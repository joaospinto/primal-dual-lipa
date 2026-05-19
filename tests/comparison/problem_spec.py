"""Solver-agnostic problem and result containers.

A ``ProblemSpec`` is the JAX-native description of a multi-shooting OCP:
stage-wise cost / dynamics / equalities / inequalities, plus a warm
start. Every adapter consumes the same ``ProblemSpec``.

A ``SolverResult`` is what every adapter returns. The fields are:

* ``iterations``: solver's outer-loop count (semantics differ across
  solvers; documented per-adapter — IPOPT outer iters, LIPA Newton
  steps, CSQP outer SQP iters).
* ``solve_time_ms``: wall clock for the solve itself, *excluding* JIT
  compile / CasADi codegen / first-call warm-up. Per-iter recording
  overhead (see ``cost_history`` and the ``kkt_*`` fields below) is
  also excluded — adapters accumulate per-iter iterates inside the
  solver call (cheap) and convert them to ``(cost, eq, ineq)`` /
  multipliers in a post-processing pass that runs *after* the timed
  window closes.
* ``final_cost``: sum of stage costs at the returned iterate, computed
  uniformly via the JAX cost so all solvers report the same number.
* ``eq_violation_inf``: infinity-norm of the equality residual vector
  (initial-state defect + dynamics defects + user equalities).
* ``ineq_violation_inf``: max(0, ineq) infinity-norm.
* ``success``: did the solver converge within its iteration budget.
* ``cost_history``, ``eq_violation_history``, ``ineq_violation_history``:
  per-iteration arrays for convergence plots. Length = iterations + 1
  (including the initial iterate). May be ``None`` for solvers that
  don't expose this.

KKT residual breakdown fields (Goal B). Computed by
``evaluate_problem(..., multipliers_eq=, multipliers_ineq=)`` when the
adapter can extract Lagrange multipliers. All measured under the LIPA
sign convention: ``L = f + λ^T c + z^T g`` with ``c = 0`` (equality
residual = init_defect ++ dyn_defects ++ user_eqs) and ``g <= 0``,
``z >= 0``. Each is an infinity-norm or ``None`` if the adapter
couldn't extract the corresponding piece.

* ``multipliers_eq``: λ array, shape matches the stacked equality
  residual ``[init_defect (n,); dyn_defects (T*n,); user_eqs ((T+1)*eq_dim,)]``.
* ``multipliers_ineq``: z array, shape ``((T+1)*ineq_dim,)``.
* ``kkt_init_violation_inf``: ``||x_0 - problem.x0||_inf``.
* ``kkt_dyn_violation_inf``: ``||x_{t+1} - dynamics(x_t, u_t, theta, t)||_inf``.
* ``kkt_eq_violation_inf``: ``||user_eq(x_t, u_t, theta, t)||_inf``.
* ``kkt_ineq_violation_inf``: ``||max(0, user_ineq)||_inf``.
* ``kkt_dual_violation_inf``: ``||max(0, -z)||_inf``. Adapters whose
  multipliers are reported under a different sign convention should
  flip-sign before populating.
* ``kkt_complementarity_inf``: ``||z * g||_inf`` evaluated at the
  final iterate. (For an active inequality with ``g(x*) = 0`` the
  product is 0; for an inactive inequality ``z* = 0`` makes it 0.
  At interior-point convergence the product equals the barrier ``μ``,
  which we let go to 0.)
* ``kkt_stationarity_inf``: ``||∇_x L||_inf`` evaluated at the final
  iterate. ``∇_x L = ∇_x f + J_eq^T λ + J_ineq^T z``.
* ``kkt_residual_inf``: max of all of the above.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
import numpy as np

# JAX function with the LIPA stage signature: (x, u, theta, t).
StageFn = Callable[[jax.Array, jax.Array, jax.Array, jnp.int32], jax.Array]
StageScalarFn = Callable[[jax.Array, jax.Array, jax.Array, jnp.int32], jnp.double]


_RESERVED_METADATA_KEYS: frozenset[str] = frozenset({
    "casadi_builder",
    "is_mjx",
    "lipa_settings",
    "lipa_warmup_cost",
    "lipa_warmup_settings",
    "lipa_enforce_inequalities",
    "sip_settings",
    "sip_mjx_extra_settings",
    "ipopt_mjx_extra_options",
})


def validate_metadata(metadata: dict, problem_name: str) -> None:
    """Warn on unrecognised ``problem.metadata`` keys.

    Catches typos like ``lipa_settngs`` that would otherwise silently
    go unread by every adapter. Prints a single warning per unknown key
    via ``warnings.warn``; never raises.
    """
    import warnings

    unknown = sorted(set(metadata) - _RESERVED_METADATA_KEYS)
    if unknown:
        warnings.warn(
            f"problem {problem_name!r} has unrecognised metadata keys "
            f"{unknown}; expected one of "
            f"{sorted(_RESERVED_METADATA_KEYS)}. Likely a typo — no "
            "adapter reads these.",
            stacklevel=2,
        )


@dataclass(frozen=True)
class ProblemSpec:
    """JAX-native description of a multi-shooting OCP.

    ``metadata`` keys reserved by the comparison driver
    -------------------------------------------------
    Problems may attach the following keys to ``metadata`` to opt into
    solver-specific behaviour. Unrecognised keys are ignored.

    * ``casadi_builder`` — callable returning ``{f, next_x, eq, ineq}``
      CasADi expressions per stage. Consumed by the CasADi-based
      analytical adapters (``ipopt_casadi``, ``acados``, ``aligator``,
      ``csqp``, ``fatrop``, and ``sip`` casadi-backend).
    * ``is_mjx`` — bool flag; problems built on MuJoCo MJX set this so
      adapters can pick MJX-appropriate code paths (rollout-vs-tile
      warm start auto-selection, callback wrapping, etc.).
    * ``lipa_settings`` — full ``lipa.Settings`` override for this
      problem (``lipa.py``).
    * ``lipa_warmup_cost`` / ``lipa_warmup_settings`` — soft-penalty
      warm-up phase config (``lipa.py``).
    * ``lipa_enforce_inequalities`` — bool; when True, MJX problems
      ship explicit inequalities to the adapters that support them.
    * ``sip_settings`` — per-problem ``sip_python`` settings overrides
      (``sip.py``).
    * ``sip_mjx_extra_settings`` — per-problem ``sip_python`` overrides
      for the MJX adapter (``sip_mjx.py``).
    * ``ipopt_mjx_extra_options`` — per-problem IPOPT option overrides
      for the sparse-callback MJX adapter (``ipopt_mjx_sparse.py``).
    """

    name: str
    T: int  # horizon (so X has shape (T+1, n), U has shape (T, m))
    n: int  # state dim
    m: int  # control dim
    theta_dim: int  # 0 if no cross-stage variable

    x0: jax.Array  # (n,) initial state

    # Stage-wise functions, all jax-jittable.
    cost: StageScalarFn  # (x, u, theta, t) -> scalar
    dynamics: StageFn  # (x, u, theta, t) -> next_state of shape (n,)
    equalities: Optional[StageFn]  # (x, u, theta, t) -> (eq_dim,) or None
    inequalities: Optional[StageFn]  # (x, u, theta, t) -> (ineq_dim,) <= 0 or None

    # Output dimensions per stage. The total counts are eq_dim*(T+1) and
    # ineq_dim*(T+1) when these functions are evaluated at every stage,
    # because the LIPA convention pads U with zeros at t=T.
    eq_dim: int
    ineq_dim: int

    # Warm start.
    X_init: jax.Array  # (T+1, n)
    U_init: jax.Array  # (T, m)
    Theta_init: jax.Array  # (theta_dim,)

    # Optional metadata used by report layer and to opt into
    # solver-specific behaviour (see class docstring for reserved keys).
    metadata: dict = field(default_factory=dict)


class KKTBreakdown(NamedTuple):
    """Per-component KKT residual measurement.

    All entries are non-negative scalars (infinity-norms). ``joint`` is
    the max of all the others. ``dual`` / ``complementarity`` /
    ``stationarity`` are ``nan`` when the corresponding multipliers were
    not provided to ``evaluate_problem``.
    """

    init: float
    dyn: float
    eq: float
    ineq: float
    dual: float  # max(0, -z); nan if no z provided
    complementarity: float  # ||z * g||_inf; nan if no z provided
    stationarity: float  # ||∇_x L||_inf; nan if no λ/z provided
    joint: float


@dataclass
class SolverResult:
    """Uniform container for a single solver's output on a single problem."""

    solver_name: str
    problem_name: str
    iterations: int
    solve_time_ms: float
    final_cost: float
    eq_violation_inf: float
    ineq_violation_inf: float
    success: bool

    # Optional per-iteration arrays for convergence plots.
    cost_history: Optional[np.ndarray] = None
    eq_violation_history: Optional[np.ndarray] = None
    ineq_violation_history: Optional[np.ndarray] = None

    # Optional final iterate (X, U, Theta) for cross-validation.
    X: Optional[np.ndarray] = None
    U: Optional[np.ndarray] = None
    Theta: Optional[np.ndarray] = None

    # KKT residual breakdown (Goal B). Adapters that don't extract
    # multipliers leave the multiplier-dependent fields at None / nan.
    multipliers_eq: Optional[np.ndarray] = None
    multipliers_ineq: Optional[np.ndarray] = None
    kkt_init_violation_inf: Optional[float] = None
    kkt_dyn_violation_inf: Optional[float] = None
    kkt_eq_violation_inf: Optional[float] = None
    kkt_ineq_violation_inf: Optional[float] = None
    kkt_dual_violation_inf: Optional[float] = None
    kkt_complementarity_inf: Optional[float] = None
    kkt_stationarity_inf: Optional[float] = None
    kkt_residual_inf: Optional[float] = None

    # Free-text reason for failure / extra notes.
    notes: str = ""


def _flatten_xut(X: np.ndarray, U: np.ndarray, Theta: np.ndarray) -> np.ndarray:
    """Pack (X, U, Theta) into a flat decision vector (matches the SIP layout)."""
    return np.concatenate([
        np.asarray(X).reshape(-1),
        np.asarray(U).reshape(-1),
        np.asarray(Theta).reshape(-1),
    ])


def _unflatten_xut(z: jax.Array, T: int, n: int, m: int, td: int):
    nx = n * (T + 1)
    nu = m * T
    X = z[:nx].reshape(T + 1, n)
    U = z[nx:nx + nu].reshape(T, m)
    Theta = z[nx + nu:nx + nu + td]
    return X, U, Theta


def evaluate_problem(
    problem: ProblemSpec,
    X: np.ndarray,
    U: np.ndarray,
    Theta: np.ndarray,
    *,
    multipliers_eq: Optional[np.ndarray] = None,
    multipliers_ineq: Optional[np.ndarray] = None,
) -> tuple[float, float, float] | tuple[float, float, float, KKTBreakdown]:
    """Compute (final_cost, eq_violation_inf, ineq_violation_inf) for an iterate.

    All adapters call this so the reported numbers are computed identically
    regardless of how the solver represented the problem internally.

    When ``multipliers_eq`` / ``multipliers_ineq`` are provided the function
    additionally returns a ``KKTBreakdown`` with the per-component KKT
    residual (init / dyn / eq / ineq / dual / complementarity /
    stationarity, plus the joint max). The multiplier shapes must match the
    stacked equality residual ``[init_defect; dyn_defects; user_eqs]`` and
    the stacked inequality residual ``user_ineqs`` produced by this
    function. For adapters whose internal sign convention is the opposite
    (``L = f - λ^T c - z^T g``), flip the sign before passing.

    Calling without multipliers preserves the legacy 3-tuple return so
    adapters that haven't been instrumented yet keep working.
    """
    X_j = jnp.asarray(X)
    U_j = jnp.asarray(U)
    Theta_j = jnp.asarray(Theta)
    T = problem.T

    # Cost: sum over t=0..T, with U padded by a zero stage at t=T.
    U_padded = jnp.concatenate([U_j, jnp.zeros((1, problem.m), dtype=U_j.dtype)], axis=0)
    ts = jnp.arange(T + 1)
    costs = jax.vmap(problem.cost, in_axes=(0, 0, None, 0))(X_j, U_padded, Theta_j, ts)
    total_cost = float(jnp.sum(costs))

    # Equality residual: initial-state defect + dynamics defects + user eqs.
    init_defect = X_j[0] - problem.x0
    dyn_defects = jax.vmap(problem.dynamics, in_axes=(0, 0, None, 0))(
        X_j[:-1], U_j, Theta_j, jnp.arange(T)
    ) - X_j[1:]
    eq_pieces = [init_defect.flatten(), dyn_defects.flatten()]
    if problem.equalities is not None:
        eq_residuals = jax.vmap(problem.equalities, in_axes=(0, 0, None, 0))(
            X_j, U_padded, Theta_j, ts
        )
        eq_pieces.append(eq_residuals.flatten())
    eq_full = jnp.concatenate(eq_pieces)
    eq_violation_inf = float(jnp.max(jnp.abs(eq_full)))

    # Inequality residual: max(0, g).
    if problem.inequalities is not None:
        ineq = jax.vmap(problem.inequalities, in_axes=(0, 0, None, 0))(
            X_j, U_padded, Theta_j, ts
        )
        ineq_full = ineq.flatten()
        ineq_violation_inf = float(jnp.max(jnp.maximum(ineq, 0.0)))
    else:
        ineq_full = jnp.zeros((0,), dtype=X_j.dtype)
        ineq_violation_inf = 0.0

    if multipliers_eq is None and multipliers_ineq is None:
        return total_cost, eq_violation_inf, ineq_violation_inf

    # ------------ KKT breakdown (Goal B) -----------------------------------
    n, m, td = problem.n, problem.m, problem.theta_dim
    init_v = float(jnp.max(jnp.abs(init_defect)))
    dyn_v = float(jnp.max(jnp.abs(dyn_defects)))
    if problem.equalities is not None:
        eq_v_only = float(jnp.max(jnp.abs(eq_pieces[2])))
    else:
        eq_v_only = 0.0

    # Dual feasibility: z >= 0 (under LIPA convention).
    if multipliers_ineq is not None and ineq_full.size > 0:
        z = jnp.asarray(multipliers_ineq).reshape(-1)
        if z.size != ineq_full.size:
            # Mismatch means the adapter passed the wrong shape; treat as
            # "couldn't extract" and keep the dual / complementarity slots
            # at nan. Stationarity also can't be computed.
            dual_v = float("nan")
            comp_v = float("nan")
            z = None
        else:
            dual_v = float(jnp.max(jnp.maximum(-z, 0.0)))
            comp_v = float(jnp.max(jnp.abs(z * ineq_full)))
    else:
        z = None
        dual_v = float("nan") if multipliers_ineq is None else 0.0
        comp_v = float("nan") if multipliers_ineq is None else 0.0

    # Stationarity: ∇_x L = ∇_x f + J_eq^T λ + J_ineq^T z.
    # We compute it by autodiff on the flat decision vector
    # ``z_var = (vec(X), vec(U), Theta)``. λ stacks
    # ``[init_defect (n,); dyn_defects (T*n,); user_eqs ((T+1)*eq_dim,)]``.
    can_compute_stationarity = (
        multipliers_eq is not None
        and not (multipliers_ineq is None and ineq_full.size > 0)
    )
    if can_compute_stationarity:
        lam = jnp.asarray(multipliers_eq).reshape(-1)
        expected_eq_size = eq_full.size
        if lam.size != expected_eq_size:
            stat_v = float("nan")
        else:
            z_var0 = _flatten_xut(X, U, Theta)

            def _flat_objective(z_var):
                Xv, Uv, Thetav = _unflatten_xut(
                    z_var, T, n, m, td,
                )
                Up = jnp.concatenate(
                    [Uv, jnp.zeros((1, m), dtype=Uv.dtype)], axis=0,
                )
                ts2 = jnp.arange(T + 1)
                return jnp.sum(jax.vmap(problem.cost, in_axes=(0, 0, None, 0))(
                    Xv, Up, Thetav, ts2,
                ))

            def _flat_eq(z_var):
                Xv, Uv, Thetav = _unflatten_xut(
                    z_var, T, n, m, td,
                )
                Up = jnp.concatenate(
                    [Uv, jnp.zeros((1, m), dtype=Uv.dtype)], axis=0,
                )
                ts2 = jnp.arange(T + 1)
                init_d = Xv[0] - problem.x0
                dyn_d = (jax.vmap(problem.dynamics, in_axes=(0, 0, None, 0))(
                    Xv[:-1], Uv, Thetav, jnp.arange(T),
                ) - Xv[1:]).flatten()
                pieces = [init_d, dyn_d]
                if problem.equalities is not None:
                    eq_r = jax.vmap(problem.equalities, in_axes=(0, 0, None, 0))(
                        Xv, Up, Thetav, ts2,
                    ).flatten()
                    pieces.append(eq_r)
                return jnp.concatenate(pieces)

            def _flat_ineq(z_var):
                if problem.inequalities is None or problem.ineq_dim == 0:
                    return jnp.zeros((0,), dtype=z_var.dtype)
                Xv, Uv, Thetav = _unflatten_xut(
                    z_var, T, n, m, td,
                )
                Up = jnp.concatenate(
                    [Uv, jnp.zeros((1, m), dtype=Uv.dtype)], axis=0,
                )
                ts2 = jnp.arange(T + 1)
                return jax.vmap(problem.inequalities, in_axes=(0, 0, None, 0))(
                    Xv, Up, Thetav, ts2,
                ).flatten()

            def _lagrangian(z_var):
                f_val = _flat_objective(z_var)
                eq_r = _flat_eq(z_var)
                term = f_val + jnp.dot(lam, eq_r)
                if z is not None and ineq_full.size > 0:
                    g_r = _flat_ineq(z_var)
                    term = term + jnp.dot(z, g_r)
                return term

            grad_L = jax.grad(_lagrangian)(jnp.asarray(z_var0))
            stat_v = float(jnp.max(jnp.abs(grad_L)))
    else:
        stat_v = float("nan")

    # Joint max — ignore nan entries when computing.
    parts = [init_v, dyn_v, eq_v_only, ineq_violation_inf]
    for v in (dual_v, comp_v, stat_v):
        if not (v != v):  # not nan
            parts.append(v)
    joint = float(max(parts)) if parts else 0.0

    breakdown = KKTBreakdown(
        init=init_v, dyn=dyn_v, eq=eq_v_only, ineq=ineq_violation_inf,
        dual=dual_v, complementarity=comp_v, stationarity=stat_v,
        joint=joint,
    )
    return total_cost, eq_violation_inf, ineq_violation_inf, breakdown


def attach_kkt(result: SolverResult, breakdown: KKTBreakdown) -> SolverResult:
    """Stamp a ``KKTBreakdown`` onto a ``SolverResult`` in place.

    Convenience wrapper so per-adapter code reads cleanly:

        cost, eq_v, ineq_v, kkt = evaluate_problem(
            problem, X, U, Theta,
            multipliers_eq=lam, multipliers_ineq=z,
        )
        result = SolverResult(...)
        attach_kkt(result, kkt)
    """
    result.kkt_init_violation_inf = breakdown.init
    result.kkt_dyn_violation_inf = breakdown.dyn
    result.kkt_eq_violation_inf = breakdown.eq
    result.kkt_ineq_violation_inf = breakdown.ineq
    result.kkt_dual_violation_inf = (
        None if breakdown.dual != breakdown.dual else breakdown.dual
    )
    result.kkt_complementarity_inf = (
        None if breakdown.complementarity != breakdown.complementarity
        else breakdown.complementarity
    )
    result.kkt_stationarity_inf = (
        None if breakdown.stationarity != breakdown.stationarity
        else breakdown.stationarity
    )
    result.kkt_residual_inf = breakdown.joint
    return result


def histories_from_iterates(
    problem: ProblemSpec,
    iterates: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert a list of ``(X, U, Theta)`` iterates into the 3 history arrays.

    Returns ``(cost_history, eq_violation_history, ineq_violation_history)``,
    each of length ``len(iterates)``. Each call to ``evaluate_problem``
    is a single fused-vmap pass and is essentially free relative to a
    full solve.
    """
    if not iterates:
        empty = np.zeros((0,), dtype=np.float64)
        return empty, empty.copy(), empty.copy()
    costs = np.empty(len(iterates), dtype=np.float64)
    eqs = np.empty(len(iterates), dtype=np.float64)
    ineqs = np.empty(len(iterates), dtype=np.float64)
    for i, (X, U, Theta) in enumerate(iterates):
        c, eq_v, ineq_v = evaluate_problem(problem, X, U, Theta)
        costs[i] = c
        eqs[i] = eq_v
        ineqs[i] = ineq_v
    return costs, eqs, ineqs


def make_failure_result(
    solver_name: str,
    problem_name: str,
    notes: str,
    *,
    solve_time_ms: float = 0.0,
) -> SolverResult:
    """Boilerplate ``SolverResult`` for early-return failures.

    Used by every adapter for unavailable / theta-skip / missing-builder
    / build-error / solver-raised paths. ``solve_time_ms`` lets callers
    record any partial-solve wall time (default 0.0 for pure early
    returns).
    """
    return SolverResult(
        solver_name=solver_name,
        problem_name=problem_name,
        iterations=0,
        solve_time_ms=solve_time_ms,
        final_cost=float("nan"),
        eq_violation_inf=float("nan"),
        ineq_violation_inf=float("nan"),
        success=False,
        notes=notes,
    )


def pack_solver_result(
    *,
    solver_name: str,
    problem_name: str,
    problem: ProblemSpec,
    X: np.ndarray,
    U: np.ndarray,
    Theta: np.ndarray,
    iterations: int,
    solve_time_ms: float,
    success: bool,
    notes: str = "",
    multipliers_eq: Optional[np.ndarray] = None,
    multipliers_ineq: Optional[np.ndarray] = None,
    compute_kkt: bool = True,
    iterates_xut: Optional[list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = None,
) -> SolverResult:
    """Run ``evaluate_problem`` on ``(X, U, Theta)`` and pack a ``SolverResult``.

    Centralises the post-solve assembly shared by the CasADi-based
    adapters: route the final iterate through ``evaluate_problem`` for
    a canonical ``(cost, eq_v, ineq_v)``, optionally compute the KKT
    breakdown when multipliers are provided, optionally compute per-iter
    histories via ``histories_from_iterates``, then build the
    ``SolverResult`` with all the standard fields filled in.

    Adapter-specific work (multiplier remapping into the
    ``[init; dyn; user_eq]`` / ``user_ineq`` stacks that
    ``evaluate_problem`` expects) stays inline at the call site — this
    helper only consumes the already-remapped arrays.

    Parameters
    ----------
    multipliers_eq, multipliers_ineq
        Already remapped to ``evaluate_problem``'s stack layout. Pass
        ``None`` when the adapter can't extract a coherent split (the
        result will still carry the raw arrays in the corresponding
        ``SolverResult`` fields if the caller stores them separately).
    compute_kkt
        When ``True`` and at least one multiplier array is non-None,
        the KKT breakdown is computed and stamped onto the result.
        Set ``False`` to skip KKT even when multipliers are present
        (e.g. when their shape doesn't match the canonical stacks but
        the caller still wants to expose them on the result).
    iterates_xut
        Optional list of per-iter ``(X, U, Theta)`` tuples; when
        non-empty, ``cost_history`` / ``eq_violation_history`` /
        ``ineq_violation_history`` are populated.
    """
    use_kkt = compute_kkt and (
        multipliers_eq is not None or multipliers_ineq is not None
    )
    if use_kkt:
        cost, eq_v, ineq_v, kkt = evaluate_problem(
            problem, X, U, Theta,
            multipliers_eq=multipliers_eq,
            multipliers_ineq=multipliers_ineq,
        )
    else:
        cost, eq_v, ineq_v = evaluate_problem(problem, X, U, Theta)
        kkt = None

    cost_hist = eq_hist = ineq_hist = None
    if iterates_xut:
        cost_hist, eq_hist, ineq_hist = histories_from_iterates(
            problem, iterates_xut,
        )

    result = SolverResult(
        solver_name=solver_name,
        problem_name=problem_name,
        iterations=iterations,
        solve_time_ms=solve_time_ms,
        final_cost=cost,
        eq_violation_inf=eq_v,
        ineq_violation_inf=ineq_v,
        success=success,
        X=X,
        U=U,
        Theta=Theta,
        multipliers_eq=multipliers_eq,
        multipliers_ineq=multipliers_ineq,
        cost_history=cost_hist,
        eq_violation_history=eq_hist,
        ineq_violation_history=ineq_hist,
        notes=notes,
    )
    if kkt is not None:
        attach_kkt(result, kkt)
    return result
