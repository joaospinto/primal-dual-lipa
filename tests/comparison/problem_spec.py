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


_RESERVED_METADATA_KEYS: frozenset[str] = frozenset(
    {
        "casadi_builder",
        "is_mjx",
        "lipa_settings",
        "lipa_warmup_settings",
        "lipa_enforce_inequalities",
        "sip_settings",
        "sip_jax_settings",
        "sip_casadi_settings",
        "sip_warmup_settings",
        "trajax_settings",
        "trajax_warmup_settings",
        "ipopt_settings",
        "ipopt_mjx_extra_options",
        "ipopt_mjx_warmup_extra_options",
        "acados_settings",
        "acados_warm_start",
        "fatrop_settings",
        "fatrop_mjx_settings",
        "fatrop_mjx_warmup_settings",
        "aligator_settings",
        "aligator_jax_settings",
        "aligator_casadi_settings",
        "aligator_warmup_settings",
        "csqp_settings",
        "csqp_jax_settings",
        "csqp_casadi_settings",
        "csqp_warmup_settings",
        # Two-phase MJX warm-up. Per-(problem, solver) opt-in via the flat
        # flag ``<solver_root>_two_phase: True``. When set, the runner
        # solves with ``warmup_cost`` and no inequalities first (Phase 1),
        # then re-solves the original problem warm-started from the Phase-1
        # iterate (Phase 2). Per-solver schedules for Phase 1 come from
        # ``<solver>_warmup_settings`` / ``<solver>_warmup_extra_options``.
        "lipa_two_phase",
        "sip_two_phase",
        "csqp_two_phase",
        "ipopt_two_phase",
        "ipopt_mjx_two_phase",
        "fatrop_two_phase",
        "fatrop_mjx_two_phase",
        "aligator_two_phase",
        "trajax_two_phase",
        "acados_two_phase",
        "warmup_cost",
        # Per-solver max_iter overrides for this problem. Dict mapping
        # solver name -> int. Used to cap solvers that are known to waste
        # iterations on a problem they can't solve (e.g. csqp on
        # quadpendulum). Honored by run_benchmark._run_one_in_process.
        "max_iter_overrides",
        # Per-problem-class success-tolerance override. Float; default
        # 1e-6. Honored by ``pack_solver_result`` (framework ok/fail
        # bar) and each adapter via ``effective_solver_tol`` (per-solver
        # primal-tol target). MJX problems set this to 1e-3 so LIPA's
        # standard schedule can reach it and all solvers target the
        # same bar.
        "success_tol",
    }
)


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
    * ``lipa_enforce_inequalities`` — bool; when True, MJX problems
      ship explicit inequalities to the adapters that support them.
    * ``<solver_root>_two_phase`` + ``warmup_cost`` + per-solver
      ``<solver>_warmup_settings`` — runner-side two-phase solve,
      opt-in per (problem, solver) pair via a flat per-solver flag
      (e.g. ``metadata['lipa_two_phase'] = True``). When set, Phase 1
      uses ``warmup_cost`` with no inequalities (per-solver schedule
      from ``<solver>_warmup_settings`` shadows the matching
      ``<solver>_settings``); Phase 2 re-solves the original problem
      warm-started from Phase 1's iterate.
    * ``sip_settings`` — per-problem nested ``sip_python.Settings``
      overrides (``sip.py``; honored on both analytical and MJX problems).
    * ``ipopt_mjx_extra_options`` — per-problem IPOPT option overrides
      for the sparse-callback MJX adapter (``ipopt_mjx_sparse.py``).
    * ``acados_warm_start`` — string selector for the acados adapter's
      warm-start strategy. ``"rollout"`` (default) forward-rolls
      ``U_init`` through ``problem.dynamics`` from ``x0``; ``"linspace"``
      uses the linspace-to-extracted-goal warm start (suitable for
      problems whose terminal equality pins ``x_T = goal`` and whose
      default rollout lands the first SQP iterate too far from the
      goal for the funnel L1 line search to recover).
    * ``max_iter_overrides`` — dict ``{solver_name: int}``. Caps the
      per-solver iteration budget for this problem; used for solvers
      that have no native time cap and provably won't converge here
      (e.g. ``{"csqp": 200}`` on quadpendulum). Honored by
      ``run_benchmark``; falls back to the CLI ``--max-iter`` value
      for any solver not listed.
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

    # Adapter-specific extra warm-start state (multipliers, slacks, etc.)
    # populated by the runner during two-phase MJX solves. The key set
    # is up to each adapter — e.g. LIPA stores Y_dyn / Y_eq / S / Z.
    # Adapters that ignore this still warm-start from X_init/U_init/Theta_init.
    warm_start: Optional[dict] = None

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

    # Adapter-specific extra warm-start state (multipliers, slacks, etc.)
    # exposed for the runner's two-phase orchestration to feed back as
    # ProblemSpec.warm_start on the Phase-2 call. Key set is up to each
    # adapter; adapters that can't / won't expose this leave it None.
    warm_start_out: Optional[dict] = None

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
    return np.concatenate(
        [
            np.asarray(X).reshape(-1),
            np.asarray(U).reshape(-1),
            np.asarray(Theta).reshape(-1),
        ]
    )


def _unflatten_xut(z: jax.Array, T: int, n: int, m: int, td: int):
    nx = n * (T + 1)
    nu = m * T
    X = z[:nx].reshape(T + 1, n)
    U = z[nx : nx + nu].reshape(T, m)
    Theta = z[nx + nu : nx + nu + td]
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
    U_padded = jnp.concatenate(
        [U_j, jnp.zeros((1, problem.m), dtype=U_j.dtype)], axis=0
    )
    ts = jnp.arange(T + 1)
    costs = jax.vmap(problem.cost, in_axes=(0, 0, None, 0))(X_j, U_padded, Theta_j, ts)
    total_cost = float(jnp.sum(costs))

    # Equality residual: initial-state defect + dynamics defects + user eqs.
    init_defect = X_j[0] - problem.x0
    dyn_defects = (
        jax.vmap(problem.dynamics, in_axes=(0, 0, None, 0))(
            X_j[:-1], U_j, Theta_j, jnp.arange(T)
        )
        - X_j[1:]
    )
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
    can_compute_stationarity = multipliers_eq is not None and not (
        multipliers_ineq is None and ineq_full.size > 0
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
                    z_var,
                    T,
                    n,
                    m,
                    td,
                )
                Up = jnp.concatenate(
                    [Uv, jnp.zeros((1, m), dtype=Uv.dtype)],
                    axis=0,
                )
                ts2 = jnp.arange(T + 1)
                return jnp.sum(
                    jax.vmap(problem.cost, in_axes=(0, 0, None, 0))(
                        Xv,
                        Up,
                        Thetav,
                        ts2,
                    )
                )

            def _flat_eq(z_var):
                Xv, Uv, Thetav = _unflatten_xut(
                    z_var,
                    T,
                    n,
                    m,
                    td,
                )
                Up = jnp.concatenate(
                    [Uv, jnp.zeros((1, m), dtype=Uv.dtype)],
                    axis=0,
                )
                ts2 = jnp.arange(T + 1)
                init_d = Xv[0] - problem.x0
                dyn_d = (
                    jax.vmap(problem.dynamics, in_axes=(0, 0, None, 0))(
                        Xv[:-1],
                        Uv,
                        Thetav,
                        jnp.arange(T),
                    )
                    - Xv[1:]
                ).flatten()
                pieces = [init_d, dyn_d]
                if problem.equalities is not None:
                    eq_r = jax.vmap(problem.equalities, in_axes=(0, 0, None, 0))(
                        Xv,
                        Up,
                        Thetav,
                        ts2,
                    ).flatten()
                    pieces.append(eq_r)
                return jnp.concatenate(pieces)

            def _flat_ineq(z_var):
                if problem.inequalities is None or problem.ineq_dim == 0:
                    return jnp.zeros((0,), dtype=z_var.dtype)
                Xv, Uv, Thetav = _unflatten_xut(
                    z_var,
                    T,
                    n,
                    m,
                    td,
                )
                Up = jnp.concatenate(
                    [Uv, jnp.zeros((1, m), dtype=Uv.dtype)],
                    axis=0,
                )
                ts2 = jnp.arange(T + 1)
                return jax.vmap(problem.inequalities, in_axes=(0, 0, None, 0))(
                    Xv,
                    Up,
                    Thetav,
                    ts2,
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
        init=init_v,
        dyn=dyn_v,
        eq=eq_v_only,
        ineq=ineq_violation_inf,
        dual=dual_v,
        complementarity=comp_v,
        stationarity=stat_v,
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
        None
        if breakdown.complementarity != breakdown.complementarity
        else breakdown.complementarity
    )
    result.kkt_stationarity_inf = (
        None
        if breakdown.stationarity != breakdown.stationarity
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


def effective_solver_tol(problem: "ProblemSpec", fallback: float) -> float:
    """Return ``problem.metadata['success_tol']`` if set, else ``fallback``.

    Each adapter that translates the CLI ``--tol`` into a solver-specific
    primal-feasibility kwarg should call this to honour the per-problem
    ``success_tol`` override declared in metadata (e.g. ``5e-4`` for MJX
    OCPs where LIPA's standard schedule cannot drive primal to ``1e-6``).
    This keeps the per-solver target consistent with the framework's
    ``ok/fail`` bar — without it, adapters end up targeting tighter than
    the bar and the comparison stops being apples-to-apples (slow
    solvers look slower because they keep iterating past the success
    point that LIPA happily exits on via its aux gate).
    """
    return float(problem.metadata.get("success_tol", fallback))


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
    warm_start_out: Optional[dict] = None,
    success_tol: Optional[float] = None,
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
            problem,
            X,
            U,
            Theta,
            multipliers_eq=multipliers_eq,
            multipliers_ineq=multipliers_ineq,
        )
    else:
        cost, eq_v, ineq_v = evaluate_problem(problem, X, U, Theta)
        kkt = None

    cost_hist = eq_hist = ineq_hist = None
    if iterates_xut:
        cost_hist, eq_hist, ineq_hist = histories_from_iterates(
            problem,
            iterates_xut,
        )

    # Framework's ok/fail label requires both solver-native convergence
    # and canonical primal feasibility. The primal check keeps every
    # adapter on the same raw constraint bar; the incoming ``success``
    # flag prevents statuses such as SIP's ITERATION_LIMIT or
    # LINE_SEARCH_FAILURE from being reported as solved merely because
    # the last iterate is feasible.
    #
    # Dual / complementarity / stationarity residuals are intentionally
    # NOT included: their numeric scales depend on each solver's
    # multiplier convention (AL-penalty-scaled, interior-point-mu-
    # scaled, raw KKT, etc.), so applying a uniform tolerance to them
    # would unfairly fail solvers whose multipliers are not normalised
    # to the framework's convention. NaN / Inf iterates naturally fail
    # the < comparison so are handled.
    #
    # Per-problem ``success_tol`` override: when the caller doesn't
    # explicitly pass one, we honour ``problem.metadata['success_tol']``
    # so MJX OCPs (where LIPA's standard schedule can't drive primal to
    # 1e-6) can declare a problem-class-specific bar. The override
    # applies uniformly to every solver run on that problem — keeping
    # the comparison fair while letting LIPA succeed at the tightest
    # tolerance it can reliably meet.
    if success_tol is None:
        success_tol = float(problem.metadata.get("success_tol", 1e-6))
    primal_success = bool(float(eq_v) <= success_tol and float(ineq_v) <= success_tol)
    success = bool(success and primal_success)

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
        warm_start_out=warm_start_out,
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
