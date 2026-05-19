"""IPOPT-on-MJX adapter using per-stage CasADi callbacks that share a
single JIT'd JAX function each (dynamics, cost, inequalities) plus a
single global ``hess_lag`` Callback that returns the per-stage
block-diagonal exact Lagrangian Hessian.

Architecture
------------
Per-stage CasADi callbacks let CasADi see the block-banded sparsity of
the OCP and infer Jacobians from the symbolic graph. Each per-stage
callback's ``eval`` delegates to a single shared ``jax.jit``'d function
with the stage index ``t`` passed as a runtime ``jnp.int32`` argument
(see ``casadi_jax_callback.PerStageJaxCallback`` for the
amortization rationale).

Alternative routes considered and rejected:

* ``ipopt_casadi.py``'s per-stage JAX-callback path closes over ``t``
  inside each ``jax.jit``, so each stage triggers a fresh trace. With
  ~3 callbacks × T stages × seconds of trace per stage, warm-up alone
  takes tens of minutes on MJX-scale problems.
* A single global JAX callback returning the entire trajectory plus a
  hand-supplied block-banded Jacobian causes ``nlpsol`` to hang for
  ≥10 min during sparsity / adjoint-graph construction, even with
  ``Callback.get_jacobian`` provided.

The unused ``_build_eq_sparsity`` / ``_build_ineq_sparsity`` /
``_pack_eq_jac_nonzeros`` / ``_pack_ineq_jac_nonzeros`` helpers in this
module document what the layout *would* be for the global-callback
route — kept for the day CasADi grows a way to feed a single sparse
Callback into ``nlpsol`` cheaply.

Lagrangian Hessian
------------------
Exact Lagrangian Hessian, per-stage symmetrized only. IPOPT's own
inertia correction (Wächter–Biegler 2006 §3.4) handles indefinite
KKT factorizations, so we do not pre-PSD-project the per-stage blocks.

The Lagrangian for a multi-shooting OCP is **block-diagonal in stages**:
each ``(x_t, u_t)`` pair only appears in ``cost_t``, ``dyn_t`` (which
the Lagrangian weights by ``-λ_dyn_t``), and ``ineq_t``. The dynamics
defect ``x_{t+1} - dyn_t(x_t,u_t)`` is linear in ``x_{t+1}``, so it
contributes nothing to the ``x_{t+1}`` block; and the init defect
``x_0 - x0`` is linear in ``x_0``. So the per-stage block is
``σ·∇²cost_t - λ_dyn_t·∇²dyn_t + z_t·∇²ineq_t`` (n+m × n+m), and the
terminal block is ``σ·∇²cost_T + z_T·∇²ineq_T`` (n × n).

We provide CasADi's ``nlpsol(..., 'ipopt', ..., {'hess_lag': cb})``
extension point with a single global Callback whose ``eval``:

1. Calls a vmap'd JAX function over ``t = 0..T-1`` to get all inner
   blocks at once (single JIT trace).
2. Calls a separate JAX function for the terminal block.
3. Packs the upper-triangular non-zeros into a pre-built sparsity
   template (block-diagonal in stages, identical layout to sip-mjx's
   ``upp_hess_template``).

The user-declared sparsity on ``get_sparsity_out`` means CasADi treats
the Hessian as opaque (no symbolic differentiation downstream) — IPOPT
just consumes the sparse triu-H. This avoids the symbolic-graph hang
that bit the global-constraint-Jacobian attempt.
"""

from __future__ import annotations

from timeit import default_timer as timer
from typing import Optional

import numpy as np

from tests.comparison.adapters import register
from tests.comparison.adapters.base import SolverAdapter
from tests.comparison.adapters.ipopt_casadi import _IterationRecorder
from tests.comparison.problem_spec import (
    ProblemSpec,
    SolverResult,
    make_failure_result,
    pack_solver_result,
)


def _import_casadi():
    import casadi as ca

    return ca


# -----------------------------------------------------------------------------
# Sparsity helpers — kept as documentation of the block-banded layout and as
# unit-testable building blocks for any future single-callback approach.
# Currently NOT used by the live adapter (see module docstring).
# -----------------------------------------------------------------------------


def _build_eq_sparsity(T: int, n: int, m: int):
    """Equality-Jacobian sparsity pattern for the multi-shooting OCP.

    Row layout: ``(T+1)*n`` rows, ``n`` rows per stage block.
    Column layout: ``(T+1)*n + T*m`` cols (vec(X) then vec(U)).

    Returns a column-major (CCS-friendly) ``casadi.Sparsity``. Values
    must be supplied in the order produced by ``_pack_eq_jac_nonzeros``.
    """
    ca = _import_casadi()
    n_rows = (T + 1) * n
    n_cols = (T + 1) * n + T * m
    nx = (T + 1) * n

    rows: list[int] = []
    cols: list[int] = []

    # X[0] columns.
    for j in range(n):
        rows.append(j)
        cols.append(j)  # init defect identity
        for i in range(n):
            rows.append(n + i)
            cols.append(j)  # -dDx[0][:,j]

    # X[t] columns for t = 1..T-1.
    for t in range(1, T):
        for j in range(n):
            rows.append(t * n + j)
            cols.append(t * n + j)  # identity
            for i in range(n):
                rows.append((t + 1) * n + i)
                cols.append(t * n + j)  # -dDx[t][:,j]

    # X[T] columns (just identities).
    for j in range(n):
        rows.append(T * n + j)
        cols.append(T * n + j)

    # U[t] columns.
    for t in range(T):
        for j in range(m):
            for i in range(n):
                rows.append((t + 1) * n + i)
                cols.append(nx + t * m + j)

    return ca.Sparsity.triplet(n_rows, n_cols, rows, cols)


def _build_ineq_sparsity(T: int, n: int, m: int, ineq_dim: int):
    """Inequality-Jacobian sparsity pattern.

    Row layout: ``(T+1)*ineq_dim`` rows. Column layout: ``(T+1)*n + T*m``.
    Each stage block depends on ``(X[t], U[t])`` (only on ``X[T]`` at t=T).
    """
    ca = _import_casadi()
    n_rows = (T + 1) * ineq_dim
    n_cols = (T + 1) * n + T * m
    nx = (T + 1) * n

    rows: list[int] = []
    cols: list[int] = []

    for t in range(T + 1):
        row_off = t * ineq_dim
        for j in range(n):
            for i in range(ineq_dim):
                rows.append(row_off + i)
                cols.append(t * n + j)

    for t in range(T):
        row_off = t * ineq_dim
        for j in range(m):
            for i in range(ineq_dim):
                rows.append(row_off + i)
                cols.append(nx + t * m + j)

    return ca.Sparsity.triplet(n_rows, n_cols, rows, cols)


def _pack_eq_jac_nonzeros(
    dDx_per_stage: np.ndarray,
    dDu_per_stage: np.ndarray,
    T: int,
    n: int,
    m: int,
) -> np.ndarray:
    """Pack per-stage dynamics Jacobian blocks into CCS-order nonzeros.

    See ``_build_eq_sparsity`` for the layout.
    """
    dDx = np.asarray(dDx_per_stage)
    dDu = np.asarray(dDu_per_stage)

    inner = np.empty((T, n, 1 + n), dtype=np.float64)
    inner[:, :, 0] = 1.0
    inner[:, :, 1:] = -np.transpose(dDx, (0, 2, 1))
    x_inner = inner.reshape(T * n * (1 + n))

    x_terminal = np.ones(n, dtype=np.float64)

    u_block = -np.transpose(dDu, (0, 2, 1))
    u_flat = u_block.reshape(T * m * n)

    out = np.concatenate([x_inner, x_terminal, u_flat])
    expected = T * n * (1 + n) + n + T * m * n
    assert out.size == expected, f"packed {out.size} entries, expected {expected}"
    return out


def _pack_ineq_jac_nonzeros(
    dGx_per_stage: np.ndarray,
    dGu_per_stage_inner: np.ndarray,
    T: int,
    n: int,
    m: int,
    ineq_dim: int,
) -> np.ndarray:
    """Pack per-stage inequality Jacobian blocks into CCS-order nonzeros."""
    dGx = np.asarray(dGx_per_stage)
    dGu = np.asarray(dGu_per_stage_inner)
    x_block = np.transpose(dGx, (0, 2, 1)).reshape((T + 1) * n * ineq_dim)
    u_block = np.transpose(dGu, (0, 2, 1)).reshape(T * m * ineq_dim)
    return np.concatenate([x_block, u_block])


# Per-stage CasADi callback that shares a single JIT'd JAX function across
# stages lives in tests.comparison.casadi_jax_callback. See that module's
# docstring for the rationale (one-trace-per-shape JIT amortization).
# IPOPT in L-BFGS mode does not need the reverse-mode (adjoint Hessian)
# plumbing, so we pass ``adj_hess_jit_fn=None`` (the default).
from tests.comparison.casadi_jax_callback import (
    PerStageJaxCallback as _PerStageJaxCallback,
)  # noqa: E402


# -----------------------------------------------------------------------------
# Hessian-of-Lagrangian Callback. Single global Callback that returns the
# upper-triangular block-diagonal exact Hessian (per-stage symmetrized only;
# IPOPT's inertia correction handles any indefiniteness).
# -----------------------------------------------------------------------------


def _build_block_diag_triu_sparsity(T: int, n: int, m: int, td: int = 0):
    """Block-diagonal-plus-theta-arrow upper-triangular sparsity.

    Block layout matches sip-mjx's ``upp_hess_template``:
      * For ``t = 0..T-1``: a dense ``(n+m, n+m)`` block on rows/cols
        ``(x_t, u_t)`` where ``x_t = [t*n, t*n+n)`` and
        ``u_t = [nx + t*m, nx + t*m + m)``, with ``nx = (T+1)*n``.
      * For ``t = T``: a dense ``(n, n)`` block on rows/cols ``x_T``.
      * If ``td > 0``: per-stage ``(n+m, td)`` cross blocks coupling
        ``(x_t, u_t)`` to theta (and ``(n, td)`` at the terminal),
        plus a single global ``(td, td)`` theta-theta corner that
        accumulates contributions from every stage.

    Only the upper triangle is stored (rows ≤ cols). Theta lives at
    ``z[nx + nu : nx + nu + td]`` in the global decision vector.
    """
    ca = _import_casadi()
    nx = (T + 1) * n
    nu = T * m
    x_dim = nx + nu + td
    theta_idx = list(range(nx + nu, nx + nu + td)) if td > 0 else []

    rows: list[int] = []
    cols: list[int] = []
    for t in range(T):
        x_idx = list(range(t * n, t * n + n))
        u_idx = list(range(nx + t * m, nx + t * m + m))
        block_idx = x_idx + u_idx
        # (x_t, u_t) × (x_t, u_t) block
        for ri in block_idx:
            for cj in block_idx:
                if ri <= cj:
                    rows.append(ri)
                    cols.append(cj)
        # (x_t, u_t) × theta cross block (always above-diagonal since
        # theta indices are at the end of z).
        for ri in block_idx:
            for cj in theta_idx:
                rows.append(ri)
                cols.append(cj)
    # Terminal stage.
    x_idx_T = list(range(T * n, T * n + n))
    for ri in x_idx_T:
        for cj in x_idx_T:
            if ri <= cj:
                rows.append(ri)
                cols.append(cj)
    for ri in x_idx_T:
        for cj in theta_idx:
            rows.append(ri)
            cols.append(cj)
    # Single shared theta-theta corner.
    for ri in theta_idx:
        for cj in theta_idx:
            if ri <= cj:
                rows.append(ri)
                cols.append(cj)
    return ca.Sparsity.triplet(x_dim, x_dim, rows, cols)


# -----------------------------------------------------------------------------
# Adapter.
# -----------------------------------------------------------------------------


def _slice_iterate(z_val: np.ndarray, problem: ProblemSpec):
    n, m, T, td = problem.n, problem.m, problem.T, problem.theta_dim
    nx = n * (T + 1)
    nu = m * T
    X = z_val[:nx].reshape(T + 1, n)
    U = z_val[nx : nx + nu].reshape(T, m)
    Theta = z_val[nx + nu : nx + nu + td] if td > 0 else np.zeros(0)
    return X, U, Theta


class IpoptMjxSparseAdapter(SolverAdapter):
    """IPOPT adapter with per-stage JAX callbacks — see module docstring.

    Handles dynamics defects, user equality constraints, and inequality
    constraints. Works on both MJX problems and analytical problems with
    user equalities (e.g. cartpole's terminal-state goal); the standard
    ``ipopt`` adapter uses CasADi's pure-symbolic builder and is the
    preferred path for analytical problems with a ``casadi_builder``.
    """

    name = "ipopt-jax"

    def __init__(
        self,
        max_iter: int = 1000,
        tol: float = 1e-6,
        print_level: int = 0,
        timeout_s: Optional[float] = None,
        # IPOPT's default mu_init=0.1 leaves the barrier too loose on
        # contact-rich MJX OCPs; 1e-4 starts closer to the boundary.
        # Per-problem overrides via metadata['ipopt_mjx_extra_options']
        # take precedence.
        ipopt_extra_options: Optional[dict] = None,
    ) -> None:
        self.max_iter = max_iter
        self.tol = tol
        self.print_level = print_level
        self.timeout_s = timeout_s
        # Stash the user-supplied options on top of our tuned defaults
        # so user keys win on collision (i.e. user can opt out of the
        # mu_init=1e-4 default by passing ipopt_extra_options={'mu_init': ...}).
        defaults = {"mu_init": 1e-4}
        merged = dict(defaults)
        if ipopt_extra_options:
            merged.update(ipopt_extra_options)
        self.ipopt_extra_options = merged

    def solve(self, problem: ProblemSpec) -> SolverResult:
        from tests.comparison.problem_spec import effective_solver_tol

        # Stash on self so the inner ``_solve`` can read it.
        self._effective_tol = effective_solver_tol(problem, self.tol)
        return self._solve(problem)

    def _solve(self, problem: ProblemSpec) -> SolverResult:
        ca = _import_casadi()
        import jax
        import jax.numpy as jnp

        T, n, m, td = problem.T, problem.n, problem.m, problem.theta_dim

        # ---------- Shared JIT'd JAX functions ----------
        # Each takes the stage index ``t`` as a runtime jnp.int32 so JAX
        # traces ONCE and reuses the compiled code for every stage. When
        # ``td > 0``, ``Theta`` is appended to the per-stage callback
        # input so each stage's symbolic graph in CasADi sees the
        # theta-coupling explicitly (IPOPT gets the right gradient
        # sparsity and the Lagrangian Hessian gets theta-arrow blocks).

        def _split_xu_theta(z):
            x = z[:n]
            u = z[n : n + m]
            theta = z[n + m : n + m + td] if td > 0 else jnp.empty(0)
            return x, u, theta

        def _split_x_theta(z):
            x = z[:n]
            theta = z[n : n + td] if td > 0 else jnp.empty(0)
            return x, theta

        @jax.jit
        def dyn_eval(z, t):
            x, u, theta = _split_xu_theta(z)
            return problem.dynamics(x, u, theta, t)

        # Jacobian wrt the (n+m+td,) input z.
        @jax.jit
        def dyn_jac(z, t):
            return jax.jacrev(lambda zz: dyn_eval(zz, t))(z)

        @jax.jit
        def cost_eval_xu(z, t):
            x, u, theta = _split_xu_theta(z)
            return jnp.array([problem.cost(x, u, theta, t)])

        @jax.jit
        def cost_jac_xu(z, t):
            return jax.jacrev(lambda zz: cost_eval_xu(zz, t))(z)

        @jax.jit
        def cost_eval_x(z, t):
            x, theta = _split_x_theta(z)
            u_zero = jnp.zeros(m, dtype=x.dtype)
            return jnp.array([problem.cost(x, u_zero, theta, t)])

        @jax.jit
        def cost_jac_x(z, t):
            return jax.jacrev(lambda zz: cost_eval_x(zz, t))(z)

        # Optional inequality.
        has_ineq = problem.inequalities is not None and problem.ineq_dim > 0
        if has_ineq:

            @jax.jit
            def ineq_eval_xu(z, t):
                x, u, theta = _split_xu_theta(z)
                return problem.inequalities(x, u, theta, t)

            @jax.jit
            def ineq_jac_xu(z, t):
                return jax.jacrev(lambda zz: ineq_eval_xu(zz, t))(z)

            @jax.jit
            def ineq_eval_x(z, t):
                x, theta = _split_x_theta(z)
                u_zero = jnp.zeros(m, dtype=x.dtype)
                return problem.inequalities(x, u_zero, theta, t)

            @jax.jit
            def ineq_jac_x(z, t):
                return jax.jacrev(lambda zz: ineq_eval_x(zz, t))(z)

        # Optional user equalities (e.g. terminal-goal constraints).
        has_user_eq = problem.equalities is not None and problem.eq_dim > 0
        if has_user_eq:

            @jax.jit
            def eq_eval_xu(z, t):
                x, u, theta = _split_xu_theta(z)
                return problem.equalities(x, u, theta, t)

            @jax.jit
            def eq_jac_xu(z, t):
                return jax.jacrev(lambda zz: eq_eval_xu(zz, t))(z)

            @jax.jit
            def eq_eval_x(z, t):
                x, theta = _split_x_theta(z)
                u_zero = jnp.zeros(m, dtype=x.dtype)
                return problem.equalities(x, u_zero, theta, t)

            @jax.jit
            def eq_jac_x(z, t):
                return jax.jacrev(lambda zz: eq_eval_x(zz, t))(z)

            # Per-stage presence probe. We can't symbolically detect
            # ``jnp.where(t == T, ..., zeros)`` patterns the way the
            # CasADi adapter does via SX expressions, so we evaluate
            # equalities at two distinct random points per stage; if
            # both return identically zero, the equality is treated
            # as structurally absent at that stage and omitted from
            # the NLP (it would otherwise produce a zero Jacobian row
            # that IPOPT flags as "Not_Enough_Degrees_Of_Freedom").
            _eq_rng = np.random.default_rng(0)
            eq_t_present = np.zeros(T + 1, dtype=bool)
            inner_dim = n + m + td
            terminal_dim = n + td
            for tt in range(T + 1):
                if tt == T:
                    xt_a = jnp.asarray(
                        _eq_rng.normal(size=terminal_dim), dtype=jnp.float64
                    )
                    xt_b = jnp.asarray(
                        _eq_rng.normal(size=terminal_dim), dtype=jnp.float64
                    )
                    va = np.asarray(eq_eval_x(xt_a, jnp.int32(tt)))
                    vb = np.asarray(eq_eval_x(xt_b, jnp.int32(tt)))
                else:
                    xu_a = jnp.asarray(
                        _eq_rng.normal(size=inner_dim), dtype=jnp.float64
                    )
                    xu_b = jnp.asarray(
                        _eq_rng.normal(size=inner_dim), dtype=jnp.float64
                    )
                    va = np.asarray(eq_eval_xu(xu_a, jnp.int32(tt)))
                    vb = np.asarray(eq_eval_xu(xu_b, jnp.int32(tt)))
                eq_t_present[tt] = bool(np.any(va != 0.0) or np.any(vb != 0.0))
        else:
            eq_t_present = np.zeros(T + 1, dtype=bool)

        # ---------- Per-stage exact Lagrangian Hessian (vmap'd) ----------
        # Single jit'd JAX function over all inner stages; one over the
        # terminal stage. We hand IPOPT the raw exact Hessian (only
        # symmetrized for floating-point cleanliness); IPOPT runs its own
        # inertia correction (Wächter–Biegler §3.4) when the assembled
        # KKT factorization has the wrong inertia.
        nx = (T + 1) * n
        nu = T * m
        x_dim = nx + nu
        ineq_dim = problem.ineq_dim if has_ineq else 0
        eq_dim = problem.eq_dim if has_user_eq else 0

        # ``lam_g`` from IPOPT's perspective is ordered:
        #   init defect (n)
        #   dynamics defects, t=0..T-1 (T*n)
        #   user_eq inner, t=0..T-1 (T*eq_dim)        if has_user_eq
        #   user_eq terminal (eq_dim)                  if has_user_eq
        #   inner ineqs, t=0..T-1 (T*ineq_dim)         if has_ineq
        #   ineq at t=T (ineq_dim)                     if has_ineq
        # We split it inside the hess_lag callback below.

        @jax.jit
        def inner_hess_blocks(
            z_jnp, sigma, lam_dyn_stack, lam_eq_stack, lam_ineq_stack
        ):
            """Per-stage ``(n+m+td, n+m+td)`` exact Lagrangian
            Hessian blocks (joint over ``(x_t, u_t, theta)``).

            ``lam_dyn_stack`` shape ``(T, n)`` (dynamics defect duals).
            ``lam_eq_stack`` shape ``(T, eq_dim)`` (inner user-eq duals)
            — pass an empty ``(T, 0)`` if ``not has_user_eq``.
            ``lam_ineq_stack`` shape ``(T, ineq_dim)`` (inner ineq duals)
            — pass an empty ``(T, 0)`` if ``not has_ineq``.
            """
            X = z_jnp[:nx].reshape(T + 1, n)
            U = z_jnp[nx : nx + nu].reshape(T, m)
            theta = z_jnp[nx + nu : nx + nu + td] if td > 0 else jnp.empty(0)
            ts = jnp.arange(T)

            def stage_lag_hess(x, u, t, lam_dyn_t, lam_eq_t, lam_ineq_t):
                def L_xut(xut):
                    xx = xut[:n]
                    uu = xut[n : n + m]
                    th = xut[n + m : n + m + td] if td > 0 else jnp.empty(0)
                    L = sigma * problem.cost(xx, uu, th, t)
                    L = L - jnp.dot(lam_dyn_t, problem.dynamics(xx, uu, th, t))
                    if has_user_eq:
                        L = L + jnp.dot(lam_eq_t, problem.equalities(xx, uu, th, t))
                    if has_ineq:
                        L = L + jnp.dot(lam_ineq_t, problem.inequalities(xx, uu, th, t))
                    return L

                xut = (
                    jnp.concatenate([x, u, theta])
                    if td > 0
                    else jnp.concatenate([x, u])
                )
                return jax.hessian(L_xut)(xut)

            blocks = jax.vmap(stage_lag_hess, in_axes=(0, 0, 0, 0, 0, 0))(
                X[:T], U, ts, lam_dyn_stack, lam_eq_stack, lam_ineq_stack
            )
            return jax.vmap(lambda H: 0.5 * (H + H.T))(blocks)

        @jax.jit
        def terminal_hess_block(z_jnp, sigma, lam_eq_T, lam_ineq_T):
            """Terminal ``(n+td, n+td)`` exact Lagrangian-Hessian block
            (joint over ``(x_T, theta)``)."""
            X = z_jnp[:nx].reshape(T + 1, n)
            x_T = X[T]
            theta = z_jnp[nx + nu : nx + nu + td] if td > 0 else jnp.empty(0)

            def L_xt(xt):
                xx = xt[:n]
                th = xt[n : n + td] if td > 0 else jnp.empty(0)
                u_zero = jnp.zeros(m, dtype=xx.dtype)
                tT = jnp.int32(T)
                L = sigma * problem.cost(xx, u_zero, th, tT)
                if has_user_eq:
                    L = L + jnp.dot(lam_eq_T, problem.equalities(xx, u_zero, th, tT))
                if has_ineq:
                    L = L + jnp.dot(
                        lam_ineq_T, problem.inequalities(xx, u_zero, th, tT)
                    )
                return L

            xt = jnp.concatenate([x_T, theta]) if td > 0 else x_T
            H = jax.hessian(L_xt)(xt)
            return 0.5 * (H + H.T)

        # ---------- Per-stage callbacks ----------
        # Pin everything in self.* so CasADi's weak-ref bookkeeping
        # doesn't reap the Python wrappers mid-solve.
        self._dyn_cbs: list = []
        self._cost_cbs: list = []
        self._eq_cbs: list = []
        self._ineq_cbs: list = []

        for t in range(T):
            self._dyn_cbs.append(
                _PerStageJaxCallback(
                    f"dyn_{t}",
                    t,
                    n + m + td,
                    n,
                    eval_jit_fn=dyn_eval,
                    jac_jit_fn=dyn_jac,
                )
            )
            self._cost_cbs.append(
                _PerStageJaxCallback(
                    f"cost_{t}",
                    t,
                    n + m + td,
                    1,
                    eval_jit_fn=cost_eval_xu,
                    jac_jit_fn=cost_jac_xu,
                )
            )
            if has_user_eq and eq_t_present[t]:
                self._eq_cbs.append(
                    _PerStageJaxCallback(
                        f"eq_{t}",
                        t,
                        n + m + td,
                        problem.eq_dim,
                        eval_jit_fn=eq_eval_xu,
                        jac_jit_fn=eq_jac_xu,
                    )
                )
            else:
                self._eq_cbs.append(None)  # placeholder
            if has_ineq:
                self._ineq_cbs.append(
                    _PerStageJaxCallback(
                        f"ineq_{t}",
                        t,
                        n + m + td,
                        problem.ineq_dim,
                        eval_jit_fn=ineq_eval_xu,
                        jac_jit_fn=ineq_jac_xu,
                    )
                )

        # Terminal stage (t = T): cost, user-eq, and ineq take (x, theta)
        # — the u-input is fixed at zero inside the per-stage JAX wrappers.
        cost_T = _PerStageJaxCallback(
            f"cost_{T}",
            T,
            n + td,
            1,
            eval_jit_fn=cost_eval_x,
            jac_jit_fn=cost_jac_x,
        )
        self._cost_cbs.append(cost_T)
        eq_T = None
        if has_user_eq and eq_t_present[T]:
            eq_T = _PerStageJaxCallback(
                f"eq_{T}",
                T,
                n + td,
                problem.eq_dim,
                eval_jit_fn=eq_eval_x,
                jac_jit_fn=eq_jac_x,
            )
            self._eq_cbs.append(eq_T)
        elif has_user_eq:
            self._eq_cbs.append(None)
        if has_ineq:
            ineq_T = _PerStageJaxCallback(
                f"ineq_{T}",
                T,
                n + td,
                problem.ineq_dim,
                eval_jit_fn=ineq_eval_x,
                jac_jit_fn=ineq_jac_x,
            )
            self._ineq_cbs.append(ineq_T)

        # ---------- Build the symbolic NLP ----------
        X_sym = ca.MX.sym("X", n, T + 1)
        U_sym = ca.MX.sym("U", m, T)
        Theta_sym = ca.MX.sym("Theta", td) if td > 0 else None

        # Per-stage callback input: vertcat(x_t, u_t, theta) for inner
        # stages and vertcat(x_T, theta) for the terminal. CasADi sees
        # the theta-coupling through every stage's symbolic graph, so
        # IPOPT gets the right cross-stage gradient and Hessian sparsity.
        def _xu_in(t):
            pieces = [X_sym[:, t], U_sym[:, t]]
            if td > 0:
                pieces.append(Theta_sym)
            return ca.vertcat(*pieces)

        def _x_in(t):
            pieces = [X_sym[:, t]]
            if td > 0:
                pieces.append(Theta_sym)
            return ca.vertcat(*pieces)

        defects = [X_sym[:, 0] - ca.DM(np.asarray(problem.x0))]
        cost_terms = []
        eq_terms = []
        ineq_terms = []

        for t in range(T):
            xu = _xu_in(t)
            next_x = self._dyn_cbs[t].cb()(xu)
            defects.append(X_sym[:, t + 1] - next_x)
            cost_terms.append(self._cost_cbs[t].cb()(xu))
            if has_user_eq and eq_t_present[t]:
                eq_terms.append(self._eq_cbs[t].cb()(xu))
            if has_ineq:
                ineq_terms.append(self._ineq_cbs[t].cb()(xu))

        # Terminal stage.
        cost_terms.append(cost_T.cb()(_x_in(T)))
        if has_user_eq and eq_t_present[T]:
            eq_terms.append(eq_T.cb()(_x_in(T)))
        if has_ineq:
            ineq_terms.append(ineq_T.cb()(_x_in(T)))

        f_sym = sum(cost_terms[1:], cost_terms[0])
        # All equality-style constraints (init defect, dynamics defects,
        # user equalities) share lbg=ubg=0; they're concatenated into a
        # single equality block here. The lam_g layout in _eval_lag_hess
        # is the same order: init / dyn / user_eq.
        eq_parts = list(defects)
        if has_user_eq:
            eq_parts.extend(eq_terms)
        g_eq_sym = ca.vertcat(*eq_parts)
        g_ineq_sym = ca.vertcat(*ineq_terms) if ineq_terms else None

        n_eq = g_eq_sym.numel()
        n_ineq = g_ineq_sym.numel() if g_ineq_sym is not None else 0
        if g_ineq_sym is not None:
            g_sym = ca.vertcat(g_eq_sym, g_ineq_sym)
        else:
            g_sym = g_eq_sym

        # Decision vector layout: vec(X), vec(U), then Theta (when td>0).
        # Matches ``_slice_iterate`` and the Lagrangian-Hessian template.
        z_parts = [ca.reshape(X_sym, -1, 1), ca.reshape(U_sym, -1, 1)]
        if td > 0:
            z_parts.append(Theta_sym)
        z_sym = ca.vertcat(*z_parts)

        lbg = np.concatenate([np.zeros(n_eq), -np.inf * np.ones(n_ineq)])
        ubg = np.concatenate([np.zeros(n_eq), np.zeros(n_ineq)])

        # ---------- Build the global Hessian-of-Lagrangian Callback ----------
        # We hand-build the upper-triangular block-diagonal sparsity
        # template, plus a permutation from (rows, cols) build order to
        # CasADi's CCS-stored order (column-major within rows). The
        # Callback's ``eval`` packs per-stage exact-Hessian blocks into
        # this template by writing into the values array in build order
        # and applying the permutation.
        triu_sparsity = _build_block_diag_triu_sparsity(T, n, m, td)

        # Build (rows, cols) in the same order as
        # ``_build_block_diag_triu_sparsity`` so the value-vector layout
        # below matches entry-for-entry. Iteration order:
        #   for t in 0..T-1:  (x_t,u_t)×(x_t,u_t) triu, then (x_t,u_t)×θ
        #   terminal:         (x_T)×(x_T) triu,         then (x_T)×θ
        #   shared θ×θ triu (sum across stages on writeback)
        theta_idx = list(range(nx + nu, nx + nu + td)) if td > 0 else []
        h_rows: list[int] = []
        h_cols: list[int] = []
        for t in range(T):
            x_idx = list(range(t * n, t * n + n))
            u_idx = list(range(nx + t * m, nx + t * m + m))
            block_idx = x_idx + u_idx
            for ri in block_idx:
                for cj in block_idx:
                    if ri <= cj:
                        h_rows.append(ri)
                        h_cols.append(cj)
            for ri in block_idx:
                for cj in theta_idx:
                    h_rows.append(ri)
                    h_cols.append(cj)
        x_idx_T = list(range(T * n, T * n + n))
        for ri in x_idx_T:
            for cj in x_idx_T:
                if ri <= cj:
                    h_rows.append(ri)
                    h_cols.append(cj)
        for ri in x_idx_T:
            for cj in theta_idx:
                h_rows.append(ri)
                h_cols.append(cj)
        for ri in theta_idx:
            for cj in theta_idx:
                if ri <= cj:
                    h_rows.append(ri)
                    h_cols.append(cj)
        h_rows_arr = np.asarray(h_rows, dtype=np.int64)
        h_cols_arr = np.asarray(h_cols, dtype=np.int64)
        h_perm = np.lexsort((h_rows_arr, h_cols_arr)).astype(np.int64)
        n_h_nz = h_perm.shape[0]

        # Per-stage block local-index extractors. Each inner stage's
        # joint block has shape (n+m+td, n+m+td); the writeback splits
        # it into three pieces:
        #   * (x,u)×(x,u) triu:    rows in [0, n+m), cols in [0, n+m)
        #   * (x,u)×θ rectangle:   rows in [0, n+m), cols in [n+m, n+m+td)
        #   * θ×θ accumulator:     rows/cols in [n+m, n+m+td)
        n_inner_xu = (n + m) * (n + m + 1) // 2
        n_inner_xt = (n + m) * td
        n_per_inner = n_inner_xu + n_inner_xt
        n_term_xx = n * (n + 1) // 2
        n_term_xt = n * td
        n_per_term = n_term_xx + n_term_xt
        n_corner = td * (td + 1) // 2
        assert n_h_nz == T * n_per_inner + n_per_term + n_corner, (
            f"hess triu nnz mismatch: {n_h_nz} vs "
            f"T*n_per_inner + n_per_term + n_corner ="
            f" {T * n_per_inner + n_per_term + n_corner}"
        )

        ii_inner_xu = np.empty(n_inner_xu, dtype=np.int64)
        jj_inner_xu = np.empty(n_inner_xu, dtype=np.int64)
        idx = 0
        for ii in range(n + m):
            for jj in range(n + m):
                if ii <= jj:
                    ii_inner_xu[idx] = ii
                    jj_inner_xu[idx] = jj
                    idx += 1
        ii_inner_xt = np.empty(n_inner_xt, dtype=np.int64)
        jj_inner_xt = np.empty(n_inner_xt, dtype=np.int64)
        idx = 0
        for ii in range(n + m):
            for jj in range(td):
                ii_inner_xt[idx] = ii
                jj_inner_xt[idx] = n + m + jj
                idx += 1
        ii_term_xx = np.empty(n_term_xx, dtype=np.int64)
        jj_term_xx = np.empty(n_term_xx, dtype=np.int64)
        idx = 0
        for ii in range(n):
            for jj in range(n):
                if ii <= jj:
                    ii_term_xx[idx] = ii
                    jj_term_xx[idx] = jj
                    idx += 1
        ii_term_xt = np.empty(n_term_xt, dtype=np.int64)
        jj_term_xt = np.empty(n_term_xt, dtype=np.int64)
        idx = 0
        for ii in range(n):
            for jj in range(td):
                ii_term_xt[idx] = ii
                jj_term_xt[idx] = n + jj
                idx += 1
        ii_corner = np.empty(n_corner, dtype=np.int64)
        jj_corner = np.empty(n_corner, dtype=np.int64)
        idx = 0
        for ii in range(td):
            for jj in range(td):
                if ii <= jj:
                    ii_corner[idx] = ii
                    jj_corner[idx] = jj
                    idx += 1

        self._h_perm = h_perm
        self._n_h_nz = n_h_nz

        # Closures for the Callback's eval.
        def _eval_lag_hess(
            x_np: np.ndarray, sigma: float, lam_g_np: np.ndarray
        ) -> np.ndarray:
            """Compute the upper-triangular Lagrangian-Hessian values vector
            in CasADi's stored (column-major) order.
            """
            # Slice IPOPT's ``lam_g`` into init / dyn / user_eq / ineq blocks.
            # Init defect (n) — linear in x_0, contributes 0 Hessian.
            # Dyn defect (T*n) — reshape (T, n).
            # User-eq inner (n_eq_inner_present * eq_dim) + terminal
            # (eq_dim if present) when has_user_eq. Stages absent from
            # eq_t_present contribute zero multipliers in the dense
            # (T, eq_dim) form the Hessian expects.
            # Inner ineq (T*ineq_dim) + terminal ineq (ineq_dim) when has_ineq.
            lam_dyn = lam_g_np[n : n + T * n].reshape(T, n)
            eq_off = n + T * n
            if has_user_eq:
                lam_eq_inner = np.zeros((T, eq_dim), dtype=np.float64)
                cursor = eq_off
                for tt in range(T):
                    if eq_t_present[tt]:
                        lam_eq_inner[tt] = lam_g_np[cursor : cursor + eq_dim]
                        cursor += eq_dim
                if eq_t_present[T]:
                    lam_eq_terminal = lam_g_np[cursor : cursor + eq_dim]
                    cursor += eq_dim
                else:
                    lam_eq_terminal = np.zeros((eq_dim,), dtype=np.float64)
                ineq_off = cursor
            else:
                lam_eq_inner = np.zeros((T, 0), dtype=np.float64)
                lam_eq_terminal = np.zeros((0,), dtype=np.float64)
                ineq_off = eq_off
            if has_ineq:
                lam_ineq_inner = lam_g_np[ineq_off : ineq_off + T * ineq_dim,].reshape(
                    T, ineq_dim
                )
                lam_ineq_terminal = lam_g_np[
                    ineq_off + T * ineq_dim : ineq_off + (T + 1) * ineq_dim,
                ]
            else:
                lam_ineq_inner = np.zeros((T, 0), dtype=np.float64)
                lam_ineq_terminal = np.zeros((0,), dtype=np.float64)

            x_jax = jnp.asarray(x_np)
            sigma_jax = jnp.asarray(sigma, dtype=x_jax.dtype)
            lam_dyn_jax = jnp.asarray(lam_dyn, dtype=x_jax.dtype)
            lam_eq_inner_jax = jnp.asarray(lam_eq_inner, dtype=x_jax.dtype)
            lam_ineq_inner_jax = jnp.asarray(lam_ineq_inner, dtype=x_jax.dtype)

            inner_blocks = np.asarray(
                inner_hess_blocks(
                    x_jax,
                    sigma_jax,
                    lam_dyn_jax,
                    lam_eq_inner_jax,
                    lam_ineq_inner_jax,
                ),
                dtype=np.float64,
            )  # (T, n+m+td, n+m+td)
            term_block = np.asarray(
                terminal_hess_block(
                    x_jax,
                    sigma_jax,
                    jnp.asarray(lam_eq_terminal, dtype=x_jax.dtype),
                    jnp.asarray(lam_ineq_terminal, dtype=x_jax.dtype),
                ),
                dtype=np.float64,
            )  # (n+td, n+td)

            # Pack in build order, then permute to CasADi storage order.
            vals = np.empty(n_h_nz, dtype=np.float64)
            offset = 0
            for t in range(T):
                blk = inner_blocks[t]
                # (x,u)×(x,u) triu
                vals[offset : offset + n_inner_xu] = blk[ii_inner_xu, jj_inner_xu]
                offset += n_inner_xu
                # (x,u)×θ rectangle
                if td > 0:
                    vals[offset : offset + n_inner_xt] = blk[ii_inner_xt, jj_inner_xt]
                    offset += n_inner_xt
            # Terminal (x_T)×(x_T) triu
            vals[offset : offset + n_term_xx] = term_block[ii_term_xx, jj_term_xx]
            offset += n_term_xx
            # Terminal (x_T)×θ rectangle
            if td > 0:
                vals[offset : offset + n_term_xt] = term_block[ii_term_xt, jj_term_xt]
                offset += n_term_xt
            # Shared θ×θ corner = sum over inner stages' bottom-right
            # block + terminal's bottom-right block. Take triu only.
            if td > 0:
                corner = term_block[n : n + td, n : n + td].copy()
                corner += inner_blocks[:, n + m : n + m + td, n + m : n + m + td].sum(
                    axis=0
                )
                vals[offset : offset + n_corner] = corner[ii_corner, jj_corner]
                offset += n_corner
            return vals[h_perm]

        n_z_total = z_sym.numel()
        n_g_total = g_sym.numel()

        class _HessLagCb(ca.Callback):
            """Global Hessian-of-Lagrangian for IPOPT.

            Signature (CasADi's standard for ``hess_lag``):
              in 0: x       (n_z_total, 1)
              in 1: p       (0, 1)               -- no parameters here
              in 2: lam_f   (1, 1)               -- σ
              in 3: lam_g   (n_g_total, 1)
            out 0: triu(H)  (n_z_total, n_z_total) with the block-diagonal
                            triu_sparsity declared on construction.
            """

            def __init__(cb_self):  # noqa: N804
                ca.Callback.__init__(cb_self)
                cb_self.construct("ipopt_mjx_hess_lag", {})

            def get_n_in(cb_self):
                return 4  # noqa: N805

            def get_n_out(cb_self):
                return 1  # noqa: N805

            def get_name_in(cb_self, i):  # noqa: N805
                return ["x", "p", "lam_f", "lam_g"][i]

            def get_name_out(cb_self, _i):  # noqa: N805
                return "triu_h"

            def get_sparsity_in(cb_self, i):  # noqa: N805
                if i == 0:
                    return ca.Sparsity.dense(n_z_total, 1)
                if i == 1:
                    return ca.Sparsity(0, 0)
                if i == 2:
                    return ca.Sparsity.dense(1, 1)
                return ca.Sparsity.dense(n_g_total, 1)

            def get_sparsity_out(cb_self, _i):  # noqa: N805
                return triu_sparsity

            def eval(cb_self, args):  # noqa: N805
                x_np = np.asarray(args[0]).reshape(-1)
                sigma = float(np.asarray(args[2]).reshape(-1)[0])
                lam_g_np = (
                    np.asarray(args[3]).reshape(-1) if n_g_total > 0 else np.zeros(0)
                )
                vals = _eval_lag_hess(x_np, sigma, lam_g_np)
                return [ca.DM(triu_sparsity, vals)]

        self._hess_lag_cb = _HessLagCb()

        ipopt_opts = {
            "print_level": self.print_level,
            "max_iter": self.max_iter,
            "tol": self._effective_tol,
            "constr_viol_tol": self._effective_tol,
            "acceptable_iter": 0,
            # Exact Lagrangian Hessian via our custom ``hess_lag``
            # Callback. The Callback returns the per-stage block-diagonal
            # triu(H) of the raw exact Hessian (symmetrized only); IPOPT
            # handles regularization of indefiniteness via its own
            # inertia-correction loop.
            "hessian_approximation": "exact",
            "sb": "yes",
        }
        if self.timeout_s is not None:
            ipopt_opts["max_wall_time"] = float(self.timeout_s)
        ipopt_opts.update(self.ipopt_extra_options)
        # Per-problem override hook: a problem can put an
        # ``ipopt_mjx_extra_options`` dict in its metadata; we merge it
        # last so problem-specific tuning wins over the adapter's
        # defaults AND the constructor-supplied ``ipopt_extra_options``.
        # This is the same per-problem-settings pattern LIPA uses via
        # ``metadata['lipa_settings']`` — it lets each MJX problem file
        # ship its own ipopt knobs (e.g. ``mu_init``, ``linear_solver``,
        # ``mu_strategy``) without polluting the adapter's signature.
        problem_extra = problem.metadata.get("ipopt_mjx_extra_options")
        if problem_extra:
            ipopt_opts.update(problem_extra)

        nlp = {"x": z_sym, "f": f_sym, "g": g_sym}
        # Per-iter recorder — captures (x, lam_g) at every IPOPT outer
        # iter inside the timed window with negligible overhead. The
        # cost / eq / ineq history arrays are computed AFTER the
        # timed solve closes, in a post-process pass.
        nx_total = z_sym.numel()
        ng_total = g_sym.numel()
        self._iter_recorder = _IterationRecorder(
            nx=nx_total,
            ng=ng_total,
            np_=0,
        )
        opts = {
            "print_time": False,
            "ipopt": ipopt_opts,
            "hess_lag": self._hess_lag_cb,
            "iteration_callback": self._iter_recorder.callback,
        }
        solver = ca.nlpsol("ipopt_mjx_solver", "ipopt", nlp, opts)

        # ---------- Warm start ----------
        z0_parts = [
            np.asarray(problem.X_init).reshape(-1),
            np.asarray(problem.U_init).reshape(-1),
        ]
        if td > 0:
            z0_parts.append(np.asarray(problem.Theta_init).reshape(-1))
        z0 = np.concatenate(z0_parts)

        # Warm up the shared JIT'd functions with a known input so the
        # first IPOPT iter doesn't include compile time. JAX caches the
        # compiled code per (input shape, dtype, static args), and since
        # ``t`` is a runtime jnp.int32 of the same shape every time, ONE
        # warmup call per shared function suffices.
        x0_jnp = jnp.asarray(problem.x0)
        u0_jnp = jnp.zeros(m)
        theta0_jnp = (
            jnp.asarray(problem.Theta_init).reshape(-1) if td > 0 else jnp.empty(0)
        )
        z_xu = (
            jnp.concatenate([x0_jnp, u0_jnp, theta0_jnp])
            if td > 0
            else jnp.concatenate([x0_jnp, u0_jnp])
        )
        z_x = jnp.concatenate([x0_jnp, theta0_jnp]) if td > 0 else x0_jnp
        t0_j = jnp.int32(0)
        _ = jax.block_until_ready(dyn_eval(z_xu, t0_j))
        _ = jax.block_until_ready(dyn_jac(z_xu, t0_j))
        _ = jax.block_until_ready(cost_eval_xu(z_xu, t0_j))
        _ = jax.block_until_ready(cost_jac_xu(z_xu, t0_j))
        _ = jax.block_until_ready(cost_eval_x(z_x, t0_j))
        _ = jax.block_until_ready(cost_jac_x(z_x, t0_j))
        if has_ineq:
            _ = jax.block_until_ready(ineq_eval_xu(z_xu, t0_j))
            _ = jax.block_until_ready(ineq_jac_xu(z_xu, t0_j))
            _ = jax.block_until_ready(ineq_eval_x(z_x, t0_j))
            _ = jax.block_until_ready(ineq_jac_x(z_x, t0_j))
        if has_user_eq:
            _ = jax.block_until_ready(eq_eval_xu(z_xu, t0_j))
            _ = jax.block_until_ready(eq_jac_xu(z_xu, t0_j))
            _ = jax.block_until_ready(eq_eval_x(z_x, t0_j))
            _ = jax.block_until_ready(eq_jac_x(z_x, t0_j))

        # Warm up the per-stage Lagrangian-Hessian JITs as well.
        try:
            z_full_parts = [
                np.asarray(problem.X_init).reshape(-1),
                np.asarray(problem.U_init).reshape(-1),
            ]
            if td > 0:
                z_full_parts.append(np.asarray(problem.Theta_init).reshape(-1))
            z_full = jnp.asarray(np.concatenate(z_full_parts))
            sigma_warm = jnp.asarray(1.0, dtype=z_full.dtype)
            lam_dyn_warm = jnp.zeros((T, n), dtype=z_full.dtype)
            lam_eq_inner_warm = jnp.zeros((T, eq_dim), dtype=z_full.dtype)
            lam_eq_term_warm = jnp.zeros((eq_dim,), dtype=z_full.dtype)
            lam_ineq_inner_warm = jnp.zeros((T, ineq_dim), dtype=z_full.dtype)
            lam_ineq_term_warm = jnp.zeros((ineq_dim,), dtype=z_full.dtype)
            _ = jax.block_until_ready(
                inner_hess_blocks(
                    z_full,
                    sigma_warm,
                    lam_dyn_warm,
                    lam_eq_inner_warm,
                    lam_ineq_inner_warm,
                ),
            )
            _ = jax.block_until_ready(
                terminal_hess_block(
                    z_full, sigma_warm, lam_eq_term_warm, lam_ineq_term_warm
                ),
            )
        except Exception:  # noqa: BLE001
            # If warmup fails, just let the first eval pay the JIT cost.
            pass

        start = timer()
        sol = solver(x0=z0, lbg=lbg, ubg=ubg)
        solve_time_ms = 1e3 * (timer() - start)

        z_val = np.asarray(sol["x"]).reshape(-1)
        X, U, Theta = _slice_iterate(z_val, problem)

        # Multipliers — split lam_g into the documented blocks
        # (init, dyn, inner ineqs, terminal ineq) and splice into
        # evaluate_problem's eq / ineq stacks. Sign-flip the dyn
        # block (IPOPT encodes X[t+1] - dyn(...) opposite from
        # evaluator's convention).
        lam_g_full = np.asarray(sol["lam_g"]).reshape(-1)
        eq_full_size = n + T * n + (T + 1) * problem.eq_dim
        ineq_full_size = (T + 1) * problem.ineq_dim if has_ineq else 0
        multipliers_eq = np.zeros(eq_full_size, dtype=np.float64)
        multipliers_ineq = (
            np.zeros(ineq_full_size, dtype=np.float64) if has_ineq else np.zeros(0)
        )
        # init block
        multipliers_eq[:n] = lam_g_full[:n]
        # dyn block (sign-flipped)
        if T > 0:
            multipliers_eq[n : n + T * n] = -lam_g_full[n : n + T * n]
        # ineq blocks: per-stage inner [t<T] then terminal [t=T]
        if has_ineq and ineq_full_size > 0:
            inner_offset = n + T * n
            for t in range(T):
                multipliers_ineq[t * problem.ineq_dim : (t + 1) * problem.ineq_dim] = (
                    lam_g_full[
                        inner_offset + t * problem.ineq_dim : inner_offset
                        + (t + 1) * problem.ineq_dim
                    ]
                )
            term_offset = inner_offset + T * problem.ineq_dim
            multipliers_ineq[T * problem.ineq_dim : (T + 1) * problem.ineq_dim] = (
                lam_g_full[term_offset : term_offset + problem.ineq_dim]
            )

        stats = solver.stats()
        iters = int(stats.get("iter_count", -1))
        success = bool(stats.get("success", False))

        # Per-iter histories (post-process; outside the timed window).
        iterates_xut = None
        recorded = self._iter_recorder.iterates
        if recorded:
            iterates_xut = [
                _slice_iterate(x_iter, problem) for x_iter, _lam in recorded
            ]

        return pack_solver_result(
            solver_name=self.name,
            problem_name=problem.name,
            problem=problem,
            X=X,
            U=U,
            Theta=Theta,
            iterations=iters,
            solve_time_ms=solve_time_ms,
            success=success,
            notes=stats.get("return_status", ""),
            multipliers_eq=multipliers_eq,
            multipliers_ineq=multipliers_ineq if has_ineq else None,
            iterates_xut=iterates_xut,
        )


@register("ipopt-jax")
def _factory(**kwargs) -> SolverAdapter:
    return IpoptMjxSparseAdapter(**kwargs)
