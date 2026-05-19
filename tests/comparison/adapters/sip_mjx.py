"""sip_python adapter for MJX problems (sip-python >= 0.0.3).

Design overview
---------------

The analytical ``sip.py`` adapter does a global eigendecomposition on
the Lagrangian Hessian for PSD projection. That is fine for small
analytical OCPs but prohibitive at MJX scale, where ``x_dim`` is much
larger and a global eigh on the full Hessian is cubic in the horizon.

Three structural choices make this tractable:

1. **Per-stage block-diagonal Lagrangian Hessian** (cheap PSD floor)

   For a multi-shooting OCP with stage-decomposable cost, dynamics, and
   inequalities, the Lagrangian
       L(z, y, zd) = sum_t cost_t(x_t, u_t)
                   + y_init · (x_0 - x0)
                   + sum_t y_dyn_t · (x_{t+1} - dyn(x_t, u_t, t))
                   + sum_t zd_t · ineq_t(x_t, u_t)
   is exactly **block-diagonal in stages** w.r.t. z = (vec(X), vec(U)):
     * init defect contributes 0 (linear in x_0).
     * dynamics defect contributes only via the nonlinear ``-y_t · dyn``
       term — a (n+m)x(n+m) block on (x_t, u_t).
     * cost and inequality terms similarly contribute one
       (n+m)x(n+m) block per stage.
     * Terminal stage: (n, n) block on x_T only.
   NO cross-stage coupling.

   We vmap the per-stage Hessian-and-PSD-clamp over stages; eigh on each
   small ~73x73 block is cheap (~milliseconds). Assembly into a global
   sparse upper-triangle CSC follows the documented block-diagonal
   layout.

2. **Block-banded c-Jacobian, block-diagonal g-Jacobian** (sparse)

   c (init defect + dynamics defects) has only identity-on-X_{t+1} and
   ``-jac(dyn)`` blocks on (X_t, U_t). g has per-stage ineq jacobian
   blocks. Both are hand-assembled from (rows, cols) lists into
   CSR templates with the per-call data filled in place.

3. **All sip primitives in float64 via JAX** (sip-python 0.0.3 native)

   sip-python 0.0.3 accepts JAX scalars directly for ``mco.f`` and
   numpy arrays for everything else. The canonical callback pattern
   from upstream's ``tests/test_simple_constrained_lqr.py`` is used
   verbatim: multiple jit'd JAX functions (cost / grad / c / g /
   jac_c / jac_g / hess), each invoked separately inside the model
   callback, with their outputs written into pre-allocated
   ``scipy.sparse`` buffers via in-place ``.data`` mutation.

Conventions (matching upstream tests)
-------------------------------------
* Jacobians as ``scipy.sparse.csr_matrix`` of shape ``(out_dim, x_dim)``,
  with ``ProblemDimensions.is_jacobian_*_transposed = True`` so the C++
  side reinterprets the row-major CSR data as column-major CSC of the
  transpose.
* Hessian as ``scipy.sparse.csc_matrix`` upper triangle of shape
  ``(x_dim, x_dim)``.

Skips cleanly when:
  * ``metadata.get("is_mjx") is not True`` — use ``sip`` for analytical.
  * ``problem.theta_dim > 0`` — sip doesn't support cross-stage vars.
  * ``problem.equalities is not None`` — only init+dyn defects supported
    in this version of the assembly.
"""

from __future__ import annotations

import os
from timeit import default_timer as timer
from typing import Optional

import numpy as np

from tests.comparison.adapters import register
from tests.comparison.adapters.base import SolverAdapter
from tests.comparison.problem_spec import (
    ProblemSpec,
    SolverResult,
    make_failure_result,
    pack_solver_result,
)
from tests.comparison.warm_starts import rollout_warm_start


def _import_sip():
    import sip_python  # noqa: F401

    return sip_python


def _import_jax():
    import jax  # noqa: F401
    import jax.numpy as jnp  # noqa: F401

    jax.config.update("jax_enable_x64", True)
    return jax, jnp


def _slice_iterate(z_val: np.ndarray, problem: ProblemSpec):
    n, m, T = problem.n, problem.m, problem.T
    nx = n * (T + 1)
    nu = m * T
    X = z_val[:nx].reshape(T + 1, n)
    U = z_val[nx:nx + nu].reshape(T, m)
    Theta = np.zeros(0, dtype=z_val.dtype)
    return X, U, Theta


class SipMjxAdapter(SolverAdapter):
    """sip_python adapter specialized for MJX-scale OCPs.

    See module docstring for design overview.
    """

    name = "sip-mjx"

    def __init__(
        self,
        max_iter: int = 1000,
        tol: float = 1e-6,
        # Floor for the per-stage Lagrangian-Hessian PSD projection.
        psd_reg_delta: float = 1e-6,
        enable_elastics: bool = False,
        elastic_var_cost_coeff: float = 1e6,
        # Penalty / barrier defaults aligned with LIPA's MJX configs.
        # h1 problems prefer a smaller initial_penalty_parameter +
        # faster ramp; per-problem overrides handle that via
        # ``metadata['sip_mjx_extra_settings']``.
        penalty_parameter_increase_factor: float = 1.1,
        mu_update_factor: float = 0.95,
        initial_mu: float = 1e-2,
        initial_penalty_parameter: float = 1e9,
        # Skip the forward rollout: on MJX problems the
        # problem-shipped reference X_init is a much better warm
        # start than the rolled-out (|c|=0, huge f) trajectory.
        warmstart_via_rollout: bool = False,
        # Tolerate LS_FAILURE and keep iterating; the near-stationary
        # plateau at convergence backtracks heavily but still makes
        # progress per outer iter.
        line_search_failures_enabled: bool = True,
        max_ls_iterations: int = 100000,
        print_logs: bool = False,
        sip_extra_settings: Optional[dict] = None,
    ) -> None:
        self.max_iter = max_iter
        self.tol = tol
        self.psd_reg_delta = float(psd_reg_delta)
        self.enable_elastics = bool(enable_elastics)
        self.elastic_var_cost_coeff = float(elastic_var_cost_coeff)
        self.penalty_parameter_increase_factor = float(penalty_parameter_increase_factor)
        self.mu_update_factor = float(mu_update_factor)
        self.initial_mu = float(initial_mu)
        self.initial_penalty_parameter = float(initial_penalty_parameter)
        self.warmstart_via_rollout = bool(warmstart_via_rollout)
        self.line_search_failures_enabled = bool(line_search_failures_enabled)
        self.max_ls_iterations = int(max_ls_iterations)
        self.print_logs = bool(print_logs)
        self.sip_extra_settings = sip_extra_settings or {}

    def is_available(self) -> tuple[bool, str]:
        try:
            _import_sip()
        except ImportError as e:
            return False, f"{e}"
        return True, ""

    def solve(self, problem: ProblemSpec) -> SolverResult:  # noqa: PLR0915, PLR0912
        avail, reason = self.is_available()
        if not avail:
            return make_failure_result(
                self.name, problem.name, f"unavailable: {reason}",
            )

        if not problem.metadata.get("is_mjx"):
            return make_failure_result(
                self.name, problem.name,
                "sip-mjx adapter is MJX-only; this problem has no "
                "metadata['is_mjx'] flag. Use the 'sip' adapter for "
                "analytical problems.",
            )

        if problem.theta_dim > 0:
            return make_failure_result(
                self.name, problem.name,
                f"sip-mjx adapter does not support cross-stage Theta "
                f"(theta_dim={problem.theta_dim})",
            )

        if problem.equalities is not None and problem.eq_dim > 0:
            return make_failure_result(
                self.name, problem.name,
                "sip-mjx adapter currently only supports OCPs whose "
                "equalities come from init defect + dynamics defects; "
                "got problem.equalities != None which would require "
                "extending the block-banded Jacobian assembly.",
            )

        sip = _import_sip()
        jax, jnp = _import_jax()
        from scipy import sparse as sp

        T, n, m = problem.T, problem.n, problem.m
        nx = n * (T + 1)
        nu = m * T
        x_dim = nx + nu

        cost_fn = problem.cost
        dyn_fn = problem.dynamics
        ineq_fn = problem.inequalities
        has_ineq = ineq_fn is not None and problem.ineq_dim > 0
        ineq_dim = problem.ineq_dim if has_ineq else 0

        theta_const = jnp.asarray(problem.Theta_init)
        x0_const = jnp.asarray(problem.x0)

        y_dim = n + T * n
        s_dim = (T + 1) * ineq_dim if has_ineq else 0

        psd_reg = self.psd_reg_delta

        # ===== Per-stage JAX primitives (jit'd; one trace each) =====
        @jax.jit
        def stage_cost(x, u, t):
            return cost_fn(x, u, theta_const, t)

        @jax.jit
        def stage_dyn(x, u, t):
            return dyn_fn(x, u, theta_const, t)

        @jax.jit
        def terminal_cost(x_T):
            u_zero = jnp.zeros(m, dtype=x_T.dtype)
            return cost_fn(x_T, u_zero, theta_const, jnp.int32(T))

        if has_ineq:
            @jax.jit
            def stage_ineq(x, u, t):
                return ineq_fn(x, u, theta_const, t)

            @jax.jit
            def terminal_ineq(x_T):
                u_zero = jnp.zeros(m, dtype=x_T.dtype)
                return ineq_fn(x_T, u_zero, theta_const, jnp.int32(T))

        # ===== Flat-NLP JAX functions =====

        @jax.jit
        def f_fn(z):
            X = z[:nx].reshape(T + 1, n)
            U = z[nx:nx + nu].reshape(T, m)
            ts = jnp.arange(T)
            inner = jax.vmap(stage_cost, in_axes=(0, 0, 0))(X[:T], U, ts)
            return jnp.sum(inner) + terminal_cost(X[T])

        @jax.jit
        def grad_f_fn(z):
            return jax.grad(f_fn)(z)

        @jax.jit
        def c_fn(z):
            X = z[:nx].reshape(T + 1, n)
            U = z[nx:nx + nu].reshape(T, m)
            ts = jnp.arange(T)
            init_defect = X[0] - x0_const
            dyn_next = jax.vmap(stage_dyn, in_axes=(0, 0, 0))(X[:T], U, ts)
            dyn_defects = (X[1:] - dyn_next).reshape(-1)
            return jnp.concatenate([init_defect, dyn_defects])

        # Per-stage dyn Jacobians stacked: (T, n, n+m)
        @jax.jit
        def dyn_jac_blocks(z):
            X = z[:nx].reshape(T + 1, n)
            U = z[nx:nx + nu].reshape(T, m)
            ts = jnp.arange(T)

            def jac_one(x_in, u_in, t_in):
                return jax.jacrev(
                    lambda xu: stage_dyn(xu[:n], xu[n:n + m], t_in),
                )(jnp.concatenate([x_in, u_in]))

            return jax.vmap(jac_one, in_axes=(0, 0, 0))(X[:T], U, ts)

        if has_ineq:
            @jax.jit
            def g_fn(z):
                X = z[:nx].reshape(T + 1, n)
                U = z[nx:nx + nu].reshape(T, m)
                ts = jnp.arange(T)
                inner = jax.vmap(stage_ineq, in_axes=(0, 0, 0))(X[:T], U, ts)
                term = terminal_ineq(X[T])
                return jnp.concatenate([inner.reshape(-1), term.reshape(-1)])

            # Stacked (T, ineq_dim, n+m) for inner stages.
            @jax.jit
            def ineq_jac_blocks(z):
                X = z[:nx].reshape(T + 1, n)
                U = z[nx:nx + nu].reshape(T, m)
                ts = jnp.arange(T)

                def jac_one(x_in, u_in, t_in):
                    return jax.jacrev(
                        lambda xu: stage_ineq(xu[:n], xu[n:n + m], t_in),
                    )(jnp.concatenate([x_in, u_in]))

                return jax.vmap(jac_one, in_axes=(0, 0, 0))(X[:T], U, ts)

            @jax.jit
            def terminal_ineq_jac(z):
                X = z[:nx].reshape(T + 1, n)
                return jax.jacrev(terminal_ineq)(X[T])

        # Per-stage Lagrangian Hessian (PSD-projected). Inner (T,n+m,n+m).
        @jax.jit
        def inner_hess_blocks(z, y, zd):
            X = z[:nx].reshape(T + 1, n)
            U = z[nx:nx + nu].reshape(T, m)
            ts = jnp.arange(T)
            Y_dyn = y[n:n + T * n].reshape(T, n)
            if has_ineq:
                Z_inner = zd[:T * ineq_dim].reshape(T, ineq_dim)
            else:
                Z_inner = jnp.zeros((T, 0), dtype=z.dtype)

            def stage_lag_hess(x, u, t, y_dyn_t, z_ineq_t):
                def L_xu(xu):
                    xx = xu[:n]
                    uu = xu[n:n + m]
                    terms = stage_cost(xx, uu, t) - jnp.dot(
                        y_dyn_t, stage_dyn(xx, uu, t),
                    )
                    if has_ineq:
                        terms = terms + jnp.dot(z_ineq_t, stage_ineq(xx, uu, t))
                    return terms

                return jax.hessian(L_xu)(jnp.concatenate([x, u]))

            def proj_psd_block(H):
                H = 0.5 * (H + H.T)
                S, V = jnp.linalg.eigh(H)
                return (V * jnp.maximum(S, psd_reg)) @ V.T

            blocks = jax.vmap(stage_lag_hess, in_axes=(0, 0, 0, 0, 0))(
                X[:T], U, ts, Y_dyn, Z_inner,
            )
            return jax.vmap(proj_psd_block)(blocks)

        @jax.jit
        def terminal_hess_block(z, zd):
            X = z[:nx].reshape(T + 1, n)
            X_T = X[T]
            if has_ineq:
                Z_T = zd[T * ineq_dim:]
            else:
                Z_T = jnp.zeros((0,), dtype=z.dtype)

            def L_x(xx):
                terms = terminal_cost(xx)
                if has_ineq:
                    terms = terms + jnp.dot(Z_T, terminal_ineq(xx))
                return terms

            H = jax.hessian(L_x)(X_T)
            H = 0.5 * (H + H.T)
            S, V = jnp.linalg.eigh(H)
            return (V * jnp.maximum(S, psd_reg)) @ V.T

        # ===== Build sparse matrix templates =====
        # Equality (c) Jacobian: init(n) ++ dyn_t(n) for t in 0..T-1.
        # Cols of z: vec(X) (n*(T+1)) then vec(U) (m*T).
        c_rows: list[int] = []
        c_cols: list[int] = []
        for j in range(n):
            c_rows.append(j); c_cols.append(j)  # init identity
        for t in range(T):
            row_off = n + t * n
            for i in range(n):
                c_rows.append(row_off + i); c_cols.append((t + 1) * n + i)
            for i in range(n):
                for j in range(n):
                    c_rows.append(row_off + i); c_cols.append(t * n + j)
            for i in range(n):
                for j in range(m):
                    c_rows.append(row_off + i); c_cols.append(nx + t * m + j)
        c_rows_arr = np.asarray(c_rows, dtype=np.int32)
        c_cols_arr = np.asarray(c_cols, dtype=np.int32)
        jac_c_template = sp.csr_matrix(
            (np.ones(c_rows_arr.shape[0], dtype=np.float64),
             (c_rows_arr, c_cols_arr)),
            shape=(y_dim, x_dim),
        )

        if has_ineq:
            g_rows: list[int] = []
            g_cols: list[int] = []
            for t in range(T):
                row_off = t * ineq_dim
                for i in range(ineq_dim):
                    for j in range(n):
                        g_rows.append(row_off + i); g_cols.append(t * n + j)
                for i in range(ineq_dim):
                    for j in range(m):
                        g_rows.append(row_off + i); g_cols.append(nx + t * m + j)
            row_off = T * ineq_dim
            for i in range(ineq_dim):
                for j in range(n):
                    g_rows.append(row_off + i); g_cols.append(T * n + j)
            g_rows_arr = np.asarray(g_rows, dtype=np.int32)
            g_cols_arr = np.asarray(g_cols, dtype=np.int32)
            jac_g_template = sp.csr_matrix(
                (np.ones(g_rows_arr.shape[0], dtype=np.float64),
                 (g_rows_arr, g_cols_arr)),
                shape=(s_dim, x_dim),
            )
        else:
            g_rows_arr = np.zeros(0, dtype=np.int32)
            g_cols_arr = np.zeros(0, dtype=np.int32)
            jac_g_template = sp.csr_matrix((s_dim, x_dim))

        # Lagrangian Hessian upper triangle (block-diagonal in stages).
        h_rows: list[int] = []
        h_cols: list[int] = []
        for t in range(T):
            x_idx = list(range(t * n, t * n + n))
            u_idx = list(range(nx + t * m, nx + t * m + m))
            block_idx = x_idx + u_idx
            for ri in block_idx:
                for cj in block_idx:
                    if ri <= cj:
                        h_rows.append(ri); h_cols.append(cj)
        x_idx_T = list(range(T * n, T * n + n))
        for ri in x_idx_T:
            for cj in x_idx_T:
                if ri <= cj:
                    h_rows.append(ri); h_cols.append(cj)
        h_rows_arr = np.asarray(h_rows, dtype=np.int32)
        h_cols_arr = np.asarray(h_cols, dtype=np.int32)
        upp_hess_template = sp.csc_matrix(
            (np.ones(h_rows_arr.shape[0], dtype=np.float64),
             (h_rows_arr, h_cols_arr)),
            shape=(x_dim, x_dim),
        )

        # ===== Pre-compute per-call permutations from (rows,cols) order to
        #       canonical CSR/CSC order (so we can write .data in our
        #       natural assembly order). =====
        # Build a per-template "perm" such that ``template.data ==
        # vals_in_natural_order[perm]`` after the COO->CSR/CSC conversion.
        # Numerically: we have indices_natural = (rows, cols) in our build
        # order; scipy.sparse.coo_matrix(...).tocsr() returns data
        # sorted by (row, col within row). We compute perm such that
        # natural_order[perm] == sorted_order.
        def _csr_perm(rows: np.ndarray, cols: np.ndarray, n_rows: int) -> np.ndarray:
            # Stable sort by (row, col)
            order = np.lexsort((cols, rows))
            return order.astype(np.int64)

        def _csc_perm(rows: np.ndarray, cols: np.ndarray, n_cols: int) -> np.ndarray:
            order = np.lexsort((rows, cols))
            return order.astype(np.int64)

        c_perm = _csr_perm(c_rows_arr, c_cols_arr, y_dim)
        h_perm = _csc_perm(h_rows_arr, h_cols_arr, x_dim)
        if has_ineq:
            g_perm = _csr_perm(g_rows_arr, g_cols_arr, s_dim)
        else:
            g_perm = np.zeros(0, dtype=np.int64)

        # Reusable buffers (mirror upstream pattern: same scipy sparse
        # object reused per call, ``.data`` mutated in place).
        jac_c_buf = jac_c_template.copy()
        jac_g_buf = jac_g_template.copy()
        upp_hess_buf = upp_hess_template.copy()

        # ===== ProblemDimensions / QDLDLSettings (0.0.3 single call) =====
        pd = sip.ProblemDimensions()
        pd.x_dim = x_dim
        pd.s_dim = s_dim
        pd.y_dim = y_dim
        pd.upper_hessian_lagrangian_nnz = upp_hess_template.nnz
        pd.jacobian_c_nnz = jac_c_template.nnz
        pd.jacobian_g_nnz = jac_g_template.nnz
        pd.is_jacobian_c_transposed = True
        pd.is_jacobian_g_transposed = True

        qs = sip.QDLDLSettings()
        qs.permute_kkt_system = True

        # AMD-first KKT permutation (falls back to RCM when cvxopt is
        # unavailable). Lives in tests/comparison/sip_kkt_perm.py so the
        # analytical sip adapter can reuse it.
        from tests.comparison.sip_kkt_perm import compute_kkt_perm_inv_and_nnzs
        _perm_result = compute_kkt_perm_inv_and_nnzs(
            upp_hess_template, jac_c_template, jac_g_template,
        )
        qs.kkt_pinv = _perm_result.perm_inv
        pd.kkt_nnz = _perm_result.kkt_nnz
        pd.kkt_L_nnz = _perm_result.L_nnz

        # ===== Settings =====
        ss = sip.Settings()
        ss.max_iterations = int(self.max_iter)
        ss.max_kkt_violation = float(self.tol)
        ss.enable_elastics = self.enable_elastics
        if self.enable_elastics:
            ss.elastic_var_cost_coeff = self.elastic_var_cost_coeff
        ss.penalty_parameter_increase_factor = self.penalty_parameter_increase_factor
        ss.mu_update_factor = self.mu_update_factor
        ss.initial_mu = self.initial_mu
        ss.initial_penalty_parameter = self.initial_penalty_parameter
        ss.enable_line_search_failures = self.line_search_failures_enabled
        ss.max_ls_iterations = self.max_ls_iterations
        ss.print_logs = self.print_logs
        if not self.print_logs:
            ss.print_line_search_logs = False
            ss.print_search_direction_logs = False
            ss.print_derivative_check_logs = False
        ss.assert_checks_pass = False
        # Per-problem override hook (same pattern as ipopt_mjx_sparse.py).
        # A problem can put a ``sip_mjx_extra_settings`` dict in its
        # metadata; we apply it AFTER the constructor-supplied
        # ``sip_extra_settings`` so per-problem tuning wins.
        for k, v in self.sip_extra_settings.items():
            setattr(ss, k, v)
        problem_extra = problem.metadata.get("sip_mjx_extra_settings")
        if problem_extra:
            for k, v in problem_extra.items():
                setattr(ss, k, v)

        # ===== Safety cap on |x| (avoid MJX vmap process-killing) =====
        # At extreme step magnitudes, vmap'd MJX dynamics has been
        # observed to SEGFAULT rather than return NaN. Cap |x|max at
        # ~10x typical warm-start magnitude (varies per problem) and
        # return a sentinel that forces sip's line search to backtrack.
        x_norm_safety_cap = 1e5

        debug = os.environ.get("SIP_MJX_DEBUG", "") == "1"
        _call_idx = [0]
        # Track the last "well-behaved" iterate from inside mc. If
        # sip ends with a non-SOLVED status, vars_in.x might be a
        # corrupted iterate (e.g. NaN-infested) — fall back to this
        # snapshot for cost / constraint reporting.
        last_good_x = [None]
        best_f = [float("inf")]
        best_x = [None]
        # Per-iter recorder. Append the iterate AT EACH mc call so we
        # can post-process the snapshots into cost / eq / ineq history
        # arrays after the timed solve closes.
        iter_xut: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

        # ===== Model callback (mirrors upstream test pattern) =====
        def mc(mci):
            mco = sip.ModelCallbackOutput()

            x_np = np.array(mci.x, dtype=np.float64)
            x_inf = float(np.max(np.abs(x_np))) if x_np.size else 0.0
            x_finite = bool(np.all(np.isfinite(x_np)))

            # Per-iter snapshot — capture even non-finite iterates so
            # the history reflects the line-search failures honestly.
            try:
                Xi, Ui, Ti = _slice_iterate(x_np.copy(), problem)
                iter_xut.append((Xi, Ui, Ti))
            except Exception:  # noqa: BLE001
                pass

            if debug:
                _call_idx[0] += 1
                print(
                    f"  mc#{_call_idx[0] - 1} |x|={x_inf:.2e} fin={x_finite}",
                    flush=True,
                )

            if (not x_finite) or x_inf > x_norm_safety_cap:
                mco.f = 1e30
                mco.gradient_f = np.zeros(x_dim, dtype=np.float64)
                mco.c = np.full(y_dim, 1e8, dtype=np.float64)
                mco.g = (
                    np.full(s_dim, 1e8, dtype=np.float64)
                    if has_ineq else np.zeros(0, dtype=np.float64)
                )
                mco.jacobian_c = jac_c_buf
                mco.jacobian_g = jac_g_buf
                mco.upper_hessian_lagrangian = upp_hess_buf
                return mco

            x_jax = mci.x  # numpy view; jit'd JAX functions accept it directly
            y_jax = mci.y
            zd_jax = mci.z if has_ineq else np.zeros(0, dtype=np.float64)

            # Cost / grad / c / g — set as JAX arrays per upstream pattern.
            mco.f = f_fn(x_jax)
            mco.gradient_f = np.array(grad_f_fn(x_jax))
            mco.c = np.array(c_fn(x_jax))
            if has_ineq:
                mco.g = np.array(g_fn(x_jax))
            else:
                mco.g = np.zeros(0, dtype=np.float64)

            # Track best iterate by total constraint+cost merit. We use
            # f + 10·sum(|c|) + 10·sum(max(0,g)) — a crude P1 merit.
            f_val = float(np.asarray(mco.f, dtype=np.float64))
            c_norm = float(np.sum(np.abs(mco.c)))
            g_pos = (
                float(np.sum(np.maximum(mco.g, 0.0))) if has_ineq else 0.0
            )
            merit = f_val + 10.0 * c_norm + 10.0 * g_pos
            if np.isfinite(merit) and merit < best_f[0]:
                best_f[0] = merit
                best_x[0] = x_np.copy()
            last_good_x[0] = x_np  # always keep last finite x

            # ----- c-Jacobian -----
            dyn_J = np.asarray(dyn_jac_blocks(x_jax), dtype=np.float64)
            c_vals = np.empty(c_rows_arr.shape[0], dtype=np.float64)
            idx = 0
            c_vals[idx:idx + n] = 1.0; idx += n
            for t in range(T):
                c_vals[idx:idx + n] = 1.0; idx += n
                c_vals[idx:idx + n * n] = -dyn_J[t, :, :n].reshape(-1)
                idx += n * n
                c_vals[idx:idx + n * m] = -dyn_J[t, :, n:n + m].reshape(-1)
                idx += n * m
            jac_c_buf.data[:] = c_vals[c_perm]
            mco.jacobian_c = jac_c_buf

            # ----- g-Jacobian -----
            if has_ineq:
                ineq_J = np.asarray(ineq_jac_blocks(x_jax), dtype=np.float64)
                ineq_J_T = np.asarray(terminal_ineq_jac(x_jax), dtype=np.float64)
                g_vals = np.empty(g_rows_arr.shape[0], dtype=np.float64)
                idx = 0
                for t in range(T):
                    g_vals[idx:idx + ineq_dim * n] = ineq_J[t, :, :n].reshape(-1)
                    idx += ineq_dim * n
                    g_vals[idx:idx + ineq_dim * m] = ineq_J[t, :, n:n + m].reshape(-1)
                    idx += ineq_dim * m
                g_vals[idx:idx + ineq_dim * n] = ineq_J_T.reshape(-1)
                idx += ineq_dim * n
                jac_g_buf.data[:] = g_vals[g_perm]
                mco.jacobian_g = jac_g_buf
            else:
                mco.jacobian_g = jac_g_buf

            # ----- Lagrangian Hessian -----
            inner_blocks = np.asarray(
                inner_hess_blocks(x_jax, y_jax, zd_jax), dtype=np.float64,
            )
            terminal_block = np.asarray(
                terminal_hess_block(x_jax, zd_jax), dtype=np.float64,
            )
            h_vals = np.empty(h_rows_arr.shape[0], dtype=np.float64)
            idx = 0
            for t in range(T):
                blk = inner_blocks[t]
                for ii in range(n + m):
                    if ii < n:
                        ri = t * n + ii
                    else:
                        ri = nx + t * m + (ii - n)
                    for jj in range(n + m):
                        if jj < n:
                            cj = t * n + jj
                        else:
                            cj = nx + t * m + (jj - n)
                        if ri <= cj:
                            h_vals[idx] = blk[ii, jj]
                            idx += 1
            for ii in range(n):
                for jj in range(n):
                    ri = T * n + ii
                    cj = T * n + jj
                    if ri <= cj:
                        h_vals[idx] = terminal_block[ii, jj]
                        idx += 1
            upp_hess_buf.data[:] = h_vals[h_perm]
            mco.upper_hessian_lagrangian = upp_hess_buf

            return mco

        # ===== Construct solver =====
        solver = sip.Solver(ss, qs, pd, mc)

        # ===== Warm start =====
        # Forward-rollout warm start (with fallback to LIPA-shipped
        # X_init on non-finite values or dynamics errors) lives in
        # tests.comparison.warm_starts.rollout_warm_start.
        if self.warmstart_via_rollout:
            X_roll, U_roll = rollout_warm_start(problem, fallback_to_x_init=True)
            z_init = np.concatenate([X_roll.reshape(-1), U_roll.reshape(-1)])
        else:
            z_init = np.concatenate([
                np.asarray(problem.X_init, dtype=np.float64).reshape(-1),
                np.asarray(problem.U_init, dtype=np.float64).reshape(-1),
            ])

        vars_in = sip.Variables(pd)
        vars_in.x[:] = z_init
        vars_in.s[:] = 1.0
        vars_in.y[:] = 0.0
        vars_in.e[:] = 0.0
        vars_in.z[:] = 1.0

        # ===== Warm-up jit caches on z_init =====
        try:
            z_warm = jnp.asarray(z_init, dtype=jnp.float64)
            y_warm = jnp.zeros(y_dim, dtype=jnp.float64)
            zd_warm = (
                jnp.ones(s_dim, dtype=jnp.float64)
                if has_ineq else jnp.zeros(0, dtype=jnp.float64)
            )
            jax.block_until_ready(f_fn(z_warm))
            jax.block_until_ready(grad_f_fn(z_warm))
            jax.block_until_ready(c_fn(z_warm))
            jax.block_until_ready(dyn_jac_blocks(z_warm))
            if has_ineq:
                jax.block_until_ready(g_fn(z_warm))
                jax.block_until_ready(ineq_jac_blocks(z_warm))
                jax.block_until_ready(terminal_ineq_jac(z_warm))
            jax.block_until_ready(inner_hess_blocks(z_warm, y_warm, zd_warm))
            jax.block_until_ready(terminal_hess_block(z_warm, zd_warm))
        except Exception:  # noqa: BLE001
            pass

        # ===== Solve =====
        start = timer()
        try:
            output = solver.solve(vars_in)
            err = ""
        except Exception as e:  # noqa: BLE001
            output = None
            err = f"{type(e).__name__}: {e}"
        solve_time_ms = 1e3 * (timer() - start)

        z_val = np.asarray(vars_in.x, dtype=np.float64).copy()

        # If sip exited with a corrupted iterate (NaN / inf / huge), or
        # if a non-SOLVED status leaves vars_in.x at the iterate where
        # line-search failed, fall back to the best clean iterate from
        # inside mc.
        z_val_finite = bool(np.all(np.isfinite(z_val)))
        z_val_norm = float(np.max(np.abs(z_val))) if z_val.size else 0.0
        bad_exit = (
            (output is not None and output.exit_status != sip.Status.SOLVED)
            or (not z_val_finite)
            or (z_val_norm > x_norm_safety_cap)
        )
        if bad_exit and best_x[0] is not None:
            z_val = best_x[0].copy()

        X, U, Theta = _slice_iterate(z_val, problem)

        # Multipliers — sip-mjx doesn't filter rows so the mapping is
        # direct. The c() function encodes the dyn defect as
        # X[t+1] - dyn(...) (opposite from evaluate_problem) so we
        # sign-flip those rows. Init / user-eq / ineq match.
        try:
            multipliers_eq_full = np.asarray(
                vars_in.y, dtype=np.float64,
            ).reshape(-1).copy()
            n_p = problem.n
            T_p = problem.T
            if multipliers_eq_full.size >= n_p + T_p * n_p:
                multipliers_eq_full[n_p:n_p + T_p * n_p] = (
                    -multipliers_eq_full[n_p:n_p + T_p * n_p]
                )
        except Exception:  # noqa: BLE001
            multipliers_eq_full = None
        try:
            multipliers_ineq_full = np.asarray(
                vars_in.z, dtype=np.float64,
            ).reshape(-1).copy() if has_ineq else None
        except Exception:  # noqa: BLE001
            multipliers_ineq_full = None

        if output is None:
            return pack_solver_result(
                solver_name=self.name,
                problem_name=problem.name,
                problem=problem,
                X=X, U=U, Theta=Theta,
                iterations=0,
                solve_time_ms=solve_time_ms,
                success=False,
                notes=err,
                multipliers_eq=multipliers_eq_full,
                multipliers_ineq=multipliers_ineq_full,
                iterates_xut=iter_xut or None,
            )

        status = output.exit_status
        success = status == sip.Status.SOLVED
        return pack_solver_result(
            solver_name=self.name,
            problem_name=problem.name,
            problem=problem,
            X=X, U=U, Theta=Theta,
            iterations=int(output.num_iterations),
            solve_time_ms=solve_time_ms,
            success=success,
            notes=str(status),
            multipliers_eq=multipliers_eq_full,
            multipliers_ineq=multipliers_ineq_full,
            iterates_xut=iter_xut or None,
        )


@register("sip-mjx")
def _factory(**kwargs) -> SolverAdapter:
    return SipMjxAdapter(**kwargs)
