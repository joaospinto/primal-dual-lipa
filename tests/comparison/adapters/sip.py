"""sip_python adapter for the analytical comparison suite.

Builds the multi-shooting NLP from a ``ProblemSpec`` and solves it via
``sip_python.Solver``. Decision vector::

    z = (vec(X), vec(U), Theta)

``theta_dim`` may be zero or positive; unlike the OCP-structured
solvers, sip_python is a generic NLP solver and doesn't care whether a
component of ``z`` is stage-local or cross-stage.

Two model-evaluation backends:

* ``backend="jax"`` (default): cost / equality / inequality come from
  the ``ProblemSpec`` JAX functions; gradients, Jacobians and the
  Hessian are built via ``jax.grad`` / ``jax.jacrev`` / ``jax.hessian``
  and JIT-compiled once per solve. Sparsity is detected from a small
  number of dense probe evaluations at perturbed warm starts.
* ``backend="casadi"``: the model is built from the problem's
  ``metadata["casadi_builder"]``. Sparsity comes from CasADi's
  symbolic graph. Wall-clock is typically several times faster than
  the JAX backend on the analytical suite.

The CasADi backend supports ``hessian_mode="cost"`` only.

Hessian modes:

* ``"cost"`` (default): eigen-clamp ``hessian(f)``, take ``triu``.
  Drops second-order constraint terms (Gauss-Newton-flavoured).
* ``"lagrangian"``: include ``y^T hess(c) + z^T hess(g)``, symmetrize,
  PSD-clamp, ``triu``.

Conventions (from sip_python's tests/test_simple_constrained_lqr.py):

* Jacobians as ``scipy.sparse.csr_matrix`` of shape ``(c_dim, x_dim)``;
  reinterpreting CSR data as CSC yields the transpose, hence
  ``ProblemDimensions.is_jacobian_c_transposed = True``.
* Hessian as ``scipy.sparse.csc_matrix`` of shape ``(x_dim, x_dim)``
  containing the upper triangle of a PSD approximation.

The sparsity pattern is determined once per solve from dense JAX
evaluations at perturbed warm starts. At each callback we re-evaluate
the dense Jacobian/Hessian and overwrite ``.data`` of the pre-built
sparse matrices.
"""

from __future__ import annotations

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


def _import_sip():
    import sip_python  # noqa: F401

    return sip_python


def _import_jax():
    import jax  # noqa: F401
    import jax.numpy as jnp  # noqa: F401

    # SIP's QDLDL backend does the linear solves in float64; if JAX runs in
    # float32 we'd silently degrade convergence. Match the test files.
    jax.config.update("jax_enable_x64", True)
    return jax, jnp


def _filter_zero_rows(c_fn, c_dim: int, x_dim: int, mock_zs: list):
    """Detect rows of c that are identically zero across all probe inputs.

    Used to drop trivially-zero equality rows (some problems pad
    inactive stages with zeros). Such rows generate spurious dual
    variables and inflate the KKT system; filtering them out leaves
    only the constraints with actual content. Returns
    ``(keep_mask, c_filtered)`` where ``c_filtered(z) = c(z)[keep_mask]``.

    Probe inputs should be a list of dense ``z`` vectors; rows are
    kept if any probe produces a non-zero entry.
    """
    import jax  # noqa: F401
    import jax.numpy as jnp

    if c_dim == 0:
        return np.ones(0, dtype=bool), c_fn

    keep = np.zeros(c_dim, dtype=bool)
    c_jit = jax.jit(c_fn)
    for z in mock_zs:
        cz = np.asarray(c_jit(jnp.asarray(z)), dtype=np.float64)
        keep = keep | (np.abs(cz) > 0.0)
    # Also probe the Jacobian at the warm start — a row could happen
    # to evaluate to 0 at a probe but have a non-zero derivative
    # elsewhere (e.g. nonlinear constraint that only "lights up" at
    # certain x). Take the union with rows that have non-zero
    # gradient at any probe.
    jac_jit = jax.jit(jax.jacrev(c_fn))
    for z in mock_zs:
        Jz = np.asarray(jac_jit(jnp.asarray(z)), dtype=np.float64)
        keep = keep | (np.abs(Jz).sum(axis=1) > 0.0)

    keep_indices = np.where(keep)[0]

    def c_filtered(z):
        return c_fn(z)[jnp.asarray(keep_indices)]

    return keep, c_filtered


def _slice_iterate(
    z_val: np.ndarray, problem: ProblemSpec
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n, m, T, td = problem.n, problem.m, problem.T, problem.theta_dim
    nx = n * (T + 1)
    nu = m * T
    X = z_val[:nx].reshape(T + 1, n)
    U = z_val[nx : nx + nu].reshape(T, m)
    Theta = z_val[nx + nu : nx + nu + td] if td > 0 else np.zeros(0, dtype=z_val.dtype)
    return X, U, Theta


def _import_casadi():
    import casadi as ca  # local import so a missing CasADi only fails this backend

    return ca


def _build_casadi_nlp(problem: ProblemSpec):
    """Build the flat NLP via the problem's ``casadi_builder``.

    Mirrors ``_build_jax_nlp`` but everything happens symbolically in
    CasADi SX. Returns a dict bundling:

    * ``f_fn(z)``       — scalar cost ``casadi.Function``
    * ``c_fn(z)``       — equality residual ``casadi.Function``
                          (init defect + dynamics defects + non-None
                          eq rows). Already filtered: stages where
                          ``casadi_builder`` returns ``eq=None`` are
                          omitted entirely (analogous to the JAX
                          backend's ``_filter_zero_rows`` step).
    * ``g_fn(z)``       — inequality residual ``casadi.Function``,
                          convention ``g <= 0`` (only stages where
                          ``casadi_builder`` returns a non-None
                          ``ineq``).
    * ``grad_f_fn(z)``  — gradient of f, dense (n_z,) output.
    * ``jac_c_fn(z)``   — Jacobian of c, CasADi-sparse.
    * ``jac_g_fn(z)``   — Jacobian of g, CasADi-sparse.
    * ``stage_hess_meta`` — list of per-stage cost-Hessian descriptors.
                          Each entry is a dict ``{"fn", "z_idx",
                          "block_size", "sparsity"}`` where ``fn`` is a
                          ``casadi.Function`` taking the block's
                          flattened input ``[x_t; (u_t if t < T);
                          theta]`` and returning the dense (small)
                          per-stage Hessian wrt that vector. ``z_idx``
                          gives the indices in the global ``z`` for
                          writing the PSD-projected block back to the
                          upper-triangle sparse buffer; ``sparsity`` is
                          the CasADi symbolic sparsity of the block
                          (used to size the global Hessian template).
    * ``y_dim``, ``s_dim``, ``x_dim`` — dimensions.
    * ``jac_c_sparsity`` / ``jac_g_sparsity`` — symbolic CasADi
                          sparsity patterns (used to build the
                          ``scipy.sparse`` template buffers).
    * ``upp_hess_pat`` — (x_dim, x_dim) sparse boolean upper-triangle
                          pattern: union of per-stage block symbolic
                          nonzeros ∪ the diagonal of every block's
                          z-index scope. Used to size the sparse
                          Hessian template handed to sip_python.

    The decision vector layout matches the JAX path exactly:
    ``z = [vec(X) (n*(T+1)), vec(U) (m*T), Theta (td)]``.

    Per-stage cost Hessian extraction is the key optimisation that
    lets the CasADi backend scale to large x_dim: the global Hessian
    is block-diagonal (plus an arrow from the cross-stage ``theta``
    if any), so the eigen-clamp factorises across stage blocks of
    size at most (n+m+theta_dim) x (n+m+theta_dim) instead of doing
    a single O(x_dim^3) eigh.
    """
    from scipy import sparse as sp

    ca = _import_casadi()
    casadi_builder = problem.metadata["casadi_builder"]

    T, n, m, td = problem.T, problem.n, problem.m, problem.theta_dim
    nx = n * (T + 1)
    nu = m * T
    x_dim = nx + nu + td

    X_sx = ca.SX.sym("X", n, T + 1)
    U_sx = ca.SX.sym("U", m, T)
    Theta_sx = ca.SX.sym(
        "Theta", max(td, 1)
    )  # CasADi can't build a 0-length sym; slice it off below

    z_pieces = [ca.reshape(X_sx, -1, 1), ca.reshape(U_sx, -1, 1)]
    if td > 0:
        z_pieces.append(Theta_sx[:td])
    z = ca.vertcat(*z_pieces)

    theta_arg = Theta_sx[:td] if td > 0 else ca.SX.zeros(0)

    # Substitute the Theta_sx symbol with a 0-length placeholder when
    # td == 0 so the resulting Functions only take z as input.
    # Otherwise theta_arg is just a slice of Theta_sx and is part of z.
    f_total = ca.SX(0.0)
    c_pieces = []  # init defect + dyn defects + non-None eq rows
    g_pieces = []  # non-None ineq rows
    # Per-stage Hessian metadata: list of (stage_hess_fn, idx_array)
    # where idx_array is the array of z-indices for the stage's inputs
    # (X[:, t] then U[:, t] then optional Theta). At terminal stages
    # U is omitted (no u arg).
    stage_hess_meta = []

    # Initial-state defect (always present).
    c_pieces.append(X_sx[:, 0] - ca.DM(np.asarray(problem.x0)))

    # Helper to compute z-indices for a stage's inputs.
    def _stage_indices(t: int, include_u: bool) -> np.ndarray:
        # X[:, t] occupies z[t*n : (t+1)*n]
        idx = list(range(t * n, (t + 1) * n))
        if include_u and t < T:
            # U[:, t] occupies z[nx + t*m : nx + (t+1)*m]
            idx.extend(range(nx + t * m, nx + (t + 1) * m))
        if td > 0:
            idx.extend(range(nx + nu, nx + nu + td))
        return np.asarray(idx, dtype=np.intp)

    for t in range(T + 1):
        # Per-stage cost: re-build the stage in isolated SX symbols so we
        # can take its Hessian w.r.t. just (x_t, u_t, theta) without the
        # global graph. This is the "split symbolic" trick that keeps the
        # per-stage Hessian Function small.
        include_u = t < T
        x_local = ca.SX.sym(f"xs_{t}", n)
        u_local = ca.SX.sym(f"us_{t}", m) if include_u else ca.SX.zeros(m, 1)
        theta_local = ca.SX.sym(f"th_{t}", max(td, 1))
        theta_local_arg = theta_local[:td] if td > 0 else ca.SX.zeros(0)
        stage_local = casadi_builder(x_local, u_local, theta_local_arg, t)
        f_local = stage_local["f"]
        # Hessian variables for this stage block. Order: x_t, [u_t,] theta.
        # Matches _stage_indices() ordering above.
        hess_vars = [x_local]
        if include_u:
            hess_vars.append(u_local)
        if td > 0:
            hess_vars.append(theta_local_arg)
        xu_local = ca.vertcat(*hess_vars)
        H_local, _ = ca.hessian(f_local, xu_local)
        # Build the per-stage Hessian Function. Input is the same
        # variables; output is the (block_size, block_size) dense matrix.
        # We want the Function to take a single flat input matching the
        # global z slice — this lets per-call dispatch be a single
        # ``np.ndarray`` slice + Function call rather than a build-up.
        fn_name = f"sip_stage_hess_{t}"
        # Use the same xu_local as the Function input.
        stage_hess_fn = ca.Function(fn_name, [xu_local], [H_local])
        stage_hess_meta.append(
            {
                "fn": stage_hess_fn,
                "z_idx": _stage_indices(t, include_u),
                "block_size": xu_local.numel(),
                "sparsity": H_local.sparsity(),
            }
        )
        # ---- Global-graph pieces (cost, c, g, jacobians) ---------------------
        u_t = U_sx[:, t] if t < T else ca.SX.zeros(m, 1)
        stage = casadi_builder(X_sx[:, t], u_t, theta_arg, t)
        f_total = f_total + stage["f"]
        if t < T:
            # Dynamics defect: x_{t+1} - f(x_t, u_t, theta, t) = 0
            c_pieces.append(X_sx[:, t + 1] - stage["next_x"])
        eq_t = stage.get("eq")
        if eq_t is not None and (not hasattr(eq_t, "numel") or eq_t.numel() > 0):
            c_pieces.append(ca.reshape(eq_t, eq_t.numel(), 1))
        ineq_t = stage.get("ineq")
        if ineq_t is not None and (not hasattr(ineq_t, "numel") or ineq_t.numel() > 0):
            g_pieces.append(ca.reshape(ineq_t, ineq_t.numel(), 1))

    c_expr = ca.vertcat(*c_pieces) if c_pieces else ca.SX.zeros(0, 1)
    g_expr = ca.vertcat(*g_pieces) if g_pieces else ca.SX.zeros(0, 1)
    y_dim = c_expr.numel()
    s_dim = g_expr.numel()

    # Autodiff for the global-graph pieces (cost, jacobians).
    grad_f = ca.gradient(f_total, z)
    jac_c = ca.jacobian(c_expr, z) if y_dim > 0 else ca.SX.zeros(0, x_dim)
    jac_g = ca.jacobian(g_expr, z) if s_dim > 0 else ca.SX.zeros(0, x_dim)

    f_fn = ca.Function("sip_cost_val", [z], [f_total], ["z"], ["f"])
    c_fn = ca.Function("sip_eq_val", [z], [c_expr], ["z"], ["c"])
    g_fn = ca.Function("sip_ineq_val", [z], [g_expr], ["z"], ["g"])
    grad_f_fn = ca.Function("sip_cost_grad", [z], [grad_f], ["z"], ["grad_f"])
    jac_c_fn = ca.Function("sip_eq_jac", [z], [jac_c], ["z"], ["jac_c"])
    jac_g_fn = ca.Function("sip_ineq_jac", [z], [jac_g], ["z"], ["jac_g"])

    # Build the (x_dim, x_dim) upper-triangle sparsity pattern as the
    # union of (a) each per-stage block's symbolic sparsity, mapped from
    # block-local (i, j) to global (z_idx[i], z_idx[j]), and (b) the
    # diagonal (the eigen-clamp may add k*I to every block, which lights
    # up every diagonal entry within a block's z_idx scope — strictly
    # speaking we only need diagonal entries that some stage covers, but
    # adding the full diagonal is cheap and avoids edge cases).
    #
    # Using the symbolic per-stage nnz (vs the full block) keeps the
    # global Hessian template close to truly minimal.
    upp_hess_pat = sp.lil_matrix((x_dim, x_dim), dtype=bool)
    for meta in stage_hess_meta:
        idx = meta["z_idx"]
        sp_block = meta["sparsity"]
        # CasADi sparsity: iterate (row, col) of structurally non-zero
        # entries. Use sparsity.row() / sparsity.colind() (CSC).
        col_ind = np.asarray(sp_block.colind(), dtype=np.intp)
        row_ind = np.asarray(sp_block.row(), dtype=np.intp)
        for j in range(sp_block.size2()):  # local cols
            for k in range(col_ind[j], col_ind[j + 1]):
                i = row_ind[k]  # local row
                gi = int(idx[i])
                gj = int(idx[j])
                if gi <= gj:  # upper triangle
                    upp_hess_pat[gi, gj] = True
                else:
                    upp_hess_pat[gj, gi] = True  # mirror to upper triangle
        # Also mark the diagonal entries the block can populate via
        # k*I clamp (every variable in z_idx may get a k*I bump).
        for i in idx:
            upp_hess_pat[i, i] = True
    upp_hess_pat = upp_hess_pat.tocsc()
    upp_hess_pat = upp_hess_pat.astype(np.float64) != 0

    return {
        "f_fn": f_fn,
        "c_fn": c_fn,
        "g_fn": g_fn,
        "grad_f_fn": grad_f_fn,
        "jac_c_fn": jac_c_fn,
        "jac_g_fn": jac_g_fn,
        "stage_hess_meta": stage_hess_meta,
        "upp_hess_pat": upp_hess_pat,
        "x_dim": x_dim,
        "y_dim": y_dim,
        "s_dim": s_dim,
        "jac_c_sparsity": jac_c.sparsity() if y_dim > 0 else None,
        "jac_g_sparsity": jac_g.sparsity() if s_dim > 0 else None,
    }


class SipAdapter(SolverAdapter):
    """sip_python solver adapter."""

    name = "sip"

    def __init__(
        self,
        max_iter: int = 1000,
        tol: float = 1e-6,
        # PSD-projection floor for the cost Hessian. Matches sip_python's
        # test default. Larger values trade convergence speed for
        # robustness on hard nonconvex sub-problems.
        psd_reg_delta: float = 1e-6,
        # Elastic mode lets the solver soften inequalities by absorbing
        # infeasibility into an L1-penalised slack. Off by default: the
        # dynamics defects in our OCPs are stiff equalities that elastic
        # relaxation would silently corrupt.
        enable_elastics: bool = False,
        elastic_var_cost_coeff: float = 1e6,
        # Default penalty / barrier ramp. Per-problem overrides via
        # problem.metadata["sip_settings"] dial in different schedules
        # for problems whose defaults don't converge.
        penalty_parameter_increase_factor: float = 2.0,
        mu_update_factor: float = 0.9,
        initial_mu: float = 1e-1,
        initial_penalty_parameter: float = 1.0,
        # Hessian mode: "lagrangian" (full y^T hess(c) + z^T hess(g) +
        # hess(f), PSD-projected) or "cost" (just hess(f) PSD-projected,
        # matching the sip_python test convention).
        hessian_mode: str = "cost",
        # PSD-projection scheme for per-stage Hessian blocks (JAX
        # backend only; CasADi backend always uses "kappa_shift"):
        # * "schur" — LIPA's Schur-complement scheme via
        #   ``block_schur_psd_projection``; guarantees global PD when
        #   theta is present at the cost of more aggressive eigenvalue
        #   re-shaping.
        # * "eig_clip" — V @ max(S, delta) @ V.T eigenvalue clip via
        #   ``project_psd_cone``. Flattens negative eigenvalues to
        #   delta; the legacy JAX-backend default for td=0.
        # * "kappa_shift" — Q + max(-s_min, 0) * I + delta * I; the
        #   CasADi backend's scheme. Preserves the eigenvector basis
        #   and only shifts the spectrum, which converges faster on
        #   problems where the Hessian basis is informative even when
        #   nearly singular.
        hessian_psd_mode: str = "schur",
        print_logs: bool = False,
        sip_extra_settings: Optional[dict] = None,
        # Backend for per-iter model evaluations. "casadi" reuses the
        # problem's casadi_builder (same one IPOPT / acados / fatrop /
        # CSQP use) and is several times faster wall-clock on
        # analytical problems while producing the same iter counts.
        # MJX problems have no casadi_builder, so requesting
        # "unavailable" SolverResult.
        backend: str = "jax",
    ) -> None:
        self.max_iter = max_iter
        self.tol = tol
        self.psd_reg_delta = float(psd_reg_delta)
        self.enable_elastics = bool(enable_elastics)
        self.elastic_var_cost_coeff = float(elastic_var_cost_coeff)
        self.penalty_parameter_increase_factor = float(
            penalty_parameter_increase_factor
        )
        self.mu_update_factor = float(mu_update_factor)
        self.initial_mu = float(initial_mu)
        self.initial_penalty_parameter = float(initial_penalty_parameter)
        self.hessian_mode = str(hessian_mode)
        if self.hessian_mode not in ("lagrangian", "cost"):
            raise ValueError(
                f"hessian_mode must be 'lagrangian' or 'cost', got {hessian_mode!r}",
            )
        self.hessian_psd_mode = str(hessian_psd_mode)
        if self.hessian_psd_mode not in ("schur", "eig_clip", "kappa_shift"):
            raise ValueError(
                "hessian_psd_mode must be one of 'schur', 'eig_clip', "
                f"'kappa_shift'; got {hessian_psd_mode!r}",
            )
        self.print_logs = bool(print_logs)
        self.sip_extra_settings = sip_extra_settings or {}
        if backend not in {"jax", "casadi"}:
            raise ValueError(f"backend must be 'jax' or 'casadi', got {backend!r}")
        self.backend = backend
        if backend == "casadi" and self.hessian_mode != "cost":
            # The CasADi backend currently only implements the cost-only
            # Hessian PSD-projection path.
            raise ValueError(
                "backend='casadi' currently only supports hessian_mode='cost'; "
                f"got hessian_mode={self.hessian_mode!r}",
            )

    def is_available(self) -> tuple[bool, str]:
        try:
            _import_sip()
        except ImportError as e:
            return False, f"{e}"
        if self.backend == "casadi":
            try:
                _import_casadi()
            except ImportError as e:
                return False, f"casadi backend requested but casadi import failed: {e}"
        return True, ""

    def solve(self, problem: ProblemSpec) -> SolverResult:  # noqa: PLR0915
        from tests.comparison.problem_spec import effective_solver_tol

        # Stored on self so the inner ``_solve_jax_per_stage`` /
        # ``_solve_casadi`` dispatch targets can read it.
        self._effective_tol = effective_solver_tol(problem, self.tol)
        avail, reason = self.is_available()
        if not avail:
            return make_failure_result(
                self.name,
                problem.name,
                f"unavailable: {reason}",
            )

        # NOTE: sip_python is a generic NLP solver — unlike CSQP /
        # Aligator / acados / fatrop it has no native OCP layout and
        # doesn't care whether a component of ``z`` is stage-local or
        # cross-stage, so theta_dim > 0 is handled by extending ``z``
        # with the Theta block (see ``_build_jax_nlp`` /
        # ``_build_casadi_nlp``).

        # Backend dispatch. The CasADi backend reuses all of the SIP
        # solver wiring (Settings / QDLDLSettings / ProblemDimensions /
        # solver.solve / Variables) but builds the model callback from
        # compiled CasADi Functions instead of jit'd JAX functions; see
        # ``_solve_casadi`` for the alternate model-construction path.
        if self.backend == "casadi":
            if "casadi_builder" not in problem.metadata:
                return make_failure_result(
                    self.name,
                    problem.name,
                    "backend='casadi' requires casadi_builder in problem.metadata "
                    f"(problem={problem.name!r} doesn't ship one — re-run with backend='jax')",
                )
            return self._solve_casadi(problem)

        return self._solve_jax_per_stage(problem)

    def _solve_jax_per_stage(self, problem: ProblemSpec) -> SolverResult:  # noqa: PLR0915, PLR0912
        """JAX backend with per-stage Jacobian / Hessian assembly.

        Replaces the original ``_solve`` body that did global
        ``jax.jacrev(c)`` / ``jax.jacrev(g)`` and produced dense per-iter
        Jacobians (e.g. 162 MB on quadpendulum_theta). The per-stage
        pattern mirrors what ``sip_mjx`` does — vmap of stage-local
        ``jax.jacrev`` returning compact ``(T, out_dim, n+m+td)`` blocks
        — but adds theta-coupling support and a user-equality vmap path
        so it works on every problem class (analytical + MJX, with or
        without theta and user equalities).

        Hessian PSD-projection per stage uses LIPA's Schur-complement
        scheme (``primal_dual_lipa.kkt_builder.block_schur_psd_projection``)
        when ``theta_dim > 0``, otherwise plain per-block eigh. The
        Schur scheme guarantees the assembled global Hessian is
        positive definite even after theta-coupling rows are summed
        across stages; per-block eigh + scatter-add would only give PSD.
        """
        sip = _import_sip()
        jax, jnp = _import_jax()
        from scipy import sparse as sp
        from primal_dual_lipa.kkt_builder import block_schur_psd_projection
        from regularized_lqr_jax.helpers import project_psd_cone

        T, n, m = problem.T, problem.n, problem.m
        td = problem.theta_dim
        nx = n * (T + 1)
        nu = m * T
        x_dim = nx + nu + td

        cost_fn = problem.cost
        dyn_fn = problem.dynamics
        eq_fn = problem.equalities
        ineq_fn = problem.inequalities
        has_user_eq = eq_fn is not None and problem.eq_dim > 0
        has_ineq = ineq_fn is not None and problem.ineq_dim > 0
        eq_dim_user = problem.eq_dim if has_user_eq else 0
        ineq_dim = problem.ineq_dim if has_ineq else 0

        # FULL equality stack: init(n) + dyn(T*n) + user_eq((T+1)*eq_dim_user).
        # We filter zero rows (problems like cartpole/quadpendulum pad the
        # inner user-eq stages with jnp.where-zeros so eq() returns
        # structurally-zero rows for t < T) — the filtered y_dim / s_dim
        # are computed below after the row-keep mask is built.
        c_full_dim = n + T * n + (T + 1) * eq_dim_user
        g_full_dim = (T + 1) * ineq_dim

        x0_const = jnp.asarray(problem.x0)
        psd_delta = self.psd_reg_delta
        is_cost_mode = self.hessian_mode == "cost"
        # MJX problems need the Lagrangian Hessian (multiplier
        # contributions matter for stiff contact constraints);
        # cost-only Hessian gives a non-stationary local minimum.
        # Mirror the historical sip_mjx adapter's always-Lagrangian
        # behaviour.
        if problem.metadata.get("is_mjx") and is_cost_mode:
            is_cost_mode = False

        def _slice_xut(z):
            X = z[:nx].reshape(T + 1, n)
            U = z[nx : nx + nu].reshape(T, m)
            theta = z[nx + nu : nx + nu + td] if td > 0 else jnp.zeros(0, dtype=z.dtype)
            return X, U, theta

        # ===== Per-stage JAX primitives =====
        @jax.jit
        def stage_cost(x, u, theta, t):
            return cost_fn(x, u, theta, t)

        @jax.jit
        def stage_dyn(x, u, theta, t):
            return dyn_fn(x, u, theta, t)

        @jax.jit
        def terminal_cost(x_T, theta):
            u_zero = jnp.zeros(m, dtype=x_T.dtype)
            return cost_fn(x_T, u_zero, theta, jnp.int32(T))

        if has_user_eq:

            @jax.jit
            def stage_user_eq(x, u, theta, t):
                return eq_fn(x, u, theta, t)

            @jax.jit
            def terminal_user_eq(x_T, theta):
                u_zero = jnp.zeros(m, dtype=x_T.dtype)
                return eq_fn(x_T, u_zero, theta, jnp.int32(T))

        if has_ineq:

            @jax.jit
            def stage_ineq(x, u, theta, t):
                return ineq_fn(x, u, theta, t)

            @jax.jit
            def terminal_ineq(x_T, theta):
                u_zero = jnp.zeros(m, dtype=x_T.dtype)
                return ineq_fn(x_T, u_zero, theta, jnp.int32(T))

        # ===== Flat-NLP scalar / vector functions =====
        @jax.jit
        def f_fn(z):
            X, U, theta = _slice_xut(z)
            ts = jnp.arange(T)
            inner = jax.vmap(stage_cost, in_axes=(0, 0, None, 0))(
                X[:T],
                U,
                theta,
                ts,
            )
            return jnp.sum(inner) + terminal_cost(X[T], theta)

        @jax.jit
        def grad_f_fn(z):
            return jax.grad(f_fn)(z)

        @jax.jit
        def c_fn(z):
            X, U, theta = _slice_xut(z)
            ts = jnp.arange(T)
            init_defect = X[0] - x0_const
            dyn_next = jax.vmap(stage_dyn, in_axes=(0, 0, None, 0))(
                X[:T],
                U,
                theta,
                ts,
            )
            # SIGN convention: c() returns X[t+1] - dyn(...) (opposite from
            # evaluate_problem's dyn(...) - X[t+1]); sign-flipped in the
            # multiplier extraction post-solve.
            dyn_defects = (X[1:] - dyn_next).reshape(-1)
            pieces = [init_defect, dyn_defects]
            if has_user_eq:
                inner_eq = jax.vmap(stage_user_eq, in_axes=(0, 0, None, 0))(
                    X[:T],
                    U,
                    theta,
                    ts,
                )
                term_eq = terminal_user_eq(X[T], theta)
                pieces.append(
                    jnp.concatenate(
                        [inner_eq.reshape(-1), term_eq.reshape(-1)],
                    )
                )
            return jnp.concatenate(pieces)

        if has_ineq:

            @jax.jit
            def g_fn(z):
                X, U, theta = _slice_xut(z)
                ts = jnp.arange(T)
                inner_ineq = jax.vmap(stage_ineq, in_axes=(0, 0, None, 0))(
                    X[:T],
                    U,
                    theta,
                    ts,
                )
                term_ineq = terminal_ineq(X[T], theta)
                return jnp.concatenate(
                    [inner_ineq.reshape(-1), term_ineq.reshape(-1)],
                )

        # ===== Row-keep mask: filter out structurally-zero rows =====
        # Problems whose equality function pads inner stages with
        # ``jnp.where(t == T, ..., zeros)`` produce structurally-zero c
        # rows that, if handed to sip, make its KKT system singular (the
        # corresponding dual variables are unbounded). Same for ineq with
        # constant trivially-satisfied rows whose Jacobian is zero.
        # Probe at warm-start + perturbations to determine which rows
        # are structurally active. y_dim / s_dim shrink accordingly.
        z_init_np = np.concatenate(
            [
                np.asarray(problem.X_init, dtype=np.float64).reshape(-1),
                np.asarray(problem.U_init, dtype=np.float64).reshape(-1),
                np.asarray(problem.Theta_init, dtype=np.float64).reshape(-1),
            ]
        )
        _rng = np.random.default_rng(0)
        probe_zs_np = [
            z_init_np,
            z_init_np + 0.01 * _rng.standard_normal(z_init_np.shape),
            z_init_np + 0.1 * _rng.standard_normal(z_init_np.shape),
        ]
        if td > 0:
            # Force a non-zero theta probe so theta-dependent rows aren't
            # missed when Theta_init defaults to zero.
            z_theta_probe = z_init_np.copy()
            z_theta_probe[nx + nu : nx + nu + td] = 0.1
            probe_zs_np.append(z_theta_probe)
        probe_zs = [jnp.asarray(z, dtype=jnp.float64) for z in probe_zs_np]
        keep_mask_eq, c_fn_filtered = _filter_zero_rows(
            c_fn,
            c_full_dim,
            x_dim,
            probe_zs,
        )
        if has_ineq:
            keep_mask_ineq, g_fn_filtered = _filter_zero_rows(
                g_fn,
                g_full_dim,
                x_dim,
                probe_zs,
            )
        else:
            keep_mask_ineq = np.zeros(0, dtype=bool)
            g_fn_filtered = None
        y_dim = int(keep_mask_eq.sum())
        s_dim = int(keep_mask_ineq.sum())

        # Build full→filtered row index map (used for sparsity template +
        # multiplier scatter). Rows that were filtered out get -1 so any
        # bug that uses them is easy to spot.
        full_to_filtered_eq = np.full(c_full_dim, -1, dtype=np.int64)
        full_to_filtered_eq[keep_mask_eq] = np.arange(y_dim)
        full_to_filtered_ineq = np.full(g_full_dim, -1, dtype=np.int64)
        if has_ineq:
            full_to_filtered_ineq[keep_mask_ineq] = np.arange(s_dim)

        # ===== Per-stage Jacobian builders =====
        # All produce per-stage blocks of shape (..., out_dim, n+m+td) for
        # inner stages and (out_dim, n+td) for terminal stage.

        def _stage_input(x, u, theta):
            # Concatenate (x, u, theta) for the per-stage Jacobian/Hessian
            # input. theta may be empty; concatenation handles that.
            return jnp.concatenate([x, u, theta])

        def _terminal_input(x_T, theta):
            return jnp.concatenate([x_T, theta])

        @jax.jit
        def dyn_jac_blocks(z):
            X, U, theta = _slice_xut(z)
            ts = jnp.arange(T)

            def jac_one(x_in, u_in, t_in):
                def _f(xut):
                    xx = xut[:n]
                    uu = xut[n : n + m]
                    thth = (
                        xut[n + m : n + m + td]
                        if td > 0
                        else jnp.zeros(0, dtype=xut.dtype)
                    )
                    return stage_dyn(xx, uu, thth, t_in)

                return jax.jacrev(_f)(_stage_input(x_in, u_in, theta))

            return jax.vmap(jac_one, in_axes=(0, 0, 0))(X[:T], U, ts)

        if has_user_eq:

            @jax.jit
            def user_eq_jac_inner_blocks(z):
                X, U, theta = _slice_xut(z)
                ts = jnp.arange(T)

                def jac_one(x_in, u_in, t_in):
                    def _f(xut):
                        xx = xut[:n]
                        uu = xut[n : n + m]
                        thth = (
                            xut[n + m : n + m + td]
                            if td > 0
                            else jnp.zeros(0, dtype=xut.dtype)
                        )
                        return stage_user_eq(xx, uu, thth, t_in)

                    return jax.jacrev(_f)(_stage_input(x_in, u_in, theta))

                return jax.vmap(jac_one, in_axes=(0, 0, 0))(X[:T], U, ts)

            @jax.jit
            def user_eq_jac_terminal(z):
                X, _U, theta = _slice_xut(z)

                def _f(xt):
                    xx = xt[:n]
                    thth = xt[n : n + td] if td > 0 else jnp.zeros(0, dtype=xt.dtype)
                    return terminal_user_eq(xx, thth)

                return jax.jacrev(_f)(_terminal_input(X[T], theta))

        if has_ineq:

            @jax.jit
            def ineq_jac_inner_blocks(z):
                X, U, theta = _slice_xut(z)
                ts = jnp.arange(T)

                def jac_one(x_in, u_in, t_in):
                    def _f(xut):
                        xx = xut[:n]
                        uu = xut[n : n + m]
                        thth = (
                            xut[n + m : n + m + td]
                            if td > 0
                            else jnp.zeros(0, dtype=xut.dtype)
                        )
                        return stage_ineq(xx, uu, thth, t_in)

                    return jax.jacrev(_f)(_stage_input(x_in, u_in, theta))

                return jax.vmap(jac_one, in_axes=(0, 0, 0))(X[:T], U, ts)

            @jax.jit
            def ineq_jac_terminal(z):
                X, _U, theta = _slice_xut(z)

                def _f(xt):
                    xx = xt[:n]
                    thth = xt[n : n + td] if td > 0 else jnp.zeros(0, dtype=xt.dtype)
                    return terminal_ineq(xx, thth)

                return jax.jacrev(_f)(_terminal_input(X[T], theta))

        # ===== Per-stage Hessian builders =====
        # PSD-project per stage. The scheme is controlled by the
        # ``sip_hessian_psd_mode`` problem-metadata key (falls back to
        # ``self.hessian_psd_mode``); see the SipAdapter docstring for
        # the per-mode tradeoffs:
        # * "schur" — LIPA's Schur-complement scheme (when theta is
        #   present) falling back to eigenvalue-clip via
        #   ``project_psd_cone`` for theta-free stages.
        # * "eig_clip" — eigenvalue-clip everywhere (V @ max(S, delta)
        #   @ V.T).
        # * "kappa_shift" — Q + max(-s_min, 0) * I + delta * I.
        #   Preserves eigenvector basis; matches the CasADi backend.
        psd_mode = problem.metadata.get("sip_hessian_psd_mode", self.hessian_psd_mode)
        if psd_mode not in ("schur", "eig_clip", "kappa_shift"):
            raise ValueError(
                "sip_hessian_psd_mode must be 'schur' / 'eig_clip' / "
                f"'kappa_shift'; got {psd_mode!r}"
            )

        def _kappa_shift(H):
            S = jnp.linalg.eigvalsh(0.5 * (H + H.T))
            shift = jnp.maximum(-S[0], 0.0) + psd_delta
            return H + shift * jnp.eye(H.shape[0], dtype=H.dtype)

        if psd_mode == "schur":

            def _psd_project_inner(H):
                if td > 0:
                    return block_schur_psd_projection(H, dims=(td, m, n), eps=psd_delta)
                return project_psd_cone(H, delta=psd_delta)

            def _psd_project_terminal(H):
                if td > 0:
                    return block_schur_psd_projection(H, dims=(td, n), eps=psd_delta)
                return project_psd_cone(H, delta=psd_delta)

        elif psd_mode == "eig_clip":

            def _psd_project_inner(H):
                return project_psd_cone(H, delta=psd_delta)

            def _psd_project_terminal(H):
                return project_psd_cone(H, delta=psd_delta)

        else:  # kappa_shift

            def _psd_project_inner(H):
                return _kappa_shift(H)

            def _psd_project_terminal(H):
                return _kappa_shift(H)

        @jax.jit
        def inner_hess_blocks(z, y, zd):
            X, U, theta = _slice_xut(z)
            ts = jnp.arange(T)
            # Multiplier slices (only consulted in lagrangian mode).
            Y_dyn = y[n : n + T * n].reshape(T, n)
            if has_user_eq:
                Y_eq_inner = y[n + T * n : n + T * n + T * eq_dim_user].reshape(
                    T, eq_dim_user
                )
            else:
                Y_eq_inner = jnp.zeros((T, 0), dtype=z.dtype)
            if has_ineq:
                Z_inner = zd[: T * ineq_dim].reshape(T, ineq_dim)
            else:
                Z_inner = jnp.zeros((T, 0), dtype=z.dtype)

            def stage_hess(x, u, t, y_dyn_t, y_eq_t, z_ineq_t):
                def L_xut(xut):
                    xx = xut[:n]
                    uu = xut[n : n + m]
                    thth = (
                        xut[n + m : n + m + td]
                        if td > 0
                        else jnp.zeros(0, dtype=xut.dtype)
                    )
                    if is_cost_mode:
                        return stage_cost(xx, uu, thth, t)
                    # Lagrangian = cost - y_dyn·dyn (sign matches c's
                    # X[t+1] - dyn convention) + y_eq·eq + z_ineq·ineq.
                    terms = stage_cost(xx, uu, thth, t) - jnp.dot(
                        y_dyn_t,
                        stage_dyn(xx, uu, thth, t),
                    )
                    if has_user_eq:
                        terms = terms + jnp.dot(
                            y_eq_t,
                            stage_user_eq(xx, uu, thth, t),
                        )
                    if has_ineq:
                        terms = terms + jnp.dot(
                            z_ineq_t,
                            stage_ineq(xx, uu, thth, t),
                        )
                    return terms

                return jax.hessian(L_xut)(_stage_input(x, u, theta))

            blocks = jax.vmap(stage_hess, in_axes=(0, 0, 0, 0, 0, 0))(
                X[:T],
                U,
                ts,
                Y_dyn,
                Y_eq_inner,
                Z_inner,
            )
            return jax.vmap(_psd_project_inner)(blocks)

        @jax.jit
        def terminal_hess_block(z, y, zd):
            X, _U, theta = _slice_xut(z)
            if has_user_eq:
                y_eq_T = y[n + T * n + T * eq_dim_user :].reshape(eq_dim_user)
            else:
                y_eq_T = jnp.zeros(0, dtype=z.dtype)
            if has_ineq:
                z_ineq_T = zd[T * ineq_dim :].reshape(ineq_dim)
            else:
                z_ineq_T = jnp.zeros(0, dtype=z.dtype)

            def L_xt(xt):
                xx = xt[:n]
                thth = xt[n : n + td] if td > 0 else jnp.zeros(0, dtype=xt.dtype)
                if is_cost_mode:
                    return terminal_cost(xx, thth)
                terms = terminal_cost(xx, thth)
                if has_user_eq:
                    terms = terms + jnp.dot(
                        y_eq_T,
                        terminal_user_eq(xx, thth),
                    )
                if has_ineq:
                    terms = terms + jnp.dot(
                        z_ineq_T,
                        terminal_ineq(xx, thth),
                    )
                return terms

            H = jax.hessian(L_xt)(_terminal_input(X[T], theta))
            return _psd_project_terminal(H)

        # ===== Sparsity templates (hand-assembled) =====
        # Column layout of z = [X (n*(T+1)), U (m*T), Theta (td)].
        def _x_col(t, i):
            return t * n + i

        def _u_col(t, i):
            return nx + t * m + i

        def _theta_col(i):
            return nx + nu + i

        # --- c-Jacobian rows: init(n) + dyn(T*n) + user_eq((T+1)*eq_dim_user) ---
        c_rows: list[int] = []
        c_cols: list[int] = []
        # init: identity on x_0.
        for i in range(n):
            c_rows.append(i)
            c_cols.append(_x_col(0, i))
        # dyn defects: per stage t, identity on x_{t+1}, then dense block
        # on (x_t, u_t, theta) with sign -1 (we don't track signs in the
        # sparsity pattern, just nonzero positions).
        for t in range(T):
            row_off = n + t * n
            for i in range(n):
                c_rows.append(row_off + i)
                c_cols.append(_x_col(t + 1, i))
            for i in range(n):
                for j in range(n):
                    c_rows.append(row_off + i)
                    c_cols.append(_x_col(t, j))
            for i in range(n):
                for j in range(m):
                    c_rows.append(row_off + i)
                    c_cols.append(_u_col(t, j))
            if td > 0:
                for i in range(n):
                    for j in range(td):
                        c_rows.append(row_off + i)
                        c_cols.append(_theta_col(j))
        # user equalities: per stage t in 0..T-1 (inner) — dense block on
        # (x_t, u_t, theta); terminal — dense block on (x_T, theta).
        # Block-major ordering (all i's, x-cols block; then all i's, u-cols
        # block; then theta-cols) so the per-iter writeback can use
        # ueq_J[t, :, :n].reshape(-1) etc. directly.
        if has_user_eq:
            ueq_off_inner = n + T * n
            for t in range(T):
                row_off = ueq_off_inner + t * eq_dim_user
                # x-cols block
                for i in range(eq_dim_user):
                    for j in range(n):
                        c_rows.append(row_off + i)
                        c_cols.append(_x_col(t, j))
                # u-cols block
                for i in range(eq_dim_user):
                    for j in range(m):
                        c_rows.append(row_off + i)
                        c_cols.append(_u_col(t, j))
                # theta-cols block
                if td > 0:
                    for i in range(eq_dim_user):
                        for j in range(td):
                            c_rows.append(row_off + i)
                            c_cols.append(_theta_col(j))
            row_off = ueq_off_inner + T * eq_dim_user
            # Terminal: x-cols block, then theta-cols block.
            for i in range(eq_dim_user):
                for j in range(n):
                    c_rows.append(row_off + i)
                    c_cols.append(_x_col(T, j))
            if td > 0:
                for i in range(eq_dim_user):
                    for j in range(td):
                        c_rows.append(row_off + i)
                        c_cols.append(_theta_col(j))
        # c_rows / c_cols are in the FULL row indexing (size c_full_dim).
        # Build c_entry_keep mask (per natural-order entry: True if this
        # entry's full-row is kept by the active-row filter). Use it to
        # both build the filtered sparsity template and to filter the
        # per-iter c_vals before scattering.
        c_rows_full_arr = np.asarray(c_rows, dtype=np.int64)
        c_cols_full_arr = np.asarray(c_cols, dtype=np.int32)
        c_entry_keep = keep_mask_eq[c_rows_full_arr]
        c_rows_filtered_arr = full_to_filtered_eq[c_rows_full_arr[c_entry_keep]].astype(
            np.int32
        )
        c_cols_filtered_arr = c_cols_full_arr[c_entry_keep]
        # Keep both names around: c_rows_arr / c_cols_arr for the
        # per-iter writeback (it indexes into the FULL c_vals natural
        # order); the *_filtered_arr versions are what goes into the
        # sparse template.
        c_rows_arr = c_rows_full_arr.astype(np.int32)
        c_cols_arr = c_cols_full_arr
        jac_c_template = sp.csr_matrix(
            (
                np.ones(c_rows_filtered_arr.shape[0], dtype=np.float64),
                (c_rows_filtered_arr, c_cols_filtered_arr),
            ),
            shape=(y_dim, x_dim),
        )

        # --- g-Jacobian rows: (T+1) * ineq_dim ---
        # Block-major ordering (all i's, x-cols block; then all i's, u-cols
        # block; then theta-cols) so the per-iter writeback can use
        # ineq_J[t, :, :n].reshape(-1) etc. directly.
        if has_ineq:
            g_rows: list[int] = []
            g_cols: list[int] = []
            for t in range(T):
                row_off = t * ineq_dim
                # x-cols block
                for i in range(ineq_dim):
                    for j in range(n):
                        g_rows.append(row_off + i)
                        g_cols.append(_x_col(t, j))
                # u-cols block
                for i in range(ineq_dim):
                    for j in range(m):
                        g_rows.append(row_off + i)
                        g_cols.append(_u_col(t, j))
                # theta-cols block
                if td > 0:
                    for i in range(ineq_dim):
                        for j in range(td):
                            g_rows.append(row_off + i)
                            g_cols.append(_theta_col(j))
            row_off = T * ineq_dim
            # Terminal: x-cols block, then theta-cols block.
            for i in range(ineq_dim):
                for j in range(n):
                    g_rows.append(row_off + i)
                    g_cols.append(_x_col(T, j))
            if td > 0:
                for i in range(ineq_dim):
                    for j in range(td):
                        g_rows.append(row_off + i)
                        g_cols.append(_theta_col(j))
            g_rows_full_arr = np.asarray(g_rows, dtype=np.int64)
            g_cols_full_arr = np.asarray(g_cols, dtype=np.int32)
            g_entry_keep = keep_mask_ineq[g_rows_full_arr]
            g_rows_filtered_arr = full_to_filtered_ineq[
                g_rows_full_arr[g_entry_keep]
            ].astype(np.int32)
            g_cols_filtered_arr = g_cols_full_arr[g_entry_keep]
            g_rows_arr = g_rows_full_arr.astype(np.int32)
            g_cols_arr = g_cols_full_arr
            jac_g_template = sp.csr_matrix(
                (
                    np.ones(g_rows_filtered_arr.shape[0], dtype=np.float64),
                    (g_rows_filtered_arr, g_cols_filtered_arr),
                ),
                shape=(s_dim, x_dim),
            )
        else:
            g_rows_arr = np.zeros(0, dtype=np.int32)
            g_cols_arr = np.zeros(0, dtype=np.int32)
            g_entry_keep = np.zeros(0, dtype=bool)
            g_rows_filtered_arr = np.zeros(0, dtype=np.int32)
            g_cols_filtered_arr = np.zeros(0, dtype=np.int32)
            jac_g_template = sp.csr_matrix((s_dim, x_dim))

        # --- Lagrangian Hessian upper triangle (block-diagonal in stages,
        # theta-coupling shared across stages) ---
        h_rows: list[int] = []
        h_cols: list[int] = []
        h_writebacks: list[dict] = []  # bookkeeping for theta scatter-add
        for t in range(T):
            x_idx = [_x_col(t, i) for i in range(n)]
            u_idx = [_u_col(t, i) for i in range(m)]
            th_idx = [_theta_col(i) for i in range(td)]
            block_idx = x_idx + u_idx + th_idx  # block layout = (x, u, theta)
            # Track the natural-order index range for this stage's block.
            start_idx = len(h_rows)
            for ii, ri in enumerate(block_idx):
                for jj, cj in enumerate(block_idx):
                    if ri <= cj:
                        h_rows.append(ri)
                        h_cols.append(cj)
            end_idx = len(h_rows)
            h_writebacks.append(
                {
                    "kind": "inner",
                    "t": t,
                    "start": start_idx,
                    "end": end_idx,
                    "block_size": n + m + td,
                }
            )
        x_idx_T = [_x_col(T, i) for i in range(n)]
        th_idx = [_theta_col(i) for i in range(td)]
        block_idx_T = x_idx_T + th_idx
        start_idx = len(h_rows)
        for ii, ri in enumerate(block_idx_T):
            for jj, cj in enumerate(block_idx_T):
                if ri <= cj:
                    h_rows.append(ri)
                    h_cols.append(cj)
        end_idx = len(h_rows)
        h_writebacks.append(
            {
                "kind": "terminal",
                "start": start_idx,
                "end": end_idx,
                "block_size": n + td,
            }
        )
        h_rows_arr = np.asarray(h_rows, dtype=np.int32)
        h_cols_arr = np.asarray(h_cols, dtype=np.int32)
        upp_hess_template = sp.csc_matrix(
            (np.ones(h_rows_arr.shape[0], dtype=np.float64), (h_rows_arr, h_cols_arr)),
            shape=(x_dim, x_dim),
        )

        # Detect duplicate (row, col) entries — this happens when theta
        # is present, because every stage's block has the theta-theta
        # entries in the same global positions. scipy's COO->CSC sums
        # duplicates by default, but our natural-order writeback needs
        # to account for it. We compute per-natural-index targets in
        # the canonical CSC data array; entries that collide get the
        # same target and we use np.add.at for the writeback.
        # For the non-theta case there are no collisions and the simple
        # permutation suffices.
        def _csr_perm_with_duplicates(
            rows: np.ndarray, cols: np.ndarray, template
        ) -> np.ndarray:
            """Map natural-order indices to canonical CSR/CSC .data slots.

            ``template.data[target[k]] += values[k]`` reconstructs the
            sparse matrix value-correctly even when (rows[k], cols[k])
            has duplicates across k.
            """
            # CSR sorts by (row, col within row). For each natural-order
            # entry, find which (sorted) data slot it belongs to.
            # Use scipy's own coo->csr machinery to get the sorted order.
            # The template was built by COO->CSR/CSC, which sums dupes.
            # We re-derive the canonical (row, col) per data slot from
            # the template, then for each (rows[k], cols[k]) find that
            # slot via a hash lookup.
            from scipy.sparse import csr_matrix, csc_matrix

            if isinstance(template, csr_matrix):
                ax_major, ax_minor = template.indptr, template.indices
                n_major = template.shape[0]

                def _slot(r, c):
                    lo, hi = ax_major[r], ax_major[r + 1]
                    # binary search for c in ax_minor[lo:hi]
                    sub = ax_minor[lo:hi]
                    pos = np.searchsorted(sub, c)
                    return lo + pos
            elif isinstance(template, csc_matrix):
                ax_major, ax_minor = template.indptr, template.indices

                def _slot(r, c):
                    lo, hi = ax_major[c], ax_major[c + 1]
                    sub = ax_minor[lo:hi]
                    pos = np.searchsorted(sub, r)
                    return lo + pos
            else:
                raise TypeError(f"unsupported sparse type {type(template).__name__}")
            target = np.empty(rows.shape[0], dtype=np.int64)
            for k in range(rows.shape[0]):
                target[k] = _slot(int(rows[k]), int(cols[k]))
            return target

        # c_target / g_target use the FILTERED arrays — they point into the
        # filtered sparse template's data slots. At per-iter time we
        # assemble c_vals over the FULL natural order then filter via
        # c_entry_keep before scattering through c_target.
        c_target = _csr_perm_with_duplicates(
            c_rows_filtered_arr,
            c_cols_filtered_arr,
            jac_c_template,
        )
        h_target = _csr_perm_with_duplicates(h_rows_arr, h_cols_arr, upp_hess_template)
        if has_ineq:
            g_target = _csr_perm_with_duplicates(
                g_rows_filtered_arr,
                g_cols_filtered_arr,
                jac_g_template,
            )
        else:
            g_target = np.zeros(0, dtype=np.int64)

        # Per-stage writeback index lookups (for the Hessian, we need to
        # know which .data slots each stage's natural-order block maps to).
        # For c-Jacobian and g-Jacobian, the targets are already non-
        # duplicated (each entry has a unique (row, col)) except in the
        # presence of theta-coupling, where dyn / user_eq / ineq rows have
        # theta cols shared across stages — but those are SEPARATE
        # (row, col) pairs since the rows differ, so no duplicates.
        # For the Hessian, theta-theta entries DO have the same (row, col)
        # across stages, so we use np.add.at to accumulate.

        # Reusable buffers — same scipy sparse object reused per call.
        jac_c_buf = jac_c_template.copy()
        jac_g_buf = jac_g_template.copy()
        upp_hess_buf = upp_hess_template.copy()

        # ===== ProblemDimensions / QDLDLSettings =====
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
        from tests.comparison.sip_kkt_perm import compute_kkt_perm_inv_and_nnzs

        _perm_result = compute_kkt_perm_inv_and_nnzs(
            upp_hess_template,
            jac_c_template,
            jac_g_template,
        )
        qs.kkt_pinv = _perm_result.perm_inv
        pd.kkt_nnz = _perm_result.kkt_nnz
        pd.kkt_L_nnz = _perm_result.L_nnz

        # ===== Settings =====
        # Layering: constructor defaults <- problem.metadata['sip_settings']
        # <- problem.metadata['sip_<backend>_settings'] <- CLI
        # self.sip_extra_settings. The backend-specific key lets a
        # problem ship per-backend tuning where the JAX and CasADi
        # numerics paths need different schedules.
        problem_overrides = dict(problem.metadata.get("sip_settings", {}))
        backend_key = f"sip_{self.backend}_settings"
        problem_overrides.update(problem.metadata.get(backend_key, {}))
        ss = sip.Settings()
        ss.max_iterations = int(self.max_iter)
        ss.max_kkt_violation = float(self._effective_tol)
        ss.enable_elastics = self.enable_elastics
        if self.enable_elastics:
            ss.elastic_var_cost_coeff = self.elastic_var_cost_coeff
        ss.penalty_parameter_increase_factor = self.penalty_parameter_increase_factor
        ss.mu_update_factor = self.mu_update_factor
        ss.initial_mu = self.initial_mu
        ss.initial_penalty_parameter = self.initial_penalty_parameter
        ss.print_logs = self.print_logs
        if not self.print_logs:
            ss.print_line_search_logs = False
            ss.print_search_direction_logs = False
            ss.print_derivative_check_logs = False
        ss.assert_checks_pass = False
        for k, v in problem_overrides.items():
            setattr(ss, k, v)
        for k, v in self.sip_extra_settings.items():
            setattr(ss, k, v)

        # ===== Warm start =====
        z_init = np.concatenate(
            [
                np.asarray(problem.X_init, dtype=np.float64).reshape(-1),
                np.asarray(problem.U_init, dtype=np.float64).reshape(-1),
                np.asarray(problem.Theta_init, dtype=np.float64).reshape(-1),
            ]
        )

        # ===== Per-iter recording =====
        iter_xut: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

        # ===== Model callback =====
        def mc(mci):
            mco = sip.ModelCallbackOutput()
            z_jax = jnp.asarray(mci.x, dtype=jnp.float64)
            z_np_iter = np.asarray(mci.x, dtype=np.float64).copy()
            Xi, Ui, Ti = _slice_iterate(z_np_iter, problem)
            iter_xut.append((Xi, Ui, Ti))

            # Scalar / vector primitives.
            mco.f = float(np.asarray(f_fn(z_jax)))
            mco.c = np.asarray(c_fn_filtered(z_jax), dtype=np.float64).copy()
            if has_ineq:
                mco.g = np.asarray(g_fn_filtered(z_jax), dtype=np.float64).copy()
            else:
                mco.g = np.zeros(0, dtype=np.float64)
            mco.gradient_f = np.asarray(
                grad_f_fn(z_jax),
                dtype=np.float64,
            ).copy()

            # ----- c-Jacobian (init + dyn + user_eq) -----
            dyn_J = np.asarray(dyn_jac_blocks(z_jax), dtype=np.float64)
            c_vals = np.empty(c_rows_arr.shape[0], dtype=np.float64)
            idx = 0
            # init: identity (n entries).
            c_vals[idx : idx + n] = 1.0
            idx += n
            # dyn defects: per stage, identity on x_{t+1} + (-dyn_J[t]).
            for t in range(T):
                c_vals[idx : idx + n] = 1.0
                idx += n  # x_{t+1} identity
                # (-) dyn / dx (n*n entries)
                c_vals[idx : idx + n * n] = -dyn_J[t, :, :n].reshape(-1)
                idx += n * n
                # (-) dyn / du (n*m entries)
                c_vals[idx : idx + n * m] = -dyn_J[t, :, n : n + m].reshape(-1)
                idx += n * m
                if td > 0:
                    c_vals[idx : idx + n * td] = -dyn_J[
                        t, :, n + m : n + m + td
                    ].reshape(-1)
                    idx += n * td
            # user equalities.
            if has_user_eq:
                ueq_inner_J = np.asarray(
                    user_eq_jac_inner_blocks(z_jax),
                    dtype=np.float64,
                )
                ueq_term_J = np.asarray(
                    user_eq_jac_terminal(z_jax),
                    dtype=np.float64,
                )
                for t in range(T):
                    c_vals[idx : idx + eq_dim_user * n] = ueq_inner_J[t, :, :n].reshape(
                        -1
                    )
                    idx += eq_dim_user * n
                    c_vals[idx : idx + eq_dim_user * m] = ueq_inner_J[
                        t, :, n : n + m
                    ].reshape(-1)
                    idx += eq_dim_user * m
                    if td > 0:
                        c_vals[idx : idx + eq_dim_user * td] = ueq_inner_J[
                            t, :, n + m : n + m + td
                        ].reshape(-1)
                        idx += eq_dim_user * td
                # terminal user eq.
                c_vals[idx : idx + eq_dim_user * n] = ueq_term_J[:, :n].reshape(-1)
                idx += eq_dim_user * n
                if td > 0:
                    c_vals[idx : idx + eq_dim_user * td] = ueq_term_J[
                        :, n : n + td
                    ].reshape(-1)
                    idx += eq_dim_user * td
            jac_c_buf.data[:] = 0.0
            # c_vals is in natural FULL order; filter to active entries
            # before scattering through the filtered template's data slots.
            np.add.at(jac_c_buf.data, c_target, c_vals[c_entry_keep])
            mco.jacobian_c = jac_c_buf

            # ----- g-Jacobian (per-stage ineq + terminal) -----
            if has_ineq:
                ineq_J = np.asarray(
                    ineq_jac_inner_blocks(z_jax),
                    dtype=np.float64,
                )
                ineq_term_J = np.asarray(
                    ineq_jac_terminal(z_jax),
                    dtype=np.float64,
                )
                g_vals = np.empty(g_rows_arr.shape[0], dtype=np.float64)
                idx = 0
                for t in range(T):
                    g_vals[idx : idx + ineq_dim * n] = ineq_J[t, :, :n].reshape(-1)
                    idx += ineq_dim * n
                    g_vals[idx : idx + ineq_dim * m] = ineq_J[t, :, n : n + m].reshape(
                        -1
                    )
                    idx += ineq_dim * m
                    if td > 0:
                        g_vals[idx : idx + ineq_dim * td] = ineq_J[
                            t, :, n + m : n + m + td
                        ].reshape(-1)
                        idx += ineq_dim * td
                g_vals[idx : idx + ineq_dim * n] = ineq_term_J[:, :n].reshape(-1)
                idx += ineq_dim * n
                if td > 0:
                    g_vals[idx : idx + ineq_dim * td] = ineq_term_J[
                        :, n : n + td
                    ].reshape(-1)
                    idx += ineq_dim * td
                jac_g_buf.data[:] = 0.0
                np.add.at(jac_g_buf.data, g_target, g_vals[g_entry_keep])
                mco.jacobian_g = jac_g_buf
            else:
                mco.jacobian_g = jac_g_buf

            # ----- Lagrangian Hessian (per-stage PSD-projected blocks) -----
            # SIP gives us multipliers in the FILTERED layout (mci.y has
            # size y_dim). The Hessian builders index multipliers by FULL
            # row layout (so per-stage Y_dyn / Y_eq_inner slices line up
            # with all-T-stage vmaps). Scatter mci.y / mci.z back into
            # full vectors first.
            y_full = np.zeros(c_full_dim, dtype=np.float64)
            y_full[keep_mask_eq] = np.asarray(mci.y, dtype=np.float64)
            y_jax = jnp.asarray(y_full, dtype=jnp.float64)
            if has_ineq:
                zd_full = np.zeros(g_full_dim, dtype=np.float64)
                zd_full[keep_mask_ineq] = np.asarray(mci.z, dtype=np.float64)
                zd_jax = jnp.asarray(zd_full, dtype=jnp.float64)
            else:
                zd_jax = jnp.zeros(0, dtype=jnp.float64)
            inner_blocks = np.asarray(
                inner_hess_blocks(z_jax, y_jax, zd_jax),
                dtype=np.float64,
            )
            terminal_block = np.asarray(
                terminal_hess_block(z_jax, y_jax, zd_jax),
                dtype=np.float64,
            )
            h_vals = np.empty(h_rows_arr.shape[0], dtype=np.float64)
            for wb in h_writebacks:
                if wb["kind"] == "inner":
                    blk = inner_blocks[wb["t"]]
                else:
                    blk = terminal_block
                bsz = wb["block_size"]
                vidx = wb["start"]
                for ii in range(bsz):
                    for jj in range(bsz):
                        ri = (
                            blk_row_for_inner(ii, n, m, T, td, wb)
                            if wb["kind"] == "inner"
                            else blk_row_for_terminal(ii, n, T, td)
                        )
                        cj = (
                            blk_row_for_inner(jj, n, m, T, td, wb)
                            if wb["kind"] == "inner"
                            else blk_row_for_terminal(jj, n, T, td)
                        )
                        if ri <= cj:
                            h_vals[vidx] = blk[ii, jj]
                            vidx += 1
            upp_hess_buf.data[:] = 0.0
            np.add.at(upp_hess_buf.data, h_target, h_vals)
            mco.upper_hessian_lagrangian = upp_hess_buf

            return mco

        # Helper closures used inside mc for Hessian writeback row/col mapping.
        # Defined OUTSIDE mc to avoid per-call closure overhead.
        def blk_row_for_inner(
            local_idx: int, n: int, m: int, T_: int, td: int, wb: dict
        ) -> int:
            t = wb["t"]
            if local_idx < n:
                return t * n + local_idx
            if local_idx < n + m:
                return nx + t * m + (local_idx - n)
            return nx + nu + (local_idx - n - m)

        def blk_row_for_terminal(local_idx: int, n: int, T_: int, td: int) -> int:
            if local_idx < n:
                return T_ * n + local_idx
            return nx + nu + (local_idx - n)

        # ===== Construct solver =====
        solver = sip.Solver(ss, qs, pd, mc)

        vars_in = sip.Variables(pd)
        vars_in.x[:] = z_init
        vars_in.s[:] = 1.0
        vars_in.y[:] = 0.0
        vars_in.e[:] = 0.0
        vars_in.z[:] = 1.0

        # ===== Warm up JIT caches on z_init =====
        try:
            z_warm = jnp.asarray(z_init, dtype=jnp.float64)
            # Hessian builders index multipliers in the FULL row layout,
            # so warm-up vectors must match c_full_dim / g_full_dim, not
            # the filtered y_dim / s_dim that sip sees.
            y_warm = jnp.zeros(c_full_dim, dtype=jnp.float64)
            zd_warm = (
                jnp.ones(g_full_dim, dtype=jnp.float64)
                if has_ineq
                else jnp.zeros(0, dtype=jnp.float64)
            )
            jax.block_until_ready(f_fn(z_warm))
            jax.block_until_ready(grad_f_fn(z_warm))
            jax.block_until_ready(c_fn(z_warm))
            jax.block_until_ready(dyn_jac_blocks(z_warm))
            if has_user_eq:
                jax.block_until_ready(user_eq_jac_inner_blocks(z_warm))
                jax.block_until_ready(user_eq_jac_terminal(z_warm))
            if has_ineq:
                jax.block_until_ready(g_fn(z_warm))
                jax.block_until_ready(ineq_jac_inner_blocks(z_warm))
                jax.block_until_ready(ineq_jac_terminal(z_warm))
            jax.block_until_ready(inner_hess_blocks(z_warm, y_warm, zd_warm))
            jax.block_until_ready(terminal_hess_block(z_warm, y_warm, zd_warm))
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
        X, U, Theta = _slice_iterate(z_val, problem)

        # ===== Multiplier extraction =====
        # SIP's vars_in.y has shape (y_dim,) in the FILTERED layout;
        # evaluate_problem expects the FULL layout
        # [init(n); dyn(T*n); user_eq((T+1)*eq_dim_user)]. Scatter
        # filtered → full and sign-flip the dyn rows (c() encodes
        # X[t+1] - dyn(...) while evaluate_problem measures
        # dyn(...) - X[t+1]).
        try:
            y_filtered = np.asarray(vars_in.y, dtype=np.float64).reshape(-1)
            multipliers_eq_full = np.zeros(c_full_dim, dtype=np.float64)
            multipliers_eq_full[keep_mask_eq] = y_filtered
            multipliers_eq_full[n : n + T * n] = -multipliers_eq_full[n : n + T * n]
        except Exception:  # noqa: BLE001
            multipliers_eq_full = None
        try:
            if has_ineq:
                z_filtered = np.asarray(vars_in.z, dtype=np.float64).reshape(-1)
                multipliers_ineq_full = np.zeros(g_full_dim, dtype=np.float64)
                multipliers_ineq_full[keep_mask_ineq] = z_filtered
            else:
                multipliers_ineq_full = None
        except Exception:  # noqa: BLE001
            multipliers_ineq_full = None

        if output is None:
            return pack_solver_result(
                solver_name=self.name,
                problem_name=problem.name,
                problem=problem,
                X=X,
                U=U,
                Theta=Theta,
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
            X=X,
            U=U,
            Theta=Theta,
            iterations=int(output.num_iterations),
            solve_time_ms=solve_time_ms,
            success=success,
            notes=str(status),
            multipliers_eq=multipliers_eq_full,
            multipliers_ineq=multipliers_ineq_full,
            iterates_xut=iter_xut or None,
        )

    def _solve_casadi(self, problem: ProblemSpec) -> SolverResult:  # noqa: PLR0915
        """CasADi-backend variant of ``solve``.

        Same SIP wiring as the JAX path (Settings / QDLDLSettings /
        ProblemDimensions / Solver / Variables), but the per-iter model
        callback dispatches to compiled CasADi Functions instead of
        jit'd JAX functions. The decision-vector layout is identical
        (``z = vec(X) ++ vec(U) ++ Theta``) so iter counts should match
        the JAX backend to within FP noise on analytical problems.

        Sparsity is taken directly from CasADi's symbolic-graph
        analysis (no mock-eval probe), and the Jacobian / Hessian
        templates are built from the CasADi sparsity patterns. The
        cost-only Hessian is materialised dense per-call for the
        eigen-clamp PSD projection — same as the JAX path.

        Caller is responsible for verifying
        ``problem.metadata["casadi_builder"]`` is present and the
        Hessian mode is "cost"; the constructor + ``solve`` enforce both.
        """
        sip = _import_sip()
        ca = _import_casadi()
        from scipy import sparse as sp

        # --- Build cost + constraint callables --------------------------------
        nlp = _build_casadi_nlp(problem)
        f_fn = nlp["f_fn"]
        c_fn = nlp["c_fn"]
        g_fn = nlp["g_fn"]
        grad_f_fn = nlp["grad_f_fn"]
        jac_c_fn = nlp["jac_c_fn"]
        jac_g_fn = nlp["jac_g_fn"]
        stage_hess_meta = nlp["stage_hess_meta"]
        upp_hess_pat_dense = nlp["upp_hess_pat"]
        x_dim = nlp["x_dim"]
        y_dim = nlp["y_dim"]
        s_dim = nlp["s_dim"]
        jac_c_sparsity = nlp["jac_c_sparsity"]
        jac_g_sparsity = nlp["jac_g_sparsity"]

        # --- Build initial guess ---------------------------------------------
        z_init = np.concatenate(
            [
                np.asarray(problem.X_init, dtype=np.float64).reshape(-1),
                np.asarray(problem.U_init, dtype=np.float64).reshape(-1),
                np.asarray(problem.Theta_init, dtype=np.float64).reshape(-1),
            ]
        )
        assert z_init.size == x_dim, (
            f"CasADi flat NLP got x_dim={x_dim} but z_init.size={z_init.size}; "
            "this indicates a mismatch between ProblemSpec dimensions and the "
            "CasADi builder output."
        )

        # --- Build sparse-matrix templates from CasADi sparsity --------------
        # SIP's convention: jacobians as CSR of shape (out_dim, x_dim) — the
        # C++ side reinterprets CSR row-major data as CSC of the transpose,
        # so ProblemDimensions.is_jacobian_*_transposed = True. CasADi
        # gives us the Jacobian in CSC; we need to permute the per-call
        # nonzero values (CSC order) into the CSR template's .data slots
        # (row-major order). The permutation is computed once via the
        # idx-CSC -> idx-CSR trick: build a CSC whose data is the
        # 0..nnz-1 enumeration of CasADi's nz, convert to CSR, and the
        # resulting .data is exactly the permutation we need.
        def _build_jac_template(sparsity, out_dim: int):
            """Return (csr_template, casadi_nz_to_csr_perm).

            ``csr_template`` is a scipy.sparse.csr_matrix with the right
            structural nnz pattern (zero values).
            ``casadi_nz_to_csr_perm`` is an ``np.ndarray`` of indices such
            that per-call we can do
                csr_template.data[:] = casadi_dm.nonzeros()[perm]
            to fill in the new values without going through dense.
            """
            if sparsity is None or out_dim == 0:
                empty = sp.csr_matrix((out_dim, x_dim), dtype=np.float64)
                return empty, np.zeros(0, dtype=np.intp)
            nnz = sparsity.nnz()
            csc_pat = sp.csc_matrix(
                (
                    np.ones(nnz, dtype=np.float64),
                    np.asarray(sparsity.row(), dtype=np.intp),
                    np.asarray(sparsity.colind(), dtype=np.intp),
                ),
                shape=(out_dim, x_dim),
            )
            csr_template = csc_pat.tocsr()
            # Build the permutation: idx_csc has 0..nnz-1 as its data, in
            # CasADi's CSC order; idx_csr.data is then perm[i] = which
            # CasADi nz index ends up at CSR position i.
            idx_csc = sp.csc_matrix(
                (
                    np.arange(nnz, dtype=np.intp),
                    np.asarray(sparsity.row(), dtype=np.intp),
                    np.asarray(sparsity.colind(), dtype=np.intp),
                ),
                shape=(out_dim, x_dim),
            )
            perm = np.asarray(idx_csc.tocsr().data, dtype=np.intp)
            return csr_template, perm

        jac_c_template, jac_c_perm = _build_jac_template(jac_c_sparsity, y_dim)
        jac_g_template, jac_g_perm = _build_jac_template(jac_g_sparsity, s_dim)

        # Hessian: CSC of upper triangle, shape (x_dim, x_dim). The sparse
        # template comes from the union of per-stage block symbolic
        # nonzeros (built in ``_build_casadi_nlp`` as ``upp_hess_pat``).
        # Each per-call PSD-projected stage block is written into the
        # corresponding slot via fancy indexing.
        upp_hess_template = upp_hess_pat_dense.astype(np.float64).tocsc()

        # --- Build ProblemDimensions / QDLDLSettings -------------------------
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
        # AMD-first KKT permutation (see comment + import in the JAX
        # backend block above; same module).
        from tests.comparison.sip_kkt_perm import compute_kkt_perm_inv_and_nnzs

        _perm_result = compute_kkt_perm_inv_and_nnzs(
            upp_hess_template,
            jac_c_template,
            jac_g_template,
        )
        qs.kkt_pinv = _perm_result.perm_inv
        pd.kkt_nnz = _perm_result.kkt_nnz
        pd.kkt_L_nnz = _perm_result.L_nnz

        # --- Settings --------------------------------------------------------
        problem_overrides = problem.metadata.get("sip_settings", {})
        ss = sip.Settings()
        ss.max_iterations = int(self.max_iter)
        ss.max_kkt_violation = float(self._effective_tol)
        ss.enable_elastics = self.enable_elastics
        if self.enable_elastics:
            ss.elastic_var_cost_coeff = self.elastic_var_cost_coeff
        ss.penalty_parameter_increase_factor = self.penalty_parameter_increase_factor
        ss.mu_update_factor = self.mu_update_factor
        ss.initial_mu = self.initial_mu
        ss.initial_penalty_parameter = self.initial_penalty_parameter
        ss.print_logs = self.print_logs
        if not self.print_logs:
            ss.print_line_search_logs = False
            ss.print_search_direction_logs = False
            ss.print_derivative_check_logs = False
        ss.assert_checks_pass = False
        for k, v in problem_overrides.items():
            setattr(ss, k, v)
        for k, v in self.sip_extra_settings.items():
            setattr(ss, k, v)

        # --- Build the model callback ---------------------------------------
        # Reusable buffers, written in-place each callback. We bind them to
        # the closure so SIP sees the same memory each call.
        jac_c_buf = jac_c_template.copy()
        jac_g_buf = jac_g_template.copy()
        upp_hess_buf = upp_hess_template.copy()

        psd_reg_delta = self.psd_reg_delta

        # Pre-compute a write-back plan for each per-stage Hessian block:
        # for each stage, determine which entries of the dense block (i, j
        # ranging over [0, block_size)) land in which slot of
        # ``upp_hess_buf.data``. Computed once; per-call we just iterate
        # the stages and do an in-place assignment.
        #
        # The buffer is CSC; CSC stores .data in column-major order, and
        # column k's entries are at indices [indptr[k], indptr[k+1]) with
        # row indices ``indices[indptr[k]:indptr[k+1]]``. We build, for
        # each stage block, two flattened arrays:
        #
        #   ``block_src_i[k], block_src_j[k]`` — block-local (row, col)
        #     of the k-th nonzero we want to write back (block-local in
        #     [0, block_size)). We always write the upper-triangle entries
        #     (row <= col), summing contributions across stages where
        #     blocks overlap (theta cross-stage entries).
        #   ``buf_data_idx[k]`` — index into ``upp_hess_buf.data`` that
        #     this entry writes to.
        #
        # Since multiple stages can hit the same global (i, j) entry (the
        # theta-theta block in particular), we sum contributions by
        # `np.add.at` per call. Build the index dictionary now to map
        # (global_i, global_j) -> upp_hess_buf.data slot.
        buf_indptr = upp_hess_buf.indptr
        buf_indices = upp_hess_buf.indices
        gij_to_data_idx = {}
        for col in range(upp_hess_buf.shape[1]):
            for k in range(buf_indptr[col], buf_indptr[col + 1]):
                row = buf_indices[k]
                gij_to_data_idx[(row, col)] = k

        # For each stage, build the index arrays.
        stage_writebacks = []
        for meta in stage_hess_meta:
            idx = meta["z_idx"]
            bs = meta["block_size"]
            src_i = []
            src_j = []
            data_idx = []
            for i in range(bs):
                gi = int(idx[i])
                for j in range(bs):
                    gj = int(idx[j])
                    if gi > gj:
                        continue  # only upper triangle
                    pos = gij_to_data_idx.get((gi, gj))
                    if pos is None:
                        # Shouldn't happen — upp_hess_pat covers every
                        # stage-block entry by construction.
                        continue
                    src_i.append(i)
                    src_j.append(j)
                    data_idx.append(pos)
            stage_writebacks.append(
                {
                    "fn": meta["fn"],
                    "z_idx": idx,
                    "block_size": bs,
                    "src_i": np.asarray(src_i, dtype=np.intp),
                    "src_j": np.asarray(src_j, dtype=np.intp),
                    "data_idx": np.asarray(data_idx, dtype=np.intp),
                }
            )

        # Per-iter recorder (CasADi backend variant).
        iter_xut: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

        def mc(mci):
            mco = sip.ModelCallbackOutput()
            z_np = np.asarray(mci.x, dtype=np.float64)
            # Per-iter snapshot for post-process history extraction.
            Xi, Ui, Ti = _slice_iterate(z_np.copy(), problem)
            iter_xut.append((Xi, Ui, Ti))

            # CasADi Function calls accept numpy arrays directly; returns
            # casadi.DM (sparse) for sparse outputs, dense for dense.
            mco.f = float(f_fn(z_np))
            mco.c = np.asarray(c_fn(z_np), dtype=np.float64).reshape(-1).copy()
            if s_dim > 0:
                mco.g = np.asarray(g_fn(z_np), dtype=np.float64).reshape(-1).copy()
            else:
                mco.g = np.zeros(0, dtype=np.float64)
            mco.gradient_f = (
                np.asarray(grad_f_fn(z_np), dtype=np.float64).reshape(-1).copy()
            )

            # Jacobians: extract values directly from the casadi.DM's
            # nonzeros() (CSC-ordered list, ~40x faster than
            # np.asarray(DM) which densifies). The pre-computed permutation
            # ``jac_*_perm`` maps from CSC nz index -> CSR template .data
            # slot.
            if y_dim > 0:
                jc_nz = np.asarray(jac_c_fn(z_np).nonzeros(), dtype=np.float64)
                jac_c_buf.data[:] = jc_nz[jac_c_perm]
            mco.jacobian_c = jac_c_buf

            if s_dim > 0:
                jg_nz = np.asarray(jac_g_fn(z_np).nonzeros(), dtype=np.float64)
                jac_g_buf.data[:] = jg_nz[jac_g_perm]
            mco.jacobian_g = jac_g_buf

            # Per-stage cost Hessian + eigen-clamp PSD projection. Each
            # block is small (the per-stage dimension), so the eigh is
            # essentially free; the key benefit vs a single global eigh on
            # the full Hessian is avoiding cubic scaling in the horizon.
            #
            # Subtle: for problems with cross-stage theta (td > 0), the
            # per-stage blocks overlap on the theta dimension(s). We
            # ZERO the .data slots first and then ADD each stage's
            # contribution — this correctly sums the theta-theta blocks
            # across stages (matching what the global Hessian would give
            # before PSD-projection). After per-stage PSD-projection the
            # resulting Hessian differs from a global eigen-clamp; this
            # is the standard local-block-regularization trade-off that
            # SQP-with-stage-decomposed-Hessians takes, and is acceptable
            # because the cost-only mode is already an SQP approximation.
            upp_hess_buf.data[:] = 0.0
            for wb in stage_writebacks:
                z_slice = z_np[wb["z_idx"]]
                H_block = np.asarray(wb["fn"](z_slice), dtype=np.float64)
                if H_block.size == 0:
                    continue
                # PSD-project this block.
                if H_block.shape[0] > 1:
                    S, _V = np.linalg.eigh(H_block)
                    s_min = float(S[0])
                else:
                    s_min = float(H_block[0, 0])
                k = max(-s_min, 0.0) + psd_reg_delta
                if k > 0.0:
                    # Add k*I to the block.
                    bs = wb["block_size"]
                    H_block.flat[:: bs + 1] += k
                # Write back upper-triangle entries (sum across stages
                # for overlapping entries — np.add.at, scatter-add).
                vals = H_block[wb["src_i"], wb["src_j"]]
                np.add.at(upp_hess_buf.data, wb["data_idx"], vals)
            mco.upper_hessian_lagrangian = upp_hess_buf

            return mco

        # --- Construct the solver --------------------------------------------
        solver = sip.Solver(ss, qs, pd, mc)

        # --- Initial Variables ----------------------------------------------
        vars_in = sip.Variables(pd)
        vars_in.x[:] = z_init
        vars_in.s[:] = 1.0
        vars_in.y[:] = 0.0
        vars_in.e[:] = 0.0
        vars_in.z[:] = 1.0

        # --- Warm-up call -----------------------------------------------------
        # CasADi Functions have no JIT to amortise but we still call each
        # once to mirror the JAX path and to catch any structural issues
        # before timing begins.
        try:
            _ = float(f_fn(z_init))
            _ = np.asarray(c_fn(z_init))
            if s_dim > 0:
                _ = np.asarray(g_fn(z_init))
            _ = np.asarray(grad_f_fn(z_init))
            if y_dim > 0:
                _ = np.asarray(jac_c_fn(z_init))
            if s_dim > 0:
                _ = np.asarray(jac_g_fn(z_init))
            # Warm up each per-stage Hessian Function.
            for wb in stage_writebacks:
                _ = np.asarray(wb["fn"](z_init[wb["z_idx"]]))
        except Exception:  # noqa: BLE001
            pass

        # --- Solve -----------------------------------------------------------
        start = timer()
        try:
            output = solver.solve(vars_in)
            err = ""
        except Exception as e:  # noqa: BLE001
            output = None
            err = f"{type(e).__name__}: {e}"
        solve_time_ms = 1e3 * (timer() - start)

        z_val = np.asarray(vars_in.x, dtype=np.float64).copy()
        X, U, Theta = _slice_iterate(z_val, problem)

        # Multipliers — the CasADi backend doesn't filter rows so the
        # mapping is direct (one-to-one with evaluate_problem's
        # constraint stacks). The CasADi NLP encodes the dyn defect
        # as X[t+1] - dyn(...) (same as the JAX backend, opposite
        # from evaluate_problem) so we sign-flip just the dyn-defect
        # rows.
        try:
            multipliers_eq_full = np.asarray(
                vars_in.y,
                dtype=np.float64,
            ).reshape(-1)
            n_p = problem.n
            T_p = problem.T
            if multipliers_eq_full.size >= n_p + T_p * n_p:
                multipliers_eq_full[n_p : n_p + T_p * n_p] = -multipliers_eq_full[
                    n_p : n_p + T_p * n_p
                ]
        except Exception:  # noqa: BLE001
            multipliers_eq_full = None
        try:
            multipliers_ineq_full = np.asarray(
                vars_in.z,
                dtype=np.float64,
            ).reshape(-1)
        except Exception:  # noqa: BLE001
            multipliers_ineq_full = None

        if output is None:
            return pack_solver_result(
                solver_name=self.name,
                problem_name=problem.name,
                problem=problem,
                X=X,
                U=U,
                Theta=Theta,
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
            X=X,
            U=U,
            Theta=Theta,
            iterations=int(output.num_iterations),
            solve_time_ms=solve_time_ms,
            success=success,
            notes=str(status),
            multipliers_eq=multipliers_eq_full,
            multipliers_ineq=multipliers_ineq_full,
            iterates_xut=iter_xut or None,
        )


@register("sip")
def _factory(**kwargs) -> SolverAdapter:
    return SipAdapter(**kwargs)
