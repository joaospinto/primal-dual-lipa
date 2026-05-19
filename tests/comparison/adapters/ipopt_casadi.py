"""IPOPT-via-CasADi adapter.

Builds the multi-shooting NLP from a ``ProblemSpec`` and solves it with
IPOPT through CasADi's ``nlpsol`` interface. For analytical problems the
``ProblemSpec`` exposes a ``casadi_builder`` that returns the dynamics,
cost, and constraint expressions in CasADi (``SX``); for MJX problems
the ``ProblemSpec`` instead exposes a JAX callable wrapped via
``casadi.Callback`` so CasADi can still build a sparse NLP graph.

Both routes reduce to the same NLP:

    z   = (vec(X), vec(U), Theta)
    min  sum_t cost(x_t, u_t, theta, t)
    s.t. x_0 = x0
         x_{t+1} = dynamics(x_t, u_t, theta, t)         t = 0..T-1
         eq(x_t, u_t, theta, t) = 0                     t = 0..T
         ineq(x_t, u_t, theta, t) <= 0                  t = 0..T
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


def _import_casadi():
    import casadi as ca  # local import so missing CasADi only fails this adapter

    return ca


class _IterationRecorder:
    """Wraps a CasADi nlpsol iteration_callback to record per-iter (x, lam_g).

    The callback API expects an object whose input shapes match the
    nlpsol's outputs (``x``, ``f``, ``g``, ``lam_x``, ``lam_g``,
    ``lam_p``); we return a single scalar so CasADi treats us as a
    "no-op" iteration callback whose only side-effect is appending to
    ``self.iterates``.

    Lifetime: CasADi keeps a strong reference to the iteration_callback
    once nlpsol is constructed, so self.callback can be passed in
    safely without an explicit pin. (Unlike the JAX-callback Jacobians
    we create elsewhere.)
    """

    def __init__(self, nx: int, ng: int, np_: int):
        ca = _import_casadi()
        self.nx = nx
        self.ng = ng
        self.np_ = np_
        self.iterates: list[tuple[np.ndarray, np.ndarray]] = []

        outer_self = self

        class _Cb(ca.Callback):
            def __init__(cb_self):  # noqa: N804
                ca.Callback.__init__(cb_self)
                cb_self.construct("ipopt_iter_recorder", {})

            def get_n_in(cb_self):  # noqa: N805
                return ca.nlpsol_n_out()

            def get_n_out(cb_self):  # noqa: N805
                return 1

            def get_name_in(cb_self, i):  # noqa: N805
                return ca.nlpsol_out(i)

            def get_name_out(cb_self, _i):  # noqa: N805
                return "ret"

            def get_sparsity_in(cb_self, i):  # noqa: N805
                name = ca.nlpsol_out(i)
                if name == "f":
                    return ca.Sparsity.scalar()
                if name in ("x", "lam_x"):
                    return ca.Sparsity.dense(outer_self.nx, 1)
                if name in ("g", "lam_g"):
                    return ca.Sparsity.dense(outer_self.ng, 1)
                if name == "lam_p":
                    if outer_self.np_ > 0:
                        return ca.Sparsity.dense(outer_self.np_, 1)
                    return ca.Sparsity(0, 0)
                return ca.Sparsity.scalar()

            def eval(cb_self, args):  # noqa: N805
                x = np.asarray(args[0]).reshape(-1).copy()
                lam_g = np.asarray(args[4]).reshape(-1).copy()
                outer_self.iterates.append((x, lam_g))
                return [0]

        self.callback = _Cb()


class _JaxToCasadiCallback:
    """Adapt a JAX scalar/vector function to a CasADi Callback.

    Critical lifetime detail: CasADi keeps only weak references to its
    Callback objects. When CasADi later asks for a Jacobian via
    ``get_jacobian``, the returned Callback must outlive the parent
    (and the parent must outlive the IPOPT solve) — otherwise the
    runtime explodes with "Callback object has been deleted". We
    address this by keeping every callback we ever construct (parent +
    Jacobian + transposed-Jacobian) as instance attributes on this
    wrapper, and the user of the wrapper must hold the wrapper itself
    for the duration of the solve.
    """

    def __init__(self, name: str, fn, in_dim: int, out_dim: int):
        ca = _import_casadi()
        import jax

        self._name = name
        self._fn = jax.jit(fn)
        self._in_dim = in_dim
        self._out_dim = out_dim

        # JIT-compile the value and Jacobian computations once. Choose
        # reverse-mode when output-dim <= input-dim (fewer VJPs than
        # forward-mode JVPs), forward otherwise.
        if out_dim <= in_dim:
            self._jac_compiled = jax.jit(jax.jacrev(fn))
        else:
            self._jac_compiled = jax.jit(jax.jacfwd(fn))

        wrapper_self = self  # captured by the inner Callback class

        class _Cb(ca.Callback):
            def __init__(cb_self):  # noqa: N804
                ca.Callback.__init__(cb_self)
                cb_self.construct(name, {})

            def get_n_in(cb_self):  # noqa: N805
                return 1

            def get_n_out(cb_self):  # noqa: N805
                return 1

            def get_sparsity_in(cb_self, _i):  # noqa: N805
                return ca.Sparsity.dense(in_dim, 1)

            def get_sparsity_out(cb_self, _i):  # noqa: N805
                return ca.Sparsity.dense(out_dim, 1)

            def eval(cb_self, args):  # noqa: N805
                z = np.asarray(args[0]).reshape(-1)
                y = np.asarray(wrapper_self._fn(z)).reshape(out_dim, 1)
                return [y]

            def has_jacobian(cb_self):  # noqa: N805
                return True

            def get_jacobian(cb_self, name_unused, inames, onames, opts):  # noqa: N805, ARG002
                # Build (or reuse) the Jacobian-Callback and stash it on
                # the wrapper so it survives garbage collection.
                if wrapper_self._jac_cb is not None:
                    return wrapper_self._jac_cb

                in_d = in_dim
                out_d = out_dim
                jac_fn = wrapper_self._jac_compiled

                class _JacCb(ca.Callback):
                    def __init__(jac_self):  # noqa: N804
                        ca.Callback.__init__(jac_self)
                        jac_self.construct(name + "_jac", {})

                    def get_n_in(jac_self):  # noqa: N805
                        return 2  # x, dummy_out (CasADi convention for derivative callbacks)

                    def get_n_out(jac_self):  # noqa: N805
                        return 1

                    def get_sparsity_in(jac_self, i):  # noqa: N805
                        if i == 0:
                            return ca.Sparsity.dense(in_d, 1)
                        return ca.Sparsity.dense(out_d, 1)

                    def get_sparsity_out(jac_self, _i):  # noqa: N805
                        return ca.Sparsity.dense(out_d, in_d)

                    def eval(jac_self, args):  # noqa: N805
                        z = np.asarray(args[0]).reshape(-1)
                        J = np.asarray(jac_fn(z)).reshape(out_d, in_d)
                        return [J]

                wrapper_self._jac_cb = _JacCb()
                return wrapper_self._jac_cb

        self._jac_cb = None  # populated lazily by get_jacobian
        self._cb = _Cb()

    def cb(self):
        return self._cb


def _build_casadi_nlp_pure(problem: ProblemSpec, casadi_builder):
    """Build the NLP using a pure-CasADi expression builder.

    ``casadi_builder(SX_x, SX_u, SX_theta, t)`` returns a dict with keys:
    ``f`` (scalar cost), ``next_x`` (n-vector dynamics), ``eq`` (eq_dim
    vector), ``ineq`` (ineq_dim vector). Either ``eq`` or ``ineq`` may be
    None (or absent) to indicate "no such constraint at this stage".

    Returns ``(nlp, n_eq, n_ineq, z, X_sx, U_sx, eq_t_present, ineq_t_present)``
    where ``eq_t_present`` and ``ineq_t_present`` are length-(T+1) bool
    arrays recording, for each stage ``t``, whether the builder emitted
    a non-empty ``eq`` / ``ineq`` block. These are used by the multiplier
    remapping below to splice IPOPT's compact lam_g back into the full
    per-stage eq / ineq stack that ``evaluate_problem`` expects.
    """
    ca = _import_casadi()
    T, n, m, td = problem.T, problem.n, problem.m, problem.theta_dim

    X_sx = ca.SX.sym("X", n, T + 1)
    U_sx = ca.SX.sym("U", m, T)
    Theta_sx = ca.SX.sym("Theta", max(td, 1))  # CasADi can't build a 0-length sym; we slice it off

    z_pieces = [ca.reshape(X_sx, -1, 1), ca.reshape(U_sx, -1, 1)]
    if td > 0:
        z_pieces.append(Theta_sx[:td])
    z = ca.vertcat(*z_pieces)

    theta_arg = Theta_sx[:td] if td > 0 else ca.SX.zeros(0)

    f_total = ca.SX(0.0)
    g_eq = []
    g_ineq = []
    eq_t_present = np.zeros(T + 1, dtype=bool)
    ineq_t_present = np.zeros(T + 1, dtype=bool)

    # Initial state defect.
    g_eq.append(X_sx[:, 0] - ca.DM(np.asarray(problem.x0)))

    for t in range(T + 1):
        u_t = U_sx[:, t] if t < T else ca.SX.zeros(m)
        stage = casadi_builder(X_sx[:, t], u_t, theta_arg, t)
        f_total += stage["f"]
        if t < T:
            # Dynamics defect: x_{t+1} - f(x_t, u_t, theta, t) = 0
            g_eq.append(X_sx[:, t + 1] - stage["next_x"])
        eq_t = stage.get("eq")
        if eq_t is not None and eq_t.numel() > 0:
            g_eq.append(eq_t)
            eq_t_present[t] = True
        ineq_t = stage.get("ineq")
        if ineq_t is not None and ineq_t.numel() > 0:
            g_ineq.append(ineq_t)
            ineq_t_present[t] = True

    g_eq_vec = ca.vertcat(*g_eq) if g_eq else ca.SX.zeros(0)
    g_ineq_vec = ca.vertcat(*g_ineq) if g_ineq else ca.SX.zeros(0)
    g = ca.vertcat(g_eq_vec, g_ineq_vec)
    n_eq = g_eq_vec.numel()
    n_ineq = g_ineq_vec.numel()

    nlp = {"x": z, "f": f_total, "g": g}
    return nlp, n_eq, n_ineq, z, X_sx, U_sx, eq_t_present, ineq_t_present


def _remap_ipopt_eq_lam(
    lam_eq_compact: np.ndarray,
    problem: ProblemSpec,
    eq_t_present: np.ndarray,
) -> np.ndarray:
    """Splice IPOPT's compact eq lam_g (init + dyn defects + non-empty user eqs)
    into the full per-stage eq stack expected by evaluate_problem
    ``[init_defect (n,); dyn_defects (T*n,); user_eqs ((T+1)*eq_dim,)]``.

    Compact layout: ``[init (n,); dyn[0] (n,); dyn[1] (n,); ...; dyn[T-1] (n,);
                       user_eq[t] for each t with eq_t_present[t]]``.

    Sign convention: IPOPT's ``lam_g`` satisfies ``∇f + lam_g^T ∇g = 0``
    where ``g`` is the constraint AS WRITTEN to CasADi. The IPOPT NLP
    layout uses:

    * init defect: ``X[0] - x0``           (matches evaluator)
    * dyn defects: ``X[t+1] - dyn(...)``   (OPPOSITE sign from
      evaluator, which uses ``dyn(...) - X[t+1]``)
    * user eq:    ``equalities(...)``      (matches evaluator)

    So we negate the dyn-defect multipliers to bring them into the
    evaluator's convention. (init / user-eq need no flip.)

    Stages with ``eq_t_present[t] == False`` get zeros in the user-eq slot
    (their constraint was structurally absent from the IPOPT NLP and so
    has no associated multiplier; the corresponding row in
    evaluate_problem's eq stack is identically zero, so the multiplier
    we put there has no effect on the stationarity computation).
    """
    n = problem.n
    T = problem.T
    eq_dim = problem.eq_dim
    full_size = n + T * n + (T + 1) * eq_dim
    out = np.zeros(full_size, dtype=np.float64)
    # init defect — same sign.
    out[:n] = lam_eq_compact[:n]
    # dyn defects — sign-flip (IPOPT NLP encodes them as
    # X[t+1] - dyn(x_t, u_t) while evaluate_problem measures the
    # opposite sign).
    if T > 0:
        out[n:n + T * n] = -lam_eq_compact[n:n + T * n]
    # user eqs (per-stage; only stages where eq was present get filled)
    cursor = n + T * n
    src = n + T * n
    if eq_dim > 0:
        for t in range(T + 1):
            dst = cursor + t * eq_dim
            if eq_t_present[t]:
                out[dst:dst + eq_dim] = lam_eq_compact[src:src + eq_dim]
                src += eq_dim
            # else leave as zeros
    return out


def _remap_ipopt_ineq_lam(
    lam_ineq_compact: np.ndarray,
    problem: ProblemSpec,
    ineq_t_present: np.ndarray,
) -> np.ndarray:
    """Same idea as ``_remap_ipopt_eq_lam`` but for the inequality stack
    ``((T+1)*ineq_dim,)`` expected by evaluate_problem.
    """
    T = problem.T
    ineq_dim = problem.ineq_dim
    full_size = (T + 1) * ineq_dim
    out = np.zeros(full_size, dtype=np.float64)
    if ineq_dim == 0:
        return out
    src = 0
    for t in range(T + 1):
        dst = t * ineq_dim
        if ineq_t_present[t]:
            out[dst:dst + ineq_dim] = lam_ineq_compact[src:src + ineq_dim]
            src += ineq_dim
        # else leave as zeros
    return out


# Warm-start helpers live in tests.comparison.warm_starts; see that
# module's docstring for the rationale.
from tests.comparison.warm_starts import rollout_warm_start as _rollout_warm_start  # noqa: E402
from tests.comparison.warm_starts import (  # noqa: E402
    linspace_to_extracted_goal_warm_start as _auto_warm_start,
)


def _slice_iterate(z_val: np.ndarray, problem: ProblemSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n, m, T, td = problem.n, problem.m, problem.T, problem.theta_dim
    nx = n * (T + 1)
    nu = m * T
    X = z_val[:nx].reshape(T + 1, n)
    U = z_val[nx:nx + nu].reshape(T, m)
    Theta = z_val[nx + nu:nx + nu + td] if td > 0 else np.zeros(0)
    return X, U, Theta


class IpoptCasadiAdapter(SolverAdapter):
    """IPOPT through CasADi for any ``ProblemSpec`` exposing a CasADi builder.

    The ``ProblemSpec`` is expected to have ``metadata["casadi_builder"]``
    when going through the pure-CasADi route (analytical problems), or
    ``metadata["jax_nlp"]`` for the JAX-callback route (MJX problems).
    """

    name = "ipopt"

    def __init__(
        self,
        max_iter: int = 1000,
        tol: float = 1e-6,
        hessian_approximation: str = "exact",
        print_level: int = 0,
        timeout_s: Optional[float] = None,
        warm_start_strategy: str = "auto",
        # Defaults: a larger initial barrier (mu_init=10) and a looser
        # infeasibility-reduction threshold (0.99) reduce spurious
        # early restoration entries on warm starts that aren't already
        # in the feasible interior. Caller-provided
        # ``ipopt_extra_options`` win over these and the base options.
        mu_init: float = 1e1,
        required_infeasibility_reduction: float = 0.99,
        ipopt_extra_options: Optional[dict] = None,
    ) -> None:
        self.max_iter = max_iter
        self.tol = tol
        self.hessian_approximation = hessian_approximation
        self.print_level = print_level
        self.timeout_s = timeout_s
        # warm_start_strategy:
        #   "auto"    -> if X_init is a degenerate ``tile(x0)`` AND the
        #                terminal equality is ``x_T - goal == 0``, infer
        #                ``goal`` from the equality callback and use
        #                ``linspace(x0, goal, T+1)``; otherwise use the
        #                shipped X_init.
        #   "as_init" -> use problem.X_init / U_init verbatim.
        #   "rollout" -> forward-roll U_init through problem.dynamics
        #                from x0.
        self.warm_start_strategy = warm_start_strategy
        self.mu_init = mu_init
        self.required_infeasibility_reduction = required_infeasibility_reduction
        # Pass-through dict of arbitrary IPOPT options merged into the base
        # ones. Keeps the constructor clean while letting users experiment
        # without re-editing the adapter.
        self.ipopt_extra_options = ipopt_extra_options or {}

    def _solve_pure_casadi(self, problem: ProblemSpec) -> SolverResult:
        ca = _import_casadi()
        casadi_builder = problem.metadata["casadi_builder"]
        nlp, n_eq, n_ineq, z_sym, X_sx, U_sx, eq_t_present, ineq_t_present = (
            _build_casadi_nlp_pure(problem, casadi_builder)
        )
        # Equality constraints encoded as g[0:n_eq] = 0; inequality as g[n_eq:] <= 0.
        lbg = np.concatenate([np.zeros(n_eq), -np.inf * np.ones(n_ineq)])
        ubg = np.concatenate([np.zeros(n_eq), np.zeros(n_ineq)])
        # Per-iter recorder: a CasADi iteration_callback that appends
        # (x, lam_g) at every IPOPT outer iteration. We hold the recorder
        # on ``self`` so its callback object outlives the solve.
        nx_total = z_sym.numel()
        self._iter_recorder = _IterationRecorder(
            nx=nx_total, ng=n_eq + n_ineq, np_=0,
        )
        ipopt_opts = {
            "print_level": self.print_level,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "constr_viol_tol": self.tol,
            "hessian_approximation": self.hessian_approximation,
            "sb": "yes",  # suppress IPOPT banner
            # Tuned IPOPT defaults — see __init__ for the rationale.
            "mu_init": float(self.mu_init),
            "required_infeasibility_reduction": float(
                self.required_infeasibility_reduction,
            ),
        }
        if self.timeout_s is not None:
            ipopt_opts["max_wall_time"] = float(self.timeout_s)
        # Apply caller-provided overrides last so they win.
        ipopt_opts.update(self.ipopt_extra_options)
        opts = {
            "print_time": False,
            "ipopt": ipopt_opts,
            "iteration_callback": self._iter_recorder.callback,
        }
        solver = ca.nlpsol("ipopt_solver", "ipopt", nlp, opts)

        # Initial guess. The "auto" strategy detects a degenerate
        # ``tile(x0)`` warm start and substitutes a
        # linspace-to-inferred-goal — see ``_auto_warm_start``.
        if self.warm_start_strategy == "auto":
            X_ws, U_ws = _auto_warm_start(problem)
        elif self.warm_start_strategy == "rollout":
            X_ws, U_ws = _rollout_warm_start(problem)
        elif self.warm_start_strategy == "as_init":
            X_ws = np.asarray(problem.X_init)
            U_ws = np.asarray(problem.U_init)
        else:
            raise ValueError(
                f"unknown warm_start_strategy={self.warm_start_strategy!r}; "
                "expected 'auto', 'rollout', or 'as_init'",
            )
        z0 = np.concatenate([
            X_ws.reshape(-1),
            U_ws.reshape(-1),
            np.asarray(problem.Theta_init).reshape(-1),
        ])

        start = timer()
        sol = solver(x0=z0, lbg=lbg, ubg=ubg)
        solve_time_ms = 1e3 * (timer() - start)

        z_val = np.asarray(sol["x"]).reshape(-1)
        X, U, Theta = _slice_iterate(z_val, problem)
        # Multipliers: IPOPT's lam_g is the compact concatenation of
        # init defect + dyn defects + (only the stages that contributed)
        # user equalities, then the (only present at some stages)
        # inequality multipliers. We splice them into the full per-stage
        # eq / ineq stacks expected by evaluate_problem so missing
        # stages are filled with zeros (those rows are structurally
        # zero in the JAX evaluator's stack).
        lam_g_full = np.asarray(sol["lam_g"]).reshape(-1)
        lam_eq_compact = lam_g_full[:n_eq] if n_eq > 0 else np.zeros(0)
        lam_ineq_compact = lam_g_full[n_eq:n_eq + n_ineq] if n_ineq > 0 else np.zeros(0)
        # IPOPT's sign convention matches LIPA's (L = f + λ^T c + z^T g
        # with g <= 0, lam_g >= 0 at active upper bound).
        multipliers_eq = _remap_ipopt_eq_lam(lam_eq_compact, problem, eq_t_present)
        multipliers_ineq = _remap_ipopt_ineq_lam(lam_ineq_compact, problem, ineq_t_present)

        stats = solver.stats()
        iters = int(stats.get("iter_count", -1))
        success = bool(stats.get("success", False))

        # Per-iter histories from the recorded iterates. Recording
        # overhead happens AFTER the timed solve closes — the
        # _IterationRecorder only appends raw arrays inside the timed
        # window, the (cost, eq_v, ineq_v) computation lives here.
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
            X=X, U=U, Theta=Theta,
            iterations=iters,
            solve_time_ms=solve_time_ms,
            success=success,
            notes=stats.get("return_status", ""),
            multipliers_eq=multipliers_eq,
            multipliers_ineq=multipliers_ineq,
            iterates_xut=iterates_xut,
        )

    def _solve_jax_callback(self, problem: ProblemSpec) -> SolverResult:
        """Build the flat NLP through *per-stage* CasADi callbacks → JAX.

        Used for MJX problems. The key reason this works at MJX scale (where
        a single global JAX-callback approach fails because the resulting
        Jacobian is dense and exceeds MUMPS-tractable size) is that we
        register a SEPARATE CasADi callback per stage:

        * ``dyn_t``   for ``f_t = dynamics(x_t, u_t, t)``
        * ``cost_t``  for the scalar stage cost
        * ``ineq_t``  for the per-stage inequality vector

        CasADi then sees the symbolic graph
        ``g_eq = [x_0 - x0, x_1 - dyn_0(x_0,u_0), x_2 - dyn_1(x_1,u_1), ...]``
        and infers the block-banded sparsity exactly, reducing the
        total Jacobian nnz from O(T^2·n^2) (dense) to O(T·(n+m)·n)
        (sparse) so MUMPS can factor it.

        We deliberately use ``hessian_approximation = limited-memory``
        (L-BFGS) so we don't need a Lagrangian-Hessian callback.
        """
        ca = _import_casadi()
        import jax
        import jax.numpy as jnp

        T, n, m, td = problem.T, problem.n, problem.m, problem.theta_dim
        if td > 0:
            return make_failure_result(
                self.name, problem.name,
                f"JAX-callback path doesn't support theta_dim>0 (got {td}). "
                "Provide a casadi_builder for analytical problems with theta.",
            )

        # Per-stage callbacks. Each operates on z = vertcat(x_t, u_t).
        nxu = n + m
        theta_const = jnp.empty(0)
        x0_jnp = jnp.asarray(problem.x0)

        # Hold the wrappers in lists so CasADi's weak-ref Callback
        # bookkeeping doesn't reap them mid-solve.
        self._dyn_wrappers = []
        self._cost_wrappers = []
        self._ineq_wrappers = []

        # Per-stage dynamics callback: input (x, u), output next_x of dim n.
        def _make_dyn_fn(t_const):
            t_jax = jnp.int32(t_const)
            def fn(z):
                x_t = z[:n]
                u_t = z[n:n + m]
                return problem.dynamics(x_t, u_t, theta_const, t_jax)
            return fn

        # Per-stage cost callback: input (x, u), output 1-element [cost].
        def _make_cost_fn(t_const, has_u):
            t_jax = jnp.int32(t_const)
            if has_u:
                def fn(z):
                    x_t = z[:n]
                    u_t = z[n:n + m]
                    return jnp.array([problem.cost(x_t, u_t, theta_const, t_jax)])
            else:
                u_zero = jnp.zeros(m)
                def fn(z):
                    x_t = z[:n]
                    return jnp.array([problem.cost(x_t, u_zero, theta_const, t_jax)])
            return fn

        # Per-stage ineq callback: input (x, u), output (ineq_dim,) vector.
        def _make_ineq_fn(t_const, has_u):
            t_jax = jnp.int32(t_const)
            if has_u:
                def fn(z):
                    x_t = z[:n]
                    u_t = z[n:n + m]
                    return problem.inequalities(x_t, u_t, theta_const, t_jax)
            else:
                u_zero = jnp.zeros(m)
                def fn(z):
                    x_t = z[:n]
                    return problem.inequalities(x_t, u_zero, theta_const, t_jax)
            return fn

        # Symbolic decision vars (CasADi MX)
        X_sym = ca.MX.sym("X", n, T + 1)
        U_sym = ca.MX.sym("U", m, T)

        # Build defects + cost + ineq via per-stage callbacks.
        defects = [X_sym[:, 0] - ca.DM(np.asarray(problem.x0))]
        cost_terms = []
        ineq_terms = []

        for t in range(T):
            xu = ca.vertcat(X_sym[:, t], U_sym[:, t])
            wrap = _JaxToCasadiCallback(f"dyn_{t}", _make_dyn_fn(t), nxu, n)
            self._dyn_wrappers.append(wrap)
            next_x = wrap.cb()(xu)
            defects.append(X_sym[:, t + 1] - next_x)

            cwrap = _JaxToCasadiCallback(f"cost_{t}", _make_cost_fn(t, True), nxu, 1)
            self._cost_wrappers.append(cwrap)
            cost_terms.append(cwrap.cb()(xu))

            if problem.inequalities is not None and problem.ineq_dim > 0:
                iwrap = _JaxToCasadiCallback(f"ineq_{t}", _make_ineq_fn(t, True), nxu, problem.ineq_dim)
                self._ineq_wrappers.append(iwrap)
                ineq_terms.append(iwrap.cb()(xu))

        # Terminal stage: u is missing — use a 1-arg callback over x only.
        def _make_x_only(fn_const, t_const, out_dim):
            # fn_const expects z = vertcat(x). We make a thin n-input callback.
            t_jax = jnp.int32(t_const)
            return fn_const  # already takes z = x only when has_u=False
        cwrap_T = _JaxToCasadiCallback(
            f"cost_{T}", _make_cost_fn(T, False), n, 1,
        )
        self._cost_wrappers.append(cwrap_T)
        cost_terms.append(cwrap_T.cb()(X_sym[:, T]))
        if problem.inequalities is not None and problem.ineq_dim > 0:
            iwrap_T = _JaxToCasadiCallback(
                f"ineq_{T}", _make_ineq_fn(T, False), n, problem.ineq_dim,
            )
            self._ineq_wrappers.append(iwrap_T)
            ineq_terms.append(iwrap_T.cb()(X_sym[:, T]))

        f = sum(cost_terms[1:], cost_terms[0]) if cost_terms else ca.MX(0.0)
        g_eq = ca.vertcat(*defects)
        g_ineq = ca.vertcat(*ineq_terms) if ineq_terms else None

        n_eq = g_eq.numel()
        n_ineq = g_ineq.numel() if g_ineq is not None else 0
        if g_ineq is not None:
            g = ca.vertcat(g_eq, g_ineq)
        else:
            g = g_eq

        # Pack symbolic z = vec(X) + vec(U) to match _slice_iterate.
        z = ca.vertcat(ca.reshape(X_sym, -1, 1), ca.reshape(U_sym, -1, 1))

        lbg = np.concatenate([np.zeros(n_eq), -np.inf * np.ones(n_ineq)])
        ubg = np.concatenate([np.zeros(n_eq), np.zeros(n_ineq)])

        ipopt_opts = {
            "print_level": self.print_level,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "constr_viol_tol": self.tol,
            "hessian_approximation": "limited-memory",
            "sb": "yes",
        }
        if self.timeout_s is not None:
            ipopt_opts["max_wall_time"] = float(self.timeout_s)
        # Per-iter recorder (same shape as the pure-CasADi path —
        # captures the flat (X, U) decision vector and lam_g per
        # IPOPT outer iter for post-process history extraction).
        nx_total = z.numel()
        self._iter_recorder = _IterationRecorder(
            nx=nx_total, ng=n_eq + n_ineq, np_=0,
        )
        opts = {
            "print_time": False,
            "ipopt": ipopt_opts,
            "iteration_callback": self._iter_recorder.callback,
        }
        nlp = {"x": z, "f": f, "g": g}
        solver = ca.nlpsol("ipopt_jax_solver", "ipopt", nlp, opts)

        z0 = np.concatenate([
            np.asarray(problem.X_init).reshape(-1),
            np.asarray(problem.U_init).reshape(-1),
        ])

        # Warm-up: call each per-stage JAX function once (cheap, ~ms each)
        # so the JIT cache is populated. This dwarfs IPOPT's per-iter
        # cost in MJX-scale problems where each stage's JAX exec is
        # non-trivial.
        for w in self._dyn_wrappers + self._cost_wrappers + self._ineq_wrappers:
            try:
                _ = np.asarray(w._fn(np.zeros(w._in_dim)))
            except Exception:  # noqa: BLE001
                pass

        start = timer()
        sol = solver(x0=z0, lbg=lbg, ubg=ubg)
        solve_time_ms = 1e3 * (timer() - start)

        z_val = np.asarray(sol["x"]).reshape(-1)
        X, U, Theta = _slice_iterate(z_val, problem)
        # Multipliers. The JAX-callback NLP layout is:
        #   eq rows  = [init_defect (n,); dyn_defects (T*n,)]   -- no user eqs
        #   ineq rows = [stage 0 (ineq_dim,); ...; stage T (ineq_dim,)]
        #              when problem.inequalities is not None
        # so the splice into evaluate_problem's full stack is trivial
        # for eq (init + dyn already match) and one-to-one for ineq
        # (every stage contributes).
        lam_g_full = np.asarray(sol["lam_g"]).reshape(-1)
        lam_eq_compact = lam_g_full[:n_eq] if n_eq > 0 else np.zeros(0)
        lam_ineq_compact = lam_g_full[n_eq:n_eq + n_ineq] if n_ineq > 0 else np.zeros(0)
        # eq_t_present is False for all stages (no user eqs in this path);
        # ineq_t_present is True for every stage when inequalities are on.
        eq_t_present = np.zeros(T + 1, dtype=bool)
        ineq_t_present = (
            np.ones(T + 1, dtype=bool)
            if (problem.inequalities is not None and problem.ineq_dim > 0)
            else np.zeros(T + 1, dtype=bool)
        )
        multipliers_eq = _remap_ipopt_eq_lam(lam_eq_compact, problem, eq_t_present)
        multipliers_ineq = _remap_ipopt_ineq_lam(lam_ineq_compact, problem, ineq_t_present)

        stats = solver.stats()
        iters = int(stats.get("iter_count", -1))
        success = bool(stats.get("success", False))

        # Per-iter histories from the recorded iterates (post-process,
        # outside the timed window).
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
            X=X, U=U, Theta=Theta,
            iterations=iters,
            solve_time_ms=solve_time_ms,
            success=success,
            notes=stats.get("return_status", ""),
            multipliers_eq=multipliers_eq,
            multipliers_ineq=multipliers_ineq,
            iterates_xut=iterates_xut,
        )

    def solve(self, problem: ProblemSpec) -> SolverResult:
        if "casadi_builder" in problem.metadata:
            return self._solve_pure_casadi(problem)
        return self._solve_jax_callback(problem)


@register("ipopt")
def _factory(**kwargs) -> SolverAdapter:
    return IpoptCasadiAdapter(**kwargs)
