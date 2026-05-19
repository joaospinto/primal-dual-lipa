"""fatrop-on-MJX adapter — fatrop's CasADi structure-aware backend with
per-stage CasADi callbacks that share a single JIT'd JAX function each.

This is the MJX counterpart to ``fatrop.py``. The analytical adapter
goes straight through ``casadi_builder`` (a pure-symbolic CasADi
representation), but the MJX problems only ship JAX functions, so we
have to take the same per-stage callback route ``ipopt_mjx_sparse.py``
takes for IPOPT.

Architecture (see also the docstrings in ``fatrop.py`` and
``ipopt_mjx_sparse.py``):

* CasADi-Opti is the entry point so we can hand fatrop the explicit
  multi-shooting block layout via ``structure_detection='manual'``.
  Variables are stage-interleaved ``[x_0, u_0, x_1, u_1, ..., x_T]``.
* Each stage gets one CasADi ``Callback`` for dynamics, one for cost,
  and (if the problem has inequalities) one for inequalities. Each
  callback delegates to a single shared ``jax.jit``'d function with
  the stage index ``t`` passed as a runtime ``jnp.int32`` argument; see
  ``tests.comparison.casadi_jax_callback.PerStageJaxCallback`` for the
  full rationale of this O(1)-JIT-trace amortization.
* Per-stage path constraints (initial-state defect at k=0, plus user
  inequalities anywhere they occur) go into ``ng[k]``. The dynamics
  defect ``x_{k+1} - dyn(x_k, u_k) = 0`` is added separately as a
  ``subject_to`` equality and is what fatrop's Riccati recursion
  exploits — it does NOT count toward ``ng``.

Hessian (the critical bit, vs ipopt_mjx_sparse):

fatrop *requires* an exact Lagrangian Hessian — the OCP-structured
Riccati linear solve at each iter inverts the KKT system that includes
``∇²L``. So we hand ``PerStageJaxCallback`` an ``adj_hess_jit_fn``
(a ``jax.jacfwd(jax.jacrev(seed.dot(f)))`` traced once per
stage-shape); that triggers the shared module's reverse-mode plumbing
(``get_reverse`` -> ``_RevCb`` -> ``get_jacobian`` -> ``_RevJacCb``)
so CasADi can ask for second derivatives without falling through to
``has_derivative()`` assertions during fatrop's ``init`` step. This
gives us the same O(1)-JIT amortization for second derivatives as we
get for first derivatives.

Warm start (per the project instructions):

1. First try a *dynamics rollout* — forward-propagate ``U_init`` through
   ``problem.dynamics`` from ``problem.x0`` and use the resulting
   trajectory as ``X_ws``. This is dynamics-feasible by construction so
   fatrop only sees inequality / terminal-cost residuals on iter 0.
2. If the rollout produces a non-finite value, fall back to the
   LIPA-shipped ``problem.X_init``.

Skip policy:

* ``theta_dim > 0``: fatrop's OCP layout has no notion of cross-stage
  decision variables.

Iteration count quirk:

When ``opti.solve`` raises (which happens on max_iter and on
``LocalInfeasibility``), CasADi's outer ``stats['iter_count']`` and
fatrop's nested ``stats['fatrop']['iterations_count']`` both get reset
to 0. We fall back to ``stats['fatrop']['eval_hess_count']`` as a
proxy: fatrop calls ``eval_hess`` exactly once per IPM iteration, which
matches the ``it`` column in fatrop's print output.
"""

from __future__ import annotations

from timeit import default_timer as timer

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
    import casadi as ca

    return ca


def _fatrop_plugin_available(ca) -> tuple[bool, str]:
    """Return (True, '') if ``nlpsol(..., 'fatrop', ...)`` works, else (False, reason).

    Mirrors ``fatrop.py``'s probe so we surface a clear "plugin not
    found" message instead of a deep CasADi traceback.
    """
    try:
        nlp = {"x": ca.SX.sym("x"), "f": ca.SX(1.0), "g": ca.SX(0.0)}
        ca.nlpsol(
            "probe",
            "fatrop",
            nlp,
            {
                "structure_detection": "manual",
                "nx": [1, 0],
                "nu": [0, 0],
                "ng": [0, 0],
                "N": 1,
                "fatrop": {"print_level": 0},
            },
        )
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        if "Plugin" in msg and ("could not be loaded" in msg or "not found" in msg):
            return False, f"CasADi fatrop plugin missing: {msg.splitlines()[0]}"
        return True, ""
    return True, ""


# Per-stage CasADi callback that shares a single JIT'd JAX function across
# stages lives in tests.comparison.casadi_jax_callback. See that module's
# docstring for the rationale. fatrop needs the reverse-mode (adjoint
# Hessian) plumbing because of its OCP-structured Riccati linear solve;
# callers pass ``adj_hess_jit_fn=<a jax.jacfwd(jax.jacrev(...))>`` to
# enable it.
from tests.comparison.casadi_jax_callback import (
    PerStageJaxCallback as _PerStageJaxCallback,
)  # noqa: E402


from tests.comparison.warm_starts import (  # noqa: E402
    linspace_to_extracted_goal_warm_start,
)


def _initial_iterate(problem: ProblemSpec) -> tuple[np.ndarray, np.ndarray]:
    """Pick (X_ws, U_ws).

    Uses the same linspace-toward-extracted-goal heuristic as the
    casadi adapter (``fatrop.py``): when the problem has an affine
    terminal-equality goal, override a degenerate ``tile(x0)`` X_init
    with ``linspace(x0, goal)`` so fatrop's filter line search has
    something to make progress against. Otherwise pass ``problem.X_init``
    through unchanged.
    """
    return linspace_to_extracted_goal_warm_start(problem)


class FatropMjxAdapter(SolverAdapter):
    """fatrop on MJX problems via per-stage JAX-callback CasADi-Opti.

    Skips cleanly (returns a SolverResult with ``success=False``) when
    the problem isn't an MJX problem (use ``fatrop`` instead) or has
    cross-stage Theta variables (fatrop OCP layout doesn't support it).
    """

    name = "fatrop-jax"

    def __init__(
        self,
        max_iter: int = 1000,
        tol: float = 1e-6,
        fatrop_extra_options: dict | None = None,
    ) -> None:
        self.max_iter = max_iter
        self.tol = tol
        # Free-form dict merged into ``fatrop_opts`` (passed under the
        # ``"fatrop"`` plugin key) so tuning sweeps can override any
        # registered fatrop option (mu_init, delta_w0, max_soc,
        # acceptable_tol, …) without changing the adapter signature.
        # Plumbed in via ``--solver-kwargs-json '{"fatrop_extra_options":
        # {...}}'``.
        self.fatrop_extra_options = dict(fatrop_extra_options or {})

    def is_available(self) -> tuple[bool, str]:
        try:
            ca = _import_casadi()
        except ImportError as e:
            return False, f"casadi not importable: {e}"
        ok, reason = _fatrop_plugin_available(ca)
        if not ok:
            return False, reason
        try:
            import jax  # noqa: F401
        except ImportError as e:
            return False, f"jax not importable: {e}"
        return True, ""

    def solve(self, problem: ProblemSpec) -> SolverResult:  # noqa: PLR0912, PLR0915
        from tests.comparison.problem_spec import effective_solver_tol

        # Stash on self so the inner ``_solve`` can read it.
        self._effective_tol = effective_solver_tol(problem, self.tol)
        avail, reason = self.is_available()
        if not avail:
            return make_failure_result(
                self.name,
                problem.name,
                f"unavailable: {reason}",
            )

        if problem.theta_dim > 0:
            return make_failure_result(
                self.name,
                problem.name,
                f"fatrop does not natively support cross-stage Theta "
                f"(theta_dim={problem.theta_dim})",
            )

        return self._solve(problem)

    def _solve(self, problem: ProblemSpec) -> SolverResult:  # noqa: PLR0912, PLR0915
        ca = _import_casadi()
        import jax
        import jax.numpy as jnp

        T, n, m = problem.T, problem.n, problem.m
        K = T + 1

        # ---------- Shared JIT'd JAX functions ----------
        # Each takes ``t`` as a runtime jnp.int32 so JAX traces ONCE
        # and reuses the compiled code for every stage.
        theta_const = jnp.empty(0)

        @jax.jit
        def dyn_eval(z, t):
            x = z[:n]
            u = z[n : n + m]
            return problem.dynamics(x, u, theta_const, t)

        @jax.jit
        def dyn_jac(z, t):
            return jax.jacrev(lambda zz: dyn_eval(zz, t))(z)

        @jax.jit
        def dyn_adj_hess(z, seed, t):
            # ∂/∂z (seed^T J(z)) = ∂²(seed^T f)/∂z² — symmetric in_dim x in_dim.
            return jax.jacfwd(jax.jacrev(lambda zz: jnp.dot(seed, dyn_eval(zz, t))))(z)

        @jax.jit
        def cost_eval_xu(z, t):
            x = z[:n]
            u = z[n : n + m]
            return jnp.array([problem.cost(x, u, theta_const, t)])

        @jax.jit
        def cost_jac_xu(z, t):
            return jax.jacrev(lambda zz: cost_eval_xu(zz, t))(z)

        @jax.jit
        def cost_adj_hess_xu(z, seed, t):
            return jax.jacfwd(
                jax.jacrev(lambda zz: jnp.dot(seed, cost_eval_xu(zz, t)))
            )(z)

        @jax.jit
        def cost_eval_x(x, t):
            u_zero = jnp.zeros(m, dtype=x.dtype)
            return jnp.array([problem.cost(x, u_zero, theta_const, t)])

        @jax.jit
        def cost_jac_x(x, t):
            return jax.jacrev(lambda xx: cost_eval_x(xx, t))(x)

        @jax.jit
        def cost_adj_hess_x(x, seed, t):
            return jax.jacfwd(jax.jacrev(lambda xx: jnp.dot(seed, cost_eval_x(xx, t))))(
                x
            )

        has_ineq = problem.inequalities is not None and problem.ineq_dim > 0
        if has_ineq:

            @jax.jit
            def ineq_eval_xu(z, t):
                x = z[:n]
                u = z[n : n + m]
                return problem.inequalities(x, u, theta_const, t)

            @jax.jit
            def ineq_jac_xu(z, t):
                return jax.jacrev(lambda zz: ineq_eval_xu(zz, t))(z)

            @jax.jit
            def ineq_adj_hess_xu(z, seed, t):
                return jax.jacfwd(
                    jax.jacrev(lambda zz: jnp.dot(seed, ineq_eval_xu(zz, t)))
                )(z)

            @jax.jit
            def ineq_eval_x(x, t):
                u_zero = jnp.zeros(m, dtype=x.dtype)
                return problem.inequalities(x, u_zero, theta_const, t)

            @jax.jit
            def ineq_jac_x(x, t):
                return jax.jacrev(lambda xx: ineq_eval_x(xx, t))(x)

            @jax.jit
            def ineq_adj_hess_x(x, seed, t):
                return jax.jacfwd(
                    jax.jacrev(lambda xx: jnp.dot(seed, ineq_eval_x(xx, t)))
                )(x)

        has_user_eq = problem.equalities is not None and problem.eq_dim > 0
        if has_user_eq:

            @jax.jit
            def eq_eval_xu(z, t):
                x = z[:n]
                u = z[n : n + m]
                return problem.equalities(x, u, theta_const, t)

            @jax.jit
            def eq_jac_xu(z, t):
                return jax.jacrev(lambda zz: eq_eval_xu(zz, t))(z)

            @jax.jit
            def eq_adj_hess_xu(z, seed, t):
                return jax.jacfwd(
                    jax.jacrev(lambda zz: jnp.dot(seed, eq_eval_xu(zz, t)))
                )(z)

            @jax.jit
            def eq_eval_x(x, t):
                u_zero = jnp.zeros(m, dtype=x.dtype)
                return problem.equalities(x, u_zero, theta_const, t)

            @jax.jit
            def eq_jac_x(x, t):
                return jax.jacrev(lambda xx: eq_eval_x(xx, t))(x)

            @jax.jit
            def eq_adj_hess_x(x, seed, t):
                return jax.jacfwd(
                    jax.jacrev(lambda xx: jnp.dot(seed, eq_eval_x(xx, t)))
                )(x)

            # Per-stage presence probe: equalities that are identically
            # zero (e.g. ``jnp.where(t == T, ..., zeros)`` patterns)
            # would otherwise produce rank-deficient Jacobian rows that
            # cause "Not_Enough_Degrees_Of_Freedom" failures.
            _eq_rng = np.random.default_rng(0)
            eq_t_present = np.zeros(T + 1, dtype=bool)
            for tt in range(T + 1):
                xu_a = jnp.asarray(_eq_rng.normal(size=n + m), dtype=jnp.float64)
                xu_b = jnp.asarray(_eq_rng.normal(size=n + m), dtype=jnp.float64)
                if tt == T:
                    va = np.asarray(eq_eval_x(xu_a[:n], jnp.int32(tt)))
                    vb = np.asarray(eq_eval_x(xu_b[:n], jnp.int32(tt)))
                else:
                    va = np.asarray(eq_eval_xu(xu_a, jnp.int32(tt)))
                    vb = np.asarray(eq_eval_xu(xu_b, jnp.int32(tt)))
                eq_t_present[tt] = bool(np.any(va != 0.0) or np.any(vb != 0.0))
        else:
            eq_t_present = np.zeros(T + 1, dtype=bool)

        # ---------- Per-stage callbacks (pinned on self.* for lifetime) ----------
        self._dyn_cbs: list = []
        self._cost_cbs: list = []
        self._eq_cbs: list = []
        self._ineq_cbs: list = []

        for t in range(T):
            self._dyn_cbs.append(
                _PerStageJaxCallback(
                    f"dyn_{t}",
                    t,
                    n + m,
                    n,
                    eval_jit_fn=dyn_eval,
                    jac_jit_fn=dyn_jac,
                    adj_hess_jit_fn=dyn_adj_hess,
                )
            )
            self._cost_cbs.append(
                _PerStageJaxCallback(
                    f"cost_{t}",
                    t,
                    n + m,
                    1,
                    eval_jit_fn=cost_eval_xu,
                    jac_jit_fn=cost_jac_xu,
                    adj_hess_jit_fn=cost_adj_hess_xu,
                )
            )
            if has_user_eq and eq_t_present[t]:
                self._eq_cbs.append(
                    _PerStageJaxCallback(
                        f"eq_{t}",
                        t,
                        n + m,
                        problem.eq_dim,
                        eval_jit_fn=eq_eval_xu,
                        jac_jit_fn=eq_jac_xu,
                        adj_hess_jit_fn=eq_adj_hess_xu,
                    )
                )
            else:
                self._eq_cbs.append(None)
            if has_ineq:
                self._ineq_cbs.append(
                    _PerStageJaxCallback(
                        f"ineq_{t}",
                        t,
                        n + m,
                        problem.ineq_dim,
                        eval_jit_fn=ineq_eval_xu,
                        jac_jit_fn=ineq_jac_xu,
                        adj_hess_jit_fn=ineq_adj_hess_xu,
                    )
                )

        # Terminal stage (t = T): cost, user-eq, and ineq take only x.
        cost_T_cb = _PerStageJaxCallback(
            f"cost_{T}",
            T,
            n,
            1,
            eval_jit_fn=cost_eval_x,
            jac_jit_fn=cost_jac_x,
            adj_hess_jit_fn=cost_adj_hess_x,
        )
        self._cost_cbs.append(cost_T_cb)
        eq_T_cb = None
        if has_user_eq and eq_t_present[T]:
            eq_T_cb = _PerStageJaxCallback(
                f"eq_{T}",
                T,
                n,
                problem.eq_dim,
                eval_jit_fn=eq_eval_x,
                jac_jit_fn=eq_jac_x,
                adj_hess_jit_fn=eq_adj_hess_x,
            )
            self._eq_cbs.append(eq_T_cb)
        elif has_user_eq:
            self._eq_cbs.append(None)
        if has_ineq:
            ineq_T_cb = _PerStageJaxCallback(
                f"ineq_{T}",
                T,
                n,
                problem.ineq_dim,
                eval_jit_fn=ineq_eval_x,
                jac_jit_fn=ineq_jac_x,
                adj_hess_jit_fn=ineq_adj_hess_x,
            )
            self._ineq_cbs.append(ineq_T_cb)

        # ---------- Warm up the JITs (one-shot per shared fn) ----------
        x0_jnp = jnp.asarray(problem.x0)
        u0_jnp = jnp.zeros(m)
        z_xu = jnp.concatenate([x0_jnp, u0_jnp])
        t0_j = jnp.int32(0)
        seed_n = jnp.zeros(n)  # adjoint seed for dynamics output (size n)
        seed_1 = jnp.zeros(1)  # adjoint seed for scalar cost
        seed_ineq = jnp.zeros(problem.ineq_dim) if has_ineq else None
        _ = jax.block_until_ready(dyn_eval(z_xu, t0_j))
        _ = jax.block_until_ready(dyn_jac(z_xu, t0_j))
        _ = jax.block_until_ready(dyn_adj_hess(z_xu, seed_n, t0_j))
        _ = jax.block_until_ready(cost_eval_xu(z_xu, t0_j))
        _ = jax.block_until_ready(cost_jac_xu(z_xu, t0_j))
        _ = jax.block_until_ready(cost_adj_hess_xu(z_xu, seed_1, t0_j))
        _ = jax.block_until_ready(cost_eval_x(x0_jnp, t0_j))
        _ = jax.block_until_ready(cost_jac_x(x0_jnp, t0_j))
        _ = jax.block_until_ready(cost_adj_hess_x(x0_jnp, seed_1, t0_j))
        if has_ineq:
            _ = jax.block_until_ready(ineq_eval_xu(z_xu, t0_j))
            _ = jax.block_until_ready(ineq_jac_xu(z_xu, t0_j))
            _ = jax.block_until_ready(ineq_adj_hess_xu(z_xu, seed_ineq, t0_j))
            _ = jax.block_until_ready(ineq_eval_x(x0_jnp, t0_j))
            _ = jax.block_until_ready(ineq_jac_x(x0_jnp, t0_j))
            _ = jax.block_until_ready(ineq_adj_hess_x(x0_jnp, seed_ineq, t0_j))
        if has_user_eq:
            seed_eq = jnp.zeros(problem.eq_dim)
            _ = jax.block_until_ready(eq_eval_xu(z_xu, t0_j))
            _ = jax.block_until_ready(eq_jac_xu(z_xu, t0_j))
            _ = jax.block_until_ready(eq_adj_hess_xu(z_xu, seed_eq, t0_j))
            _ = jax.block_until_ready(eq_eval_x(x0_jnp, t0_j))
            _ = jax.block_until_ready(eq_jac_x(x0_jnp, t0_j))
            _ = jax.block_until_ready(eq_adj_hess_x(x0_jnp, seed_eq, t0_j))

        # ---------- Build the Opti problem with stage-interleaved layout ----------
        opti = ca.Opti()
        x_vars: list = []
        u_vars: list = []
        for k in range(K):
            x_vars.append(opti.variable(n))
            if k < T:
                u_vars.append(opti.variable(m))
        u_vars.append(ca.MX.zeros(m, 1))  # dummy zero control at t=T

        nx_list = [n] * K
        nu_list = [m] * T + [0]
        ng_list: list[int] = []
        # Track per-block constraint structure so we can splice
        # opti.lam_g back into evaluate_problem's eq/ineq stacks.
        constraint_blocks: list = []

        # Initial-state defect lives in k=0 path constraints.
        init_eq = x_vars[0] - ca.DM(np.asarray(problem.x0, dtype=np.float64))

        f_total = ca.MX(0.0)

        for k in range(K):
            is_terminal = k == T
            x_k = x_vars[k]

            if not is_terminal:
                xu_k = ca.vertcat(x_k, u_vars[k])
                # Cost.
                f_total = f_total + self._cost_cbs[k].cb()(xu_k)
                # Dynamics defect: x_{k+1} = dyn(x_k, u_k).
                next_x = self._dyn_cbs[k].cb()(xu_k)
                opti.subject_to(x_vars[k + 1] == next_x)
                constraint_blocks.append({"kind": "dyn", "stage": k, "size": n})
            else:
                # Terminal cost takes x only.
                f_total = f_total + cost_T_cb.cb()(x_k)

            # Path constraints: equality (init defect at k=0 + optional
            # user equalities at present stages) + inequality.
            eq_rows: list = []
            ineq_rows: list = []
            if k == 0:
                eq_rows.append(init_eq)
            if has_user_eq and eq_t_present[k]:
                if not is_terminal:
                    user_eq_k = self._eq_cbs[k].cb()(ca.vertcat(x_k, u_vars[k]))
                else:
                    user_eq_k = eq_T_cb.cb()(x_k)
                eq_rows.append(user_eq_k)

            if has_ineq:
                if not is_terminal:
                    ineq_k = self._ineq_cbs[k].cb()(ca.vertcat(x_k, u_vars[k]))
                else:
                    ineq_k = ineq_T_cb.cb()(x_k)
                ineq_rows.append(ineq_k)

            n_eq = sum(int(r.numel()) for r in eq_rows)
            n_ineq = sum(int(r.numel()) for r in ineq_rows)

            if eq_rows:
                opti.subject_to(ca.vertcat(*eq_rows) == ca.MX.zeros(n_eq, 1))
                if k == 0:
                    constraint_blocks.append({"kind": "init", "stage": 0, "size": n})
                if has_user_eq and eq_t_present[k]:
                    constraint_blocks.append(
                        {"kind": "user_eq", "stage": k, "size": problem.eq_dim}
                    )
            if ineq_rows:
                opti.subject_to(ca.vertcat(*ineq_rows) <= ca.MX.zeros(n_ineq, 1))
                constraint_blocks.append(
                    {
                        "kind": "ineq",
                        "stage": k,
                        "size": int(n_ineq),
                    }
                )

            ng_list.append(n_eq + n_ineq)

        opti.minimize(f_total)

        fatrop_opts = {
            "tol": float(self._effective_tol),
            "max_iter": int(self.max_iter),
            "print_level": int(getattr(self, "fatrop_print_level", 0)),
            "mu_init": 1e-1,
        }
        # Per-problem metadata (fatrop_mjx_settings) layered BEFORE the
        # CLI extra_options so CLI overrides still win.
        fatrop_opts.update(problem.metadata.get("fatrop_mjx_settings", {}))
        fatrop_opts.update(self.fatrop_extra_options)
        plugin_opts = {
            "structure_detection": "manual",
            "nx": nx_list,
            "nu": nu_list,
            "ng": ng_list,
            "N": T,
            # CRITICAL: ``expand`` would attempt to convert the MX graph
            # (which contains opaque CasADi Callbacks) into SX, which
            # cannot represent Callbacks. Leave it false.
            "expand": False,
            "fatrop": fatrop_opts,
            "print_time": False,
        }
        opti.solver("fatrop", plugin_opts)

        # ---------- Warm start ----------
        xs_ws, us_ws = _initial_iterate(problem)
        for k in range(K):
            opti.set_initial(x_vars[k], xs_ws[k])
        for k in range(T):
            opti.set_initial(u_vars[k], us_ws[k])

        # Per-iter recorder via opti.callback. fatrop calls back every
        # outer IPM iteration; we read live values via opti.debug.value
        # and append (X, U) snapshots for post-process history.
        iter_xu: list = []

        def _iter_cb(_i):
            try:
                Xi = np.zeros((T + 1, n), dtype=np.float64)
                Ui = np.zeros((T, m), dtype=np.float64)
                for k in range(T + 1):
                    Xi[k] = np.asarray(
                        opti.debug.value(x_vars[k]),
                        dtype=np.float64,
                    ).reshape(-1)
                for k in range(T):
                    Ui[k] = np.asarray(
                        opti.debug.value(u_vars[k]),
                        dtype=np.float64,
                    ).reshape(-1)
                iter_xu.append((Xi, Ui, np.zeros(0)))
            except Exception:  # noqa: BLE001
                pass

        opti.callback(_iter_cb)

        notes_pieces: list[str] = []
        xs = np.zeros((T + 1, n), dtype=np.float64)
        us = np.zeros((T, m), dtype=np.float64)
        lam_g_final = np.zeros(0)
        start = timer()
        try:
            sol = opti.solve()
            solve_time_ms = 1e3 * (timer() - start)
            stats = opti.stats()
            for k in range(T + 1):
                xs[k] = np.asarray(sol.value(x_vars[k]), dtype=np.float64).reshape(-1)
            for k in range(T):
                us[k] = np.asarray(sol.value(u_vars[k]), dtype=np.float64).reshape(-1)
            try:
                lam_g_final = np.asarray(
                    sol.value(opti.lam_g), dtype=np.float64
                ).reshape(-1)
            except Exception:  # noqa: BLE001
                lam_g_final = np.zeros(0)
        except Exception as e:  # noqa: BLE001
            solve_time_ms = 1e3 * (timer() - start)
            try:
                stats = opti.stats()
            except Exception:  # noqa: BLE001
                stats = {"iter_count": -1, "success": False, "return_status": "error"}
            try:
                for k in range(T + 1):
                    xs[k] = np.asarray(
                        opti.debug.value(x_vars[k]),
                        dtype=np.float64,
                    ).reshape(-1)
                for k in range(T):
                    us[k] = np.asarray(
                        opti.debug.value(u_vars[k]),
                        dtype=np.float64,
                    ).reshape(-1)
            except Exception:  # noqa: BLE001
                pass
            try:
                lam_g_final = np.asarray(
                    opti.debug.value(opti.lam_g),
                    dtype=np.float64,
                ).reshape(-1)
            except Exception:  # noqa: BLE001
                lam_g_final = np.zeros(0)
            # Keep just the first interesting line of the exception —
            # CasADi nests every layer of context, so the full message
            # is hundreds of chars of internal stack frames. The most
            # useful bits (return status, "max iter reached", etc.)
            # are usually in the deepest line.
            err_lines = [line.strip() for line in str(e).split("\n") if line.strip()]
            err_text = err_lines[-1] if err_lines else type(e).__name__
            # If fatrop failed cleanly with "return_status is 'N'",
            # capture only that — it's the actionable signal.
            if "return_status is" in err_text:
                err_text = err_text.split("return_status is", 1)[1].strip(" .")
                err_text = f"return_status={err_text}"
            notes_pieces.append(f"{type(e).__name__}: {err_text[:200]}")

        Theta = np.asarray(problem.Theta_init, dtype=np.float64)

        # Multiplier remap (mirrors fatrop.py's _remap_fatrop_lam_g but
        # inlined here since fatrop_mjx doesn't import that helper).
        eq_full_size = n + T * n + (T + 1) * problem.eq_dim
        ineq_full_size = (T + 1) * problem.ineq_dim if has_ineq else 0
        multipliers_eq = np.zeros(eq_full_size, dtype=np.float64)
        multipliers_ineq = (
            np.zeros(ineq_full_size, dtype=np.float64)
            if ineq_full_size > 0
            else np.zeros(0)
        )
        if lam_g_final.size > 0:
            cursor = 0
            for blk in constraint_blocks:
                sz = blk["size"]
                if cursor + sz > lam_g_final.size:
                    break
                slice_lam = lam_g_final[cursor : cursor + sz]
                cursor += sz
                kind = blk["kind"]
                stage = blk["stage"]
                if kind == "init":
                    multipliers_eq[:n] = slice_lam
                elif kind == "dyn":
                    # Sign flip — fatrop encodes X[t+1] - dyn(...) = 0.
                    multipliers_eq[n + stage * n : n + (stage + 1) * n] = -slice_lam
                elif kind == "user_eq":
                    # Slot after init (n) + dyn (T*n) blocks.
                    base = n + T * n + stage * problem.eq_dim
                    multipliers_eq[base : base + sz] = slice_lam
                elif kind == "ineq" and ineq_full_size > 0:
                    multipliers_ineq[
                        stage * problem.ineq_dim : stage * problem.ineq_dim + sz
                    ] = slice_lam

        # Iteration count: CasADi's ``iter_count`` and fatrop's
        # ``iterations_count`` both report 0 when ``opti.solve`` raises
        # (fatrop hit max_iter or LocalInfeasibility) — they're only
        # populated on a clean exit. As a workaround, use
        # ``stats['fatrop']['eval_hess_count']`` (Hessian evaluations)
        # as a proxy: fatrop calls ``eval_hess`` exactly once per IPM
        # iteration. This matches the printed ``it`` column count.
        iters = -1
        if isinstance(stats.get("fatrop"), dict):
            iters = int(stats["fatrop"].get("iterations_count", 0))
            if iters <= 0:
                iters = int(stats["fatrop"].get("eval_hess_count", -1))
        if iters <= 0:
            iters = int(stats.get("iter_count", -1))
        success = bool(stats.get("success", False))
        return_status = stats.get("return_status", "")
        notes = "; ".join([*notes_pieces, str(return_status)]).strip("; ")

        return pack_solver_result(
            solver_name=self.name,
            problem_name=problem.name,
            problem=problem,
            X=xs,
            U=us,
            Theta=Theta,
            iterations=iters,
            solve_time_ms=solve_time_ms,
            success=success,
            notes=notes,
            multipliers_eq=multipliers_eq,
            multipliers_ineq=multipliers_ineq if ineq_full_size > 0 else None,
            iterates_xut=iter_xu or None,
        )


@register("fatrop-jax")
def _factory(**kwargs) -> SolverAdapter:
    return FatropMjxAdapter(**kwargs)
