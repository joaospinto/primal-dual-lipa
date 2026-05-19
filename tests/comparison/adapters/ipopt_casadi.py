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
    Theta_sx = ca.SX.sym(
        "Theta", max(td, 1)
    )  # CasADi can't build a 0-length sym; we slice it off

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
        out[n : n + T * n] = -lam_eq_compact[n : n + T * n]
    # user eqs (per-stage; only stages where eq was present get filled)
    cursor = n + T * n
    src = n + T * n
    if eq_dim > 0:
        for t in range(T + 1):
            dst = cursor + t * eq_dim
            if eq_t_present[t]:
                out[dst : dst + eq_dim] = lam_eq_compact[src : src + eq_dim]
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
            out[dst : dst + ineq_dim] = lam_ineq_compact[src : src + ineq_dim]
            src += ineq_dim
        # else leave as zeros
    return out


# Warm-start helpers live in tests.comparison.warm_starts; see that
# module's docstring for the rationale.
from tests.comparison.warm_starts import rollout_warm_start as _rollout_warm_start  # noqa: E402
from tests.comparison.warm_starts import (  # noqa: E402
    linspace_to_extracted_goal_warm_start as _auto_warm_start,
)


def _slice_iterate(
    z_val: np.ndarray, problem: ProblemSpec
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n, m, T, td = problem.n, problem.m, problem.T, problem.theta_dim
    nx = n * (T + 1)
    nu = m * T
    X = z_val[:nx].reshape(T + 1, n)
    U = z_val[nx : nx + nu].reshape(T, m)
    Theta = z_val[nx + nu : nx + nu + td] if td > 0 else np.zeros(0)
    return X, U, Theta


class IpoptCasadiAdapter(SolverAdapter):
    """IPOPT through CasADi for any ``ProblemSpec`` exposing a CasADi builder.

    The ``ProblemSpec`` is expected to have ``metadata["casadi_builder"]``
    when going through the pure-CasADi route (analytical problems), or
    ``metadata["jax_nlp"]`` for the JAX-callback route (MJX problems).
    """

    name = "ipopt-casadi"

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
            nx=nx_total,
            ng=n_eq + n_ineq,
            np_=0,
        )
        ipopt_opts = {
            "print_level": self.print_level,
            "max_iter": self.max_iter,
            "tol": self._effective_tol,
            "constr_viol_tol": self._effective_tol,
            "acceptable_iter": 0,
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
        # Per-problem metadata override layer (ipopt_settings), applied
        # AFTER the constructor defaults and BEFORE self.ipopt_extra_options
        # so a CLI experiment can still shadow a problem-baked tuning.
        ipopt_opts.update(problem.metadata.get("ipopt_settings", {}))
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
        z0 = np.concatenate(
            [
                X_ws.reshape(-1),
                U_ws.reshape(-1),
                np.asarray(problem.Theta_init).reshape(-1),
            ]
        )

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
        lam_ineq_compact = (
            lam_g_full[n_eq : n_eq + n_ineq] if n_ineq > 0 else np.zeros(0)
        )
        # IPOPT's sign convention matches LIPA's (L = f + λ^T c + z^T g
        # with g <= 0, lam_g >= 0 at active upper bound).
        multipliers_eq = _remap_ipopt_eq_lam(lam_eq_compact, problem, eq_t_present)
        multipliers_ineq = _remap_ipopt_ineq_lam(
            lam_ineq_compact, problem, ineq_t_present
        )

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
            X=X,
            U=U,
            Theta=Theta,
            iterations=iters,
            solve_time_ms=solve_time_ms,
            success=success,
            notes=stats.get("return_status", ""),
            multipliers_eq=multipliers_eq,
            multipliers_ineq=multipliers_ineq,
            iterates_xut=iterates_xut,
        )

    def solve(self, problem: ProblemSpec) -> SolverResult:
        from tests.comparison.problem_spec import effective_solver_tol

        # Per-problem ``success_tol`` (e.g. 1e-3 for MJX) overrides the
        # CLI default for this solve, so the comparison stays uniform.
        self._effective_tol = effective_solver_tol(problem, self.tol)
        if "casadi_builder" in problem.metadata:
            return self._solve_pure_casadi(problem)
        return make_failure_result(
            self.name,
            problem.name,
            "ipopt-casadi requires problem.metadata['casadi_builder']; "
            "for JAX-callback problems (MJX), use the 'ipopt-jax' adapter "
            "(ipopt_mjx_sparse), which uses per-stage exact Lagrangian "
            "Hessians instead of L-BFGS.",
        )


@register("ipopt-casadi")
def _factory(**kwargs) -> SolverAdapter:
    return IpoptCasadiAdapter(**kwargs)
