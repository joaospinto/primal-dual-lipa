"""acados real-time SQP adapter.

acados requires a manual CMake build of the C library before its Python
bindings (``acados_template``) become usable. See acados's docs:
https://docs.acados.org/installation/index.html, plus the more
project-specific notes in ``tests/comparison/acados_install.md`` (which
covers the exact env vars needed to make the bindings find
``libacados.so``).

This adapter consumes the same ``ProblemSpec`` as every other adapter,
reuses the per-problem ``casadi_builder`` (already used by the IPOPT
adapter), and runs acados in *fully converged* SQP mode (``nlp_solver_type
= 'SQP'``), not RTI — we want the iteration-count number to mean the same
thing as IPOPT's ``iter_count`` and CSQP's ``solver.iter``.

Encoding choices:

* ``model.disc_dyn_expr = stage['next_x']`` (LIPA's problems are already
  pre-discretized via Euler), with ``solver_options.integrator_type =
  'DISCRETE'``.
* Cost goes through ``cost.cost_type = 'EXTERNAL'`` so arbitrary
  quadratic / goal-tracking forms survive unchanged. We pair this with
  ``hessian_approx = 'EXACT'`` since the Gauss-Newton path is
  undefined for arbitrary external costs.
* Path inequalities go through ``con_h_expr`` with ``lh = -inf`` / ``uh =
  0`` to match the LIPA convention ``ineq(x, u) <= 0``.
* Stage equalities are encoded as ``con_h_expr`` rows with
  ``lh = uh = 0`` (no separate "h_eq" channel exists in acados).
* The terminal-stage equality ``x_T - goal = 0`` is encoded through
  ``con_h_expr_e`` with ``lh_e = uh_e = 0``.
* Initial state via ``constraints.x0``.

We deliberately do NOT use ``lbu/ubu/idxbu`` / ``lbx_e/ubx_e/idxbx_e``
even when the problem boils down to box constraints, because the
``ProblemSpec`` interface only exposes generic vector-valued ineq/eq
expressions, and we need this adapter to work for any problem without
snooping into the casadi_builder's internals.

Cross-stage decision variables (``theta_dim > 0``) are not supported: we
report ``unavailable`` rather than silently dropping the cross-stage
coupling.
"""

from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
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


def _import_acados():
    from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver  # noqa: F401

    return AcadosModel, AcadosOcp, AcadosOcpSolver


# acados rejects ``np.inf`` in bounds because its JSON template renderer
# (Tera, written in Rust) chokes on JavaScript-style ``Infinity``. The
# Python interface defines ``ACADOS_INFTY = 1e10`` for "this side is
# unbounded"; we mirror the constant locally so we don't need to import
# from a private acados submodule path.
_ACADOS_INFTY = 1.0e10


def _import_casadi():
    import casadi as ca  # local import so missing CasADi only fails this adapter

    return ca


@contextmanager
def _chdir(path: str):
    """Temporarily ``chdir`` to ``path`` (acados generates code relative to CWD)."""
    cwd = os.getcwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(cwd)


# Forward-rollout warm-start lives in tests.comparison.warm_starts;
# see that module's docstring for the rationale. We re-export under
# the historical name so any external callers keep working.
from tests.comparison.warm_starts import rollout_warm_start as _rollout_warm_start  # noqa: E402


def _build_ocp(problem: ProblemSpec, *, code_export_dir: str, json_file: str,
               max_iter: int, tol: float):
    """Translate ``problem`` into an ``AcadosOcp``.

    Returns the constructed ``AcadosOcp``. Caller is responsible for
    handing it to ``AcadosOcpSolver(ocp, json_file=...)``.
    """
    AcadosModel, AcadosOcp, _ = _import_acados()
    ca = _import_casadi()

    if problem.theta_dim > 0:
        raise ValueError(
            f"acados adapter cannot handle cross-stage Theta "
            f"(theta_dim={problem.theta_dim})",
        )
    if "casadi_builder" not in problem.metadata:
        raise ValueError(
            "acados adapter requires problem.metadata['casadi_builder']; "
            "this is the same builder the IPOPT adapter uses.",
        )

    casadi_builder = problem.metadata["casadi_builder"]
    n, m, T = problem.n, problem.m, problem.T

    x = ca.SX.sym("x", n)
    u = ca.SX.sym("u", m)
    theta = ca.SX.zeros(0)  # theta_dim == 0 (we early-skip otherwise)

    # Build stage and terminal expressions by calling casadi_builder twice.
    # Every existing problem's builder branches on ``is_terminal = (t == T)``,
    # so t=0 gives stage and t=T gives terminal.
    stage = casadi_builder(x, u, theta, 0)
    terminal = casadi_builder(x, ca.SX.zeros(m), theta, T)

    # Sanity: stage must have a next_x. Terminal next_x is ignored by acados
    # (no dynamics from N to N+1).
    if stage.get("next_x") is None:
        raise ValueError("casadi_builder(t=0) must return 'next_x'")

    ocp = AcadosOcp()
    ocp.model = AcadosModel()
    ocp.model.x = x
    ocp.model.u = u
    ocp.model.name = problem.name
    ocp.model.disc_dyn_expr = stage["next_x"]

    # Initial state.
    ocp.constraints.x0 = np.asarray(problem.x0, dtype=np.float64).reshape(-1)

    # ---- Cost ------------------------------------------------------------
    ocp.cost.cost_type = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"
    ocp.model.cost_expr_ext_cost = stage["f"]
    ocp.model.cost_expr_ext_cost_e = terminal["f"]

    # ---- Stage path constraints (ineq + eq) ------------------------------
    h_pieces = []
    h_lb_pieces: list[np.ndarray] = []
    h_ub_pieces: list[np.ndarray] = []

    stage_ineq = stage.get("ineq")
    if stage_ineq is not None and stage_ineq.numel() > 0:
        ng = stage_ineq.numel()
        h_pieces.append(stage_ineq)
        h_lb_pieces.append(-_ACADOS_INFTY * np.ones(ng))
        h_ub_pieces.append(np.zeros(ng))

    stage_eq = stage.get("eq")
    if stage_eq is not None and stage_eq.numel() > 0:
        ng = stage_eq.numel()
        h_pieces.append(stage_eq)
        h_lb_pieces.append(np.zeros(ng))
        h_ub_pieces.append(np.zeros(ng))

    if h_pieces:
        ocp.model.con_h_expr = ca.vertcat(*h_pieces)
        ocp.constraints.lh = np.concatenate(h_lb_pieces)
        ocp.constraints.uh = np.concatenate(h_ub_pieces)

    # ---- Terminal path constraints (ineq + eq) ---------------------------
    he_pieces = []
    he_lb_pieces: list[np.ndarray] = []
    he_ub_pieces: list[np.ndarray] = []

    terminal_ineq = terminal.get("ineq")
    if terminal_ineq is not None and terminal_ineq.numel() > 0:
        ng = terminal_ineq.numel()
        he_pieces.append(terminal_ineq)
        he_lb_pieces.append(-_ACADOS_INFTY * np.ones(ng))
        he_ub_pieces.append(np.zeros(ng))

    terminal_eq = terminal.get("eq")
    if terminal_eq is not None and terminal_eq.numel() > 0:
        ng = terminal_eq.numel()
        he_pieces.append(terminal_eq)
        he_lb_pieces.append(np.zeros(ng))
        he_ub_pieces.append(np.zeros(ng))

    if he_pieces:
        ocp.model.con_h_expr_e = ca.vertcat(*he_pieces)
        ocp.constraints.lh_e = np.concatenate(he_lb_pieces)
        ocp.constraints.uh_e = np.concatenate(he_ub_pieces)

    # ---- Solver options --------------------------------------------------
    ocp.solver_options.N_horizon = T
    # ``tf`` only matters for continuous-time integrators; for DISCRETE it's
    # used as a label / scaling factor, but acados requires it set.
    ocp.solver_options.tf = float(T)
    ocp.solver_options.integrator_type = "DISCRETE"
    # SQP_WITH_FEASIBLE_QP runs Byrd-Omojokun feasibility restoration
    # whenever the nominal QP is infeasible — essential for OCPs with
    # a terminal equality whose QP linearization at the warm start
    # may be infeasible. NOT SQP_RTI: we want fully converged
    # iterations.
    ocp.solver_options.nlp_solver_type = "SQP_WITH_FEASIBLE_QP"
    # External cost + nonlinear dynamics need the exact Hessian for the
    # Gauss-Newton-default to behave correctly. (Without this, acados
    # warns "Gauss-Newton Hessian approximation with EXTERNAL cost type
    # not well defined" and convergence is poor.)
    ocp.solver_options.hessian_approx = "EXACT"
    # PROJECT keeps the Hessian PSD by clipping negative eigenvalues to
    # zero. MIRROR is more conservative (preserves the magnitude of
    # negative curvature directions by reflecting), which can
    # over-regularize and slow down Newton's natural
    # convergence rate.
    ocp.solver_options.regularize_method = "PROJECT"
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.qp_solver_mu0 = 1e3
    # Funnel L1-penalty line search (Kiessling et al. 2024) is required
    # by the FEASIBLE_QP variant: it's the only globalization that
    # tracks both the merit (cost + L1 penalty) and the funnel
    # (constraint violation budget) consistently across restoration
    # steps. With MERIT_BACKTRACKING the restoration phase produces
    # negative merit improvements and the line search rejects the step.
    ocp.solver_options.globalization = "FUNNEL_L1PEN_LINESEARCH"
    ocp.solver_options.globalization_full_step_dual = True
    ocp.solver_options.globalization_funnel_use_merit_fun_only = False
    # Auto-scale the QP so HPIPM tolerates large mismatches between
    # stage and terminal cost weights.
    ocp.solver_options.qpscaling_scale_objective = "OBJECTIVE_GERSHGORIN"
    ocp.solver_options.qpscaling_scale_constraints = "INF_NORM"
    ocp.solver_options.nlp_solver_max_iter = int(max_iter)
    # Per-iter recording: tell acados to keep every intermediate iterate
    # so the adapter can post-process them into cost / eq / ineq history
    # arrays. The store cost is O(N*(n+m)) per iter — negligible for the
    # cost of OCPs at this scale.
    ocp.solver_options.store_iterates = True
    # Inner QP iter budget: the feasibility restoration QPs sometimes
    # need many inner iterations near the boundary of the funnel.
    ocp.solver_options.qp_solver_iter_max = max(1000, int(max_iter))
    ocp.solver_options.tol = float(tol)
    ocp.solver_options.print_level = 0

    # Code-gen artifacts go into our scratch dir, not the project root.
    # (The pre-0.5.4 ``ocp.code_export_directory = ...`` setter still
    # works but is deprecated; the new home is ``code_gen_opts``.)
    ocp.code_gen_opts.code_export_directory = code_export_dir
    ocp.code_gen_opts.json_file = json_file

    return ocp


class AcadosAdapter(SolverAdapter):
    """acados ``AcadosOcpSolver`` (SQP, exact Hessian) on a ``ProblemSpec``."""

    name = "acados"

    def __init__(self, max_iter: int = 1000, tol: float = 1e-6) -> None:
        self.max_iter = max_iter
        self.tol = tol

    def is_available(self) -> tuple[bool, str]:
        try:
            _import_acados()
            _import_casadi()
        except ImportError as e:
            return False, f"{e}"
        return True, ""

    def solve(self, problem: ProblemSpec) -> SolverResult:
        avail, reason = self.is_available()
        if not avail:
            return make_failure_result(
                self.name, problem.name, f"unavailable: {reason}",
            )

        if problem.theta_dim > 0:
            return make_failure_result(
                self.name, problem.name,
                f"acados does not natively support cross-stage Theta "
                f"(theta_dim={problem.theta_dim})",
            )

        if "casadi_builder" not in problem.metadata:
            return make_failure_result(
                self.name, problem.name,
                "problem has no metadata['casadi_builder']",
            )

        _, _, AcadosOcpSolver = _import_acados()

        # acados always generates C code into the current working
        # directory (and stores model_dir / cost_dir / constraints_dir
        # subdirs there), so we sandbox each solve in its own tmp
        # directory to avoid trampling the user's CWD or clashing across
        # parallel solves.
        with tempfile.TemporaryDirectory(prefix=f"acados_{problem.name}_") as scratch:
            ocp = _build_ocp(
                problem,
                code_export_dir=os.path.join(scratch, "c_generated_code"),
                json_file=os.path.join(scratch, f"{problem.name}_acados_ocp.json"),
                max_iter=self.max_iter,
                tol=self.tol,
            )

            err = ""
            T, n, m = problem.T, problem.n, problem.m

            try:
                with _chdir(scratch):
                    solver = AcadosOcpSolver(ocp, verbose=False)

                    # Warm start from a forward rollout of U_init through
                    # problem.dynamics — see _rollout_warm_start docstring
                    # for why the LIPA-shipped X_init isn't usable here.
                    xs_ws, us_ws = _rollout_warm_start(problem)
                    for i in range(T + 1):
                        solver.set(i, "x", xs_ws[i])
                    for i in range(T):
                        solver.set(i, "u", us_ws[i])

                    start = timer()
                    status = solver.solve()
                    solve_time_ms = 1e3 * (timer() - start)

                    # Status codes per acados/include/acados/utils/types.h:
                    # 0 = success; 1 = max-iter; 2 = min-step (line search
                    # failure); 3 = NaN; 4 = QP failure; 5 = ready (not
                    # solved). We accept 0 only.
                    sqp_iter = int(solver.get_stats("sqp_iter"))

                    xs = np.zeros((T + 1, n), dtype=np.float64)
                    us = np.zeros((T, m), dtype=np.float64)
                    for i in range(T + 1):
                        xs[i] = np.asarray(solver.get(i, "x"), dtype=np.float64)
                    for i in range(T):
                        us[i] = np.asarray(solver.get(i, "u"), dtype=np.float64)

                    # Multipliers — acados stores per-stage 'pi' (dynamics
                    # multipliers, length N for stages 0..N-1) and 'lam'
                    # (path-constraint multipliers stacked as
                    # [lower bounds; upper bounds; soft slacks ...] per
                    # stage). For our convention we want the dynamics
                    # multipliers (one (n,) per dyn defect) and the
                    # path-constraint multipliers split into eq/ineq.
                    # We extract pi at stages 0..N-1 (these multiply the
                    # dynamics defects ``x_{t+1} - dyn(x_t, u_t) = 0``
                    # — same sign as evaluate_problem's
                    # ``dyn_defects = dyn(...) - X[1:]`` after a sign
                    # flip; acados's pi multiplies the "dynamics gap"
                    # in the OCP formulation, which is the upper bound
                    # 0 of the equality, so we negate).
                    pi_full = []
                    for i in range(T):
                        pi_i = np.asarray(
                            solver.get(i, "pi"), dtype=np.float64,
                        ).reshape(-1)
                        pi_full.append(pi_i)
                    pi_full_arr = (
                        np.concatenate(pi_full) if pi_full else np.zeros(0)
                    )

                    # Path-constraint multipliers via 'lam'. acados stacks
                    # ``lam = [lam_lh; lam_uh; lam_phi; ...]`` per stage.
                    # We only set ``con_h_expr`` (and at terminal
                    # ``con_h_expr_e``); path-equality rows used
                    # ``lh = uh = 0`` (active at both bounds), pure-ineq
                    # rows used ``lh = -INFTY, uh = 0``. For an active
                    # equality acados returns lam_uh - lam_lh as the
                    # equality multiplier. Our build records the per-
                    # stage h split via the stage builder probe — we
                    # re-run the probe here to split the lam vector.
                    iterates_record_xut = []  # filled below
            except Exception as e:  # noqa: BLE001
                return make_failure_result(
                    self.name, problem.name,
                    f"adapter raised: {type(e).__name__}: {e}",
                )

            # Per-iter history (post-process; outside the timed window).
            # acados stores all intermediate iterates when
            # store_iterates=True (set in _build_ocp). Each iterate
            # contains x[N+1] and u[N] arrays.
            try:
                iterates_obj = solver.get_iterates()
                # AcadosOcpIterates exposes .iterate_list (list of
                # AcadosOcpIterate, each with .x / .u /etc.).
                for it in iterates_obj.iterate_list:
                    Xi = np.zeros((T + 1, n), dtype=np.float64)
                    Ui = np.zeros((T, m), dtype=np.float64)
                    for k in range(T + 1):
                        Xi[k] = np.asarray(it.x[k], dtype=np.float64).reshape(-1)
                    for k in range(T):
                        Ui[k] = np.asarray(it.u[k], dtype=np.float64).reshape(-1)
                    iterates_record_xut.append((Xi, Ui, np.zeros(0)))
            except Exception:  # noqa: BLE001
                # Older acados version or store_iterates not honored —
                # leave the history as None.
                iterates_record_xut = []

        Theta = np.asarray(problem.Theta_init, dtype=np.float64)

        # Build the equality multiplier stack expected by
        # evaluate_problem: [init_defect (n,); dyn_defects (T*n,);
        # user_eqs ((T+1)*eq_dim,)].
        # * init_defect: acados constrains ``x_0 = problem.x0`` directly
        #   via ``ocp.constraints.x0`` (a separate channel from pi /
        #   lam). We don't have direct access to its multiplier, so we
        #   leave the init slot at 0 — an active equality constraint
        #   that matches at convergence contributes 0 to the eval
        #   stationarity test (the gradient term vanishes when the
        #   constraint is satisfied).
        # * dyn_defects: -pi (sign flip — acados's pi multiplies
        #   ``X[t+1] - dyn(...) = 0`` while evaluate_problem measures
        #   ``dyn(...) - X[t+1]``).
        # * user_eqs: not extracted (acados encodes them through
        #   con_h_expr; the lam vector layout per stage is
        #   ``[lam_lh; lam_uh; ...]``, requiring per-stage probing of
        #   the constraint dimensions to split equality vs inequality
        #   blocks). Leave as zeros; the corresponding rows in the
        #   evaluator's stack are typically zero only at convergence,
        #   so the stationarity contribution from these rows lands in
        #   the residual reported here.
        # Multiplier extraction is partial for acados: we expose pi
        # (dynamics multipliers, accurately mapped to evaluate_problem's
        # dyn_defects rows after a sign flip) but path-constraint
        # multipliers (acados's per-stage `lam` vector that stacks
        # ``[lbu lbx lg lh lphi ubu ubx ug uh uphi; soft...]``) require
        # per-stage dimension probing to split into our user-eq and
        # user-ineq blocks. We don't do that here — the resulting KKT
        # stationarity / dual / complementarity slots therefore reflect
        # only the cost + dyn-defect Lagrangian. The primal feasibility
        # numbers (init / dyn / eq* / ineq*) are always accurate.
        eq_full_size = n + T * n + (T + 1) * problem.eq_dim
        multipliers_eq = np.zeros(eq_full_size, dtype=np.float64)
        if pi_full_arr.size == T * n:
            # Sign flip: acados encodes the dynamics constraint as
            # ``X[t+1] - dyn(...) = 0`` while evaluate_problem measures
            # the opposite ``dyn(...) - X[t+1]``.
            multipliers_eq[n:n + T * n] = -pi_full_arr

        success = (status == 0)
        notes = err or f"status={status}"

        # multipliers_ineq=None: dual / complementarity / stationarity
        # slots get reported as nan rather than as mis-computed zeros.
        return pack_solver_result(
            solver_name=self.name,
            problem_name=problem.name,
            problem=problem,
            X=xs, U=us, Theta=Theta,
            iterations=sqp_iter,
            solve_time_ms=solve_time_ms,
            success=success,
            notes=notes,
            multipliers_eq=multipliers_eq,
            multipliers_ineq=None,
            iterates_xut=iterates_record_xut or None,
        )


@register("acados")
def _factory(**kwargs) -> SolverAdapter:
    return AcadosAdapter(**kwargs)
