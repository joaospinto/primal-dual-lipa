"""Google Trajax adapter — JAX-native iLQR / AL-iLQR.

Trajax (https://github.com/google/trajax — *not* the unrelated PyPI
``trajax`` package) is a JAX-native trajectory optimization library
from Google. It exposes two primary entry points relevant to this
comparison:

* ``trajax.optimizers.ilqr`` — unconstrained iLQR (single-shooting).
* ``trajax.optimizers.constrained_ilqr`` — Augmented-Lagrangian iLQR
  with explicit support for stage-wise equality + inequality
  constraints (single-shooting).

Both are pre-jitted (``PjitFunction``) and operate on JAX-traceable
``cost(x, u, t)`` / ``dynamics(x, u, t)`` callables — there is no
Python-callback overhead like ipopt-mjx / fatrop-mjx pay through
CasADi. This makes Trajax the closest non-LIPA solver in the
comparison set on per-iteration cost for JAX-native (MJX) problems.

## Algorithmic positioning vs LIPA

LIPA is a primal-dual interior-point Riccati-based solver. Trajax's
constrained_ilqr is an outer Augmented-Lagrangian wrapper around
iLQR's DDP-style Bellman recursion. Both end up solving a sequence of
LQR-shaped sub-problems, but:

* LIPA carries a barrier ``μ`` and inequality slack variables; AL-iLQR
  carries a penalty ``ρ`` and dual-multiplier estimates that get
  promoted at each outer iter.
* LIPA's KKT system is solved exactly per Newton step via a structured
  Riccati factorization; iLQR linearizes the rollout and runs a
  backwards-pass with line search on the controls.
* SQP-family solvers (CSQP / Aligator / acados-EXACT) can struggle on
  non-convex multi-basin problems because their filter / merit
  function does not always jump between disconnected feasible
  components from a bad warm start. AL-iLQR's penalty merit *should*
  in principle handle this — that is a hypothesis worth testing here.

## Single-shooting vs multi-shooting

Trajax is **single-shooting**: the only decision variable is the
control sequence ``U`` of shape ``(T, m)``; the state trajectory is
recomputed by rolling out ``dynamics`` from ``x0``. Our
``ProblemSpec`` is multi-shooting (state + control both decision
variables, dynamics defects penalized), so this adapter:

1. Discards ``problem.X_init`` and uses only ``problem.U_init`` as the
   warm start.
2. Computes the state trajectory inside trajax by single-shooting
   rollout, so dynamics defects are zero by construction.
3. Passes ``cost`` / ``equalities`` / ``inequalities`` through
   directly — the problem-spec stage functions already follow the
   trajax convention of evaluating uniformly at every ``t`` and using
   ``jnp.where(t == T, terminal_branch, stage_branch)`` to special-case
   the terminal step (so the output dimensions are constant across
   ``t``, which trajax's vmap requires).

A multi-shooting variant of iLQR (e.g. Crocoddyl's FDDP) is not what
Trajax exposes; option 1 above matches Trajax's design intent.

## Theta handling

Trajax stage signatures are ``(x, u, t)`` — there is no cross-stage
``theta`` argument. Following the same policy as CSQP / Aligator /
acados / fatrop, we **skip cleanly** when ``problem.theta_dim > 0``
with a ``notes="trajax does not natively support cross-stage Theta
(theta_dim=N)"`` result. LIPA, IPOPT, and SIP are the only solvers
in this comparison that ingest theta natively.

## Constraint dimensions

Trajax requires the per-stage equality / inequality outputs to have
**constant dimension** across all ``t`` (the inner ``vmap`` over
the time axis demands it). Our ``ProblemSpec`` stage functions
already use the ``jnp.where(t == T, terminal_branch, stage_branch)``
pattern (e.g. a terminal goal constraint at ``t == T`` and
``zeros(n)`` otherwise — both shape ``(n,)``), so this is satisfied
without further massaging.

## Solver choice

We use ``constrained_ilqr`` whenever the problem has any equality or
inequality constraints, and fall back to plain ``ilqr`` for fully
unconstrained cost-only problems. The same wall-clock budget applies
to either.

## Hardcoded settings

* ``maxiter_al`` (constrained outer AL loop): we use a relatively
  large default (50) so the penalty parameter ramps high enough to
  drive the constraint violation below ``constraints_threshold``.
* ``maxiter_ilqr`` (per-AL-iter inner iLQR cap): scaled from the
  ``max_iter`` constructor arg.
* ``constraints_threshold``: we tie it to the standard ``tol``
  constructor arg.
* ``make_psd=True``, ``psd_delta=1e-6``: keep Hessians PSD during
  the backwards pass; the trajax test suite's defaults for
  constrained_ilqr.

## Warm-up call

We do a single non-timed solve before the timed solve so JAX trace
+ compile costs aren't included in ``solve_time_ms``. Same convention
as sip / sip-mjx / fatrop-mjx adapters. The warm-up uses ``maxiter=1``
to keep its wall time minimal while still triggering compilation of
the entire trace.

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


def _import_trajax():
    import trajax  # noqa: F401

    return trajax


def _import_jax():
    import jax  # noqa: F401
    import jax.numpy as jnp  # noqa: F401

    # Trajax (and our problem specs) operate in float64 by default; align
    # JAX so the per-iter linear solves don't silently fall back to float32.
    jax.config.update("jax_enable_x64", True)
    return jax, jnp


def _build_trajax_callables(problem: ProblemSpec):
    """Adapt LIPA-style ``(x, u, theta, t) -> ...`` stage functions to
    trajax's ``(x, u, t) -> ...`` signature.

    Theta is captured by closure from ``problem.Theta_init`` (we only
    ever call this path when ``theta_dim == 0`` — see ``solve()`` —
    but plumbing it through here keeps the closure types clean if a
    future caller relaxes that guard).

    Returns ``(tj_cost, tj_dyn, tj_eq, tj_ineq)`` where the last two
    are ``None`` when the problem has no equality / inequality
    constraints respectively.
    """
    _, jnp = _import_jax()

    theta = jnp.asarray(problem.Theta_init)
    cost_fn = problem.cost
    dyn_fn = problem.dynamics
    eq_fn = problem.equalities
    ineq_fn = problem.inequalities

    def tj_cost(x, u, t):
        return cost_fn(x, u, theta, t)

    def tj_dyn(x, u, t):
        return dyn_fn(x, u, theta, t)

    tj_eq = None
    if eq_fn is not None and problem.eq_dim > 0:
        def tj_eq(x, u, t):
            return eq_fn(x, u, theta, t)

    tj_ineq = None
    if ineq_fn is not None and problem.ineq_dim > 0:
        def tj_ineq(x, u, t):
            return ineq_fn(x, u, theta, t)

    return tj_cost, tj_dyn, tj_eq, tj_ineq


class TrajaxAdapter(SolverAdapter):
    """Google Trajax (iLQR / AL-iLQR) adapter."""

    name = "trajax"

    def __init__(
        self,
        max_iter: int = 1000,
        tol: float = 1e-6,
        # AL outer-loop iteration cap (only used by ``constrained_ilqr``).
        # 50 outer AL updates with a 10x penalty ramp lets penalty go up
        # to 1e50 — well past anything we'd ever need for the analytical
        # problems. The runner-passed ``max_iter`` caps the inner iLQR
        # budget per AL iter.
        maxiter_al: int = 50,
        # PSD floor for the quadratized cost Hessian. The trajax test
        # suite defaults to 0 for the unconstrained ilqr and uses
        # ``make_psd=True`` for constrained_ilqr; we mirror that here.
        psd_delta: float = 1e-6,
        # Penalty schedule.
        penalty_init: float = 1.0,
        penalty_update_rate: float = 10.0,
        # iLQR line-search lower bound; trajax default 5e-5 is fine for
        # most problems but MJX problems sometimes need a smaller floor.
        alpha_min: float = 5e-5,
    ) -> None:
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.maxiter_al = int(maxiter_al)
        self.psd_delta = float(psd_delta)
        self.penalty_init = float(penalty_init)
        self.penalty_update_rate = float(penalty_update_rate)
        self.alpha_min = float(alpha_min)

    def is_available(self) -> tuple[bool, str]:
        try:
            _import_trajax()
        except ImportError as e:
            return False, f"{e}"
        try:
            from trajax.optimizers import ilqr, constrained_ilqr  # noqa: F401
        except ImportError as e:
            return False, (
                f"trajax is installed but missing ilqr / constrained_ilqr "
                f"entry points: {e}"
            )
        return True, ""

    def solve(self, problem: ProblemSpec) -> SolverResult:  # noqa: PLR0915
        avail, reason = self.is_available()
        if not avail:
            return make_failure_result(
                self.name, problem.name, f"unavailable: {reason}",
            )

        if problem.theta_dim > 0:
            return make_failure_result(
                self.name, problem.name,
                f"trajax does not natively support cross-stage Theta "
                f"(theta_dim={problem.theta_dim})",
            )

        jax, jnp = _import_jax()
        from trajax.optimizers import constrained_ilqr, ilqr

        T, n, m = problem.T, problem.n, problem.m

        tj_cost, tj_dyn, tj_eq, tj_ineq = _build_trajax_callables(problem)

        x0 = jnp.asarray(problem.x0, dtype=jnp.float64).reshape(n)
        U0 = jnp.asarray(problem.U_init, dtype=jnp.float64).reshape(T, m)

        has_constraints = (tj_eq is not None) or (tj_ineq is not None)

        # --- Warm-up call (excluded from solve_time_ms) ----------------------
        # Trajax's ilqr / constrained_ilqr are PjitFunctions — the first
        # invocation incurs the JAX trace + compile cost. Use maxiter=1 to
        # keep the warm-up cheap while still compiling the full trace.
        #
        # constrained_ilqr internally calls ``np.max(np.abs(...))`` on the
        # equality / inequality residuals; that reduction has no identity
        # for a zero-sized array, so we MUST supply at least one row in
        # each constraint stub even when the problem has none of that
        # kind. A constant-zero ``(1,)`` row is benign — its dual stays
        # at zero through the AL updates and it doesn't perturb the
        # iLQR sub-problem.
        try:
            if has_constraints:
                eq_for_call = tj_eq if tj_eq is not None else (
                    lambda x, u, t: jnp.zeros(1, dtype=jnp.float64)
                )
                ineq_for_call = tj_ineq if tj_ineq is not None else (
                    lambda x, u, t: -jnp.ones(1, dtype=jnp.float64)
                )
                _warm = constrained_ilqr(
                    tj_cost, tj_dyn, x0, U0,
                    equality_constraint=eq_for_call,
                    inequality_constraint=ineq_for_call,
                    maxiter_al=1,
                    maxiter_ilqr=1,
                    constraints_threshold=self.tol,
                    penalty_init=self.penalty_init,
                    penalty_update_rate=self.penalty_update_rate,
                    make_psd=True,
                    psd_delta=self.psd_delta,
                    alpha_min=self.alpha_min,
                )
                jax.block_until_ready(_warm[0])
            else:
                _warm = ilqr(
                    tj_cost, tj_dyn, x0, U0,
                    maxiter=1,
                    grad_norm_threshold=self.tol,
                    make_psd=True,
                    psd_delta=self.psd_delta,
                    alpha_min=self.alpha_min,
                )
                jax.block_until_ready(_warm[0])
        except Exception:  # noqa: BLE001
            pass

        # --- Timed solve ----------------------------------------------------
        notes_pieces: list[str] = []
        start = timer()
        try:
            if has_constraints:
                eq_for_call = tj_eq if tj_eq is not None else (
                    lambda x, u, t: jnp.zeros(1, dtype=jnp.float64)
                )
                ineq_for_call = tj_ineq if tj_ineq is not None else (
                    lambda x, u, t: -jnp.ones(1, dtype=jnp.float64)
                )
                sol = constrained_ilqr(
                    tj_cost, tj_dyn, x0, U0,
                    equality_constraint=eq_for_call,
                    inequality_constraint=ineq_for_call,
                    maxiter_al=self.maxiter_al,
                    maxiter_ilqr=self.max_iter,
                    constraints_threshold=self.tol,
                    penalty_init=self.penalty_init,
                    penalty_update_rate=self.penalty_update_rate,
                    make_psd=True,
                    psd_delta=self.psd_delta,
                    alpha_min=self.alpha_min,
                )
                jax.block_until_ready(sol[0])
                # constrained_ilqr returns:
                # (X, U, dual_eq, dual_ineq, penalty, eq_violation,
                #  ineq_violation, max_violation, obj, gradient,
                #  iter_ilqr, iter_al)
                X_sol = np.asarray(sol[0], dtype=np.float64)
                U_sol = np.asarray(sol[1], dtype=np.float64)
                # dual_eq / dual_ineq are per-stage AL multiplier
                # estimates; their exact shape depends on the
                # per-stage constraint dimensions (which trajax
                # requires to be uniform — we use shape-(1,) zero
                # stubs for the missing-constraint side, see warmup).
                trajax_dual_eq = np.asarray(sol[2], dtype=np.float64)
                trajax_dual_ineq = np.asarray(sol[3], dtype=np.float64)
                max_viol = float(sol[7])
                iters_ilqr = int(sol[10])
                iters_al = int(sol[11])
                iterations = iters_ilqr  # report cumulative inner iLQR count
                notes_pieces.append(
                    f"al_iters={iters_al} ilqr_iters_total={iters_ilqr} "
                    f"max_viol={max_viol:.2e}"
                )
                success = max_viol <= self.tol
            else:
                sol = ilqr(
                    tj_cost, tj_dyn, x0, U0,
                    maxiter=self.max_iter,
                    grad_norm_threshold=self.tol,
                    make_psd=True,
                    psd_delta=self.psd_delta,
                    alpha_min=self.alpha_min,
                )
                jax.block_until_ready(sol[0])
                # ilqr returns:
                # (X, U, obj, gradient, adjoints, lqr, iteration)
                X_sol = np.asarray(sol[0], dtype=np.float64)
                U_sol = np.asarray(sol[1], dtype=np.float64)
                trajax_dual_eq = None
                trajax_dual_ineq = None
                grad_norm = float(np.linalg.norm(np.asarray(sol[3])))
                iterations = int(sol[6])
                notes_pieces.append(
                    f"ilqr_iters={iterations} grad_norm={grad_norm:.2e}"
                )
                # Plain iLQR has no constraint violation; success is purely
                # a gradient-norm check. The trajax convergence criterion
                # is identical, but it may also exit on the line-search
                # alpha floor / iteration cap before grad_norm hits tol —
                # report based on grad norm directly.
                success = grad_norm <= self.tol or iterations < self.max_iter
        except Exception as e:  # noqa: BLE001
            solve_time_ms = 1e3 * (timer() - start)
            return make_failure_result(
                self.name, problem.name,
                f"{type(e).__name__}: {str(e).splitlines()[0]}",
                solve_time_ms=solve_time_ms,
            )
        solve_time_ms = 1e3 * (timer() - start)

        Theta = np.asarray(problem.Theta_init, dtype=np.float64)
        # Re-evaluate cost / violations via the canonical evaluator so the
        # number we report is computed identically to every other solver
        # (including padding the U axis to length T+1 with a zero stage).
        # Map trajax's per-stage AL duals into evaluate_problem's stacks
        # when present. Trajax's single-shooting layout means there are
        # no dynamics defects (rolled out internally), so multipliers_eq
        # only reflects the user-eq rows. Init-defect is also implicit
        # (X[0] = problem.x0 by construction).
        T = problem.T
        n = problem.n
        eq_dim = problem.eq_dim
        ineq_dim = problem.ineq_dim
        eq_full_size = n + T * n + (T + 1) * eq_dim
        ineq_full_size = (T + 1) * ineq_dim
        multipliers_eq = np.zeros(eq_full_size, dtype=np.float64) if eq_full_size > 0 else None
        multipliers_ineq = np.zeros(ineq_full_size, dtype=np.float64) if ineq_full_size > 0 else None
        if trajax_dual_eq is not None and eq_dim > 0 and multipliers_eq is not None:
            try:
                # trajax shape (T+1, eq_dim) — splice into the user-eq tail.
                de = np.asarray(trajax_dual_eq, dtype=np.float64).reshape(-1)
                if de.size == (T + 1) * eq_dim:
                    multipliers_eq[n + T * n:] = de
            except Exception:  # noqa: BLE001
                pass
        if trajax_dual_ineq is not None and ineq_dim > 0 and multipliers_ineq is not None:
            try:
                di = np.asarray(trajax_dual_ineq, dtype=np.float64).reshape(-1)
                if di.size == (T + 1) * ineq_dim:
                    multipliers_ineq[:] = di
            except Exception:  # noqa: BLE001
                pass

        # If we have ineq multipliers but no eq multipliers, materialise
        # a zero-filled eq stack of the right shape so evaluate_problem
        # can compute the partial KKT (stationarity needs both).
        multipliers_eq_for_kkt = multipliers_eq
        if multipliers_eq_for_kkt is None and multipliers_ineq is not None:
            multipliers_eq_for_kkt = np.zeros(eq_full_size)

        # Per-iter history: trajax's PjitFunction makes per-iter
        # iterates inaccessible without forking the library (the
        # while_loop is fully JIT-compiled). Per the project brief,
        # we leave history as None — the convergence plot falls back
        # to endpoint markers for trajax.
        return pack_solver_result(
            solver_name=self.name,
            problem_name=problem.name,
            problem=problem,
            X=X_sol, U=U_sol, Theta=Theta,
            iterations=iterations,
            solve_time_ms=solve_time_ms,
            success=bool(success),
            notes="; ".join(notes_pieces),
            multipliers_eq=multipliers_eq_for_kkt,
            multipliers_ineq=multipliers_ineq,
        )


@register("trajax")
def _factory(**kwargs) -> SolverAdapter:
    return TrajaxAdapter(**kwargs)
