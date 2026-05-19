"""Adapter that runs the in-tree LIPA solver and returns a uniform ``SolverResult``."""

from __future__ import annotations

from timeit import default_timer as timer
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from primal_dual_lipa.optimizers import solve as lipa_solve
from primal_dual_lipa.types import SolverSettings, Variables

from tests.comparison.adapters import register
from tests.comparison.adapters.base import SolverAdapter
from tests.comparison.problem_spec import (
    ProblemSpec,
    SolverResult,
    pack_solver_result,
)


def _empty_inequalities(x, u, theta, t):  # noqa: ARG001
    return jnp.empty(0)


def _empty_equalities(x, u, theta, t):  # noqa: ARG001
    return jnp.empty(0)


class LipaAdapter(SolverAdapter):
    """Thin wrapper around ``primal_dual_lipa.optimizers.solve``."""

    name = "lipa"

    def __init__(
        self,
        settings: Optional[SolverSettings] = None,
        max_iter: int = 500,
        tol: float = 1e-6,
    ) -> None:
        # Translate generic CLI knobs into a SolverSettings if no explicit
        # settings object was given. The mapping is intentionally loose:
        # we square the requested KKT tolerance because LIPA's threshold is
        # on the squared residual, not its norm.
        if settings is None:
            settings = SolverSettings(
                max_iterations=max_iter,
                residual_sq_threshold=tol * tol,
                print_logs=False,
            )
        self.settings = settings

    def solve(self, problem: ProblemSpec) -> SolverResult:
        T = problem.T
        eq_dim = problem.eq_dim
        ineq_dim = problem.ineq_dim

        # Per-problem override: use ``metadata["lipa_settings"]`` if the
        # problem ships its own SolverSettings.
        settings = problem.metadata.get("lipa_settings", self.settings)

        # Auto-enable parallel LQR + parallel line search on GPU. Many of
        # the problem-shipped settings were copied verbatim from the
        # in-tree unit tests, which were written for CPU. With sequential
        # LQR LIPA spends most of its iter time waiting on serial scans
        # — flipping these on costs nothing on CPU and yields ~10–25x on
        # GPU. Mirrors the auto-enable in tests/mpc_examples/offline_solver.
        try:
            on_gpu = any(d.platform == "gpu" for d in jax.devices())
        except Exception:  # noqa: BLE001
            on_gpu = False
        if on_gpu and not settings.use_parallel_lqr:
            import dataclasses as _dc
            settings = _dc.replace(
                settings,
                use_parallel_lqr=True,
                num_parallel_line_search_steps=max(
                    settings.num_parallel_line_search_steps, 8,
                ),
            )

        # Optional two-phase warm start: an initial soft-penalty solve
        # produces an iterate that the main constrained solve then
        # refines. Mirrors the scheme used by
        # tests/mpc_examples/offline_solver.run_lipa_offline. Useful
        # for problems where the IPM otherwise parks at a degenerate
        # iterate (e.g. quaternion sign-flip singularities).
        warmup_cost = problem.metadata.get("lipa_warmup_cost")
        warmup_settings = problem.metadata.get("lipa_warmup_settings", settings)

        eq_fn = problem.equalities if problem.equalities is not None else _empty_equalities
        ineq_fn = problem.inequalities if problem.inequalities is not None else _empty_inequalities

        # Build initial Variables. LIPA reserves Y_dyn for dynamics
        # multipliers, Y_eq for general-equality multipliers, S/Z for the
        # primal-dual interior-point slack/multiplier pair.
        X0 = jnp.asarray(problem.X_init)
        U0 = jnp.asarray(problem.U_init)
        Theta0 = jnp.asarray(problem.Theta_init)
        S0 = jnp.zeros((T + 1, ineq_dim), dtype=X0.dtype)
        Y_dyn0 = jnp.zeros_like(X0)
        Y_eq0 = jnp.zeros((T + 1, eq_dim), dtype=X0.dtype)
        Z0 = jnp.zeros((T + 1, ineq_dim), dtype=X0.dtype)
        vars_in = Variables(X=X0, U=U0, S=S0, Y_dyn=Y_dyn0, Y_eq=Y_eq0, Z=Z0, Theta=Theta0)

        x0 = jnp.asarray(problem.x0)

        warmup_iters = 0
        warmup_time_ms = 0.0
        if warmup_cost is not None:
            # Phase 1: soft-penalty cost, no inequalities. We need to
            # build a vars_in with empty S/Z arrays since LIPA's KKT
            # builder shape-checks them against the inequality output.
            S0_empty = jnp.zeros((T + 1, 0), dtype=X0.dtype)
            Z0_empty = jnp.zeros((T + 1, 0), dtype=X0.dtype)
            vars_warmup = Variables(
                X=X0, U=U0, S=S0_empty, Y_dyn=Y_dyn0, Y_eq=Y_eq0,
                Z=Z0_empty, Theta=Theta0,
            )
            warmup_out, w_iter, _ = lipa_solve(
                vars_in=vars_warmup,
                x0=x0,
                cost=warmup_cost,
                dynamics=problem.dynamics,
                equalities=eq_fn,
                inequalities=_empty_inequalities,
                settings=warmup_settings,
            )
            jax.block_until_ready(warmup_out.X)

            # Time a real Phase 1 run (the JIT cache is warm now).
            start = timer()
            warmup_out, w_iter, _ = lipa_solve(
                vars_in=vars_warmup,
                x0=x0,
                cost=warmup_cost,
                dynamics=problem.dynamics,
                equalities=eq_fn,
                inequalities=_empty_inequalities,
                settings=warmup_settings,
            )
            jax.block_until_ready(warmup_out.X)
            warmup_time_ms = 1e3 * (timer() - start)
            warmup_iters = int(w_iter)

            # Hand the warm-started iterate to Phase 2 (now with the
            # full ineq-dim S/Z slots).
            vars_in = Variables(
                X=warmup_out.X, U=warmup_out.U,
                S=jnp.zeros_like(S0), Y_dyn=warmup_out.Y_dyn,
                Y_eq=warmup_out.Y_eq, Z=jnp.zeros_like(Z0),
                Theta=warmup_out.Theta,
            )

        # Warm-up call so JIT compile time is excluded from the Phase 2
        # measurement. We do this UNCONDITIONALLY — even when Phase 1 ran
        # — because Phase 2 uses a different cost / settings / `vars_in`
        # combination than Phase 1, so it has its own JIT cache key. The
        # earlier "skip warmup if Phase 1 done" optimization was wrong:
        # it left Phase 2's JIT compile inside the timed window. The
        # cost is one extra full Phase 2 solve, but `solve_time_ms` is
        # then a clean per-iter-x-iter count number.
        warmup_out, _, _ = lipa_solve(
            vars_in=vars_in,
            x0=x0,
            cost=problem.cost,
            dynamics=problem.dynamics,
            equalities=eq_fn,
            inequalities=ineq_fn,
            settings=settings,
        )
        jax.block_until_ready(warmup_out.X)

        start = timer()
        vars_out, iterations, no_errors = lipa_solve(
            vars_in=vars_in,
            x0=x0,
            cost=problem.cost,
            dynamics=problem.dynamics,
            equalities=eq_fn,
            inequalities=ineq_fn,
            settings=settings,
        )
        jax.block_until_ready(vars_out.X)
        solve_time_ms = 1e3 * (timer() - start) + warmup_time_ms

        X = np.asarray(vars_out.X)
        U = np.asarray(vars_out.U)
        Theta = np.asarray(vars_out.Theta)

        # Multiplier extraction. LIPA's Lagrangian convention (see
        # primal_dual_lipa.lagrangian_helpers.build_lagrangian):
        #
        #   L = f + Y_dyn[t+1] · dynamics(x_t, u_t)
        #         + Y_dyn[t] · (x0 - X[t])  if t == 0  else  Y_dyn[t] · (-X[t])
        #         + Y_eq · equalities + Z · (inequalities + S)
        #         - μ · sum_log_S
        #
        # Re-grouping the dynamics terms across t gives, for the inner
        # stages, ``Y_dyn[t+1] · (dynamics(x_t, u_t) - x_{t+1})`` — the
        # sign matches evaluate_problem's ``dyn_defects = dynamics - X[1:]``
        # convention. The init-defect term, however, is
        # ``Y_dyn[0] · (x0 - X[0])`` whereas evaluate_problem measures
        # ``init_defect = X[0] - x0`` — so Y_dyn[0] must be negated when
        # we hand it to the canonical evaluator. (This is purely a
        # sign-convention mapping; the underlying KKT condition is the
        # same.)
        Y_dyn = np.asarray(vars_out.Y_dyn)  # shape (T+1, n)
        Y_eq_arr = np.asarray(vars_out.Y_eq)  # shape (T+1, eq_dim)
        Z_arr = np.asarray(vars_out.Z)  # shape (T+1, ineq_dim)
        # Stack lambdas to match evaluate_problem's eq stack:
        #   [-Y_dyn[0] (init defect: X[0] - x0, n,),
        #    Y_dyn[1:T+1] (dyn defects: dyn - X[1:], T*n,),
        #    Y_eq.flatten() ((T+1)*eq_dim,)]
        lam_pieces = [(-Y_dyn[0]).reshape(-1), Y_dyn[1:].reshape(-1)]
        if eq_dim > 0:
            lam_pieces.append(Y_eq_arr.reshape(-1))
        multipliers_eq = np.concatenate(lam_pieces) if lam_pieces else np.zeros(0)
        if ineq_dim > 0:
            multipliers_ineq = Z_arr.reshape(-1)
        else:
            multipliers_ineq = np.zeros(0)

        total_iters = int(iterations) + warmup_iters
        notes = "two-phase warm start" if warmup_cost is not None else ""

        # Per-iter history: LIPA's solve is a fully-jitted while_loop and
        # we are forbidden from touching primal_dual_lipa/. We deliberately
        # leave the *_history arrays as None — the report layer falls back
        # to endpoint markers, matching the precedent set by Trajax. A
        # full per-iter trace would require either jax.lax.scan inside
        # primal_dual_lipa.optimizers.solve (out of scope) or a brittle
        # log-parse + re-solve scheme.
        return pack_solver_result(
            solver_name=self.name,
            problem_name=problem.name,
            problem=problem,
            X=X, U=U, Theta=Theta,
            iterations=total_iters,
            solve_time_ms=solve_time_ms,
            success=bool(no_errors),
            notes=notes,
            multipliers_eq=multipliers_eq,
            multipliers_ineq=multipliers_ineq,
        )


@register("lipa")
def _factory(**kwargs) -> SolverAdapter:
    return LipaAdapter(**kwargs)
