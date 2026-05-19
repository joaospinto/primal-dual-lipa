"""Adapter that runs the in-tree LIPA solver and returns a uniform ``SolverResult``."""

from __future__ import annotations

from timeit import default_timer as timer
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from primal_dual_lipa.optimizers import solve as lipa_solve
from primal_dual_lipa.types import Parameters, SolverSettings, Variables

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
                    settings.num_parallel_line_search_steps,
                    8,
                ),
            )

        # Dev-only: ``LIPA_DEV_MAX_ITERATIONS`` overrides the per-problem
        # max_iterations. Used during near-conv schedule debugging when
        # we want a uniform iter budget across problems regardless of
        # what their shipped settings declare.
        import os as _os

        _ovr = _os.environ.get("LIPA_DEV_MAX_ITERATIONS")
        if _ovr:
            import dataclasses as _dc

            settings = _dc.replace(settings, max_iterations=int(_ovr))

        # Per-problem ``success_tol`` (e.g. 1e-3 for MJX) overrides
        # LIPA's aux-gate primal-violation threshold so LIPA targets the
        # SAME bar as every other adapter — see ``effective_solver_tol``.
        # The per-problem ``lipa_settings.primal_violation_threshold``
        # in the config still wins when it's tighter; we only RELAX to
        # ``success_tol``, never tighten beyond what the config asked for.
        from tests.comparison.problem_spec import effective_solver_tol as _eff_tol

        _success_tol = _eff_tol(problem, settings.primal_violation_threshold)
        if _success_tol > settings.primal_violation_threshold:
            import dataclasses as _dc

            settings = _dc.replace(settings, primal_violation_threshold=_success_tol)

        eq_fn = (
            problem.equalities if problem.equalities is not None else _empty_equalities
        )
        ineq_fn = (
            problem.inequalities
            if problem.inequalities is not None
            else _empty_inequalities
        )

        # Dev-only: load full (X, U, S, Y_dyn, Y_eq, Z, Theta) from a
        # cached previous solve when LIPA_DEV_WARM_CACHE_DIR is set, so
        # iteration on the near-convergence schedule can skip the slow
        # "get near the solution" phase. Empty / disabled by default.
        from tests.comparison._dev import lipa_warm_cache as _lwc

        cached = _lwc.load(problem.name)

        # Build initial Variables. LIPA reserves Y_dyn for dynamics
        # multipliers, Y_eq for general-equality multipliers, S/Z for the
        # primal-dual interior-point slack/multiplier pair. The runner
        # can pre-populate problem.warm_start with dual+slack arrays
        # from a prior Phase-1 solve (two-phase MJX orchestration).
        X0 = jnp.asarray(
            cached["X"] if cached is not None and "X" in cached else problem.X_init
        )
        U0 = jnp.asarray(
            cached["U"] if cached is not None and "U" in cached else problem.U_init
        )
        Theta0 = jnp.asarray(
            cached["Theta"]
            if cached is not None and "Theta" in cached
            else problem.Theta_init
        )
        ws = problem.warm_start or {}
        if cached is not None:
            # Cache takes precedence over runner's warm_start (cache has full
            # Vars; runner's warm_start has only the post-Phase-1 dual+slack).
            ws = {**ws, **cached}

        def _ws(key, fallback):
            v = ws.get(key)
            return jnp.asarray(v) if v is not None else fallback

        Y_dyn0 = _ws("Y_dyn", jnp.zeros_like(X0))
        Y_eq0 = _ws("Y_eq", jnp.zeros((T + 1, eq_dim), dtype=X0.dtype))
        S0 = _ws("S", jnp.zeros((T + 1, ineq_dim), dtype=X0.dtype))
        Z0 = _ws("Z", jnp.zeros((T + 1, ineq_dim), dtype=X0.dtype))
        # Cross-phase shape mismatch (Phase 1 had ineq_dim=0 and
        # therefore S/Z shape (T+1, 0); Phase 2 has full ineq_dim)
        # falls back to fresh zeros for the mismatched array, matching
        # the historical in-adapter two-phase behavior.
        if S0.shape != (T + 1, ineq_dim):
            S0 = jnp.zeros((T + 1, ineq_dim), dtype=X0.dtype)
        if Z0.shape != (T + 1, ineq_dim):
            Z0 = jnp.zeros((T + 1, ineq_dim), dtype=X0.dtype)
        vars_in = Variables(
            X=X0, U=U0, S=S0, Y_dyn=Y_dyn0, Y_eq=Y_eq0, Z=Z0, Theta=Theta0
        )

        x0 = jnp.asarray(problem.x0)

        # Dev-only: rebuild Parameters from cache when present (mu/eta
        # values from the prior run). Skips the slow η ramp on re-solves.
        params_in = None
        if cached is not None and all(
            k in cached for k in ("µ", "η_dyn", "η_eq", "η_ineq")
        ):
            cached_η_eq = jnp.asarray(cached["η_eq"])
            cached_η_ineq = jnp.asarray(cached["η_ineq"])
            # Reject cached η shapes that don't match this run (e.g.,
            # cross-phase ineq_dim mismatch). Fresh init otherwise.
            if cached_η_eq.shape == (T + 1, eq_dim) and cached_η_ineq.shape == (
                T + 1,
                ineq_dim,
            ):
                params_in = Parameters(
                    µ=jnp.asarray(cached["µ"]),
                    η_dyn=jnp.asarray(cached["η_dyn"]),
                    η_eq=cached_η_eq,
                    η_ineq=cached_η_ineq,
                )

        # Warm-up call so JIT compile time is excluded from the timed
        # solve. The Phase-2 vs Phase-1 split historically lived here;
        # it now lives in run_benchmark.py via the shared two-phase
        # orchestration (gated on problem.metadata['lipa_two_phase']).
        warmup_out, _, _, _ = lipa_solve(
            vars_in=vars_in,
            x0=x0,
            cost=problem.cost,
            dynamics=problem.dynamics,
            equalities=eq_fn,
            inequalities=ineq_fn,
            settings=settings,
            params_in=params_in,
        )
        jax.block_until_ready(warmup_out.X)

        start = timer()
        vars_out, iterations, no_errors, final_params = lipa_solve(
            vars_in=vars_in,
            x0=x0,
            cost=problem.cost,
            dynamics=problem.dynamics,
            equalities=eq_fn,
            inequalities=ineq_fn,
            settings=settings,
            params_in=params_in,
        )
        jax.block_until_ready(vars_out.X)
        solve_time_ms = 1e3 * (timer() - start)

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

        total_iters = int(iterations)

        # Per-iter history: LIPA's solve is a fully-jitted while_loop and
        # we are forbidden from touching primal_dual_lipa/. We deliberately
        # leave the *_history arrays as None — the report layer falls back
        # to endpoint markers, matching the precedent set by Trajax. A
        # full per-iter trace would require either jax.lax.scan inside
        # primal_dual_lipa.optimizers.solve (out of scope) or a brittle
        # log-parse + re-solve scheme.
        # Expose the full internal LIPA state for the runner's
        # two-phase orchestration to feed back as Phase-2 warm_start.
        # Y_dyn / Y_eq / S / Z are LIPA's dual and slack variables; the
        # runner doesn't interpret them — it just plumbs them through.
        S_arr = np.asarray(vars_out.S)
        # Dev-only: persist the full iterate to disk for next-run
        # warm-start. No-op unless LIPA_DEV_WARM_CACHE_DIR is set.
        _lwc.save(
            problem.name,
            {
                "X": X,
                "U": U,
                "Theta": Theta,
                "S": S_arr,
                "Y_dyn": Y_dyn,
                "Y_eq": Y_eq_arr,
                "Z": Z_arr,
                "µ": np.asarray(final_params.µ),
                "η_dyn": np.asarray(final_params.η_dyn),
                "η_eq": np.asarray(final_params.η_eq),
                "η_ineq": np.asarray(final_params.η_ineq),
            },
        )
        warm_start_out = {
            "Y_dyn": Y_dyn,
            "Y_eq": Y_eq_arr,
            "S": S_arr,
            "Z": Z_arr,
        }
        return pack_solver_result(
            solver_name=self.name,
            problem_name=problem.name,
            problem=problem,
            X=X,
            U=U,
            Theta=Theta,
            iterations=total_iters,
            solve_time_ms=solve_time_ms,
            success=bool(no_errors),
            multipliers_eq=multipliers_eq,
            multipliers_ineq=multipliers_ineq,
            warm_start_out=warm_start_out,
        )


@register("lipa")
def _factory(**kwargs) -> SolverAdapter:
    return LipaAdapter(**kwargs)
