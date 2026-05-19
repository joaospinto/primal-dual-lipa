"""LIPA offline-solve driver.

Translates the ``mpx``-style ``solve(reference, parameter, W, x0, X0,
U0, V0)`` interface to LIPA's ``Variables``-pytree call. Implements the
two-phase warm-start path (soft-penalty cost first, then constrained
solve) that's load-bearing for ``barrel_roll`` — see the inline comment
in ``run_lipa_offline``.

The two-phase warm start is load-bearing for ``barrel_roll`` (multi-shooting
quaternion defect at the apex hits a sign-flip singularity that the cold-
start solve cannot escape) — see the comments in ``run_lipa_offline``.
"""

from functools import partial
from timeit import default_timer as timer

import jax
import jax.numpy as jnp
import numpy as np

from primal_dual_lipa.optimizers import solve as lipa_solve
from primal_dual_lipa.types import SolverSettings, Variables


def _wrap_cost(cost):
    def lipa_cost(W, reference, x, u, theta, t):
        del theta
        return cost(W, reference, x, u, t)

    return lipa_cost


def _wrap_dynamics(dynamics):
    def lipa_dynamics(parameter, x, u, theta, t):
        del theta
        return dynamics(x, u, t, parameter=parameter)

    return lipa_dynamics


def _wrap_inequalities(inequalities):
    def lipa_inequalities(reference, x, u, theta, t):
        del theta
        return inequalities(reference, x, u, t)

    return lipa_inequalities


def _empty_inequalities(reference, x, u, t):
    del reference, x, u, t
    return jnp.empty(0)


def _model_evaluator(cost, dynamics, x0, X, U):
    """Sum-of-stage-costs and dynamics-defect residual for a (X, U) trajectory."""
    T = U.shape[0]
    costs = jax.vmap(cost)(X, jnp.pad(U, [[0, 1], [0, 0]]), jnp.arange(T + 1))
    g = jnp.sum(costs)

    def residual_fn(t):
        return dynamics(X[t], U[t], t) - X[t + 1]

    c = jnp.vstack([x0 - X[0], jax.vmap(residual_fn)(jnp.arange(T))])
    return g, c


@partial(jax.jit, static_argnames=("cost", "dynamics", "inequalities"))
def _lipa_solve_with_stats(
    cost,
    dynamics,
    inequalities,
    settings,
    reference,
    parameter,
    W,
    x0,
    X_in,
    U_in,
    V_in,
):
    """Single LIPA call returning final variables plus the iteration count."""
    lipa_cost = partial(_wrap_cost(cost), W, reference)
    lipa_dynamics = partial(_wrap_dynamics(dynamics), parameter)

    ineq_callable = inequalities if inequalities is not None else _empty_inequalities
    lipa_inequalities = partial(_wrap_inequalities(ineq_callable), reference)

    T = U_in.shape[0]
    sample_g = lipa_inequalities(X_in[0], U_in[0], jnp.empty(0, dtype=X_in.dtype), 0)
    g_dim = sample_g.shape[0]

    vars_in = Variables(
        X=X_in,
        U=U_in,
        S=jnp.zeros((T + 1, g_dim), dtype=X_in.dtype),
        Y_dyn=V_in,
        Y_eq=jnp.zeros((T + 1, 0), dtype=X_in.dtype),
        Z=jnp.zeros((T + 1, g_dim), dtype=X_in.dtype),
        Theta=jnp.empty(0, dtype=X_in.dtype),
    )

    vars_out, iterations, no_errors, _ = lipa_solve(
        vars_in=vars_in,
        x0=x0,
        cost=lipa_cost,
        dynamics=lipa_dynamics,
        inequalities=lipa_inequalities,
        settings=settings,
    )
    return vars_out.X, vars_out.U, vars_out.Y_dyn, iterations, no_errors


def _default_settings():
    on_gpu = any(d.platform == "gpu" for d in jax.devices())
    common = dict(
        max_iterations=2000,
        η0=1e3,
        η_update_factor=1.0,
        µ_update_factor=0.9,
        cost_improvement_threshold=1e-3,
        primal_violation_threshold=1e-6,
    )
    if on_gpu:
        return SolverSettings(
            use_parallel_lqr=True,
            num_parallel_line_search_steps=8,
            **common,
        )
    return SolverSettings(**common)


def lipa_pick_cost_and_inequalities(config, cost):
    """Choose (main_cost, ineqs, settings, warmup_cost, warmup_settings) per config.

    Mirrors the LIPA branch of mpx's ``lipa_pick_cost_and_inequalities``:

    * If ``config.lipa_enforce_inequalities`` is False (or unset): use ``cost``
      (the soft-penalty version) with no inequalities, and the config's
      ``lipa_settings``.
    * If True: use ``cost_smooth`` + ``inequalities`` with
      ``lipa_settings_enforce or lipa_settings``, and return ``cost`` /
      ``lipa_settings`` as the warm-start pair so the offline path can do a
      two-phase solve.
    * If ``config.lipa_skip_warmup_phase`` is True: same as above but the
      warm-start pair is suppressed (returned as None), so the offline path
      goes single-phase. Useful when the constrained IPM converges well
      enough from the bare reference initial guess that the warmup phase
      adds latency without improving the result.
    """
    enforce = getattr(config, "lipa_enforce_inequalities", False)
    base_settings = getattr(config, "lipa_settings", None)
    if not enforce:
        return cost, None, base_settings, None, None
    cost_smooth = getattr(config, "cost_smooth", None)
    inequalities = getattr(config, "inequalities", None)
    if cost_smooth is None or inequalities is None:
        msg = (
            "lipa_enforce_inequalities=True requires both `cost_smooth` and "
            "`inequalities` to be defined on the config."
        )
        raise ValueError(msg)
    enforce_settings = getattr(config, "lipa_settings_enforce", None) or base_settings
    skip_warmup = getattr(config, "lipa_skip_warmup_phase", False)
    if skip_warmup:
        return cost_smooth, inequalities, enforce_settings, None, None
    return cost_smooth, inequalities, enforce_settings, cost, base_settings


def run_lipa_offline(
    cost,
    dynamics,
    reference,
    parameter,
    W,
    x0,
    X0,
    U0,
    V0,
    *,
    settings=None,
    inequalities=None,
    warmup_cost=None,
    warmup_settings=None,
    verbose=True,
):
    """Solve a single OCP with LIPA, return ``(X, U, V, history, stats)``.

    Two-phase warm start when ``warmup_cost`` is provided: an initial LIPA
    solve on the inequality-free formulation produces the warm start for the
    main inequality-enforcing solve. This sidesteps a class of local-basin
    pitfalls — notably the multi-shooting quaternion-defect sign-flip
    singularity at the apex of the barrel-roll maneuver, where the AL term
    η·Jᵀc dominates and the IPM parks at a degenerate iterate.

    NB: we deliberately do NOT call ``_lipa_solve_with_stats`` twice in
    single-phase mode (warmup + timed) here — the parallel-LQR scan reduction
    is not bit-deterministic across back-to-back invocations of the same
    compiled function on the same inputs (different floating-point summation
    order can land on numerically different iterates). For two-phase, the
    phase-1 wall time inevitably includes any first-call JIT compile.
    """
    if settings is None:
        settings = _default_settings()

    offline_cost = partial(cost, W, reference)
    offline_dynamics = partial(dynamics, parameter=parameter)
    model_evaluator = jax.jit(
        partial(_model_evaluator, offline_cost, offline_dynamics, x0)
    )

    g0, c0 = model_evaluator(X0, U0)
    initial_objective = float(g0)
    initial_l2_cost = float(np.sqrt(np.sum(np.asarray(g0) * np.asarray(g0))))
    initial_dynamics_violation = float(np.sum(np.asarray(c0) * np.asarray(c0)))

    if verbose:
        print(
            "{:<10} {:<20} {:<20} {:<20}".format(
                "Iter", "Cost", "Constraint", "Time [ms]"
            )
        )
        print(
            "{:<10d} {:<20.5f} {:<20.5f} {:<20}".format(
                0, initial_l2_cost, initial_dynamics_violation, "-"
            )
        )

    do_warmup_phase = warmup_cost is not None and inequalities is not None
    warmup_phase_settings = warmup_settings if warmup_settings is not None else settings
    warmup_iters = 0
    warmup_time_ms = 0.0

    if do_warmup_phase:
        start = timer()
        Xp1, Up1, Vp1, iters_p1, _ = _lipa_solve_with_stats(
            warmup_cost,
            dynamics,
            None,
            warmup_phase_settings,
            reference,
            parameter,
            W,
            x0,
            X0,
            U0,
            V0,
        )
        Xp1.block_until_ready()
        warmup_time_ms = 1e3 * (timer() - start)
        warmup_iters = int(iters_p1)
        if verbose:
            print(
                "{:<10s} {:<20s} {:<20s} {:<20.5f}".format(
                    "ph1", "(warmup)", "(warmup)", warmup_time_ms
                )
            )
            print(f"  Phase 1 (soft-penalty warm start): {warmup_iters} iters")
        X0, U0, V0 = Xp1, Up1, Vp1
    else:
        # Single-phase mode: warmup-then-timed pattern (timed run is not the
        # first call into the compiled function so JIT compile time is excluded).
        Xw, _, _, _, _ = _lipa_solve_with_stats(
            cost,
            dynamics,
            inequalities,
            settings,
            reference,
            parameter,
            W,
            x0,
            X0,
            U0,
            V0,
        )
        Xw.block_until_ready()

    start = timer()
    X, U, V, iterations, no_errors = _lipa_solve_with_stats(
        cost, dynamics, inequalities, settings, reference, parameter, W, x0, X0, U0, V0
    )
    X.block_until_ready()
    iteration_time_ms = 1e3 * (timer() - start)

    g, c = model_evaluator(X, U)
    final_objective = float(g)
    final_l2_cost = float(np.sqrt(np.sum(np.asarray(g) * np.asarray(g))))
    final_dynamics_violation = float(np.sum(np.asarray(c) * np.asarray(c)))
    n_iters = int(iterations) + warmup_iters
    converged = bool(no_errors)

    if verbose:
        print(
            "{:<10d} {:<20.5f} {:<20.5f} {:<20.5f}".format(
                1, final_l2_cost, final_dynamics_violation, iteration_time_ms
            )
        )
        if do_warmup_phase:
            print(
                f"  Phase 2 (constrained): {int(iterations)} iters, no_errors: {converged}\n"
                f"  Total LIPA internal iterations: {n_iters}"
            )
        else:
            print(f"  LIPA internal iterations: {n_iters}, no_errors: {converged}")

    history = [X0, X]
    stats = {
        "n_iterations": n_iters,
        "warmup_iterations": warmup_iters,
        "converged": converged,
        "objective_history": [initial_objective, final_objective],
        "l2_cost_history": [initial_l2_cost, final_l2_cost],
        "dynamics_violation_history": [
            initial_dynamics_violation,
            final_dynamics_violation,
        ],
        "metric_iteration_history": [0, 1],
        "iteration_time_ms_history": [iteration_time_ms + warmup_time_ms],
        "initial_objective": initial_objective,
        "initial_l2_cost": initial_l2_cost,
        "initial_dynamics_violation": initial_dynamics_violation,
        "average_iteration_time_ms": iteration_time_ms + warmup_time_ms,
        "final_objective": final_objective,
        "final_l2_cost": final_l2_cost,
        "final_dynamics_violation": final_dynamics_violation,
    }
    return X, U, V, history, stats
