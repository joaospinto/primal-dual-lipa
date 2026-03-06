"""Defines the main solve function, implementing the Primal-Dual LIPA algorithm."""

from functools import partial

import jax
from jax import numpy as jnp

from primal_dual_lipa.kkt_builder import build_kkt, build_kkt_rhs
from primal_dual_lipa.kkt_helpers import (
    compute_kkt_residual,
    factor_kkt,
    solve_kkt,
)
from primal_dual_lipa.lagrangian_helpers import directional_augmented_lagrangian, pad
from primal_dual_lipa.types import (
    CostFunction,
    Function,
    KKTSystem,
    Parameters,
    SolverSettings,
    Variables,
)
from primal_dual_lipa.vectorization_helpers import vectorize


@partial(
    jax.jit,
    static_argnames=[
        "cost",
        "dynamics",
        "equalities",
        "inequalities",
        "settings",
    ],
)
def solve(
    vars_in: Variables,
    x0: jnp.ndarray,
    cost: CostFunction,
    dynamics: Function,
    settings: SolverSettings,
    equalities: Function = lambda x, u, theta, t: jnp.empty(0),
    inequalities: Function = lambda x, u, theta, t: jnp.empty(0),
) -> tuple[Variables, jnp.int32, jnp.bool]:
    """Implement the Primal-Dual LIPA algorithm for discrete-time optimal control.

    Args:
      vars_in:       Variables object containing warm-start values for X, U, S, Y_dyn, Y_eq, Z, Theta.
      x0:            [n]           numpy array (initial state).
      cost:          cost function with signature cost(x, u, theta, t).
      dynamics:      dynamics function with signature dynamics(x, u, theta, t).
      equalities:    equalities(x, u, theta, t) = 0 should hold; output is (c_dim,).
      inequalities:  inequalities(x, u, theta, t) <= 0 should hold; output is (g_dim,).
      settings:      the solver settings.

    Returns:
      Variables:   Variables object containing the solution for X, U, S, Y_dyn, Y_eq, Z, Theta.
      iterations:  the number of iterations it took to converge.
      no_errors:   whether no errors were encountered during the solve.

    """
    vars_current = Variables(
        X=vars_in.X,
        U=vars_in.U,
        S=jnp.maximum(vars_in.S, 1e-3),
        Y_dyn=vars_in.Y_dyn,
        Y_eq=vars_in.Y_eq,
        Z=jnp.maximum(vars_in.Z, 1e-3),
        Theta=vars_in.Theta,
    )

    # Infer residual shapes from inputs to initialize η
    T = vars_current.X.shape[0] - 1
    n = vars_current.X.shape[1]
    c_dim = vars_current.Y_eq.shape[-1]
    g_dim = vars_current.S.shape[-1]

    η_dyn = jnp.full((T + 1, n), settings.η0)
    η_eq = jnp.full((T + 1, c_dim), settings.η0)
    η_ineq = jnp.full((T + 1, g_dim), settings.η0)
    params_current = Parameters(µ=settings.µ0, η_dyn=η_dyn, η_eq=η_eq, η_ineq=η_ineq)

    kkt_system = build_kkt(
        cost=cost,
        dynamics=dynamics,
        equalities=equalities,
        inequalities=inequalities,
        x0=x0,
        vars=vars_current,
        params=params_current,
    )

    if settings.print_logs and not settings.print_ls_logs:
        jax.debug.print(
            "{:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10}".format(  # noqa: E501
                "iteration",
                "α",
                "cost",
                "|c|",
                "|g+s|",
                "merit",
                "dmerit/dα",
                "dmerit/dαdX",
                "dmerit/dαdS",
                "|dx|+|du|",
                "|ds|",
                "|dy|",
                "|dz|",
                "|dθ|",
                "avg(η)",
                "µ",
                "linsys_res",
            ),
            ordered=True,
        )

    def main_loop_body(
        inputs: tuple[
            Variables,
            Parameters,
            KKTSystem,
            jnp.int32,
        ],
    ) -> tuple[
        Variables,
        Parameters,
        KKTSystem,
        jnp.int32,
    ]:
        (
            vars,
            params,
            kkt_system,
            iteration,
        ) = inputs

        factorization_outputs = factor_kkt(
            inputs=kkt_system.lhs,
            use_parallel_lqr=settings.use_parallel_lqr,
        )

        deltas = solve_kkt(
            factorization_outputs=factorization_outputs,
            factorization_inputs=kkt_system.lhs,
            rhs=kkt_system.rhs,
            use_parallel_lqr=settings.use_parallel_lqr,
        )

        def iterative_refinement_loop_continuation_criteria(
            x: tuple[Variables, jnp.int32],
        ) -> jnp.bool:
            return x[-1] < settings.num_iterative_refinement_steps

        def iterative_refinement_loop_body(
            x: tuple[Variables, jnp.int32],
        ) -> tuple[Variables, jnp.int32]:
            deltas_inner, it = x
            residuals = compute_kkt_residual(
                factorization_inputs=kkt_system.lhs,
                solve_inputs=kkt_system.rhs,
                solution=deltas_inner,
            )

            refinement_deltas = solve_kkt(
                factorization_outputs=factorization_outputs,
                factorization_inputs=kkt_system.lhs,
                rhs=residuals,
                use_parallel_lqr=settings.use_parallel_lqr,
            )

            return (
                Variables(
                    X=deltas_inner.X + refinement_deltas.X,
                    U=deltas_inner.U + refinement_deltas.U,
                    S=deltas_inner.S + refinement_deltas.S,
                    Y_dyn=deltas_inner.Y_dyn + refinement_deltas.Y_dyn,
                    Y_eq=deltas_inner.Y_eq + refinement_deltas.Y_eq,
                    Z=deltas_inner.Z + refinement_deltas.Z,
                    Theta=deltas_inner.Theta + refinement_deltas.Theta,
                ),
                it + 1,
            )

        deltas, _ = jax.lax.while_loop(
            iterative_refinement_loop_continuation_criteria,
            iterative_refinement_loop_body,
            (deltas, 0),
        )

        τ = jnp.maximum(settings.τ_min, 1.0 - params.µ)

        def compute_alpha_max(v, dv, τ):
            mask = dv < 0
            safe_dv = jnp.where(mask, dv, -1.0)
            alphas = jnp.where(mask, -(τ * v) / safe_dv, 1.0)
            return jnp.min(jnp.concatenate([alphas.flatten(), jnp.array([1.0])]))

        α_max_s = compute_alpha_max(vars.S, deltas.S, τ)

        T_range = jnp.arange(T)
        Tp1_range = jnp.arange(T + 1)

        dal = directional_augmented_lagrangian(
            cost=cost,
            dynamics=dynamics,
            equalities=equalities,
            inequalities=inequalities,
            x0=x0,
            params=params,
            τ=τ,
            T=T,
            vars=vars,
            deltas=deltas,
        )

        dal_x = directional_augmented_lagrangian(
            cost=cost,
            dynamics=dynamics,
            equalities=equalities,
            inequalities=inequalities,
            x0=x0,
            params=params,
            τ=τ,
            T=T,
            vars=vars,
            deltas=Variables(
                X=deltas.X,
                U=deltas.U,
                S=jnp.zeros_like(deltas.S),
                Y_dyn=jnp.zeros_like(deltas.Y_dyn),
                Y_eq=jnp.zeros_like(deltas.Y_eq),
                Z=jnp.zeros_like(deltas.Z),
                Theta=jnp.zeros_like(deltas.Theta),
            ),
        )

        dal_s = directional_augmented_lagrangian(
            cost=cost,
            dynamics=dynamics,
            equalities=equalities,
            inequalities=inequalities,
            x0=x0,
            params=params,
            τ=τ,
            T=T,
            vars=vars,
            deltas=Variables(
                X=jnp.zeros_like(deltas.X),
                U=jnp.zeros_like(deltas.U),
                S=deltas.S,
                Y_dyn=jnp.zeros_like(deltas.Y_dyn),
                Y_eq=jnp.zeros_like(deltas.Y_eq),
                Z=jnp.zeros_like(deltas.Z),
                Theta=jnp.zeros_like(deltas.Theta),
            ),
        )

        baseline_merit = dal(0.0)

        merit_grad = jax.grad(dal)(0.0)
        merit_grad_x = jax.grad(dal_x)(0.0)
        merit_grad_s = jax.grad(dal_s)(0.0)

        if settings.print_ls_logs:
            jax.debug.print(
                "{:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10}".format(
                    "",
                    "α",
                    "cost",
                    "|c|",
                    "|g+s|",
                    "dmerit",
                    "dmerit/α",
                ),
                ordered=True,
            )

        def line_search_loop_body(α: jnp.double) -> jnp.double:
            return α * settings.α_update_factor

        def line_search_loop_continuation_criteria(α: jnp.double) -> jnp.bool:
            candidate_merit = dal(α)
            armijo_condition_not_met = (
                candidate_merit - baseline_merit
                > α * settings.armijo_factor * merit_grad
            )
            if settings.print_ls_logs:
                cX = vars.X + α * deltas.X
                cU = vars.U + α * deltas.U
                cU_pad = pad(cU)
                cS = jnp.maximum(vars.S + α * deltas.S, (1.0 - τ) * vars.S)
                cTheta = vars.Theta + α * deltas.Theta

                c_dyn0 = x0 - cX[0]
                c_dyn = vectorize(dynamics)(cX[:-1], cU, cTheta, T_range) - cX[1:]
                c_eq = vectorize(equalities)(cX, cU_pad, cTheta, Tp1_range)
                c = jnp.concatenate([c_dyn0, c_dyn.flatten(), c_eq.flatten()])

                g = vectorize(inequalities)(cX, cU_pad, cTheta, Tp1_range)
                gps = (g + cS).flatten()
                dmerit = candidate_merit - baseline_merit
                jax.debug.print(
                    "{:^10} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g}",  # noqa: E501
                    "",
                    α,
                    vectorize(cost)(cX, cU_pad, cTheta, Tp1_range).sum(),
                    jnp.linalg.norm(c),
                    jnp.linalg.norm(gps),
                    dmerit,
                    dmerit / α,
                    ordered=True,
                )
            return jnp.logical_and(α > settings.α_min, armijo_condition_not_met)

        α = jax.lax.while_loop(
            line_search_loop_continuation_criteria,
            line_search_loop_body,
            α_max_s,
        )

        if settings.print_logs:
            residuals = compute_kkt_residual(
                factorization_inputs=kkt_system.lhs,
                solve_inputs=kkt_system.rhs,
                solution=deltas,
            )
            residual_vec = jnp.concatenate(
                [
                    residuals.X.flatten(),
                    residuals.U.flatten(),
                    residuals.S.flatten(),
                    residuals.Y_dyn.flatten(),
                    residuals.Y_eq.flatten(),
                    residuals.Z.flatten(),
                    residuals.Theta.flatten(),
                ]
            )

            U_pad = pad(vars.U)
            T_range = jnp.arange(T)
            Tp1_range = jnp.arange(T + 1)

            c_dyn0 = x0 - vars.X[0]
            c_dyn = (
                vectorize(dynamics)(vars.X[:-1], vars.U, vars.Theta, T_range)
                - vars.X[1:]
            )
            c_eq = vectorize(equalities)(vars.X, U_pad, vars.Theta, Tp1_range)
            c = jnp.concatenate([c_dyn0, c_dyn.flatten(), c_eq.flatten()])

            g = vectorize(inequalities)(vars.X, U_pad, vars.Theta, Tp1_range)
            gps = (g + vars.S).flatten()

            avg_η = jnp.mean(
                jnp.concatenate(
                    [
                        params.η_dyn.flatten(),
                        params.η_eq.flatten(),
                        params.η_ineq.flatten(),
                    ]
                )
            )

            if settings.print_ls_logs:
                jax.debug.print(
                    "{:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10}".format(  # noqa: E501
                        "iteration",
                        "α",
                        "cost",
                        "|c|",
                        "|g+s|",
                        "merit",
                        "dmerit/dα",
                        "dmerit/dαdX",
                        "dmerit/dαdS",
                        "|dx|+|du|",
                        "|ds|",
                        "|dy|",
                        "|dz|",
                        "|dθ|",
                        "avg(η)",
                        "µ",
                        "linsys_res",
                    ),
                    ordered=True,
                )

            jax.debug.print(
                "{:^+10} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g}",  # noqa: E501
                iteration,
                α,
                vectorize(cost)(vars.X, U_pad, vars.Theta, Tp1_range).sum(),
                jnp.linalg.norm(c),
                jnp.linalg.norm(gps),
                baseline_merit,
                merit_grad,
                merit_grad_x,
                merit_grad_s,
                jnp.linalg.norm(deltas.X) + jnp.linalg.norm(pad(deltas.U)),
                jnp.linalg.norm(deltas.S),
                jnp.linalg.norm(deltas.Y_dyn) + jnp.linalg.norm(deltas.Y_eq),
                jnp.linalg.norm(deltas.Z),
                jnp.linalg.norm(deltas.Theta),
                avg_η,
                params.µ,
                jnp.linalg.norm(residual_vec),
                ordered=True,
            )

        updated_vars = Variables(
            X=vars.X + α * deltas.X,
            U=vars.U + α * deltas.U,
            S=jnp.maximum(vars.S + α * deltas.S, (1.0 - τ) * vars.S),
            Y_dyn=vars.Y_dyn + α * deltas.Y_dyn,
            Y_eq=vars.Y_eq + α * deltas.Y_eq,
            Z=jnp.maximum(vars.Z + α * deltas.Z, (1.0 - τ) * vars.Z),
            Theta=vars.Theta + α * deltas.Theta,
        )

        new_residual = build_kkt_rhs(
            cost=cost,
            dynamics=dynamics,
            equalities=equalities,
            inequalities=inequalities,
            x0=x0,
            vars=updated_vars,
            params=params,
        )

        improved_dyn = jnp.abs(
            new_residual.Y_dyn
        ) < settings.η_improvement_threshold * jnp.abs(kkt_system.rhs.Y_dyn)
        η_dyn_new = jnp.where(
            improved_dyn,
            params.η_dyn,
            jnp.minimum(params.η_dyn * settings.η_update_factor, settings.η_max),
        )

        improved_eq = jnp.abs(
            new_residual.Y_eq
        ) < settings.η_improvement_threshold * jnp.abs(kkt_system.rhs.Y_eq)
        η_eq_new = jnp.where(
            improved_eq,
            params.η_eq,
            jnp.minimum(params.η_eq * settings.η_update_factor, settings.η_max),
        )

        improved_ineq = jnp.abs(
            new_residual.Z
        ) < settings.η_improvement_threshold * jnp.abs(kkt_system.rhs.Z)
        η_ineq_new = jnp.where(
            improved_ineq,
            params.η_ineq,
            jnp.minimum(params.η_ineq * settings.η_update_factor, settings.η_max),
        )

        residual = jnp.concatenate(
            [
                new_residual.X.flatten(),
                new_residual.U.flatten(),
                # Note: absolutely do NOT just use new_residual.S.flatten() here.
                # Using (vars.S * new_residual.S).flatten() is also suboptimal,
                # but would be a lot closer to being OK.
                (vars.S * vars.Z).flatten(),
                new_residual.Y_dyn.flatten(),
                new_residual.Y_eq.flatten(),
                new_residual.Z.flatten(),
                new_residual.Theta.flatten(),
            ]
        )
        µ_new = jnp.where(
            jnp.abs(residual).max() > settings.κ * params.µ,
            params.µ,
            jnp.maximum(
                jnp.minimum(params.µ * settings.µ_update_factor, params.µ**1.5),
                settings.µ_min,
            ),
        )

        updated_params = Parameters(
            µ=µ_new,
            η_dyn=η_dyn_new,
            η_eq=η_eq_new,
            η_ineq=η_ineq_new,
        )

        kkt_system_new = build_kkt(
            cost=cost,
            dynamics=dynamics,
            equalities=equalities,
            inequalities=inequalities,
            x0=x0,
            vars=updated_vars,
            params=updated_params,
        )

        iteration += 1

        return (
            updated_vars,
            updated_params,
            kkt_system_new,
            iteration,
        )

    def main_loop_continuation_criteria(
        inputs: tuple[
            Variables,
            Parameters,
            KKTSystem,
            jnp.int32,
        ],
    ) -> jnp.bool:
        (
            vars,
            _unused_params,
            kkt_system,
            iteration,
        ) = inputs

        residual = jnp.concatenate(
            [
                kkt_system.rhs.X.flatten(),
                kkt_system.rhs.U.flatten(),
                # Note: absolutely do NOT just use kkt_system.rhs.S.flatten() here.
                # Using (vars.S * kkt_system.rhs.S).flatten() is also suboptimal,
                # but would be a lot closer to being OK.
                (vars.S * vars.Z).flatten(),
                kkt_system.rhs.Y_dyn.flatten(),
                kkt_system.rhs.Y_eq.flatten(),
                kkt_system.rhs.Z.flatten(),
                kkt_system.rhs.Theta.flatten(),
            ]
        )
        not_converged = jnp.sum(jnp.square(residual)) > settings.residual_sq_threshold
        no_iteration_limit = iteration < settings.max_iterations
        return jnp.logical_and(not_converged, no_iteration_limit)

    (
        vars_out,
        _unused_params,  # noqa: RUF059
        _unused_kkt_system,
        iteration,
    ) = jax.lax.while_loop(
        main_loop_continuation_criteria,
        main_loop_body,
        (
            vars_current,
            params_current,
            kkt_system,
            0,
        ),
    )

    no_errors = iteration < settings.max_iterations

    return vars_out, iteration, no_errors
