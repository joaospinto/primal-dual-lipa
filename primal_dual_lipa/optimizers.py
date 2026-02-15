"""Defines the main solve function, implementing the Primal-Dual LIPA algorithm."""

from functools import partial

import jax
from jax import numpy as jnp

from primal_dual_lipa.kkt_builder import kkt_builder
from primal_dual_lipa.kkt_helpers import compute_kkt_residual, solve_kkt
from primal_dual_lipa.lagrangian_helpers import directional_augmented_lagrangian, pad
from primal_dual_lipa.types import CostFunction, Function, SolverSettings


@partial(
    jax.jit,
    static_argnames=[
        "cost",
        "dynamics",
        "equalities",
        "inequalities",
    ],
)
def solve(
    X_in: jnp.ndarray,
    U_in: jnp.ndarray,
    S_in: jnp.ndarray,
    Y_dyn_in: jnp.ndarray,
    Y_eq_in: jnp.ndarray,
    Z_in: jnp.ndarray,
    x0: jnp.ndarray,
    cost: CostFunction,
    dynamics: Function,
    settings: SolverSettings,
    equalities: Function = lambda x, u, t: jnp.empty(0),
    inequalities: Function = lambda x, u, t: jnp.empty(0),
) -> tuple[
    jnp.ndarray,  # X
    jnp.ndarray,  # U
    jnp.ndarray,  # S
    jnp.ndarray,  # Y_dyn
    jnp.ndarray,  # Y_eq
    jnp.ndarray,  # Z
    jnp.int32,  # iterations
    jnp.bool,  # no_errors
]:
    """Implement the Primal-Dual LIPA algorithm for discrete-time optimal control.

    Args:
      X_in:          [T+1, n]      numpy array (state warm-start).
      U_in:          [T, m]        numpy array (control warm-start).
      S_in:          [T+1, g_dim]  numpy array (slacks warm-start).
      Y_dyn_in:      [T, n]        numpy array (costates warm-start).
      Y_eq_in:       [T+1, c_dim]  numpy array (equalities warm-start).
      Z_in:          [T+1, g_dim]  numpy array (inequalities warm-start).
      x0:            [n]           numpy array (initial state).
      cost:          cost function with signature cost(x, u, t).
      dynamics:      dynamics function with signature dynamics(x, u, t).
      equalities:    equalities(x, u, t) = 0 should hold; output is (c_dim,).
      inequalities:  inequalities(x, u, t) <= 0 should hold; output is (g_dim,).
      settings:      the solver settings.

    Returns:
      X:           [T+1, n]      numpy array (state solution).
      U:           [T, m]        numpy array (control solution).
      S:           [T+1, g_dim]  numpy array (slacks solution).
      Y_dyn:       [T, n]        numpy array (costates solution).
      Y_eq:        [T+1, c_dim]  numpy array (equalities solution).
      Z:           [T+1, g_dim]  numpy array (inequalities solution).
      iterations:  the number of iterations it took to converge.
      no_errors:   whether no errors were encountered during the solve.

    """
    S_in = jnp.maximum(S_in, 1e-3)
    Z_in = jnp.maximum(Z_in, 1e-3)

    T = X_in.shape[0] - 1

    P, D, E, G, r_x, r_s, r_y_dyn, r_y_eq, r_z = kkt_builder(
        cost=cost,
        dynamics=dynamics,
        equalities=equalities,
        inequalities=inequalities,
        x0=x0,
        X=X_in,
        U=U_in,
        S=S_in,
        Y_dyn=Y_dyn_in,
        Y_eq=Y_eq_in,
        Z=Z_in,
        µ=settings.µ0,
    )

    if settings.print_logs:
        jax.debug.print(
            "{:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10}".format(  # noqa: E501
                "iteration",
                "α",
                "cost",
                "|c|",
                "|g+s|",
                "merit",
                "dmerit/dα",
                "|dx|+|du|",
                "|ds|",
                "|dy|",
                "|dz|",
                "η",
                "µ",
                "linsys_res",
            )
        )

    def main_loop_body(
        inputs: tuple[
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.int32,
        ],
    ) -> tuple[
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.double,
        jnp.double,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.int32,
    ]:
        (
            X,
            U,
            S,
            Y_dyn,
            Y_eq,
            Z,
            η,
            µ,
            P,
            D,
            E,
            G,
            r_x,
            r_s,
            r_y_dyn,
            r_y_eq,
            r_z,
            iteration,
        ) = inputs
        w_inv = jnp.maximum(Z / S, 1e-8)
        dX, dU, dS, dY_dyn, dY_eq, dZ = solve_kkt(
            P=P,
            D=D,
            E=E,
            G=G,
            w_inv=w_inv,
            r_x=r_x,
            r_s=r_s,
            r_y_dyn=r_y_dyn,
            r_y_eq=r_y_eq,
            r_z=r_z,
            η=η,
            use_parallel_lqr=settings.use_parallel_lqr,
        )

        def iterative_refinement_loop_continuation_criteria(
            x: tuple[
                jnp.ndarray,
                jnp.ndarray,
                jnp.ndarray,
                jnp.ndarray,
                jnp.ndarray,
                jnp.ndarray,
                jnp.int32,
            ],
        ) -> jnp.bool:
            return x[-1] < settings.num_iterative_refinement_steps

        def iterative_refinement_loop_body(
            x: tuple[
                jnp.ndarray,
                jnp.ndarray,
                jnp.ndarray,
                jnp.ndarray,
                jnp.ndarray,
                jnp.ndarray,
                jnp.int32,
            ],
        ) -> tuple[
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.int32,
        ]:
            dX, dU, dS, dY_dyn, dY_eq, dZ, it = x
            res_X, res_U, res_S, res_Y_dyn, res_Y_eq, res_Z = compute_kkt_residual(
                P=P,
                D=D,
                E=E,
                G=G,
                w_inv=w_inv,
                r_x=r_x,
                r_s=r_s,
                r_y_dyn=r_y_dyn,
                r_y_eq=r_y_eq,
                r_z=r_z,
                dX=dX,
                dU=dU,
                dS=dS,
                dY_dyn=dY_dyn,
                dY_eq=dY_eq,
                dZ=dZ,
                η=η,
            )

            res_XU = jnp.concatenate([res_X, res_U], axis=-1)

            ddX, ddU, ddS, ddY_dyn, ddY_eq, ddZ = solve_kkt(
                P=P,
                D=D,
                E=E,
                G=G,
                w_inv=w_inv,
                r_x=res_XU,
                r_s=res_S,
                r_y_dyn=res_Y_dyn,
                r_y_eq=res_Y_eq,
                r_z=res_Z,
                η=η,
                use_parallel_lqr=settings.use_parallel_lqr,
            )

            return (
                dX + ddX,
                dU + ddU,
                dS + ddS,
                dY_dyn + ddY_dyn,
                dY_eq + ddY_eq,
                dZ + ddZ,
                it + 1,
            )

        dX, dU, dS, dY_dyn, dY_eq, dZ, _ = jax.lax.while_loop(
            iterative_refinement_loop_continuation_criteria,
            iterative_refinement_loop_body,
            (dX, dU, dS, dY_dyn, dY_eq, dZ, 0),
        )

        dU = dU[:-1]

        dal = directional_augmented_lagrangian(
            cost=cost,
            dynamics=dynamics,
            equalities=equalities,
            inequalities=inequalities,
            x0=x0,
            µ=µ,
            η=η,
            τ=settings.τ,
            T=T,
            X=X,
            U=U,
            S=S,
            Y_dyn=Y_dyn,
            Y_eq=Y_eq,
            Z=Z,
            dX=dX,
            dU=dU,
            dS=dS,
        )

        baseline_merit = dal(0.0)

        merit_grad = jax.grad(dal)(0.0)

        def line_search_loop_body(α: jnp.double) -> jnp.double:
            return α * settings.α_update_factor

        def line_search_loop_continuation_criteria(α: jnp.double) -> jnp.bool:
            candidate_merit = dal(α)
            armijo_condition_not_met = (
                candidate_merit - baseline_merit
                > α * settings.armijo_factor * merit_grad
            )
            return jnp.logical_and(α > settings.α_min, armijo_condition_not_met)

        α = jax.lax.while_loop(
            line_search_loop_continuation_criteria,
            line_search_loop_body,
            1.0,
        )

        if settings.print_logs:
            res_X, res_U, res_S, res_Y_dyn, res_Y_eq, res_Z = compute_kkt_residual(
                P=P,
                D=D,
                E=E,
                G=G,
                w_inv=w_inv,
                r_x=r_x,
                r_s=r_s,
                r_y_dyn=r_y_dyn,
                r_y_eq=r_y_eq,
                r_z=r_z,
                dX=dX,
                dU=pad(dU),
                dS=dS,
                dY_dyn=dY_dyn,
                dY_eq=dY_eq,
                dZ=dZ,
                η=η,
            )
            residual = jnp.concatenate(
                [
                    res_X.flatten(),
                    res_U.flatten(),
                    res_S.flatten(),
                    res_Y_dyn.flatten(),
                    res_Y_eq.flatten(),
                    res_Z.flatten(),
                ]
            )

            U_pad = pad(U)
            T_range = jnp.arange(T)
            Tp1_range = jnp.arange(T + 1)

            c_dyn0 = x0 - X[0]
            c_dyn = jax.vmap(dynamics)(X[:-1], U, T_range) - X[1:]
            c_eq = jax.vmap(equalities)(X, U_pad, Tp1_range)
            c = jnp.concatenate([c_dyn0, c_dyn.flatten(), c_eq.flatten()])

            g = (jax.vmap(inequalities)(X, U_pad, Tp1_range) + S).flatten()

            jax.debug.print(
                "{:^+10} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g}",  # noqa: E501
                iteration,
                α,
                jax.vmap(cost)(X, U_pad, Tp1_range).sum(),
                jnp.linalg.norm(c),
                jnp.linalg.norm(g),
                baseline_merit,
                merit_grad,
                jnp.linalg.norm(dX) + jnp.linalg.norm(dU),
                jnp.linalg.norm(dS),
                jnp.linalg.norm(dY_dyn) + jnp.linalg.norm(dY_eq),
                jnp.linalg.norm(dZ),
                η,
                µ,
                jnp.linalg.norm(residual),
            )

        X += α * dX
        U += α * dU
        S = jnp.maximum(S + α * dS, (1.0 - settings.τ) * S)
        Y_dyn += α * dY_dyn
        Y_eq += α * dY_eq
        Z = jnp.maximum(Z + α * dZ, (1.0 - settings.τ) * Z)

        P, D, E, G, r_x, r_s, r_y_dyn, r_y_eq, r_z = kkt_builder(
            cost=cost,
            dynamics=dynamics,
            equalities=equalities,
            inequalities=inequalities,
            x0=x0,
            X=X,
            U=U,
            S=S,
            Y_dyn=Y_dyn,
            Y_eq=Y_eq,
            Z=Z,
            µ=µ,
        )

        η = jnp.minimum(η * settings.η_update_factor, settings.η_max)
        µ = jnp.maximum(µ * settings.µ_update_factor, settings.µ_min)

        iteration += 1

        return (
            X,
            U,
            S,
            Y_dyn,
            Y_eq,
            Z,
            η,
            µ,
            P,
            D,
            E,
            G,
            r_x,
            r_s,
            r_y_dyn,
            r_y_eq,
            r_z,
            iteration,
        )

    def main_loop_continuation_criteria(
        inputs: tuple[
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.int32,
        ],
    ) -> jnp.bool:
        (
            _unused_X,
            _unused_U,
            _unused_S,
            _unused_Y_dyn,
            _unused_Y_eq,
            _unused_Z,
            _unused_η,
            _unused_µ,
            _unused_P,
            _unused_D,
            _unused_E,
            _unused_G,
            r_x,
            r_s,
            r_y_dyn,
            r_y_eq,
            r_z,
            iteration,
        ) = inputs
        residual = jnp.concatenate(
            [
                r_x.flatten(),
                r_s.flatten(),
                r_y_dyn.flatten(),
                r_y_eq.flatten(),
                r_z.flatten(),
            ]
        )
        not_converged = jnp.sum(jnp.square(residual)) > settings.residual_sq_threshold
        no_iteration_limit = iteration < settings.max_iterations
        return jnp.logical_and(not_converged, no_iteration_limit)

    (
        X,
        U,
        S,
        Y_dyn,
        Y_eq,
        Z,
        _unused_η,  # noqa: RUF059
        _unused_µ,  # noqa: RUF059
        _unused_P,
        _unused_D,
        _unused_E,
        _unused_G,
        _unused_r_x,
        _unused_r_s,
        _unused_r_y_dyn,
        _unused_r_y_eq,
        _unused_r_z,
        iteration,
    ) = jax.lax.while_loop(
        main_loop_continuation_criteria,
        main_loop_body,
        (
            X_in,
            U_in,
            S_in,
            Y_dyn_in,
            Y_eq_in,
            Z_in,
            settings.η0,
            settings.µ0,
            P,
            D,
            E,
            G,
            r_x,
            r_s,
            r_y_dyn,
            r_y_eq,
            r_z,
            0,
        ),
    )

    no_errors = iteration < settings.max_iterations

    return X, U, S, Y_dyn, Y_eq, Z, iteration, no_errors
