"""Defines the main solve function, implementing the Primal-Dual LIPA algorithm."""

from functools import partial

import jax
from jax import numpy as jnp

from primal_dual_lipa.kkt_builder import kkt_builder
from primal_dual_lipa.kkt_helpers import solve_kkt
from primal_dual_lipa.lagrangian_helpers import directional_augmented_lagrangian
from primal_dual_lipa.types import Function, SolverSettings


@partial(
    jax.jit,
    static_argnames=[
        "cost",
        "dynamics",
        "equalities",
        "inequalities",
        "use_parallel_lqr",
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
    cost: Function,
    dynamics: Function,
    settings: SolverSettings,
    equalities: Function = lambda x, u, t: jnp.empty(1),
    inequalities: Function = lambda x, u, t: jnp.empty(1),
) -> tuple[
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.int32,
    jnp.bool,
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

    def main_loop_body(
        X: jnp.ndarray,
        U: jnp.ndarray,
        S: jnp.ndarray,
        Y_dyn: jnp.ndarray,
        Y_eq: jnp.ndarray,
        Z: jnp.ndarray,
        η: jnp.double,
        µ: jnp.double,
        P: jnp.ndarray,
        D: jnp.ndarray,
        E: jnp.ndarray,
        G: jnp.ndarray,
        r_x: jnp.ndarray,
        r_s: jnp.ndarray,
        r_y_dyn: jnp.ndarray,
        r_y_eq: jnp.ndarray,
        r_z: jnp.ndarray,
        iteration: jnp.int32,
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
        dX, dU, dS, dY_dyn, dY_eq, dZ = solve_kkt(
            P=P,
            D=D,
            E=E,
            G=G,
            s=S,
            z=Z,
            r_x=r_x,
            r_s=r_s,
            r_y_dyn=r_y_dyn,
            r_y_eq=r_y_eq,
            r_z=r_z,
            η=η,
            use_parallel_lqr=use_parallel_lqr,
        )

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

        η *= settings.η_update_factor
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
        _unused_X: jnp.ndarray,
        _unused_U: jnp.ndarray,
        _unused_S: jnp.ndarray,
        _unused_Y_dyn: jnp.ndarray,
        _unused_Y_eq: jnp.ndarray,
        _unused_Z: jnp.ndarray,
        _unused_η: jnp.double,  # noqa: ARG001
        _unused_µ: jnp.double,  # noqa: ARG001
        _unused_P: jnp.ndarray,
        _unused_D: jnp.ndarray,
        _unused_E: jnp.ndarray,
        _unused_G: jnp.ndarray,
        r_x: jnp.ndarray,
        r_s: jnp.ndarray,
        r_y_dyn: jnp.ndarray,
        r_y_eq: jnp.ndarray,
        r_z: jnp.ndarray,
        iteration: jnp.int32,
    ) -> jnp.bool:
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
