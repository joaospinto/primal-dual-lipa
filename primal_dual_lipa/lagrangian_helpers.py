"""Provides helper methods for computing the Lagrangian and Augmented Lagrangian."""

import jax
from jax import numpy as jnp

from primal_dual_lipa.types import CostFunction, Function, Parameters, Variables


def pad(A: jnp.ndarray) -> jnp.ndarray:
    """Pad with zeros along the first axis by an extra element."""
    return jnp.pad(A, [[0, 1], [0, 0]])


def build_lagrangian(  # noqa: ANN201
    cost: CostFunction,
    dynamics: Function,
    equalities: Function,
    inequalities: Function,
    x0: jnp.ndarray,
    µ: jnp.double,
):
    """Return a function to evaluate the associated Lagrangian."""

    def lagrangian(
        X: jnp.ndarray,
        U: jnp.ndarray,
        t: jnp.int32,
        S: jnp.ndarray,
        next_Y_dyn: jnp.ndarray,
        Y_dyn: jnp.ndarray,
        Y_eq: jnp.ndarray,
        Z: jnp.ndarray,
    ) -> jnp.double:
        c1 = cost(X, U, t)
        c2 = jnp.dot(next_Y_dyn, dynamics(X, U, t))
        c3 = jnp.dot(Y_dyn, jax.lax.select(t == 0, x0 - X, -X))
        c4 = -µ * jnp.sum(jnp.log(S))
        c5 = jnp.dot(Y_eq, equalities(X, U, t))
        c6 = jnp.dot(Z, inequalities(X, U, t) + S)
        return c1 + c2 + c3 + c4 + c5 + c6

    return lagrangian


def build_total_augmented_lagrangian(  # noqa: ANN201
    cost: CostFunction,
    dynamics: Function,
    equalities: Function,
    inequalities: Function,
    x0: jnp.ndarray,
    params: Parameters,
    T: jnp.int32,
):
    """Return a function to evaluate the associated Augmented Lagrangian."""
    lagrangian = build_lagrangian(
        cost=cost,
        dynamics=dynamics,
        equalities=equalities,
        inequalities=inequalities,
        x0=x0,
        µ=params.µ,
    )

    T_range = jnp.arange(T)
    Tp1_range = jnp.arange(T + 1)

    def augmented_lagrangian(
        vars: Variables,
    ) -> jnp.double:
        U_pad = pad(vars.U)
        next_Y_dyn = pad(vars.Y_dyn[1:])
        c1 = jnp.sum(
            jax.vmap(
                lambda t: lagrangian(
                    X=vars.X[t],
                    U=U_pad[t],
                    t=t,
                    S=vars.S[t],
                    next_Y_dyn=next_Y_dyn[t],
                    Y_dyn=vars.Y_dyn[t],
                    Y_eq=vars.Y_eq[t],
                    Z=vars.Z[t],
                )
            )(Tp1_range)
        )
        c2 = 0.5 * jnp.sum(
            params.η_eq * jnp.square(jax.vmap(equalities)(vars.X, U_pad, Tp1_range))
        )
        c3 = 0.5 * jnp.sum(
            params.η_ineq
            * jnp.square(jax.vmap(inequalities)(vars.X, U_pad, Tp1_range) + vars.S)
        )
        c4 = 0.5 * jnp.sum(
            params.η_dyn[1:]
            * jnp.square(jax.vmap(dynamics)(vars.X[:-1], vars.U, T_range) - vars.X[1:])
        )
        c5 = 0.5 * jnp.sum(params.η_dyn[0] * jnp.square(x0 - vars.X[0]))
        return c1 + c2 + c3 + c4 + c5

    return augmented_lagrangian


def directional_augmented_lagrangian(  # noqa: ANN201
    cost: CostFunction,
    dynamics: Function,
    equalities: Function,
    inequalities: Function,
    x0: jnp.ndarray,
    params: Parameters,
    τ: jnp.double,
    T: jnp.int32,
    vars: Variables,
    deltas: Variables,
):
    """Define the directional Augmented Lagrangian used in the line search."""
    augmented_lagrangian = build_total_augmented_lagrangian(
        cost=cost,
        dynamics=dynamics,
        equalities=equalities,
        inequalities=inequalities,
        x0=x0,
        params=params,
        T=T,
    )

    def dal(α: jnp.double) -> jnp.double:
        return augmented_lagrangian(
            Variables(
                X=(vars.X + α * deltas.X),
                U=(vars.U + α * deltas.U),
                S=jnp.maximum(vars.S + α * deltas.S, (1.0 - τ) * vars.S),
                Y_dyn=vars.Y_dyn,
                Y_eq=vars.Y_eq,
                Z=vars.Z,
            )
        )

    return dal
