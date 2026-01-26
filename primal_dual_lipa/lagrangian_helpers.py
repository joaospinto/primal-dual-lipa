"""Provides helper methods for computing the Lagrangian and Augmented Lagrangian."""

import jax
from jax import numpy as jnp

from primal_dual_lipa.types import Function


@jax.jit
def pad(A: jnp.ndarray) -> jnp.ndarray:
    """Pads with zeros along the first axis by an extra element."""
    return jnp.pad(A, [[0, 1], [0, 0]])


@jax.jit
def build_lagrangian(  # noqa: ANN201
    cost: Function,
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
        S: jnp.ndarray,
        next_Y_dyn: jnp.ndarray,
        Y_dyn: jnp.ndarray,
        Y_eq: jnp.ndarray,
        Z: jnp.ndarray,
        t: jnp.int32,
    ) -> jnp.ndarray:
        c1 = cost(X, U, t)
        c2 = jnp.dot(next_Y_dyn, dynamics(X, U, t))
        c3 = jnp.dot(Y_dyn, jax.lax.select(t == 0, x0 - X, -X))
        c4 = -µ * jnp.sum(jnp.log(S))
        c5 = jnp.dot(Y_eq, equalities(X, U, t))
        c6 = jnp.dot(Z, inequalities(X, U, t) + S)
        return c1 + c2 + c3 + c4 + c5 + c6

    return lagrangian


@jax.jit
def build_total_augmented_lagrangian(  # noqa: ANN201
    cost: Function,
    dynamics: Function,
    equalities: Function,
    inequalities: Function,
    x0: jnp.ndarray,
    µ: jnp.double,
    η: jnp.double,
    T: jnp.int32,
):
    """Return a function to evaluate the associated Augmented Lagrangian."""
    lagrangian = build_lagrangian(
        cost=cost,
        dynamics=dynamics,
        equalities=equalities,
        inequalities=inequalities,
        x0=x0,
        µ=µ,
    )

    T_range = jnp.arange(T + 1)

    def augmented_lagrangian(
        X: jnp.ndarray,
        U: jnp.ndarray,
        S: jnp.ndarray,
        Y_dyn: jnp.ndarray,
        Y_eq: jnp.ndarray,
        Z: jnp.ndarray,
    ) -> jnp.ndarray:
        next_Y_dyn = pad(Y_dyn[1:])
        next_X = pad(X[1:])
        c1 = jnp.sum(
            jax.vmap(
                lambda t: lagrangian(
                    X=X[t],
                    U=U[t],
                    S=S[t],
                    next_Y_dyn=next_Y_dyn[t],
                    Y_dyn=Y_dyn[t],
                    Y_eq=Y_eq[t],
                    Z=Z[t],
                    t=t,
                )
            )(T_range)
        )
        c2 = 0.5 * η * jnp.sum(jnp.square(jax.vmap(equalities)(X, U, T_range)))
        c3 = 0.5 * η * jnp.sum(jnp.square(jax.vmap(inequalities)(X, U, T_range) + S))
        c4 = 0.5 * η * jnp.sum(jnp.square(jax.vmap(dynamics)(X, U, T_range) - next_X))
        c5 = 0.5 * η * jnp.sum(jnp.square(x0 - X[0]))
        return c1 + c2 + c3 + c4 + c5

    return augmented_lagrangian


@jax.jit
def directional_augmented_lagrangian(  # noqa: ANN201
    cost: Function,
    dynamics: Function,
    equalities: Function,
    inequalities: Function,
    x0: jnp.ndarray,
    µ: jnp.double,
    η: jnp.double,
    τ: jnp.double,
    T: jnp.int32,
    X: jnp.ndarray,
    U: jnp.ndarray,
    S: jnp.ndarray,
    Y_dyn: jnp.ndarray,
    Y_eq: jnp.ndarray,
    Z: jnp.ndarray,
    dX: jnp.ndarray,
    dU: jnp.ndarray,
    dS: jnp.ndarray,
):
    """Define the directional Augmented Lagrangian used in the line search."""
    augmented_lagrangian = build_total_augmented_lagrangian(
        cost=cost,
        dynamics=dynamics,
        equalities=equalities,
        inequalities=inequalities,
        x0=x0,
        µ=µ,
        η=η,
        T=T,
    )

    def dal(α: jnp.double) -> jnp.double:
        return augmented_lagrangian(
            X=(X + α * dX),
            U=(U + α * dU),
            S=jnp.maximum(S + α * dS, (1.0 - τ) * S),
            Y_dyn=Y_dyn,
            Y_eq=Y_eq,
            Z=Z,
        )

    return dal
