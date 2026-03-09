"""Add transcription helpers."""

import jax
from jax import numpy as jnp

from primal_dual_lipa.types import Function


def euler(dynamics: Function, dt: jnp.double) -> Function:
    """Apply Euler transcription."""
    return lambda x, u, theta, t: x + dt * dynamics(x, u, theta, t)


def midpoint(dynamics: Function, dt: jnp.double) -> Function:
    """Apply midpoint transcription."""

    def integrator(
        x: jax.Array, u: jax.Array, theta: jax.Array, t: jnp.double
    ) -> jax.Array:
        dt2 = 0.5 * dt
        k1 = dynamics(x, u, theta, t)
        k2 = dynamics(x + dt2 * k1, u, theta, t + dt2)
        return x + dt * k2

    return integrator


def rk4(dynamics: Function, dt: jnp.double) -> Function:
    """Apply Runge-Kutta 4 transcription."""

    def integrator(
        x: jax.Array, u: jax.Array, theta: jax.Array, t: jnp.double
    ) -> jax.Array:
        dt2 = 0.5 * dt
        k1 = dynamics(x, u, theta, t)
        k2 = dynamics(x + dt2 * k1, u, theta, t + dt2)
        k3 = dynamics(x + dt2 * k2, u, theta, t + dt2)
        k4 = dynamics(x + dt * k3, u, theta, t + dt)
        return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    return integrator


def rollout(
    dynamics: Function, U: jax.Array, x0: jax.Array, theta: jax.Array
) -> jax.Array:
    """Rollout the dynamics for a control sequence."""

    def dynamics_for_scan(
        x: jax.Array, ut: tuple[jax.Array, jnp.int32]
    ) -> tuple[jax.Array, jax.Array]:
        u, t = ut
        x_next = dynamics(x, u, theta, t)
        return x_next, x_next

    return jnp.vstack(
        (
            x0,
            jax.lax.scan(dynamics_for_scan, x0, (U, jnp.arange(U.shape[0])))[1],
        )
    )
