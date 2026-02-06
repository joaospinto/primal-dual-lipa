"""Add transcription helpers."""

import jax
from jax import numpy as jnp

from primal_dual_lipa.types import Function


def euler(dynamics: Function, dt: jnp.double) -> Function:
    """Apply Euler transcription."""
    return lambda x, u, t: x + dt * dynamics(x, u, t)


def midpoint(dynamics: Function, dt: jnp.double) -> Function:
    """Apply midpoint transcription."""

    def integrator(x: jnp.ndarray, u: jnp.ndarray, t: jnp.double) -> jnp.ndarray:
        dt2 = 0.5 * dt
        k1 = dynamics(x, u, t)
        k2 = dynamics(x + dt2 * k1, u, t + dt2)
        return x + dt * k2

    return integrator


def rk4(dynamics: Function, dt: jnp.double) -> Function:
    """Apply Runge-Kutta 4 transcription."""

    def integrator(x: jnp.ndarray, u: jnp.ndarray, t: jnp.double) -> jnp.ndarray:
        dt2 = 0.5 * dt
        k1 = dynamics(x, u, t)
        k2 = dynamics(x + dt2 * k1, u, t + dt2)
        k3 = dynamics(x + dt2 * k2, u, t + dt2)
        k4 = dynamics(x + dt * k3, u, t + dt)
        return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    return integrator


def rollout(dynamics: Function, U: jnp.ndarray, x0: jnp.ndarray) -> jnp.ndarray:
    """Rollout the dynamics for a control sequence."""

    def dynamics_for_scan(
        x: jnp.ndarray, ut: tuple[jnp.ndarray, jnp.int32]
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        u, t = ut
        x_next = dynamics(x, u, t)
        return x_next, x_next

    return jnp.vstack(
        (x0, jax.lax.scan(dynamics_for_scan, x0, (U, jnp.arange(U.shape[0])))[1])
    )
