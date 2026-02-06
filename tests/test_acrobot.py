# From https://github.com/google/trajax/blob/main/tests/optimizers_test.py

"""Test goal-reaching with Acrobot."""

import unittest
from functools import partial

import jax
from jax import numpy as jnp

from primal_dual_lipa.integrators import euler
from primal_dual_lipa.lagrangian_helpers import pad
from primal_dual_lipa.optimizers import SolverSettings, solve

jax.config.update("jax_enable_x64", True)  # noqa: FBT003


@jax.jit
def acrobot(
    x: jnp.ndarray, u: jnp.ndarray, t: jnp.double, params: jnp.ndarray
) -> jnp.ndarray:
    """Classic Acrobot system.

    Note this implementation emulates the OpenAI gym implementation of
    Acrobot-v2, which itself is based on Stutton's Reinforcement Learning book.

    Args:
      x: state, (4, ) array
      u: control, (1, ) array
      t: scalar time. Disregarded because system is time-invariant.
      params: tuple of (LINK_MASS_1, LINK_MASS_2, LINK_LENGTH_1, LINK_COM_POS_1,
        LINK_COM_POS_2 LINK_MOI_1, LINK_MOI_2)

    Returns:
      xdot: state time derivative, (4, )

    """
    del t  # Unused

    m1, m2, l1, lc1, lc2, I1, I2 = params
    g = 9.8
    a = u[0]
    theta1 = x[0]
    theta2 = x[1]
    dtheta1 = x[2]
    dtheta2 = x[3]
    d1 = m1 * lc1**2 + m2 * (l1**2 + lc2**2 + 2 * l1 * lc2 * jnp.cos(theta2)) + I1 + I2
    d2 = m2 * (lc2**2 + l1 * lc2 * jnp.cos(theta2)) + I2
    phi2 = m2 * lc2 * g * jnp.cos(theta1 + theta2 - jnp.pi / 2.0)
    phi1 = (
        -m2 * l1 * lc2 * dtheta2**2 * jnp.sin(theta2)
        - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * jnp.sin(theta2)
        + (m1 * lc1 + m2 * l1) * g * jnp.cos(theta1 - jnp.pi / 2)
        + phi2
    )
    ddtheta2 = (
        a + d2 / d1 * phi1 - m2 * l1 * lc2 * dtheta1**2 * jnp.sin(theta2) - phi2
    ) / (m2 * lc2**2 + I2 - d2**2 / d1)
    ddtheta1 = -(d2 * ddtheta2 + phi1) / d1
    return jnp.array([dtheta1, dtheta2, ddtheta1, ddtheta2])


@jax.jit
def goal_cost(
    x: jnp.ndarray,
    u: jnp.ndarray,
    t: jnp.int32,
    params: jnp.ndarray,
    goal: jnp.ndarray,
    T: jnp.int32,
) -> jnp.double:
    """Define the cost function."""
    delta = x - goal
    terminal_cost = 0.5 * params[0] * jnp.dot(delta, delta)
    stagewise_cost = 0.5 * params[1] * jnp.dot(delta, delta) + 0.5 * params[
        2
    ] * jnp.dot(u, u)
    return jnp.where(t == T, terminal_cost, stagewise_cost)


class TestAcrobot(unittest.TestCase):
    """Define the Acrobot test."""

    def test(self) -> None:
        """Run the test."""
        T = 50
        n = 4
        m = 1
        c_dim = 0
        g_dim = 0

        goal = jnp.array([jnp.pi, 0.0, 0.0, 0.0])

        dynamics_params = jnp.array([1.0, 1.0, 1.0, 0.5, 0.5, 1.0, 1.0])
        cost_params = jnp.array([1000.0, 0.1, 0.01])

        dynamics = partial(acrobot, params=dynamics_params)
        dynamics = euler(dynamics, dt=0.1)

        cost = partial(goal_cost, params=cost_params, goal=goal, T=T)

        X = jnp.zeros([T + 1, n])
        U = jnp.zeros([T, m])
        S = jnp.zeros([T + 1, g_dim])
        Y_dyn = jnp.zeros_like(X)
        Y_eq = jnp.zeros([T + 1, c_dim])
        Z = jnp.zeros([T + 1, g_dim])

        x0 = jnp.zeros(n)

        settings = SolverSettings(print_logs=True)

        print("Acrobot problem")  # noqa: T201
        X, U, S, Y_dyn, Y_eq, Z, iterations, no_errors = solve(
            X_in=X,
            U_in=U,
            S_in=S,
            Y_dyn_in=Y_dyn,
            Y_eq_in=Y_eq,
            Z_in=Z,
            x0=x0,
            cost=cost,
            dynamics=dynamics,
            settings=settings,
        )
        self.assertTrue(no_errors)  # noqa: PT009
        self.assertLess(jax.vmap(cost)(X, pad(U), jnp.arange(T + 1)).sum(), 45.0)  # noqa: PT009


if __name__ == "__main__":
    jnp.set_printoptions(threshold=1000000)
    jnp.set_printoptions(linewidth=1000000)
    unittest.main()
