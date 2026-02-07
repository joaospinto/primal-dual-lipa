# From https://github.com/google/trajax/blob/main/benchmarks/ilqr_benchmark.py

"""Test goal-reaching with cartpole."""

import unittest
from functools import partial

import jax
from jax import numpy as jnp
from jax import scipy as jsp

from primal_dual_lipa.integrators import euler, rollout
from primal_dual_lipa.lagrangian_helpers import pad
from primal_dual_lipa.optimizers import SolverSettings, solve

jax.config.update("jax_enable_x64", True)  # noqa: FBT003


@jax.jit
def cartpole(
    state: jnp.ndarray,
    action: jnp.ndarray,
    timestep: jnp.double,
    params: jnp.ndarray,
) -> jnp.ndarray:
    """Classic cartpole system.

    Args:
      state: state, (4, ) array
      action: control, (1, ) array
      timestep: scalar time
      params: tuple of (MASS_CART, MASS_POLE, LENGTH_POLE)

    Returns:
      xdot: state time derivative, (4, )

    """
    del timestep  # Unused

    mc, mp, l = params  # noqa: E741
    g = 9.81

    q = state[0:2]
    qd = state[2:]
    s = jnp.sin(q[1])
    c = jnp.cos(q[1])

    H = jnp.array([[mc + mp, mp * l * c], [mp * l * c, mp * l * l]])
    C = jnp.array([[0.0, -mp * qd[1] * l * s], [0.0, 0.0]])

    G = jnp.array([[0.0], [mp * g * l * s]])
    B = jnp.array([[1.0], [0.0]])

    CqdG = jnp.dot(C, jnp.expand_dims(qd, 1)) + G
    f = jnp.concatenate((qd, jnp.squeeze(-jsp.linalg.solve(H, CqdG, assume_a="pos"))))

    v = jnp.squeeze(jsp.linalg.solve(H, B, assume_a="pos"))
    g = jnp.concatenate((jnp.zeros(2), v))

    return f + g * action


def angle_wrap(th: jnp.double) -> jnp.double:
    """Wrap the input angle."""
    return (th) % (2 * jnp.pi)


def state_wrap(s: jnp.ndarray) -> jnp.ndarray:
    """Wrap the angles in the state."""
    return jnp.array([s[0], angle_wrap(s[1]), s[2], s[3]])


@jax.jit
def goal_cost(
    x: jnp.ndarray, u: jnp.ndarray, t: jnp.int32, goal: jnp.int32, T: jnp.int32
) -> jnp.double:
    """Define the cost."""
    err = state_wrap(x - goal)
    stage_cost = 0.1 * jnp.dot(err, err) + 0.01 * jnp.dot(u, u)
    final_cost = 1000 * jnp.dot(err, err)
    return jnp.where(t == T, final_cost, stage_cost)


@jax.jit
def goal_equality(
    x: jnp.ndarray, _u: jnp.ndarray, t: jnp.int32, goal: jnp.int32, T: jnp.int32
) -> jnp.ndarray:
    """Define the final state constraint."""
    return jnp.where(t == T, state_wrap(x - goal), jnp.zeros_like(x))


@jax.jit
def inequalities(_x: jnp.ndarray, u: jnp.ndarray, _t: jnp.int32) -> jnp.ndarray:
    """Define the control bounds."""
    return jnp.array([u[0] - 5.0, -5.0 - u[0]])


class TestCartpole(unittest.TestCase):
    """Define the Cartpole test."""

    def test(self) -> None:
        """Run the test."""
        T = 50
        n = 4
        m = 1
        c_dim = n
        g_dim = 2

        x0 = jnp.array([0.0, 0.2, 0.0, -0.1])

        dynamics_params = jnp.array([10.0, 1.0, 0.5])
        dynamics = partial(cartpole, params=dynamics_params)

        dynamics = euler(dynamics, dt=0.1)

        goal = jnp.array([0, jnp.pi, 0, 0])
        cost = partial(goal_cost, goal=goal, T=T)

        equalities = partial(goal_equality, goal=goal, T=T)

        U = jnp.zeros([T, m])
        # X = rollout(dynamics, U, x0)
        X = jnp.linspace(start=x0, stop=goal, num=T + 1)
        S = jnp.zeros([T + 1, g_dim])
        Y_dyn = jnp.zeros_like(X)
        Y_eq = jnp.zeros([T + 1, c_dim])
        Z = jnp.zeros([T + 1, g_dim])

        # TODO(joao): only change print_logs, if possible.
        settings = SolverSettings(
            µ_update_factor=0.99,
            η_update_factor=1.1,
            print_logs=True,
        )

        print("Cartpole problem")  # noqa: T201
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
            equalities=equalities,
            inequalities=inequalities,
            settings=settings,
        )
        self.assertTrue(no_errors)  # noqa: PT009
        # TODO(joao): define the right value.
        self.assertLess(jax.vmap(cost)(X, pad(U), jnp.arange(T + 1)).sum(), 102.0)  # noqa: PT009


if __name__ == "__main__":
    jnp.set_printoptions(threshold=1000000)
    jnp.set_printoptions(linewidth=1000000)
    unittest.main()
