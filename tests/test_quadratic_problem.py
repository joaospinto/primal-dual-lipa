"""Unit test for a simple linear quadratic regulator problem."""

import unittest

import jax
from jax import numpy as jnp

jax.config.update("jax_enable_x64", True)  # noqa: FBT003

from primal_dual_lipa.optimizers import SolverSettings, solve
from primal_dual_lipa.types import Variables


class TestLqrSolve(unittest.TestCase):
    """Solve a simple LQR problem."""

    def test(self) -> None:
        """Run the test."""

        @jax.jit
        def dynamics(
            x: jnp.ndarray, u: jnp.ndarray, theta: jnp.ndarray, t: jnp.int32
        ) -> jnp.ndarray:
            del theta
            A = 6 * t * jnp.ones([2, 2]) + jnp.arange(4).reshape([2, 2])
            B = 6 * t * jnp.ones([2, 1]) + jnp.arange(4, 6).reshape([2, 1])
            c = 6 * t * jnp.ones(2) + jnp.arange(6, 8)
            return A @ x + B @ u + c

        @jax.jit
        def cost(
            x: jnp.ndarray, u: jnp.ndarray, theta: jnp.ndarray, t: jnp.int32
        ) -> jnp.double:
            del theta
            Q = (t + 1) * jnp.diag(jnp.arange(1, 3))
            R = (t + 1) * jnp.array([3.0]).reshape([1, 1])
            q = -(t + 1) * jnp.arange(91, 93)
            r = -(t + 1) * jnp.array([99.0])
            M = (t + 1) / 100.0 * jnp.arange(55, 57).reshape([2, 1])
            return (
                0.5 * x.T @ Q @ x + 0.5 * u.T @ R @ u + x.T @ M @ u + q.T @ x + r.T @ u
            )

        x0 = jnp.array([-11.0, -22.0])

        T = 2
        n = 2
        m = 1
        c_dim = 0
        g_dim = 0

        X = jnp.zeros([T + 1, n])
        U = jnp.zeros([T, m])
        S = jnp.zeros([T + 1, g_dim])
        Y_dyn = jnp.zeros_like(X)
        Y_eq = jnp.zeros([T + 1, c_dim])
        Z = jnp.zeros([T + 1, g_dim])

        vars_in = Variables(
            X=X, U=U, S=S, Y_dyn=Y_dyn, Y_eq=Y_eq, Z=Z, Theta=jnp.empty(0)
        )

        settings = SolverSettings(print_logs=True)

        print("Quadratic problem")  # noqa: T201
        vars_out, iterations, no_errors = solve(
            vars_in=vars_in,
            x0=x0,
            cost=cost,
            dynamics=dynamics,
            settings=settings,
        )
        self.assertTrue(no_errors)  # noqa: PT009


if __name__ == "__main__":
    jnp.set_printoptions(threshold=1000000)
    jnp.set_printoptions(linewidth=1000000)
    unittest.main()
