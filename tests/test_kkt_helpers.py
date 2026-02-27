"""Test the KKT helpers."""

import unittest

import jax
from jax import numpy as jnp
from regularized_lqr_jax.helpers import regularize, symmetrize

from primal_dual_lipa.kkt_helpers import (
    compute_kkt_residual,
    factor_kkt,
    solve_kkt,
)
from primal_dual_lipa.types import (
    KKTFactorizationInputs,
    Parameters,
    Variables,
)

jax.config.update("jax_enable_x64", True)  # noqa: FBT003


class TestKKTSolves(unittest.TestCase):
    """Test solving the KKT system."""

    def setUp(self) -> None:
        """Set up the unit test."""
        n = 4
        m = 2
        T = 30

        c_dim = 3
        g_dim = 3
        p_dim = 2

        self.T = T

        key = jax.random.PRNGKey(0)

        key, subkey = jax.random.split(key)
        self.D = jax.random.uniform(subkey, (T, n, n + m))

        key, subkey = jax.random.split(key)
        self.E = jax.random.uniform(subkey, (T + 1, c_dim, n + m))
        self.E = self.E.at[-1, :, n:].set(0.0)

        key, subkey = jax.random.split(key)
        self.G = jax.random.uniform(subkey, (T + 1, g_dim, n + m))
        self.G = self.G.at[-1, :, n:].set(0.0)

        key, subkey = jax.random.split(key)
        Q = jax.random.uniform(subkey, (T + 1, n, n))
        Q = jax.vmap(symmetrize)(Q)

        key, subkey = jax.random.split(key)
        M = jax.random.uniform(subkey, (T, n, m))

        key, subkey = jax.random.split(key)
        R = jax.random.uniform(subkey, (T, m, m))
        R = jax.vmap(symmetrize)(R)

        Q, R = regularize(Q=Q, R=R, M=M, psd_delta=1e-3)

        M = jnp.concatenate([M, jnp.zeros_like(M[0])[None, ...]], axis=0)

        R = jnp.concatenate([R, jnp.eye(m)[None, ...] * 1e-3], axis=0)

        self.P = jax.vmap(lambda q, m, r: jnp.block([[q, m], [m.T, r]]))(Q, M, R)

        key, subkey = jax.random.split(key)
        s = jnp.abs(jax.random.uniform(subkey, (T + 1, g_dim)))

        key, subkey = jax.random.split(key)
        z = jnp.abs(jax.random.uniform(subkey, (T + 1, g_dim)))

        self.w_inv = z / s

        key, subkey = jax.random.split(key)
        self.r_x = jax.random.uniform(subkey, (T + 1, n))

        key, subkey = jax.random.split(key)
        self.r_u = jax.random.uniform(subkey, (T, m))

        key, subkey = jax.random.split(key)
        self.r_s = jax.random.uniform(subkey, (T + 1, g_dim))

        key, subkey = jax.random.split(key)
        self.r_y_dyn = jax.random.uniform(subkey, (T + 1, n))

        key, subkey = jax.random.split(key)
        self.r_y_eq = jax.random.uniform(subkey, (T + 1, c_dim))

        key, subkey = jax.random.split(key)
        self.r_z = jax.random.uniform(subkey, (T + 1, g_dim))

        # Global parameter initializations
        key, subkey = jax.random.split(key)
        self.Theta = jax.random.uniform(subkey, (p_dim,))
        self.r_theta = jax.random.uniform(subkey, (p_dim,))
        self.H_theta_theta = jax.random.uniform(subkey, (p_dim, p_dim))
        self.H_theta_theta = symmetrize(self.H_theta_theta) + jnp.eye(p_dim)

        self.H_theta_X = jax.random.uniform(subkey, (T + 1, n, p_dim))
        self.H_theta_U = jax.random.uniform(subkey, (T, m, p_dim))
        self.H_theta_y_dyn = jax.random.uniform(subkey, (T + 1, n, p_dim))
        self.H_theta_y_eq = jax.random.uniform(subkey, (T + 1, c_dim, p_dim))
        self.H_theta_z = jax.random.uniform(subkey, (T + 1, g_dim, p_dim))

        self.parameters = Parameters(
            η_dyn=jnp.ones((T + 1, n)) * 10.0,
            η_eq=jnp.ones((T + 1, c_dim)) * 10.0,
            η_ineq=jnp.ones((T + 1, g_dim)) * 10.0,
            µ=0.1,
        )

    def test(self) -> None:
        """Run the test."""
        for use_parallel_lqr in [False, True]:
            with self.subTest(use_parallel_lqr=use_parallel_lqr):
                factorization_inputs = KKTFactorizationInputs(
                    P=self.P,
                    D=self.D,
                    E=self.E,
                    G=self.G,
                    w_inv=self.w_inv,
                    params=self.parameters,
                    H_theta_theta=self.H_theta_theta,
                    H_theta_X=self.H_theta_X,
                    H_theta_U=self.H_theta_U,
                    H_theta_y_dyn=self.H_theta_y_dyn,
                    H_theta_y_eq=self.H_theta_y_eq,
                    H_theta_z=self.H_theta_z,
                )
                solve_inputs = Variables(
                    X=self.r_x,
                    U=self.r_u,
                    S=self.r_s,
                    Y_dyn=self.r_y_dyn,
                    Y_eq=self.r_y_eq,
                    Z=self.r_z,
                    Theta=self.r_theta,
                )

                factorization_outputs = factor_kkt(
                    inputs=factorization_inputs,
                    use_parallel_lqr=use_parallel_lqr,
                )

                deltas = solve_kkt(
                    factorization_outputs=factorization_outputs,
                    factorization_inputs=factorization_inputs,
                    rhs=solve_inputs,
                    use_parallel_lqr=use_parallel_lqr,
                )

                residuals = compute_kkt_residual(
                    factorization_inputs=factorization_inputs,
                    solve_inputs=solve_inputs,
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

                if use_parallel_lqr:
                    self.assertLess(jnp.linalg.norm(residual_vec), 1e-3)  # noqa: PT009
                else:
                    self.assertLess(jnp.linalg.norm(residual_vec), 1e-9)  # noqa: PT009


if __name__ == "__main__":
    unittest.main()
