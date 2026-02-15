"""Test the KKT helpers."""

import unittest

import jax
from jax import numpy as jnp
from regularized_lqr_jax.helpers import regularize, symmetrize

from primal_dual_lipa.kkt_helpers import compute_kkt_residual, solve_kkt

jax.config.update("jax_enable_x64", True)  # noqa: FBT003


class TestKKTSolves(unittest.TestCase):
    """Test the KKT helpers."""

    def setUp(self) -> None:
        """Set up the unit test."""
        n = 4
        m = 2
        T = 30

        c_dim = 3
        g_dim = 3

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
        self.r_x = jax.random.uniform(subkey, (T + 1, n + m))
        self.r_x = self.r_x.at[-1, n:].set(0.0)

        key, subkey = jax.random.split(key)
        self.r_s = jax.random.uniform(subkey, (T + 1, g_dim))

        key, subkey = jax.random.split(key)
        self.r_y_dyn = jax.random.uniform(subkey, (T + 1, n))

        key, subkey = jax.random.split(key)
        self.r_y_eq = jax.random.uniform(subkey, (T + 1, c_dim))

        key, subkey = jax.random.split(key)
        self.r_z = jax.random.uniform(subkey, (T + 1, g_dim))

        key, subkey = jax.random.split(key)
        self.η = jnp.abs(jax.random.uniform(subkey, (1,)))[0]

    def test(self) -> None:
        """Run the test."""
        for use_parallel_lqr in [False, True]:
            with self.subTest(use_parallel_lqr=use_parallel_lqr):
                dX, dU, dS, dY_dyn, dY_eq, dZ = solve_kkt(
                    P=self.P,
                    D=self.D,
                    E=self.E,
                    G=self.G,
                    w_inv=self.w_inv,
                    r_x=self.r_x,
                    r_s=self.r_s,
                    r_y_dyn=self.r_y_dyn,
                    r_y_eq=self.r_y_eq,
                    r_z=self.r_z,
                    η=self.η,
                    use_parallel_lqr=use_parallel_lqr,
                )

                res_X, res_U, res_S, res_Y_dyn, res_Y_eq, res_Z = compute_kkt_residual(
                    P=self.P,
                    D=self.D,
                    E=self.E,
                    G=self.G,
                    w_inv=self.w_inv,
                    r_x=self.r_x,
                    r_s=self.r_s,
                    r_y_dyn=self.r_y_dyn,
                    r_y_eq=self.r_y_eq,
                    r_z=self.r_z,
                    dX=dX,
                    dU=dU,
                    dS=dS,
                    dY_dyn=dY_dyn,
                    dY_eq=dY_eq,
                    dZ=dZ,
                    η=self.η,
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

                if use_parallel_lqr:
                    self.assertLess(jnp.linalg.norm(residual), 1e-3)  # noqa: PT009
                else:
                    self.assertLess(jnp.linalg.norm(residual), 1e-9)  # noqa: PT009


if __name__ == "__main__":
    unittest.main()
