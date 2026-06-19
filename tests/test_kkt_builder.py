"""Test the KKT builders and regularization logic."""

import unittest

import jax
from jax import numpy as jnp
from regularized_lqr_jax.helpers import symmetrize

from primal_dual_lipa.kkt_builder import (
    add_scalar_hessian_regularization_delta,
    regularize_primal_hessian_blocks,
)
from primal_dual_lipa.types import KKTFactorizationInputs, KKTSystem, Parameters, Variables

jax.config.update("jax_enable_x64", True)  # noqa: FBT003


class TestKKTRegularization(unittest.TestCase):
    """Test the robust regularization of the KKT Hessian."""

    def test_primal_hessian_regularization_shifts_only_primal_blocks(self) -> None:
        """Verify IPOPT-style scalar regularization changes only primal blocks."""
        n = 3
        m = 2
        p = 2
        T = 4
        delta = 1e-2

        key = jax.random.PRNGKey(7)
        key, *subkeys = jax.random.split(key, 4)
        Q_raw = jax.random.normal(subkeys[0], (T + 1, n, n))
        R_raw = jax.random.normal(subkeys[1], (T, m, m))
        H_theta_raw = jax.random.normal(subkeys[2], (T + 1, p, p))

        Q_reg, R_reg, H_theta_reg = regularize_primal_hessian_blocks(
            Q=Q_raw,
            R=R_raw,
            H_theta_theta_per_stage=H_theta_raw,
            hessian_regularization=delta,
        )

        Q_expected = jax.vmap(symmetrize)(Q_raw) + delta * jnp.eye(n)
        R_expected = jax.vmap(symmetrize)(R_raw) + delta * jnp.eye(m)
        H_theta_expected = jax.vmap(symmetrize)(H_theta_raw)

        self.assertTrue(jnp.allclose(Q_reg, Q_expected))  # noqa: PT009
        self.assertTrue(jnp.allclose(R_reg, R_expected))  # noqa: PT009
        self.assertTrue(jnp.allclose(H_theta_reg, H_theta_expected))  # noqa: PT009

    def test_scalar_regularization_delta_shifts_cached_primal_blocks(self) -> None:
        """Verify retry regularization updates cached raw/LQR/theta blocks only."""
        T = 2
        n = 2
        m = 1
        p = 2
        c_dim = 1
        g_dim = 1
        delta = 0.3
        xu_dim = n + m

        params = Parameters(
            µ=0.1,
            η_dyn=jnp.ones((T + 1, n)),
            η_eq=jnp.ones((T + 1, c_dim)),
            η_ineq=jnp.ones((T + 1, g_dim)),
        )
        lhs = KKTFactorizationInputs(
            P=jnp.zeros((T + 1, xu_dim, xu_dim)),
            P_lqr=jnp.ones((T + 1, xu_dim, xu_dim)),
            D=jnp.zeros((T, n, xu_dim)),
            E=jnp.zeros((T + 1, c_dim, xu_dim)),
            G=jnp.zeros((T + 1, g_dim, xu_dim)),
            w_inv=jnp.ones((T + 1, g_dim)),
            params=params,
            H_theta_theta=jnp.zeros((p, p)),
            H_theta_X=jnp.zeros((T + 1, n, p)),
            H_theta_U=jnp.zeros((T, m, p)),
            H_theta_y_dyn=jnp.zeros((T + 1, n, p)),
            H_theta_y_eq=jnp.zeros((T + 1, c_dim, p)),
            H_theta_z=jnp.zeros((T + 1, g_dim, p)),
        )
        rhs = Variables(
            X=jnp.zeros((T + 1, n)),
            U=jnp.zeros((T, m)),
            S=jnp.zeros((T + 1, g_dim)),
            Y_dyn=jnp.zeros((T + 1, n)),
            Y_eq=jnp.zeros((T + 1, c_dim)),
            Z=jnp.zeros((T + 1, g_dim)),
            Theta=jnp.zeros((p,)),
        )

        shifted = add_scalar_hessian_regularization_delta(
            KKTSystem(lhs=lhs, rhs=rhs), delta
        )

        expected_p_shift = delta * jnp.eye(xu_dim)[None, ...]
        self.assertTrue(jnp.allclose(shifted.lhs.P, lhs.P + expected_p_shift))  # noqa: PT009
        self.assertTrue(jnp.allclose(shifted.lhs.P_lqr, lhs.P_lqr + expected_p_shift))  # noqa: PT009
        self.assertTrue(  # noqa: PT009
            jnp.allclose(shifted.lhs.H_theta_theta, delta * jnp.eye(p))
        )
        self.assertTrue(jnp.allclose(shifted.lhs.E, lhs.E))  # noqa: PT009
        self.assertTrue(jnp.allclose(shifted.lhs.G, lhs.G))  # noqa: PT009

if __name__ == "__main__":
    unittest.main()
