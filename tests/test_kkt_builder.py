"""Test the KKT builders and regularization logic."""

import unittest

import jax
from jax import numpy as jnp
from regularized_lqr_jax.helpers import symmetrize

from primal_dual_lipa.kkt_builder import (
    add_scalar_hessian_regularization_delta,
    regularize_primal_hessian_blocks,
)
from primal_dual_lipa.types import (
    KKTFactorizationInputs,
    KKTSystem,
    NodeAndEdgeIndices,
    NodeAndEdgeValues,
    TreeParameters,
    TreeVariables,
)

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
        H_theta_raw = jax.random.normal(subkeys[2], (p, p))

        Q_reg, R_reg, H_theta_reg = regularize_primal_hessian_blocks(
            Q=Q_raw,
            R=R_raw,
            H_theta_theta=H_theta_raw,
            hessian_regularization=delta,
        )

        Q_expected = jax.vmap(symmetrize)(Q_raw) + delta * jnp.eye(n)
        R_expected = jax.vmap(symmetrize)(R_raw) + delta * jnp.eye(m)
        H_theta_expected = symmetrize(H_theta_raw) + delta * jnp.eye(p)

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

        params = TreeParameters(
            µ=0.1,
            η_dyn=jnp.ones((T + 1, n)),
            η_eq=NodeAndEdgeValues(
                node=jnp.ones((T + 1, c_dim)), edge=jnp.ones((T, c_dim))
            ),
            η_ineq=NodeAndEdgeValues(
                node=jnp.ones((T + 1, g_dim)), edge=jnp.ones((T, g_dim))
            ),
        )
        lhs = KKTFactorizationInputs(
            Q=jnp.zeros((T + 1, n, n)),
            M=jnp.zeros((T, n, m)),
            R=jnp.zeros((T, m, m)),
            Q_lqr=jnp.ones((T + 1, n, n)),
            M_lqr=jnp.ones((T, n, m)),
            R_lqr=jnp.ones((T, m, m)),
            D=jnp.zeros((T, n, xu_dim)),
            E=NodeAndEdgeValues(
                node=jnp.zeros((T + 1, c_dim, n)),
                edge=jnp.zeros((T, c_dim, xu_dim)),
            ),
            G=NodeAndEdgeValues(
                node=jnp.zeros((T + 1, g_dim, n)),
                edge=jnp.zeros((T, g_dim, xu_dim)),
            ),
            w_inv=NodeAndEdgeValues(
                node=jnp.ones((T + 1, g_dim)), edge=jnp.ones((T, g_dim))
            ),
            params=params,
            H_theta_theta=jnp.zeros((p, p)),
            H_theta_X=jnp.zeros((T + 1, n, p)),
            H_theta_U=jnp.zeros((T, m, p)),
            H_theta_y_dyn=jnp.zeros((T + 1, n, p)),
            H_theta_y_eq=NodeAndEdgeValues(
                node=jnp.zeros((T + 1, c_dim, p)),
                edge=jnp.zeros((T, c_dim, p)),
            ),
            H_theta_z=NodeAndEdgeValues(
                node=jnp.zeros((T + 1, g_dim, p)),
                edge=jnp.zeros((T, g_dim, p)),
            ),
            equality_locations=NodeAndEdgeIndices(
                node=jnp.arange(T + 1), edge=jnp.arange(T)
            ),
            inequality_locations=NodeAndEdgeIndices(
                node=jnp.arange(T + 1), edge=jnp.arange(T)
            ),
        )
        rhs = TreeVariables(
            X=jnp.zeros((T + 1, n)),
            U=jnp.zeros((T, m)),
            S=NodeAndEdgeValues(
                node=jnp.zeros((T + 1, g_dim)), edge=jnp.zeros((T, g_dim))
            ),
            Y_dyn=jnp.zeros((T + 1, n)),
            Y_eq=NodeAndEdgeValues(
                node=jnp.zeros((T + 1, c_dim)), edge=jnp.zeros((T, c_dim))
            ),
            Z=NodeAndEdgeValues(
                node=jnp.zeros((T + 1, g_dim)), edge=jnp.zeros((T, g_dim))
            ),
            Theta=jnp.zeros((p,)),
        )

        shifted = add_scalar_hessian_regularization_delta(
            KKTSystem(lhs=lhs, rhs=rhs), delta
        )

        expected_q_shift = delta * jnp.eye(n)
        expected_r_shift = delta * jnp.eye(m)
        self.assertTrue(jnp.allclose(shifted.lhs.Q, lhs.Q + expected_q_shift))  # noqa: PT009
        self.assertTrue(jnp.allclose(shifted.lhs.R, lhs.R + expected_r_shift))  # noqa: PT009
        self.assertTrue(  # noqa: PT009
            jnp.allclose(shifted.lhs.Q_lqr, lhs.Q_lqr + expected_q_shift)
        )
        self.assertTrue(  # noqa: PT009
            jnp.allclose(shifted.lhs.R_lqr, lhs.R_lqr + expected_r_shift)
        )
        self.assertTrue(jnp.allclose(shifted.lhs.M, lhs.M))  # noqa: PT009
        self.assertTrue(jnp.allclose(shifted.lhs.M_lqr, lhs.M_lqr))  # noqa: PT009
        self.assertTrue(  # noqa: PT009
            jnp.allclose(shifted.lhs.H_theta_theta, delta * jnp.eye(p))
        )
        for shifted_leaf, original_leaf in zip(
            jax.tree_util.tree_leaves(shifted.lhs.E),
            jax.tree_util.tree_leaves(lhs.E),
            strict=True,
        ):
            self.assertTrue(jnp.allclose(shifted_leaf, original_leaf))  # noqa: PT009
        for shifted_leaf, original_leaf in zip(
            jax.tree_util.tree_leaves(shifted.lhs.G),
            jax.tree_util.tree_leaves(lhs.G),
            strict=True,
        ):
            self.assertTrue(jnp.allclose(shifted_leaf, original_leaf))  # noqa: PT009


if __name__ == "__main__":
    unittest.main()
