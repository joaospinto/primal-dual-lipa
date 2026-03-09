"""Test the KKT builders and regularization logic."""

import unittest

import jax
from jax import numpy as jnp
from regularized_lqr_jax.helpers import symmetrize

from primal_dual_lipa.kkt_builder import regularize

jax.config.update("jax_enable_x64", True)  # noqa: FBT003


class TestKKTRegularization(unittest.TestCase):
    """Test the robust regularization of the KKT Hessian."""

    def test_joint_hessian_psd(self) -> None:
        """Verify that the joint Hessian is PSD after regularization.

        Note: The joint Hessian here refers to the block-diagonal Hessian of the
        Lagrangian with respect to (X, U, Theta). In the Newton-KKT system,
        the coupling between stages comes from the dynamics constraints, not
        the Hessian itself (which is block-diagonal in (X, U)).
        """
        n = 3
        m = 2
        p = 2
        T = 5
        psd_delta = 1e-3

        key = jax.random.PRNGKey(42)

        # Generate random indefinite components
        key, *subkeys = jax.random.split(key, 7)
        Q = jax.vmap(symmetrize)(jax.random.normal(subkeys[0], (T + 1, n, n)))
        R = jax.vmap(symmetrize)(jax.random.normal(subkeys[1], (T, m, m)))
        M = jax.random.normal(subkeys[2], (T, n, m))
        H_theta_theta = symmetrize(jax.random.normal(subkeys[3], (p, p)))
        # Distribute the theta Hessian across stages for the test
        H_theta_theta_per_stage = jnp.stack([H_theta_theta / (T + 1)] * (T + 1))
        H_x_theta = jax.random.normal(subkeys[4], (T + 1, n, p))
        H_u_theta = jax.random.normal(subkeys[5], (T, m, p))

        # Regularize
        Q_reg, R_reg, H_theta_theta_per_stage_reg = regularize(
            Q=Q,
            R=R,
            M=M,
            psd_delta=psd_delta,
            H_theta_theta_per_stage=H_theta_theta_per_stage,
            H_x_theta=H_x_theta,
            H_u_theta=H_u_theta,
        )
        H_theta_theta_reg = jnp.sum(H_theta_theta_per_stage_reg, axis=0)

        # Assemble the full joint Hessian
        # Structure:
        # [ diag(P_0, ..., P_T)   H_xu_theta_total ]
        # [ H_xu_theta_total^T    H_theta_theta_reg ]
        # where P_t = [Q_t M_t; M_t^T R_t] for t < T, and P_T = Q_T

        P_blocks = []
        H_xu_theta_list = []
        for t in range(T):
            # IMPORTANT: Use stage-specific M[t], Q_reg[t], R_reg[t]
            P_t = jnp.block([[Q_reg[t], M[t]], [M[t].T, R_reg[t]]])
            P_blocks.append(P_t)
            # Match the P_t block structure: [x; u]
            H_xu_theta_list.append(
                jnp.concatenate([H_x_theta[t], H_u_theta[t]], axis=0)
            )

        P_blocks.append(Q_reg[T])
        H_xu_theta_list.append(H_x_theta[T])

        P_total = jax.scipy.linalg.block_diag(*P_blocks)
        H_xu_theta_total = jnp.concatenate(H_xu_theta_list, axis=0)

        # Joint matrix:
        # J = [ P_total          H_xu_theta_total ]
        #     [ H_xu_theta_total^T  H_theta_theta_reg ]
        joint_hessian = jnp.block(
            [
                [P_total, H_xu_theta_total],
                [H_xu_theta_total.T, H_theta_theta_reg],
            ]
        )

        # Force symmetry explicitly to avoid numerical noise issues
        joint_hessian = (joint_hessian + joint_hessian.T) / 2.0

        # Check eigenvalues
        eigvals = jnp.linalg.eigvalsh(joint_hessian)
        min_eig = jnp.min(eigvals)

        self.assertGreater(min_eig, 1e-12)  # noqa: PT009


if __name__ == "__main__":
    unittest.main()
