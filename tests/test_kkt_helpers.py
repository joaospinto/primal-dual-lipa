"""Test the KKT helpers."""

import unittest
from dataclasses import replace

import jax
from jax import numpy as jnp
from regularized_lqr_jax.helpers import symmetrize

from primal_dual_lipa.kkt_builder import regularize_primal_hessian_blocks
from primal_dual_lipa.kkt_helpers import (
    compute_kkt_residual,
    factorization_is_valid,
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

    def test_factorization_validity_detects_primal_pd_breaks(self) -> None:
        """Check sequential and parallel Riccati validity with scalar shifts."""
        T = 2
        n = 1
        m = 1
        c_dim = 0
        g_dim = 0
        p_dim = 0

        D = jnp.zeros((T, n, n + m))
        E = jnp.zeros((T + 1, c_dim, n + m))
        G = jnp.zeros((T + 1, g_dim, n + m))
        w_inv = jnp.zeros((T + 1, g_dim))
        params = Parameters(
            η_dyn=jnp.ones((T + 1, n)),
            η_eq=jnp.ones((T + 1, c_dim)),
            η_ineq=jnp.ones((T + 1, g_dim)),
            µ=0.1,
        )

        def make_inputs(delta: float) -> KKTFactorizationInputs:
            Q = jnp.ones((T + 1, n, n)) * delta
            R = jnp.ones((T, m, m)) * (-1.0 + delta)
            M = jnp.zeros((T, n, m))
            M_pad = jnp.concatenate([M, jnp.zeros_like(M[0])[None, ...]], axis=0)
            R_pad = jnp.concatenate([R, jnp.ones((1, m, m)) * delta], axis=0)
            P = jax.vmap(lambda q, m_block, r: jnp.block([[q, m_block], [m_block.T, r]]))(
                Q, M_pad, R_pad
            )
            return KKTFactorizationInputs(
                P=P,
                P_lqr=P,
                D=D,
                E=E,
                G=G,
                w_inv=w_inv,
                params=params,
                H_theta_theta=jnp.zeros((p_dim, p_dim)),
                H_theta_X=jnp.zeros((T + 1, n, p_dim)),
                H_theta_U=jnp.zeros((T, m, p_dim)),
                H_theta_y_dyn=jnp.zeros((T + 1, n, p_dim)),
                H_theta_y_eq=jnp.zeros((T + 1, c_dim, p_dim)),
                H_theta_z=jnp.zeros((T + 1, g_dim, p_dim)),
            )

        for use_parallel_lqr in [False, True]:
            with self.subTest(use_parallel_lqr=use_parallel_lqr):
                bad_outputs = factor_kkt(
                    inputs=make_inputs(delta=0.0),
                    use_parallel_lqr=use_parallel_lqr,
                )
                self.assertFalse(bool(factorization_is_valid(bad_outputs)))  # noqa: PT009

                good_outputs = factor_kkt(
                    inputs=make_inputs(delta=2.0),
                    use_parallel_lqr=use_parallel_lqr,
                )
                self.assertTrue(bool(factorization_is_valid(good_outputs)))  # noqa: PT009

                if hasattr(good_outputs.lqr_outputs, "S_cho"):
                    bad_lqr_outputs = replace(
                        good_outputs.lqr_outputs,
                        S_cho=good_outputs.lqr_outputs.S_cho.at[0, 0, 0].set(0.0),
                    )
                else:
                    bad_lqr_outputs = replace(
                        good_outputs.lqr_outputs,
                        F_lu=good_outputs.lqr_outputs.F_lu.at[0, 0, 0].set(0.0),
                    )
                bad_f_outputs = replace(good_outputs, lqr_outputs=bad_lqr_outputs)
                self.assertFalse(bool(factorization_is_valid(bad_f_outputs)))  # noqa: PT009

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

        # Global parameter initializations
        key, subkey = jax.random.split(key)
        self.Theta = jax.random.uniform(subkey, (p_dim,))
        self.r_theta = jax.random.uniform(subkey, (p_dim,))

        key, subkey = jax.random.split(key)
        H_theta_theta_per_stage = jax.random.uniform(subkey, (T + 1, p_dim, p_dim))
        H_theta_theta_per_stage = jax.vmap(symmetrize)(H_theta_theta_per_stage)

        self.H_theta_X = jax.random.uniform(subkey, (T + 1, n, p_dim))
        self.H_theta_U = jax.random.uniform(subkey, (T, m, p_dim))
        self.H_theta_y_dyn = jax.random.uniform(subkey, (T + 1, n, p_dim))
        self.H_theta_y_eq = jax.random.uniform(subkey, (T + 1, c_dim, p_dim))
        self.H_theta_z = jax.random.uniform(subkey, (T + 1, g_dim, p_dim))

        hessian_regularization = 10.0
        Q, R, H_theta_theta_per_stage = regularize_primal_hessian_blocks(
            Q=Q,
            R=R,
            H_theta_theta_per_stage=H_theta_theta_per_stage,
            hessian_regularization=hessian_regularization,
        )
        self.H_theta_theta = jnp.sum(H_theta_theta_per_stage, axis=0)
        self.H_theta_theta = self.H_theta_theta + hessian_regularization * jnp.eye(
            p_dim
        )

        M_pad = jnp.concatenate([M, jnp.zeros_like(M[0])[None, ...]], axis=0)

        R_pad = jnp.concatenate(
            [R, jnp.eye(m)[None, ...] * hessian_regularization], axis=0
        )

        self.P = jax.vmap(lambda q, m, r: jnp.block([[q, m], [m.T, r]]))(
            Q, M_pad, R_pad
        )

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

        self.parameters = Parameters(
            η_dyn=jnp.ones((T + 1, n)) * 10.0,
            η_eq=jnp.ones((T + 1, c_dim)) * 10.0,
            η_ineq=jnp.ones((T + 1, g_dim)) * 10.0,
            µ=0.1,
        )
        w = 1.0 / self.w_inv
        reg_w_inv = 1.0 / (w + 1.0 / self.parameters.η_ineq)
        bmm = jax.vmap(jnp.matmul)
        self.P_lqr = (
            self.P
            + bmm(self.E.mT, self.parameters.η_eq[..., None] * self.E)
            + bmm(self.G.mT, reg_w_inv[..., None] * self.G)
        )

    def test(self) -> None:
        """Run the test."""
        for use_parallel_lqr in [False, True]:
            with self.subTest(use_parallel_lqr=use_parallel_lqr):
                factorization_inputs = KKTFactorizationInputs(
                    P=self.P,
                    P_lqr=self.P_lqr,
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
                singular_schur_outputs = replace(
                    factorization_outputs,
                    schur_complement=jnp.zeros_like(
                        factorization_outputs.schur_complement
                    ),
                )
                self.assertFalse(  # noqa: PT009
                    bool(factorization_is_valid(singular_schur_outputs))
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
                    self.assertLess(jnp.linalg.norm(residual_vec), 1e-7)  # noqa: PT009
                else:
                    self.assertLess(jnp.linalg.norm(residual_vec), 1e-7)  # noqa: PT009


if __name__ == "__main__":
    unittest.main()
