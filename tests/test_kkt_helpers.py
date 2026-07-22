"""Test the KKT helpers."""

import unittest
from dataclasses import replace

import jax
from jax import numpy as jnp
from regularized_lqr_jax.helpers import symmetrize

from primal_dual_lipa.kkt_builder import regularize_primal_hessian_blocks
from primal_dual_lipa.kkt_helpers import (
    compute_kkt_residual,
    factor_kkt,
    factorization_is_valid,
    solve_kkt,
)
from primal_dual_lipa.types import (
    KKTFactorizationInputs,
    NodeAndEdgeIndices,
    NodeAndEdgeValues,
    TreeParameters,
    TreeVariables,
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
        E = NodeAndEdgeValues(
            node=jnp.zeros((T + 1, c_dim, n)),
            edge=jnp.zeros((T, c_dim, n + m)),
        )
        G = NodeAndEdgeValues(
            node=jnp.zeros((T + 1, g_dim, n)),
            edge=jnp.zeros((T, g_dim, n + m)),
        )
        w_inv = NodeAndEdgeValues(
            node=jnp.zeros((T + 1, g_dim)), edge=jnp.zeros((T, g_dim))
        )
        params = TreeParameters(
            η_dyn=jnp.ones((T + 1, n)),
            η_eq=NodeAndEdgeValues(
                node=jnp.ones((T + 1, c_dim)), edge=jnp.ones((T, c_dim))
            ),
            η_ineq=NodeAndEdgeValues(
                node=jnp.ones((T + 1, g_dim)), edge=jnp.ones((T, g_dim))
            ),
            µ=0.1,
        )

        def make_inputs(delta: float) -> KKTFactorizationInputs:
            Q = jnp.ones((T + 1, n, n)) * delta
            R = jnp.ones((T, m, m)) * (-1.0 + delta)
            M = jnp.zeros((T, n, m))
            return KKTFactorizationInputs(
                Q=Q,
                M=M,
                R=R,
                Q_lqr=Q,
                M_lqr=M,
                R_lqr=R,
                D=D,
                E=E,
                G=G,
                w_inv=w_inv,
                params=params,
                H_theta_theta=jnp.zeros((p_dim, p_dim)),
                H_theta_X=jnp.zeros((T + 1, n, p_dim)),
                H_theta_U=jnp.zeros((T, m, p_dim)),
                H_theta_y_dyn=jnp.zeros((T + 1, n, p_dim)),
                H_theta_y_eq=NodeAndEdgeValues(
                    node=jnp.zeros((T + 1, c_dim, p_dim)),
                    edge=jnp.zeros((T, c_dim, p_dim)),
                ),
                H_theta_z=NodeAndEdgeValues(
                    node=jnp.zeros((T + 1, g_dim, p_dim)),
                    edge=jnp.zeros((T, g_dim, p_dim)),
                ),
                equality_locations=NodeAndEdgeIndices(
                    node=jnp.arange(T + 1), edge=jnp.arange(T)
                ),
                inequality_locations=NodeAndEdgeIndices(
                    node=jnp.arange(T + 1), edge=jnp.arange(T)
                ),
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
        self.all_locations = NodeAndEdgeIndices(
            node=jnp.arange(T + 1), edge=jnp.arange(T)
        )

        key = jax.random.PRNGKey(0)

        key, subkey = jax.random.split(key)
        self.D = jax.random.uniform(subkey, (T, n, n + m))

        key, subkey = jax.random.split(key)
        self.E = NodeAndEdgeValues(
            node=jax.random.uniform(subkey, (T + 1, c_dim, n)),
            edge=jax.random.uniform(subkey, (T, c_dim, n + m)),
        )

        key, subkey = jax.random.split(key)
        self.G = NodeAndEdgeValues(
            node=jax.random.uniform(subkey, (T + 1, g_dim, n)),
            edge=jax.random.uniform(subkey, (T, g_dim, n + m)),
        )

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
        H_theta_theta = symmetrize(jax.random.uniform(subkey, (p_dim, p_dim)))

        self.H_theta_X = jax.random.uniform(subkey, (T + 1, n, p_dim))
        self.H_theta_U = jax.random.uniform(subkey, (T, m, p_dim))
        self.H_theta_y_dyn = jax.random.uniform(subkey, (T + 1, n, p_dim))
        self.H_theta_y_eq = NodeAndEdgeValues(
            node=jax.random.uniform(subkey, (T + 1, c_dim, p_dim)),
            edge=jax.random.uniform(subkey, (T, c_dim, p_dim)),
        )
        self.H_theta_z = NodeAndEdgeValues(
            node=jax.random.uniform(subkey, (T + 1, g_dim, p_dim)),
            edge=jax.random.uniform(subkey, (T, g_dim, p_dim)),
        )

        hessian_regularization = 10.0
        self.Q, self.R, self.H_theta_theta = regularize_primal_hessian_blocks(
            Q=Q,
            R=R,
            H_theta_theta=H_theta_theta,
            hessian_regularization=hessian_regularization,
        )
        self.M = M

        key, subkey = jax.random.split(key)
        s = NodeAndEdgeValues(
            node=jnp.abs(jax.random.uniform(subkey, (T + 1, g_dim))),
            edge=jnp.abs(jax.random.uniform(subkey, (T, g_dim))),
        )

        key, subkey = jax.random.split(key)
        z = NodeAndEdgeValues(
            node=jnp.abs(jax.random.uniform(subkey, (T + 1, g_dim))),
            edge=jnp.abs(jax.random.uniform(subkey, (T, g_dim))),
        )

        self.w_inv = NodeAndEdgeValues(node=z.node / s.node, edge=z.edge / s.edge)

        key, subkey = jax.random.split(key)
        self.r_x = jax.random.uniform(subkey, (T + 1, n))

        key, subkey = jax.random.split(key)
        self.r_u = jax.random.uniform(subkey, (T, m))

        key, subkey = jax.random.split(key)
        self.r_s = NodeAndEdgeValues(
            node=jax.random.uniform(subkey, (T + 1, g_dim)),
            edge=jax.random.uniform(subkey, (T, g_dim)),
        )

        key, subkey = jax.random.split(key)
        self.r_y_dyn = jax.random.uniform(subkey, (T + 1, n))

        key, subkey = jax.random.split(key)
        self.r_y_eq = NodeAndEdgeValues(
            node=jax.random.uniform(subkey, (T + 1, c_dim)),
            edge=jax.random.uniform(subkey, (T, c_dim)),
        )

        key, subkey = jax.random.split(key)
        self.r_z = NodeAndEdgeValues(
            node=jax.random.uniform(subkey, (T + 1, g_dim)),
            edge=jax.random.uniform(subkey, (T, g_dim)),
        )

        self.parameters = TreeParameters(
            η_dyn=jnp.ones((T + 1, n)) * 10.0,
            η_eq=NodeAndEdgeValues(
                node=jnp.ones((T + 1, c_dim)) * 10.0,
                edge=jnp.ones((T, c_dim)) * 10.0,
            ),
            η_ineq=NodeAndEdgeValues(
                node=jnp.ones((T + 1, g_dim)) * 10.0,
                edge=jnp.ones((T, g_dim)) * 10.0,
            ),
            µ=0.1,
        )
        reg_w_inv_node = 1.0 / (
            1.0 / self.w_inv.node + 1.0 / self.parameters.η_ineq.node
        )
        reg_w_inv_edge = 1.0 / (
            1.0 / self.w_inv.edge + 1.0 / self.parameters.η_ineq.edge
        )
        node_constraint_hessian = jnp.einsum(
            "vki,vk,vkj->vij",
            self.E.node,
            self.parameters.η_eq.node,
            self.E.node,
        ) + jnp.einsum("vgi,vg,vgj->vij", self.G.node, reg_w_inv_node, self.G.node)
        edge_constraint_hessian = jnp.einsum(
            "eki,ek,ekj->eij",
            self.E.edge,
            self.parameters.η_eq.edge,
            self.E.edge,
        ) + jnp.einsum("egi,eg,egj->eij", self.G.edge, reg_w_inv_edge, self.G.edge)
        self.Q_lqr = self.Q + node_constraint_hessian
        self.Q_lqr = self.Q_lqr.at[:-1].add(edge_constraint_hessian[:, :n, :n])
        self.M_lqr = self.M + edge_constraint_hessian[:, :n, n:]
        self.R_lqr = self.R + edge_constraint_hessian[:, n:, n:]

    def test(self) -> None:
        """Run the test."""
        for use_parallel_lqr in [False, True]:
            with self.subTest(use_parallel_lqr=use_parallel_lqr):
                factorization_inputs = KKTFactorizationInputs(
                    Q=self.Q,
                    M=self.M,
                    R=self.R,
                    Q_lqr=self.Q_lqr,
                    M_lqr=self.M_lqr,
                    R_lqr=self.R_lqr,
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
                    equality_locations=self.all_locations,
                    inequality_locations=self.all_locations,
                )
                solve_inputs = TreeVariables(
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
                        residuals.S.node.flatten(),
                        residuals.S.edge.flatten(),
                        residuals.Y_dyn.flatten(),
                        residuals.Y_eq.node.flatten(),
                        residuals.Y_eq.edge.flatten(),
                        residuals.Z.node.flatten(),
                        residuals.Z.edge.flatten(),
                        residuals.Theta.flatten(),
                    ]
                )

                if use_parallel_lqr:
                    self.assertLess(jnp.linalg.norm(residual_vec), 1e-7)  # noqa: PT009
                else:
                    self.assertLess(jnp.linalg.norm(residual_vec), 1e-7)  # noqa: PT009


if __name__ == "__main__":
    unittest.main()
