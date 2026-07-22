"""Independent dense verification of the rooted-tree Newton-KKT solve."""

# ruff: noqa: ANN001, ANN202

import unittest

import jax
import numpy as np
from jax import numpy as jnp

from primal_dual_lipa.kkt_builder import build_kkt
from primal_dual_lipa.kkt_helpers import (
    compute_kkt_residual,
    factor_kkt,
    solve_kkt,
)
from primal_dual_lipa.topology import make_tree_ocp_topology
from primal_dual_lipa.types import (
    NodeAndEdgeIndices,
    NodeAndEdgeValues,
    OCPCallbackLocations,
    TreeParameters,
    TreeVariables,
)

jax.config.update("jax_enable_x64", True)  # noqa: FBT003


def _flatten_variables(values: TreeVariables) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(values.X).reshape(-1),
            np.asarray(values.U).reshape(-1),
            np.asarray(values.S.node).reshape(-1),
            np.asarray(values.S.edge).reshape(-1),
            np.asarray(values.Y_dyn).reshape(-1),
            np.asarray(values.Y_eq.node).reshape(-1),
            np.asarray(values.Y_eq.edge).reshape(-1),
            np.asarray(values.Z.node).reshape(-1),
            np.asarray(values.Z.edge).reshape(-1),
            np.asarray(values.Theta).reshape(-1),
        ]
    )


def _dense_tree_kkt(topology, inputs) -> np.ndarray:  # noqa: PLR0915
    """Assemble the explicit node/edge KKT matrix from its blocks."""
    V, E = topology.num_nodes, topology.num_edges
    n, m = inputs.Q.shape[-1], inputs.R.shape[-1]
    num_eq_nodes, cn = inputs.E.node.shape[:2]
    num_eq_edges, ce = inputs.E.edge.shape[:2]
    num_ineq_nodes, gn = inputs.G.node.shape[:2]
    num_ineq_edges, ge = inputs.G.edge.shape[:2]
    p = inputs.H_theta_theta.shape[0]
    sizes = (
        V * n,
        E * m,
        num_ineq_nodes * gn,
        num_ineq_edges * ge,
        V * n,
        num_eq_nodes * cn,
        num_eq_edges * ce,
        num_ineq_nodes * gn,
        num_ineq_edges * ge,
        p,
    )
    offsets = np.cumsum((0, *sizes))
    matrix = np.zeros((offsets[-1], offsets[-1]))

    def block(group, item, width):
        start = offsets[group] + item * width
        return slice(start, start + width)

    parents = np.asarray(topology.plan.edge_parents)
    children = np.asarray(topology.plan.edge_children)
    Q, M, R, D = map(np.asarray, (inputs.Q, inputs.M, inputs.R, inputs.D))
    En, Ee = np.asarray(inputs.E.node), np.asarray(inputs.E.edge)
    Gn, Ge = np.asarray(inputs.G.node), np.asarray(inputs.G.edge)

    for node in range(V):
        xs, ys = block(0, node, n), block(4, node, n)
        matrix[xs, xs] += Q[node]
        matrix[xs, ys] -= np.eye(n)
        matrix[ys, xs] -= np.eye(n)
        matrix[ys, ys] -= np.diag(1.0 / np.asarray(inputs.params.η_dyn[node]))
        matrix[xs, offsets[9] :] += np.asarray(inputs.H_theta_X[node])
        matrix[ys, offsets[9] :] += np.asarray(inputs.H_theta_y_dyn[node])

    for row, node in enumerate(np.asarray(inputs.equality_locations.node)):
        xs, eqs = block(0, int(node), n), block(5, row, cn)
        matrix[xs, eqs] += En[row].T
        matrix[eqs, xs] += En[row]
        matrix[eqs, eqs] -= np.diag(1.0 / np.asarray(inputs.params.η_eq.node[row]))
        matrix[eqs, offsets[9] :] += np.asarray(inputs.H_theta_y_eq.node[row])

    for row, node in enumerate(np.asarray(inputs.inequality_locations.node)):
        xs = block(0, int(node), n)
        ss, zs = block(2, row, gn), block(7, row, gn)
        matrix[xs, zs] += Gn[row].T
        matrix[zs, xs] += Gn[row]
        matrix[ss, ss] += np.diag(np.asarray(inputs.w_inv.node[row]))
        matrix[ss, zs] += np.eye(gn)
        matrix[zs, ss] += np.eye(gn)
        matrix[zs, zs] -= np.diag(1.0 / np.asarray(inputs.params.η_ineq.node[row]))
        matrix[zs, offsets[9] :] += np.asarray(inputs.H_theta_z.node[row])

    for edge, (parent, child) in enumerate(zip(parents, children, strict=True)):
        xp = block(0, int(parent), n)
        us = block(1, edge, m)
        yc = block(4, int(child), n)
        Dx, Du = D[edge, :, :n], D[edge, :, n:]
        matrix[xp, us] += M[edge]
        matrix[us, xp] += M[edge].T
        matrix[us, us] += R[edge]
        matrix[xp, yc] += Dx.T
        matrix[us, yc] += Du.T
        matrix[yc, xp] += Dx
        matrix[yc, us] += Du
        matrix[us, offsets[9] :] += np.asarray(inputs.H_theta_U[edge])

    for row, edge in enumerate(np.asarray(inputs.equality_locations.edge)):
        parent = parents[edge]
        xp, us = block(0, int(parent), n), block(1, int(edge), m)
        eqs = block(6, row, ce)
        Ex, Eu = Ee[row, :, :n], Ee[row, :, n:]
        matrix[xp, eqs] += Ex.T
        matrix[us, eqs] += Eu.T
        matrix[eqs, xp] += Ex
        matrix[eqs, us] += Eu
        matrix[eqs, eqs] -= np.diag(1.0 / np.asarray(inputs.params.η_eq.edge[row]))
        matrix[eqs, offsets[9] :] += np.asarray(inputs.H_theta_y_eq.edge[row])

    for row, edge in enumerate(np.asarray(inputs.inequality_locations.edge)):
        parent = parents[edge]
        xp, us = block(0, int(parent), n), block(1, int(edge), m)
        ss, zs = block(3, row, ge), block(8, row, ge)
        Gx, Gu = Ge[row, :, :n], Ge[row, :, n:]
        matrix[xp, zs] += Gx.T
        matrix[us, zs] += Gu.T
        matrix[zs, xp] += Gx
        matrix[zs, us] += Gu
        matrix[ss, ss] += np.diag(np.asarray(inputs.w_inv.edge[row]))
        matrix[ss, zs] += np.eye(ge)
        matrix[zs, ss] += np.eye(ge)
        matrix[zs, zs] -= np.diag(1.0 / np.asarray(inputs.params.η_ineq.edge[row]))
        matrix[zs, offsets[9] :] += np.asarray(inputs.H_theta_z.edge[row])

    theta = slice(offsets[9], offsets[10])
    matrix[theta, theta] += np.asarray(inputs.H_theta_theta)
    theta_blocks = (
        (inputs.H_theta_X, 0),
        (inputs.H_theta_U, 1),
        (inputs.H_theta_y_dyn, 4),
        (inputs.H_theta_y_eq.node, 5),
        (inputs.H_theta_y_eq.edge, 6),
        (inputs.H_theta_z.node, 7),
        (inputs.H_theta_z.edge, 8),
    )
    for values, group in theta_blocks:
        matrix[theta, offsets[group] : offsets[group + 1]] += (
            np.asarray(values).transpose(2, 0, 1).reshape(p, sizes[group])
        )
    return matrix


class TestTreeKKT(unittest.TestCase):
    """Verify the full branching KKT matrix independently."""

    def test_permuted_root_tree_matches_independent_dense_solve(  # noqa: C901, PLR0915
        self,
    ) -> None:
        """Compare tree Riccati recovery with a direct dense solve."""
        parent_nodes = [2, 2, -1, 0, 0]
        topology = make_tree_ocp_topology(
            parent_nodes,
            use_parallel_lqr=False,
        )
        V, E = topology.num_nodes, topology.num_edges

        def dynamics(x, u, theta, edge):
            del edge
            return jnp.array(
                [
                    0.7 * x[0] + 0.1 * x[1] + u[0] + 0.1 * theta[0],
                    -0.2 * x[0] + 0.5 * x[1] + 0.3 * u[0] - 0.05 * theta[0],
                ]
            )

        def node_cost(x, theta, node):
            weight = 1.0 + 0.03 * node
            return weight * (
                0.4 * jnp.dot(x, x)
                + 0.2 * jnp.square(theta[0])
                + 0.05 * theta[0] * x[1]
            )

        def edge_cost(x, u, theta, edge):
            weight = 1.0 + 0.05 * edge
            return weight * (
                0.2 * jnp.dot(x, x)
                + 0.75 * jnp.square(u[0])
                + 0.1 * x[0] * u[0]
                + 0.1 * jnp.square(theta[0])
            )

        def node_equalities(x, theta, node):
            return jnp.array([x[0] + 0.1 * theta[0] - 0.01 * node, x[1] - 0.02 * node])

        def edge_equalities(x, u, theta, edge):
            return jnp.array([x[0] + 0.2 * u[0] + 0.1 * theta[0] - 0.01 * edge])

        def node_inequalities(x, theta, node):
            del node
            return jnp.array([x[1] - 2.0 + 0.05 * theta[0]])

        def edge_inequalities(x, u, theta, edge):
            del edge
            return jnp.array([x[0] + 0.1 * u[0] - 2.0, -u[0] - 1.5 + 0.02 * theta[0]])

        locations = OCPCallbackLocations(
            cost=NodeAndEdgeIndices(node=jnp.array([3, 0]), edge=jnp.array([2, 1])),
            equalities=NodeAndEdgeIndices(
                node=jnp.array([4, 0]), edge=jnp.array([3, 1])
            ),
            inequalities=NodeAndEdgeIndices(
                node=jnp.array([4, 1, 2]), edge=jnp.array([3, 0, 2])
            ),
        )
        rng = np.random.default_rng(12)
        variables = TreeVariables(
            X=jnp.asarray(0.1 * rng.standard_normal((V, 2))),
            U=jnp.asarray(0.1 * rng.standard_normal((E, 1))),
            S=NodeAndEdgeValues(
                node=jnp.asarray(1.5 + 0.1 * rng.random((3, 1))),
                edge=jnp.asarray(1.5 + 0.1 * rng.random((3, 2))),
            ),
            Y_dyn=jnp.asarray(0.1 * rng.standard_normal((V, 2))),
            Y_eq=NodeAndEdgeValues(
                node=jnp.asarray(0.1 * rng.standard_normal((2, 2))),
                edge=jnp.asarray(0.1 * rng.standard_normal((2, 1))),
            ),
            Z=NodeAndEdgeValues(
                node=jnp.asarray(0.8 + 0.1 * rng.random((3, 1))),
                edge=jnp.asarray(0.8 + 0.1 * rng.random((3, 2))),
            ),
            Theta=jnp.array([0.2]),
        )
        params = TreeParameters(
            µ=0.1,
            η_dyn=jnp.full((V, 2), 7.0),
            η_eq=NodeAndEdgeValues(
                node=jnp.full((2, 2), 5.0), edge=jnp.full((2, 1), 4.0)
            ),
            η_ineq=NodeAndEdgeValues(
                node=jnp.full((3, 1), 6.0), edge=jnp.full((3, 2), 8.0)
            ),
        )
        system = build_kkt(
            node_cost=node_cost,
            edge_cost=edge_cost,
            dynamics=dynamics,
            node_equalities=node_equalities,
            edge_equalities=edge_equalities,
            node_inequalities=node_inequalities,
            edge_inequalities=edge_inequalities,
            x0=jnp.array([0.3, -0.2]),
            vars=variables,
            params=params,
            hessian_regularization=2.0,
            regularize_slack_elimination_with_mu=False,
            topology=topology,
            locations=locations,
        )

        parents = topology.plan.edge_parents
        children = topology.plan.edge_children
        primal0 = jnp.concatenate(
            [variables.X.reshape(-1), variables.U.reshape(-1), variables.Theta]
        )

        def full_lagrangian(primal):
            """Independently form the unregularized fixed-dual Lagrangian."""
            x_end = V * 2
            u_end = x_end + E
            X = primal[:x_end].reshape(V, 2)
            U = primal[x_end:u_end].reshape(E, 1)
            theta = primal[u_end:]
            edges = jnp.arange(E, dtype=jnp.int32)
            cost_nodes = locations.cost.node
            cost_edges = locations.cost.edge
            equality_nodes = locations.equalities.node
            equality_edges = locations.equalities.edge
            inequality_nodes = locations.inequalities.node
            inequality_edges = locations.inequalities.edge
            node_cost_terms = jax.vmap(lambda x, node: node_cost(x, theta, node))(
                X[cost_nodes], cost_nodes
            ).sum()
            edge_cost_terms = jax.vmap(lambda x, u, edge: edge_cost(x, u, theta, edge))(
                X[parents[cost_edges]], U[cost_edges], cost_edges
            ).sum()
            node_equality_terms = jax.vmap(
                lambda x, node, y_eq: jnp.dot(y_eq, node_equalities(x, theta, node))
            )(X[equality_nodes], equality_nodes, variables.Y_eq.node).sum()
            edge_equality_terms = jax.vmap(
                lambda x, u, edge, y_eq: jnp.dot(
                    y_eq, edge_equalities(x, u, theta, edge)
                )
            )(
                X[parents[equality_edges]],
                U[equality_edges],
                equality_edges,
                variables.Y_eq.edge,
            ).sum()
            node_inequality_terms = jax.vmap(
                lambda x, node, slack, z: (
                    -params.µ * jnp.log(slack).sum()
                    + jnp.dot(z, node_inequalities(x, theta, node) + slack)
                )
            )(
                X[inequality_nodes],
                inequality_nodes,
                variables.S.node,
                variables.Z.node,
            ).sum()
            edge_inequality_terms = jax.vmap(
                lambda x, u, edge, slack, z: (
                    -params.µ * jnp.log(slack).sum()
                    + jnp.dot(z, edge_inequalities(x, u, theta, edge) + slack)
                )
            )(
                X[parents[inequality_edges]],
                U[inequality_edges],
                inequality_edges,
                variables.S.edge,
                variables.Z.edge,
            ).sum()
            dynamics_terms = jax.vmap(
                lambda x, u, edge, y: jnp.dot(y, dynamics(x, u, theta, edge))
            )(X[parents], U, edges, variables.Y_dyn[children]).sum()
            return (
                node_cost_terms
                + edge_cost_terms
                + node_equality_terms
                + edge_equality_terms
                + node_inequality_terms
                + edge_inequality_terms
                + dynamics_terms
                - jnp.sum(variables.Y_dyn * X)
            )

        expected_gradient = jax.grad(full_lagrangian)(primal0)
        expected_hessian = jax.hessian(full_lagrangian)(primal0)
        assembled_hessian = np.zeros_like(np.asarray(expected_hessian))
        x_size, u_size = V * 2, E
        for node in range(V):
            xs = slice(node * 2, (node + 1) * 2)
            assembled_hessian[xs, xs] = np.asarray(system.lhs.Q[node]) - 2.0 * np.eye(2)
            assembled_hessian[xs, x_size + u_size :] = np.asarray(
                system.lhs.H_theta_X[node]
            )
            assembled_hessian[x_size + u_size :, xs] = np.asarray(
                system.lhs.H_theta_X[node]
            ).T
        for edge, parent in enumerate(np.asarray(parents)):
            xs = slice(int(parent) * 2, (int(parent) + 1) * 2)
            us = slice(x_size + edge, x_size + edge + 1)
            assembled_hessian[xs, us] = np.asarray(system.lhs.M[edge])
            assembled_hessian[us, xs] = np.asarray(system.lhs.M[edge]).T
            assembled_hessian[us, us] = np.asarray(system.lhs.R[edge]) - 2.0
            assembled_hessian[us, x_size + u_size :] = np.asarray(
                system.lhs.H_theta_U[edge]
            )
            assembled_hessian[x_size + u_size :, us] = np.asarray(
                system.lhs.H_theta_U[edge]
            ).T
        assembled_hessian[x_size + u_size :, x_size + u_size :] = np.asarray(
            system.lhs.H_theta_theta
        ) - 2.0 * np.eye(variables.Theta.shape[0])
        np.testing.assert_allclose(
            assembled_hessian, expected_hessian, atol=2e-10, rtol=2e-10
        )
        assembled_gradient = np.concatenate(
            [
                np.asarray(system.rhs.X).reshape(-1),
                np.asarray(system.rhs.U).reshape(-1),
                np.asarray(system.rhs.Theta),
            ]
        )
        np.testing.assert_allclose(
            assembled_gradient, expected_gradient, atol=2e-10, rtol=2e-10
        )

        dense_solution = np.linalg.solve(
            _dense_tree_kkt(topology, system.lhs), -_flatten_variables(system.rhs)
        )
        for use_parallel_lqr in (False, True):
            selected_topology = make_tree_ocp_topology(
                parent_nodes,
                use_parallel_lqr=use_parallel_lqr,
            )
            factorization = factor_kkt(
                system.lhs,
                use_parallel_lqr=use_parallel_lqr,
                topology=selected_topology,
            )
            solution = solve_kkt(
                factorization,
                system.lhs,
                system.rhs,
                use_parallel_lqr=use_parallel_lqr,
                topology=selected_topology,
            )
            np.testing.assert_allclose(
                _flatten_variables(solution),
                dense_solution,
                atol=3e-9,
                rtol=3e-9,
            )

            residual = compute_kkt_residual(
                system.lhs,
                system.rhs,
                solution,
                topology=selected_topology,
            )
            np.testing.assert_allclose(
                _flatten_variables(residual), 0.0, atol=2e-9, rtol=2e-9
            )


if __name__ == "__main__":
    unittest.main()
