"""End-to-end tests for rooted-tree optimal control problems."""

# ruff: noqa: ANN001, ANN202

import unittest

import jax
from jax import numpy as jnp
from jax_bidirectional_tree_rake_compress import ContractionExecutor, plan_statistics

from primal_dual_lipa.optimizers import solve, solve_tree
from primal_dual_lipa.topology import make_tree_ocp_topology
from primal_dual_lipa.types import (
    NodeAndEdgeIndices,
    NodeAndEdgeValues,
    OCPCallbackLocations,
    SolverSettings,
    TreeParameters,
    TreeVariables,
    Variables,
)

jax.config.update("jax_enable_x64", True)  # noqa: FBT003


def _empty_tree_variables(num_nodes: int) -> TreeVariables:
    num_edges = num_nodes - 1
    return TreeVariables(
        X=jnp.zeros((num_nodes, 1)),
        U=jnp.zeros((num_edges, 1)),
        S=NodeAndEdgeValues(
            node=jnp.zeros((num_nodes, 0)), edge=jnp.zeros((num_edges, 0))
        ),
        Y_dyn=jnp.zeros((num_nodes, 1)),
        Y_eq=NodeAndEdgeValues(
            node=jnp.zeros((num_nodes, 0)), edge=jnp.zeros((num_edges, 0))
        ),
        Z=NodeAndEdgeValues(
            node=jnp.zeros((num_nodes, 0)), edge=jnp.zeros((num_edges, 0))
        ),
        Theta=jnp.empty(0),
    )


class TestTreeOCPSolve(unittest.TestCase):
    """Verify tree topology, objective ownership, and chain compatibility."""

    def test_tree_schedule_follows_parallel_setting(self) -> None:
        """Sequential tree solves use rake-only plans; parallel solves compress."""
        parents = [-1, 0, 1, 1, 3, 4, 0]
        parallel = make_tree_ocp_topology(parents, use_parallel_lqr=True)
        sequential = make_tree_ocp_topology(parents, use_parallel_lqr=False)

        self.assertTrue(parallel.use_parallel_lqr)  # noqa: PT009
        self.assertFalse(sequential.use_parallel_lqr)  # noqa: PT009
        self.assertIs(parallel.plan.executor, ContractionExecutor.UNROLLED)  # noqa: PT009
        self.assertIs(sequential.plan.executor, ContractionExecutor.UNROLLED)  # noqa: PT009
        self.assertGreater(plan_statistics(parallel.plan).num_compressions, 0)  # noqa: PT009
        self.assertEqual(plan_statistics(sequential.plan).num_compressions, 0)  # noqa: PT009

        with self.assertRaisesRegex(ValueError, "topology was created with"):  # noqa: PT027
            solve_tree(
                vars_in=_empty_tree_variables(sequential.num_nodes),
                x0=jnp.zeros(1),
                dynamics=lambda x, u, theta, edge: x,
                settings=SolverSettings(max_iterations=0, use_parallel_lqr=True),
                topology=sequential,
            )

    def test_chain_topology_selects_scan_executor(self) -> None:
        """Chains avoid unrolled contraction loops in either solver mode."""
        parents = [-1, 0, 1, 2]
        parallel = make_tree_ocp_topology(parents, use_parallel_lqr=True)
        sequential = make_tree_ocp_topology(parents, use_parallel_lqr=False)

        self.assertIs(  # noqa: PT009
            parallel.plan.executor, ContractionExecutor.ASSOCIATIVE_SCAN
        )
        self.assertIs(sequential.plan.executor, ContractionExecutor.SCAN)  # noqa: PT009

    def test_sequential_and_parallel_tree_schedules_match(self) -> None:
        """Both schedules solve the same depth-two branching quadratic OCP."""
        parents = [-1, 0, 0, 1, 2]
        goals = jnp.array([2.0, -3.0])
        first_leaf = 3

        def dynamics(x, u, theta, edge):
            del theta, edge
            return x + u

        def node_cost(x, theta, node):
            del theta
            goal = goals[jnp.clip(node - first_leaf, 0, goals.shape[0] - 1)]
            return jnp.where(
                node >= first_leaf,
                0.5 * jnp.square(x[0] - goal),
                0.0,
            )

        def edge_cost(x, u, theta, edge):
            del x, theta, edge
            return 0.5 * jnp.square(u[0])

        results = []
        for use_parallel_lqr in (False, True):
            with self.subTest(use_parallel_lqr=use_parallel_lqr):
                topology = make_tree_ocp_topology(
                    parents,
                    use_parallel_lqr=use_parallel_lqr,
                )
                result, iterations, no_errors, _ = solve_tree(
                    vars_in=_empty_tree_variables(topology.num_nodes),
                    x0=jnp.array([0.5]),
                    dynamics=dynamics,
                    settings=SolverSettings(
                        residual_sq_threshold=1e-20,
                        num_iterative_refinement_steps=1,
                        use_parallel_lqr=use_parallel_lqr,
                    ),
                    node_cost=node_cost,
                    edge_cost=edge_cost,
                    topology=topology,
                )
                self.assertTrue(bool(no_errors))  # noqa: PT009
                self.assertLess(int(iterations), 20)  # noqa: PT009
                results.append(result)

        self.assertTrue(jnp.allclose(results[0].X, results[1].X, atol=1e-8))  # noqa: PT009
        self.assertTrue(jnp.allclose(results[0].U, results[1].U, atol=1e-8))  # noqa: PT009

    def test_star_tree_matches_analytic_solution(self) -> None:
        """Solve two independent child branches sharing a fixed root state."""
        topology = make_tree_ocp_topology([-1, 0, 0], use_parallel_lqr=True)
        goals = jnp.array([2.0, -4.0])

        def dynamics(x, u, theta, edge):
            del theta, edge
            return x + u

        def node_cost(x, theta, node):
            del theta
            goal = goals[jnp.clip(node - 1, 0, goals.shape[0] - 1)]
            return jnp.where(node == 0, 0.0, 0.5 * jnp.square(x[0] - goal))

        def edge_cost(x, u, theta, edge):
            del x, theta, edge
            return 0.5 * jnp.square(u[0])

        x0 = jnp.array([0.5])
        variables = _empty_tree_variables(topology.num_nodes)
        result, iterations, no_errors, _ = solve_tree(
            vars_in=variables,
            x0=x0,
            dynamics=dynamics,
            settings=SolverSettings(
                residual_sq_threshold=1e-20,
                num_iterative_refinement_steps=1,
                use_parallel_lqr=True,
            ),
            node_cost=node_cost,
            edge_cost=edge_cost,
            topology=topology,
        )

        expected_u = 0.5 * (goals - x0[0])
        expected_x = jnp.concatenate([x0, x0 + expected_u])[:, None]
        self.assertTrue(bool(no_errors))  # noqa: PT009
        self.assertLess(int(iterations), 20)  # noqa: PT009
        self.assertTrue(jnp.allclose(result.U[:, 0], expected_u, atol=1e-8))  # noqa: PT009
        self.assertTrue(jnp.allclose(result.X, expected_x, atol=1e-8))  # noqa: PT009

    def test_single_node_tree(self) -> None:
        """A root-only tree has one node and no edge controls."""
        topology = make_tree_ocp_topology([-1], use_parallel_lqr=False)
        variables = _empty_tree_variables(1)

        result, iterations, no_errors, _ = solve_tree(
            vars_in=variables,
            x0=jnp.array([0.75]),
            dynamics=lambda x, u, theta, edge: x,
            settings=SolverSettings(residual_sq_threshold=1e-20),
            node_cost=lambda x, theta, node: 0.5 * jnp.square(x[0] - 1.0),
            topology=topology,
        )

        self.assertTrue(bool(no_errors))  # noqa: PT009
        self.assertLess(int(iterations), 10)  # noqa: PT009
        self.assertEqual(result.U.shape, (0, 1))  # noqa: PT009
        self.assertTrue(jnp.allclose(result.X, jnp.array([[0.75]]), atol=1e-10))  # noqa: PT009

    def test_explicit_chain_topology_matches_simple_solve(self) -> None:
        """The simple chain facade and explicit tree API use the same solver path."""
        horizon = 3
        topology = make_tree_ocp_topology(
            [-1, 0, 1, 2],
            use_parallel_lqr=False,
        )

        def dynamics(x, u, theta, stage):
            del theta, stage
            return 0.8 * x + u

        def edge_cost(x, u, theta, edge):
            del theta
            running = 0.5 * jnp.square(x[0]) + 0.2 * jnp.square(u[0])
            del edge
            return running

        def node_cost(x, theta, node):
            del theta, node
            return 2.0 * jnp.square(x[0] - 1.0)

        def cost(x, u, theta, stage):
            return jnp.where(
                stage == horizon,
                node_cost(x, theta, stage),
                edge_cost(x, u, theta, stage),
            )

        simple_variables = Variables(
            X=jnp.zeros((horizon + 1, 1)),
            U=jnp.zeros((horizon, 1)),
            S=jnp.zeros((horizon + 1, 0)),
            Y_dyn=jnp.zeros((horizon + 1, 1)),
            Y_eq=jnp.zeros((horizon + 1, 0)),
            Z=jnp.zeros((horizon + 1, 0)),
            Theta=jnp.empty(0),
        )
        chain_locations = NodeAndEdgeIndices(
            node=jnp.array([horizon]), edge=jnp.arange(horizon)
        )
        locations = OCPCallbackLocations(
            cost=chain_locations,
            equalities=chain_locations,
            inequalities=chain_locations,
        )
        tree_variables = TreeVariables(
            X=simple_variables.X,
            U=simple_variables.U,
            S=NodeAndEdgeValues(
                node=simple_variables.S[-1:], edge=simple_variables.S[:-1]
            ),
            Y_dyn=simple_variables.Y_dyn,
            Y_eq=NodeAndEdgeValues(
                node=simple_variables.Y_eq[-1:], edge=simple_variables.Y_eq[:-1]
            ),
            Z=NodeAndEdgeValues(
                node=simple_variables.Z[-1:], edge=simple_variables.Z[:-1]
            ),
            Theta=simple_variables.Theta,
        )
        settings = SolverSettings(residual_sq_threshold=1e-20)
        simple, simple_iterations, simple_ok, _ = solve(
            vars_in=simple_variables,
            x0=jnp.array([-0.25]),
            cost=cost,
            dynamics=dynamics,
            settings=settings,
        )
        tree, tree_iterations, tree_ok, _ = solve_tree(
            vars_in=tree_variables,
            x0=jnp.array([-0.25]),
            dynamics=dynamics,
            settings=settings,
            node_cost=node_cost,
            edge_cost=edge_cost,
            topology=topology,
            locations=locations,
        )

        self.assertTrue(bool(simple_ok))  # noqa: PT009
        self.assertTrue(bool(tree_ok))  # noqa: PT009
        self.assertEqual(int(simple_iterations), int(tree_iterations))  # noqa: PT009
        self.assertTrue(jnp.allclose(tree.X, simple.X, atol=1e-9, rtol=1e-9))  # noqa: PT009
        self.assertTrue(jnp.allclose(tree.U, simple.U, atol=1e-9, rtol=1e-9))  # noqa: PT009

    def test_constrained_tree_reaches_branch_terminal_targets(self) -> None:
        """Exercise tree equality, inequality, slack, and barrier bookkeeping."""
        topology = make_tree_ocp_topology([-1, 0, 0], use_parallel_lqr=False)
        goals = jnp.array([0.4, -0.6])

        def dynamics(x, u, theta, edge):
            del theta, edge
            return x + u

        def node_cost(x, theta, node):
            del theta, node
            return 0.05 * jnp.square(x[0])

        def edge_cost(x, u, theta, edge):
            del x, theta, edge
            return 0.5 * jnp.square(u[0])

        def edge_equalities(x, u, theta, edge):
            del theta
            return jnp.array([x[0] + u[0] - goals[edge]])

        def edge_inequalities(x, u, theta, edge):
            del x, theta
            del edge
            return jnp.array([u[0] - 1.0, -u[0] - 1.0])

        all_nodes = jnp.arange(topology.num_nodes, dtype=jnp.int32)
        all_edges = jnp.arange(topology.num_edges, dtype=jnp.int32)
        no_nodes = jnp.empty(0, dtype=jnp.int32)
        locations = OCPCallbackLocations(
            cost=NodeAndEdgeIndices(node=all_nodes, edge=all_edges),
            equalities=NodeAndEdgeIndices(node=no_nodes, edge=all_edges),
            inequalities=NodeAndEdgeIndices(node=no_nodes, edge=all_edges),
        )
        variables = TreeVariables(
            X=jnp.zeros((topology.num_nodes, 1)),
            U=jnp.zeros((topology.num_edges, 1)),
            S=NodeAndEdgeValues(
                node=jnp.ones((0, 0)),
                edge=jnp.ones((topology.num_edges, 2)),
            ),
            Y_dyn=jnp.zeros((topology.num_nodes, 1)),
            Y_eq=NodeAndEdgeValues(
                node=jnp.zeros((0, 0)),
                edge=jnp.zeros((topology.num_edges, 1)),
            ),
            Z=NodeAndEdgeValues(
                node=jnp.ones((0, 0)),
                edge=jnp.ones((topology.num_edges, 2)),
            ),
            Theta=jnp.empty(0),
        )
        result, iterations, no_errors, _ = solve_tree(
            vars_in=variables,
            x0=jnp.zeros(1),
            dynamics=dynamics,
            settings=SolverSettings(
                max_iterations=100,
                residual_sq_threshold=1e-14,
                print_logs=True,
            ),
            node_cost=node_cost,
            edge_cost=edge_cost,
            edge_equalities=edge_equalities,
            edge_inequalities=edge_inequalities,
            topology=topology,
            locations=locations,
        )

        self.assertTrue(bool(no_errors))  # noqa: PT009
        self.assertLess(int(iterations), 100)  # noqa: PT009
        self.assertTrue(jnp.allclose(result.X[1:, 0], goals, atol=2e-6))  # noqa: PT009
        self.assertTrue(jnp.allclose(result.U[:, 0], goals, atol=2e-6))  # noqa: PT009
        edge_inequalities = jax.vmap(lambda u: jnp.array([u[0] - 1.0, -u[0] - 1.0]))(
            result.U
        )
        self.assertLessEqual(float(jnp.max(edge_inequalities)), 1e-8)  # noqa: PT009

    def test_invalid_tree_variable_layout_is_reported(self) -> None:
        """Reject a local-stage warm start with the wrong row count."""
        topology = make_tree_ocp_topology([-1, 0, 0], use_parallel_lqr=False)
        all_nodes = jnp.arange(topology.num_nodes, dtype=jnp.int32)
        all_edges = jnp.arange(topology.num_edges, dtype=jnp.int32)
        all_locations = NodeAndEdgeIndices(node=all_nodes, edge=all_edges)
        duplicate_locations = OCPCallbackLocations(
            cost=NodeAndEdgeIndices(node=jnp.array([0, 0]), edge=all_edges),
            equalities=all_locations,
            inequalities=all_locations,
        )
        with self.assertRaisesRegex(ValueError, "must not contain duplicate"):  # noqa: PT027
            solve_tree(
                vars_in=_empty_tree_variables(topology.num_nodes),
                x0=jnp.zeros(1),
                dynamics=lambda x, u, theta, edge: x,
                settings=SolverSettings(max_iterations=0),
                topology=topology,
                locations=duplicate_locations,
            )

        variables = _empty_tree_variables(topology.num_nodes)
        variables.S = NodeAndEdgeValues(
            node=jnp.zeros((topology.num_nodes - 1, 0)),
            edge=variables.S.edge,
        )

        with self.assertRaisesRegex(  # noqa: PT027
            ValueError, "one row per selected node"
        ):
            solve_tree(
                vars_in=variables,
                x0=jnp.zeros(1),
                dynamics=lambda x, u, theta, edge: x,
                settings=SolverSettings(max_iterations=0),
                topology=topology,
            )

        valid_variables = _empty_tree_variables(topology.num_nodes)
        bad_parameters = TreeParameters(
            µ=1e-3,
            η_dyn=jnp.ones((topology.num_nodes - 1, 1)),
            η_eq=NodeAndEdgeValues(
                node=jnp.ones((topology.num_nodes, 0)),
                edge=jnp.ones((topology.num_edges, 0)),
            ),
            η_ineq=NodeAndEdgeValues(
                node=jnp.ones((topology.num_nodes, 0)),
                edge=jnp.ones((topology.num_edges, 0)),
            ),
        )
        with self.assertRaisesRegex(ValueError, "params_in.η_dyn"):  # noqa: PT027
            solve_tree(
                vars_in=valid_variables,
                x0=jnp.zeros(1),
                dynamics=lambda x, u, theta, edge: x,
                settings=SolverSettings(max_iterations=0),
                params_in=bad_parameters,
                topology=topology,
            )


if __name__ == "__main__":
    unittest.main()
