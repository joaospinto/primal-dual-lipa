"""Integration tests for self-contained branching Dymos formulations."""

import unittest

import jax
import numpy as np
from jax import numpy as jnp

jax.config.update("jax_enable_x64", True)  # noqa: FBT003

from primal_dual_lipa.kkt_builder import build_kkt_rhs  # noqa: E402
from primal_dual_lipa.lagrangian_helpers import (  # noqa: E402
    dynamics_residuals,
    evaluate_node_edge,
)
from primal_dual_lipa.optimizers import solve_tree  # noqa: E402
from primal_dual_lipa.types import (  # noqa: E402
    NodeAndEdgeValues,
    TreeVariables,
    node_edge_flatten,
    node_edge_map,
    node_edge_sum,
)
from tests.dymos_tree_problems import (  # noqa: E402
    TreeTestProblem,
    make_balanced_field_problem,
    make_battery_multibranch_problem,
)


def _solve_problem(problem: TreeTestProblem):  # noqa: ANN202
    return solve_tree(
        vars_in=problem.variables,
        x0=problem.x0,
        dynamics=problem.dynamics,
        settings=problem.settings,
        node_cost=problem.node_cost,
        edge_cost=problem.edge_cost,
        node_equalities=problem.node_equalities,
        edge_equalities=problem.edge_equalities,
        node_inequalities=problem.node_inequalities,
        edge_inequalities=problem.edge_inequalities,
        topology=problem.topology,
        locations=problem.locations,
    )


def _physical_residuals(
    problem: TreeTestProblem, variables: TreeVariables
) -> tuple[jax.Array, NodeAndEdgeValues, NodeAndEdgeValues]:
    dynamics = dynamics_residuals(
        problem.dynamics, problem.x0, variables, problem.topology
    )
    equalities = evaluate_node_edge(
        problem.node_equalities,
        problem.edge_equalities,
        variables.X,
        variables.U,
        variables.Theta,
        problem.topology,
        problem.locations.equalities,
    )
    inequalities = evaluate_node_edge(
        problem.node_inequalities,
        problem.edge_inequalities,
        variables.X,
        variables.U,
        variables.Theta,
        problem.topology,
        problem.locations.inequalities,
    )
    return dynamics, equalities, inequalities


class TestDymosTreeProblems(unittest.TestCase):
    """Exercise both genuinely branching Dymos graph formulations."""

    def test_battery_multibranch_solves(self) -> None:
        """Solve three battery/motor continuations from one shared SOC node."""
        problem = make_battery_multibranch_problem(print_logs=True)
        variables, iterations, no_errors, parameters = _solve_problem(problem)
        dynamics, equalities, inequalities = _physical_residuals(problem, variables)

        branch_node = problem.metadata["branch_node"]
        edge_parents = np.asarray(problem.topology.plan.edge_parents)
        edge_children = np.asarray(problem.topology.plan.edge_children)
        actual_branch_children = np.sort(edge_children[edge_parents == branch_node])
        np.testing.assert_array_equal(
            actual_branch_children, problem.metadata["branch_children"]
        )
        self.assertTrue(bool(no_errors))  # noqa: PT009
        self.assertLess(int(iterations), problem.settings.max_iterations)  # noqa: PT009
        self.assertLess(float(jnp.max(jnp.abs(dynamics))), 1e-6)  # noqa: PT009
        self.assertLess(  # noqa: PT009
            float(jnp.max(jnp.abs(node_edge_flatten(equalities)))), 1e-6
        )
        self.assertLessEqual(  # noqa: PT009
            float(jnp.max(node_edge_flatten(inequalities))), 1e-8
        )
        self.assertLess(  # noqa: PT009
            float(
                jnp.max(
                    jnp.abs(
                        node_edge_flatten(
                            node_edge_map(
                                lambda slack, z: slack * z - parameters.µ,
                                variables.S,
                                variables.Z,
                            )
                        )
                    )
                )
            ),
            1e-6,
        )

    def test_balanced_field_solves(self) -> None:
        """Solve rejected-takeoff and continued-climb branches jointly."""
        problem = make_balanced_field_problem(print_logs=True)
        variables, iterations, no_errors, parameters = _solve_problem(problem)
        dynamics, equalities, inequalities = _physical_residuals(problem, variables)

        branch_node = problem.metadata["branch_node"]
        edge_parents = np.asarray(problem.topology.plan.edge_parents)
        edge_children = np.asarray(problem.topology.plan.edge_children)
        actual_branch_children = np.sort(edge_children[edge_parents == branch_node])
        np.testing.assert_array_equal(
            actual_branch_children, problem.metadata["branch_children"]
        )

        self.assertTrue(bool(no_errors))  # noqa: PT009
        self.assertLess(int(iterations), problem.settings.max_iterations)  # noqa: PT009
        self.assertLess(float(jnp.max(jnp.abs(dynamics))), 1e-6)  # noqa: PT009
        self.assertLess(  # noqa: PT009
            float(jnp.max(jnp.abs(node_edge_flatten(equalities)))), 1e-6
        )
        self.assertLessEqual(  # noqa: PT009
            float(jnp.max(node_edge_flatten(inequalities))), 1e-6
        )
        self.assertLess(  # noqa: PT009
            float(
                jnp.max(
                    jnp.abs(
                        node_edge_flatten(
                            node_edge_map(
                                lambda slack, z: slack * z - parameters.µ,
                                variables.S,
                                variables.Z,
                            )
                        )
                    )
                )
            ),
            1e-6,
        )

        stationarity = build_kkt_rhs(
            node_cost=problem.node_cost,
            edge_cost=problem.edge_cost,
            dynamics=problem.dynamics,
            node_equalities=problem.node_equalities,
            edge_equalities=problem.edge_equalities,
            node_inequalities=problem.node_inequalities,
            edge_inequalities=problem.edge_inequalities,
            x0=problem.x0,
            vars=variables,
            params=parameters,
            topology=problem.topology,
            locations=problem.locations,
        )
        dual_stationarity = jnp.concatenate(
            [stationarity.X.ravel(), stationarity.U.ravel(), stationarity.Theta]
        )
        self.assertLess(float(jnp.max(jnp.abs(dual_stationarity))), 1e-6)  # noqa: PT009

        rto_final = problem.metadata["rto_final"]
        climb_final = problem.metadata["climb_final"]
        field_length = variables.Theta[5]
        self.assertAlmostEqual(  # noqa: PT009
            float(variables.X[rto_final, 0]), float(field_length), delta=1e-4
        )
        self.assertAlmostEqual(  # noqa: PT009
            float(variables.X[climb_final, 0]), float(field_length), delta=1e-4
        )
        self.assertAlmostEqual(float(variables.X[rto_final, 1]), 0.0, delta=1e-5)  # noqa: PT009
        self.assertAlmostEqual(  # noqa: PT009
            float(variables.X[climb_final, 1]), 35.0 * 0.3048, delta=1e-4
        )
        self.assertAlmostEqual(  # noqa: PT009
            float(variables.X[climb_final, 3]), 5.0 * np.pi / 180.0, delta=1e-6
        )

        runway_nodes = np.logical_not(problem.metadata["node_is_climb"])
        self.assertLess(  # noqa: PT009
            float(jnp.max(jnp.abs(variables.X[runway_nodes, 2:]))), 1e-6
        )
        objective = node_edge_sum(
            evaluate_node_edge(
                problem.node_cost,
                problem.edge_cost,
                variables.X,
                variables.U,
                variables.Theta,
                problem.topology,
                problem.locations.cost,
            )
        )
        self.assertGreater(float(objective), 2.15)  # noqa: PT009
        self.assertLess(float(objective), 2.25)  # noqa: PT009


if __name__ == "__main__":
    unittest.main()
