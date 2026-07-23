"""Tests for the regularized and proximal PD-LIPA modes."""

import unittest

import jax
from jax import numpy as jnp

from primal_dual_lipa.lagrangian_helpers import (
    build_total_augmented_lagrangian,
    directional_augmented_lagrangian,
)
from primal_dual_lipa.optimizers import solve, solve_tree
from primal_dual_lipa.topology import make_tree_ocp_topology
from primal_dual_lipa.types import (
    NodeAndEdgeIndices,
    NodeAndEdgeValues,
    OCPCallbackLocations,
    SolverMode,
    SolverSettings,
    TreeParameters,
    TreeVariables,
    Variables,
)

jax.config.update("jax_enable_x64", True)  # noqa: FBT003


def _empty_edge_function(x, u, theta, edge):  # noqa: ANN001, ANN202
    del x, u, theta, edge
    return jnp.empty(0)


def _zero_edge_cost(x, u, theta, edge):  # noqa: ANN001, ANN202
    del x, u, theta, edge
    return 0.0


class TestSolverModeMerits(unittest.TestCase):
    """Check the three merit-function interpretations directly."""

    def setUp(self) -> None:
        """Construct a one-node problem with every multiplier type."""
        empty_indices = jnp.empty(0, dtype=jnp.int32)
        node_indices = jnp.array([0], dtype=jnp.int32)
        self.locations = OCPCallbackLocations(
            cost=NodeAndEdgeIndices(node=node_indices, edge=empty_indices),
            equalities=NodeAndEdgeIndices(node=node_indices, edge=empty_indices),
            inequalities=NodeAndEdgeIndices(node=node_indices, edge=empty_indices),
        )
        self.reference = TreeVariables(
            X=jnp.array([[1.0]]),
            U=jnp.empty((0, 1)),
            S=NodeAndEdgeValues(node=jnp.array([[2.0]]), edge=jnp.empty((0, 0))),
            Y_dyn=jnp.array([[3.0]]),
            Y_eq=NodeAndEdgeValues(node=jnp.array([[4.0]]), edge=jnp.empty((0, 0))),
            Z=NodeAndEdgeValues(node=jnp.array([[5.0]]), edge=jnp.empty((0, 0))),
            Theta=jnp.array([0.5]),
        )
        self.candidate = TreeVariables(
            X=jnp.array([[1.2]]),
            U=jnp.empty((0, 1)),
            S=NodeAndEdgeValues(node=jnp.array([[1.8]]), edge=jnp.empty((0, 0))),
            Y_dyn=jnp.array([[3.3]]),
            Y_eq=NodeAndEdgeValues(node=jnp.array([[3.6]]), edge=jnp.empty((0, 0))),
            Z=NodeAndEdgeValues(node=jnp.array([[4.5]]), edge=jnp.empty((0, 0))),
            Theta=jnp.array([0.7]),
        )
        self.params = TreeParameters(
            µ=jnp.array(0.2),
            η_dyn=jnp.array([[2.0]]),
            η_eq=NodeAndEdgeValues(node=jnp.array([[3.0]]), edge=jnp.empty((0, 0))),
            η_ineq=NodeAndEdgeValues(node=jnp.array([[4.0]]), edge=jnp.empty((0, 0))),
        )

    @staticmethod
    def _node_cost(x, theta, node):  # noqa: ANN001, ANN205
        del node
        return 0.5 * jnp.square(x[0]) + theta[0]

    @staticmethod
    def _dynamics(x, u, theta, edge):  # noqa: ANN001, ANN205
        del u, theta, edge
        return x

    @staticmethod
    def _node_equalities(x, theta, node):  # noqa: ANN001, ANN205
        del node
        return jnp.array([x[0] + theta[0] - 1.0])

    @staticmethod
    def _node_inequalities(x, theta, node):  # noqa: ANN001, ANN205
        del theta, node
        return jnp.array([x[0] - 2.0])

    def _merit(self, mode: SolverMode):  # noqa: ANN202
        return build_total_augmented_lagrangian(
            node_cost=self._node_cost,
            edge_cost=_zero_edge_cost,
            dynamics=self._dynamics,
            node_equalities=self._node_equalities,
            edge_equalities=_empty_edge_function,
            node_inequalities=self._node_inequalities,
            edge_inequalities=_empty_edge_function,
            x0=jnp.array([0.0]),
            params=self.params,
            topology=None,
            locations=self.locations,
            mode=mode,
            hessian_regularization=jnp.array(7.0),
            reference_variables=self.reference,
        )

    def test_default_mode_is_regularized_ipm(self) -> None:
        """Preserve the existing regularized-IPM behavior by default."""
        self.assertIs(SolverSettings().mode, SolverMode.REGULARIZED_IPM)  # noqa: PT009

    def test_primal_proximal_adds_centered_primal_term(self) -> None:
        """Add exactly the scalar Hessian regularization around the center."""
        regular_candidate = TreeVariables(
            X=self.candidate.X,
            U=self.candidate.U,
            S=self.candidate.S,
            Y_dyn=self.reference.Y_dyn,
            Y_eq=self.reference.Y_eq,
            Z=self.reference.Z,
            Theta=self.candidate.Theta,
        )
        regular_merit = self._merit(SolverMode.REGULARIZED_IPM)(regular_candidate)
        proximal_merit = self._merit(SolverMode.PRIMAL_PROXIMAL_IPM)(regular_candidate)
        expected_proximal_term = 0.5 * 7.0 * (0.2**2 + 0.2**2)
        self.assertAlmostEqual(  # noqa: PT009
            float(proximal_merit - regular_merit), expected_proximal_term
        )
        self.assertAlmostEqual(  # noqa: PT009
            float(
                self._merit(SolverMode.PRIMAL_PROXIMAL_IPM)(self.reference)
                - self._merit(SolverMode.REGULARIZED_IPM)(self.reference)
            ),
            0.0,
        )

    def test_primal_dual_proximal_matches_pdal_formula(self) -> None:
        """Match SIP's centered dual-proximal augmented Lagrangian."""
        merit = self._merit(SolverMode.PRIMAL_DUAL_PROXIMAL_IPM)(self.candidate)
        cost = 0.5 * 1.2**2 + 0.7
        barrier = -0.2 * jnp.log(1.8)

        def pdal_term(residual, dual, center, eta):  # noqa: ANN001, ANN202
            regularized_residual = residual - (dual - center) / eta
            return center * residual + 0.5 * eta * (
                residual**2 + regularized_residual**2
            )

        expected = (
            cost
            + barrier
            + pdal_term(-1.2, 3.3, 3.0, 2.0)
            + pdal_term(0.9, 3.6, 4.0, 3.0)
            + pdal_term(1.0, 4.5, 5.0, 4.0)
            + 0.5 * 7.0 * (0.2**2 + 0.2**2)
        )
        self.assertAlmostEqual(float(merit), float(expected))  # noqa: PT009

        deltas = jax.tree.map(
            lambda candidate, reference: candidate - reference,
            self.candidate,
            self.reference,
        )
        directional_merit = directional_augmented_lagrangian(
            node_cost=self._node_cost,
            edge_cost=_zero_edge_cost,
            dynamics=self._dynamics,
            node_equalities=self._node_equalities,
            edge_equalities=_empty_edge_function,
            node_inequalities=self._node_inequalities,
            edge_inequalities=_empty_edge_function,
            x0=jnp.array([0.0]),
            params=self.params,
            τ=jnp.array(0.995),
            topology=None,
            locations=self.locations,
            variables=self.reference,
            deltas=deltas,
            mode=SolverMode.PRIMAL_DUAL_PROXIMAL_IPM,
            hessian_regularization=jnp.array(7.0),
        )
        self.assertAlmostEqual(float(directional_merit(1.0)), float(merit))  # noqa: PT009


class TestSolverModesEndToEnd(unittest.TestCase):
    """Exercise every mode through the shared chain/tree implementation."""

    def test_chain_problem_solves_in_all_modes(self) -> None:
        """Reach the same constrained chain solution in every mode."""

        def dynamics(x, u, theta, stage):  # noqa: ANN001, ANN202
            del theta, stage
            return x + u

        def cost(x, u, theta, stage):  # noqa: ANN001, ANN202
            del theta
            return jnp.where(
                stage == 1,
                0.5 * jnp.square(x[0] - 1.0),
                0.5 * jnp.square(u[0]),
            )

        def inequalities(x, u, theta, stage):  # noqa: ANN001, ANN202
            del theta
            value = jnp.where(stage == 1, x[0], u[0])
            return jnp.array([value - 2.0])

        variables = Variables(
            X=jnp.zeros((2, 1)),
            U=jnp.zeros((1, 1)),
            S=jnp.full((2, 1), 2.0),
            Y_dyn=jnp.zeros((2, 1)),
            Y_eq=jnp.empty((2, 0)),
            Z=jnp.ones((2, 1)),
            Theta=jnp.empty(0),
        )
        solutions = []
        for mode in SolverMode:
            with self.subTest(mode=mode):
                solution, iterations, no_errors, _ = solve(
                    vars_in=variables,
                    x0=jnp.array([0.0]),
                    cost=cost,
                    dynamics=dynamics,
                    inequalities=inequalities,
                    settings=SolverSettings(
                        mode=mode,
                        residual_sq_threshold=1e-14,
                        num_iterative_refinement_steps=1,
                    ),
                )
                self.assertTrue(bool(no_errors))  # noqa: PT009
                self.assertLess(int(iterations), 100)  # noqa: PT009
                self.assertAlmostEqual(  # noqa: PT009
                    float(solution.U[0, 0]), 0.5, delta=1e-5
                )
                solutions.append(solution)

        for solution in solutions[1:]:
            self.assertTrue(  # noqa: PT009
                jnp.allclose(solution.X, solutions[0].X, atol=1e-5)
            )
            self.assertTrue(  # noqa: PT009
                jnp.allclose(solution.U, solutions[0].U, atol=1e-5)
            )

    def test_tree_problem_uses_shared_proximal_path(self) -> None:
        """Solve a branched OCP through the same proximal implementation."""
        topology = make_tree_ocp_topology([-1, 0, 0], use_parallel_lqr=False)
        variables = TreeVariables(
            X=jnp.zeros((3, 1)),
            U=jnp.zeros((2, 1)),
            S=NodeAndEdgeValues(node=jnp.empty((3, 0)), edge=jnp.empty((2, 0))),
            Y_dyn=jnp.zeros((3, 1)),
            Y_eq=NodeAndEdgeValues(node=jnp.empty((3, 0)), edge=jnp.empty((2, 0))),
            Z=NodeAndEdgeValues(node=jnp.empty((3, 0)), edge=jnp.empty((2, 0))),
            Theta=jnp.empty(0),
        )
        goals = jnp.array([2.0, -4.0])

        def dynamics(x, u, theta, edge):  # noqa: ANN001, ANN202
            del theta, edge
            return x + u

        def node_cost(x, theta, node):  # noqa: ANN001, ANN202
            del theta
            goal = goals[jnp.clip(node - 1, 0, 1)]
            return jnp.where(node == 0, 0.0, 0.5 * jnp.square(x[0] - goal))

        def edge_cost(x, u, theta, edge):  # noqa: ANN001, ANN202
            del x, theta, edge
            return 0.5 * jnp.square(u[0])

        solution, iterations, no_errors, _ = solve_tree(
            vars_in=variables,
            x0=jnp.array([0.5]),
            dynamics=dynamics,
            settings=SolverSettings(
                mode=SolverMode.PRIMAL_PROXIMAL_IPM,
                residual_sq_threshold=1e-16,
                num_iterative_refinement_steps=1,
            ),
            node_cost=node_cost,
            edge_cost=edge_cost,
            topology=topology,
        )
        self.assertTrue(bool(no_errors))  # noqa: PT009
        self.assertLess(int(iterations), 20)  # noqa: PT009
        self.assertTrue(  # noqa: PT009
            jnp.allclose(solution.U[:, 0], 0.5 * (goals - 0.5), atol=1e-8)
        )


if __name__ == "__main__":
    unittest.main()
