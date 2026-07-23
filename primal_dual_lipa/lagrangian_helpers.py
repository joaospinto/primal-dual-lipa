"""Helpers for node/edge Lagrangians and augmented Lagrangians."""

import jax
from jax import numpy as jnp

from primal_dual_lipa.topology import (
    TreeOCPTopology,
    edge_children,
    edge_parents,
    root_node,
)
from primal_dual_lipa.types import (
    EdgeCostFunction,
    EdgeFunction,
    NodeAndEdgeIndices,
    NodeAndEdgeValues,
    NodeCostFunction,
    NodeFunction,
    OCPCallbackLocations,
    SolverMode,
    TreeParameters,
    TreeVariables,
    node_edge_map,
    node_edge_sum,
)
from primal_dual_lipa.vectorization_helpers import vectorize_edge, vectorize_node


def pad(A: jax.Array) -> jax.Array:
    """Pad a time-indexed array by one zero row (legacy test helper)."""
    return jnp.pad(A, [[0, 1], [0, 0]])


def build_node_equality_lagrangian(equalities: NodeFunction):  # noqa: ANN201
    """Return the multiplier term for one selected node equality block."""

    def lagrangian(x, theta, node, y_eq):  # noqa: ANN001, ANN202
        return jnp.dot(y_eq, equalities(x, theta, node))

    return lagrangian


def build_node_inequality_lagrangian(  # noqa: ANN201
    inequalities: NodeFunction, µ: jnp.double
):
    """Return the barrier/multiplier term for one selected node block."""

    def lagrangian(x, theta, node, s, z):  # noqa: ANN001, ANN202
        return -µ * jnp.sum(jnp.log(s)) + jnp.dot(z, inequalities(x, theta, node) + s)

    return lagrangian


def build_edge_equality_lagrangian(equalities: EdgeFunction):  # noqa: ANN201
    """Return the multiplier term for one selected edge equality block."""

    def lagrangian(x, u, theta, edge, y_eq):  # noqa: ANN001, ANN202
        return jnp.dot(y_eq, equalities(x, u, theta, edge))

    return lagrangian


def build_edge_inequality_lagrangian(  # noqa: ANN201
    inequalities: EdgeFunction, µ: jnp.double
):
    """Return the barrier/multiplier term for one selected edge block."""

    def lagrangian(x, u, theta, edge, s, z):  # noqa: ANN001, ANN202
        return -µ * jnp.sum(jnp.log(s)) + jnp.dot(
            z, inequalities(x, u, theta, edge) + s
        )

    return lagrangian


def build_dynamics_lagrangian(dynamics: EdgeFunction):  # noqa: ANN201
    """Return the dynamics-multiplier term for one directed edge."""

    def dynamics_lagrangian(
        X_parent: jax.Array,
        U: jax.Array,
        Theta: jax.Array,
        edge: jnp.int32,
        Y_child: jax.Array,
    ) -> jnp.double:
        return jnp.dot(Y_child, dynamics(X_parent, U, Theta, edge))

    return dynamics_lagrangian


def evaluate_nodes(  # noqa: ANN201
    fun: NodeCostFunction | NodeFunction,
    X: jax.Array,
    Theta: jax.Array,
    indices: jax.Array,
):
    """Evaluate a callback at selected nodes, preserving selection order."""
    return vectorize_node(fun)(X[indices], Theta, indices)


def evaluate_edges(  # noqa: ANN201
    fun: EdgeCostFunction | EdgeFunction,
    X: jax.Array,
    U: jax.Array,
    Theta: jax.Array,
    topology: TreeOCPTopology | None,
    indices: jax.Array,
):
    """Evaluate a callback at selected edges, preserving selection order."""
    parents = edge_parents(topology, U.shape[0])
    return vectorize_edge(fun)(X[parents[indices]], U[indices], Theta, indices)


def evaluate_node_edge(
    node_fun: NodeCostFunction | NodeFunction,
    edge_fun: EdgeCostFunction | EdgeFunction,
    X: jax.Array,
    U: jax.Array,
    Theta: jax.Array,
    topology: TreeOCPTopology | None,
    locations: NodeAndEdgeIndices,
) -> NodeAndEdgeValues:
    """Evaluate callbacks at explicitly selected node and edge locations."""
    return NodeAndEdgeValues(
        node=evaluate_nodes(node_fun, X, Theta, locations.node),
        edge=evaluate_edges(edge_fun, X, U, Theta, topology, locations.edge),
    )


def evaluate_dynamics(  # noqa: ANN201
    dynamics: EdgeFunction,
    X: jax.Array,
    U: jax.Array,
    Theta: jax.Array,
    topology: TreeOCPTopology | None,
):
    """Evaluate dynamics once per edge in contraction-plan edge order."""
    indices = jnp.arange(U.shape[0], dtype=jnp.int32)
    return evaluate_edges(dynamics, X, U, Theta, topology, indices)


def dynamics_residuals(
    dynamics: EdgeFunction,
    x0: jax.Array,
    variables: TreeVariables,
    topology: TreeOCPTopology | None,
) -> jax.Array:
    """Return root/edge dynamics defects in node order."""
    children = edge_children(topology, variables.U.shape[0])
    root = root_node(topology)
    residuals = -variables.X
    residuals = residuals.at[root].add(x0)
    return residuals.at[children].add(
        evaluate_dynamics(dynamics, variables.X, variables.U, variables.Theta, topology)
    )


def _dual_proximal_term(
    residual: jax.Array,
    dual: jax.Array,
    center: jax.Array,
    eta: jax.Array,
) -> jnp.double:
    regularized_residual = residual - (dual - center) / eta
    return jnp.sum(
        center * residual
        + 0.5 * eta * (jnp.square(residual) + jnp.square(regularized_residual))
    )


def build_total_augmented_lagrangian(  # noqa: ANN201
    node_cost: NodeCostFunction,
    edge_cost: EdgeCostFunction,
    dynamics: EdgeFunction,
    node_equalities: NodeFunction,
    edge_equalities: EdgeFunction,
    node_inequalities: NodeFunction,
    edge_inequalities: EdgeFunction,
    x0: jax.Array,
    params: TreeParameters,
    topology: TreeOCPTopology | None,
    locations: OCPCallbackLocations,
    mode: SolverMode,
    hessian_regularization: jnp.double,
    reference_variables: TreeVariables,
):
    """Return a function evaluating the associated augmented Lagrangian."""
    node_equality_lagrangian = build_node_equality_lagrangian(node_equalities)
    node_inequality_lagrangian = build_node_inequality_lagrangian(
        node_inequalities, params.µ
    )
    edge_equality_lagrangian = build_edge_equality_lagrangian(edge_equalities)
    edge_inequality_lagrangian = build_edge_inequality_lagrangian(
        edge_inequalities, params.µ
    )
    dynamics_lagrangian = build_dynamics_lagrangian(dynamics)

    def augmented_lagrangian(variables: TreeVariables) -> jnp.double:
        local_cost = node_edge_sum(
            evaluate_node_edge(
                node_cost,
                edge_cost,
                variables.X,
                variables.U,
                variables.Theta,
                topology,
                locations.cost,
            )
        )
        dyn_residual = dynamics_residuals(dynamics, x0, variables, topology)
        equality_residual = evaluate_node_edge(
            node_equalities,
            edge_equalities,
            variables.X,
            variables.U,
            variables.Theta,
            topology,
            locations.equalities,
        )
        inequality_residual = node_edge_map(
            lambda value, slack: value + slack,
            evaluate_node_edge(
                node_inequalities,
                edge_inequalities,
                variables.X,
                variables.U,
                variables.Theta,
                topology,
                locations.inequalities,
            ),
            variables.S,
        )
        if mode.uses_dual_center:
            barrier = -params.µ * node_edge_sum(node_edge_map(jnp.log, variables.S))
            merit = (
                local_cost
                + barrier
                + _dual_proximal_term(
                    dyn_residual,
                    variables.Y_dyn,
                    reference_variables.Y_dyn,
                    params.η_dyn,
                )
                + _dual_proximal_term(
                    equality_residual.node,
                    variables.Y_eq.node,
                    reference_variables.Y_eq.node,
                    params.η_eq.node,
                )
                + _dual_proximal_term(
                    equality_residual.edge,
                    variables.Y_eq.edge,
                    reference_variables.Y_eq.edge,
                    params.η_eq.edge,
                )
                + _dual_proximal_term(
                    inequality_residual.node,
                    variables.Z.node,
                    reference_variables.Z.node,
                    params.η_ineq.node,
                )
                + _dual_proximal_term(
                    inequality_residual.edge,
                    variables.Z.edge,
                    reference_variables.Z.edge,
                    params.η_ineq.edge,
                )
            )
        else:
            num_edges = variables.U.shape[0]
            parents = edge_parents(topology, num_edges)
            children = edge_children(topology, num_edges)
            root = root_node(topology)
            edge_indices = jnp.arange(num_edges, dtype=jnp.int32)
            local_node_equality = vectorize_node(node_equality_lagrangian)(
                variables.X[locations.equalities.node],
                variables.Theta,
                locations.equalities.node,
                variables.Y_eq.node,
            ).sum()
            local_node_inequality = vectorize_node(node_inequality_lagrangian)(
                variables.X[locations.inequalities.node],
                variables.Theta,
                locations.inequalities.node,
                variables.S.node,
                variables.Z.node,
            ).sum()
            equality_edges = locations.equalities.edge
            local_edge_equality = vectorize_edge(edge_equality_lagrangian)(
                variables.X[parents[equality_edges]],
                variables.U[equality_edges],
                variables.Theta,
                equality_edges,
                variables.Y_eq.edge,
            ).sum()
            inequality_edges = locations.inequalities.edge
            local_edge_inequality = vectorize_edge(edge_inequality_lagrangian)(
                variables.X[parents[inequality_edges]],
                variables.U[inequality_edges],
                variables.Theta,
                inequality_edges,
                variables.S.edge,
                variables.Z.edge,
            ).sum()
            dynamics_term = vectorize_edge(dynamics_lagrangian)(
                variables.X[parents],
                variables.U,
                variables.Theta,
                edge_indices,
                variables.Y_dyn[children],
            ).sum()
            state_term = -jnp.sum(variables.Y_dyn * variables.X) + jnp.dot(
                variables.Y_dyn[root], x0
            )
            dyn_penalty = 0.5 * jnp.sum(params.η_dyn * jnp.square(dyn_residual))
            equality_penalty = 0.5 * (
                jnp.sum(params.η_eq.node * jnp.square(equality_residual.node))
                + jnp.sum(params.η_eq.edge * jnp.square(equality_residual.edge))
            )
            inequality_penalty = 0.5 * (
                jnp.sum(params.η_ineq.node * jnp.square(inequality_residual.node))
                + jnp.sum(params.η_ineq.edge * jnp.square(inequality_residual.edge))
            )
            merit = (
                local_cost
                + local_node_equality
                + local_node_inequality
                + local_edge_equality
                + local_edge_inequality
                + dynamics_term
                + state_term
                + dyn_penalty
                + equality_penalty
                + inequality_penalty
            )

        if mode.uses_primal_center:
            merit += (
                0.5
                * hessian_regularization
                * (
                    jnp.sum(jnp.square(variables.X - reference_variables.X))
                    + jnp.sum(jnp.square(variables.U - reference_variables.U))
                    + jnp.sum(jnp.square(variables.Theta - reference_variables.Theta))
                )
            )
        return merit

    return augmented_lagrangian


def directional_augmented_lagrangian(  # noqa: ANN201
    node_cost: NodeCostFunction,
    edge_cost: EdgeCostFunction,
    dynamics: EdgeFunction,
    node_equalities: NodeFunction,
    edge_equalities: EdgeFunction,
    node_inequalities: NodeFunction,
    edge_inequalities: EdgeFunction,
    x0: jax.Array,
    params: TreeParameters,
    τ: jnp.double,
    topology: TreeOCPTopology | None,
    locations: OCPCallbackLocations,
    variables: TreeVariables,
    deltas: TreeVariables,
    mode: SolverMode,
    hessian_regularization: jnp.double,
):
    """Define the directional augmented Lagrangian used in line search."""
    augmented_lagrangian = build_total_augmented_lagrangian(
        node_cost=node_cost,
        edge_cost=edge_cost,
        dynamics=dynamics,
        node_equalities=node_equalities,
        edge_equalities=edge_equalities,
        node_inequalities=node_inequalities,
        edge_inequalities=edge_inequalities,
        x0=x0,
        params=params,
        topology=topology,
        locations=locations,
        mode=mode,
        hessian_regularization=hessian_regularization,
        reference_variables=variables,
    )

    def dal(α: jnp.double) -> jnp.double:
        return augmented_lagrangian(
            TreeVariables(
                X=variables.X + α * deltas.X,
                U=variables.U + α * deltas.U,
                S=node_edge_map(
                    lambda value, delta: jnp.maximum(
                        value + α * delta, (1.0 - τ) * value
                    ),
                    variables.S,
                    deltas.S,
                ),
                Y_dyn=(
                    variables.Y_dyn + α * deltas.Y_dyn
                    if mode.uses_dual_center
                    else variables.Y_dyn
                ),
                Y_eq=(
                    node_edge_map(
                        lambda value, delta: value + α * delta,
                        variables.Y_eq,
                        deltas.Y_eq,
                    )
                    if mode.uses_dual_center
                    else variables.Y_eq
                ),
                Z=(
                    node_edge_map(
                        lambda value, delta: jnp.maximum(
                            value + α * delta, (1.0 - τ) * value
                        ),
                        variables.Z,
                        deltas.Z,
                    )
                    if mode.uses_dual_center
                    else variables.Z
                ),
                Theta=variables.Theta + α * deltas.Theta,
            )
        )

    return dal
