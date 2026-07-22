"""Build the topology-aware Newton-KKT system used by primal-dual LIPA."""

from functools import partial

import jax
from jax import numpy as jnp
from regularized_lqr_jax.helpers import symmetrize

from primal_dual_lipa.lagrangian_helpers import (
    build_dynamics_lagrangian,
    build_edge_equality_lagrangian,
    build_edge_inequality_lagrangian,
    build_node_equality_lagrangian,
    build_node_inequality_lagrangian,
    dynamics_residuals,
    evaluate_node_edge,
)
from primal_dual_lipa.topology import TreeOCPTopology, edge_children, edge_parents
from primal_dual_lipa.types import (
    EdgeCostFunction,
    EdgeFunction,
    KKTFactorizationInputs,
    KKTSystem,
    NodeAndEdgeIndices,
    NodeAndEdgeValues,
    NodeCostFunction,
    NodeFunction,
    OCPCallbackLocations,
    TreeParameters,
    TreeVariables,
    node_edge_map,
)
from primal_dual_lipa.vectorization_helpers import (
    linearize_edge,
    linearize_node,
    quadratize_edge,
    quadratize_node,
)


@jax.jit
def regularize_primal_hessian_blocks(
    Q: jax.Array,
    R: jax.Array,
    H_theta_theta: jax.Array,
    hessian_regularization: jnp.double,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Symmetrize Hessian blocks and shift every primal variable once."""
    Q = jax.vmap(symmetrize)(Q)
    R = jax.vmap(symmetrize)(R)
    H_theta_theta = symmetrize(H_theta_theta)

    Q = Q + hessian_regularization * jnp.eye(Q.shape[-1])
    R = R + hessian_regularization * jnp.eye(R.shape[-1])
    H_theta_theta = H_theta_theta + hessian_regularization * jnp.eye(
        H_theta_theta.shape[-1]
    )
    return Q, R, H_theta_theta


@jax.jit
def add_scalar_hessian_regularization_delta(
    kkt_system: KKTSystem,
    hessian_regularization_delta: jnp.double,
) -> KKTSystem:
    """Shift cached primal Hessian blocks after a regularization change."""
    lhs = kkt_system.lhs
    theta_dim = lhs.H_theta_theta.shape[-1]
    Q_shift = hessian_regularization_delta * jnp.eye(lhs.Q.shape[-1])
    R_shift = hessian_regularization_delta * jnp.eye(lhs.R.shape[-1])
    H_theta_theta_shift = hessian_regularization_delta * jnp.eye(theta_dim)
    lhs_new = KKTFactorizationInputs(
        Q=lhs.Q + Q_shift,
        M=lhs.M,
        R=lhs.R + R_shift,
        Q_lqr=lhs.Q_lqr + Q_shift,
        M_lqr=lhs.M_lqr,
        R_lqr=lhs.R_lqr + R_shift,
        D=lhs.D,
        E=lhs.E,
        G=lhs.G,
        w_inv=lhs.w_inv,
        params=lhs.params,
        H_theta_theta=lhs.H_theta_theta + H_theta_theta_shift,
        H_theta_X=lhs.H_theta_X,
        H_theta_U=lhs.H_theta_U,
        H_theta_y_dyn=lhs.H_theta_y_dyn,
        H_theta_y_eq=lhs.H_theta_y_eq,
        H_theta_z=lhs.H_theta_z,
        equality_locations=lhs.equality_locations,
        inequality_locations=lhs.inequality_locations,
    )
    return KKTSystem(lhs=lhs_new, rhs=kkt_system.rhs)


_STATIC_CALLBACKS = (
    "node_cost",
    "edge_cost",
    "dynamics",
    "node_equalities",
    "edge_equalities",
    "node_inequalities",
    "edge_inequalities",
)


def _all_callback_locations(vars: TreeVariables) -> OCPCallbackLocations:  # noqa: A002
    """Return the default layout in which every callback runs everywhere."""
    all_locations = NodeAndEdgeIndices(
        node=jnp.arange(vars.X.shape[0], dtype=jnp.int32),
        edge=jnp.arange(vars.U.shape[0], dtype=jnp.int32),
    )
    return OCPCallbackLocations(
        cost=all_locations,
        equalities=all_locations,
        inequalities=all_locations,
    )


@partial(
    jax.jit,
    static_argnames=(*_STATIC_CALLBACKS, "regularize_slack_elimination_with_mu"),
)
def build_kkt_lhs(
    node_cost: NodeCostFunction,
    edge_cost: EdgeCostFunction,
    dynamics: EdgeFunction,
    node_equalities: NodeFunction,
    edge_equalities: EdgeFunction,
    node_inequalities: NodeFunction,
    edge_inequalities: EdgeFunction,
    x0: jax.Array,
    vars: TreeVariables,  # noqa: A002
    params: TreeParameters,
    hessian_regularization: jnp.double,
    regularize_slack_elimination_with_mu: bool,
    topology: TreeOCPTopology | None = None,
    locations: OCPCallbackLocations | None = None,
) -> KKTFactorizationInputs:
    """Build the LHS for a chain or rooted-tree Newton-KKT system."""
    del x0  # The initial-state term is affine and has no LHS contribution.
    num_nodes = vars.X.shape[0]
    num_edges = vars.U.shape[0]
    x_dim = vars.X.shape[1]
    parents = edge_parents(topology, num_edges)
    children = edge_children(topology, num_edges)
    edge_indices = jnp.arange(num_edges, dtype=jnp.int32)
    if locations is None:
        locations = _all_callback_locations(vars)

    cost_nodes = locations.cost.node
    Q_node_cost, Htt_node_cost, Hx_theta_node_cost = quadratize_node(node_cost)(
        vars.X[cost_nodes], vars.Theta, cost_nodes
    )
    equality_nodes = locations.equalities.node
    Q_node_eq, Htt_node_eq, Hx_theta_node_eq = quadratize_node(
        build_node_equality_lagrangian(node_equalities)
    )(
        vars.X[equality_nodes],
        vars.Theta,
        equality_nodes,
        vars.Y_eq.node,
    )
    inequality_nodes = locations.inequalities.node
    Q_node_ineq, Htt_node_ineq, Hx_theta_node_ineq = quadratize_node(
        build_node_inequality_lagrangian(node_inequalities, params.µ)
    )(
        vars.X[inequality_nodes],
        vars.Theta,
        inequality_nodes,
        vars.S.node,
        vars.Z.node,
    )

    cost_edges = locations.cost.edge
    (
        Q_edge_cost,
        R_edge_cost,
        M_edge_cost,
        Htt_edge_cost,
        Hx_theta_edge_cost,
        Hu_theta_edge_cost,
    ) = quadratize_edge(edge_cost)(
        vars.X[parents[cost_edges]],
        vars.U[cost_edges],
        vars.Theta,
        cost_edges,
    )
    equality_edges = locations.equalities.edge
    (
        Q_edge_eq,
        R_edge_eq,
        M_edge_eq,
        Htt_edge_eq,
        Hx_theta_edge_eq,
        Hu_theta_edge_eq,
    ) = quadratize_edge(build_edge_equality_lagrangian(edge_equalities))(
        vars.X[parents[equality_edges]],
        vars.U[equality_edges],
        vars.Theta,
        equality_edges,
        vars.Y_eq.edge,
    )
    inequality_edges = locations.inequalities.edge
    (
        Q_edge_ineq,
        R_edge_ineq,
        M_edge_ineq,
        Htt_edge_ineq,
        Hx_theta_edge_ineq,
        Hu_theta_edge_ineq,
    ) = quadratize_edge(build_edge_inequality_lagrangian(edge_inequalities, params.µ))(
        vars.X[parents[inequality_edges]],
        vars.U[inequality_edges],
        vars.Theta,
        inequality_edges,
        vars.S.edge,
        vars.Z.edge,
    )

    (
        Q_edge_dyn,
        R_edge_dyn,
        M_edge_dyn,
        Htt_edge_dyn,
        Hx_theta_edge_dyn,
        Hu_theta_edge_dyn,
    ) = quadratize_edge(build_dynamics_lagrangian(dynamics))(
        vars.X[parents],
        vars.U,
        vars.Theta,
        edge_indices,
        vars.Y_dyn[children],
    )

    theta_dim = vars.Theta.shape[0]
    Q = jnp.zeros((num_nodes, x_dim, x_dim), dtype=vars.X.dtype)
    H_theta_X = jnp.zeros((num_nodes, x_dim, theta_dim), dtype=vars.X.dtype)
    H_theta_theta = Htt_edge_dyn.sum(axis=0)
    for indices, q, htt, hx_theta in (
        (cost_nodes, Q_node_cost, Htt_node_cost, Hx_theta_node_cost),
        (equality_nodes, Q_node_eq, Htt_node_eq, Hx_theta_node_eq),
        (inequality_nodes, Q_node_ineq, Htt_node_ineq, Hx_theta_node_ineq),
    ):
        Q = Q.at[indices].add(q)
        H_theta_X = H_theta_X.at[indices].add(hx_theta)
        H_theta_theta = H_theta_theta + htt.sum(axis=0)

    Q = Q.at[parents].add(Q_edge_dyn)
    H_theta_X = H_theta_X.at[parents].add(Hx_theta_edge_dyn)
    M = M_edge_dyn
    R = R_edge_dyn
    H_theta_U = Hu_theta_edge_dyn
    for indices, q, r, m, htt, hx_theta, hu_theta in (
        (
            cost_edges,
            Q_edge_cost,
            R_edge_cost,
            M_edge_cost,
            Htt_edge_cost,
            Hx_theta_edge_cost,
            Hu_theta_edge_cost,
        ),
        (
            equality_edges,
            Q_edge_eq,
            R_edge_eq,
            M_edge_eq,
            Htt_edge_eq,
            Hx_theta_edge_eq,
            Hu_theta_edge_eq,
        ),
        (
            inequality_edges,
            Q_edge_ineq,
            R_edge_ineq,
            M_edge_ineq,
            Htt_edge_ineq,
            Hx_theta_edge_ineq,
            Hu_theta_edge_ineq,
        ),
    ):
        Q = Q.at[parents[indices]].add(q)
        M = M.at[indices].add(m)
        R = R.at[indices].add(r)
        H_theta_X = H_theta_X.at[parents[indices]].add(hx_theta)
        H_theta_U = H_theta_U.at[indices].add(hu_theta)
        H_theta_theta = H_theta_theta + htt.sum(axis=0)

    Q, R, H_theta_theta = regularize_primal_hessian_blocks(
        Q, R, H_theta_theta, hessian_regularization
    )

    A, B, H_theta_y_dyn_edges = linearize_edge(dynamics)(
        vars.X[parents], vars.U, vars.Theta, edge_indices
    )
    D = jnp.concatenate([A, B], axis=-1)

    E_node, H_theta_y_eq_node = linearize_node(node_equalities)(
        vars.X[equality_nodes], vars.Theta, equality_nodes
    )
    E_edge_x, E_edge_u, H_theta_y_eq_edge = linearize_edge(edge_equalities)(
        vars.X[parents[equality_edges]],
        vars.U[equality_edges],
        vars.Theta,
        equality_edges,
    )
    E = NodeAndEdgeValues(
        node=E_node,
        edge=jnp.concatenate([E_edge_x, E_edge_u], axis=-1),
    )

    G_node, H_theta_z_node = linearize_node(node_inequalities)(
        vars.X[inequality_nodes], vars.Theta, inequality_nodes
    )
    G_edge_x, G_edge_u, H_theta_z_edge = linearize_edge(edge_inequalities)(
        vars.X[parents[inequality_edges]],
        vars.U[inequality_edges],
        vars.Theta,
        inequality_edges,
    )
    G = NodeAndEdgeValues(
        node=G_node,
        edge=jnp.concatenate([G_edge_x, G_edge_u], axis=-1),
    )

    w_inv = node_edge_map(lambda z, s: jnp.clip(z / s, 1e-8, 1e8), vars.Z, vars.S)
    if regularize_slack_elimination_with_mu:
        w_inv = node_edge_map(lambda value: value + params.µ, w_inv)
    w = node_edge_map(lambda value: 1.0 / value, w_inv)
    reg_w_inv = node_edge_map(
        lambda value, eta: 1.0 / (value + 1.0 / eta), w, params.η_ineq
    )
    equality_hessian_node = jnp.einsum(
        "nki,nk,nkj->nij", E.node, params.η_eq.node, E.node
    )
    inequality_hessian_node = jnp.einsum(
        "ngi,ng,ngj->nij", G.node, reg_w_inv.node, G.node
    )
    equality_hessian_edge = jnp.einsum(
        "eki,ek,ekj->eij", E.edge, params.η_eq.edge, E.edge
    )
    inequality_hessian_edge = jnp.einsum(
        "egi,eg,egj->eij", G.edge, reg_w_inv.edge, G.edge
    )
    Q_lqr = Q.at[equality_nodes].add(equality_hessian_node)
    Q_lqr = Q_lqr.at[inequality_nodes].add(inequality_hessian_node)
    Q_lqr = Q_lqr.at[parents[equality_edges]].add(
        equality_hessian_edge[:, :x_dim, :x_dim]
    )
    Q_lqr = Q_lqr.at[parents[inequality_edges]].add(
        inequality_hessian_edge[:, :x_dim, :x_dim]
    )
    M_lqr = M.at[equality_edges].add(equality_hessian_edge[:, :x_dim, x_dim:])
    M_lqr = M_lqr.at[inequality_edges].add(inequality_hessian_edge[:, :x_dim, x_dim:])
    R_lqr = R.at[equality_edges].add(equality_hessian_edge[:, x_dim:, x_dim:])
    R_lqr = R_lqr.at[inequality_edges].add(inequality_hessian_edge[:, x_dim:, x_dim:])

    H_theta_y_dyn = (
        jnp.zeros((num_nodes, x_dim, theta_dim), dtype=H_theta_y_dyn_edges.dtype)
        .at[children]
        .add(H_theta_y_dyn_edges)
    )

    return KKTFactorizationInputs(
        Q=Q,
        M=M,
        R=R,
        Q_lqr=Q_lqr,
        M_lqr=M_lqr,
        R_lqr=R_lqr,
        D=D,
        E=E,
        G=G,
        w_inv=w_inv,
        params=params,
        H_theta_theta=H_theta_theta,
        H_theta_X=H_theta_X,
        H_theta_U=H_theta_U,
        H_theta_y_dyn=H_theta_y_dyn,
        H_theta_y_eq=NodeAndEdgeValues(node=H_theta_y_eq_node, edge=H_theta_y_eq_edge),
        H_theta_z=NodeAndEdgeValues(node=H_theta_z_node, edge=H_theta_z_edge),
        equality_locations=locations.equalities,
        inequality_locations=locations.inequalities,
    )


@partial(jax.jit, static_argnames=_STATIC_CALLBACKS)
def build_kkt_rhs(
    node_cost: NodeCostFunction,
    edge_cost: EdgeCostFunction,
    dynamics: EdgeFunction,
    node_equalities: NodeFunction,
    edge_equalities: EdgeFunction,
    node_inequalities: NodeFunction,
    edge_inequalities: EdgeFunction,
    x0: jax.Array,
    vars: TreeVariables,  # noqa: A002
    params: TreeParameters,
    topology: TreeOCPTopology | None = None,
    locations: OCPCallbackLocations | None = None,
) -> TreeVariables:
    """Build the RHS for a chain or rooted-tree Newton-KKT system."""
    num_edges = vars.U.shape[0]
    parents = edge_parents(topology, num_edges)
    children = edge_children(topology, num_edges)
    edge_indices = jnp.arange(num_edges, dtype=jnp.int32)
    if locations is None:
        locations = _all_callback_locations(vars)

    cost_nodes = locations.cost.node
    q_node_cost, theta_node_cost = linearize_node(node_cost)(
        vars.X[cost_nodes], vars.Theta, cost_nodes
    )
    equality_nodes = locations.equalities.node
    q_node_eq, theta_node_eq = linearize_node(
        build_node_equality_lagrangian(node_equalities)
    )(
        vars.X[equality_nodes],
        vars.Theta,
        equality_nodes,
        vars.Y_eq.node,
    )
    inequality_nodes = locations.inequalities.node
    q_node_ineq, theta_node_ineq = linearize_node(
        build_node_inequality_lagrangian(node_inequalities, params.µ)
    )(
        vars.X[inequality_nodes],
        vars.Theta,
        inequality_nodes,
        vars.S.node,
        vars.Z.node,
    )

    cost_edges = locations.cost.edge
    q_edge_cost, r_edge_cost, theta_edge_cost = linearize_edge(edge_cost)(
        vars.X[parents[cost_edges]],
        vars.U[cost_edges],
        vars.Theta,
        cost_edges,
    )
    equality_edges = locations.equalities.edge
    q_edge_eq, r_edge_eq, theta_edge_eq = linearize_edge(
        build_edge_equality_lagrangian(edge_equalities)
    )(
        vars.X[parents[equality_edges]],
        vars.U[equality_edges],
        vars.Theta,
        equality_edges,
        vars.Y_eq.edge,
    )
    inequality_edges = locations.inequalities.edge
    q_edge_ineq, r_edge_ineq, theta_edge_ineq = linearize_edge(
        build_edge_inequality_lagrangian(edge_inequalities, params.µ)
    )(
        vars.X[parents[inequality_edges]],
        vars.U[inequality_edges],
        vars.Theta,
        inequality_edges,
        vars.S.edge,
        vars.Z.edge,
    )
    q_edge_dyn, r_edge_dyn, theta_edge_dyn = linearize_edge(
        build_dynamics_lagrangian(dynamics)
    )(
        vars.X[parents],
        vars.U,
        vars.Theta,
        edge_indices,
        vars.Y_dyn[children],
    )

    q = jnp.zeros_like(vars.X)
    theta = theta_edge_dyn.sum(axis=0)
    for indices, q_local, theta_local in (
        (cost_nodes, q_node_cost, theta_node_cost),
        (equality_nodes, q_node_eq, theta_node_eq),
        (inequality_nodes, q_node_ineq, theta_node_ineq),
    ):
        q = q.at[indices].add(q_local)
        theta = theta + theta_local.sum(axis=0)
    q = q.at[parents].add(q_edge_dyn) - vars.Y_dyn
    r = r_edge_dyn
    for indices, q_local, r_local, theta_local in (
        (cost_edges, q_edge_cost, r_edge_cost, theta_edge_cost),
        (equality_edges, q_edge_eq, r_edge_eq, theta_edge_eq),
        (inequality_edges, q_edge_ineq, r_edge_ineq, theta_edge_ineq),
    ):
        q = q.at[parents[indices]].add(q_local)
        r = r.at[indices].add(r_local)
        theta = theta + theta_local.sum(axis=0)
    r_s = node_edge_map(lambda z, s: z - params.µ / s, vars.Z, vars.S)
    r_y_dyn = dynamics_residuals(dynamics, x0, vars, topology)
    r_y_eq = evaluate_node_edge(
        node_equalities,
        edge_equalities,
        vars.X,
        vars.U,
        vars.Theta,
        topology,
        locations.equalities,
    )
    r_z = node_edge_map(
        lambda value, slack: value + slack,
        evaluate_node_edge(
            node_inequalities,
            edge_inequalities,
            vars.X,
            vars.U,
            vars.Theta,
            topology,
            locations.inequalities,
        ),
        vars.S,
    )

    return TreeVariables(
        X=q,
        U=r,
        S=r_s,
        Y_dyn=r_y_dyn,
        Y_eq=r_y_eq,
        Z=r_z,
        Theta=theta,
    )


@partial(
    jax.jit,
    static_argnames=(*_STATIC_CALLBACKS, "regularize_slack_elimination_with_mu"),
)
def build_kkt(
    node_cost: NodeCostFunction,
    edge_cost: EdgeCostFunction,
    dynamics: EdgeFunction,
    node_equalities: NodeFunction,
    edge_equalities: EdgeFunction,
    node_inequalities: NodeFunction,
    edge_inequalities: EdgeFunction,
    x0: jax.Array,
    vars: TreeVariables,  # noqa: A002
    params: TreeParameters,
    hessian_regularization: jnp.double = 0.0,
    regularize_slack_elimination_with_mu: bool = True,
    topology: TreeOCPTopology | None = None,
    locations: OCPCallbackLocations | None = None,
) -> KKTSystem:
    """Build both sides of a chain or rooted-tree Newton-KKT system."""
    callbacks = {
        "node_cost": node_cost,
        "edge_cost": edge_cost,
        "dynamics": dynamics,
        "node_equalities": node_equalities,
        "edge_equalities": edge_equalities,
        "node_inequalities": node_inequalities,
        "edge_inequalities": edge_inequalities,
    }
    return KKTSystem(
        lhs=build_kkt_lhs(
            **callbacks,
            x0=x0,
            vars=vars,
            params=params,
            hessian_regularization=hessian_regularization,
            regularize_slack_elimination_with_mu=regularize_slack_elimination_with_mu,
            topology=topology,
            locations=locations,
        ),
        rhs=build_kkt_rhs(
            **callbacks,
            x0=x0,
            vars=vars,
            params=params,
            topology=topology,
            locations=locations,
        ),
    )
