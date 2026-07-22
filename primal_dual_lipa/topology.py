"""Topology and array-layout helpers for chain and rooted-tree OCPs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import numpy as np
from jax import numpy as jnp
from jax_bidirectional_tree_rake_compress import (
    ContractionExecutor,
    ContractionSchedule,
    TreeContractionPlan,
    make_tree_contraction_plan,
)

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

from primal_dual_lipa.types import (
    NodeAndEdgeIndices,
    NodeAndEdgeValues,
    OCPCallbackLocations,
)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, eq=False)
class TreeOCPTopology:
    """Static rooted-tree layout used by the nonlinear OCP solver.

    Node arrays retain the caller's node order. Edge arrays follow
    ``plan.edge_children``.
    """

    plan: TreeContractionPlan
    use_parallel_lqr: bool

    def tree_flatten(self) -> tuple[tuple[TreeContractionPlan], bool]:
        """Keep the solver mode as static PyTree metadata."""
        return (self.plan,), self.use_parallel_lqr

    @classmethod
    def tree_unflatten(
        cls,
        use_parallel_lqr: bool,
        children: tuple[TreeContractionPlan],
    ) -> TreeOCPTopology:
        """Reconstruct a topology after a JAX transformation."""
        (plan,) = children
        return cls(plan=plan, use_parallel_lqr=use_parallel_lqr)

    @property
    def num_nodes(self) -> int:
        """Return the number of state nodes."""
        return self.plan.num_nodes

    @property
    def num_edges(self) -> int:
        """Return the number of dynamics/control edges."""
        return self.plan.num_edges


def make_tree_ocp_topology(
    parents: ArrayLike,
    *,
    use_parallel_lqr: bool,
    root: int | None = None,
) -> TreeOCPTopology:
    """Validate and preprocess one rooted-tree schedule on the host."""
    use_parallel_lqr = bool(use_parallel_lqr)
    schedule = (
        ContractionSchedule.RAKE_COMPRESS
        if use_parallel_lqr
        else ContractionSchedule.RAKE_ONLY
    )
    plan = make_tree_contraction_plan(
        parents,
        root=root,
        schedule=schedule,
        executor=ContractionExecutor.AUTO,
    )
    return TreeOCPTopology(plan=plan, use_parallel_lqr=use_parallel_lqr)


def validate_tree_ocp_schedule(
    topology: TreeOCPTopology | None,
    *,
    use_parallel_lqr: bool,
) -> None:
    """Reject a topology planned for a different LQR execution mode."""
    if topology is None:
        return
    if topology.use_parallel_lqr != bool(use_parallel_lqr):
        message = (
            "topology was created with "
            f"use_parallel_lqr={topology.use_parallel_lqr}, but solver settings "
            f"use_parallel_lqr={bool(use_parallel_lqr)}"
        )
        raise ValueError(message)


def root_node(topology: TreeOCPTopology | None) -> jax.Array | int:
    """Return the root node; chains without explicit topology start at zero."""
    return 0 if topology is None else topology.plan.root


def edge_parents(
    topology: TreeOCPTopology | None,
    num_edges: int,
) -> jax.Array:
    """Return one parent-node index per dynamics/control edge."""
    if topology is None:
        return jnp.arange(num_edges, dtype=jnp.int32)
    return topology.plan.edge_parents


def edge_children(
    topology: TreeOCPTopology | None,
    num_edges: int,
) -> jax.Array:
    """Return one child-node index per dynamics/control edge."""
    if topology is None:
        return jnp.arange(1, num_edges + 1, dtype=jnp.int32)
    return topology.plan.edge_children


def validate_callback_locations(
    locations: OCPCallbackLocations,
    *,
    num_nodes: int,
    num_edges: int,
) -> OCPCallbackLocations:
    """Validate callback indices on the host and normalize them to int32 arrays."""

    def validate_indices(
        indices: jax.Array,
        *,
        callback: str,
        domain: str,
        upper_bound: int,
    ) -> jax.Array:
        host_indices = np.asarray(indices)
        name = f"locations.{callback}.{domain}"
        if host_indices.ndim != 1:
            message = f"{name} must be one-dimensional; got shape {host_indices.shape}"
            raise ValueError(message)
        if not np.issubdtype(host_indices.dtype, np.integer):
            message = f"{name} must contain integer indices; got {host_indices.dtype}"
            raise ValueError(message)
        if np.any(host_indices < 0) or np.any(host_indices >= upper_bound):
            message = (
                f"{name} entries must be in [0, {upper_bound}); got {host_indices}"
            )
            raise ValueError(message)
        if np.unique(host_indices).size != host_indices.size:
            message = f"{name} must not contain duplicate indices; got {host_indices}"
            raise ValueError(message)
        return jnp.asarray(host_indices, dtype=jnp.int32)

    normalized = {}
    for callback in ("cost", "equalities", "inequalities"):
        selected = getattr(locations, callback)
        normalized[callback] = NodeAndEdgeIndices(
            node=validate_indices(
                selected.node,
                callback=callback,
                domain="node",
                upper_bound=num_nodes,
            ),
            edge=validate_indices(
                selected.edge,
                callback=callback,
                domain="edge",
                upper_bound=num_edges,
            ),
        )
    return OCPCallbackLocations(**normalized)


def validate_tree_shapes(
    topology: TreeOCPTopology | None,
    *,
    X: jax.Array,
    U: jax.Array,
    S: NodeAndEdgeValues,
    Y_dyn: jax.Array,
    Y_eq: NodeAndEdgeValues,
    Z: NodeAndEdgeValues,
    locations: OCPCallbackLocations,
) -> None:
    """Raise an actionable error for inconsistent chain/tree warm starts."""
    num_nodes = X.shape[0]
    expected_edges = num_nodes - 1
    if topology is not None and topology.num_nodes != num_nodes:
        message = f"X has {num_nodes} nodes but topology has {topology.num_nodes}"
        raise ValueError(message)
    if U.shape[0] != expected_edges:
        message = f"U must have one row per edge ({expected_edges}); got {U.shape[0]}"
        raise ValueError(message)
    if Y_dyn.shape != X.shape:
        message = f"Y_dyn must have the same shape as X ({X.shape}); got {Y_dyn.shape}"
        raise ValueError(message)
    expected_locations = {
        "S": locations.inequalities,
        "Y_eq": locations.equalities,
        "Z": locations.inequalities,
    }
    for name, value in (("S", S), ("Y_eq", Y_eq), ("Z", Z)):
        selected = expected_locations[name]
        for domain in ("node", "edge"):
            actual_rows = getattr(value, domain).shape[0]
            expected_rows = getattr(selected, domain).shape[0]
            if actual_rows != expected_rows:
                message = (
                    f"{name}.{domain} must have one row per selected {domain} "
                    f"({expected_rows}); got {actual_rows}"
                )
                raise ValueError(message)
    if S.node.shape != Z.node.shape or S.edge.shape != Z.edge.shape:
        message = (
            "S and Z must have identical node and edge shapes; got "
            f"{S.node.shape}/{S.edge.shape} and {Z.node.shape}/{Z.edge.shape}"
        )
        raise ValueError(message)
