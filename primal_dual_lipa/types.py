"""Defines some types."""

from collections.abc import Callable
from dataclasses import dataclass, field

import jax
from jax import numpy as jnp
from regularized_lqr_jax.types import (
    FactorizationInputs as LQRFactorizationInputs,
)
from regularized_lqr_jax.types import (
    ParallelFactorizationOutputs as LQRParallelFactorizationOutputs,
)
from regularized_lqr_jax.types import (
    SequentialFactorizationOutputs as LQRSequentialFactorizationOutputs,
)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class HessianRegularizationSettings:
    """Settings for IPOPT-style primal Hessian regularization."""

    initial: jnp.double = 0.0
    first_positive: jnp.double = 1e-8
    minimum: jnp.double = 0.0
    maximum: jnp.double = 1e8
    increase_factor: jnp.double = 10.0
    decrease_factor: jnp.double = 0.1
    pd_tol: jnp.double = 0.0
    singular_tol: jnp.double = 0.0
    descent_tol: jnp.double = 0.0
    max_attempts: jnp.int32 = field(default=18, metadata={"static": True})


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SolverSettings:
    """Encapsulate a few solver settings."""

    max_iterations: jnp.int32 = 500
    residual_sq_threshold: jnp.double = 1e-16
    # Auxiliary SQP-style termination criterion: declared converged when
    # the cost improvement and the primal violation (inf-norm of
    # init/dyn defects, equality residuals, and positive-part inequality
    # residuals) both fall strictly below their thresholds. Both default
    # to 0, which disables this criterion (no real value of either is < 0).
    # LIPA terminates if EITHER this criterion or `residual_sq_threshold`
    # is satisfied.
    cost_improvement_threshold: jnp.double = 0.0
    primal_violation_threshold: jnp.double = 0.0
    α_min: jnp.double = 3e-6
    α_update_factor: jnp.double = 0.5
    η0: jnp.double = 1e3
    η_update_factor: jnp.double = 2.0
    η_max: jnp.double = 1e12
    η_improvement_threshold: jnp.double = 0.9
    µ0: jnp.double = 1e-3
    µ_update_factor: jnp.double = 0.8
    µ_min: jnp.double = 1e-16
    τ_min: jnp.double = 0.995
    κ: jnp.double = 10.0
    # Mehrotra-style centering µ update: when True, replace the default
    # residual-gated rule with a complementarity-tracking rule
    #
    #     µ_target = σ · mean(S·Z)
    #     µ_new    = clamp(µ_target,
    #                      µ_update_factor · µ,    # lower clamp
    #                      µ)                      # non-increasing
    #
    # Default False preserves the existing residual-gated rule.
    mehrotra_mu: jnp.bool = field(default=False, metadata={"static": True})
    # Centering parameter for the Mehrotra rule above. Lower σ is more
    # "predictor-like" (faster complementarity tightening); higher σ is
    # more "corrector-like" (more centered iterates). σ = 0.1 is the
    # textbook conservative default. Only consulted when ``mehrotra_mu``
    # is True.
    mehrotra_sigma: jnp.double = 0.1
    armijo_factor: jnp.double = 1e-4
    # Filter line-search (Wächter–Biegler 2006, Math. Prog. 106) —
    # optional alternative to the merit-function Armijo line search.
    # When enabled, a trial step is accepted if it improves EITHER the
    # barrier-augmented objective f OR the primal-violation measure θ
    # separately, rather than their merit-function combination. We
    # maintain a single (θ, f) envelope as the "filter" (the
    # Fletcher–Leyffer simplification), bounding the jitted while_loop
    # carry. The Armijo check is ORed with the filter check, so the
    # filter strictly accepts a superset of merit-Armijo alphas.
    use_filter_line_search: jnp.bool = field(default=False, metadata={"static": True})
    # Filter "wedge" margins: a trial point (θ_α, f_α) is acceptable iff
    #   θ_α ≤ (1 − γ_θ) · θ_F  OR  f_α ≤ f_F − γ_f · θ_F
    # for the current envelope (θ_F, f_F). Defaults follow the IPOPT
    # paper §3.2 (Eqns. 18-19).
    filter_gamma_theta: jnp.double = 1e-5
    filter_gamma_f: jnp.double = 1e-5
    num_iterative_refinement_steps: jnp.int32 = 0
    num_parallel_line_search_steps: jnp.int32 = field(
        default=1, metadata={"static": True}
    )
    use_parallel_lqr: jnp.bool = field(default=False, metadata={"static": True})
    skip_line_search: jnp.bool = field(default=False, metadata={"static": True})
    # Add a barrier-scaled diagonal term to W^{-1}=Z/S in the slack
    # elimination. False matches the regularized-IPM derivation exactly;
    # True preserves the historically useful extra regularization.
    regularize_slack_elimination_with_mu: jnp.bool = field(
        default=True,
        metadata={"static": True},
    )
    print_logs: jnp.bool = field(default=False, metadata={"static": True})
    print_ls_logs: jnp.bool = field(default=False, metadata={"static": True})
    hessian_regularization: HessianRegularizationSettings = field(
        default_factory=HessianRegularizationSettings
    )


NodeFunction = Callable[[jax.Array, jax.Array, jnp.int32], jax.Array]
NodeCostFunction = Callable[[jax.Array, jax.Array, jnp.int32], jnp.double]
EdgeFunction = Callable[[jax.Array, jax.Array, jax.Array, jnp.int32], jax.Array]
EdgeCostFunction = Callable[[jax.Array, jax.Array, jax.Array, jnp.int32], jnp.double]
Function = EdgeFunction
CostFunction = EdgeCostFunction


@jax.tree_util.register_dataclass
@dataclass
class NodeAndEdgeValues:
    """Values stored in independent node and edge row sets."""

    node: jax.Array
    edge: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class NodeAndEdgeIndices:
    """Node and edge indices at which a callback is evaluated."""

    node: jax.Array
    edge: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class OCPCallbackLocations:
    """Locations carrying cost, equality, and inequality callbacks."""

    cost: NodeAndEdgeIndices
    equalities: NodeAndEdgeIndices
    inequalities: NodeAndEdgeIndices


def node_edge_map(fun: Callable, *values: NodeAndEdgeValues) -> NodeAndEdgeValues:
    """Apply ``fun`` to corresponding node and edge arrays."""
    return NodeAndEdgeValues(
        node=fun(*(value.node for value in values)),
        edge=fun(*(value.edge for value in values)),
    )


def node_edge_flatten(value: NodeAndEdgeValues) -> jax.Array:
    """Flatten and concatenate the node and edge arrays."""
    return jnp.concatenate([value.node.reshape(-1), value.edge.reshape(-1)])


def node_edge_sum(value: NodeAndEdgeValues) -> jax.Array:
    """Sum all entries in the node and edge arrays."""
    return jnp.sum(value.node) + jnp.sum(value.edge)


@jax.tree_util.register_dataclass
@dataclass
class TreeParameters:
    """Penalty parameters split into explicit node and edge blocks."""

    µ: jnp.double
    η_dyn: jax.Array
    η_eq: NodeAndEdgeValues
    η_ineq: NodeAndEdgeValues


@jax.tree_util.register_dataclass
@dataclass
class Parameters:
    """Penalty parameters for a chain's ``T + 1`` local stages."""

    µ: jnp.double
    η_dyn: jax.Array
    η_eq: jax.Array
    η_ineq: jax.Array


@jax.tree_util.register_dataclass
@dataclass
class KKTFactorizationInputs:
    """Inputs to the KKT factorization."""

    Q: jax.Array
    M: jax.Array
    R: jax.Array
    Q_lqr: jax.Array
    M_lqr: jax.Array
    R_lqr: jax.Array
    D: jax.Array
    E: NodeAndEdgeValues
    G: NodeAndEdgeValues
    w_inv: NodeAndEdgeValues
    params: TreeParameters
    H_theta_theta: jax.Array
    H_theta_X: jax.Array
    H_theta_U: jax.Array
    H_theta_y_dyn: jax.Array
    H_theta_y_eq: NodeAndEdgeValues
    H_theta_z: NodeAndEdgeValues
    equality_locations: NodeAndEdgeIndices
    inequality_locations: NodeAndEdgeIndices


@jax.tree_util.register_dataclass
@dataclass
class KKTFactorizationOutputs:
    """KKT factorization outputs."""

    lqr_inputs: LQRFactorizationInputs
    lqr_outputs: LQRSequentialFactorizationOutputs | LQRParallelFactorizationOutputs
    schur_complement: jax.Array
    B_inv_C_X: jax.Array
    B_inv_C_U: jax.Array
    B_inv_C_S: NodeAndEdgeValues
    B_inv_C_Y_dyn: jax.Array
    B_inv_C_Y_eq: NodeAndEdgeValues
    B_inv_C_Z: NodeAndEdgeValues


@jax.tree_util.register_dataclass
@dataclass
class TreeVariables:
    """Variables with independent node and edge constraint blocks."""

    X: jax.Array
    U: jax.Array
    S: NodeAndEdgeValues
    Y_dyn: jax.Array
    Y_eq: NodeAndEdgeValues
    Z: NodeAndEdgeValues
    Theta: jax.Array


@jax.tree_util.register_dataclass
@dataclass
class Variables:
    """Variables for a chain's ``T + 1`` local stages."""

    X: jax.Array
    U: jax.Array
    S: jax.Array
    Y_dyn: jax.Array
    Y_eq: jax.Array
    Z: jax.Array
    Theta: jax.Array


@jax.tree_util.register_dataclass
@dataclass
class KKTSystem:
    """Encapsulate the KKT system (LHS and RHS)."""

    lhs: KKTFactorizationInputs
    rhs: TreeVariables
