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
class SolverSettings:
    """Encapsulate a few solver settings."""

    max_iterations: jnp.int32 = 500
    residual_sq_threshold: jnp.double = 1e-16
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
    armijo_factor: jnp.double = 1e-4
    num_iterative_refinement_steps: jnp.int32 = 0
    num_ruiz_scaling_steps: jnp.int32 = 0
    num_parallel_line_search_steps: jnp.int32 = field(
        default=1, metadata={"static": True}
    )
    use_parallel_lqr: jnp.bool = field(default=False, metadata={"static": True})
    print_logs: jnp.bool = field(default=False, metadata={"static": True})
    print_ls_logs: jnp.bool = field(default=False, metadata={"static": True})


Function = Callable[[jax.Array, jax.Array, jax.Array, jnp.int32], jax.Array]
CostFunction = Callable[[jax.Array, jax.Array, jax.Array, jnp.int32], jnp.double]


@jax.tree_util.register_dataclass
@dataclass
class Parameters:
    """Encapsulate µ and η variables."""

    µ: jnp.double
    η_dyn: jax.Array
    η_eq: jax.Array
    η_ineq: jax.Array


@jax.tree_util.register_dataclass
@dataclass
class KKTFactorizationInputs:
    """Inputs to the KKT factorization."""

    P: jax.Array
    D: jax.Array
    E: jax.Array
    G: jax.Array
    w_inv: jax.Array
    params: Parameters
    H_theta_theta: jax.Array
    H_theta_X: jax.Array
    H_theta_U: jax.Array
    H_theta_y_dyn: jax.Array
    H_theta_y_eq: jax.Array
    H_theta_z: jax.Array


@jax.tree_util.register_dataclass
@dataclass
class KKTFactorizationOutputs:
    """KKT factorization outputs."""

    lqr_inputs: LQRFactorizationInputs
    lqr_outputs: LQRSequentialFactorizationOutputs | LQRParallelFactorizationOutputs
    schur_complement: jax.Array
    B_inv_C_X: jax.Array
    B_inv_C_U: jax.Array
    B_inv_C_S: jax.Array
    B_inv_C_Y_dyn: jax.Array
    B_inv_C_Y_eq: jax.Array
    B_inv_C_Z: jax.Array


@jax.tree_util.register_dataclass
@dataclass
class Variables:
    """Generic variables container."""

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
    rhs: Variables
