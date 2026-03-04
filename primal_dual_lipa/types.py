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
    use_parallel_lqr: jnp.bool = field(default=False, metadata={"static": True})
    print_logs: jnp.bool = field(default=False, metadata={"static": True})
    print_ls_logs: jnp.bool = field(default=False, metadata={"static": True})


Function = Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.int32], jnp.ndarray]
CostFunction = Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.int32], jnp.double]


@jax.tree_util.register_dataclass
@dataclass
class Parameters:
    """Encapsulate µ and η variables."""

    µ: jnp.double
    η_dyn: jnp.ndarray
    η_eq: jnp.ndarray
    η_ineq: jnp.ndarray
    η_x: jnp.ndarray
    η_u: jnp.ndarray
    η_s: jnp.ndarray


@jax.tree_util.register_dataclass
@dataclass
class KKTFactorizationInputs:
    """Inputs to the KKT factorization."""

    P: jnp.ndarray
    D: jnp.ndarray
    E: jnp.ndarray
    G: jnp.ndarray
    w_inv: jnp.ndarray
    params: Parameters
    H_theta_theta: jnp.ndarray
    H_theta_X: jnp.ndarray
    H_theta_U: jnp.ndarray
    H_theta_y_dyn: jnp.ndarray
    H_theta_y_eq: jnp.ndarray
    H_theta_z: jnp.ndarray


@jax.tree_util.register_dataclass
@dataclass
class KKTFactorizationOutputs:
    """KKT factorization outputs."""

    lqr_inputs: LQRFactorizationInputs
    lqr_outputs: LQRSequentialFactorizationOutputs | LQRParallelFactorizationOutputs
    schur_complement: jnp.ndarray
    B_inv_C_X: jnp.ndarray
    B_inv_C_U: jnp.ndarray
    B_inv_C_S: jnp.ndarray
    B_inv_C_Y_dyn: jnp.ndarray
    B_inv_C_Y_eq: jnp.ndarray
    B_inv_C_Z: jnp.ndarray


@jax.tree_util.register_dataclass
@dataclass
class Variables:
    """Generic variables container."""

    X: jnp.ndarray
    U: jnp.ndarray
    S: jnp.ndarray
    Y_dyn: jnp.ndarray
    Y_eq: jnp.ndarray
    Z: jnp.ndarray
    Theta: jnp.ndarray


@jax.tree_util.register_dataclass
@dataclass
class KKTSystem:
    """Encapsulate the KKT system (LHS and RHS)."""

    lhs: KKTFactorizationInputs
    rhs: Variables
