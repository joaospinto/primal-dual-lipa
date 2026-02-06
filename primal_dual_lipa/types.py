"""Defines some types."""

from collections.abc import Callable
from dataclasses import dataclass, field

import jax
from jax import numpy as jnp


@jax.tree_util.register_dataclass
@dataclass
class SolverSettings:
    """Encapsulate a few solver settings."""

    max_iterations: jnp.int32 = 500
    residual_sq_threshold: jnp.double = 1e-16
    α_min: jnp.double = 3e-6
    α_update_factor: jnp.double = 0.5
    η0: jnp.double = 1e3
    η_update_factor: jnp.double = 10.0
    η_max: jnp.double = 1e12
    µ0: jnp.double = 1e-3
    µ_update_factor: jnp.double = 0.5
    µ_min: jnp.double = 1e-16
    τ: jnp.double = 0.995
    armijo_factor: jnp.double = 1e-4
    use_parallel_lqr: jnp.bool = field(default=False, metadata={"static": True})
    print_logs: jnp.bool = field(default=False, metadata={"static": True})


Function = Callable[[jnp.ndarray, jnp.ndarray, jnp.int32], jnp.ndarray]
CostFunction = Callable[[jnp.ndarray, jnp.ndarray, jnp.int32], jnp.double]
