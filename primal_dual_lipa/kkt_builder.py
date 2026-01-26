"""Provides the helper method for building the Newton-KKT system.

This is used to compute the line search direction at each optimization step.
"""

import jax
from jax import numpy as jnp
from regularized_lqr_jax.helpers import regularize

from primal_dual_lipa.lagrangian_helpers import build_lagrangian, pad
from primal_dual_lipa.types import Function
from primal_dual_lipa.vectorization_helpers import linearize, quadratize, vectorize


def kkt_builder(
    cost: Function,
    dynamics: Function,
    equalities: Function,
    inequalities: Function,
    x0: jnp.ndarray,
    X: jnp.ndarray,
    U: jnp.ndarray,
    S: jnp.ndarray,
    Y_dyn: jnp.ndarray,
    Y_eq: jnp.ndarray,
    Z: jnp.ndarray,
    µ: jnp.double,
) -> tuple[
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
]:
    """Build the Newton-KKT system used to compute the line search direction."""
    T = X.shape[0] - 1

    U_pad = pad(U)

    lagrangian = build_lagrangian(cost, dynamics, equalities, inequalities, x0, µ)

    quadratizer = quadratize(lagrangian, argnums=7)
    Q, R_pad, M_pad = quadratizer(
        X, U_pad, S, pad(Y_dyn[1:]), Y_dyn, Y_eq, Z, jnp.arange(T + 1)
    )

    M = M_pad[:-1]
    R = R_pad[:-1]

    Q, R = regularize(Q=Q, R=R, M=M, psd_delta=1e-3)

    M_pad = jnp.concatenate([M, jnp.zeros_like(M[0])[None, ...]], axis=0)

    R_pad = jnp.concatenate([R, jnp.eye(R.shape[-1])[None, ...] * 1e-3], axis=0)

    P = jax.vmap(lambda q, m, r: jnp.block([[q, m], [m.T, r]]))(Q, M_pad, R_pad)

    linearizer = linearize(lagrangian, argnums=7)
    q, r_pad = linearizer(
        X, U_pad, S, pad(Y_dyn[1:]), Y_dyn, Y_eq, Z, jnp.arange(T + 1)
    )
    r_pad = r_pad.at[-1].set(0.0)

    r_x = jnp.stack([q, r_pad], axis=-1)

    dynamics_linearizer = linearize(dynamics)
    A_pad, B_pad = dynamics_linearizer(X, U_pad, jnp.arange(T + 1))
    A = A_pad[:-1]
    B = B_pad[:-1]
    D = jnp.concatenate([A, B], axis=-1)

    r_s = Z - µ / S

    r_y_dyn = vectorize(dynamics)(X[:-1], U, jnp.arange(T + 1)) - X[1:]

    r_y_eq = vectorize(equalities)(X, U_pad, jnp.arange(T + 1))

    r_z = vectorize(inequalities)(X, U_pad, jnp.arange(T + 1)) + S

    E_x, E_u = linearize(equalities)(X, U_pad, jnp.arange(T + 1))
    E = jnp.concatenate([E_x, E_u], axis=-1)

    G_x, G_u = linearize(inequalities)(X, U_pad, jnp.arange(T + 1))
    G = jnp.concatenate([G_x, G_u], axis=-1)

    return P, D, E, G, r_x, r_s, r_y_dyn, r_y_eq, r_z
