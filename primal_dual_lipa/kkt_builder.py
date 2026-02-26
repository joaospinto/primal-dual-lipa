"""Provides the helper method for building the Newton-KKT system.

This is used to compute the line search direction at each optimization step.
"""

from functools import partial

import jax
from jax import numpy as jnp
from regularized_lqr_jax.helpers import regularize

from primal_dual_lipa.lagrangian_helpers import build_lagrangian, pad
from primal_dual_lipa.types import (
    CostFunction,
    Function,
    KKTFactorizationInputs,
    KKTSystem,
    Parameters,
    Variables,
)
from primal_dual_lipa.vectorization_helpers import linearize, quadratize, vectorize


@partial(
    jax.jit,
    static_argnames=[
        "cost",
        "dynamics",
        "equalities",
        "inequalities",
    ],
)
def kkt_builder(
    cost: CostFunction,
    dynamics: Function,
    equalities: Function,
    inequalities: Function,
    x0: jnp.ndarray,
    vars: Variables,
    params: Parameters,
) -> KKTSystem:
    """Build the Newton-KKT system used to compute the line search direction."""
    T = vars.X.shape[0] - 1

    T_range = jnp.arange(T)
    Tp1_range = jnp.arange(T + 1)

    U_pad = pad(vars.U)

    lagrangian = build_lagrangian(
        cost=cost,
        dynamics=dynamics,
        equalities=equalities,
        inequalities=inequalities,
        x0=x0,
        µ=params.µ,
    )

    quadratizer = quadratize(lagrangian, argnums=8)
    Q, R_pad, M_pad = quadratizer(
        vars.X,
        U_pad,
        Tp1_range,
        vars.S,
        pad(vars.Y_dyn[1:]),
        vars.Y_dyn,
        vars.Y_eq,
        vars.Z,
    )

    M = M_pad[:-1]
    R = R_pad[:-1]

    Q, R = regularize(Q=Q, R=R, M=M, psd_delta=1e-3)

    M_pad = jnp.concatenate([M, jnp.zeros_like(M[0])[None, ...]], axis=0)

    R_pad = jnp.concatenate([R, jnp.eye(R.shape[-1])[None, ...] * 1e-3], axis=0)

    P = jax.vmap(lambda q, m, r: jnp.block([[q, m], [m.T, r]]))(Q, M_pad, R_pad)

    linearizer = linearize(lagrangian, argnums=8)
    q, r_pad = linearizer(
        vars.X,
        U_pad,
        Tp1_range,
        vars.S,
        pad(vars.Y_dyn[1:]),
        vars.Y_dyn,
        vars.Y_eq,
        vars.Z,
    )

    dynamics_linearizer = linearize(dynamics)
    A, B = dynamics_linearizer(vars.X[:-1], vars.U, T_range)
    D = jnp.concatenate([A, B], axis=-1)

    r_s = vars.Z - params.µ / vars.S

    r_y_dyn = vectorize(dynamics)(vars.X[:-1], vars.U, T_range) - vars.X[1:]
    r_y_dyn = jnp.concatenate([(x0 - vars.X[0])[None, ...], r_y_dyn])

    r_y_eq = vectorize(equalities)(vars.X, U_pad, Tp1_range)

    r_z = vectorize(inequalities)(vars.X, U_pad, Tp1_range) + vars.S

    E_x, E_u = linearize(equalities)(vars.X, U_pad, Tp1_range)
    E = jnp.concatenate([E_x, E_u], axis=-1)

    G_x, G_u = linearize(inequalities)(vars.X, U_pad, Tp1_range)
    G = jnp.concatenate([G_x, G_u], axis=-1)

    w_inv = jnp.clip(vars.Z / vars.S, 1e-8, 1e8)

    return KKTSystem(
        lhs=KKTFactorizationInputs(
            P=P,
            D=D,
            E=E,
            G=G,
            w_inv=w_inv,
            params=params,
        ),
        rhs=Variables(
            X=q,
            U=r_pad[:-1],
            S=r_s,
            Y_dyn=r_y_dyn,
            Y_eq=r_y_eq,
            Z=r_z,
        ),
    )
