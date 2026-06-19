"""Provides the helper method for building the Newton-KKT system.

This is used to compute the line search direction at each optimization step.
"""

from functools import partial

import jax
from jax import numpy as jnp
from regularized_lqr_jax.helpers import symmetrize

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


@jax.jit
def regularize_primal_hessian_blocks(
    Q: jax.Array,
    R: jax.Array,
    H_theta_theta_per_stage: jax.Array,
    hessian_regularization: jnp.double,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Symmetrize Hessian blocks and shift only the primal blocks."""
    Q = jax.vmap(symmetrize)(Q)
    R = jax.vmap(symmetrize)(R)
    H_theta_theta_per_stage = jax.vmap(symmetrize)(H_theta_theta_per_stage)

    x_dim = Q.shape[-1]
    u_dim = R.shape[-1]
    Q = Q + hessian_regularization * jnp.eye(x_dim)
    R = R + hessian_regularization * jnp.eye(u_dim)

    return Q, R, H_theta_theta_per_stage


@jax.jit
def add_scalar_hessian_regularization_delta(
    kkt_system: KKTSystem,
    hessian_regularization_delta: jnp.double,
) -> KKTSystem:
    """Shift cached primal Hessian blocks after a scalar regularization change."""
    lhs = kkt_system.lhs
    xu_dim = lhs.P.shape[-1]
    theta_dim = lhs.H_theta_theta.shape[-1]
    P_shift = hessian_regularization_delta * jnp.eye(xu_dim)[None, ...]
    H_theta_theta_shift = hessian_regularization_delta * jnp.eye(theta_dim)
    lhs_new = KKTFactorizationInputs(
        P=lhs.P + P_shift,
        P_lqr=lhs.P_lqr + P_shift,
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
    )
    return KKTSystem(lhs=lhs_new, rhs=kkt_system.rhs)


@partial(
    jax.jit,
    static_argnames=[
        "cost",
        "dynamics",
        "equalities",
        "inequalities",
        "regularize_slack_elimination_with_mu",
    ],
)
def build_kkt_lhs(
    cost: CostFunction,
    dynamics: Function,
    equalities: Function,
    inequalities: Function,
    x0: jax.Array,
    vars: Variables,
    params: Parameters,
    hessian_regularization: jnp.double,
    regularize_slack_elimination_with_mu: bool,
) -> KKTFactorizationInputs:
    """Build the LHS of the Newton-KKT system."""
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

    quadratizer = quadratize(lagrangian)
    Q, R_pad, M_pad, H_theta_theta_per_stage, H_x_theta, H_u_theta_pad = quadratizer(
        vars.X,
        U_pad,
        vars.Theta,
        Tp1_range,
        vars.S,
        pad(vars.Y_dyn[1:]),
        vars.Y_dyn,
        vars.Y_eq,
        vars.Z,
    )

    M = M_pad[:-1]
    R = R_pad[:-1]

    Q, R, H_theta_theta_per_stage = regularize_primal_hessian_blocks(
        Q=Q,
        R=R,
        H_theta_theta_per_stage=H_theta_theta_per_stage,
        hessian_regularization=hessian_regularization,
    )

    H_theta_theta = jnp.sum(H_theta_theta_per_stage, axis=0)

    M_pad = jnp.concatenate([M, jnp.zeros_like(M[0])[None, ...]], axis=0)

    R_pad = jnp.concatenate(
        [R, jnp.eye(R.shape[-1])[None, ...] * hessian_regularization],
        axis=0,
    )

    P = jax.vmap(lambda q, m, r: jnp.block([[q, m], [m.T, r]]))(Q, M_pad, R_pad)
    H_theta_theta = H_theta_theta + hessian_regularization * jnp.eye(
        H_theta_theta.shape[-1]
    )

    dynamics_linearizer = linearize(dynamics)
    A, B, H_theta_y_dyn = dynamics_linearizer(vars.X[:-1], vars.U, vars.Theta, T_range)
    D = jnp.concatenate([A, B], axis=-1)

    equalities_linearizer = linearize(equalities)
    E_x, E_u, H_theta_y_eq = equalities_linearizer(vars.X, U_pad, vars.Theta, Tp1_range)
    E = jnp.concatenate([E_x, E_u], axis=-1)

    inequalities_linearizer = linearize(inequalities)
    G_x, G_u, H_theta_z = inequalities_linearizer(vars.X, U_pad, vars.Theta, Tp1_range)
    G = jnp.concatenate([G_x, G_u], axis=-1)

    w_inv = jnp.clip(vars.Z / vars.S, 1e-8, 1e8)
    if regularize_slack_elimination_with_mu:
        w_inv = w_inv + params.μ
    w = 1.0 / w_inv
    reg_w_inv = 1.0 / (w + 1.0 / params.η_ineq)
    bmm = jax.vmap(jnp.matmul)
    P_lqr = (
        P
        + bmm(E.mT, params.η_eq[..., None] * E)
        + bmm(G.mT, reg_w_inv[..., None] * G)
    )

    H_theta_y_dyn_full = jnp.concatenate(
        [jnp.zeros_like(H_theta_y_dyn[0])[None, ...], H_theta_y_dyn], axis=0
    )

    return KKTFactorizationInputs(
        P=P,
        P_lqr=P_lqr,
        D=D,
        E=E,
        G=G,
        w_inv=w_inv,
        params=params,
        H_theta_theta=H_theta_theta,
        H_theta_X=H_x_theta,
        H_theta_U=H_u_theta_pad[:-1],
        H_theta_y_dyn=H_theta_y_dyn_full,
        H_theta_y_eq=H_theta_y_eq,
        H_theta_z=H_theta_z,
    )


@partial(
    jax.jit,
    static_argnames=[
        "cost",
        "dynamics",
        "equalities",
        "inequalities",
    ],
)
def build_kkt_rhs(
    cost: CostFunction,
    dynamics: Function,
    equalities: Function,
    inequalities: Function,
    x0: jax.Array,
    vars: Variables,
    params: Parameters,
) -> Variables:
    """Build the RHS of the Newton-KKT system."""
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

    linearizer = linearize(lagrangian)
    q, r_pad, jac_theta_per_stage = linearizer(
        vars.X,
        U_pad,
        vars.Theta,
        Tp1_range,
        vars.S,
        pad(vars.Y_dyn[1:]),
        vars.Y_dyn,
        vars.Y_eq,
        vars.Z,
    )

    r_s = vars.Z - params.µ / vars.S

    r_y_dyn = vectorize(dynamics)(vars.X[:-1], vars.U, vars.Theta, T_range) - vars.X[1:]
    r_y_dyn = jnp.concatenate([(x0 - vars.X[0])[None, ...], r_y_dyn])

    r_y_eq = vectorize(equalities)(vars.X, U_pad, vars.Theta, Tp1_range)

    r_z = vectorize(inequalities)(vars.X, U_pad, vars.Theta, Tp1_range) + vars.S

    return Variables(
        X=q,
        U=r_pad[:-1],
        S=r_s,
        Y_dyn=r_y_dyn,
        Y_eq=r_y_eq,
        Z=r_z,
        Theta=jnp.sum(jac_theta_per_stage, axis=0),
    )


@partial(
    jax.jit,
    static_argnames=[
        "cost",
        "dynamics",
        "equalities",
        "inequalities",
        "regularize_slack_elimination_with_mu",
    ],
)
def build_kkt(
    cost: CostFunction,
    dynamics: Function,
    equalities: Function,
    inequalities: Function,
    x0: jax.Array,
    vars: Variables,
    params: Parameters,
    hessian_regularization: jnp.double = 0.0,
    regularize_slack_elimination_with_mu: bool = True,
) -> KKTSystem:
    return KKTSystem(
        lhs=build_kkt_lhs(
            cost=cost,
            dynamics=dynamics,
            equalities=equalities,
            inequalities=inequalities,
            x0=x0,
            vars=vars,
            params=params,
            hessian_regularization=hessian_regularization,
            regularize_slack_elimination_with_mu=regularize_slack_elimination_with_mu,
        ),
        rhs=build_kkt_rhs(
            cost=cost,
            dynamics=dynamics,
            equalities=equalities,
            inequalities=inequalities,
            x0=x0,
            vars=vars,
            params=params,
        ),
    )
