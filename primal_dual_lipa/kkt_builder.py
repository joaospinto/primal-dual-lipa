"""Provides the helper method for building the Newton-KKT system.

This is used to compute the line search direction at each optimization step.
"""

from functools import partial

import jax
import jax.scipy as jsp
from jax import numpy as jnp
from regularized_lqr_jax.helpers import project_psd_cone

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


@partial(jax.jit, static_argnames=["dims"])
def block_schur_psd_projection(M, dims, eps):
    """Recursively computes a faster approximate PSD projection of matrix M
    using Schur complements over block sizes specified in `dims`.
    """
    # Base case: only one block left
    if len(dims) == 1:
        return project_psd_cone(M, delta=eps)

    d = dims[0]

    # 1. Partition the matrix M = [[A, B], [B.T, D]]
    A = M[:-d, :-d]
    B = M[:-d, -d:]
    D = M[-d:, -d:]

    # 2. Project the bottom-right corner to the PSD cone
    D_plus = project_psd_cone(D, delta=eps)

    # 3. Compute the Schur residual (S = A - B * D_plus^-1 * B^T)
    # Using jsp.linalg.solve with assume_a='pos' triggers a fast Cholesky solve
    D_inv_B_T = jsp.linalg.solve(D_plus, B.T, assume_a="pos")
    S = A - B @ D_inv_B_T

    # 4. Recursively do the same thing with dims[1:] on the Schur residual
    S_plus = block_schur_psd_projection(S, dims[1:], eps)

    # 5. Reconstruct the original matrix
    # Since S = A - B * D_plus^-1 * B^T, we substitute A_new for A
    A_new = S_plus + B @ D_inv_B_T

    # Reassemble blocks
    top = jnp.concatenate([A_new, B], axis=1)
    bottom = jnp.concatenate([B.T, D_plus], axis=1)
    return jnp.concatenate([top, bottom], axis=0)


@jax.jit
def regularize(
    Q: jnp.ndarray,
    R: jnp.ndarray,
    M: jnp.ndarray,
    psd_delta: jnp.double,
    H_theta_theta_per_stage: jnp.ndarray,
    H_x_theta: jnp.ndarray,
    H_u_theta: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Regularizes the KKT Hessian blocks including parameters.

    This method ensures that the joint Hessian (wrt states, controls, and parameters)
    is positive definite by building the joint cost matrix per stage and
    projecting it to the PSD cone using recursive Schur complements.
    """
    T, n, m = M.shape
    p = H_theta_theta_per_stage.shape[-1]

    # 1. Stage-wise projection for t = 0, ..., T-1
    # Matrix layout: [Q, H_x_theta, M], [H_x_theta.T, H_theta_theta, H_u_theta.T], [M.T, H_u_theta, R]
    # This matches the user request: states, thetas, controls order.
    def project_stage(q, r, m_block, h_theta, h_x_theta, h_u_theta):
        top = jnp.concatenate([q, h_x_theta, m_block], axis=1)
        mid = jnp.concatenate([h_x_theta.T, h_theta, h_u_theta.T], axis=1)
        bot = jnp.concatenate([m_block.T, h_u_theta, r], axis=1)
        full = jnp.concatenate([top, mid, bot], axis=0)

        # dims=(m, p, n) projects R first, then theta, then Q
        full_reg = block_schur_psd_projection(full, (m, p, n), psd_delta)

        # Extract diagonal blocks
        return (
            full_reg[:n, :n],
            full_reg[n + p :, n + p :],
            full_reg[n : n + p, n : n + p],
        )

    Q_reg_stages, R_reg, H_theta_per_stage_reg = jax.vmap(project_stage)(
        Q[:-1], R, M, H_theta_theta_per_stage[:-1], H_x_theta[:-1], H_u_theta
    )

    # 2. Terminal stage projection
    def project_terminal(q, h_theta, h_x_theta):
        top = jnp.concatenate([q, h_x_theta], axis=1)
        bot = jnp.concatenate([h_x_theta.T, h_theta], axis=1)
        full = jnp.concatenate([top, bot], axis=0)

        # dims=(p, n) projects theta first, then Q
        full_reg = block_schur_psd_projection(full, (p, n), psd_delta)
        return full_reg[:n, :n], full_reg[n:, n:]

    Q_reg_T, H_theta_T_reg = project_terminal(
        Q[T], H_theta_theta_per_stage[T], H_x_theta[T]
    )

    Q_reg = jnp.concatenate([Q_reg_stages, Q_reg_T[None, ...]], axis=0)
    H_theta_per_stage_reg = jnp.concatenate(
        [H_theta_per_stage_reg, H_theta_T_reg[None, ...]], axis=0
    )

    return Q_reg, R_reg, H_theta_per_stage_reg


@partial(
    jax.jit,
    static_argnames=[
        "cost",
        "dynamics",
        "equalities",
        "inequalities",
    ],
)
def build_kkt_lhs(
    cost: CostFunction,
    dynamics: Function,
    equalities: Function,
    inequalities: Function,
    x0: jnp.ndarray,
    vars: Variables,
    params: Parameters,
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

    Q, R, H_theta_theta_per_stage = regularize(
        Q=Q,
        R=R,
        M=M,
        psd_delta=1e-3,
        H_theta_theta_per_stage=H_theta_theta_per_stage,
        H_x_theta=H_x_theta,
        H_u_theta=H_u_theta_pad[:-1],
    )

    H_theta_theta = jnp.sum(H_theta_theta_per_stage, axis=0)

    M_pad = jnp.concatenate([M, jnp.zeros_like(M[0])[None, ...]], axis=0)

    R_pad = jnp.concatenate([R, jnp.eye(R.shape[-1])[None, ...] * 1e-3], axis=0)

    P = jax.vmap(lambda q, m, r: jnp.block([[q, m], [m.T, r]]))(Q, M_pad, R_pad)

    dynamics_linearizer = linearize(dynamics)
    A, B, H_theta_y_dyn = dynamics_linearizer(vars.X[:-1], vars.U, vars.Theta, T_range)
    D = jnp.concatenate([A, B], axis=-1)

    equalities_linearizer = linearize(equalities)
    E_x, E_u, H_theta_y_eq = equalities_linearizer(vars.X, U_pad, vars.Theta, Tp1_range)
    E = jnp.concatenate([E_x, E_u], axis=-1)

    inequalities_linearizer = linearize(inequalities)
    G_x, G_u, H_theta_z = inequalities_linearizer(vars.X, U_pad, vars.Theta, Tp1_range)
    G = jnp.concatenate([G_x, G_u], axis=-1)

    w_inv = jnp.clip(vars.Z / vars.S, 1e-8, 1e8) + params.μ

    H_theta_y_dyn_full = jnp.concatenate(
        [jnp.zeros_like(H_theta_y_dyn[0])[None, ...], H_theta_y_dyn], axis=0
    )

    return KKTFactorizationInputs(
        P=P,
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
    x0: jnp.ndarray,
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
    ],
)
def build_kkt(
    cost: CostFunction,
    dynamics: Function,
    equalities: Function,
    inequalities: Function,
    x0: jnp.ndarray,
    vars: Variables,
    params: Parameters,
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
