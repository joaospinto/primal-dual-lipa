"""Provides utilities for handling the Newton-KKT linear systems.

The block-5x5 Newton-KKT linear system is of the form
    [  H_pp     H_xup   0      H_yp    H_zp  ] [∆p ] = - [ r_p  ],
    [ H_xup^T     P     0      C^T      G^T  ] [∆xu]     [ r_xu ],
    [   0         0   W^{-1}    0        I   ] [∆s ]     [ r_s  ]
    [  H_yp^T     C     0    -(1/η)I     0   ] [∆y ]     [ r_y  ]
    [  H_zp^T     G     I       0    -(1/η)I ] [∆z ]     [ r_z  ]
where
    P = diag(P_0, ..., P_N),
    W^{-1} = diag(w_inv),
    C = stack_rows(Jacobian(dynamics), E),
    Jacobian(dynamics) = [  -I                                                                 ],
                            [ D_x_0  D_u_0  -I                                                    ]
                            [               D_x_1  D_u_1   -I                                     ]
                            [                              D_x_2  D_u_2  -I                       ]
                            [                                             (...)                   ]
                            [                                           D_x_{N-1}  D_u_{N-1}  -I  ]
    E = [ E_x_0  E_u_0                                                           ]
        [               E_x_1  E_u_1                                             ]
        [                             E_x_2  E_u_2                               ]
        [                                        (...)                           ]
        [                                            E_x_{N-1}  E_u_{N-1}        ]
        [                                                                  E_x_N ]
    G = [ G_x_0  G_u_0                                                           ],
        [               G_x_1  G_u_1                                             ]
        [                             G_x_2  G_u_2                               ]
        [                                        (...)                           ]
        [                                            G_x_{N-1}  G_u_{N-1}        ]
        [                                                                  G_x_N ]
    r_y = stack_rows(r_y_dyn, r_y_eq).
"""  # noqa: E501

from functools import partial

import jax
from jax import numpy as jnp
from regularized_lqr_jax.solver import (
    factor,
    factor_parallel,
    solve,
    solve_parallel,
)
from regularized_lqr_jax.types import FactorizationInputs as LQRFactorizationInputs
from regularized_lqr_jax.types import SolveInputs as LQRSolveInputs
from regularized_lqr_jax.types import SolveOutputs as LQRSolveOutputs

from primal_dual_lipa.lagrangian_helpers import pad
from primal_dual_lipa.types import (
    KKTFactorizationInputs,
    KKTFactorizationOutputs,
    KKTSystem,
    Parameters,
    Variables,
)


@partial(
    jax.jit,
    static_argnames=("num_steps",),
)
def ruiz_scaling(
    kkt_system: KKTSystem, num_steps: int
) -> tuple[KKTSystem, Variables]:
    """Perform Ruiz scaling on the KKT system."""
    lhs = kkt_system.lhs
    rhs = kkt_system.rhs

    T = rhs.X.shape[0] - 1
    n = rhs.X.shape[1]
    m = rhs.U.shape[1]
    g = rhs.S.shape[1]
    c = rhs.Y_eq.shape[1]
    p = rhs.Theta.shape[0]

    # Initialize scaling
    d_X = jnp.ones((T + 1, n))
    d_U = jnp.ones((T, m))
    d_S = jnp.ones((T + 1, g))
    d_Y_dyn = jnp.ones((T + 1, n))
    d_Y_eq = jnp.ones((T + 1, c))
    d_Z = jnp.ones((T + 1, g))
    d_Theta = jnp.ones(p)

    # Precompute transposes
    Dx = lhs.D[:, :, :n]
    Du = lhs.D[:, :, n:]
    Dx_T = jnp.transpose(Dx, (0, 2, 1))
    Du_T = jnp.transpose(Du, (0, 2, 1))

    Ex = lhs.E[:, :, :n]
    Eu = lhs.E[:, :, n:]
    Ex_T = jnp.transpose(Ex, (0, 2, 1))
    Eu_T = jnp.transpose(Eu, (0, 2, 1))

    Gx = lhs.G[:, :, :n]
    Gu = lhs.G[:, :, n:]
    Gx_T = jnp.transpose(Gx, (0, 2, 1))
    Gu_T = jnp.transpose(Gu, (0, 2, 1))

    def get_row_max(A, d_col):
        return jnp.max(jnp.abs(A) * d_col[..., None, :], axis=-1, initial=0.0)

    def body_fn(i, scaling):
        dX, dU, dS, dYd, dYe, dZ, dTh = scaling

        dU_pad = jnp.concatenate([dU, jnp.ones((1, m))], axis=0)

        # Row X
        rX = get_row_max(lhs.P[:, :n, :n], dX)
        rX = jnp.maximum(rX, get_row_max(lhs.P[:, :n, n:], dU_pad))
        rX = jnp.maximum(rX, dYd)
        rX = jnp.maximum(
            rX,
            jnp.concatenate(
                [get_row_max(Dx_T, dYd[1:]), jnp.zeros((1, n))],
                axis=0,
            ),
        )
        rX = jnp.maximum(rX, get_row_max(Ex_T, dYe))
        rX = jnp.maximum(rX, get_row_max(Gx_T, dZ))
        rX = jnp.maximum(rX, get_row_max(lhs.H_theta_X, dTh))
        rX = dX * rX

        # Row U
        rU = get_row_max(lhs.P[:-1, n:, :n], dX[:-1])
        rU = jnp.maximum(rU, get_row_max(lhs.P[:-1, n:, n:], dU))
        rU = jnp.maximum(rU, get_row_max(Du_T, dYd[1:]))
        rU = jnp.maximum(rU, get_row_max(Eu_T[:-1], dYe[:-1]))
        rU = jnp.maximum(rU, get_row_max(Gu_T[:-1], dZ[:-1]))
        rU = jnp.maximum(rU, get_row_max(lhs.H_theta_U, dTh))
        rU = dU * rU

        # Row S
        rS = lhs.w_inv * dS
        rS = jnp.maximum(rS, dZ)
        rS = dS * rS

        # Row Y_eq
        rYe = get_row_max(Ex, dX)
        rYe = jnp.maximum(
            rYe,
            jnp.concatenate(
                [get_row_max(Eu[:-1], dU), jnp.zeros((1, c))],
                axis=0,
            ),
        )
        rYe = jnp.maximum(rYe, (1.0 / lhs.params.η_eq) * dYe)
        rYe = jnp.maximum(rYe, get_row_max(lhs.H_theta_y_eq, dTh))
        rYe = dYe * rYe

        # Row Theta
        rTh = get_row_max(lhs.H_theta_theta, dTh)
        rTh = jnp.maximum(
            rTh,
            jnp.max(
                jnp.abs(lhs.H_theta_X) * dX[..., None], axis=(0, 1), initial=0.0
            ),
        )
        rTh = jnp.maximum(
            rTh,
            jnp.max(
                jnp.abs(lhs.H_theta_U) * dU[..., None], axis=(0, 1), initial=0.0
            ),
        )
        rTh = jnp.maximum(
            rTh,
            jnp.max(
                jnp.abs(lhs.H_theta_y_dyn) * dYd[..., None],
                axis=(0, 1),
                initial=0.0,
            ),
        )
        rTh = jnp.maximum(
            rTh,
            jnp.max(
                jnp.abs(lhs.H_theta_y_eq) * dYe[..., None],
                axis=(0, 1),
                initial=0.0,
            ),
        )
        rTh = jnp.maximum(
            rTh,
            jnp.max(
                jnp.abs(lhs.H_theta_z) * dZ[..., None], axis=(0, 1), initial=0.0
            ),
        )
        rTh = dTh * rTh

        # Update scaling factors for the free variables
        dX_new = dX / jnp.sqrt(jnp.clip(rX, a_min=1e-12))
        dU_new = dU / jnp.sqrt(jnp.clip(rU, a_min=1e-12))
        dS_new = dS / jnp.sqrt(jnp.clip(rS, a_min=1e-12))
        dYe_new = dYe / jnp.sqrt(jnp.clip(rYe, a_min=1e-12))
        dTh_new = dTh / jnp.sqrt(jnp.clip(rTh, a_min=1e-12))

        # Enforce structural constraints for the dependent variables
        dYd_new = 1.0 / dX_new
        dZ_new = 1.0 / dS_new

        return (dX_new, dU_new, dS_new, dYd_new, dYe_new, dZ_new, dTh_new)

    scaling = jax.lax.fori_loop(
        0, num_steps, body_fn, (d_X, d_U, d_S, d_Y_dyn, d_Y_eq, d_Z, d_Theta)
    )
    dX, dU, dS, dYd, dYe, dZ, dTh = scaling

    # Scale LHS
    dU_pad = jnp.concatenate([dU, jnp.ones((1, m))], axis=0)
    d_XU = jnp.concatenate([dX, dU_pad], axis=1)

    P_scaled = lhs.P * d_XU[..., :, None] * d_XU[..., None, :]
    D_scaled = lhs.D * dYd[1:, :, None] * d_XU[:-1, None, :]
    E_scaled = lhs.E * dYe[..., :, None] * d_XU[..., None, :]
    G_scaled = lhs.G * dZ[..., :, None] * d_XU[..., None, :]

    w_inv_scaled = lhs.w_inv * dS * dS
    η_dyn_scaled = lhs.params.η_dyn / jnp.square(dYd)
    η_eq_scaled = lhs.params.η_eq / jnp.square(dYe)
    η_ineq_scaled = lhs.params.η_ineq / jnp.square(dZ)

    H_theta_theta_scaled = lhs.H_theta_theta * dTh[:, None] * dTh[None, :]
    H_theta_X_scaled = lhs.H_theta_X * dX[..., None] * dTh[None, None, :]
    H_theta_U_scaled = lhs.H_theta_U * dU[..., None] * dTh[None, None, :]
    H_theta_y_dyn_scaled = (
        lhs.H_theta_y_dyn * dYd[..., None] * dTh[None, None, :]
    )
    H_theta_y_eq_scaled = lhs.H_theta_y_eq * dYe[..., None] * dTh[None, None, :]
    H_theta_z_scaled = lhs.H_theta_z * dZ[..., None] * dTh[None, None, :]

    scaled_lhs = KKTFactorizationInputs(
        P=P_scaled,
        D=D_scaled,
        E=E_scaled,
        G=G_scaled,
        w_inv=w_inv_scaled,
        params=Parameters(
            µ=lhs.params.µ,
            η_dyn=η_dyn_scaled,
            η_eq=η_eq_scaled,
            η_ineq=η_ineq_scaled,
        ),
        H_theta_theta=H_theta_theta_scaled,
        H_theta_X=H_theta_X_scaled,
        H_theta_U=H_theta_U_scaled,
        H_theta_y_dyn=H_theta_y_dyn_scaled,
        H_theta_y_eq=H_theta_y_eq_scaled,
        H_theta_z=H_theta_z_scaled,
    )

    scaled_rhs = Variables(
        X=rhs.X * dX,
        U=rhs.U * dU,
        S=rhs.S * dS,
        Y_dyn=rhs.Y_dyn * dYd,
        Y_eq=rhs.Y_eq * dYe,
        Z=rhs.Z * dZ,
        Theta=rhs.Theta * dTh,
    )

    scaling_factors = Variables(
        X=dX,
        U=dU,
        S=dS,
        Y_dyn=dYd,
        Y_eq=dYe,
        Z=dZ,
        Theta=dTh,
    )

    return KKTSystem(lhs=scaled_lhs, rhs=scaled_rhs), scaling_factors


@partial(
    jax.jit,
    static_argnames=("use_parallel_lqr",),
)
def lqr_solve_kkt(
    lqr_inputs: LQRFactorizationInputs,
    lqr_outputs: jax.Array,
    factorization_inputs: KKTFactorizationInputs,
    rhs: Variables,
    use_parallel_lqr: bool,
) -> Variables:
    """Solve the LQR part of the Newton-KKT system (ignoring global parameters)."""
    w = 1.0 / factorization_inputs.w_inv
    reg_w_inv = 1.0 / (w + 1.0 / factorization_inputs.params.η_ineq)
    bmm = jax.vmap(jnp.matmul)
    x_dim = factorization_inputs.D.shape[1]

    term_G = bmm(
        factorization_inputs.G.mT,
        reg_w_inv * (rhs.Z - w * rhs.S),
    )
    term_E = bmm(factorization_inputs.E.mT, factorization_inputs.params.η_eq * rhs.Y_eq)

    r_x = rhs.X + term_G[:, :x_dim] + term_E[:, :x_dim]
    r_u = rhs.U + term_G[:-1, x_dim:] + term_E[:-1, x_dim:]

    solve_fn = solve_parallel if use_parallel_lqr else solve

    solve_outputs: LQRSolveOutputs = solve_fn(
        factorization_inputs=lqr_inputs,
        factorization_outputs=lqr_outputs,
        solve_inputs=LQRSolveInputs(
            q=r_x,
            r=r_u,
            c=rhs.Y_dyn,
        ),
    )

    dX = solve_outputs.X
    dU = solve_outputs.U
    dY_dyn = solve_outputs.Y

    dU_padded = pad(dU)

    dY_eq = factorization_inputs.params.η_eq * (
        bmm(factorization_inputs.E[:, :, :x_dim], dX)
        + bmm(factorization_inputs.E[:, :, x_dim:], dU_padded)
        + rhs.Y_eq
    )
    dZ = reg_w_inv * (
        bmm(factorization_inputs.G[:, :, :x_dim], dX)
        + bmm(factorization_inputs.G[:, :, x_dim:], dU_padded)
        + (rhs.Z - w * rhs.S)
    )
    dS = -w * (rhs.S + dZ)

    return Variables(
        X=dX,
        U=dU,
        S=dS,
        Y_dyn=dY_dyn,
        Y_eq=dY_eq,
        Z=dZ,
        Theta=jnp.empty(0),
    )


@partial(
    jax.jit,
    static_argnames=("use_parallel_lqr",),
)
def factor_kkt(
    inputs: KKTFactorizationInputs,
    use_parallel_lqr: bool,
) -> KKTFactorizationOutputs:
    """Factorize the Newton-KKT linear system."""
    w = 1.0 / inputs.w_inv
    reg_w_inv = 1.0 / (w + 1.0 / inputs.params.η_ineq)
    bmm = jax.vmap(jnp.matmul)
    x_dim = inputs.D.shape[1]

    P_2x2 = (
        inputs.P
        + bmm(inputs.E.mT, inputs.params.η_eq[..., None] * inputs.E)
        + bmm(inputs.G.mT, reg_w_inv[..., None] * inputs.G)
    )

    Δ = jax.vmap(jnp.diag)(1.0 / inputs.params.η_dyn)

    factor_fn = factor_parallel if use_parallel_lqr else factor
    lqr_inputs = LQRFactorizationInputs(
        A=inputs.D[:, :, :x_dim],
        B=inputs.D[:, :, x_dim:],
        Q=P_2x2[:, :x_dim, :x_dim],
        M=P_2x2[:-1, :x_dim, x_dim:],
        R=P_2x2[:-1, x_dim:, x_dim:],
        Δ=Δ,
    )
    lqr_outputs = factor_fn(lqr_inputs)

    T = inputs.P.shape[0] - 1

    H_theta_X_T = jnp.moveaxis(inputs.H_theta_X, -1, 0)
    H_theta_U_T = jnp.moveaxis(inputs.H_theta_U, -1, 0)
    H_theta_y_dyn_T = jnp.moveaxis(inputs.H_theta_y_dyn, -1, 0)
    H_theta_y_eq_T = jnp.moveaxis(inputs.H_theta_y_eq, -1, 0)
    H_theta_z_T = jnp.moveaxis(inputs.H_theta_z, -1, 0)

    def partial_solve(
        h_x: jax.Array,
        h_u: jax.Array,
        b_dyn: jax.Array,
        b_eq: jax.Array,
        b_ineq: jax.Array,
    ) -> Variables:
        rhs = Variables(
            X=h_x,
            U=h_u,
            S=jnp.zeros((T + 1, inputs.G.shape[1])),
            Y_dyn=b_dyn,
            Y_eq=b_eq,
            Z=b_ineq,
            Theta=jnp.empty(0),
        )
        return lqr_solve_kkt(
            lqr_inputs=lqr_inputs,
            lqr_outputs=lqr_outputs,
            factorization_inputs=inputs,
            rhs=rhs,
            use_parallel_lqr=use_parallel_lqr,
        )

    sol = jax.vmap(partial_solve)(
        H_theta_X_T, H_theta_U_T, H_theta_y_dyn_T, H_theta_y_eq_T, H_theta_z_T
    )

    schur_term = (
        jnp.einsum("tnp,jtn->pj", inputs.H_theta_X, sol.X)
        + jnp.einsum("tmp,jtm->pj", inputs.H_theta_U, sol.U)
        + jnp.einsum("tnp,jtn->pj", inputs.H_theta_y_dyn, sol.Y_dyn)
        + jnp.einsum("tkp,jtk->pj", inputs.H_theta_y_eq, sol.Y_eq)
        + jnp.einsum("tkp,jtk->pj", inputs.H_theta_z, sol.Z)
    )
    schur_complement = inputs.H_theta_theta + schur_term

    return KKTFactorizationOutputs(
        lqr_inputs=lqr_inputs,
        lqr_outputs=lqr_outputs,
        schur_complement=schur_complement,
        B_inv_C_X=sol.X,
        B_inv_C_U=sol.U,
        B_inv_C_S=sol.S,
        B_inv_C_Y_dyn=sol.Y_dyn,
        B_inv_C_Y_eq=sol.Y_eq,
        B_inv_C_Z=sol.Z,
    )


@partial(
    jax.jit,
    static_argnames=("use_parallel_lqr",),
)
def solve_kkt(
    factorization_outputs: KKTFactorizationOutputs,
    factorization_inputs: KKTFactorizationInputs,
    rhs: Variables,
    use_parallel_lqr: bool,
) -> Variables:
    """Solve the Newton-KKT linear system with a pre-computed factorization."""
    sol0 = lqr_solve_kkt(
        lqr_inputs=factorization_outputs.lqr_inputs,
        lqr_outputs=factorization_outputs.lqr_outputs,
        factorization_inputs=factorization_inputs,
        rhs=rhs,
        use_parallel_lqr=use_parallel_lqr,
    )

    B_sol0 = (
        jnp.einsum("tnp,tn->p", factorization_inputs.H_theta_X, sol0.X)
        + jnp.einsum("tmp,tm->p", factorization_inputs.H_theta_U, sol0.U)
        + jnp.einsum("tnp,tn->p", factorization_inputs.H_theta_y_dyn, sol0.Y_dyn)
        + jnp.einsum("tkp,tk->p", factorization_inputs.H_theta_y_eq, sol0.Y_eq)
        + jnp.einsum("tkp,tk->p", factorization_inputs.H_theta_z, sol0.Z)
    )

    dTheta = jnp.linalg.solve(
        factorization_outputs.schur_complement, -rhs.Theta - B_sol0
    )

    dX = sol0.X + jnp.einsum("p,p...->...", dTheta, factorization_outputs.B_inv_C_X)
    dU = sol0.U + jnp.einsum("p,p...->...", dTheta, factorization_outputs.B_inv_C_U)
    dS = sol0.S + jnp.einsum("p,p...->...", dTheta, factorization_outputs.B_inv_C_S)
    dY_dyn = sol0.Y_dyn + jnp.einsum(
        "p,p...->...", dTheta, factorization_outputs.B_inv_C_Y_dyn
    )
    dY_eq = sol0.Y_eq + jnp.einsum(
        "p,p...->...", dTheta, factorization_outputs.B_inv_C_Y_eq
    )
    dZ = sol0.Z + jnp.einsum("p,p...->...", dTheta, factorization_outputs.B_inv_C_Z)

    return Variables(
        X=dX,
        U=dU,
        S=dS,
        Y_dyn=dY_dyn,
        Y_eq=dY_eq,
        Z=dZ,
        Theta=dTheta,
    )


def compute_kkt_residual(
    factorization_inputs: KKTFactorizationInputs,
    solve_inputs: Variables,
    solution: Variables,
) -> Variables:
    """Compute the residual of the Newton-KKT linear system."""
    bmm = jax.vmap(jnp.matmul)
    x_dim = factorization_inputs.D.shape[1]

    Q = factorization_inputs.P[:, :x_dim, :x_dim]
    M = factorization_inputs.P[:, :x_dim, x_dim:]
    R = factorization_inputs.P[:, x_dim:, x_dim:]

    Dx = factorization_inputs.D[:, :, :x_dim]
    Du = factorization_inputs.D[:, :, x_dim:]

    Ex = factorization_inputs.E[:, :, :x_dim]
    Eu = factorization_inputs.E[:, :, x_dim:]

    Gx = factorization_inputs.G[:, :, :x_dim]
    Gu = factorization_inputs.G[:, :, x_dim:]

    dU_padded = pad(solution.U)

    DxdX_plus_DudU = jnp.concatenate(
        [
            jnp.zeros_like(solution.X[0])[None, ...],
            bmm(Dx, solution.X[:-1]) + bmm(Du, solution.U),
        ]
    )

    Dx_T_dY_dyn = jnp.concatenate(
        [
            bmm(Dx.mT, solution.Y_dyn[1:]),
            jnp.zeros_like(solution.X[0])[None, ...],
        ]
    )
    Du_T_dY_dyn = bmm(Du.mT, solution.Y_dyn[1:])

    res_X = (
        bmm(Q, solution.X)
        + bmm(M, dU_padded)
        + Dx_T_dY_dyn
        - solution.Y_dyn
        + bmm(Ex.mT, solution.Y_eq)
        + bmm(Gx.mT, solution.Z)
        + jnp.einsum("tnp,p->tn", factorization_inputs.H_theta_X, solution.Theta)
        + solve_inputs.X
    )

    res_U = (
        bmm(M[:-1].mT, solution.X[:-1])
        + bmm(R[:-1], solution.U)
        + Du_T_dY_dyn
        + bmm(Eu[:-1].mT, solution.Y_eq[:-1])
        + bmm(Gu[:-1].mT, solution.Z[:-1])
        + jnp.einsum("tmp,p->tm", factorization_inputs.H_theta_U, solution.Theta)
        + solve_inputs.U
    )

    res_S = factorization_inputs.w_inv * solution.S + solution.Z + solve_inputs.S

    res_Y_dyn = (
        DxdX_plus_DudU
        - solution.X
        - (1.0 / factorization_inputs.params.η_dyn) * solution.Y_dyn
        + jnp.einsum("tnp,p->tn", factorization_inputs.H_theta_y_dyn, solution.Theta)
        + solve_inputs.Y_dyn
    )

    res_Y_eq = (
        bmm(Ex, solution.X)
        + jnp.concatenate(
            [bmm(Eu[:-1], solution.U), jnp.zeros_like(solve_inputs.Y_eq[0])[None, ...]]
        )
        - (1.0 / factorization_inputs.params.η_eq) * solution.Y_eq
        + jnp.einsum("tkp,p->tk", factorization_inputs.H_theta_y_eq, solution.Theta)
        + solve_inputs.Y_eq
    )

    res_Z = (
        bmm(Gx, solution.X)
        + jnp.concatenate(
            [bmm(Gu[:-1], solution.U), jnp.zeros_like(solve_inputs.Z[0])[None, ...]]
        )
        + solution.S
        - (1.0 / factorization_inputs.params.η_ineq) * solution.Z
        + jnp.einsum("tkp,p->tk", factorization_inputs.H_theta_z, solution.Theta)
        + solve_inputs.Z
    )

    B_dZ = (
        jnp.einsum("tnp,tn->p", factorization_inputs.H_theta_X, solution.X)
        + jnp.einsum("tmp,tm->p", factorization_inputs.H_theta_U, solution.U)
        + jnp.einsum("tnp,tn->p", factorization_inputs.H_theta_y_dyn, solution.Y_dyn)
        + jnp.einsum("tkp,tk->p", factorization_inputs.H_theta_y_eq, solution.Y_eq)
        + jnp.einsum("tkp,tk->p", factorization_inputs.H_theta_z, solution.Z)
    )
    res_Theta = (
        factorization_inputs.H_theta_theta @ solution.Theta + B_dZ + solve_inputs.Theta
    )

    return Variables(
        X=res_X,
        U=res_U,
        S=res_S,
        Y_dyn=res_Y_dyn,
        Y_eq=res_Y_eq,
        Z=res_Z,
        Theta=res_Theta,
    )
