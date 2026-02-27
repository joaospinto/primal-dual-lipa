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
    Variables,
)


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
