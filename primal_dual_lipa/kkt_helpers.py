"""Provides utilities for handling the Newton-KKT linear systems.

The block-4x4 Newton-KKT linear system is of the form
    [ P     0      C^T      G^T  ] [∆x] = - [ r_x ],
    [ 0   W^{-1}    0        I   ] [∆s]     [ r_s ]
    [ C     0    -(1/η)I     0   ] [∆y]     [ r_y ]
    [ G     I       0    -(1/η)I ] [∆z]     [ r_z ]
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
from regularized_lqr_jax.solver import factor, factor_parallel, solve, solve_parallel
from regularized_lqr_jax.types import (
    FactorizationInputs,
    SolveInputs,
    SolveOutputs,
)

from primal_dual_lipa.types import (
    KKTFactorizationInputs,
    KKTFactorizationOutputs,
    Variables,
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
    lqr_inputs = FactorizationInputs(
        A=inputs.D[:, :, :x_dim],
        B=inputs.D[:, :, x_dim:],
        Q=P_2x2[:, :x_dim, :x_dim],
        M=P_2x2[:-1, :x_dim, x_dim:],
        R=P_2x2[:-1, x_dim:, x_dim:],
        Δ=Δ,
    )
    lqr_outputs = factor_fn(lqr_inputs)

    return KKTFactorizationOutputs(lqr_inputs=lqr_inputs, lqr_outputs=lqr_outputs)


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

    solve_outputs: SolveOutputs = solve_fn(
        factorization_inputs=factorization_outputs.lqr_inputs,
        factorization_outputs=factorization_outputs.lqr_outputs,
        solve_inputs=SolveInputs(
            q=r_x,
            r=r_u,
            c=rhs.Y_dyn,
        ),
    )

    dX = solve_outputs.X
    dU = solve_outputs.U
    dY_dyn = solve_outputs.Y

    dU_padded = jnp.concatenate([dU, jnp.zeros_like(dU[0])[None, ...]])

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
    )


@jax.jit
def compute_kkt_residual(
    factorization_inputs: KKTFactorizationInputs,
    solve_inputs: Variables,
    solution: Variables,
) -> Variables:
    """Compute the residual of the Newton-KKT system."""
    N, x_dim = factorization_inputs.D.shape[:2]
    Q = factorization_inputs.P[:, :x_dim, :x_dim]
    M = factorization_inputs.P[:, :x_dim, x_dim:]
    R = factorization_inputs.P[:, x_dim:, x_dim:]

    dU_padded = jnp.concatenate([solution.U, jnp.zeros_like(solution.U[0])[None, ...]])
    Dx = factorization_inputs.D[:, :, :x_dim]
    Du = factorization_inputs.D[:, :, x_dim:]
    Ex = factorization_inputs.E[:, :, :x_dim]
    Eu = factorization_inputs.E[:, :, x_dim:]
    Gx = factorization_inputs.G[:, :, :x_dim]
    Gu = factorization_inputs.G[:, :, x_dim:]
    bmm = jax.vmap(jnp.matmul)
    DxdX_plus_DudU = bmm(Dx, solution.X[:-1]) + bmm(Du, solution.U)
    DxdX_plus_DudU = jnp.concatenate(
        [jnp.zeros_like(DxdX_plus_DudU[0])[None, ...], DxdX_plus_DudU]
    )
    Dx_T_dY_dyn = bmm(Dx.mT, solution.Y_dyn[1:])
    Dx_T_dY_dyn = jnp.concatenate(
        [Dx_T_dY_dyn, jnp.zeros_like(Dx_T_dY_dyn[0])[None, ...]]
    )
    Du_T_dY_dyn = bmm(Du.mT, solution.Y_dyn[1:])

    res_X = (
        bmm(Q, solution.X)
        + bmm(M, dU_padded)
        + Dx_T_dY_dyn
        - solution.Y_dyn
        + bmm(Ex.mT, solution.Y_eq)
        + bmm(Gx.mT, solution.Z)
        + solve_inputs.X
    )

    res_U = (
        bmm(M[:-1].mT, solution.X[:-1])
        + bmm(R[:-1], solution.U)
        + Du_T_dY_dyn
        + bmm(Eu[:-1].mT, solution.Y_eq[:-1])
        + bmm(Gu[:-1].mT, solution.Z[:-1])
        + solve_inputs.U
    )

    res_S = factorization_inputs.w_inv * solution.S + solution.Z + solve_inputs.S

    res_Y_dyn = (
        DxdX_plus_DudU
        - solution.X
        - (1.0 / factorization_inputs.params.η_dyn) * solution.Y_dyn
        + solve_inputs.Y_dyn
    )

    res_Y_eq = (
        bmm(Ex, solution.X)
        + jnp.concatenate(
            [bmm(Eu[:-1], solution.U), jnp.zeros_like(solve_inputs.Y_eq[0])[None, ...]]
        )
        - (1.0 / factorization_inputs.params.η_eq) * solution.Y_eq
        + solve_inputs.Y_eq
    )

    res_Z = (
        bmm(Gx, solution.X)
        + jnp.concatenate(
            [bmm(Gu[:-1], solution.U), jnp.zeros_like(solve_inputs.Z[0])[None, ...]]
        )
        + solution.S
        - (1.0 / factorization_inputs.params.η_ineq) * solution.Z
        + solve_inputs.Z
    )

    return Variables(
        X=res_X,
        U=res_U,
        S=res_S,
        Y_dyn=res_Y_dyn,
        Y_eq=res_Y_eq,
        Z=res_Z,
    )
