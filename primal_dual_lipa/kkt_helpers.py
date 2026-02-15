"""Provides a helper method for solving the Newton-KKT systems.

This relies in the Regularized LQR algorithm.
"""

from functools import partial

import jax
from jax import numpy as jnp
from regularized_lqr_jax.solver import solve, solve_parallel


@partial(
    jax.jit,
    static_argnames=("use_parallel_lqr",),
)
def solve_kkt(
    P: jnp.ndarray,
    D: jnp.ndarray,
    E: jnp.ndarray,
    G: jnp.ndarray,
    w_inv: jnp.ndarray,
    r_x: jnp.ndarray,
    r_s: jnp.ndarray,
    r_y_dyn: jnp.ndarray,
    r_y_eq: jnp.ndarray,
    r_z: jnp.ndarray,
    η: float,
    use_parallel_lqr: bool,
) -> tuple[
    jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray
]:
    """Solve the Newton-KKT linear systems.

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
    w = 1.0 / w_inv
    dX, dU, dY_dyn, dY_eq, dZ = _solve_kkt_3x3(
        P=P,
        D=D,
        E=E,
        G=G,
        w=w,
        r_x=r_x,
        r_y_dyn=r_y_dyn,
        r_y_eq=r_y_eq,
        r_z=(r_z - w * r_s),
        η=η,
        use_parallel_lqr=use_parallel_lqr,
    )
    dS = -w * (r_s + dZ)
    return dX, dU, dS, dY_dyn, dY_eq, dZ


@jax.jit
def compute_kkt_residual(
    P: jnp.ndarray,
    D: jnp.ndarray,
    E: jnp.ndarray,
    G: jnp.ndarray,
    w_inv: jnp.ndarray,
    r_x: jnp.ndarray,
    r_s: jnp.ndarray,
    r_y_dyn: jnp.ndarray,
    r_y_eq: jnp.ndarray,
    r_z: jnp.ndarray,
    dX: jnp.ndarray,
    dU: jnp.ndarray,
    dS: jnp.ndarray,
    dY_dyn: jnp.ndarray,
    dY_eq: jnp.ndarray,
    dZ: jnp.ndarray,
    η: float,
) -> tuple[
    jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray
]:
    """Compute the residual of the Newton-KKT system from solve_kkt."""
    N, x_dim = D.shape[:2]
    Q = P[:, :x_dim, :x_dim]
    M = P[:, :x_dim, x_dim:]
    R = P[:, x_dim:, x_dim:]
    M = M.at[-1].set(0.0)
    R = R.at[-1].set(0.0)
    dU = dU.at[N].set(0.0)
    Dx = D[:, :, :x_dim]
    Du = D[:, :, x_dim:]
    Ex = E[:, :, :x_dim]
    Eu = E[::, :, x_dim:]
    Eu = Eu.at[-1].set(0.0)
    Gx = G[:, :, :x_dim]
    Gu = G[:, :, x_dim:]
    Gu = Gu.at[-1].set(0.0)
    η_inv = 1.0 / η
    r_x = r_x.at[-1, x_dim:].set(0.0)
    bmm = jax.vmap(jnp.matmul)
    DxdX_plus_DudU = bmm(Dx, dX[:-1]) + bmm(Du, dU[:-1])
    DxdX_plus_DudU = jnp.concatenate(
        [jnp.zeros_like(DxdX_plus_DudU[0])[None, ...], DxdX_plus_DudU]
    )
    Dx_T_dY_dyn = bmm(Dx.mT, dY_dyn[1:])
    Dx_T_dY_dyn = jnp.concatenate(
        [Dx_T_dY_dyn, jnp.zeros_like(Dx_T_dY_dyn[0])[None, ...]]
    )
    Du_T_dY_dyn = bmm(Du.mT, dY_dyn[1:])
    Du_T_dY_dyn = jnp.concatenate(
        [Du_T_dY_dyn, jnp.zeros_like(Du_T_dY_dyn[0])[None, ...]]
    )

    res_X = (
        bmm(Q, dX)
        + bmm(M, dU)
        + Dx_T_dY_dyn
        - dY_dyn
        + bmm(Ex.mT, dY_eq)
        + bmm(Gx.mT, dZ)
        + r_x[:, :x_dim]
    )

    res_U = (
        bmm(M.mT, dX)
        + bmm(R, dU)
        + Du_T_dY_dyn
        + bmm(Eu.mT, dY_eq)
        + bmm(Gu.mT, dZ)
        + r_x[:, x_dim:]
    )

    res_S = w_inv * dS + dZ + r_s

    res_Y_dyn = DxdX_plus_DudU - dX - η_inv * dY_dyn + r_y_dyn

    res_Y_eq = bmm(Ex, dX) + bmm(Eu, dU) - η_inv * dY_eq + r_y_eq

    res_Z = bmm(Gx, dX) + bmm(Gu, dU) + dS - η_inv * dZ + r_z

    return res_X, res_U, res_S, res_Y_dyn, res_Y_eq, res_Z


def _solve_kkt_3x3(
    P: jnp.ndarray,
    D: jnp.ndarray,
    E: jnp.ndarray,
    G: jnp.ndarray,
    w: jnp.ndarray,
    r_x: jnp.ndarray,
    r_y_dyn: jnp.ndarray,
    r_y_eq: jnp.ndarray,
    r_z: jnp.ndarray,
    η: float,
    use_parallel_lqr: bool,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    x_dim = D.shape[1]
    reg_w_inv = 1.0 / (w + 1.0 / η)
    bmm = jax.vmap(jnp.matmul)
    dX, dU, dY_dyn = _solve_kkt_2x2(
        P=(P + η * bmm(E.mT, E) + bmm(G.mT, reg_w_inv[..., None] * G)),
        D=D,
        r_x=(r_x + bmm(G.mT, reg_w_inv * r_z) + bmm(E.mT, η * r_y_eq)),
        r_y=r_y_dyn,
        η=η,
        use_parallel_lqr=use_parallel_lqr,
    )
    Ex = E[:, :, :x_dim]
    Eu = E[:, :, x_dim:]
    dY_eq = η * (bmm(Ex, dX) + bmm(Eu, dU) + r_y_eq)
    Gx = G[:, :, :x_dim]
    Gu = G[:, :, x_dim:]
    dZ = reg_w_inv * (bmm(Gx, dX) + bmm(Gu, dU) + r_z)
    return dX, dU, dY_dyn, dY_eq, dZ


def _solve_kkt_2x2(
    P: jnp.ndarray,
    D: jnp.ndarray,
    r_x: jnp.ndarray,
    r_y: jnp.ndarray,
    η: float,
    use_parallel_lqr: bool,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    solve_fn = solve_parallel if use_parallel_lqr else solve
    N = P.shape[0] - 1
    x_dim = D.shape[1]
    dX, dU, dY_dyn, _V, _v, _K, _k = solve_fn(
        A=D[:, :, :x_dim],
        B=D[:, :, x_dim:],
        Q=P[:, :x_dim, :x_dim],
        M=P[:-1, :x_dim, x_dim:],
        R=P[:-1, x_dim:, x_dim:],
        q=r_x[:, :x_dim],
        r=r_x[:-1, x_dim:],
        c=r_y,
        Δ=jnp.full(shape=(N + 1,), fill_value=(1.0 / η)),
    )
    dU = jnp.concatenate([dU, jnp.zeros_like(dU[0])[None, ...]])
    return dX, dU, dY_dyn
