"""Topology-aware factorization and solve helpers for Newton-KKT systems."""

from functools import partial

import jax
from jax import numpy as jnp
from regularized_lqr_jax.solver import (
    factor,
    factor_parallel,
    solve,
    solve_parallel,
)
from regularized_lqr_jax.tree_solver import factor_tree, solve_tree
from regularized_lqr_jax.types import FactorizationInputs as LQRFactorizationInputs
from regularized_lqr_jax.types import FactorizationOutputs as LQRFactorizationOutputs
from regularized_lqr_jax.types import SolveInputs as LQRSolveInputs
from regularized_lqr_jax.types import SolveOutputs as LQRSolveOutputs

from primal_dual_lipa.topology import TreeOCPTopology, edge_children, edge_parents
from primal_dual_lipa.types import (
    KKTFactorizationInputs,
    KKTFactorizationOutputs,
    NodeAndEdgeValues,
    TreeVariables,
    node_edge_map,
)


def tree_all_finite(tree: object) -> jax.Array:
    """Return whether all array leaves in a pytree are finite."""
    result = jnp.ones((), dtype=jnp.bool_)
    for leaf in jax.tree_util.tree_leaves(tree):
        result = jnp.logical_and(result, jnp.all(jnp.isfinite(leaf)))
    return result


def _regularized_inequality_weight(inputs: KKTFactorizationInputs) -> NodeAndEdgeValues:
    w = node_edge_map(lambda value: 1.0 / value, inputs.w_inv)
    return node_edge_map(
        lambda value, eta: 1.0 / (value + 1.0 / eta), w, inputs.params.η_ineq
    )


@partial(jax.jit, static_argnames=("use_parallel_lqr",))
def lqr_solve_kkt(
    lqr_inputs: LQRFactorizationInputs,
    lqr_outputs: LQRFactorizationOutputs,
    factorization_inputs: KKTFactorizationInputs,
    rhs: TreeVariables,
    use_parallel_lqr: bool,
    topology: TreeOCPTopology | None = None,
) -> TreeVariables:
    """Solve the LQR portion of a chain or tree Newton-KKT system."""
    inputs = factorization_inputs
    num_edges = inputs.M.shape[0]
    x_dim = inputs.Q.shape[-1]
    parents = edge_parents(topology, num_edges)
    equality_nodes = inputs.equality_locations.node
    equality_edges = inputs.equality_locations.edge
    inequality_nodes = inputs.inequality_locations.node
    inequality_edges = inputs.inequality_locations.edge
    w = node_edge_map(lambda value: 1.0 / value, inputs.w_inv)
    reg_w_inv = _regularized_inequality_weight(inputs)

    equality_node_rhs = jnp.einsum(
        "nki,nk->ni", inputs.E.node, inputs.params.η_eq.node * rhs.Y_eq.node
    )
    inequality_node_rhs = jnp.einsum(
        "ngi,ng->ni",
        inputs.G.node,
        reg_w_inv.node * (rhs.Z.node - w.node * rhs.S.node),
    )
    equality_edge_rhs = jnp.einsum(
        "eki,ek->ei", inputs.E.edge, inputs.params.η_eq.edge * rhs.Y_eq.edge
    )
    inequality_edge_rhs = jnp.einsum(
        "egi,eg->ei",
        inputs.G.edge,
        reg_w_inv.edge * (rhs.Z.edge - w.edge * rhs.S.edge),
    )
    r_x = rhs.X.at[equality_nodes].add(equality_node_rhs)
    r_x = r_x.at[inequality_nodes].add(inequality_node_rhs)
    r_x = r_x.at[parents[equality_edges]].add(equality_edge_rhs[:, :x_dim])
    r_x = r_x.at[parents[inequality_edges]].add(inequality_edge_rhs[:, :x_dim])
    r_u = rhs.U.at[equality_edges].add(equality_edge_rhs[:, x_dim:])
    r_u = r_u.at[inequality_edges].add(inequality_edge_rhs[:, x_dim:])

    solve_inputs = LQRSolveInputs(q=r_x, r=r_u, c=rhs.Y_dyn)
    if topology is None:
        solve_fn = solve_parallel if use_parallel_lqr else solve
        solve_outputs: LQRSolveOutputs = solve_fn(
            factorization_inputs=lqr_inputs,
            factorization_outputs=lqr_outputs,
            solve_inputs=solve_inputs,
        )
    else:
        solve_outputs = solve_tree(
            topology.plan,
            factorization_inputs=lqr_inputs,
            factorization_outputs=lqr_outputs,
            solve_inputs=solve_inputs,
        )

    dX = solve_outputs.X
    dU = solve_outputs.U
    dY_dyn = solve_outputs.Y
    primal_product_E = NodeAndEdgeValues(
        node=jnp.einsum("vki,vi->vk", inputs.E.node, dX[equality_nodes]),
        edge=(
            jnp.einsum(
                "ekn,en->ek",
                inputs.E.edge[:, :, :x_dim],
                dX[parents[equality_edges]],
            )
            + jnp.einsum("ekm,em->ek", inputs.E.edge[:, :, x_dim:], dU[equality_edges])
        ),
    )
    primal_product_G = NodeAndEdgeValues(
        node=jnp.einsum("vgi,vi->vg", inputs.G.node, dX[inequality_nodes]),
        edge=(
            jnp.einsum(
                "egn,en->eg",
                inputs.G.edge[:, :, :x_dim],
                dX[parents[inequality_edges]],
            )
            + jnp.einsum(
                "egm,em->eg", inputs.G.edge[:, :, x_dim:], dU[inequality_edges]
            )
        ),
    )
    dY_eq = node_edge_map(
        lambda eta, product, residual: eta * (product + residual),
        inputs.params.η_eq,
        primal_product_E,
        rhs.Y_eq,
    )
    dZ = node_edge_map(
        lambda weight, product, residual, inv_weight, slack_residual: (
            weight * (product + residual - inv_weight * slack_residual)
        ),
        reg_w_inv,
        primal_product_G,
        rhs.Z,
        w,
        rhs.S,
    )
    dS = node_edge_map(
        lambda inv_weight, slack_residual, dz: -inv_weight * (slack_residual + dz),
        w,
        rhs.S,
        dZ,
    )

    return TreeVariables(
        X=dX,
        U=dU,
        S=dS,
        Y_dyn=dY_dyn,
        Y_eq=dY_eq,
        Z=dZ,
        Theta=jnp.empty(0),
    )


@partial(jax.jit, static_argnames=("use_parallel_lqr",))
def factor_kkt(
    inputs: KKTFactorizationInputs,
    use_parallel_lqr: bool,
    topology: TreeOCPTopology | None = None,
) -> KKTFactorizationOutputs:
    """Factorize a chain or rooted-tree Newton-KKT linear system."""
    x_dim = inputs.Q.shape[-1]
    lqr_inputs = LQRFactorizationInputs(
        A=inputs.D[:, :, :x_dim],
        B=inputs.D[:, :, x_dim:],
        Q=inputs.Q_lqr,
        M=inputs.M_lqr,
        R=inputs.R_lqr,
        Δ_L=jax.vmap(jnp.diag)(jnp.sqrt(1.0 / inputs.params.η_dyn)),
    )
    if topology is None:
        factor_fn = factor_parallel if use_parallel_lqr else factor
        lqr_outputs = factor_fn(lqr_inputs)
    else:
        lqr_outputs = factor_tree(topology.plan, lqr_inputs)

    H_theta_X_T = jnp.moveaxis(inputs.H_theta_X, -1, 0)
    H_theta_U_T = jnp.moveaxis(inputs.H_theta_U, -1, 0)
    H_theta_y_dyn_T = jnp.moveaxis(inputs.H_theta_y_dyn, -1, 0)
    H_theta_y_eq_T = node_edge_map(
        lambda value: jnp.moveaxis(value, -1, 0), inputs.H_theta_y_eq
    )
    H_theta_z_T = node_edge_map(
        lambda value: jnp.moveaxis(value, -1, 0), inputs.H_theta_z
    )

    def partial_solve(
        h_x: jax.Array,
        h_u: jax.Array,
        b_dyn: jax.Array,
        b_eq: NodeAndEdgeValues,
        b_ineq: NodeAndEdgeValues,
    ) -> TreeVariables:
        rhs = TreeVariables(
            X=h_x,
            U=h_u,
            S=NodeAndEdgeValues(
                node=jnp.zeros_like(inputs.G.node[:, :, 0]),
                edge=jnp.zeros_like(inputs.G.edge[:, :, 0]),
            ),
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
            topology=topology,
        )

    sol = jax.vmap(partial_solve)(
        H_theta_X_T,
        H_theta_U_T,
        H_theta_y_dyn_T,
        H_theta_y_eq_T,
        H_theta_z_T,
    )
    schur_term = (
        jnp.einsum("vnp,jvn->pj", inputs.H_theta_X, sol.X)
        + jnp.einsum("emp,jem->pj", inputs.H_theta_U, sol.U)
        + jnp.einsum("vnp,jvn->pj", inputs.H_theta_y_dyn, sol.Y_dyn)
        + jnp.einsum("nkp,jnk->pj", inputs.H_theta_y_eq.node, sol.Y_eq.node)
        + jnp.einsum("ekp,jek->pj", inputs.H_theta_y_eq.edge, sol.Y_eq.edge)
        + jnp.einsum("ngp,jng->pj", inputs.H_theta_z.node, sol.Z.node)
        + jnp.einsum("egp,jeg->pj", inputs.H_theta_z.edge, sol.Z.edge)
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


def factorization_is_valid(
    outputs: KKTFactorizationOutputs,
    pd_tol: jnp.double = 0.0,
    singular_tol: jnp.double = 0.0,
) -> jax.Array:
    """Check whether KKT factorization outputs are numerically usable."""
    g_diag = jnp.diagonal(outputs.lqr_outputs.G_cho, axis1=-2, axis2=-1)
    valid = jnp.logical_and(tree_all_finite(outputs), jnp.all(g_diag > pd_tol))
    if hasattr(outputs.lqr_outputs, "S_cho"):
        s_diag = jnp.diagonal(outputs.lqr_outputs.S_cho, axis1=-2, axis2=-1)
        valid = jnp.logical_and(valid, jnp.all(s_diag > pd_tol))
    else:
        f_diag = jnp.abs(jnp.diagonal(outputs.lqr_outputs.F_lu, axis1=-2, axis2=-1))
        valid = jnp.logical_and(valid, jnp.all(f_diag > singular_tol))

    if outputs.schur_complement.shape[0] > 0:
        schur_singular_values = jnp.linalg.svd(
            outputs.schur_complement, compute_uv=False
        )
        valid = jnp.logical_and(valid, jnp.all(schur_singular_values > singular_tol))
    return valid


@partial(jax.jit, static_argnames=("use_parallel_lqr",))
def solve_kkt(
    factorization_outputs: KKTFactorizationOutputs,
    factorization_inputs: KKTFactorizationInputs,
    rhs: TreeVariables,
    use_parallel_lqr: bool,
    topology: TreeOCPTopology | None = None,
) -> TreeVariables:
    """Solve a Newton-KKT system with a precomputed factorization."""
    sol0 = lqr_solve_kkt(
        lqr_inputs=factorization_outputs.lqr_inputs,
        lqr_outputs=factorization_outputs.lqr_outputs,
        factorization_inputs=factorization_inputs,
        rhs=rhs,
        use_parallel_lqr=use_parallel_lqr,
        topology=topology,
    )
    B_sol0 = (
        jnp.einsum("vnp,vn->p", factorization_inputs.H_theta_X, sol0.X)
        + jnp.einsum("emp,em->p", factorization_inputs.H_theta_U, sol0.U)
        + jnp.einsum("vnp,vn->p", factorization_inputs.H_theta_y_dyn, sol0.Y_dyn)
        + jnp.einsum(
            "nkp,nk->p", factorization_inputs.H_theta_y_eq.node, sol0.Y_eq.node
        )
        + jnp.einsum(
            "ekp,ek->p", factorization_inputs.H_theta_y_eq.edge, sol0.Y_eq.edge
        )
        + jnp.einsum("ngp,ng->p", factorization_inputs.H_theta_z.node, sol0.Z.node)
        + jnp.einsum("egp,eg->p", factorization_inputs.H_theta_z.edge, sol0.Z.edge)
    )
    dTheta = jnp.linalg.solve(
        factorization_outputs.schur_complement, -rhs.Theta - B_sol0
    )

    def add_parameter_correction(base: jax.Array, correction: jax.Array) -> jax.Array:
        return base + jnp.einsum("p,p...->...", dTheta, correction)

    return TreeVariables(
        X=add_parameter_correction(sol0.X, factorization_outputs.B_inv_C_X),
        U=add_parameter_correction(sol0.U, factorization_outputs.B_inv_C_U),
        S=node_edge_map(
            add_parameter_correction, sol0.S, factorization_outputs.B_inv_C_S
        ),
        Y_dyn=add_parameter_correction(sol0.Y_dyn, factorization_outputs.B_inv_C_Y_dyn),
        Y_eq=node_edge_map(
            add_parameter_correction,
            sol0.Y_eq,
            factorization_outputs.B_inv_C_Y_eq,
        ),
        Z=node_edge_map(
            add_parameter_correction, sol0.Z, factorization_outputs.B_inv_C_Z
        ),
        Theta=dTheta,
    )


def compute_kkt_residual(
    factorization_inputs: KKTFactorizationInputs,
    solve_inputs: TreeVariables,
    solution: TreeVariables,
    topology: TreeOCPTopology | None = None,
) -> TreeVariables:
    """Compute the residual of a chain or rooted-tree Newton-KKT system."""
    inputs = factorization_inputs
    num_edges = inputs.M.shape[0]
    x_dim = inputs.Q.shape[-1]
    parents = edge_parents(topology, num_edges)
    children = edge_children(topology, num_edges)
    equality_nodes = inputs.equality_locations.node
    equality_edges = inputs.equality_locations.edge
    inequality_nodes = inputs.inequality_locations.node
    inequality_edges = inputs.inequality_locations.edge
    Dx = inputs.D[:, :, :x_dim]
    Du = inputs.D[:, :, x_dim:]
    Eex = inputs.E.edge[:, :, :x_dim]
    Eeu = inputs.E.edge[:, :, x_dim:]
    Gex = inputs.G.edge[:, :, :x_dim]
    Geu = inputs.G.edge[:, :, x_dim:]

    res_X = (
        jnp.einsum("vnm,vm->vn", inputs.Q, solution.X)
        - solution.Y_dyn
        + jnp.einsum("vnp,p->vn", inputs.H_theta_X, solution.Theta)
        + solve_inputs.X
    )
    dynamics_outgoing = jnp.einsum("enm,em->en", inputs.M, solution.U) + jnp.einsum(
        "eni,ei->en", jnp.swapaxes(Dx, -2, -1), solution.Y_dyn[children]
    )
    res_X = res_X.at[parents].add(dynamics_outgoing)
    res_X = res_X.at[equality_nodes].add(
        jnp.einsum("vki,vk->vi", inputs.E.node, solution.Y_eq.node)
    )
    res_X = res_X.at[inequality_nodes].add(
        jnp.einsum("vgi,vg->vi", inputs.G.node, solution.Z.node)
    )
    res_X = res_X.at[parents[equality_edges]].add(
        jnp.einsum("ekn,ek->en", Eex, solution.Y_eq.edge)
    )
    res_X = res_X.at[parents[inequality_edges]].add(
        jnp.einsum("egn,eg->en", Gex, solution.Z.edge)
    )

    res_U = (
        jnp.einsum("enm,en->em", inputs.M, solution.X[parents])
        + jnp.einsum("emn,en->em", inputs.R, solution.U)
        + jnp.einsum("emn,en->em", jnp.swapaxes(Du, -2, -1), solution.Y_dyn[children])
        + jnp.einsum("emp,p->em", inputs.H_theta_U, solution.Theta)
        + solve_inputs.U
    )
    res_U = res_U.at[equality_edges].add(
        jnp.einsum("ekm,ek->em", Eeu, solution.Y_eq.edge)
    )
    res_U = res_U.at[inequality_edges].add(
        jnp.einsum("egm,eg->em", Geu, solution.Z.edge)
    )
    res_S = node_edge_map(
        lambda weight, slack, z, rhs: weight * slack + z + rhs,
        inputs.w_inv,
        solution.S,
        solution.Z,
        solve_inputs.S,
    )

    res_Y_dyn = (
        -solution.X
        - (1.0 / inputs.params.η_dyn) * solution.Y_dyn
        + jnp.einsum("vnp,p->vn", inputs.H_theta_y_dyn, solution.Theta)
        + solve_inputs.Y_dyn
    )
    edge_dynamics = jnp.einsum("eni,ei->en", Dx, solution.X[parents]) + jnp.einsum(
        "enm,em->en", Du, solution.U
    )
    res_Y_dyn = res_Y_dyn.at[children].add(edge_dynamics)

    res_Y_eq = NodeAndEdgeValues(
        node=(
            jnp.einsum("vki,vi->vk", inputs.E.node, solution.X[equality_nodes])
            - (1.0 / inputs.params.η_eq.node) * solution.Y_eq.node
            + jnp.einsum("nkp,p->nk", inputs.H_theta_y_eq.node, solution.Theta)
            + solve_inputs.Y_eq.node
        ),
        edge=(
            jnp.einsum("ekn,en->ek", Eex, solution.X[parents[equality_edges]])
            + jnp.einsum("ekm,em->ek", Eeu, solution.U[equality_edges])
            - (1.0 / inputs.params.η_eq.edge) * solution.Y_eq.edge
            + jnp.einsum("ekp,p->ek", inputs.H_theta_y_eq.edge, solution.Theta)
            + solve_inputs.Y_eq.edge
        ),
    )
    res_Z = NodeAndEdgeValues(
        node=(
            jnp.einsum("vgi,vi->vg", inputs.G.node, solution.X[inequality_nodes])
            + solution.S.node
            - (1.0 / inputs.params.η_ineq.node) * solution.Z.node
            + jnp.einsum("ngp,p->ng", inputs.H_theta_z.node, solution.Theta)
            + solve_inputs.Z.node
        ),
        edge=(
            jnp.einsum("egn,en->eg", Gex, solution.X[parents[inequality_edges]])
            + jnp.einsum("egm,em->eg", Geu, solution.U[inequality_edges])
            + solution.S.edge
            - (1.0 / inputs.params.η_ineq.edge) * solution.Z.edge
            + jnp.einsum("egp,p->eg", inputs.H_theta_z.edge, solution.Theta)
            + solve_inputs.Z.edge
        ),
    )
    B_dZ = (
        jnp.einsum("vnp,vn->p", inputs.H_theta_X, solution.X)
        + jnp.einsum("emp,em->p", inputs.H_theta_U, solution.U)
        + jnp.einsum("vnp,vn->p", inputs.H_theta_y_dyn, solution.Y_dyn)
        + jnp.einsum("nkp,nk->p", inputs.H_theta_y_eq.node, solution.Y_eq.node)
        + jnp.einsum("ekp,ek->p", inputs.H_theta_y_eq.edge, solution.Y_eq.edge)
        + jnp.einsum("ngp,ng->p", inputs.H_theta_z.node, solution.Z.node)
        + jnp.einsum("egp,eg->p", inputs.H_theta_z.edge, solution.Z.edge)
    )
    res_Theta = inputs.H_theta_theta @ solution.Theta + B_dZ + solve_inputs.Theta

    return TreeVariables(
        X=res_X,
        U=res_U,
        S=res_S,
        Y_dyn=res_Y_dyn,
        Y_eq=res_Y_eq,
        Z=res_Z,
        Theta=res_Theta,
    )
