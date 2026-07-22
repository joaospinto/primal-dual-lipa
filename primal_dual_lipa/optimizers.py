"""Defines the main solve function, implementing the Primal-Dual LIPA algorithm."""

from functools import partial

import jax
from jax import numpy as jnp

from primal_dual_lipa.kkt_builder import (
    add_scalar_hessian_regularization_delta,
    build_kkt,
    build_kkt_rhs,
)
from primal_dual_lipa.kkt_helpers import (
    compute_kkt_residual,
    factor_kkt,
    factorization_is_valid,
    solve_kkt,
    tree_all_finite,
)
from primal_dual_lipa.lagrangian_helpers import (
    directional_augmented_lagrangian,
    dynamics_residuals,
    evaluate_node_edge,
)
from primal_dual_lipa.topology import (
    TreeOCPTopology,
    validate_callback_locations,
    validate_tree_shapes,
)
from primal_dual_lipa.types import (
    CostFunction,
    EdgeCostFunction,
    EdgeFunction,
    Function,
    KKTSystem,
    NodeAndEdgeIndices,
    NodeAndEdgeValues,
    NodeCostFunction,
    NodeFunction,
    OCPCallbackLocations,
    Parameters,
    SolverSettings,
    TreeParameters,
    TreeVariables,
    Variables,
    node_edge_flatten,
    node_edge_map,
    node_edge_sum,
)


def _zero_node_cost(x, theta, node):  # noqa: ANN001, ANN202
    del x, theta, node
    return 0.0


def _zero_edge_cost(x, u, theta, edge):  # noqa: ANN001, ANN202
    del x, u, theta, edge
    return 0.0


def _empty_node_function(x, theta, node):  # noqa: ANN001, ANN202
    del x, theta, node
    return jnp.empty(0)


def _empty_edge_function(x, u, theta, edge):  # noqa: ANN001, ANN202
    del x, u, theta, edge
    return jnp.empty(0)


def _all_callback_locations(num_nodes: int, num_edges: int) -> OCPCallbackLocations:
    """Select every node and edge for all three callback categories."""
    all_locations = NodeAndEdgeIndices(
        node=jnp.arange(num_nodes, dtype=jnp.int32),
        edge=jnp.arange(num_edges, dtype=jnp.int32),
    )
    return OCPCallbackLocations(
        cost=all_locations,
        equalities=all_locations,
        inequalities=all_locations,
    )


@partial(
    jax.jit,
    static_argnames=[
        "node_cost",
        "edge_cost",
        "dynamics",
        "node_equalities",
        "edge_equalities",
        "node_inequalities",
        "edge_inequalities",
    ],
)
def _solve_node_edge(
    vars_in: TreeVariables,
    x0: jax.Array,
    dynamics: EdgeFunction,
    settings: SolverSettings,
    *,
    node_cost: NodeCostFunction = _zero_node_cost,
    edge_cost: EdgeCostFunction = _zero_edge_cost,
    node_equalities: NodeFunction = _empty_node_function,
    edge_equalities: EdgeFunction = _empty_edge_function,
    node_inequalities: NodeFunction = _empty_node_function,
    edge_inequalities: EdgeFunction = _empty_edge_function,
    params_in: TreeParameters | None = None,
    topology: TreeOCPTopology | None = None,
    locations: OCPCallbackLocations | None = None,
) -> tuple[TreeVariables, jnp.int32, jnp.bool, TreeParameters]:
    """Implement the Primal-Dual LIPA algorithm for discrete-time optimal control.

    Args:
      vars_in:       TreeVariables object containing warm-start values for X, U, S, Y_dyn, Y_eq, Z, Theta.
      x0:            [n]           numpy array (initial state).
      dynamics:      dynamics function with signature dynamics(x, u, theta, t).
      settings:      the solver settings.
      node_cost:     cost function with signature node_cost(x, theta, node).
      edge_cost:     cost function with signature edge_cost(x, u, theta, edge).
      node_equalities: constraints node_equalities(x, theta, node) = 0.
      edge_equalities: constraints edge_equalities(x, u, theta, edge) = 0.
      node_inequalities: constraints node_inequalities(x, theta, node) <= 0.
      edge_inequalities: constraints edge_inequalities(x, u, theta, edge) <= 0.
      params_in:     Optional warm-start TreeParameters (µ, η_dyn, η_eq, η_ineq).
                     If None, initialized from settings.µ0 / settings.η0.
                     Useful for debugging and iterating from a previously
                     saved iterate without re-paying the ramp-up phase.
      topology:      Optional rooted-tree topology. If omitted, use the legacy
                     chain layout. State callbacks and dynamics duals are node
                     ordered; edge callbacks and controls are edge ordered.

    Returns:
      TreeVariables:   TreeVariables object containing the solution for X, U, S, Y_dyn, Y_eq, Z, Theta.
      iterations:  the number of iterations it took to converge.
      no_errors:   whether no errors were encountered during the solve.
      params:      Final TreeParameters (µ, η_dyn, η_eq, η_ineq) at termination.

    """

    if locations is None:
        locations = _all_callback_locations(vars_in.X.shape[0], vars_in.U.shape[0])

    def evaluate_equalities(X, U, Theta):  # noqa: ANN001, ANN202
        return evaluate_node_edge(
            node_equalities,
            edge_equalities,
            X,
            U,
            Theta,
            topology,
            locations.equalities,
        )

    def evaluate_inequalities(X, U, Theta):  # noqa: ANN001, ANN202
        return evaluate_node_edge(
            node_inequalities,
            edge_inequalities,
            X,
            U,
            Theta,
            topology,
            locations.inequalities,
        )

    def evaluate_cost(X, U, Theta):  # noqa: ANN001, ANN202
        return node_edge_sum(
            evaluate_node_edge(
                node_cost, edge_cost, X, U, Theta, topology, locations.cost
            )
        )

    validate_tree_shapes(
        topology,
        X=vars_in.X,
        U=vars_in.U,
        S=vars_in.S,
        Y_dyn=vars_in.Y_dyn,
        Y_eq=vars_in.Y_eq,
        Z=vars_in.Z,
        locations=locations,
    )
    vars_current = TreeVariables(
        X=vars_in.X,
        U=vars_in.U,
        S=node_edge_map(
            lambda value: jnp.maximum(value, jnp.sqrt(settings.µ0)), vars_in.S
        ),
        Y_dyn=vars_in.Y_dyn,
        Y_eq=vars_in.Y_eq,
        Z=node_edge_map(
            lambda value: jnp.maximum(value, jnp.sqrt(settings.µ0)), vars_in.Z
        ),
        Theta=vars_in.Theta,
    )

    # Infer residual shapes from inputs to initialize η
    num_nodes = vars_current.X.shape[0]
    n = vars_current.X.shape[1]

    if params_in is not None:
        expected_parameter_shapes = [
            ("η_dyn", params_in.η_dyn.shape, vars_current.Y_dyn.shape)
        ]
        for name, actual, expected in (
            ("η_eq", params_in.η_eq, vars_current.Y_eq),
            ("η_ineq", params_in.η_ineq, vars_current.Z),
        ):
            expected_parameter_shapes.extend(
                [
                    (f"{name}.node", actual.node.shape, expected.node.shape),
                    (f"{name}.edge", actual.edge.shape, expected.edge.shape),
                ]
            )
        for name, actual, expected in expected_parameter_shapes:
            if actual != expected:
                message = f"params_in.{name} must have shape {expected}; got {actual}"
                raise ValueError(message)
        params_current = TreeParameters(
            µ=params_in.µ,
            η_dyn=params_in.η_dyn,
            η_eq=params_in.η_eq,
            η_ineq=params_in.η_ineq,
        )
    else:
        η_dyn = jnp.full((num_nodes, n), settings.η0)
        η_eq = node_edge_map(
            lambda value: jnp.full_like(value, settings.η0), vars_current.Y_eq
        )
        η_ineq = node_edge_map(
            lambda value: jnp.full_like(value, settings.η0), vars_current.Z
        )
        params_current = TreeParameters(
            µ=settings.µ0, η_dyn=η_dyn, η_eq=η_eq, η_ineq=η_ineq
        )

    hess_reg_settings = settings.hessian_regularization
    hessian_regularization_current = jnp.maximum(
        hess_reg_settings.initial,
        hess_reg_settings.minimum,
    )

    kkt_system = build_kkt(
        node_cost=node_cost,
        edge_cost=edge_cost,
        dynamics=dynamics,
        node_equalities=node_equalities,
        edge_equalities=edge_equalities,
        node_inequalities=node_inequalities,
        edge_inequalities=edge_inequalities,
        x0=x0,
        vars=vars_current,
        params=params_current,
        hessian_regularization=hessian_regularization_current,
        regularize_slack_elimination_with_mu=(
            settings.regularize_slack_elimination_with_mu
        ),
        topology=topology,
        locations=locations,
    )

    if settings.print_logs and not settings.print_ls_logs:
        jax.debug.print(
            "{:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10}".format(  # noqa: E501
                "iteration",
                "α",
                "cost",
                "|c|",
                "|g+s|",
                "merit",
                "dmerit/dα",
                "dmerit/dαdX",
                "dmerit/dαdS",
                "|dx|+|du|",
                "|ds|",
                "|dy|",
                "|dz|",
                "|dθ|",
                "avg(η)",
                "µ",
                "δH",
                "δH it",
                "linsys_res",
            ),
            ordered=True,
        )

    def main_loop_body(
        inputs: tuple[
            TreeVariables,
            TreeParameters,
            KKTSystem,
            jnp.int32,
            jnp.double,
            jnp.double,
            jnp.double,
            jnp.double,
            jnp.double,
        ],
    ) -> tuple[
        TreeVariables,
        TreeParameters,
        KKTSystem,
        jnp.int32,
        jnp.double,
        jnp.double,
        jnp.double,
        jnp.double,
        jnp.double,
    ]:
        (
            vars,
            params,
            kkt_system,
            iteration,
            prev_iter_cost,
            _last_improvement,
            # Filter envelope: smallest θ and f seen so far. +∞ on entry
            # so the very first iterate is unconditionally acceptable.
            filter_theta,
            filter_f,
            hessian_regularization,
        ) = inputs

        τ = jnp.maximum(settings.τ_min, 1.0 - params.µ)

        def solve_trial(
            trial_kkt_system: KKTSystem,
        ) -> tuple[object, TreeVariables, jax.Array]:
            trial_factorization_outputs = factor_kkt(
                inputs=trial_kkt_system.lhs,
                use_parallel_lqr=settings.use_parallel_lqr,
                topology=topology,
            )
            trial_deltas = solve_kkt(
                factorization_outputs=trial_factorization_outputs,
                factorization_inputs=trial_kkt_system.lhs,
                rhs=trial_kkt_system.rhs,
                use_parallel_lqr=settings.use_parallel_lqr,
                topology=topology,
            )
            valid = jnp.logical_and(
                factorization_is_valid(
                    trial_factorization_outputs,
                    pd_tol=hess_reg_settings.pd_tol,
                    singular_tol=hess_reg_settings.singular_tol,
                ),
                tree_all_finite(trial_deltas),
            )
            trial_dal = directional_augmented_lagrangian(
                node_cost=node_cost,
                edge_cost=edge_cost,
                dynamics=dynamics,
                node_equalities=node_equalities,
                edge_equalities=edge_equalities,
                node_inequalities=node_inequalities,
                edge_inequalities=edge_inequalities,
                x0=x0,
                params=params,
                τ=τ,
                topology=topology,
                locations=locations,
                variables=vars,
                deltas=trial_deltas,
            )
            trial_merit_grad = jax.grad(trial_dal)(0.0)
            descent_valid = trial_merit_grad < -hess_reg_settings.descent_tol
            valid = jnp.logical_and(
                valid,
                jnp.logical_and(jnp.isfinite(trial_merit_grad), descent_valid),
            )
            return trial_factorization_outputs, trial_deltas, valid

        factorization_outputs, deltas, regularization_valid = solve_trial(kkt_system)

        def next_hessian_regularization(reg: jax.Array) -> jax.Array:
            from_zero = jnp.where(
                hess_reg_settings.minimum > 0.0,
                hess_reg_settings.minimum,
                hess_reg_settings.first_positive,
            )
            increased = jnp.where(
                reg >= hess_reg_settings.first_positive,
                reg * hess_reg_settings.increase_factor,
                from_zero,
            )
            return jnp.minimum(increased, hess_reg_settings.maximum)

        def regularization_loop_cond(
            state: tuple[
                jnp.double,
                jnp.int32,
                KKTSystem,
                object,
                TreeVariables,
                jnp.bool,
            ],
        ) -> jnp.bool:
            reg, attempt, _trial_kkt, _factorization, _deltas, valid = state
            can_try_more = jnp.logical_and(
                attempt < hess_reg_settings.max_attempts,
                reg < hess_reg_settings.maximum,
            )
            return jnp.logical_and(jnp.logical_not(valid), can_try_more)

        def regularization_loop_body(
            state: tuple[
                jnp.double,
                jnp.int32,
                KKTSystem,
                object,
                TreeVariables,
                jnp.bool,
            ],
        ) -> tuple[jnp.double, jnp.int32, KKTSystem, object, TreeVariables, jnp.bool]:
            reg, attempt, _trial_kkt, _factorization, _deltas, _valid = state
            reg_new = next_hessian_regularization(reg)
            trial_kkt_system = add_scalar_hessian_regularization_delta(
                _trial_kkt,
                reg_new - reg,
            )
            trial_factorization_outputs, trial_deltas, valid = solve_trial(
                trial_kkt_system
            )
            return (
                reg_new,
                attempt + 1,
                trial_kkt_system,
                trial_factorization_outputs,
                trial_deltas,
                valid,
            )

        (
            hessian_regularization_used,
            regularization_attempts,
            kkt_system,
            factorization_outputs,
            deltas,
            regularization_valid,
        ) = jax.lax.while_loop(
            regularization_loop_cond,
            regularization_loop_body,
            (
                hessian_regularization,
                jnp.array(0, dtype=jnp.int32),
                kkt_system,
                factorization_outputs,
                deltas,
                regularization_valid,
            ),
        )

        def iterative_refinement_loop_continuation_criteria(
            x: tuple[TreeVariables, jnp.int32],
        ) -> jnp.bool:
            return jnp.logical_and(
                regularization_valid,
                x[-1] < settings.num_iterative_refinement_steps,
            )

        def iterative_refinement_loop_body(
            x: tuple[TreeVariables, jnp.int32],
        ) -> tuple[TreeVariables, jnp.int32]:
            deltas_inner, it = x
            residuals = compute_kkt_residual(
                factorization_inputs=kkt_system.lhs,
                solve_inputs=kkt_system.rhs,
                solution=deltas_inner,
                topology=topology,
            )

            refinement_deltas = solve_kkt(
                factorization_outputs=factorization_outputs,
                factorization_inputs=kkt_system.lhs,
                rhs=residuals,
                use_parallel_lqr=settings.use_parallel_lqr,
                topology=topology,
            )

            return (
                TreeVariables(
                    X=deltas_inner.X + refinement_deltas.X,
                    U=deltas_inner.U + refinement_deltas.U,
                    S=node_edge_map(
                        lambda lhs, rhs: lhs + rhs,
                        deltas_inner.S,
                        refinement_deltas.S,
                    ),
                    Y_dyn=deltas_inner.Y_dyn + refinement_deltas.Y_dyn,
                    Y_eq=node_edge_map(
                        lambda lhs, rhs: lhs + rhs,
                        deltas_inner.Y_eq,
                        refinement_deltas.Y_eq,
                    ),
                    Z=node_edge_map(
                        lambda lhs, rhs: lhs + rhs,
                        deltas_inner.Z,
                        refinement_deltas.Z,
                    ),
                    Theta=deltas_inner.Theta + refinement_deltas.Theta,
                ),
                it + 1,
            )

        deltas, _ = jax.lax.while_loop(
            iterative_refinement_loop_continuation_criteria,
            iterative_refinement_loop_body,
            (deltas, 0),
        )
        regularization_valid = jnp.logical_and(
            regularization_valid,
            tree_all_finite(deltas),
        )

        def compute_alpha_max(v, dv, τ):
            mask = dv < 0
            safe_dv = jnp.where(mask, dv, -1.0)
            alphas = jnp.where(mask, -(τ * v) / safe_dv, 1.0)
            return jnp.min(jnp.concatenate([alphas.flatten(), jnp.array([1.0])]))

        α_max_s = jnp.minimum(
            compute_alpha_max(vars.S.node, deltas.S.node, τ),
            compute_alpha_max(vars.S.edge, deltas.S.edge, τ),
        )

        dal = directional_augmented_lagrangian(
            node_cost=node_cost,
            edge_cost=edge_cost,
            dynamics=dynamics,
            node_equalities=node_equalities,
            edge_equalities=edge_equalities,
            node_inequalities=node_inequalities,
            edge_inequalities=edge_inequalities,
            x0=x0,
            params=params,
            τ=τ,
            topology=topology,
            locations=locations,
            variables=vars,
            deltas=deltas,
        )

        dal_x = directional_augmented_lagrangian(
            node_cost=node_cost,
            edge_cost=edge_cost,
            dynamics=dynamics,
            node_equalities=node_equalities,
            edge_equalities=edge_equalities,
            node_inequalities=node_inequalities,
            edge_inequalities=edge_inequalities,
            x0=x0,
            params=params,
            τ=τ,
            topology=topology,
            locations=locations,
            variables=vars,
            deltas=TreeVariables(
                X=deltas.X,
                U=deltas.U,
                S=node_edge_map(jnp.zeros_like, deltas.S),
                Y_dyn=jnp.zeros_like(deltas.Y_dyn),
                Y_eq=node_edge_map(jnp.zeros_like, deltas.Y_eq),
                Z=node_edge_map(jnp.zeros_like, deltas.Z),
                Theta=jnp.zeros_like(deltas.Theta),
            ),
        )

        dal_s = directional_augmented_lagrangian(
            node_cost=node_cost,
            edge_cost=edge_cost,
            dynamics=dynamics,
            node_equalities=node_equalities,
            edge_equalities=edge_equalities,
            node_inequalities=node_inequalities,
            edge_inequalities=edge_inequalities,
            x0=x0,
            params=params,
            τ=τ,
            topology=topology,
            locations=locations,
            variables=vars,
            deltas=TreeVariables(
                X=jnp.zeros_like(deltas.X),
                U=jnp.zeros_like(deltas.U),
                S=deltas.S,
                Y_dyn=jnp.zeros_like(deltas.Y_dyn),
                Y_eq=node_edge_map(jnp.zeros_like, deltas.Y_eq),
                Z=node_edge_map(jnp.zeros_like, deltas.Z),
                Theta=jnp.zeros_like(deltas.Theta),
            ),
        )

        baseline_merit = dal(0.0)

        merit_grad = jax.grad(dal)(0.0)
        merit_grad_x = jax.grad(dal_x)(0.0)
        merit_grad_s = jax.grad(dal_s)(0.0)

        if settings.print_ls_logs:
            jax.debug.print(
                "{:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10}".format(
                    "",
                    "α",
                    "cost",
                    "|c|",
                    "|g+s|",
                    "dmerit",
                    "dmerit/α",
                ),
                ordered=True,
            )

        def compute_theta_f(α: jnp.double) -> tuple[jnp.double, jnp.double]:
            """Compute (θ, f) at trial step `α` for the filter line search.

            θ = squared L2 norm of the primal violation (dyn + eq + (g+s)).
            f = barrier-augmented objective (cost − μ·sum_log_S), i.e. the
            cost component of the merit function in `dal` excluding the
            augmented-Lagrangian dual penalty terms.
            """
            cX = vars.X + α * deltas.X
            cU = vars.U + α * deltas.U
            cS = node_edge_map(
                lambda value, delta: jnp.maximum(value + α * delta, (1.0 - τ) * value),
                vars.S,
                deltas.S,
            )
            cTheta = vars.Theta + α * deltas.Theta
            candidate_vars = TreeVariables(
                X=cX,
                U=cU,
                S=cS,
                Y_dyn=vars.Y_dyn,
                Y_eq=vars.Y_eq,
                Z=vars.Z,
                Theta=cTheta,
            )
            c_dyn = dynamics_residuals(dynamics, x0, candidate_vars, topology)
            c_eq = evaluate_equalities(cX, cU, cTheta)
            g = evaluate_inequalities(cX, cU, cTheta)
            gps = node_edge_flatten(
                node_edge_map(lambda value, slack: value + slack, g, cS)
            )
            c_all = jnp.concatenate([c_dyn.flatten(), node_edge_flatten(c_eq), gps])
            theta = jnp.sum(jnp.square(c_all))

            f_cost = evaluate_cost(cX, cU, cTheta)
            f_barrier = -params.µ * node_edge_sum(node_edge_map(jnp.log, cS))
            f_val = f_cost + f_barrier
            return theta, f_val

        def check_alpha(α: jnp.double) -> jnp.bool:
            candidate_merit = dal(α)
            armijo_condition_met = (
                candidate_merit - baseline_merit
                <= α * settings.armijo_factor * merit_grad
            )
            if settings.use_filter_line_search:
                # Accept if EITHER the primal violation θ or the
                # barrier-augmented objective f strictly improves on the
                # historical envelope, with a wedge margin (Wächter–Biegler
                # 2006 §3, eq. 18-19). ORed with the Armijo condition so
                # the filter strictly enlarges the acceptable α set.
                theta_α, f_α = compute_theta_f(α)
                filter_accept = jnp.logical_or(
                    theta_α <= (1.0 - settings.filter_gamma_theta) * filter_theta,
                    f_α <= filter_f - settings.filter_gamma_f * filter_theta,
                )
                accepted = jnp.logical_or(armijo_condition_met, filter_accept)
            else:
                accepted = armijo_condition_met

            if settings.print_ls_logs:
                cX = vars.X + α * deltas.X
                cU = vars.U + α * deltas.U
                cS = node_edge_map(
                    lambda value, delta: jnp.maximum(
                        value + α * delta, (1.0 - τ) * value
                    ),
                    vars.S,
                    deltas.S,
                )
                cTheta = vars.Theta + α * deltas.Theta
                candidate_vars = TreeVariables(
                    X=cX,
                    U=cU,
                    S=cS,
                    Y_dyn=vars.Y_dyn,
                    Y_eq=vars.Y_eq,
                    Z=vars.Z,
                    Theta=cTheta,
                )
                c_dyn = dynamics_residuals(dynamics, x0, candidate_vars, topology)
                c_eq = evaluate_equalities(cX, cU, cTheta)
                c = jnp.concatenate([c_dyn.flatten(), node_edge_flatten(c_eq)])

                g = evaluate_inequalities(cX, cU, cTheta)
                gps = node_edge_flatten(
                    node_edge_map(lambda value, slack: value + slack, g, cS)
                )
                dmerit = candidate_merit - baseline_merit
                jax.debug.print(
                    "{:^10} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g}",  # noqa: E501
                    "",
                    α,
                    evaluate_cost(cX, cU, cTheta),
                    jnp.linalg.norm(c),
                    jnp.linalg.norm(gps),
                    dmerit,
                    dmerit / α,
                    ordered=True,
                )
            return accepted

        def line_search_iteration(
            state: tuple[jnp.double, jnp.bool, jnp.double],
        ) -> tuple[jnp.double, jnp.bool, jnp.double]:
            α_base, _, _ = state
            factors = settings.α_update_factor ** jnp.arange(
                settings.num_parallel_line_search_steps
            )
            alphas = α_base * factors
            armijo_mets = jax.vmap(check_alpha)(alphas)

            combined_condition = jnp.logical_or(armijo_mets, alphas <= settings.α_min)
            indices = jnp.arange(settings.num_parallel_line_search_steps)
            valid_indices = jnp.where(
                combined_condition, indices, settings.num_parallel_line_search_steps
            )
            first_idx = jnp.min(valid_indices)

            found_now = first_idx < settings.num_parallel_line_search_steps
            α_res = alphas[
                jnp.minimum(first_idx, settings.num_parallel_line_search_steps - 1)
            ]

            new_α_base = α_base * (
                settings.α_update_factor**settings.num_parallel_line_search_steps
            )
            return new_α_base, found_now, α_res

        def line_search_cond(
            state: tuple[jnp.double, jnp.bool, jnp.double],
        ) -> jnp.bool:
            _, found, _ = state
            return jnp.logical_not(found)

        if settings.skip_line_search:
            α = α_max_s
        else:
            _, _, α = jax.lax.while_loop(
                line_search_cond,
                line_search_iteration,
                (α_max_s, False, 0.0),
            )

        if settings.print_logs:
            residuals = compute_kkt_residual(
                factorization_inputs=kkt_system.lhs,
                solve_inputs=kkt_system.rhs,
                solution=deltas,
                topology=topology,
            )
            residual_vec = jnp.concatenate(
                [
                    residuals.X.flatten(),
                    residuals.U.flatten(),
                    node_edge_flatten(residuals.S),
                    residuals.Y_dyn.flatten(),
                    node_edge_flatten(residuals.Y_eq),
                    node_edge_flatten(residuals.Z),
                    residuals.Theta.flatten(),
                ]
            )

            c_dyn = dynamics_residuals(dynamics, x0, vars, topology)
            c_eq = evaluate_equalities(vars.X, vars.U, vars.Theta)
            c = jnp.concatenate([c_dyn.flatten(), node_edge_flatten(c_eq)])

            g = evaluate_inequalities(vars.X, vars.U, vars.Theta)
            gps = node_edge_flatten(
                node_edge_map(lambda value, slack: value + slack, g, vars.S)
            )

            avg_η = jnp.mean(
                jnp.concatenate(
                    [
                        params.η_dyn.flatten(),
                        node_edge_flatten(params.η_eq),
                        node_edge_flatten(params.η_ineq),
                    ]
                )
            )

            if settings.print_ls_logs:
                jax.debug.print(
                    "{:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10}".format(  # noqa: E501
                        "iteration",
                        "α",
                        "cost",
                        "|c|",
                        "|g+s|",
                        "merit",
                        "dmerit/dα",
                        "dmerit/dαdX",
                        "dmerit/dαdS",
                        "|dx|+|du|",
                        "|ds|",
                        "|dy|",
                        "|dz|",
                        "|dθ|",
                        "avg(η)",
                        "µ",
                        "δH",
                        "δH it",
                        "linsys_res",
                    ),
                    ordered=True,
                )

            jax.debug.print(
                "{:^+10} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10} {:^+10.4g}",  # noqa: E501
                iteration,
                α,
                evaluate_cost(vars.X, vars.U, vars.Theta),
                jnp.linalg.norm(c),
                jnp.linalg.norm(gps),
                baseline_merit,
                merit_grad,
                merit_grad_x,
                merit_grad_s,
                jnp.linalg.norm(deltas.X) + jnp.linalg.norm(deltas.U),
                jnp.linalg.norm(node_edge_flatten(deltas.S)),
                jnp.linalg.norm(deltas.Y_dyn)
                + jnp.linalg.norm(node_edge_flatten(deltas.Y_eq)),
                jnp.linalg.norm(node_edge_flatten(deltas.Z)),
                jnp.linalg.norm(deltas.Theta),
                avg_η,
                params.µ,
                hessian_regularization_used,
                regularization_attempts,
                jnp.linalg.norm(residual_vec),
                ordered=True,
            )

        updated_vars = TreeVariables(
            X=vars.X + α * deltas.X,
            U=vars.U + α * deltas.U,
            S=node_edge_map(
                lambda value, delta: jnp.maximum(value + α * delta, (1.0 - τ) * value),
                vars.S,
                deltas.S,
            ),
            Y_dyn=vars.Y_dyn + α * deltas.Y_dyn,
            Y_eq=node_edge_map(
                lambda value, delta: value + α * delta, vars.Y_eq, deltas.Y_eq
            ),
            Z=node_edge_map(
                lambda value, delta: jnp.maximum(value + α * delta, (1.0 - τ) * value),
                vars.Z,
                deltas.Z,
            ),
            Theta=vars.Theta + α * deltas.Theta,
        )

        # Safety net: revert if the step produced any non-finite value
        # (NaN/Inf), and force-exit the main loop on the next iteration (by
        # jumping iteration past max_iterations). This prevents propagation
        # when the KKT factorization or line search misbehaves on
        # ill-conditioned problems. The returned `no_errors` will be False.
        nan_in_update = jnp.logical_or(
            jnp.logical_not(regularization_valid),
            jnp.logical_not(
                jnp.all(jnp.isfinite(updated_vars.X))
                & jnp.all(jnp.isfinite(updated_vars.U))
                & jnp.all(jnp.isfinite(updated_vars.Y_dyn))
                & jnp.all(jnp.isfinite(updated_vars.Theta))
                & tree_all_finite(updated_vars.S)
                & tree_all_finite(updated_vars.Y_eq)
                & tree_all_finite(updated_vars.Z)
            ),
        )
        updated_vars = jax.tree_util.tree_map(
            lambda new, old: jnp.where(nan_in_update, old, new),
            updated_vars,
            vars,
        )

        new_residual = build_kkt_rhs(
            node_cost=node_cost,
            edge_cost=edge_cost,
            dynamics=dynamics,
            node_equalities=node_equalities,
            edge_equalities=edge_equalities,
            node_inequalities=node_inequalities,
            edge_inequalities=edge_inequalities,
            x0=x0,
            vars=updated_vars,
            params=params,
            topology=topology,
            locations=locations,
        )

        improved_dyn = jnp.abs(
            new_residual.Y_dyn
        ) < settings.η_improvement_threshold * jnp.abs(kkt_system.rhs.Y_dyn)
        η_dyn_new = jnp.where(
            improved_dyn,
            params.η_dyn,
            jnp.minimum(params.η_dyn * settings.η_update_factor, settings.η_max),
        )

        η_eq_new = node_edge_map(
            lambda new, old, eta: jnp.where(
                jnp.abs(new) < settings.η_improvement_threshold * jnp.abs(old),
                eta,
                jnp.minimum(eta * settings.η_update_factor, settings.η_max),
            ),
            new_residual.Y_eq,
            kkt_system.rhs.Y_eq,
            params.η_eq,
        )

        η_ineq_new = node_edge_map(
            lambda new, old, eta: jnp.where(
                jnp.abs(new) < settings.η_improvement_threshold * jnp.abs(old),
                eta,
                jnp.minimum(eta * settings.η_update_factor, settings.η_max),
            ),
            new_residual.Z,
            kkt_system.rhs.Z,
            params.η_ineq,
        )

        residual = jnp.concatenate(
            [
                new_residual.X.flatten(),
                new_residual.U.flatten(),
                # Note: absolutely do NOT just use new_residual.S.flatten() here.
                # Using (vars.S * new_residual.S).flatten() is also suboptimal,
                # but would be a lot closer to being OK.
                node_edge_flatten(
                    node_edge_map(lambda slack, z: slack * z, vars.S, vars.Z)
                ),
                new_residual.Y_dyn.flatten(),
                node_edge_flatten(new_residual.Y_eq),
                node_edge_flatten(new_residual.Z),
                new_residual.Theta.flatten(),
            ]
        )
        µ_new_default = jnp.where(
            jnp.abs(residual).max() > settings.κ * params.µ,
            params.µ,
            jnp.maximum(
                jnp.minimum(params.µ * settings.µ_update_factor, params.µ**1.5),
                settings.µ_min,
            ),
        )

        # Guard: with no inequalities the centering rule has nothing to
        # track (S/Z are empty), so fall back to the default rule.
        num_inequalities = vars.S.node.size + vars.S.edge.size

        def _mehrotra_µ() -> jnp.double:
            """Centering µ update: ``µ_target = σ · mean(S·Z)``.

            Clamped above by ``params.µ`` (non-increasing) and below by
            ``max(µ_update_factor · µ, µ_min)`` so we don't shrink
            faster than the legacy schedule in a single step.

            σ defaults to 0.1; a full predictor-corrector would compute σ
            from an affine fraction-to-boundary solve, which we skip in
            favour of an always-on σ to avoid the extra factorisation.
            """
            sigma = settings.mehrotra_sigma
            sz_mean = jnp.mean(
                node_edge_flatten(
                    node_edge_map(lambda slack, z: slack * z, vars.S, vars.Z)
                )
            )
            µ_target = sigma * sz_mean
            # Upper clamp: don't shrink µ by more than µ_update_factor
            # in a single step. Lower clamp: µ_min.
            lower = jnp.maximum(params.µ * settings.µ_update_factor, settings.µ_min)
            upper = params.µ  # non-increasing
            return jnp.clip(µ_target, lower, upper)

        if settings.mehrotra_mu and num_inequalities > 0:
            µ_new = _mehrotra_µ()
        else:
            µ_new = µ_new_default

        updated_params = TreeParameters(
            µ=µ_new,
            η_dyn=η_dyn_new,
            η_eq=η_eq_new,
            η_ineq=η_ineq_new,
        )

        decreased_hessian_regularization = (
            hessian_regularization_used * hess_reg_settings.decrease_factor
        )
        decreased_hessian_regularization = jnp.where(
            decreased_hessian_regularization < hess_reg_settings.first_positive,
            hess_reg_settings.minimum,
            decreased_hessian_regularization,
        )
        hessian_regularization_new = jnp.where(
            regularization_valid,
            jnp.maximum(hess_reg_settings.minimum, decreased_hessian_regularization),
            hessian_regularization_used,
        )

        kkt_system_new = build_kkt(
            node_cost=node_cost,
            edge_cost=edge_cost,
            dynamics=dynamics,
            node_equalities=node_equalities,
            edge_equalities=edge_equalities,
            node_inequalities=node_inequalities,
            edge_inequalities=edge_inequalities,
            x0=x0,
            vars=updated_vars,
            params=updated_params,
            hessian_regularization=hessian_regularization_new,
            regularize_slack_elimination_with_mu=(
                settings.regularize_slack_elimination_with_mu
            ),
            topology=topology,
            locations=locations,
        )

        # On NaN we already reverted vars; jump iteration past max_iterations
        # so the continuation criterion exits on the next check.
        iteration += jnp.where(nan_in_update, settings.max_iterations + 1, 1)

        # Track cost of new iterate (and its improvement vs previous iterate)
        # for the auxiliary cost-improvement / primal-violation termination.
        # On NaN we kept old vars, so report no improvement.
        new_iter_cost = evaluate_cost(
            updated_vars.X, updated_vars.U, updated_vars.Theta
        )
        new_iter_cost = jnp.where(nan_in_update, prev_iter_cost, new_iter_cost)
        last_improvement = prev_iter_cost - new_iter_cost

        # Tighten the single-envelope filter to the smallest (θ, f) seen
        # so far. On NaN keep the old envelope. Unused when filter mode
        # is off (envelope stays at +inf).
        if settings.use_filter_line_search:
            theta_new, f_new = compute_theta_f(α)
            theta_new = jnp.where(nan_in_update, filter_theta, theta_new)
            f_new = jnp.where(nan_in_update, filter_f, f_new)
            filter_theta_new = jnp.minimum(filter_theta, theta_new)
            filter_f_new = jnp.minimum(filter_f, f_new)
        else:
            filter_theta_new = filter_theta
            filter_f_new = filter_f

        return (
            updated_vars,
            updated_params,
            kkt_system_new,
            iteration,
            new_iter_cost,
            last_improvement,
            filter_theta_new,
            filter_f_new,
            hessian_regularization_new,
        )

    def main_loop_continuation_criteria(
        inputs: tuple[
            TreeVariables,
            TreeParameters,
            KKTSystem,
            jnp.int32,
            jnp.double,
            jnp.double,
            jnp.double,
            jnp.double,
            jnp.double,
        ],
    ) -> jnp.bool:
        (
            vars,
            _unused_params,
            kkt_system,
            iteration,
            _last_iter_cost,
            last_improvement,
            _filter_theta,
            _filter_f,
            _hessian_regularization,
        ) = inputs

        residual = jnp.concatenate(
            [
                kkt_system.rhs.X.flatten(),
                kkt_system.rhs.U.flatten(),
                # Note: absolutely do NOT just use kkt_system.rhs.S.flatten() here.
                # Using (vars.S * kkt_system.rhs.S).flatten() is also suboptimal,
                # but would be a lot closer to being OK.
                node_edge_flatten(
                    node_edge_map(lambda slack, z: slack * z, vars.S, vars.Z)
                ),
                kkt_system.rhs.Y_dyn.flatten(),
                node_edge_flatten(kkt_system.rhs.Y_eq),
                node_edge_flatten(kkt_system.rhs.Z),
                kkt_system.rhs.Theta.flatten(),
            ]
        )
        converged = jnp.sum(jnp.square(residual)) <= settings.residual_sq_threshold

        # Aux-gate primal violation: inf-norm of init/dyn defects,
        # equality residuals, and positive-part inequality residuals.
        # The inequality piece uses ``max(0, g) = max(0, rhs.Z - S)``
        # (recovered without re-evaluating the inequality function) so
        # the aux gate matches the benchmark's success-check exactly.
        ineq_violation = jnp.max(
            jnp.maximum(
                node_edge_flatten(
                    node_edge_map(
                        lambda residual, slack: residual - slack,
                        kkt_system.rhs.Z,
                        vars.S,
                    )
                ),
                0.0,
            ),
            initial=0.0,
        )
        primal_violation = jnp.maximum(
            jnp.max(jnp.abs(kkt_system.rhs.Y_dyn), initial=0.0),
            jnp.maximum(
                jnp.max(jnp.abs(node_edge_flatten(kkt_system.rhs.Y_eq)), initial=0.0),
                ineq_violation,
            ),
        )
        aux_converged = jnp.logical_and(
            last_improvement < settings.cost_improvement_threshold,
            primal_violation < settings.primal_violation_threshold,
        )

        any_converged = jnp.logical_or(converged, aux_converged)
        hit_iteration_limit = iteration >= settings.max_iterations
        return jnp.logical_and(
            jnp.logical_not(any_converged), jnp.logical_not(hit_iteration_limit)
        )

    init_iter_cost = evaluate_cost(vars_current.X, vars_current.U, vars_current.Theta)
    (
        vars_out,
        final_params,
        _unused_kkt_system,
        iteration,
        _final_iter_cost,
        _final_improvement,
        _final_filter_theta,
        _final_filter_f,
        _final_hessian_regularization,
    ) = jax.lax.while_loop(
        main_loop_continuation_criteria,
        main_loop_body,
        (
            vars_current,
            params_current,
            kkt_system,
            0,
            init_iter_cost,
            jnp.array(jnp.inf, dtype=jnp.float64),
            # Filter envelope initialized to +inf so the first iterate is
            # unconditionally acceptable (no prior history to compare to).
            jnp.array(jnp.inf, dtype=jnp.float64),
            jnp.array(jnp.inf, dtype=jnp.float64),
            hessian_regularization_current,
        ),
    )

    no_errors = iteration < settings.max_iterations

    return vars_out, iteration, no_errors, final_params


def solve_tree(
    vars_in: TreeVariables,
    x0: jax.Array,
    dynamics: EdgeFunction,
    settings: SolverSettings,
    *,
    node_cost: NodeCostFunction = _zero_node_cost,
    edge_cost: EdgeCostFunction = _zero_edge_cost,
    node_equalities: NodeFunction = _empty_node_function,
    edge_equalities: EdgeFunction = _empty_edge_function,
    node_inequalities: NodeFunction = _empty_node_function,
    edge_inequalities: EdgeFunction = _empty_edge_function,
    params_in: TreeParameters | None = None,
    topology: TreeOCPTopology | None = None,
    locations: OCPCallbackLocations | None = None,
) -> tuple[TreeVariables, jnp.int32, jnp.bool, TreeParameters]:
    """Solve an OCP with explicit node and edge callbacks and storage.

    ``locations`` independently selects where cost, equality, and inequality
    callbacks are active. Constraint rows in ``vars_in`` and ``params_in``
    follow the corresponding selected-index order. If omitted, every callback
    is evaluated at every node and edge.
    """
    if locations is None:
        locations = _all_callback_locations(vars_in.X.shape[0], vars_in.U.shape[0])
    locations = validate_callback_locations(
        locations,
        num_nodes=vars_in.X.shape[0],
        num_edges=vars_in.U.shape[0],
    )
    return _solve_node_edge(
        vars_in=vars_in,
        x0=x0,
        dynamics=dynamics,
        settings=settings,
        node_cost=node_cost,
        edge_cost=edge_cost,
        node_equalities=node_equalities,
        edge_equalities=edge_equalities,
        node_inequalities=node_inequalities,
        edge_inequalities=edge_inequalities,
        params_in=params_in,
        topology=topology,
        locations=locations,
    )


def _validate_chain_shapes(vars_in: Variables) -> None:
    """Validate the legacy ``T + 1`` chain layout using static shapes."""
    num_stages = vars_in.X.shape[0]
    num_edges = num_stages - 1
    if vars_in.U.shape[0] != num_edges:
        message = f"U must have {num_edges} rows; got {vars_in.U.shape[0]}"
        raise ValueError(message)
    if vars_in.Y_dyn.shape != vars_in.X.shape:
        message = (
            f"Y_dyn must have the same shape as X ({vars_in.X.shape}); "
            f"got {vars_in.Y_dyn.shape}"
        )
        raise ValueError(message)
    for name, value in (
        ("S", vars_in.S),
        ("Y_eq", vars_in.Y_eq),
        ("Z", vars_in.Z),
    ):
        if value.shape[0] != num_stages:
            message = f"{name} must have {num_stages} rows; got {value.shape[0]}"
            raise ValueError(message)
    if vars_in.S.shape != vars_in.Z.shape:
        message = f"S and Z must have identical shapes; got {vars_in.S.shape} and {vars_in.Z.shape}"
        raise ValueError(message)


@partial(
    jax.jit,
    static_argnames=("cost", "dynamics", "equalities", "inequalities"),
)
def solve(
    vars_in: Variables,
    x0: jax.Array,
    cost: CostFunction,
    dynamics: Function,
    settings: SolverSettings,
    equalities: Function = _empty_edge_function,
    inequalities: Function = _empty_edge_function,
    params_in: Parameters | None = None,
) -> tuple[Variables, jnp.int32, jnp.bool, Parameters]:
    """Solve a chain OCP using one callback and value row per local stage.

    The callbacks use the conventional ``(x, u, theta, t)`` signature for
    ``t = 0, ..., T``.  At ``t = T`` the control argument is a zero vector.
    Internally, stages ``0, ..., T - 1`` are edge blocks and stage ``T`` is
    the sole selected node block of :func:`solve_tree`.
    """
    _validate_chain_shapes(vars_in)
    num_edges = vars_in.U.shape[0]
    edge_indices = jnp.arange(num_edges, dtype=jnp.int32)
    terminal_indices = jnp.asarray([num_edges], dtype=jnp.int32)
    chain_locations = NodeAndEdgeIndices(
        node=terminal_indices,
        edge=edge_indices,
    )
    locations = OCPCallbackLocations(
        cost=chain_locations,
        equalities=chain_locations,
        inequalities=chain_locations,
    )
    zero_control = jnp.zeros((vars_in.U.shape[1],), dtype=vars_in.U.dtype)

    def terminal_cost(x, theta, node):  # noqa: ANN001, ANN202
        return cost(x, zero_control, theta, node)

    def terminal_equalities(x, theta, node):  # noqa: ANN001, ANN202
        return equalities(x, zero_control, theta, node)

    def terminal_inequalities(x, theta, node):  # noqa: ANN001, ANN202
        return inequalities(x, zero_control, theta, node)

    tree_vars = TreeVariables(
        X=vars_in.X,
        U=vars_in.U,
        S=NodeAndEdgeValues(node=vars_in.S[-1:], edge=vars_in.S[:-1]),
        Y_dyn=vars_in.Y_dyn,
        Y_eq=NodeAndEdgeValues(node=vars_in.Y_eq[-1:], edge=vars_in.Y_eq[:-1]),
        Z=NodeAndEdgeValues(node=vars_in.Z[-1:], edge=vars_in.Z[:-1]),
        Theta=vars_in.Theta,
    )
    tree_params = None
    if params_in is not None:
        tree_params = TreeParameters(
            µ=params_in.µ,
            η_dyn=params_in.η_dyn,
            η_eq=NodeAndEdgeValues(node=params_in.η_eq[-1:], edge=params_in.η_eq[:-1]),
            η_ineq=NodeAndEdgeValues(
                node=params_in.η_ineq[-1:], edge=params_in.η_ineq[:-1]
            ),
        )

    tree_out, iterations, no_errors, tree_params_out = _solve_node_edge(
        vars_in=tree_vars,
        x0=x0,
        dynamics=dynamics,
        settings=settings,
        node_cost=terminal_cost,
        edge_cost=cost,
        node_equalities=terminal_equalities,
        edge_equalities=equalities,
        node_inequalities=terminal_inequalities,
        edge_inequalities=inequalities,
        params_in=tree_params,
        topology=None,
        locations=locations,
    )

    def flatten_stages(values: NodeAndEdgeValues) -> jax.Array:
        return jnp.concatenate([values.edge, values.node], axis=0)

    vars_out = Variables(
        X=tree_out.X,
        U=tree_out.U,
        S=flatten_stages(tree_out.S),
        Y_dyn=tree_out.Y_dyn,
        Y_eq=flatten_stages(tree_out.Y_eq),
        Z=flatten_stages(tree_out.Z),
        Theta=tree_out.Theta,
    )
    params_out = Parameters(
        µ=tree_params_out.µ,
        η_dyn=tree_params_out.η_dyn,
        η_eq=flatten_stages(tree_params_out.η_eq),
        η_ineq=flatten_stages(tree_params_out.η_ineq),
    )
    return vars_out, iterations, no_errors, params_out
