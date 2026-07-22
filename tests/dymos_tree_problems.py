"""Self-contained JAX ports of branching Dymos example formulations.

The mathematical problem data are derived from the Apache-2.0-licensed Dymos
1.15.1 ports in ``sip_examples`` at commit 768a97f.  This module deliberately
has no import-time or runtime dependency on that repository, Dymos, or CasADi.
"""

# ruff: noqa: ANN001, ANN202

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import numpy as np
from jax import numpy as jnp

from primal_dual_lipa.topology import TreeOCPTopology, make_tree_ocp_topology
from primal_dual_lipa.types import (
    HessianRegularizationSettings,
    NodeAndEdgeIndices,
    NodeAndEdgeValues,
    OCPCallbackLocations,
    SolverSettings,
    TreeVariables,
)

if TYPE_CHECKING:
    from primal_dual_lipa.types import (
        EdgeCostFunction,
        EdgeFunction,
        NodeCostFunction,
        NodeFunction,
    )


@dataclass(frozen=True)
class TreeTestProblem:
    """A complete self-contained input to the rooted-tree LIPA solver."""

    name: str
    topology: TreeOCPTopology
    variables: TreeVariables
    x0: jax.Array
    node_cost: NodeCostFunction
    edge_cost: EdgeCostFunction
    dynamics: EdgeFunction
    node_equalities: NodeFunction
    edge_equalities: EdgeFunction
    node_inequalities: NodeFunction
    edge_inequalities: EdgeFunction
    locations: OCPCallbackLocations
    settings: SolverSettings
    metadata: dict[str, object]


_TRAIN_SOC = jnp.array([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
_AKIMA_C = jnp.array(
    [
        [
            -1.923076923076894,
            -9.945609945610054,
            3.180606060606042,
            2.053333333333388,
            -30.555555555555195,
            -6.250000000000003,
        ],
        [
            1.0256410256410349,
            2.175602175602198,
            -1.376969696969688,
            0.07333333333331704,
            7.638888888888806,
            -0.41666666666662755,
        ],
        [
            0.4166666666666637,
            0.5641025641025638,
            0.5454545454545447,
            0.45333333333333364,
            0.8750000000000023,
            1.1041666666666683,
        ],
        [3.5, 3.55, 3.65, 3.75, 3.9, 4.1],
    ]
)


def _akima_v_oc(soc: jax.Array) -> jax.Array:
    """Evaluate the reference piecewise-cubic open-circuit voltage curve."""
    piece = jnp.searchsorted(_TRAIN_SOC[1:-1], soc, side="left")
    piece = jnp.clip(piece, 0, _AKIMA_C.shape[1] - 1)
    dx = soc - _TRAIN_SOC[piece]
    coefficients = _AKIMA_C[:, piece]
    return (
        coefficients[0] * dx**3
        + coefficients[1] * dx**2
        + coefficients[2] * dx
        + coefficients[3]
    )


def _battery_power_balance(
    soc: jax.Array,
    current: jax.Array,
    num_battery: jax.Array,
    num_motor: jax.Array,
) -> jax.Array:
    resistance = 0.025
    power_out_gearbox = 3.6
    pack_current = num_battery * current
    load_voltage = _akima_v_oc(soc) - current * resistance
    pack_power = pack_current * load_voltage
    efficiency = 0.9 - 0.3 * pack_current / num_motor
    return pack_power - power_out_gearbox / efficiency


def make_battery_multibranch_problem(*, print_logs: bool = True) -> TreeTestProblem:
    """Build the three-way branching battery feasibility problem."""
    parents = np.array(
        [
            -1,
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            6,
            12,
            13,
            14,
            15,
            6,
            17,
            18,
            19,
            20,
        ],
        dtype=np.int32,
    )
    topology = make_tree_ocp_topology(parents, use_parallel_lqr=True)

    X_init = np.zeros((parents.size, 1))
    X_init[1, 0] = 1.0
    phase_specs = (
        (1, range(2, 7), 0.63464982, 3.0, 3.0),
        (6, range(7, 12), 0.23794217, 3.0, 3.0),
        (6, range(12, 17), 0.0281523, 2.0, 3.0),
        (6, range(17, 22), 0.18625395, 3.0, 2.0),
    )
    edge_parameters_by_child = np.ones((parents.size, 2))
    control_guess_by_child = np.zeros(parents.size)
    for start_node, child_range, endpoint, num_battery, num_motor in phase_specs:
        children = list(child_range)
        start_soc = X_init[start_node, 0]
        guesses = np.linspace(start_soc, endpoint, len(children) + 1)
        X_init[children, 0] = guesses[1:]
        current_guess = 1.05 * (start_soc - endpoint)
        edge_parameters_by_child[children] = (num_battery, num_motor)
        control_guess_by_child[children] = current_guess

    edge_children = np.asarray(topology.plan.edge_children)
    edge_parameters = jnp.asarray(edge_parameters_by_child[edge_children])
    edge_is_controlled = jnp.asarray(edge_children != 1)
    controlled_edges = jnp.asarray(
        np.flatnonzero(np.asarray(edge_is_controlled)), dtype=jnp.int32
    )
    empty_indices = jnp.empty((0,), dtype=jnp.int32)
    locations = OCPCallbackLocations(
        cost=NodeAndEdgeIndices(node=empty_indices, edge=empty_indices),
        equalities=NodeAndEdgeIndices(node=empty_indices, edge=controlled_edges),
        inequalities=NodeAndEdgeIndices(node=empty_indices, edge=controlled_edges),
    )
    U_init = jnp.asarray(control_guess_by_child[edge_children, None])

    def dynamics(x, u, theta, edge):
        del theta
        integrated_soc = x[0] - u[0] / 5.25
        return jnp.array([jnp.where(edge_is_controlled[edge], integrated_soc, 1.0)])

    def node_cost(x, theta, node):
        del x, theta, node
        return jnp.array(0.0)

    def edge_cost(x, u, theta, edge):
        del x, u, theta, edge
        return jnp.array(0.0)

    def node_equalities(x, theta, node):
        del x, theta, node
        return jnp.empty(0)

    def edge_equalities(x, u, theta, edge):
        del theta
        num_battery, num_motor = edge_parameters[edge]
        balance = _battery_power_balance(x[0], u[0], num_battery, num_motor)
        return jnp.array([balance])

    def node_inequalities(x, theta, node):
        del x, theta, node
        return jnp.empty(0)

    def edge_inequalities(x, u, theta, edge):
        del x, theta
        del edge
        return jnp.array([-u[0], u[0] - 50.0])

    initial_inequalities = jax.vmap(edge_inequalities)(
        jnp.asarray(X_init)[topology.plan.edge_parents[controlled_edges]],
        U_init[controlled_edges],
        jnp.empty((controlled_edges.shape[0], 0)),
        controlled_edges,
    )
    variables = TreeVariables(
        X=jnp.asarray(X_init),
        U=U_init,
        S=NodeAndEdgeValues(node=jnp.empty((0, 0)), edge=-initial_inequalities),
        Y_dyn=jnp.zeros_like(jnp.asarray(X_init)),
        Y_eq=NodeAndEdgeValues(
            node=jnp.empty((0, 0)),
            edge=jnp.zeros((controlled_edges.shape[0], 1)),
        ),
        Z=NodeAndEdgeValues(
            node=jnp.empty((0, 0)),
            edge=jnp.ones((controlled_edges.shape[0], 2)),
        ),
        Theta=jnp.empty(0),
    )
    settings = SolverSettings(
        max_iterations=200,
        residual_sq_threshold=1e-12,
        η0=10.0,
        η_update_factor=1.5,
        µ0=1e-3,
        use_parallel_lqr=True,
        print_logs=print_logs,
    )
    return TreeTestProblem(
        name="battery_multibranch",
        topology=topology,
        variables=variables,
        x0=jnp.array([0.0]),
        node_cost=node_cost,
        edge_cost=edge_cost,
        dynamics=dynamics,
        node_equalities=node_equalities,
        edge_equalities=edge_equalities,
        node_inequalities=node_inequalities,
        edge_inequalities=edge_inequalities,
        locations=locations,
        settings=settings,
        metadata={
            "branch_node": 6,
            "branch_children": np.array([7, 12, 17], dtype=np.int32),
            "edge_is_controlled": np.asarray(edge_is_controlled),
        },
    )


_FT_TO_M = 0.3048
_KNOT_TO_MPS = 0.5144444444444445
_LBM_TO_KG = 0.45359237
_LBF_TO_N = 4.4482216152605
_DEG_TO_RAD = np.pi / 180.0

_DURATION_BR = 0
_DURATION_RTO = 1
_DURATION_V1 = 2
_DURATION_ROTATE = 3
_DURATION_CLIMB = 4
_THETA_FIELD_LENGTH = 5
_THETA_ALPHA_LINK = 6

_R_REF = 1000.0
_V_REF = 100.0
_H_REF = 1.0
_GAM_REF = 0.05
_ALPHA_REF = 10.0 * _DEG_TO_RAD
_FORCE_REF = 100000.0


def make_balanced_field_problem(  # noqa: C901, PLR0915
    *, print_logs: bool = True
) -> TreeTestProblem:
    """Build the rejected/continued-takeoff branching aircraft problem.

    Five active bounds implied by hard equalities are omitted without changing
    the feasible set because LIPA has no separate native-bound treatment.
    """
    segments = 6

    rho = 1.225
    s_ref = 124.7
    cd0 = 0.03
    cl0 = 0.5
    cl_max = 2.0
    alpha_max = np.radians(10.0)
    h_w = 1.0
    aspect_ratio = 9.45
    oswald_e = 0.801
    span = 35.7
    gravity = 9.80665
    mass = 174200.0 * _LBM_TO_KG
    thrust_nominal = 27000.0 * 2.0 * _LBF_TO_N
    thrust_engine_out = 27000.0 * _LBF_TO_N
    thrust_shutdown = 0.0
    mu_nominal = 0.03
    mu_braking = 0.3

    # Nodes are created in topological order. Per-child edge metadata are then
    # reordered into the contraction plan's canonical edge order below.
    parents = [-1]
    node_is_climb = [False]
    edge_kind_by_child = [-1]
    duration_index_by_child = [-1]
    thrust_by_child = [0.0]
    rolling_mu_by_child = [0.0]
    rotate_index_by_child = [0]
    control_guess_by_child = [0.0]

    def add_edge_node(
        parent: int,
        *,
        climb_node: bool,
        edge_kind: int,
        duration_index: int = 0,
        thrust: float = 0.0,
        rolling_mu: float = 0.0,
        rotate_index: int = 0,
        control_guess: float = 0.0,
    ) -> int:
        child = len(parents)
        parents.append(parent)
        node_is_climb.append(climb_node)
        edge_kind_by_child.append(edge_kind)
        duration_index_by_child.append(duration_index)
        thrust_by_child.append(thrust)
        rolling_mu_by_child.append(rolling_mu)
        rotate_index_by_child.append(rotate_index)
        control_guess_by_child.append(control_guess)
        return child

    def add_runway_phase(
        start: int,
        duration_index: int,
        thrust: float,
        rolling_mu: float,
    ) -> int:
        node = start
        for _ in range(segments):
            node = add_edge_node(
                node,
                climb_node=False,
                edge_kind=0,
                duration_index=duration_index,
                thrust=thrust,
                rolling_mu=rolling_mu,
            )
        return node

    br_initial = 0
    br_final = add_runway_phase(br_initial, _DURATION_BR, thrust_nominal, mu_nominal)
    rto_final = add_runway_phase(br_final, _DURATION_RTO, thrust_shutdown, mu_braking)
    v1_final = add_runway_phase(br_final, _DURATION_V1, thrust_engine_out, mu_nominal)
    rotate_start = add_edge_node(
        v1_final,
        climb_node=False,
        edge_kind=1,
    )
    rotate_node = rotate_start
    for rotate_index in range(segments):
        rotate_node = add_edge_node(
            rotate_node,
            climb_node=False,
            edge_kind=2,
            duration_index=_DURATION_ROTATE,
            rotate_index=rotate_index,
        )
    rotate_terminal = rotate_node
    climb_first = add_edge_node(
        rotate_terminal,
        climb_node=True,
        edge_kind=3,
    )
    climb_controls = np.array(
        [0.09735855, 0.07399866, 0.13706639, 0.15410363, 0.14712257, 0.10481729]
    )
    climb_node = climb_first
    for control_guess in climb_controls:
        climb_node = add_edge_node(
            climb_node,
            climb_node=True,
            edge_kind=4,
            duration_index=_DURATION_CLIMB,
            control_guess=float(control_guess),
        )
    climb_final = climb_node

    parents_array = np.asarray(parents, dtype=np.int32)
    topology = make_tree_ocp_topology(parents_array, use_parallel_lqr=True)
    edge_children_host = np.asarray(topology.plan.edge_children)
    edge_kind = jnp.asarray(np.asarray(edge_kind_by_child)[edge_children_host])
    duration_index = jnp.asarray(
        np.asarray(duration_index_by_child)[edge_children_host]
    )
    edge_thrust = jnp.asarray(np.asarray(thrust_by_child)[edge_children_host])
    edge_rolling_mu = jnp.asarray(np.asarray(rolling_mu_by_child)[edge_children_host])
    rotate_index = jnp.asarray(np.asarray(rotate_index_by_child)[edge_children_host])
    U_init = jnp.asarray(np.asarray(control_guess_by_child)[edge_children_host, None])

    def aero_runway(x, alpha, thrust, rolling_mu):
        velocity = x[1]
        weight = mass * gravity
        lift_coefficient = cl0 + (alpha / alpha_max) * (cl_max - cl0)
        k_nominal = 1.0 / (np.pi * aspect_ratio * oswald_e)
        half_span = span / 2.0
        height_ratio = h_w / half_span
        ground_factor = height_ratio * jnp.sqrt(height_ratio)
        induced_factor = k_nominal * 33.0 * ground_factor / (1.0 + 33.0 * ground_factor)
        dynamic_pressure = 0.5 * rho * velocity**2
        lift = dynamic_pressure * s_ref * lift_coefficient
        drag = dynamic_pressure * s_ref * (cd0 + induced_factor * lift_coefficient**2)
        cos_alpha = jnp.cos(alpha)
        sin_alpha = jnp.sin(alpha)
        normal_force = mass * gravity - lift * cos_alpha - thrust * sin_alpha
        velocity_dot = (thrust * cos_alpha - drag - normal_force * rolling_mu) / mass
        ode = jnp.array([velocity, velocity_dot, 0.0, 0.0])
        stall_ratio = velocity / jnp.sqrt(2.0 * weight / rho / s_ref / cl_max)
        return ode, normal_force, stall_ratio

    def aero_climb(x, alpha, thrust):
        _, height, velocity, gamma = x
        weight = mass * gravity
        lift_coefficient = cl0 + (alpha / alpha_max) * (cl_max - cl0)
        k_nominal = 1.0 / (np.pi * aspect_ratio * oswald_e)
        half_span = span / 2.0
        height_ratio = (height + h_w) / half_span
        ground_factor = height_ratio * jnp.sqrt(height_ratio)
        induced_factor = k_nominal * 33.0 * ground_factor / (1.0 + 33.0 * ground_factor)
        dynamic_pressure = 0.5 * rho * velocity**2
        lift = dynamic_pressure * s_ref * lift_coefficient
        drag = dynamic_pressure * s_ref * (cd0 + induced_factor * lift_coefficient**2)
        cos_alpha = jnp.cos(alpha)
        sin_alpha = jnp.sin(alpha)
        cos_gamma = jnp.cos(gamma)
        sin_gamma = jnp.sin(gamma)
        velocity_dot = (thrust * cos_alpha - drag) / mass - gravity * sin_gamma
        gamma_dot = (thrust * sin_alpha + lift) / (mass * velocity) - (
            gravity / velocity
        ) * cos_gamma
        ode = jnp.array(
            [velocity * cos_gamma, velocity * sin_gamma, velocity_dot, gamma_dot]
        )
        stall_ratio = velocity / jnp.sqrt(2.0 * weight / rho / s_ref / cl_max)
        return ode, stall_ratio

    def rk4_fixed(ode, x, h):
        k1 = ode(x)
        k2 = ode(x + 0.5 * h * k1)
        k3 = ode(x + 0.5 * h * k2)
        k4 = ode(x + h * k3)
        return x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def runway_dynamics(x, u, theta, edge):
        del u
        h = theta[duration_index[edge]] / segments
        thrust = edge_thrust[edge]
        rolling_mu = edge_rolling_mu[edge]
        return rk4_fixed(
            lambda state: aero_runway(state, 0.0, thrust, rolling_mu)[0], x, h
        )

    def link_runway_dynamics(x, u, theta, edge):
        del u, theta, edge
        return x

    def rotate_dynamics(x, u, theta, edge):
        del u
        h = theta[_DURATION_ROTATE] / segments
        index = rotate_index[edge]
        alpha0 = theta[_THETA_ALPHA_LINK] * index / segments
        alpha_mid = theta[_THETA_ALPHA_LINK] * (index + 0.5) / segments
        alpha1 = theta[_THETA_ALPHA_LINK] * (index + 1.0) / segments
        k1 = aero_runway(x, alpha0, thrust_engine_out, mu_nominal)[0]
        k2 = aero_runway(x + 0.5 * h * k1, alpha_mid, thrust_engine_out, mu_nominal)[0]
        k3 = aero_runway(x + 0.5 * h * k2, alpha_mid, thrust_engine_out, mu_nominal)[0]
        k4 = aero_runway(x + h * k3, alpha1, thrust_engine_out, mu_nominal)[0]
        return x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def link_climb_dynamics(x, u, theta, edge):
        del u, theta, edge
        return jnp.array([x[0], 0.0, x[1], 0.0])

    def climb_dynamics(x, u, theta, edge):
        del edge
        h = theta[_DURATION_CLIMB] / segments
        return rk4_fixed(
            lambda state: aero_climb(state, u[0], thrust_engine_out)[0], x, h
        )

    dynamics_branches = (
        runway_dynamics,
        link_runway_dynamics,
        rotate_dynamics,
        link_climb_dynamics,
        climb_dynamics,
    )

    def dynamics(x, u, theta, edge):
        return jax.lax.switch(edge_kind[edge], dynamics_branches, x, u, theta, edge)

    num_nodes = len(parents)
    num_edges = topology.num_edges
    empty_indices = jnp.empty((0,), dtype=jnp.int32)
    cost_nodes = jnp.asarray([rto_final], dtype=jnp.int32)
    equality_nodes = jnp.asarray(
        [rto_final, rotate_terminal, climb_final], dtype=jnp.int32
    )
    equality_edges = jnp.asarray(
        np.flatnonzero(np.asarray(topology.plan.edge_parents) == climb_first),
        dtype=jnp.int32,
    )
    inequality_nodes = jnp.arange(num_nodes, dtype=jnp.int32)
    inequality_edges = jnp.asarray(
        np.flatnonzero(np.asarray(edge_kind) == 4), dtype=jnp.int32
    )
    locations = OCPCallbackLocations(
        cost=NodeAndEdgeIndices(node=cost_nodes, edge=empty_indices),
        equalities=NodeAndEdgeIndices(
            node=equality_nodes,
            edge=equality_edges,
        ),
        inequalities=NodeAndEdgeIndices(
            node=inequality_nodes,
            edge=inequality_edges,
        ),
    )

    equality_kind_by_node = np.zeros(num_nodes, dtype=np.int32)
    equality_kind_by_node[rto_final] = 1
    equality_kind_by_node[rotate_terminal] = 2
    equality_kind_by_node[climb_final] = 3
    equality_kind = jnp.asarray(equality_kind_by_node)

    def no_equalities(x, theta, node):
        del x, theta, node
        return jnp.zeros(3)

    def rto_equalities(x, theta, node):
        del node
        return jnp.array(
            [
                x[1] / _V_REF,
                (x[0] - theta[_THETA_FIELD_LENGTH]) / _R_REF,
                0.0,
            ]
        )

    def rotate_equalities(x, theta, node):
        del node
        normal_force = aero_runway(
            x, theta[_THETA_ALPHA_LINK], thrust_engine_out, mu_nominal
        )[1]
        return jnp.array([normal_force / _FORCE_REF, 0.0, 0.0])

    def climb_final_equalities(x, theta, node):
        del node
        target_height = 35.0 * _FT_TO_M
        target_gamma = 5.0 * _DEG_TO_RAD
        return jnp.array(
            [
                (x[1] - target_height) / target_height,
                (x[3] - target_gamma) / target_gamma,
                (x[0] - theta[_THETA_FIELD_LENGTH]) / _R_REF,
            ]
        )

    equality_branches = (
        no_equalities,
        rto_equalities,
        rotate_equalities,
        climb_final_equalities,
    )

    def node_equalities(x, theta, node):
        return jax.lax.switch(equality_kind[node], equality_branches, x, theta, node)

    def edge_equalities(x, u, theta, edge):
        del x, edge
        value = (u[0] - theta[_THETA_ALPHA_LINK]) / _ALPHA_REF
        return jnp.array([value])

    inequality_kind_by_node = np.ones(num_nodes, dtype=np.int32)
    inequality_kind_by_node[br_initial] = 2
    inequality_kind_by_node[v1_final] = 3
    node_is_climb_host = np.asarray(node_is_climb)
    inequality_kind_by_node[node_is_climb_host] = 4
    inequality_kind_by_node[climb_final] = 5
    inequality_kind_by_node[rto_final] = 6
    inequality_kind_by_node[climb_first] = 7
    inequality_kind = jnp.asarray(inequality_kind_by_node)

    def inactive_inequalities(x, theta, node):
        del x, theta, node
        return -jnp.ones(14)

    def runway_inequalities(x, theta, node):
        del theta, node
        result = -jnp.ones(14)
        return result.at[:2].set(jnp.array([-x[0] / _R_REF, -x[1] / _V_REF]))

    def root_inequalities(x, theta, node):
        del node
        result = -jnp.ones(14)
        # The root range is fixed to zero by x0, so its duplicate active lower
        # bound is omitted to preserve LICQ. The velocity bound remains.
        result = result.at[1].set(-x[1] / _V_REF)
        duration_bounds = (
            (_DURATION_BR, 1.0, 1000.0, 10.0),
            (_DURATION_RTO, 1.0, 1000.0, 1.0),
            (_DURATION_V1, 1.0, 1000.0, 1.0),
            (_DURATION_ROTATE, 1.0, 5.0, 1.0),
            (_DURATION_CLIMB, 1.0, 100.0, 1.0),
        )
        for bound, (index, lower, upper, reference) in enumerate(duration_bounds):
            offset = 2 + 2 * bound
            result = result.at[offset].set((lower - theta[index]) / reference)
            result = result.at[offset + 1].set((theta[index] - upper) / reference)
        result = result.at[12].set(-theta[_THETA_ALPHA_LINK] / _ALPHA_REF)
        return result.at[13].set(
            (theta[_THETA_ALPHA_LINK] - 10.0 * _DEG_TO_RAD) / _ALPHA_REF
        )

    def v1_inequalities(x, theta, node):
        del theta, node
        result = runway_inequalities(x, jnp.zeros(7), 0)
        stall_ratio = aero_runway(x, 0.0, thrust_engine_out, mu_nominal)[2]
        return result.at[2].set((1.2 - stall_ratio) / 100.0)

    def climb_inequalities(x, theta, node):
        del theta, node
        result = -jnp.ones(14)
        result = result.at[:4].set(
            jnp.array(
                [
                    -x[0] / _R_REF,
                    -x[1] / _H_REF,
                    -x[2] / _V_REF,
                    -x[3] / _GAM_REF,
                ]
            )
        )
        return result.at[6].set((x[3] - 5.0 * _DEG_TO_RAD) / (5.0 * _DEG_TO_RAD))

    def climb_terminal_inequalities(x, theta, node):
        del node
        result = -jnp.ones(14)
        state_bounds = jnp.array(
            [
                -x[0] / _R_REF,
                -x[1] / _H_REF,
                -x[2] / _V_REF,
                -x[3] / _GAM_REF,
            ]
        )
        result = result.at[:4].set(state_bounds)
        stall_ratio = aero_climb(x, theta[_THETA_ALPHA_LINK], thrust_engine_out)[1]
        return result.at[7].set((1.25 - stall_ratio) / 1.25)

    def rto_terminal_inequalities(x, theta, node):
        del theta, node
        result = -jnp.ones(14)
        # v == 0 is already an endpoint equality; retaining v >= 0 creates
        # linearly dependent active constraints and unbounded cancelling duals.
        return result.at[0].set(-x[0] / _R_REF)

    def climb_initial_inequalities(x, theta, node):
        result = climb_inequalities(x, theta, node)
        # The runway-to-climb link fixes h == 0 and gamma == 0 exactly.
        result = result.at[1].set(-1.0)
        return result.at[3].set(-1.0)

    inequality_branches = (
        inactive_inequalities,
        runway_inequalities,
        root_inequalities,
        v1_inequalities,
        climb_inequalities,
        climb_terminal_inequalities,
        rto_terminal_inequalities,
        climb_initial_inequalities,
    )

    def node_inequalities(x, theta, node):
        return jax.lax.switch(
            inequality_kind[node], inequality_branches, x, theta, node
        )

    def edge_inequalities(x, u, theta, edge):
        del x, theta, edge
        return jnp.array(
            [
                (-10.0 * _DEG_TO_RAD - u[0]) / _ALPHA_REF,
                (u[0] - 15.0 * _DEG_TO_RAD) / _ALPHA_REF,
            ]
        )

    def node_cost(x, theta, node):
        del theta, node
        return x[0] / _R_REF

    def edge_cost(x, u, theta, edge):
        del x, u, theta, edge
        return jnp.array(0.0)

    theta_init = jnp.array(
        [
            28.12256353,
            28.15292765,
            8.29989208,
            1.06218046,
            4.02118536,
            2197.0,
            0.09735854740699715,
        ]
    )
    edge_index_by_child = np.full(num_nodes, -1, dtype=np.int32)
    edge_index_by_child[edge_children_host] = np.arange(num_edges)
    X_init = (
        jnp.zeros((num_nodes, 4)).at[br_initial].set(jnp.array([0.0, 1.0e-4, 0.0, 0.0]))
    )

    def rollout_body(child, states):
        edge = jnp.asarray(edge_index_by_child)[child]
        parent = jnp.asarray(parents_array)[child]
        next_state = dynamics(states[parent], U_init[edge], theta_init, edge)
        return states.at[child].set(next_state)

    X_init = jax.lax.fori_loop(1, num_nodes, rollout_body, X_init)
    initial_node_inequalities = jax.vmap(node_inequalities)(
        X_init[inequality_nodes],
        jnp.broadcast_to(theta_init, (inequality_nodes.shape[0], theta_init.shape[0])),
        inequality_nodes,
    )
    initial_edge_inequalities = jax.vmap(edge_inequalities)(
        X_init[topology.plan.edge_parents[inequality_edges]],
        U_init[inequality_edges],
        jnp.broadcast_to(theta_init, (inequality_edges.shape[0], theta_init.shape[0])),
        inequality_edges,
    )
    variables = TreeVariables(
        X=X_init,
        U=U_init,
        S=NodeAndEdgeValues(
            node=jnp.maximum(-initial_node_inequalities, 1e-2),
            edge=jnp.maximum(-initial_edge_inequalities, 1e-2),
        ),
        Y_dyn=jnp.zeros_like(X_init),
        Y_eq=NodeAndEdgeValues(
            node=jnp.zeros((equality_nodes.shape[0], 3)),
            edge=jnp.zeros((equality_edges.shape[0], 1)),
        ),
        Z=NodeAndEdgeValues(
            node=jnp.ones((inequality_nodes.shape[0], 14)),
            edge=jnp.ones((inequality_edges.shape[0], 2)),
        ),
        Theta=theta_init,
    )
    settings = SolverSettings(
        max_iterations=500,
        residual_sq_threshold=1e-12,
        η0=100.0,
        η_update_factor=1.5,
        η_max=1e8,
        µ0=1e-3,
        mehrotra_mu=True,
        use_filter_line_search=True,
        use_parallel_lqr=True,
        num_iterative_refinement_steps=1,
        print_logs=print_logs,
        hessian_regularization=HessianRegularizationSettings(maximum=1e12),
    )
    return TreeTestProblem(
        name="balanced_field",
        topology=topology,
        variables=variables,
        x0=jnp.array([0.0, 1.0e-4, 0.0, 0.0]),
        node_cost=node_cost,
        edge_cost=edge_cost,
        dynamics=dynamics,
        node_equalities=node_equalities,
        edge_equalities=edge_equalities,
        node_inequalities=node_inequalities,
        edge_inequalities=edge_inequalities,
        locations=locations,
        settings=settings,
        metadata={
            "branch_node": br_final,
            "branch_children": np.array(
                [br_final + 1, v1_final - segments + 1], dtype=np.int32
            ),
            "rto_final": rto_final,
            "v1_final": v1_final,
            "rotate_terminal": rotate_terminal,
            "climb_first": climb_first,
            "climb_final": climb_final,
            "node_is_climb": node_is_climb_host,
        },
    )
