"""Cartpole goal-reaching problem (mirror of tests/test_cartpole.py).

JAX dynamics + CasADi mirror so the IPOPT adapter can build a fully
symbolic NLP. Same numerical formulation as the LIPA unit test:

* T = 50, dt = 0.1, Euler integration
* state x = (cart_pos, pole_angle, cart_vel, pole_vel)
* control u = (cart_force,)
* terminal equality: x_T = goal = [0, pi, 0, 0]
* control inequality: -5 <= u_t <= 5
* cost: 0.1*||x-goal||^2 + 0.01*u^2 stagewise; 1000*||x-goal||^2 terminal
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np

from primal_dual_lipa.types import SolverSettings

from tests.comparison.problem_spec import ProblemSpec

T = 50
N_STATE = 4
N_CTRL = 1
DT = 0.1
GOAL = jnp.array([0.0, jnp.pi, 0.0, 0.0])
DYN_PARAMS = jnp.array([10.0, 1.0, 0.5])  # (mc, mp, l)


def _ode(state: jax.Array, action: jax.Array, params: jax.Array) -> jax.Array:
    mc, mp, l = params  # noqa: E741
    g = 9.81
    q = state[0:2]
    qd = state[2:]
    s = jnp.sin(q[1])
    c = jnp.cos(q[1])
    H = jnp.array([[mc + mp, mp * l * c], [mp * l * c, mp * l * l]])
    C = jnp.array([[0.0, -mp * qd[1] * l * s], [0.0, 0.0]])
    G = jnp.array([[0.0], [mp * g * l * s]])
    B = jnp.array([[1.0], [0.0]])
    CqdG = jnp.dot(C, jnp.expand_dims(qd, 1)) + G
    f = jnp.concatenate((qd, jnp.squeeze(-jsp.linalg.solve(H, CqdG, assume_a="pos"))))
    v = jnp.squeeze(jsp.linalg.solve(H, B, assume_a="pos"))
    g_term = jnp.concatenate((jnp.zeros(2), v))
    return f + g_term * action


def _cartpole_dynamics(x, u, theta, t):  # noqa: ARG001
    return x + DT * _ode(x, u, DYN_PARAMS)


def _cartpole_cost(x, u, theta, t):  # noqa: ARG001
    err = x - GOAL
    stage = 0.1 * jnp.dot(err, err) + 0.01 * jnp.dot(u, u)
    final = 1000.0 * jnp.dot(err, err)
    return jnp.where(t == T, final, stage)


def _cartpole_equalities(x, u, theta, t):  # noqa: ARG001
    return jnp.where(t == T, x - GOAL, jnp.zeros_like(x))


def _cartpole_inequalities(x, u, theta, t):  # noqa: ARG001
    # -5 <= u <= 5: written as g(x,u) <= 0
    return jnp.where(
        t == T,
        -jnp.ones(2),
        jnp.array([u[0] - 5.0, -5.0 - u[0]]),
    )


def _casadi_builder(x_sx, u_sx, theta_sx, t):  # noqa: ARG001
    """Mirror the JAX dynamics/cost/constraints in CasADi SX."""
    import casadi as ca

    mc, mp, l = float(DYN_PARAMS[0]), float(DYN_PARAMS[1]), float(DYN_PARAMS[2])
    g = 9.81

    q1, q2 = x_sx[0], x_sx[1]
    qd1, qd2 = x_sx[2], x_sx[3]
    s = ca.sin(q2)
    c = ca.cos(q2)

    H = ca.SX(2, 2)
    H[0, 0] = mc + mp
    H[0, 1] = mp * l * c
    H[1, 0] = mp * l * c
    H[1, 1] = mp * l * l

    C = ca.SX(2, 2)
    C[0, 0] = 0.0
    C[0, 1] = -mp * qd2 * l * s
    C[1, 0] = 0.0
    C[1, 1] = 0.0

    G = ca.SX(2, 1)
    G[0, 0] = 0.0
    G[1, 0] = mp * g * l * s

    B = ca.SX(2, 1)
    B[0, 0] = 1.0
    B[1, 0] = 0.0

    qd = ca.vertcat(qd1, qd2)
    CqdG = ca.mtimes(C, qd) + G
    qdd_drift = ca.solve(H, -CqdG)
    qdd_input = ca.solve(H, B) * u_sx[0]
    qdd = qdd_drift + qdd_input

    xdot = ca.vertcat(qd1, qd2, qdd[0], qdd[1])
    next_x = x_sx + DT * xdot

    # Cost (terminal vs stagewise)
    err = x_sx - ca.DM(np.asarray(GOAL))
    stage_cost = 0.1 * ca.dot(err, err) + 0.01 * ca.dot(u_sx, u_sx)
    terminal_cost = 1000.0 * ca.dot(err, err)
    is_terminal = t == T

    # Equality at t=T: x - goal = 0; nothing earlier.
    eq = err if is_terminal else None

    # Inequalities: control bounds at t < T; nothing at t = T.
    ineq = None if is_terminal else ca.vertcat(u_sx[0] - 5.0, -5.0 - u_sx[0])

    f = terminal_cost if is_terminal else stage_cost

    return {
        "f": f,
        "next_x": next_x,
        "eq": eq,
        "ineq": ineq,
    }


def make_problem() -> ProblemSpec:
    x0 = jnp.array([0.0, 0.2, 0.0, -0.1])
    X_init = jnp.linspace(start=x0, stop=GOAL, num=T + 1)
    U_init = jnp.zeros((T, N_CTRL))
    Theta_init = jnp.empty(0)

    return ProblemSpec(
        name="cartpole",
        T=T,
        n=N_STATE,
        m=N_CTRL,
        theta_dim=0,
        x0=x0,
        cost=jax.jit(_cartpole_cost),
        dynamics=jax.jit(_cartpole_dynamics),
        equalities=jax.jit(_cartpole_equalities),
        inequalities=jax.jit(_cartpole_inequalities),
        eq_dim=N_STATE,  # output dim of equalities() per stage
        ineq_dim=2,
        X_init=X_init,
        U_init=U_init,
        Theta_init=Theta_init,
        metadata={
            "casadi_builder": _casadi_builder,
            "ipopt_settings": {
                "mu_strategy": "adaptive",
                "mu_oracle": "probing",
                "alpha_for_y": "bound-mult",
            },
            "lipa_settings": SolverSettings(
                η0=100.0,
                η_update_factor=1.5,
                η_improvement_threshold=0.5,
                µ0=1.0,
                µ_update_factor=0.65,
                κ=100.0,
                num_iterative_refinement_steps=0,
                skip_line_search=True,
                max_iterations=500,
                print_logs=False,
            ),
            "sip_settings": dict(
                penalty_parameter_increase_factor=1.1,
                min_acceptable_constraint_violation_ratio=0.9,
                max_ls_iterations=5000,
                enable_line_search_failures=False,
                num_iterative_refinement_steps=0,
            ),
            "trajax_settings": dict(
                penalty_init=10.0,
                penalty_update_rate=5.0,
                maxiter_al=80,
            ),
            "fatrop_settings": dict(
                mu_init=0.25,
            ),
        },
    )
