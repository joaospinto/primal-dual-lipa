"""Acrobot goal-reaching problem (mirror of tests/test_acrobot.py).

* T = 50, dt = 0.1, Euler integration
* state x = (theta1, theta2, dtheta1, dtheta2)
* control u = (torque,)
* no constraints (cost-only problem)
* cost: 0.5*0.1*||x-goal||^2 + 0.5*0.01*u^2 stagewise; 0.5*1000*||x-goal||^2 terminal
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from primal_dual_lipa.types import SolverSettings

from tests.comparison.problem_spec import ProblemSpec

T = 50
N_STATE = 4
N_CTRL = 1
DT = 0.1
GOAL = jnp.array([jnp.pi, 0.0, 0.0, 0.0])
DYN_PARAMS = jnp.array([1.0, 1.0, 1.0, 0.5, 0.5, 1.0, 1.0])  # m1,m2,l1,lc1,lc2,I1,I2
COST_PARAMS = jnp.array([1000.0, 0.1, 0.01])  # term, stage_x, stage_u


def _ode(x: jax.Array, u: jax.Array) -> jax.Array:
    m1, m2, l1, lc1, lc2, I1, I2 = DYN_PARAMS
    g = 9.8
    a = u[0]
    th1, th2, dth1, dth2 = x[0], x[1], x[2], x[3]
    d1 = m1 * lc1**2 + m2 * (l1**2 + lc2**2 + 2 * l1 * lc2 * jnp.cos(th2)) + I1 + I2
    d2 = m2 * (lc2**2 + l1 * lc2 * jnp.cos(th2)) + I2
    phi2 = m2 * lc2 * g * jnp.cos(th1 + th2 - jnp.pi / 2.0)
    phi1 = (
        -m2 * l1 * lc2 * dth2**2 * jnp.sin(th2)
        - 2 * m2 * l1 * lc2 * dth2 * dth1 * jnp.sin(th2)
        + (m1 * lc1 + m2 * l1) * g * jnp.cos(th1 - jnp.pi / 2)
        + phi2
    )
    ddth2 = (
        a + d2 / d1 * phi1 - m2 * l1 * lc2 * dth1**2 * jnp.sin(th2) - phi2
    ) / (m2 * lc2**2 + I2 - d2**2 / d1)
    ddth1 = -(d2 * ddth2 + phi1) / d1
    return jnp.array([dth1, dth2, ddth1, ddth2])


def _dynamics(x, u, theta, t):  # noqa: ARG001
    return x + DT * _ode(x, u)


def _cost(x, u, theta, t):  # noqa: ARG001
    delta = x - GOAL
    terminal = 0.5 * COST_PARAMS[0] * jnp.dot(delta, delta)
    stagewise = 0.5 * COST_PARAMS[1] * jnp.dot(delta, delta) + 0.5 * COST_PARAMS[2] * jnp.dot(u, u)
    return jnp.where(t == T, terminal, stagewise)


def _casadi_builder(x_sx, u_sx, theta_sx, t):  # noqa: ARG001
    import casadi as ca

    m1, m2, l1, lc1, lc2, I1, I2 = (float(p) for p in DYN_PARAMS)
    g = 9.8
    a = u_sx[0]
    th1, th2, dth1, dth2 = x_sx[0], x_sx[1], x_sx[2], x_sx[3]
    d1 = m1 * lc1**2 + m2 * (l1**2 + lc2**2 + 2 * l1 * lc2 * ca.cos(th2)) + I1 + I2
    d2 = m2 * (lc2**2 + l1 * lc2 * ca.cos(th2)) + I2
    phi2 = m2 * lc2 * g * ca.cos(th1 + th2 - ca.pi / 2.0)
    phi1 = (
        -m2 * l1 * lc2 * dth2**2 * ca.sin(th2)
        - 2 * m2 * l1 * lc2 * dth2 * dth1 * ca.sin(th2)
        + (m1 * lc1 + m2 * l1) * g * ca.cos(th1 - ca.pi / 2)
        + phi2
    )
    ddth2 = (
        a + d2 / d1 * phi1 - m2 * l1 * lc2 * dth1**2 * ca.sin(th2) - phi2
    ) / (m2 * lc2**2 + I2 - d2**2 / d1)
    ddth1 = -(d2 * ddth2 + phi1) / d1
    xdot = ca.vertcat(dth1, dth2, ddth1, ddth2)
    next_x = x_sx + DT * xdot

    delta = x_sx - ca.DM(np.asarray(GOAL))
    terminal = 0.5 * float(COST_PARAMS[0]) * ca.dot(delta, delta)
    stagewise = (
        0.5 * float(COST_PARAMS[1]) * ca.dot(delta, delta)
        + 0.5 * float(COST_PARAMS[2]) * ca.dot(u_sx, u_sx)
    )
    f = terminal if t == T else stagewise

    return {"f": f, "next_x": next_x, "eq": None, "ineq": None}


def make_problem() -> ProblemSpec:
    x0 = jnp.zeros(N_STATE)
    X_init = jnp.zeros((T + 1, N_STATE))
    U_init = jnp.zeros((T, N_CTRL))
    Theta_init = jnp.empty(0)

    return ProblemSpec(
        name="acrobot",
        T=T,
        n=N_STATE,
        m=N_CTRL,
        theta_dim=0,
        x0=x0,
        cost=jax.jit(_cost),
        dynamics=jax.jit(_dynamics),
        equalities=None,
        inequalities=None,
        eq_dim=0,
        ineq_dim=0,
        X_init=X_init,
        U_init=U_init,
        Theta_init=Theta_init,
        metadata={
            "casadi_builder": _casadi_builder,
            # Unconstrained problem — the IPM line search adds no value
            # (no fraction-to-boundary cost to defend); skip it.
            "lipa_settings": SolverSettings(
                η0=100.0,
                η_update_factor=1.5,
                skip_line_search=True,
                print_logs=False,
            ),
            "sip_settings": dict(
                initial_penalty_parameter=100.0,
                penalty_parameter_increase_factor=1.5,
                skip_line_search=True,
            ),
        },
    )
