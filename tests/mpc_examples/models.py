"""Whole-body / kinodynamic dynamics for MJX MPC tasks.

Provenance: subset ported from ``mpx/utils/models.py`` —
https://github.com/iit-DLSLab/mpx (BSD-3-Clause).

Covers the two dynamics flavors used by the ported configs:

* ``quadruped_wb_dynamics`` — full whole-body forward dynamics with
  Baumgarte-stabilized rigid contact constraints (Aliengo).
* ``h1_kinodynamic_dynamics`` — kinodynamic model that treats joint
  velocities as commands and grfs as contact forces (H1).

Variants not used by the ported tasks (SRBD, talos_wb,
quadruped_wb_dynamics_explicit_contact, learned-contact-model) are
omitted.
"""

import jax
from jax import numpy as jnp
from mujoco import mjx
from mujoco.mjx._src import math


def _mask_contact_forces(grf, contact):
    return (grf.reshape(-1, 3) * contact[:, None]).reshape(-1)


def _h1_contact_kinematics(mjx_model, mjx_data, contact_id, body_id):
    fl = mjx_data.geom_xpos[contact_id[0]]
    rl = mjx_data.geom_xpos[contact_id[1]]
    fr = mjx_data.geom_xpos[contact_id[2]]
    rr = mjx_data.geom_xpos[contact_id[3]]

    j_fl, _ = mjx.jac(mjx_model, mjx_data, fl, body_id[0])
    j_rl, _ = mjx.jac(mjx_model, mjx_data, rl, body_id[0])
    j_fr, _ = mjx.jac(mjx_model, mjx_data, fr, body_id[1])
    j_rr, _ = mjx.jac(mjx_model, mjx_data, rr, body_id[1])

    feet = jnp.concatenate([fl, rl, fr, rr], axis=0)
    jacobian = jnp.concatenate([j_fl, j_rl, j_fr, j_rr], axis=1)
    return feet, jacobian


def quadruped_wb_dynamics(
    model, mjx_model, contact_id, body_id, n_joints, dt, x, u, t, parameter
):
    """Whole-body dynamics for a quadruped via MJX forward dyn + Baumgarte-stabilized rigid contact."""
    mjx_data = mjx.make_data(model)
    mjx_data = mjx_data.replace(
        qpos=x[: n_joints + 7], qvel=x[n_joints + 7 : 2 * n_joints + 13]
    )

    mjx_data = mjx.fwd_position(mjx_model, mjx_data)
    mjx_data = mjx.fwd_velocity(mjx_model, mjx_data)

    M = mjx_data.qLD
    D = mjx_data.qfrc_bias

    contact = parameter[t, :4]

    tau = jnp.concatenate([jnp.zeros(6), u])

    FL_leg = mjx_data.geom_xpos[contact_id[0]]
    FR_leg = mjx_data.geom_xpos[contact_id[1]]
    RL_leg = mjx_data.geom_xpos[contact_id[2]]
    RR_leg = mjx_data.geom_xpos[contact_id[3]]

    J_FL, _ = mjx.jac(mjx_model, mjx_data, FL_leg, body_id[0])
    J_FR, _ = mjx.jac(mjx_model, mjx_data, FR_leg, body_id[1])
    J_RL, _ = mjx.jac(mjx_model, mjx_data, RL_leg, body_id[2])
    J_RR, _ = mjx.jac(mjx_model, mjx_data, RR_leg, body_id[3])

    J = jnp.concatenate([J_FL, J_FR, J_RL, J_RR], axis=1)
    current_leg = jnp.concatenate([FL_leg, FR_leg, RL_leg, RR_leg], axis=0)
    alpha = 25
    g_dot = J.T @ x[n_joints + 7 : 13 + 2 * n_joints]
    baumgarte_term = -2 * alpha * g_dot

    JT_M_invJ = J.T @ jax.scipy.linalg.cho_solve((M, False), J)
    rhs = -J.T @ jax.scipy.linalg.cho_solve((M, False), tau - D) + baumgarte_term
    cho_JT_M_invJ = jax.scipy.linalg.cho_factor(JT_M_invJ)
    grf = jax.scipy.linalg.cho_solve(cho_JT_M_invJ, rhs)
    grf = jnp.concatenate(
        [
            grf[:3] * contact[0],
            grf[3:6] * contact[1],
            grf[6:9] * contact[2],
            grf[9:12] * contact[3],
        ]
    )

    v = (
        x[n_joints + 7 : 13 + 2 * n_joints]
        + jax.scipy.linalg.cho_solve((M, False), tau - D + J @ grf) * dt
    )

    p = x[:3] + v[:3] * dt
    quat = math.quat_integrate(x[3:7], v[3:6], dt)
    q = x[7 : 7 + n_joints] + v[6 : 6 + n_joints] * dt
    return jnp.concatenate([p, quat, q, v, current_leg, grf])


def h1_kinodynamic_dynamics(
    model, mjx_model, contact_id, body_id, n_joints, dt, x, u, t, parameter
):
    """Kinodynamic model: joint velocities are commands; ground reaction forces are control inputs."""
    qpos = x[: n_joints + 7]
    qvel = x[n_joints + 7 : 2 * n_joints + 13]
    dq = x[13 + n_joints : 13 + 2 * n_joints]
    dq_next = u[:n_joints]
    contact = parameter[t, :4]
    grf = _mask_contact_forces(u[n_joints:], contact)

    mjx_data = mjx.make_data(model)
    mjx_data = mjx_data.replace(qpos=qpos, qvel=qvel)
    mjx_data = mjx.fwd_position(mjx_model, mjx_data)
    mjx_data = mjx.fwd_velocity(mjx_model, mjx_data)

    mass_matrix = mjx.full_m(mjx_model, mjx_data)
    bias = mjx_data.qfrc_bias
    feet_next, jacobian = _h1_contact_kinematics(
        mjx_model, mjx_data, contact_id, body_id
    )

    qdd_joints = (dq_next - dq) / dt
    rhs = (jacobian @ grf)[:6] - bias[:6] - mass_matrix[:6, 6:] @ qdd_joints
    qdd_base = jnp.linalg.solve(mass_matrix[:6, :6] + 1e-6 * jnp.eye(6), rhs)

    base_velocity_next = qvel[:6] + qdd_base * dt
    qvel_next = jnp.concatenate([base_velocity_next, dq_next])

    p_next = x[:3] + qvel_next[:3] * dt
    quat_next = math.quat_integrate(x[3:7], qvel_next[3:6], dt)
    q_next = x[7 : 7 + n_joints] + dq_next * dt

    return jnp.concatenate([p_next, quat_next, q_next, qvel_next, feet_next])
