"""Shared Aliengo whole-body MPC config for tasks that inherit from it.

Provenance: mirrors ``mpx/config/config_aliengo.py`` —
https://github.com/iit-DLSLab/mpx (BSD-3-Clause).

Currently inherited only by ``config_aliengo_trot.py``. The barrel-roll
config does NOT inherit from this — it defines its own Q matrices, dt,
N, etc. inline; see ``config_barrel_roll.py``.
"""

import os
from functools import partial

import jax
import jax.numpy as jnp

from tests.mpc_examples import models, objectives

dir_path = os.path.dirname(os.path.realpath(__file__))
# Bare robot for dynamics (matches mpx). Including the scene's floor
# adds a contact pair to the MJX model that perturbs the mass matrix /
# Baumgarte residual of the rigid-contact constraint.
model_path = os.path.abspath(
    os.path.join(dir_path, "..", "data", "aliengo", "aliengo.xml")
)
# Renderable scene with floor + lighting.
scene_path = os.path.abspath(
    os.path.join(dir_path, "..", "data", "aliengo", "scene_flat.xml")
)

contact_frame = ["FL", "FR", "RL", "RR"]
body_name = ["FL_calf", "FR_calf", "RL_calf", "RR_calf"]

dt = 0.02
N = 25
mpc_frequency = 50

timer_t = jnp.array([0.5, 0.0, 0.0, 0.5])
duty_factor = 0.65
step_freq = 1.35
step_height = 0.2
initial_height = 0.1
robot_height = 0.36
grf_as_state = True

p0 = jnp.array([0, 0, robot_height])
quat0 = jnp.array([1, 0, 0, 0])
q0 = jnp.array([0.2, 0.8, -1.8, -0.2, 0.8, -1.8, 0.2, 0.8, -1.8, -0.2, 0.8, -1.8])
q0_init = jnp.array(
    [-0.2, 0.8, -1.8, -0.2, 0.8, -1.8, -0.2, 0.8, -1.8, -0.2, 0.8, -1.8]
)

p_legs0 = jnp.array(
    [
        0.27092872,
        0.193,
        0.0,
        0.27092872,
        -0.193,
        0.0,
        -0.20887128,
        0.193,
        0.0,
        -0.20887128,
        -0.193,
        0.0,
    ]
)

n_joints = 12
n_contact = len(contact_frame)
n = 13 + 2 * n_joints + 6 * n_contact
m = n_joints
u_ref = jnp.zeros(m)

Qp = jnp.diag(jnp.array([0, 0, 1e4]))
Qrot = jnp.diag(jnp.array([1000, 1000, 0]))
Qq = jnp.diag(jnp.ones(n_joints)) * 1e-1
Qdp = jnp.diag(jnp.array([1, 1, 1])) * 5e3
Qomega = jnp.diag(jnp.array([1, 1, 1])) * 1e2
Qdq = jnp.diag(jnp.ones(n_joints)) * 1e-1
Qtau = jnp.diag(jnp.ones(n_joints)) * 1e-1
Q_grf = jnp.diag(jnp.ones(3 * n_contact)) * 1e-2
Qleg = jnp.diag(jnp.tile(jnp.array([1e4, 1e4, 1e5]), n_contact))
W = jax.scipy.linalg.block_diag(Qp, Qrot, Qq, Qdp, Qomega, Qdq, Qleg, Qtau, Q_grf)

use_terrain_estimation = True

_state_extra = n - (13 + 2 * n_joints + 3 * n_contact)
initial_state = jnp.concatenate(
    [p0, quat0, q0, jnp.zeros(6 + n_joints), p_legs0, jnp.zeros(_state_extra)]
)

cost = partial(objectives.quadruped_wb_obj, True, n_joints, n_contact, N)


def dynamics(model, mjx_model, contact_id, body_id):
    return partial(
        models.quadruped_wb_dynamics,
        model,
        mjx_model,
        contact_id,
        body_id,
        n_joints,
        dt,
    )


max_torque = 35
min_torque = -35
