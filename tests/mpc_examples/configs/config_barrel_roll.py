"""Barrel-roll Aliengo task config.

Provenance: mirrors ``mpx/config/config_barrel_roll.py`` —
https://github.com/iit-DLSLab/mpx (BSD-3-Clause).

Standalone — does NOT inherit from ``_aliengo_base`` because the
upstream barrel-roll config defines its own dt / N / Q matrices /
initial state inline rather than inheriting from ``config_aliengo.py``.
"""

import os
from functools import partial

import jax
import jax.numpy as jnp

from primal_dual_lipa.types import SolverSettings

from tests.mpc_examples import models, objectives, references

task_name = "barrel_roll"

dir_path = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.abspath(
    os.path.join(dir_path, "..", "data", "aliengo", "aliengo.xml")
)
scene_path = os.path.abspath(
    os.path.join(dir_path, "..", "data", "aliengo", "scene_flat.xml")
)

contact_frame = ["FL", "FR", "RL", "RR"]
body_name = ["FL_calf", "FR_calf", "RL_calf", "RR_calf"]

dt = 0.01
N = 100
mpc_frequency = 50

timer_t = jnp.array([0.5, 0.0, 0.0, 0.5])
duty_factor = 0.65
step_freq = 1.35
step_height = 0.12
initial_height = 0.1
robot_height = 0.33
grf_as_state = True

p0 = jnp.array([0, 0, 0.33])
quat0 = jnp.array([1, 0, 0, 0])
q0 = jnp.array([0.2, 0.8, -1.8, -0.2, 0.8, -1.8, 0.2, 0.8, -1.8, -0.2, 0.8, -1.8])
q0_init = jnp.array(
    [-0.2, 0.8, -1.8, -0.2, 0.8, -1.8, -0.2, 0.8, -1.8, -0.2, 0.8, -1.8]
)

p_legs0 = jnp.array(
    [
        0.27092872,
        0.174,
        -0.31,
        0.27092872,
        -0.174,
        -0.31,
        -0.20887128,
        0.174,
        -0.31,
        -0.20887128,
        -0.174,
        -0.31,
    ]
)

n_joints = 12
n_contact = len(contact_frame)
n = 13 + 2 * n_joints + 6 * n_contact
m = n_joints
tau_ref = jnp.zeros(n_joints)
u_ref = jnp.concatenate([tau_ref])

Qp = jnp.diag(jnp.array([0, 0, 5e4]))
Qrot = jnp.diag(jnp.array([100, 100, 100]))
Qq = jnp.diag(jnp.ones(n_joints)) * 1e2
Qdp = jnp.diag(jnp.array([100, 100, 100]))
Qomega = jnp.diag(jnp.array([1, 1, 1])) * 1e2
Qdq = jnp.diag(jnp.ones(n_joints)) * 1e-1
Qtau = jnp.diag(jnp.ones(n_joints)) * 1e-1
Q_grf = jnp.diag(jnp.ones(3 * n_contact)) * 0
Qleg = jnp.diag(jnp.tile(jnp.array([1e3, 1e3, 5e4]), n_contact))
W = jax.scipy.linalg.block_diag(Qp, Qrot, Qq, Qdp, Qomega, Qdq, Qleg, Qtau, Q_grf)

use_terrain_estimation = True

_state_extra = n - (13 + 2 * n_joints + 3 * n_contact)
initial_state = jnp.concatenate(
    [p0, quat0, q0, jnp.zeros(6 + n_joints), p_legs0, jnp.zeros(_state_extra)]
)

cost = partial(objectives.quadruped_wb_obj, False, n_joints, n_contact, N)
cost_smooth = partial(
    objectives.quadruped_wb_smooth_cost, False, n_joints, n_contact, N
)
inequalities = partial(
    objectives.quadruped_wb_inequalities, n_joints, n_contact, 0.5, 50.0, 20.0
)


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


reference = references.reference_barell_roll

max_torque = 40
min_torque = -40

lipa_enforce_inequalities = True

lipa_settings = SolverSettings(
    max_iterations=100,
    η0=1e9,
    η_update_factor=1.1,
    µ_update_factor=0.9,
    cost_improvement_threshold=1e-3,
    primal_violation_threshold=1e-3,
    use_parallel_lqr=False,
    num_parallel_line_search_steps=1,
    mehrotra_mu=True,
)
