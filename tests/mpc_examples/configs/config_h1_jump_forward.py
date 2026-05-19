"""H1 humanoid jump-forward task config.

Provenance: composed from ``mpx/config/config_h1_jump_forward.py`` and
``mpx/config/config_h1_kinodynamic.py`` (the parent config it inherits
from upstream) — https://github.com/iit-DLSLab/mpx (BSD-3-Clause).
Inlined into a single file here since we don't run the other H1 tasks
that share the kinodynamic base.
"""

import os
from functools import partial

import jax
import jax.numpy as jnp

from primal_dual_lipa.types import SolverSettings

from tests.mpc_examples import models, objectives, references

task_name = "h1_jump_forward"

dir_path = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.abspath(
    os.path.join(dir_path, "..", "data", "unitree_h1", "mjx_h1_walk_real_feet.xml")
)
# The renderable scene (with floor + skybox) lives one file up the chain.
scene_path = os.path.abspath(
    os.path.join(dir_path, "..", "data", "unitree_h1", "mjx_scene_h1_walk.xml")
)

contact_frame = ["FL", "RL", "FR", "RR"]
body_name = ["left_ankle_link", "right_ankle_link"]

# Two consecutive jumps (n_jumps=2 in the reference). Per-jump stages
# at dt=0.02: 10 (crouch) + 14 (flight) + 9 (land) = 33. Plus 15 stages
# of standing between jumps (between_jumps_time=0.30). Total used: 81.
# We allocate a few extra stages for the final settle.
dt = 0.02
N = 100
mpc_frequency = 50

timer_t = jnp.array([0.5, 0.5, 0.0, 0.0])
duty_factor = 1.0
step_freq = 1.2
step_height = 0.08
initial_height = 0.9
robot_height = 0.9

p0 = jnp.array([0.0, 0.0, 0.9])
quat0 = jnp.array([1.0, 0.0, 0.0, 0.0])
q0 = jnp.array(
    [
        0.0,
        0.0,
        -0.54,
        1.2,
        -0.68,
        0.0,
        0.0,
        -0.54,
        1.2,
        -0.68,
        0.0,
        0.5,
        0.25,
        0.0,
        0.5,
        0.5,
        -0.25,
        0.0,
        0.5,
    ]
)

p_legs0 = jnp.array(
    [
        0.14738185,
        0.20541158,
        0.01398883,
        -0.00253908,
        0.2102815,
        0.01398485,
        0.14787466,
        -0.20581408,
        0.01399987,
        -0.00203967,
        -0.21088305,
        0.0139761,
    ]
)

n_joints = 19
n_contact = len(contact_frame)
n = 13 + 2 * n_joints + 3 * n_contact
m = n_joints + 3 * n_contact
grf_as_state = False
u_ref = jnp.zeros(m)

Qp = jnp.diag(jnp.array([0.0, 0.0, 1e4]))
Qrot = jnp.diag(jnp.array([1.0, 1.0, 0.0])) * 1e3
Qq = jnp.diag(
    jnp.array(
        [
            4e0,
            4e0,
            4e0,
            4e0,
            4e0,
            4e0,
            4e0,
            4e0,
            4e0,
            4e0,
            4e1,
            4e1,
            4e1,
            4e1,
            4e1,
            4e1,
            4e1,
            4e1,
            4e1,
        ]
    )
)
Qdp = jnp.diag(jnp.array([1.0, 1.0, 1.0])) * 1e3
Qomega = jnp.diag(jnp.array([1.0, 1.0, 1.0])) * 1e2
Qdq = jnp.diag(jnp.ones(n_joints)) * 1e0
Qleg = jnp.diag(jnp.tile(jnp.array([1e5, 1e5, 1e5]), n_contact))
Qdq_cmd = jnp.diag(jnp.ones(n_joints)) * 1e-2
Qgrf = jnp.diag(jnp.ones(3 * n_contact)) * 1e-3
W = jax.scipy.linalg.block_diag(Qp, Qrot, Qq, Qdp, Qomega, Qdq, Qleg, Qdq_cmd, Qgrf)

use_terrain_estimation = False
initial_state = jnp.concatenate([p0, quat0, q0, jnp.zeros(6 + n_joints), p_legs0])

joint_kp = jnp.array(
    [
        200.0,
        200.0,
        200.0,
        200.0,
        60.0,
        200.0,
        200.0,
        200.0,
        200.0,
        60.0,
        200.0,
        60.0,
        60.0,
        60.0,
        60.0,
        60.0,
        60.0,
        60.0,
        60.0,
    ]
)
joint_kd = jnp.array(
    [
        5.0,
        5.0,
        5.0,
        5.0,
        1.5,
        5.0,
        5.0,
        5.0,
        5.0,
        1.5,
        5.0,
        1.5,
        1.5,
        1.5,
        1.5,
        1.5,
        1.5,
        1.5,
        1.5,
    ]
)

torque_limits = jnp.array(
    [
        200.0,
        200.0,
        200.0,
        300.0,
        40.0,
        200.0,
        200.0,
        200.0,
        300.0,
        40.0,
        200.0,
        40.0,
        40.0,
        18.0,
        18.0,
        40.0,
        40.0,
        18.0,
        18.0,
    ]
)
max_torque = torque_limits
min_torque = -torque_limits

cost = partial(objectives.h1_kinodynamic_obj, n_joints, n_contact, N)
cost_smooth = partial(objectives.h1_kinodynamic_smooth_cost, n_joints, n_contact, N)
inequalities = partial(objectives.h1_kinodynamic_inequalities, n_joints, n_contact, 0.7)


def dynamics(model, mjx_model, contact_id, body_id):
    return partial(
        models.h1_kinodynamic_dynamics,
        model,
        mjx_model,
        contact_id,
        body_id,
        n_joints,
        dt,
    )


reference = partial(
    references.reference_humanoid_jump_forward,
    n_jumps=2,
    base_height=robot_height,
    crouch_height=0.82,
    apex_height=1.02,
    jump_distance=0.35,
    foot_shift=0.18,
    foot_lift=0.10,
    between_jumps_time=0.30,
)

lipa_enforce_inequalities = True

lipa_settings = SolverSettings(
    max_iterations=500,
    η0=1e9,
    η_update_factor=1.0,
    µ_update_factor=0.9,
    cost_improvement_threshold=1e-3,
    primal_violation_threshold=1e-3,
    num_iterative_refinement_steps=2,
    use_parallel_lqr=False,
    num_parallel_line_search_steps=1,
    mehrotra_mu=True,
)

lipa_settings_enforce = SolverSettings(
    max_iterations=500,
    η0=1e5,
    η_update_factor=2.0,
    µ_update_factor=0.9,
    cost_improvement_threshold=1e-3,
    primal_violation_threshold=1e-3,
    mehrotra_mu=True,
    num_iterative_refinement_steps=2,
    use_parallel_lqr=False,
    num_parallel_line_search_steps=1,
)
