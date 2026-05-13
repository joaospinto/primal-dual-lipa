"""H1 humanoid backflip task config.

Reuses the H1 kinodynamic dynamics, soft-penalty cost, smooth cost, and
friction inequalities from ``config_h1_jump_forward`` — only the
trajectory horizon, weighting on rotation tracking, and the reference
function differ. The single 360° pitch rotation in the flight phase
passes through the quaternion sign-flip point at the apex; we know
from the single barrel-roll case that the SQP can handle a single
traversal of that singularity.
"""

import os
from functools import partial

import jax
import jax.numpy as jnp

from primal_dual_lipa.types import SolverSettings

from tests.mpc_examples import models, objectives, references

task_name = "h1_backflip"

dir_path = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.abspath(
    os.path.join(dir_path, "..", "data", "unitree_h1", "mjx_h1_walk_real_feet.xml")
)
scene_path = os.path.abspath(
    os.path.join(dir_path, "..", "data", "unitree_h1", "mjx_scene_h1_walk.xml")
)

contact_frame = ["FL", "RL", "FR", "RR"]
body_name = ["left_ankle_link", "right_ankle_link"]

# Trajectory length matches the segment plan in
# `references.reference_humanoid_backflip`: crouch 0.25s + flight 0.9s +
# land 0.25s + the rest is final settle. We allocate a generous settle
# (~0.6s = 30 stages) so the cost has many stages to reward "stand
# still at q0" at the end and discourage trajectories that land then
# tip over.
dt = 0.02
# Bumped from 100 → 130 to give the settle phase more time to fully
# dissipate residual pitch rate. With 100 stages the body still has
# omega_y ≈ -0.4 rad/s at the trajectory end, which compounds into
# ~5° of backward tilt at t=100. Extending settle by 0.6 s gives the
# IPM enough room to fully zero out angular momentum.
N = 130
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

# Lean cost weights. Earlier iterations had everything at 1e4-1e5
# per-stage, which over-prescribed the trajectory and forced the
# optimizer to fight against the dynamics + hard-IPM constraints,
# inflating iter counts to thousands. This pares the per-stage cost
# down to the minimum needed to disambiguate the maneuver, leaving
# the dynamics and hard inequalities to enforce physics. Terminal
# pose is enforced separately via terminal_cost_multiplier=200 below
# (so the standing-still landing target binds without per-stage
# weights).
#
# Per-axis breakdown:
# * Qp x = 0: don't track body x — the launch translates the body
#   backward by ~5 cm in flight; tracking this with a per-stage
#   weight just creates cost-dynamics tension. Terminal cost handles
#   the post-landing position pin.
# * Qp y = 1e4: pin the body to the symmetry plane (without it the
#   flight phase drifted laterally and the asymmetric landing
#   sprayed pitch into yaw).
# * Qp z = 1e4: track the parabolic flight z-profile so the body
#   actually goes up and comes down on schedule.
Qp = jnp.diag(jnp.array([0.0, 1e4, 1e4]))
# Qrot all-axes 1e3: roll/yaw light suppression and pitch-tracking
# pull. Earlier 1e4 off-axis was over-prescriptive; the body is
# already pulled into the rotation by Qomega + the dynamics.
Qrot = jnp.diag(jnp.array([1e3, 1e3, 1e3]))
# Qq uniform 4e0 — light per-stage joint-position pull so the legs
# don't drift to absurd configurations, but not so heavy that it
# fights the natural leg-swing during flight. Terminal cost makes
# the landing pose actually bind. Off-sagittal arm joints get 4e2
# (mild bias toward q0) since a backflip should stay in the sagittal
# plane.
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
            4e0,
            4e0,
            4e2,
            4e2,
            4e0,
            4e0,
            4e2,
            4e2,
            4e0,
        ]
    )
)
# Light velocity regularization so the trajectory is smooth.
Qdp = jnp.diag(jnp.array([1e2, 1e2, 1e2]))
# Light angular velocity tracking. Terminal cost handles the final-
# pose-and-stillness enforcement; per-stage just needs enough weight
# to disambiguate the rotation direction (sign on omega_y).
Qomega = jnp.diag(jnp.array([1e1, 1e1, 1e1]))
Qdq = jnp.diag(jnp.ones(n_joints)) * 1e0
# Now that the flight foot ref tracks the rotating body (see
# `references.reference_humanoid_backflip`), Qleg can be high enough
# to enforce the landing foot positions. Set the same as
# config_h1_jump_forward.
Qleg = jnp.diag(jnp.tile(jnp.array([1e5, 1e5, 1e5]), n_contact))
Qdq_cmd = jnp.diag(jnp.ones(n_joints)) * 1e-2
Qgrf = jnp.diag(jnp.ones(3 * n_contact)) * 1e-3
W = jax.scipy.linalg.block_diag(Qp, Qrot, Qq, Qdp, Qomega, Qdq, Qleg, Qdq_cmd, Qgrf)

use_terrain_estimation = False
initial_state = jnp.concatenate([p0, quat0, q0, jnp.zeros(6 + n_joints), p_legs0])

# Per-joint motor velocity limits, ordered to match the H1 joint vector:
#   left/right legs (5 each: hip_yaw, hip_roll, hip_pitch, knee, ankle),
#   torso (1), left arm (4: shoulder_pitch, shoulder_roll, shoulder_yaw,
#   elbow), right arm (4). Numbers are tuned to balance physical
#   realism against IPM convergence — H1's actual no-load top speeds
#   (≈9–19 rad/s) are too tight for the backflip to be feasible at
#   dt=0.02s, but uniform 60 rad/s let the arms flail unrealistically.
#   Legs/torso get 40 rad/s, arms get 25 (since the arm motors are
#   the weakest, ±18 N·m on shoulder-yaw / elbow).
dq_max_h1_backflip = jnp.array(
    [
        40.0,
        40.0,
        40.0,
        40.0,
        40.0,  # left leg
        40.0,
        40.0,
        40.0,
        40.0,
        40.0,  # right leg
        30.0,  # torso
        25.0,
        25.0,
        25.0,
        25.0,  # left arm
        25.0,
        25.0,
        25.0,
        25.0,  # right arm
    ]
)
# qdd_max only used by the soft warmup penalty now; sized roughly so a
# joint can go from 0 to dq_max in 2 dt steps.
qdd_max_h1_backflip = jnp.array(
    [
        2000.0,
        2000.0,
        2000.0,
        2000.0,
        2000.0,
        2000.0,
        2000.0,
        2000.0,
        2000.0,
        2000.0,
        1500.0,
        1200.0,
        1200.0,
        1200.0,
        1200.0,
        1200.0,
        1200.0,
        1200.0,
        1200.0,
    ]
)

# Terminal-cost multiplier of 200 is load-bearing for the backflip
# settle phase: without it the finite-horizon optimizer leaves
# residual ω_y at the trajectory end (no cost beyond N to penalize
# the resulting drift), and the body slowly tilts backward over the
# last 5–10 stages. The other H1 tasks (jump_forward) don't need
# this — they end in a static stance with no residual angular
# momentum to dissipate — so the default in objectives.py is 1.
cost = partial(
    objectives.h1_kinodynamic_obj,
    n_joints,
    n_contact,
    N,
    dq_max=dq_max_h1_backflip,
    qdd_max=qdd_max_h1_backflip,
    dt=dt,
    terminal_cost_multiplier=200.0,
)
cost_smooth = partial(
    objectives.h1_kinodynamic_smooth_cost,
    n_joints,
    n_contact,
    N,
    terminal_cost_multiplier=200.0,
)
inequalities = partial(
    objectives.h1_kinodynamic_inequalities,
    n_joints,
    n_contact,
    0.7,
    dq_max=dq_max_h1_backflip,
    qdd_max=qdd_max_h1_backflip,
    dt=dt,
)


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
    references.reference_humanoid_backflip,
    base_height=robot_height,
    crouch_height=0.62,
    apex_height=1.5,
    backward_distance=-0.05,
    crouch_time=0.25,
    flight_time=0.9,
    land_time=0.25,
)

lipa_enforce_inequalities = True
lipa_skip_warmup_phase = True

lipa_settings = SolverSettings(
    max_iterations=500,
    η0=1e9,
    η_update_factor=1.0,
    µ_update_factor=0.9,
    cost_improvement_threshold=1.0,
    primal_violation_threshold=1.0,
    num_iterative_refinement_steps=2,
    use_parallel_lqr=False,
    num_parallel_line_search_steps=1,
)

# Single-phase IPM. With the lean cost weights (Qrot at 1e3 etc),
# foot-tracking gated by contact, and the joint-velocity hard
# inequalities, the IPM converges from the bare reference initial
# guess in ~710 iterations to defect ~6e-8. The two-phase warmup
# variant gives marginally fewer total iterations (~600) but its
# warmup phase plateaus at a cap, which we'd rather avoid for a
# clean "naturally terminated" run.
lipa_settings_enforce = SolverSettings(
    max_iterations=1000,
    η0=1e9,
    η_update_factor=1.5,
    µ_update_factor=0.9,
    cost_improvement_threshold=1.0,
    primal_violation_threshold=1.0,
    num_iterative_refinement_steps=2,
    use_parallel_lqr=False,
    num_parallel_line_search_steps=1,
)
