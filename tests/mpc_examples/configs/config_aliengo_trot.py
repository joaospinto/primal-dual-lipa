"""Aliengo trot task config.

Provenance: mirrors ``mpx/config/config_aliengo_trot_two_step.py`` —
https://github.com/iit-DLSLab/mpx (BSD-3-Clause). Renamed to drop the
``_two_step`` suffix because ``n_phases`` is now configurable.
"""

from functools import partial

from primal_dual_lipa.types import SolverSettings

from tests.mpc_examples import objectives, references
from tests.mpc_examples.configs import _aliengo_base as base

task_name = "aliengo_trot"

model_path = base.model_path
scene_path = base.scene_path
contact_frame = base.contact_frame
body_name = base.body_name

dt = 0.02

# Gait shape — N is derived from these so that bumping `n_phases`
# automatically extends the horizon proportionally and preserves the
# pre-/post-trot stance ratio. Original (n_phases=4) used N=60: 5
# initial-stance + 4*8 active + 23 final-settle. Same per-phase ratio
# applies for any n_phases.
n_phases = 8
phase_time = 0.16  # time per swing-pair phase
settle_time = 0.10  # initial standing time before the trot starts
final_settle_time = 0.46  # standing time after the last phase
step_length = 0.16

N = (
    int(settle_time / dt)
    + n_phases * int(phase_time / dt)
    + int(final_settle_time / dt)
)

mpc_frequency = base.mpc_frequency

timer_t = base.timer_t
duty_factor = 0.5
step_freq = base.step_freq
step_height = 0.08
initial_height = base.initial_height
robot_height = base.robot_height

p0 = base.p0
quat0 = base.quat0
q0 = base.q0
p_legs0 = base.p_legs0

n_joints = base.n_joints
n_contact = base.n_contact
n = base.n
m = base.m
grf_as_state = base.grf_as_state
u_ref = base.u_ref
W = base.W

use_terrain_estimation = False
initial_state = base.initial_state
dynamics = base.dynamics
max_torque = base.max_torque
min_torque = base.min_torque

cost = partial(objectives.quadruped_wb_obj, True, n_joints, n_contact, N)
cost_smooth = partial(objectives.quadruped_wb_smooth_cost, True, n_joints, n_contact, N)
inequalities = partial(
    objectives.quadruped_wb_inequalities, n_joints, n_contact, 0.5, 44.0, 10.0
)

# Forward body translation. Original config used 0.45 with n_phases=4
# (0.1125 m/phase — body slightly leads the feet, which advance step_length
# every two phases). Preserve that per-phase rate for any n_phases.
forward_per_phase = 0.45 / 4

reference = partial(
    references.reference_quadruped_trot,
    n_phases=n_phases,
    base_height=robot_height,
    total_forward=forward_per_phase * n_phases,
    step_length=step_length,
    step_height=step_height,
    settle_time=settle_time,
    phase_time=phase_time,
)

lipa_enforce_inequalities = True

lipa_settings = SolverSettings(
    max_iterations=100,
    η0=1e9,
    η_update_factor=1.0,
    µ_update_factor=0.9,
    cost_improvement_threshold=1e-3,
    primal_violation_threshold=1e-5,
    use_parallel_lqr=False,
    num_parallel_line_search_steps=1,
)
