"""Stage costs and inequalities for the MJX whole-body MPC tasks.

Provenance: subset ported from ``mpx/utils/objectives.py`` —
https://github.com/iit-DLSLab/mpx (BSD-3-Clause).

Only the quadruped whole-body and H1 kinodynamic flavors used by
``barrel_roll``, ``aliengo_trot``, and ``h1_jump_forward``. Soft-penalty
cost (``*_obj``), strictly-smooth cost (``*_smooth_cost``) for LIPA's
enforce-inequalities path, and the inequality functions are all kept;
the per-cost Gauss-Newton Hessian helpers (``*_hessian_gn``) are dropped
— LIPA computes Hessians via autodiff.
"""

import jax
from jax import numpy as jnp
from mujoco.mjx._src import math


def penalty(constraint, alpha=0.1, sigma=5):
    def safe_log(x):
        x = jnp.clip(x, 1e-10, 1e6)
        return jnp.log(x)

    quadratic_barrier = (
        alpha
        / 2
        * (jnp.square((constraint - 2 * sigma) / sigma) - jnp.ones_like(constraint))
    )
    log_barrier = -alpha * safe_log(constraint)
    return jnp.clip(
        jnp.where(constraint > sigma, log_barrier, quadratic_barrier + log_barrier),
        0,
        1e8,
    )


# ---------------------------------------------------------------------------
# Quadruped whole-body (used by barrel_roll, aliengo_trot)
# ---------------------------------------------------------------------------


def _quadruped_wb_constraint_slacks(
    n_joints, n_contact, mu, torque_limit, dq_limit, x, u, friction_eps=1e-2
):
    grf = x[13 + 2 * n_joints + 3 * n_contact :]
    tau = u[:n_joints]
    dq = x[13 + n_joints : 13 + 2 * n_joints]
    Fx = grf[0::3]
    Fy = grf[1::3]
    Fz = grf[2::3]
    s_friction = mu * Fz - jnp.sqrt(
        jnp.square(Fx) + jnp.square(Fy) + jnp.ones(n_contact) * friction_eps
    )
    sym = jnp.kron(jnp.eye(n_joints), jnp.array([-1.0, 1.0])).T
    s_torque = sym @ tau + (torque_limit + 1e-2)
    s_dq = sym @ dq + (dq_limit + 1e-2)
    return s_friction, s_torque, s_dq


def quadruped_wb_inequalities(
    n_joints,
    n_contact,
    mu,
    torque_limit,
    dq_limit,
    reference,
    x,
    u,
    t,
    friction_eps=1e-12,
):
    """LIPA-form inequalities ``g(x,u,t) <= 0`` for the quadruped whole-body problem.

    Friction is gated by the reference contact mask (vacuous in swing); torque and
    joint-speed limits are always active. At the terminal stage there is no control
    input, so all entries collapse to zero.
    """
    s_friction, s_torque, s_dq = _quadruped_wb_constraint_slacks(
        n_joints,
        n_contact,
        mu,
        torque_limit,
        dq_limit,
        x,
        u,
        friction_eps=friction_eps,
    )
    contact = reference[
        t, 13 + n_joints + 3 * n_contact : 13 + n_joints + 4 * n_contact
    ]
    g = jnp.concatenate([-contact * s_friction, -s_torque, -s_dq])
    N = reference.shape[0] - 1
    return jnp.where(t == N, jnp.zeros_like(g), g)


def quadruped_wb_smooth_cost(
    swing_tracking, n_joints, n_contact, N, W, reference, x, u, t
):
    """Stage cost without any soft-inequality penalties (friction/torque/dq)."""
    p = x[:3]
    quat = x[3:7]
    q = x[7 : 7 + n_joints]
    dp = x[7 + n_joints : 10 + n_joints]
    omega = x[10 + n_joints : 13 + n_joints]
    dq = x[13 + n_joints : 13 + 2 * n_joints]
    p_leg = x[13 + 2 * n_joints : 13 + 2 * n_joints + 3 * n_contact]
    grf = x[13 + 2 * n_joints + 3 * n_contact :]
    tau = u[:n_joints]

    p_ref = reference[t, :3]
    quat_ref = reference[t, 3:7]
    q_ref = reference[t, 7 : 7 + n_joints]
    dp_ref = reference[t, 7 + n_joints : 10 + n_joints]
    omega_ref = reference[t, 10 + n_joints : 13 + n_joints]
    p_leg_ref = reference[t, 13 + n_joints : 13 + n_joints + 3 * n_contact]
    contact = reference[
        t, 13 + n_joints + 3 * n_contact : 13 + n_joints + 4 * n_contact
    ]
    grf_ref = reference[
        t, 13 + n_joints + 4 * n_contact : 13 + n_joints + 7 * n_contact
    ]

    if swing_tracking:
        contact_map = jnp.ones(3 * n_contact)
    else:
        contact_map = jnp.array(
            [jnp.ones(3) * contact[i] for i in range(n_contact)]
        ).flatten()

    stage_cost = (
        (p - p_ref).T @ W[:3, :3] @ (p - p_ref)
        + math.quat_sub(quat, quat_ref).T @ W[3:6, 3:6] @ math.quat_sub(quat, quat_ref)
        + (q - q_ref).T @ W[6 : 6 + n_joints, 6 : 6 + n_joints] @ (q - q_ref)
        + (dp - dp_ref).T
        @ W[6 + n_joints : 9 + n_joints, 6 + n_joints : 9 + n_joints]
        @ (dp - dp_ref)
        + (omega - omega_ref).T
        @ W[9 + n_joints : 12 + n_joints, 9 + n_joints : 12 + n_joints]
        @ (omega - omega_ref)
        + dq.T
        @ W[12 + n_joints : 12 + 2 * n_joints, 12 + n_joints : 12 + 2 * n_joints]
        @ dq
        + (contact_map * (p_leg - p_leg_ref)).T
        @ W[
            12 + 2 * n_joints : 12 + 2 * n_joints + 3 * n_contact,
            12 + 2 * n_joints : 12 + 2 * n_joints + 3 * n_contact,
        ]
        @ (contact_map * (p_leg - p_leg_ref))
        + tau.T
        @ W[
            12 + 2 * n_joints + 3 * n_contact : 12 + 3 * n_joints + 3 * n_contact,
            12 + 2 * n_joints + 3 * n_contact : 12 + 3 * n_joints + 3 * n_contact,
        ]
        @ tau
        + (grf - grf_ref).T
        @ W[
            12 + 3 * n_joints + 3 * n_contact : 12 + 3 * n_joints + 6 * n_contact,
            12 + 3 * n_joints + 3 * n_contact : 12 + 3 * n_joints + 6 * n_contact,
        ]
        @ (grf - grf_ref)
    )
    term_cost = (
        (p - p_ref).T @ W[:3, :3] @ (p - p_ref)
        + math.quat_sub(quat, quat_ref).T @ W[3:6, 3:6] @ math.quat_sub(quat, quat_ref)
        + (q - q_ref).T @ W[6 : 6 + n_joints, 6 : 6 + n_joints] @ (q - q_ref)
        + (dp - dp_ref).T
        @ W[6 + n_joints : 9 + n_joints, 6 + n_joints : 9 + n_joints]
        @ (dp - dp_ref)
        + (omega - omega_ref).T
        @ W[9 + n_joints : 12 + n_joints, 9 + n_joints : 12 + n_joints]
        @ (omega - omega_ref)
        + dq.T
        @ W[12 + n_joints : 12 + 2 * n_joints, 12 + n_joints : 12 + 2 * n_joints]
        @ dq
    )
    return jnp.where(t == N, 0.5 * term_cost, 0.5 * stage_cost)


def quadruped_wb_obj(swing_tracking, n_joints, n_contact, N, W, reference, x, u, t):
    smooth = quadruped_wb_smooth_cost(
        swing_tracking, n_joints, n_contact, N, W, reference, x, u, t
    )
    s_friction, s_torque, s_dq = _quadruped_wb_constraint_slacks(
        n_joints,
        n_contact,
        0.5,
        44.0,
        10.0,
        x,
        u,
    )
    contact = reference[
        t, 13 + n_joints + 3 * n_contact : 13 + n_joints + 4 * n_contact
    ]
    soft = (
        jnp.sum(penalty(s_friction) * contact)
        + jnp.sum(penalty(s_torque, 1, 1))
        + jnp.sum(penalty(s_dq, 1, 1))
    )
    return smooth + jnp.where(t == N, 0.0, 0.5 * soft)


# ---------------------------------------------------------------------------
# H1 kinodynamic (used by h1_jump_forward)
# ---------------------------------------------------------------------------


def _h1_kinodynamic_friction_slack(n_joints, n_contact, mu, u, friction_eps=1e-1):
    grf = u[n_joints:]
    Fx = grf[0::3]
    Fy = grf[1::3]
    Fz = grf[2::3]
    return mu * Fz - jnp.sqrt(
        jnp.square(Fx) + jnp.square(Fy) + jnp.ones(n_contact) * friction_eps
    )


def h1_kinodynamic_inequalities(
    n_joints,
    n_contact,
    mu,
    reference,
    x,
    u,
    t,
    friction_eps=1e-12,
    dq_max=50.0,
    qdd_max=1500.0,
    dt=0.02,
    foot_contact_clearance=0.05,
):
    """LIPA-form ``g <= 0`` inequalities for the H1 kinodynamic problem.

    Five physical constraints:

    * ``Fz >= 0`` — a foot can only push into the ground, not pull. Without
      this, the soft-penalty optimizer happily uses negative Fz to "anchor"
      the foot, which produces unphysical jump take-offs and breaks the
      Coulomb-cone interpretation: with Fz < 0 and ``g = mu*Fz - sqrt(Fx²+Fy²)``
      the cone becomes infeasible by ``≈ |mu*Fz|`` regardless of (Fx, Fy).
    * Friction cone: ``sqrt(Fx² + Fy²) <= mu * Fz``.
    * ``foot_z >= 0`` — feet cannot penetrate the ground. Without this,
      a high-impact landing (e.g. backflip) lets feet sink ~7 cm below
      the floor and the optimizer compensates with sparse asymmetric
      GRFs, which generates unphysical yaw-torque transients.
    * ``|dq_cmd| <= dq_max`` — joint velocity commands cannot exceed
      the motor's no-load top speed. Backflip-tier maneuvers exposed
      that without this bound the optimizer commands joint velocities
      of 100+ rad/s (vs. H1 leg motors' ~15 rad/s top speed) to drive
      large joint accelerations; via the kinodynamic mass-matrix
      coupling block, those accelerations inertially drag the floating
      base, letting the body drop ~24 cm in a single 0.02 s step
      without any GRF involvement.

    Joint-acceleration is bounded only via the soft-penalty term in
    `h1_kinodynamic_obj` (the warmup-phase cost) — adding it as a hard
    inequality on top of the other constraints made the IPM unable to
    converge cleanly within reasonable iteration budgets, and at
    dt = 0.02 s a velocity bound of ~30–80 rad/s already implies a
    qdd bound of ~1500–4000 rad/s² per single step, which is in the
    right physical ballpark for H1.

    The friction/Fz constraints are gated by the reference contact mask
    (vacuous during swing), but ground penetration and the joint-velocity
    limit are forbidden in every phase.
    """
    grf = u[n_joints:]
    Fz = grf[2::3]
    s_friction = _h1_kinodynamic_friction_slack(
        n_joints, n_contact, mu, u, friction_eps=friction_eps
    )
    contact = reference[
        t, 13 + n_joints + 3 * n_contact : 13 + n_joints + 4 * n_contact
    ]
    g_friction = -contact * s_friction
    g_fz = -contact * Fz
    foot_z = x[13 + 2 * n_joints + 2 : 13 + 2 * n_joints + 3 * n_contact : 3]
    g_foot_z = -foot_z
    # When in contact, also bound foot_z from above so the foot
    # actually sits on the floor instead of hovering 5–8 cm above
    # while the optimizer still credits it as "in contact" for the
    # GRF computation. The constraint is gated by the contact mask
    # (vacuous when contact == 0), and `foot_contact_clearance`
    # leaves a small barrier-friendly margin above the floor.
    g_foot_z_contact = contact * (foot_z - foot_contact_clearance)
    dq_cmd = u[:n_joints]
    del qdd_max, dt
    dq_max_arr = jnp.broadcast_to(jnp.asarray(dq_max), (n_joints,))
    g_dq_hi = dq_cmd - dq_max_arr
    g_dq_lo = -dq_cmd - dq_max_arr
    g = jnp.concatenate(
        [g_friction, g_fz, g_foot_z, g_foot_z_contact, g_dq_hi, g_dq_lo]
    )
    N = reference.shape[0] - 1
    return jnp.where(t == N, jnp.zeros_like(g), g)


def h1_kinodynamic_smooth_cost(
    n_joints,
    n_contact,
    N,
    W,
    reference,
    x,
    u,
    t,
    terminal_cost_multiplier=1.0,
):
    """H1 kinodynamic stage cost with the friction soft-penalty stripped out.

    The optional ``terminal_cost_multiplier`` scales the final-stage
    cost (t == N) by an extra factor. Without it the finite-horizon
    optimizer leaves residual omega at the trajectory end (no cost
    beyond N to penalize the resulting drift), which compounds into a
    visible backward tilt of a few degrees over the last 5–10 stages.
    Multiplying the terminal cost makes the final pose / velocity
    targets actually bind.
    """
    p = x[:3]
    quat = x[3:7]
    q = x[7 : 7 + n_joints]
    dp = x[7 + n_joints : 10 + n_joints]
    omega = x[10 + n_joints : 13 + n_joints]
    dq = x[13 + n_joints : 13 + 2 * n_joints]
    p_leg = x[13 + 2 * n_joints : 13 + 2 * n_joints + 3 * n_contact]

    dq_cmd = u[:n_joints]
    grf = u[n_joints:]

    p_ref = reference[t, :3]
    quat_ref = reference[t, 3:7]
    q_ref = reference[t, 7 : 7 + n_joints]
    dp_ref = reference[t, 7 + n_joints : 10 + n_joints]
    omega_ref = reference[t, 10 + n_joints : 13 + n_joints]
    p_leg_ref = reference[t, 13 + n_joints : 13 + n_joints + 3 * n_contact]
    contact = reference[
        t, 13 + n_joints + 3 * n_contact : 13 + n_joints + 4 * n_contact
    ]
    grf_ref = reference[
        t, 13 + n_joints + 4 * n_contact : 13 + n_joints + 7 * n_contact
    ]
    # Foot tracking is gated by the per-foot contact mask: during
    # flight (contact == 0), the optimizer is free to let the legs
    # adopt whatever pose the dynamics + joint costs prefer; only
    # during contact phases do we insist that the foot match the
    # planned foot position. Without this gate the cost was forcing
    # the legs to track a kinematically-prescribed rotating foot
    # trajectory through the entire flight phase, which fights the
    # natural leg-swing the dynamics wants and balloons the IPM
    # iteration count for high-rotation maneuvers like the backflip.
    contact_map = jnp.repeat(contact, 3)

    # Rotation cost: direct quaternion 4-vector difference, NOT the
    # SO(3) log map. The mjx `quat_sub` returns the log map wrapped to
    # [-π, π] (the geodesic-shortest-path error), which is correct for
    # generic orientation tracking but wrong for tracking a path
    # through SO(3) with a specific winding number — e.g. a backflip
    # where the reference rotates ~360° and the optimizer can find an
    # equally-cheap alternate path that rotates ~720° because both
    # endpoints map to identity on SO(3) (where the wrap collapses).
    # Direct 4-vector difference is non-wrapping; it requires the
    # reference quaternion path to be sign-continuous on S³ (no jumps
    # from (-q) to (+q) for the same orientation). Weight is taken
    # from W[3:6, 3:6] (Qrot, 3×3) and applied to the imaginary part
    # only — the qw component is a redundant scalar (qw² + |qxyz|² = 1
    # for unit quats), so penalizing |qxyz - qxyz_ref|² captures the
    # full rotation tracking error along the prescribed path.
    quat_imag_err = quat[1:4] - quat_ref[1:4]
    rot_cost = quat_imag_err.T @ W[3:6, 3:6] @ quat_imag_err

    stage_cost = (
        (p - p_ref).T @ W[:3, :3] @ (p - p_ref)
        + rot_cost
        + (q - q_ref).T @ W[6 : 6 + n_joints, 6 : 6 + n_joints] @ (q - q_ref)
        + (dp - dp_ref).T
        @ W[6 + n_joints : 9 + n_joints, 6 + n_joints : 9 + n_joints]
        @ (dp - dp_ref)
        + (omega - omega_ref).T
        @ W[9 + n_joints : 12 + n_joints, 9 + n_joints : 12 + n_joints]
        @ (omega - omega_ref)
        + dq.T
        @ W[12 + n_joints : 12 + 2 * n_joints, 12 + n_joints : 12 + 2 * n_joints]
        @ dq
        + (contact_map * (p_leg - p_leg_ref)).T
        @ W[
            12 + 2 * n_joints : 12 + 2 * n_joints + 3 * n_contact,
            12 + 2 * n_joints : 12 + 2 * n_joints + 3 * n_contact,
        ]
        @ (contact_map * (p_leg - p_leg_ref))
        + dq_cmd.T
        @ W[
            12 + 2 * n_joints + 3 * n_contact : 12 + 3 * n_joints + 3 * n_contact,
            12 + 2 * n_joints + 3 * n_contact : 12 + 3 * n_joints + 3 * n_contact,
        ]
        @ dq_cmd
        + (grf - grf_ref).T
        @ W[
            12 + 3 * n_joints + 3 * n_contact : 12 + 3 * n_joints + 6 * n_contact,
            12 + 3 * n_joints + 3 * n_contact : 12 + 3 * n_joints + 6 * n_contact,
        ]
        @ (grf - grf_ref)
    )
    term_cost = (
        (p - p_ref).T @ W[:3, :3] @ (p - p_ref)
        + rot_cost
        + (q - q_ref).T @ W[6 : 6 + n_joints, 6 : 6 + n_joints] @ (q - q_ref)
        + (dp - dp_ref).T
        @ W[6 + n_joints : 9 + n_joints, 6 + n_joints : 9 + n_joints]
        @ (dp - dp_ref)
        + (omega - omega_ref).T
        @ W[9 + n_joints : 12 + n_joints, 9 + n_joints : 12 + n_joints]
        @ (omega - omega_ref)
        + dq.T
        @ W[12 + n_joints : 12 + 2 * n_joints, 12 + n_joints : 12 + 2 * n_joints]
        @ dq
    )
    return jnp.where(
        t == N,
        0.5 * terminal_cost_multiplier * term_cost,
        0.5 * stage_cost,
    )


def h1_kinodynamic_obj(
    n_joints,
    n_contact,
    N,
    W,
    reference,
    x,
    u,
    t,
    dq_max=50.0,
    qdd_max=1500.0,
    dt=0.02,
    foot_contact_clearance=0.05,
    terminal_cost_multiplier=1.0,
):
    """Soft-penalty H1 kinodynamic stage cost (used by the warmup phase).

    Mirrors `h1_kinodynamic_inequalities` in the limits it penalizes —
    friction cone, foot ground clearance (both penetration and
    hovering-when-in-contact), joint velocity, joint acceleration.
    Phase 2 then re-enforces the same set as hard IPM constraints;
    keeping the soft set in sync with the hard set means the warmup
    lands in the right basin.

    ``terminal_cost_multiplier`` is forwarded to the inner smooth-cost
    call so the warmup matches the second-phase terminal weighting.
    """
    smooth = h1_kinodynamic_smooth_cost(
        n_joints,
        n_contact,
        N,
        W,
        reference,
        x,
        u,
        t,
        terminal_cost_multiplier=terminal_cost_multiplier,
    )
    s_friction = _h1_kinodynamic_friction_slack(n_joints, n_contact, 0.7, u)
    contact = reference[
        t, 13 + n_joints + 3 * n_contact : 13 + n_joints + 4 * n_contact
    ]
    foot_z = x[13 + 2 * n_joints + 2 : 13 + 2 * n_joints + 3 * n_contact : 3]
    dq_cmd = u[:n_joints]
    dq = x[13 + n_joints : 13 + 2 * n_joints]
    dq_max_arr = jnp.broadcast_to(jnp.asarray(dq_max), (n_joints,))
    qdd_max_arr = jnp.broadcast_to(jnp.asarray(qdd_max), (n_joints,))
    s_foot_z = foot_z
    s_foot_z_contact = foot_contact_clearance - foot_z
    s_dq_hi = dq_max_arr - dq_cmd
    s_dq_lo = dq_max_arr + dq_cmd
    s_qdd_hi = qdd_max_arr * dt - (dq_cmd - dq)
    s_qdd_lo = qdd_max_arr * dt + (dq_cmd - dq)
    soft = (
        jnp.sum(penalty(s_friction) * contact)
        + jnp.sum(penalty(s_foot_z))
        + jnp.sum(penalty(s_foot_z_contact) * contact)
        + jnp.sum(penalty(s_dq_hi))
        + jnp.sum(penalty(s_dq_lo))
        + jnp.sum(penalty(s_qdd_hi))
        + jnp.sum(penalty(s_qdd_lo))
    )
    return smooth + jnp.where(t == N, 0.0, 0.5 * soft)
