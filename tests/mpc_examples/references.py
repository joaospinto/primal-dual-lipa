"""Offline reference (state + contact + foot) sequences for the MPC tasks.

Provenance: subset ported from ``mpx/utils/mpc_utils.py`` —
https://github.com/iit-DLSLab/mpx (BSD-3-Clause).

Only the three offline-task references — barrel roll (Aliengo), humanoid
jump-forward (H1), quadruped trot (Aliengo). The online
``reference_generator``, terrain-orientation, and timer helpers are not
used by the offline tasks and are omitted. The H1 jump-forward and trot
references add a configurable ``n_jumps`` / ``n_phases`` parameter
beyond what mpx ships.

Each function returns ``(reference, parameter)`` where ``reference`` packs
the per-stage tracking targets (position, orientation, joint angles,
velocities, foot positions, contact pattern, grfs) and ``parameter`` is
``(contact_pattern, foot_positions)`` — the per-stage inputs that the
dynamics models consume.
"""

import jax
from jax import numpy as jnp
from mujoco.mjx._src import math


def reference_barell_roll(N, dt, n_joints, n_contact, foot0, q0):
    t1 = 0.2
    t2 = 0.2
    t3 = 0.3
    t4 = 0.1
    z_start = 0.4
    z_land = 0.28
    v_lateral = -0.25 / (t2 + t3)
    v0 = (z_land - z_start + 0.5 * 9.81 * t3 * t3) / t3
    total_roll_time = t2 + t3 + t4
    roll_speed = 2 * 3.14 / total_roll_time

    def z_position(t):
        return z_start - 0.5 * 9.81 * t**2 + v0 * t

    def z_speed(t):
        return -9.81 * t + v0

    acc = v0 / t2

    n1 = int(t1 / dt)
    p1 = jnp.tile(jnp.array([0, 0, 0.33]), (n1, 1))
    p1 = p1.at[:, 1].set(jnp.arange(n1) * dt * v_lateral)
    dp1 = jnp.tile(jnp.array([0, v_lateral, 0]), (n1, 1))
    contact1 = jnp.tile(jnp.array([1, 1, 1, 1]), (n1, 1))
    quat1 = jnp.tile(jnp.array([1, 0, 0, 0]), (n1, 1))
    omega1 = jnp.tile(jnp.array([0, 0, 0]), (n1, 1))

    n2 = int(t2 / dt)
    p2 = jnp.tile(jnp.array([0, p1[-1, 1], 0.33]), (n2, 1))
    p2 = p2.at[:, 2].set(0.5 * jnp.arange(n2) * dt * jnp.arange(n2) * dt * acc + 0.33)
    p2 = p2.at[:, 1].set(jnp.arange(n2) * dt * v_lateral)
    dp2 = jnp.tile(jnp.array([0, v_lateral, 0]), (n2, 1))
    dp2 = dp2.at[:, 2].set(jnp.arange(n2) * dt * acc)
    contact2 = jnp.tile(jnp.array([0, 1, 0, 1]), (n2, 1))

    n3 = int(t3 / dt)
    p3 = jnp.tile(jnp.array([0, p2[-1, 1], p2[-1, 2]]), (n3, 1))
    p3 = p3.at[:, 1].set(jnp.arange(n3) * dt * v_lateral)
    dp3 = jnp.tile(jnp.array([0, v_lateral, 0]), (n3, 1))
    for i in range(n3):
        p3 = p3.at[i, 2].set(z_position(i * dt))
        dp3 = dp3.at[i, 2].set(z_speed(i * dt))

    def fn(t, carry):
        quat_new = math.quat_integrate(
            carry[t - 1, :], jnp.array([roll_speed, 0, 0]), dt
        )
        return carry.at[t, :].set(quat_new)

    contact3 = jnp.tile(jnp.array([0, 0, 0, 0]), (n3, 1))

    n4 = int(t4 / dt)
    p4 = jnp.tile(jnp.array([0, p3[-1, 1], z_land]), (n4, 1))
    dp4 = jnp.tile(jnp.array([0, 0, 0]), (n4, 1))
    contact4 = jnp.tile(jnp.array([1, 1, 1, 1]), (n4, 1))

    init_carry = jnp.tile(jnp.array([1.0, 0.0, 0, 0]), (n2 + n3 + n4, 1))
    quat234 = jax.lax.fori_loop(1, n2 + n3 + n4, fn, init_carry)
    omega234 = jnp.tile(jnp.array([roll_speed, 0, 0]), (n2 + n3 + n4, 1))

    n5 = N - (n1 + n2 + n3 + n4)

    p5 = jnp.tile(jnp.array([0, p4[-1, 1], z_land]), (n5, 1))
    dp5 = jnp.tile(jnp.array([0, 0, 0]), (n5, 1))
    quat5 = jnp.tile(jnp.array([1, 0, 0, 0]), (n5, 1))
    omega5 = jnp.tile(jnp.array([0, 0, 0]), (n5, 1))
    contact5 = jnp.tile(jnp.array([1, 1, 1, 1]), (n5, 1))

    p_ref = jnp.concatenate([p1, p2, p3, p4, p5], axis=0)
    quat_ref = jnp.concatenate([quat1, quat234, quat5], axis=0)
    q_ref = jnp.tile(q0, (n1 + n2 + n3 + n4 + n5, 1))
    dp_ref = jnp.concatenate([dp1, dp2, dp3, dp4, dp5], axis=0)
    omega_ref = jnp.concatenate([omega1, omega234, omega5], axis=0)
    foot_ref = jnp.tile(foot0, (n1 + n2 + n3 + n4 + n5, 1)) + jnp.tile(p_ref, n_contact)
    foot_ref = foot_ref.at[:, 2::3].set(jnp.zeros((n1 + n2 + n3 + n4 + n5, n_contact)))
    contact_sequence = jnp.concatenate(
        [contact1, contact2, contact3, contact4, contact5], axis=0
    )

    grf_ref = jnp.zeros((N, 3 * n_contact))

    return (
        jnp.concatenate(
            [
                p_ref,
                quat_ref,
                q_ref,
                dp_ref,
                omega_ref,
                foot_ref,
                contact_sequence,
                grf_ref,
            ],
            axis=1,
        ),
        jnp.concatenate([contact_sequence, foot_ref], axis=1),
    )


def reference_humanoid_jump_forward(
    N,
    dt,
    n_joints,
    n_contact,
    foot0,
    q0,
    *,
    n_jumps=1,
    base_height=0.9,
    crouch_height=0.82,
    apex_height=1.02,
    jump_distance=0.35,
    foot_shift=0.18,
    foot_lift=0.12,
    between_jumps_time=0.30,
):
    """Forward-jump reference for an H1-class humanoid.

    Each jump is the four-segment cycle (crouch → flight → land → standing),
    advancing the floating-base by ``jump_distance`` and shifting each foot
    forward by ``foot_shift``. With ``n_jumps > 1`` jumps are separated by a
    ``between_jumps_time`` standing interval; the final settle period after
    the last jump fills any remaining stages.

    The trajectory must fit within ``N`` stages — the function silently
    truncates the final settle period if ``N`` is smaller than the sum of
    per-jump and inter-jump segments.
    """
    n_crouch = max(2, int(0.20 / dt))
    n_flight = max(2, int(0.28 / dt))
    n_land = max(2, int(0.18 / dt))
    n_between = max(2, int(between_jumps_time / dt))
    n_used = n_jumps * (n_crouch + n_flight + n_land) + max(0, n_jumps - 1) * n_between
    n_settle = max(0, N - n_used)

    crouch_q = q0.at[2].set(-0.8).at[3].set(1.5).at[4].set(-0.8)
    crouch_q = crouch_q.at[7].set(-0.8).at[8].set(1.5).at[9].set(-0.8)

    flight_phase = jnp.linspace(0.0, 1.0, n_flight)

    p_segs, q_segs, foot_segs, contact_segs = [], [], [], []
    cur_x = 0.0
    cur_foot = foot0

    for k in range(n_jumps):
        # Crouch
        x_c = jnp.linspace(cur_x, cur_x + 0.05, n_crouch)
        z_c = jnp.linspace(base_height, crouch_height, n_crouch)
        p_segs.append(jnp.stack([x_c, jnp.zeros_like(x_c), z_c], axis=1))
        q_segs.append(
            jnp.stack(
                [q0 + (crouch_q - q0) * a for a in jnp.linspace(0.0, 1.0, n_crouch)],
                axis=0,
            )
        )
        contact_segs.append(jnp.tile(jnp.ones(n_contact), (n_crouch, 1)))
        foot_segs.append(jnp.tile(cur_foot, (n_crouch, 1)))

        # Flight
        new_x = cur_x + jump_distance
        x_f = jnp.linspace(x_c[-1], new_x, n_flight)
        z_f = (
            crouch_height
            + (base_height - crouch_height) * flight_phase
            + (apex_height - base_height) * 4.0 * flight_phase * (1.0 - flight_phase)
        )
        p_segs.append(jnp.stack([x_f, jnp.zeros_like(x_f), z_f], axis=1))
        q_segs.append(jnp.tile(crouch_q, (n_flight, 1)))
        contact_segs.append(jnp.tile(jnp.zeros(n_contact), (n_flight, 1)))
        ff = jnp.tile(cur_foot, (n_flight, 1))
        flight_shift = foot_shift * flight_phase
        flight_lift = foot_lift * 4.0 * flight_phase * (1.0 - flight_phase)
        ff = ff.at[:, ::3].set(ff[:, ::3] + flight_shift[:, None])
        ff = ff.at[:, 2::3].set(ff[:, 2::3] + flight_lift[:, None])
        foot_segs.append(ff)

        # Land
        new_foot = cur_foot.at[::3].set(cur_foot[::3] + foot_shift)
        x_l = jnp.linspace(new_x, new_x, n_land)
        z_l = jnp.linspace(base_height, base_height, n_land)
        p_segs.append(jnp.stack([x_l, jnp.zeros_like(x_l), z_l], axis=1))
        q_segs.append(
            jnp.stack(
                [
                    crouch_q + (q0 - crouch_q) * a
                    for a in jnp.linspace(0.0, 1.0, n_land)
                ],
                axis=0,
            )
        )
        contact_segs.append(jnp.tile(jnp.ones(n_contact), (n_land, 1)))
        foot_segs.append(jnp.tile(new_foot, (n_land, 1)))

        cur_x = new_x
        cur_foot = new_foot

        # Standing interval between jumps (skip after the last jump)
        if k < n_jumps - 1:
            x_b = jnp.linspace(cur_x, cur_x, n_between)
            z_b = jnp.linspace(base_height, base_height, n_between)
            p_segs.append(jnp.stack([x_b, jnp.zeros_like(x_b), z_b], axis=1))
            q_segs.append(jnp.tile(q0, (n_between, 1)))
            contact_segs.append(jnp.tile(jnp.ones(n_contact), (n_between, 1)))
            foot_segs.append(jnp.tile(cur_foot, (n_between, 1)))

    # Final settle to fill out N
    if n_settle > 0:
        x_s = jnp.linspace(cur_x, cur_x, n_settle)
        z_s = jnp.linspace(base_height, base_height, n_settle)
        p_segs.append(jnp.stack([x_s, jnp.zeros_like(x_s), z_s], axis=1))
        q_segs.append(jnp.tile(q0, (n_settle, 1)))
        contact_segs.append(jnp.tile(jnp.ones(n_contact), (n_settle, 1)))
        foot_segs.append(jnp.tile(cur_foot, (n_settle, 1)))

    p_ref = jnp.concatenate(p_segs, axis=0)
    q_ref = jnp.concatenate(q_segs, axis=0)
    foot_ref = jnp.concatenate(foot_segs, axis=0)
    contact_sequence = jnp.concatenate(contact_segs, axis=0)

    quat_ref = jnp.tile(jnp.array([1.0, 0.0, 0.0, 0.0]), (N, 1))
    omega_ref = jnp.zeros((N, 3))

    dp_ref = jnp.zeros((N, 3))
    dp_ref = dp_ref.at[:-1].set((p_ref[1:] - p_ref[:-1]) / dt)
    dp_ref = dp_ref.at[-1].set(dp_ref[-2])

    grf_ref = jnp.zeros((N, 3 * n_contact))

    reference = jnp.concatenate(
        [
            p_ref,
            quat_ref,
            q_ref,
            dp_ref,
            omega_ref,
            foot_ref,
            contact_sequence,
            grf_ref,
        ],
        axis=1,
    )
    parameter = jnp.concatenate([contact_sequence, foot_ref], axis=1)
    return reference, parameter


def reference_humanoid_backflip(
    N,
    dt,
    n_joints,
    n_contact,
    foot0,
    q0,
    *,
    base_height=0.9,
    crouch_height=0.62,
    apex_height=1.5,
    backward_distance=-0.05,
    crouch_time=0.25,
    flight_time=0.9,
    land_time=0.25,
):
    """Backflip reference for an H1-class humanoid.

    Five segments: standing → deep crouch → flight (ballistic z + 2π
    pitch rotation about body-y) → land → settle. Body translates
    ``backward_distance`` in the world-x direction across the flip
    (slightly negative reads as "step backward as you flip").

    The omega_y profile ramps up across the second half of crouch (when
    feet are still grounded and can build angular momentum via foot
    impulse), holds during flight (no torque available — angular
    momentum conserved), then ramps back down across the first half of
    land. This matches the physics: a real backflip generates and
    dissipates angular momentum during contact, not at the boundary
    instant.

    The pitch trajectory passes through ``qw = -1`` at the half-flip
    point. SQP can handle this single sign-traversal (same as the
    single barrel roll); chaining flips would compound the quat-
    singularity issue and almost certainly diverge.
    """
    n_crouch = max(2, int(crouch_time / dt))
    n_flight = max(2, int(flight_time / dt))
    n_land = max(2, int(land_time / dt))
    n_settle = max(0, N - (n_crouch + n_flight + n_land))

    # Deep crouch: bend hips/knees harder than the jump-forward crouch
    # so the legs can store enough energy to launch + spin.
    crouch_q = q0.at[2].set(-1.4).at[3].set(2.5).at[4].set(-1.1)
    crouch_q = crouch_q.at[7].set(-1.4).at[8].set(2.5).at[9].set(-1.1)
    # Tuck arms during flight to reduce moment of inertia about y, helps
    # the body spin faster. The H1 arm joints are ordered (per limb,
    # 4 each): shoulder_pitch, shoulder_roll, shoulder_yaw, elbow.
    # A backflip is a sagittal-plane (about-y) maneuver, so the tuck
    # must stay strictly left/right mirror-symmetric — any roll- or
    # yaw-asymmetric reference encourages the optimizer to swing one
    # arm across the body's midline (which it then has to "uncross"
    # at landing, producing the visible discontinuity). Tuck via:
    #   * shoulder_pitch (11/15): from 0.5 → 2.6, swings both arms
    #     forward + up so they hug the chest as the body rotates
    #     backward overhead.
    #   * elbow (14/18): from 0.5 → 2.0, fully flex both elbows.
    # Shoulder roll (12/16) and shoulder yaw (13/17) are LEFT AT q0
    # so that the mirror-symmetric ±0.25 rad shoulder roll in q0 is
    # preserved. The earlier reference set both shoulder rolls to
    # +1.3, which flipped the right shoulder's sign and produced the
    # "right arm pointing left, then jumping back at landing"
    # artifact.
    tuck_q = crouch_q.at[11].set(2.6).at[15].set(2.6).at[14].set(2.0).at[18].set(2.0)

    crouch_alpha = jnp.linspace(0.0, 1.0, n_crouch)
    land_alpha = jnp.linspace(0.0, 1.0, n_land)
    flight_phase = jnp.linspace(0.0, 1.0, n_flight)

    # The peak omega_y. Total rotation = -2π (NEGATIVE about +y) must
    # equal the integral of omega over the spinning interval (last half
    # crouch + flight + first half land). With linear ramps on the
    # contact phases:
    #   -2π = ½·n_crouch·dt·ω_peak/2 + n_flight·dt·ω_peak + ½·n_land·dt·ω_peak/2
    # i.e. ω_peak = -2π / (n_flight·dt + (n_crouch + n_land)·dt/4).
    #
    # Sign matters: in mjx body frame (x forward, y left, z up),
    # R_y(+90°)·ẑ = +x̂ — head pitches FORWARD = front flip. We want
    # head to pitch BACKWARD on the way up, which is a NEGATIVE rotation
    # about +y. Hence the leading minus sign.
    spin_duration = n_flight * dt + (n_crouch + n_land) * dt / 4.0
    omega_peak = -2.0 * jnp.pi / spin_duration

    # ---- crouch ----
    x_c = jnp.zeros(n_crouch)
    z_c = jnp.linspace(base_height, crouch_height, n_crouch)
    p_c = jnp.stack([x_c, jnp.zeros_like(x_c), z_c], axis=1)
    q_c = jnp.stack([q0 + (crouch_q - q0) * a for a in crouch_alpha], axis=0)
    contact_c = jnp.tile(jnp.ones(n_contact), (n_crouch, 1))
    foot_c = jnp.tile(foot0, (n_crouch, 1))
    # Omega: zero in first half, ramps 0 → ω_peak in second half.
    crouch_ramp = jnp.clip((crouch_alpha - 0.5) * 2.0, 0.0, 1.0)
    omega_c = jnp.stack(
        [
            jnp.zeros(n_crouch),
            crouch_ramp * omega_peak,
            jnp.zeros(n_crouch),
        ],
        axis=1,
    )
    # Quat trajectory in crouch: integrate omega_c. Cumulative angle is
    # cumsum(omega_y * dt), starting from 0.
    crouch_angle = jnp.cumsum(omega_c[:, 1]) * dt
    quat_c = jnp.stack(
        [
            jnp.cos(crouch_angle / 2),
            jnp.zeros(n_crouch),
            jnp.sin(crouch_angle / 2),
            jnp.zeros(n_crouch),
        ],
        axis=1,
    )

    # ---- flight ----
    x_f = jnp.linspace(0.0, backward_distance, n_flight)
    z_f = (
        crouch_height
        + (base_height - crouch_height) * flight_phase
        + (apex_height - base_height) * 4.0 * flight_phase * (1.0 - flight_phase)
    )
    p_f = jnp.stack([x_f, jnp.zeros_like(x_f), z_f], axis=1)
    omega_f = jnp.tile(jnp.array([0.0, omega_peak, 0.0]), (n_flight, 1))
    flight_start_angle = crouch_angle[-1]
    flight_angle = flight_start_angle + jnp.arange(1, n_flight + 1) * dt * omega_peak
    quat_f = jnp.stack(
        [
            jnp.cos(flight_angle / 2),
            jnp.zeros(n_flight),
            jnp.sin(flight_angle / 2),
            jnp.zeros(n_flight),
        ],
        axis=1,
    )
    q_f = jnp.tile(tuck_q, (n_flight, 1))
    contact_f = jnp.tile(jnp.zeros(n_contact), (n_flight, 1))
    # Foot ref during flight: rotate the body-frame foot offsets by the
    # current pitch and add body position. Without this evolution the
    # ref says feet should stay at their starting world position while
    # the body flies through a 360° rotation overhead — a contradiction
    # that bullies the cost into either fighting the rotation or
    # ignoring foot tracking entirely.
    foot_local = (
        foot0 - jnp.tile(jnp.array([0.0, 0.0, base_height]), n_contact)
    ).reshape(n_contact, 3)
    cos_a = jnp.cos(flight_angle)
    sin_a = jnp.sin(flight_angle)
    # R_y(θ)·(x,y,z) = (x cosθ + z sinθ, y, -x sinθ + z cosθ)
    rot_x = (
        cos_a[:, None] * foot_local[None, :, 0]
        + sin_a[:, None] * foot_local[None, :, 2]
    )
    rot_y = jnp.broadcast_to(foot_local[None, :, 1], (n_flight, n_contact))
    rot_z = (
        -sin_a[:, None] * foot_local[None, :, 0]
        + cos_a[:, None] * foot_local[None, :, 2]
    )
    foot_world_flight = jnp.stack([rot_x, rot_y, rot_z], axis=-1)
    foot_world_flight = foot_world_flight + p_f[:, None, :]
    foot_f = foot_world_flight.reshape(n_flight, -1)

    # ---- land ----
    # z ramps from a deeper-than-crouch absorption pose at touchdown
    # back up to standing. Without this dip the reference asks the body
    # to be at standing height the instant the feet hit, which is
    # physically impossible with the inbound vertical momentum from the
    # flight — the optimizer will overshoot the apex and bottom out
    # somewhere unreasonable trying to satisfy an impossible target.
    x_l = jnp.full(n_land, backward_distance)
    z_l = jnp.linspace(crouch_height - 0.05, base_height, n_land)
    p_l = jnp.stack([x_l, jnp.zeros_like(x_l), z_l], axis=1)
    q_l = jnp.stack(
        [crouch_q + (q0 - crouch_q) * a for a in land_alpha],
        axis=0,
    )
    contact_l = jnp.tile(jnp.ones(n_contact), (n_land, 1))
    landed_foot = foot0.at[::3].set(foot0[::3] + backward_distance)
    foot_l = jnp.tile(landed_foot, (n_land, 1))
    # Omega: ramps ω_peak → 0 in first half, zero in second half.
    land_ramp = jnp.clip((0.5 - land_alpha) * 2.0, 0.0, 1.0)
    omega_l = jnp.stack(
        [
            jnp.zeros(n_land),
            land_ramp * omega_peak,
            jnp.zeros(n_land),
        ],
        axis=1,
    )
    flight_end_angle = flight_angle[-1]
    land_angle = flight_end_angle + jnp.cumsum(omega_l[:, 1]) * dt
    quat_l = jnp.stack(
        [
            jnp.cos(land_angle / 2),
            jnp.zeros(n_land),
            jnp.sin(land_angle / 2),
            jnp.zeros(n_land),
        ],
        axis=1,
    )

    # ---- final settle ----
    if n_settle > 0:
        p_s = jnp.tile(jnp.array([backward_distance, 0.0, base_height]), (n_settle, 1))
        q_s = jnp.tile(q0, (n_settle, 1))
        contact_s = jnp.tile(jnp.ones(n_contact), (n_settle, 1))
        foot_s = jnp.tile(landed_foot, (n_settle, 1))
        omega_s = jnp.zeros((n_settle, 3))
        # Sign-continuous quaternion path. End of land sits at
        # quat ≈ (-1, 0, 0, 0) (the equivalent representation of
        # identity orientation after a full -2π rotation about y).
        # Continuing settle with (+1, 0, 0, 0) jumps the quaternion to
        # the antipodal point on S³ — same orientation, but a
        # 4-vector-difference cost reads it as a maximal error step.
        # Tiling the end-of-land quaternion keeps the path continuous
        # so the direct (q - q_ref)² cost stays meaningful all the way
        # through the trajectory.
        quat_s = jnp.tile(quat_l[-1], (n_settle, 1))
        p_segs = [p_c, p_f, p_l, p_s]
        q_segs = [q_c, q_f, q_l, q_s]
        contact_segs = [contact_c, contact_f, contact_l, contact_s]
        foot_segs = [foot_c, foot_f, foot_l, foot_s]
        omega_segs = [omega_c, omega_f, omega_l, omega_s]
        quat_segs = [quat_c, quat_f, quat_l, quat_s]
    else:
        p_segs = [p_c, p_f, p_l]
        q_segs = [q_c, q_f, q_l]
        contact_segs = [contact_c, contact_f, contact_l]
        foot_segs = [foot_c, foot_f, foot_l]
        omega_segs = [omega_c, omega_f, omega_l]
        quat_segs = [quat_c, quat_f, quat_l]

    p_ref = jnp.concatenate(p_segs, axis=0)
    q_ref = jnp.concatenate(q_segs, axis=0)
    foot_ref = jnp.concatenate(foot_segs, axis=0)
    contact_sequence = jnp.concatenate(contact_segs, axis=0)
    quat_ref = jnp.concatenate(quat_segs, axis=0)
    omega_ref = jnp.concatenate(omega_segs, axis=0)

    dp_ref = jnp.zeros((N, 3))
    dp_ref = dp_ref.at[:-1].set((p_ref[1:] - p_ref[:-1]) / dt)
    dp_ref = dp_ref.at[-1].set(dp_ref[-2])

    grf_ref = jnp.zeros((N, 3 * n_contact))

    reference = jnp.concatenate(
        [
            p_ref,
            quat_ref,
            q_ref,
            dp_ref,
            omega_ref,
            foot_ref,
            contact_sequence,
            grf_ref,
        ],
        axis=1,
    )
    parameter = jnp.concatenate([contact_sequence, foot_ref], axis=1)
    return reference, parameter


def reference_quadruped_trot(
    N,
    dt,
    n_joints,
    n_contact,
    foot0,
    q0,
    *,
    n_phases=4,
    base_height=0.36,
    total_forward=0.45,
    step_length=0.16,
    step_height=0.08,
    settle_time=0.10,
    phase_time=0.16,
):
    """Forward trot with ``n_phases`` alternating diagonal-pair swings.

    The base translates from 0 to ``total_forward`` over the full trajectory
    while each foot pair lifts and steps forward by ``step_length`` once per
    phase. ``n_phases`` should generally be even so that both diagonals get
    equal swing time.
    """
    del n_joints
    n_stance = max(2, int(settle_time / dt))
    n_phase = max(2, int(phase_time / dt))
    n_settle = max(0, N - (n_stance + n_phases * n_phase))

    p_ref = jnp.zeros((N, 3))
    p_ref = p_ref.at[:, 0].set(jnp.linspace(0.0, total_forward, N))
    p_ref = p_ref.at[:, 2].set(base_height)
    quat_ref = jnp.tile(jnp.array([1.0, 0.0, 0.0, 0.0]), (N, 1))
    q_ref = jnp.tile(q0, (N, 1))
    dp_ref = jnp.zeros((N, 3))
    dp_ref = dp_ref.at[:, 0].set(total_forward / ((N - 1) * dt + 1e-6))
    omega_ref = jnp.zeros((N, 3))

    foot_ref = jnp.tile(foot0, (N, 1))
    footholds = foot0.reshape(n_contact, 3)
    contact_sequence = jnp.tile(jnp.ones(n_contact), (N, 1))

    trot_a = jnp.array([1.0, 0.0, 0.0, 1.0])
    trot_b = jnp.array([0.0, 1.0, 1.0, 0.0])
    patterns = [trot_a if i % 2 == 0 else trot_b for i in range(n_phases)]

    start_idx = n_stance
    for pattern in patterns:
        end_idx = min(start_idx + n_phase, N)
        contact_sequence = contact_sequence.at[start_idx:end_idx].set(
            jnp.tile(pattern, (end_idx - start_idx, 1))
        )
        swing_ids = jnp.where(pattern == 0.0)[0]
        phase = jnp.linspace(0.0, 1.0, end_idx - start_idx)
        start_feet = footholds
        end_feet = footholds.at[swing_ids, 0].add(step_length)
        swing_xyz = (
            start_feet[None, :, :]
            + (end_feet - start_feet)[None, :, :] * phase[:, None, None]
        )
        swing_xyz = swing_xyz.at[:, swing_ids, 2].set(
            start_feet[swing_ids, 2][None, :]
            + step_height * 4.0 * phase[:, None] * (1.0 - phase[:, None])
        )
        foot_ref = foot_ref.at[start_idx:end_idx].set(
            swing_xyz.reshape(end_idx - start_idx, -1)
        )
        footholds = end_feet
        start_idx = end_idx

    if start_idx < N:
        foot_ref = foot_ref.at[start_idx:].set(
            jnp.tile(footholds.reshape(-1), (N - start_idx, 1))
        )
        contact_sequence = contact_sequence.at[start_idx:].set(
            jnp.tile(jnp.ones(n_contact), (N - start_idx, 1))
        )

    grf_ref = jnp.zeros((N, 3 * n_contact))
    reference = jnp.concatenate(
        [
            p_ref,
            quat_ref,
            q_ref,
            dp_ref,
            omega_ref,
            foot_ref,
            contact_sequence,
            grf_ref,
        ],
        axis=1,
    )
    parameter = jnp.concatenate([contact_sequence, foot_ref], axis=1)
    return reference, parameter
