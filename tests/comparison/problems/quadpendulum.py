"""Quadpendulum problem (mirror of tests/test_quadpendulum.py).

We support two modes via ``with_theta``:

* ``with_theta=False`` (default): drop the cross-stage ``Theta`` margin
  variable. The obstacle constraints become ``dist^2 >= (r1+r2)^2``
  (hard contact avoidance with no slack), and the cost loses the
  ``-w[3]*theta`` term. This is the "fair-comparison" formulation that
  every adapter can ingest.
* ``with_theta=True``: keep ``Theta`` as a single global decision
  variable. Only LIPA and IPOPT support this in the comparison driver
  (their NLP can include the extra var natively); CSQP/Aligator/acados
  would need extra plumbing.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from primal_dual_lipa.types import SolverSettings

from tests.comparison.problem_spec import ProblemSpec

# --- problem constants (mirror tests/test_quadpendulum.py) -----------------
T = 200
N_STATE = 8
N_CTRL = 2
DT = 0.025
MASS = 0.486
MASS_POLE = 0.2 * MASS
GRAV = 9.81
L_QUAD = 0.25  # quad half-arm
L_POLE = 2.0 * L_QUAD
J_QUAD = 0.00383
FRIC = 0.01
U_HOVER = 0.5 * (MASS + MASS_POLE) * GRAV * jnp.ones((N_CTRL,))

R_JOINT = 0.05 * L_QUAD
R_TIP = 0.1 * L_QUAD
R_T = 0.3 * L_QUAD

OBS = (
    (jnp.array([-1.0, 0.5]), 0.5),
    (jnp.array([0.75, -1.0]), 0.75),
    (jnp.array([-2.0, -1.0]), 0.5),
    (jnp.array([2.0, 1.0]), 0.5),
)
WORLD_LO = jnp.array([-4.0, -2.0])
WORLD_HI = jnp.array([4.0, 2.0])
THETA_LIM = 3.0 * jnp.pi / 4.0  # quad rotation cap
CTRL_LO = 0.1 * MASS * GRAV * jnp.ones((N_CTRL,))
CTRL_HI = 3.0 * MASS * GRAV * jnp.ones((N_CTRL,))

POS_0 = jnp.array([-2.5, 1.5, 0.0, 0.0])
POS_G = jnp.array([3.0, -1.5, 0.0, jnp.pi])
GOAL = jnp.concatenate((POS_G, jnp.zeros(4)))
WEIGHTS = jnp.array((0.01, 0.05, 5.0, 10.0))
QT = jnp.array((10.0, 10.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0))


# --- dynamics --------------------------------------------------------------
def _mass_matrix(q):
    phi = q[-1]
    a = MASS + MASS_POLE
    b = MASS_POLE * L_POLE * jnp.cos(phi)
    c = MASS_POLE * L_POLE * jnp.sin(phi)
    d = MASS_POLE * L_POLE * L_POLE
    return jnp.array([
        [a, 0.0, 0.0, b],
        [0.0, a, 0.0, c],
        [0.0, 0.0, J_QUAD, 0.0],
        [b, c, 0.0, d],
    ])


def _ode(x, u):
    q, qd = x[:4], x[4:]
    phi = q[-1]
    theta = q[2]
    M = _mass_matrix(q)

    # dM/dt times qd. Only entries depending on phi are non-zero.
    dphi = qd[-1]
    dM_dphi = jnp.array([
        [0.0, 0.0, 0.0, -MASS_POLE * L_POLE * jnp.sin(phi)],
        [0.0, 0.0, 0.0, MASS_POLE * L_POLE * jnp.cos(phi)],
        [0.0, 0.0, 0.0, 0.0],
        [-MASS_POLE * L_POLE * jnp.sin(phi), MASS_POLE * L_POLE * jnp.cos(phi), 0.0, 0.0],
    ])
    Mdot_qd = (dphi * dM_dphi) @ qd

    # dL/dq from Lagrangian: kinetic depends on phi via M(q); potential adds
    # gravitational terms in q[1] and q[-1].
    # d(KE)/dphi = 0.5 * qd^T * dM/dphi * qd
    dKE_dphi = 0.5 * jnp.dot(qd, dM_dphi @ qd)
    # d(PE)/dq[1] = (M + m) * g
    # d(PE)/dphi = m * g * L * sin(phi)
    dPE_dq1 = (MASS + MASS_POLE) * GRAV
    dPE_dphi = MASS_POLE * GRAV * L_POLE * jnp.sin(phi)
    dL_dq = jnp.array([0.0, -dPE_dq1, 0.0, dKE_dphi - dPE_dphi])

    torque_fric_pole = -FRIC * (qd[-1] - qd[-2])
    F_q = jnp.array([
        -jnp.sum(u) * jnp.sin(theta),
        jnp.sum(u) * jnp.cos(theta),
        (u[0] - u[1]) * L_QUAD - torque_fric_pole,
        torque_fric_pole,
    ])
    qdd = jnp.linalg.solve(M, F_q + dL_dq - Mdot_qd)
    return jnp.concatenate((qd, qdd))


def _dynamics(x, u, theta, t):  # noqa: ARG001
    return x + DT * _ode(x, u)


# --- cost ------------------------------------------------------------------
def _cost(x, u, theta, t):
    delta = x - GOAL
    pos_cost = jnp.dot(delta[:3], delta[:3]) + (1.0 + jnp.cos(x[3]))
    ctrl_cost = jnp.dot(u - U_HOVER, u - U_HOVER)
    stage_cost = WEIGHTS[0] * pos_cost + WEIGHTS[1] * ctrl_cost
    term_cost = WEIGHTS[2] * jnp.dot(delta, QT * delta)
    # `theta` is either empty (with_theta=False) or shape (1,). Use `sum` so
    # the empty case yields 0 without indexing into a length-0 array.
    margin_term = -WEIGHTS[3] * jnp.sum(theta)
    return jnp.where(t == T, 0.5 * term_cost, 0.5 * stage_cost) + jnp.where(t == 0, margin_term, 0.0)


# --- geometry: returns (centers, radii) of the body, plus pole endpoints ---
def _geometry(q):
    pos = q[:2]
    theta_quad = q[2]
    phi = q[-1]
    R = jnp.array([
        [jnp.cos(theta_quad), -jnp.sin(theta_quad)],
        [jnp.sin(theta_quad), jnp.cos(theta_quad)],
    ])
    pos_c = pos + R @ jnp.array([0.0, 0.15 * L_QUAD])
    pos_lt = pos + R @ jnp.array([-L_QUAD, 0.3 * L_QUAD])
    pos_rt = pos + R @ jnp.array([L_QUAD, 0.3 * L_QUAD])
    pole_tip = pos + jnp.array([L_POLE * jnp.sin(phi), -L_POLE * jnp.cos(phi)])
    centers = jnp.stack([pos, pos_c, pos_lt, pos_rt, pole_tip])
    radii = jnp.array([R_JOINT, L_QUAD, R_T, R_T, R_TIP])
    return centers, radii, pos, pole_tip


def _equalities(x, u, theta, t):  # noqa: ARG001
    return jnp.where(t == T, x - GOAL, jnp.zeros_like(x))


def _inequalities(x, u, theta, t):
    """Return concatenated inequality vector (g <= 0)."""
    # `theta` is empty (with_theta=False) or shape (1,) — sum handles both.
    margin = jnp.sum(theta)
    centers, radii, pos, pole_tip = _geometry(x[:4])

    # Quad rotation cap: |theta_quad| <= THETA_LIM
    theta_cons = jnp.array([x[2] - THETA_LIM, -x[2] - THETA_LIM])

    # World-bounds: lo + r <= c <= hi - r
    world_lo = (-centers + (WORLD_LO + radii[:, None])).flatten()
    world_hi = (centers - (WORLD_HI - radii[:, None])).flatten()

    # Obstacle avoidance: -(dist^2 - (R + margin)^2) <= 0 for each (body_circle, obstacle)
    obs_centers = jnp.stack([o[0] for o in OBS])
    obs_radii = jnp.array([o[1] for o in OBS])

    def _per_obs(oc, orad):
        # Body circles vs obstacle circle
        deltas = centers - oc[None, :]
        dist_sq = jnp.sum(deltas * deltas, axis=1)
        thresh_sq = (orad + radii + margin) ** 2
        circ_cons = -(dist_sq - thresh_sq)
        # Pole closest-point vs obstacle
        seg_dir = pole_tip - pos
        seg_norm_sq = jnp.dot(seg_dir, seg_dir) + 1e-12
        t_proj = jnp.clip(jnp.dot(oc - pos, seg_dir) / seg_norm_sq, 0.0, 1.0)
        closest = pos + t_proj * seg_dir
        delta_pole = closest - oc
        pole_dist_sq = jnp.dot(delta_pole, delta_pole)
        pole_thresh_sq = (orad + margin) ** 2
        pole_con = -(pole_dist_sq - pole_thresh_sq)
        return jnp.concatenate([circ_cons, jnp.array([pole_con])])

    obs_cons = jnp.concatenate([_per_obs(oc, orad) for oc, orad in zip(obs_centers, obs_radii)])

    # Control bounds at t < T; vacuous (-1) at t = T
    u_lo = jnp.where(t == T, -jnp.ones_like(u), CTRL_LO - u)
    u_hi = jnp.where(t == T, -jnp.ones_like(u), u - CTRL_HI)

    return jnp.concatenate([theta_cons, world_lo, world_hi, obs_cons, u_lo, u_hi])


# --- CasADi mirror ---------------------------------------------------------
def _casadi_builder_factory(with_theta: bool):
    def _builder(x_sx, u_sx, theta_sx, t):
        import casadi as ca

        margin = theta_sx[0] if with_theta else ca.SX(0.0)

        # Dynamics
        q = x_sx[:4]
        qd = x_sx[4:]
        phi = q[-1]
        theta_q = q[2]
        # Mass matrix
        a = float(MASS + MASS_POLE)
        b = MASS_POLE * L_POLE * ca.cos(phi)
        c = MASS_POLE * L_POLE * ca.sin(phi)
        d = float(MASS_POLE * L_POLE * L_POLE)
        M = ca.SX(4, 4)
        M[0, 0] = a; M[0, 3] = b
        M[1, 1] = a; M[1, 3] = c
        M[2, 2] = J_QUAD
        M[3, 0] = b; M[3, 1] = c; M[3, 3] = d

        # dM/dphi * qd
        dphi = qd[-1]
        dMdphi = ca.SX(4, 4)
        dMdphi[0, 3] = -MASS_POLE * L_POLE * ca.sin(phi)
        dMdphi[1, 3] = MASS_POLE * L_POLE * ca.cos(phi)
        dMdphi[3, 0] = -MASS_POLE * L_POLE * ca.sin(phi)
        dMdphi[3, 1] = MASS_POLE * L_POLE * ca.cos(phi)
        Mdot_qd = ca.mtimes(dphi * dMdphi, qd)

        # dL/dq
        dKE_dphi = 0.5 * ca.dot(qd, ca.mtimes(dMdphi, qd))
        dPE_dq1 = (MASS + MASS_POLE) * GRAV
        dPE_dphi = MASS_POLE * GRAV * L_POLE * ca.sin(phi)
        dL_dq = ca.vertcat(0.0, -dPE_dq1, 0.0, dKE_dphi - dPE_dphi)

        torque_fric = -FRIC * (qd[-1] - qd[-2])
        F_q = ca.vertcat(
            -(u_sx[0] + u_sx[1]) * ca.sin(theta_q),
            (u_sx[0] + u_sx[1]) * ca.cos(theta_q),
            (u_sx[0] - u_sx[1]) * L_QUAD - torque_fric,
            torque_fric,
        )
        qdd = ca.solve(M, F_q + dL_dq - Mdot_qd)
        next_x = x_sx + DT * ca.vertcat(qd, qdd)

        # Cost
        delta = x_sx - ca.DM(np.asarray(GOAL))
        pos_cost = ca.dot(delta[:3], delta[:3]) + (1.0 + ca.cos(x_sx[3]))
        ctrl_cost = ca.dot(u_sx - ca.DM(np.asarray(U_HOVER)), u_sx - ca.DM(np.asarray(U_HOVER)))
        stage_cost = WEIGHTS[0] * pos_cost + WEIGHTS[1] * ctrl_cost
        term_cost = WEIGHTS[2] * ca.dot(delta, ca.DM(np.asarray(QT)) * delta)
        f = 0.5 * (term_cost if t == T else stage_cost)
        if with_theta and t == 0:
            f = f - WEIGHTS[3] * margin

        eq = delta if t == T else None

        # Inequalities
        # Geometry
        pos = q[:2]
        Rmat = ca.SX(2, 2)
        Rmat[0, 0] = ca.cos(theta_q); Rmat[0, 1] = -ca.sin(theta_q)
        Rmat[1, 0] = ca.sin(theta_q); Rmat[1, 1] = ca.cos(theta_q)
        offsets = ca.SX(5, 2)
        offsets[0, :] = ca.SX([0.0, 0.0])
        offsets[1, :] = ca.SX([0.0, 0.15 * L_QUAD])
        offsets[2, :] = ca.SX([-L_QUAD, 0.3 * L_QUAD])
        offsets[3, :] = ca.SX([L_QUAD, 0.3 * L_QUAD])
        centers = ca.SX(5, 2)
        for i in range(4):
            local = ca.vertcat(offsets[i, 0], offsets[i, 1])
            world = pos + ca.mtimes(Rmat, local)
            centers[i, 0] = world[0]
            centers[i, 1] = world[1]
        pole_tip = ca.vertcat(pos[0] + L_POLE * ca.sin(phi), pos[1] - L_POLE * ca.cos(phi))
        centers[4, 0] = pole_tip[0]
        centers[4, 1] = pole_tip[1]
        radii = ca.DM([R_JOINT, L_QUAD, R_T, R_T, R_TIP])

        # Theta cap
        theta_cons = ca.vertcat(theta_q - THETA_LIM, -theta_q - THETA_LIM)
        # World bounds
        world_lo_pieces = []
        world_hi_pieces = []
        for i in range(5):
            world_lo_pieces.append(-centers[i, 0] + WORLD_LO[0] + radii[i])
            world_lo_pieces.append(-centers[i, 1] + WORLD_LO[1] + radii[i])
            world_hi_pieces.append(centers[i, 0] - WORLD_HI[0] + radii[i])
            world_hi_pieces.append(centers[i, 1] - WORLD_HI[1] + radii[i])
        world_lo_v = ca.vertcat(*world_lo_pieces)
        world_hi_v = ca.vertcat(*world_hi_pieces)

        # Obstacle avoidance
        obs_pieces = []
        for ob in OBS:
            oc = ca.DM(np.asarray(ob[0]))
            orad = float(ob[1])
            for i in range(5):
                cx = centers[i, 0] - oc[0]
                cy = centers[i, 1] - oc[1]
                dist_sq = cx * cx + cy * cy
                thresh = (orad + radii[i] + margin) ** 2
                obs_pieces.append(-(dist_sq - thresh))
            # Pole closest point
            seg = pole_tip - pos
            seg_norm_sq = seg[0] * seg[0] + seg[1] * seg[1] + 1e-12
            tproj = ca.dot(oc - pos, seg) / seg_norm_sq
            tproj_cl = ca.fmin(1.0, ca.fmax(0.0, tproj))
            closest = pos + tproj_cl * seg
            dx = closest[0] - oc[0]
            dy = closest[1] - oc[1]
            pole_dist_sq = dx * dx + dy * dy
            pole_thresh = (orad + margin) ** 2
            obs_pieces.append(-(pole_dist_sq - pole_thresh))
        obs_v = ca.vertcat(*obs_pieces)

        if t == T:
            u_lo = -ca.DM.ones(N_CTRL)
            u_hi = -ca.DM.ones(N_CTRL)
        else:
            u_lo = ca.DM(np.asarray(CTRL_LO)) - u_sx
            u_hi = u_sx - ca.DM(np.asarray(CTRL_HI))

        ineq = ca.vertcat(theta_cons, world_lo_v, world_hi_v, obs_v, u_lo, u_hi)
        return {"f": f, "next_x": next_x, "eq": eq, "ineq": ineq}

    return _builder


def _ineq_dim() -> int:
    """Compute the per-stage ineq vector length (must match _inequalities/CasADi)."""
    # theta_cons (2) + world_lo (5*2=10) + world_hi (10) + obs (4 obstacles * (5 circles + 1 pole) = 24) + u_lo (2) + u_hi (2)
    return 2 + 10 + 10 + 4 * (5 + 1) + 2 + 2


def make_problem(with_theta: bool = False) -> ProblemSpec:
    x0 = jnp.concatenate((POS_0, jnp.zeros(4)))
    # The tile(x0) warm start lands every IPM in a high-cost basin;
    # linspace(x0, GOAL) takes them all to the same lower-cost basin.
    X_init = jnp.linspace(x0, GOAL, T + 1)
    U_init = jnp.tile(U_HOVER, (T, 1))
    if with_theta:
        Theta_init = jnp.array([0.0])
        theta_dim = 1
    else:
        Theta_init = jnp.empty(0)
        theta_dim = 0

    name = "quadpendulum_theta" if with_theta else "quadpendulum"

    # The with_theta variant has a narrow basin; the no-theta variant
    # tolerates a more aggressive (κ, η0, η_update) combination.
    if with_theta:
        lipa_settings = SolverSettings(
            max_iterations=2000,
            residual_sq_threshold=1e-8,
            η0=10.0,
            η_max=1e9,
            η_update_factor=1.1,
            µ0=0.1,
            µ_update_factor=0.9,
            µ_min=1e-16,
            num_iterative_refinement_steps=0,
            skip_line_search=True,
            print_logs=False,
        )
    else:
        lipa_settings = SolverSettings(
            max_iterations=2000,
            residual_sq_threshold=1e-8,
            η0=1.0,
            η_max=1e9,
            η_update_factor=1.18,
            µ0=0.1,
            µ_update_factor=0.9,
            µ_min=1e-16,
            κ=100.0,
            num_iterative_refinement_steps=0,
            skip_line_search=True,
            print_logs=False,
        )

    # SIP settings — broadly mirror the LIPA fields. The with_theta
    # variant additionally needs a smaller initial_mu, line search on,
    # LS failures absorbed, and an eta-decrease factor < 1 to avoid
    # saturating at max_penalty_parameter.
    if with_theta:
        sip_settings = dict(
            max_iterations=2000,
            initial_penalty_parameter=10.0,
            max_penalty_parameter=1e9,
            penalty_parameter_increase_factor=1.1,
            penalty_parameter_decrease_factor=0.9,
            initial_mu=0.01,
            mu_update_factor=0.9,
            mu_min=1e-16,
            mu_update_kappa=100.0,
            num_iterative_refinement_steps=0,
            skip_line_search=False,
            enable_line_search_failures=True,
            max_ls_iterations=5000,
        )
    else:
        sip_settings = dict(
            max_iterations=2000,
            initial_penalty_parameter=1.0,
            max_penalty_parameter=1e9,
            penalty_parameter_increase_factor=1.18,
            initial_mu=0.1,
            mu_update_factor=0.9,
            mu_min=1e-16,
            mu_update_kappa=100.0,
            num_iterative_refinement_steps=0,
            skip_line_search=True,
        )

    return ProblemSpec(
        name=name,
        T=T,
        n=N_STATE,
        m=N_CTRL,
        theta_dim=theta_dim,
        x0=x0,
        cost=jax.jit(_cost),
        dynamics=jax.jit(_dynamics),
        equalities=jax.jit(_equalities),
        inequalities=jax.jit(_inequalities),
        eq_dim=N_STATE,
        ineq_dim=_ineq_dim(),
        X_init=X_init,
        U_init=U_init,
        Theta_init=Theta_init,
        metadata={
            "casadi_builder": _casadi_builder_factory(with_theta),
            "lipa_settings": lipa_settings,
            "sip_settings": sip_settings,
        },
    )
