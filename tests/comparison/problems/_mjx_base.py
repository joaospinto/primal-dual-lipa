"""Shared builder for MJX-based ``ProblemSpec`` factories.

Every MJX problem (``mjx_barrel_roll``, ``mjx_h1_backflip``,
``mjx_h1_jump_forward``, ``mjx_aliengo_trot``) wraps an existing
``tests.mpc_examples.configs.config_*`` module the same way, so the
construction lives here once. Per-problem files supply only the config
module path, a display name, and any per-problem ``metadata`` tweaks.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from tests.comparison.problem_spec import ProblemSpec


def build_mjx_problem(config_module: str, name: str) -> ProblemSpec:
    """Build a ProblemSpec from one of the existing MJX configs."""
    import importlib

    config = importlib.import_module(config_module)

    import mujoco
    from mujoco import mjx

    from tests.mpc_examples import fetch_assets

    # Trigger asset fetch (mirror of run_offline._ensure_assets).
    robots = []
    if "aliengo" in config.model_path:
        robots.append("aliengo")
    if "unitree_h1" in config.model_path:
        robots.append("unitree_h1")
    if robots:
        fetch_assets.fetch(robots)

    model = mujoco.MjModel.from_xml_path(config.model_path)
    mjx_model = mjx.put_model(model)

    contact_id = [
        mjx.name2id(mjx_model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in config.contact_frame
    ]
    body_id = [
        mjx.name2id(mjx_model, mujoco.mjtObj.mjOBJ_BODY, name) for name in config.body_name
    ]

    dyn = config.dynamics(model, mjx_model, contact_id, body_id)

    reference, parameter = config.reference(
        config.N + 1,
        config.dt,
        config.n_joints,
        config.n_contact,
        config.p_legs0,
        config.q0,
    )

    # Initial state: nominal configuration + measured feet from FK
    qpos0 = jnp.concatenate([config.p0, config.quat0, config.q0])
    qvel0 = jnp.zeros(6 + config.n_joints)

    data = mujoco.MjData(model)
    data.qpos = np.asarray(qpos0)
    mujoco.mj_kinematics(model, data)
    contact_id_mj = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in config.contact_frame
    ]
    foot_op = jnp.array([data.geom_xpos[idx] for idx in contact_id_mj]).flatten()

    qpos_dim = 7 + config.n_joints
    qvel_stop = qpos_dim + 6 + config.n_joints
    foot_stop = qvel_stop + 3 * config.n_contact

    x0 = (
        config.initial_state.at[:qpos_dim]
        .set(qpos0)
        .at[qpos_dim:qvel_stop]
        .set(qvel0)
        .at[qvel_stop:foot_stop]
        .set(foot_op)
    )

    initial_X0 = jnp.tile(config.initial_state, (config.N + 1, 1))
    X_init = initial_X0.at[:, : 13 + config.n_joints].set(reference[:, : 13 + config.n_joints])
    U_init = jnp.tile(config.u_ref, (config.N, 1))

    # LIPA-style stage signature: (x, u, theta, t).
    # The MJX cost / dyn / ineq have their own signatures — wrap them.
    def lipa_cost(x, u, theta, t):  # noqa: ARG001
        return config.cost(config.W, reference, x, u, t)

    def lipa_dyn(x, u, theta, t):  # noqa: ARG001
        return dyn(x, u, t, parameter=parameter)

    def lipa_ineq(x, u, theta, t):  # noqa: ARG001
        return config.inequalities(reference, x, u, t)

    # Probe ineq dim so we can size LIPA's slack arrays correctly.
    ineq_probe = lipa_ineq(x0, U_init[0], jnp.empty(0), jnp.int32(0))
    ineq_dim = int(ineq_probe.shape[0])

    # LIPA settings: mirror the two-phase scheme from the existing
    # tests/mpc_examples/offline_solver.run_lipa_offline driver.
    base_settings = config.lipa_settings
    enforce_ineq = getattr(config, "lipa_enforce_inequalities", False)
    skip_warmup = getattr(config, "lipa_skip_warmup_phase", False)
    enforce_settings = getattr(config, "lipa_settings_enforce", None) or base_settings

    metadata = {"is_mjx": True}
    if enforce_ineq:
        # Phase 2: smooth cost, constrained solve.
        cost_smooth = getattr(config, "cost_smooth", None)
        if cost_smooth is not None:
            def lipa_cost(x, u, theta, t):  # noqa: ARG001
                return cost_smooth(config.W, reference, x, u, t)
        metadata["lipa_settings"] = enforce_settings
        if not skip_warmup:
            # Phase 1: original soft-penalty cost, no inequalities.
            original_cost = lambda x, u, theta, t: config.cost(  # noqa: ARG005, E731
                config.W, reference, x, u, t
            )
            metadata["lipa_warmup_cost"] = jax.jit(original_cost)
            metadata["lipa_warmup_settings"] = base_settings
    else:
        metadata["lipa_settings"] = base_settings

    return ProblemSpec(
        name=name,
        T=config.N,
        n=config.n,
        m=config.m,
        theta_dim=0,
        x0=x0,
        cost=jax.jit(lipa_cost),
        dynamics=jax.jit(lipa_dyn),
        equalities=None,
        inequalities=jax.jit(lipa_ineq) if enforce_ineq else None,
        eq_dim=0,
        ineq_dim=ineq_dim if enforce_ineq else 0,
        X_init=X_init,
        U_init=U_init,
        Theta_init=jnp.empty(0),
        metadata=metadata,
    )
