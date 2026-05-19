"""Shared builder for MJX-based ``ProblemSpec`` factories.

Every MJX problem (``mjx_barrel_roll``, ``mjx_backflip``,
``mjx_jump``, ``mjx_trot``) wraps an existing
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

    # Silence mujoco-mjx's "Failed to import warp / mujoco_warp" stdout
    # warnings — the NVIDIA Warp backend is an OPTIONAL accelerator for
    # MJX, and its absence is expected on CPU-only hosts (Docker on
    # macOS, GPU-less CI, etc.). The warnings print to stdout (not the
    # logging module) at module-import time, so we redirect stdout
    # around the import. The redirect doesn't suppress real errors —
    # only the cosmetic startup banner.
    import contextlib
    import io as _io
    import os as _os

    with contextlib.redirect_stdout(_io.StringIO()):
        import mujoco  # noqa: F401
        from mujoco import mjx  # noqa: F401
    # Re-import to bind the names in this scope (the with-block did the
    # heavy lifting already; re-import is a fast no-op since cached).
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
        mjx.name2id(mjx_model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in config.contact_frame
    ]
    body_id = [
        mjx.name2id(mjx_model, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in config.body_name
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
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)
        for n in config.contact_frame
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
    X_init = initial_X0.at[:, : 13 + config.n_joints].set(
        reference[:, : 13 + config.n_joints]
    )
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

    metadata: dict = {
        "is_mjx": True,
        # Per-problem success-tolerance override (uniform across all
        # solvers via ``pack_solver_result`` / ``effective_solver_tol``).
        "success_tol": 1e-3,
        # Per-solver iter caps on MJX problems to bound benchmark runtime.
        "max_iter_overrides": {"csqp": 200, "fatrop-jax": 500},
        "csqp_jax_settings": {
            # ProxQP relative/absolute tolerances matched to success_tol.
            "eps_abs": 1e-3,
            "eps_rel": 1e-3,
            "filter_size": 10,
        },
        "csqp_two_phase": True,
        # SIP defaults inherited from the historical sip_mjx adapter;
        # individual problems can override via dict.update below.
        "sip_settings": {
            "initial_penalty_parameter": 1e9,
            "initial_mu": 1e-2,
            "penalty_parameter_increase_factor": 1.1,
            "mu_update_factor": 0.95,
            "enable_line_search_failures": True,
            "max_ls_iterations": 100000,
        },
    }
    if enforce_ineq:
        # Phase 2: smooth cost, constrained solve.
        cost_smooth = getattr(config, "cost_smooth", None)
        if cost_smooth is not None:

            def lipa_cost(x, u, theta, t):  # noqa: ARG001
                return cost_smooth(config.W, reference, x, u, t)

        metadata["lipa_settings"] = enforce_settings
        metadata["trajax_settings"] = {
            "penalty_init": 1e3,
            "penalty_update_rate": 5.0,
            "alpha_min": 1e-8,
            "maxiter_al": 30,
        }
        if not skip_warmup:
            # Phase 1: original soft-penalty cost, no inequalities.
            # Wired via the shared two-phase orchestration in
            # run_benchmark.py; each solver opts in per-problem via the
            # flat ``<solver_root>_two_phase: True`` metadata flag and
            # can ship per-phase schedules via ``<solver>_warmup_settings``.
            original_cost = lambda x, u, theta, t: config.cost(  # noqa: ARG005, E731
                config.W, reference, x, u, t
            )
            metadata["lipa_two_phase"] = True
            metadata["aligator_two_phase"] = True
            metadata["warmup_cost"] = jax.jit(original_cost)
            metadata["lipa_warmup_settings"] = base_settings
            metadata["fatrop_mjx_settings"] = {
                "warm_start_init_point": True,
            }
            metadata["trajax_two_phase"] = True
            metadata["trajax_warmup_settings"] = {
                "max_iter": 50,
                "tol": 1e-3,
                "alpha_min": 1e-8,
            }
            metadata["aligator_warmup_settings"] = {
                "mu_init": 1.0,
                "max_al_iters": 10,
                "max_iters": 50,
            }
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
