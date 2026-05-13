"""CLI entry point: solve one offline MJX MPC task with LIPA, optionally render output.

Provenance: CLI shape and overall solve→render flow inspired by
``mpx/examples/offline_task.py`` —
https://github.com/iit-DLSLab/mpx (BSD-3-Clause). Specialized to the
LIPA solver here.

Usage (from repo root):

    uv run --extra mpc-examples python -m tests.mpc_examples.run_offline \\
        --task barrel_roll --png barrel_roll.png

Tasks: ``barrel_roll``, ``aliengo_trot``, ``h1_jump_forward``,
``h1_backflip``.

If ``--video`` or ``--png`` is passed, the script picks a sensible
``MUJOCO_GL`` backend before importing mujoco. You can override the
auto-pick by exporting ``MUJOCO_GL`` yourself; we honor whatever you
already set.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys


def _default_mujoco_gl() -> str:
    """Pick a default ``MUJOCO_GL`` for headless offscreen rendering.

    Per platform:

    * macOS  → ``cgl`` (Apple's Core OpenGL — always available).
    * Windows → ``wgl`` (the Windows OpenGL backend).
    * Linux with no display (``DISPLAY`` / ``WAYLAND_DISPLAY`` unset) →
      ``egl`` (NVIDIA's headless backend, the standard headless-Linux
      choice).
    * Linux with a display → ``glfw`` (a windowed backend that doesn't
      require EGL libraries; works on the typical desktop install).

    All of this is overridable: if the user has already exported
    ``MUJOCO_GL``, we leave it alone (``setdefault`` semantics).
    """
    if sys.platform == "darwin":
        return "cgl"
    if sys.platform.startswith("win"):
        return "wgl"
    # Linux (and other Unixes) — distinguish headless vs display.
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        return "glfw"
    return "egl"


# Headless rendering requires a working GL backend BEFORE the first
# `import mujoco`. Pick a platform-appropriate default when --video or
# --png is requested; honor any pre-existing MUJOCO_GL override.
if "--video" in sys.argv or "--png" in sys.argv:
    os.environ.setdefault("MUJOCO_GL", _default_mujoco_gl())
    # PYOPENGL_PLATFORM only matters for EGL; setting it to a non-EGL
    # value confuses pyopengl on some installs.
    if os.environ.get("MUJOCO_GL") == "egl":
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
# JIT compile cache reuse across repeated runs.
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

# Note: jax x64 is enabled in tests.mpc_examples.__init__.py because LIPA
# needs float64 sentinels — see the comment there.
from tests.mpc_examples.offline_solver import (
    lipa_pick_cost_and_inequalities,
    run_lipa_offline,
)

jax.config.update("jax_compilation_cache_dir", "./jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)


TASKS = {
    "barrel_roll": "tests.mpc_examples.configs.config_barrel_roll",
    "h1_jump_forward": "tests.mpc_examples.configs.config_h1_jump_forward",
    "h1_backflip": "tests.mpc_examples.configs.config_h1_backflip",
    "aliengo_trot": "tests.mpc_examples.configs.config_aliengo_trot",
}


def _load_config(task_name: str):
    return importlib.import_module(TASKS[task_name])


def _foot_positions(config) -> jnp.ndarray:
    """MuJoCo forward-kinematics-derived initial foot positions for the warm start."""
    model = mujoco.MjModel.from_xml_path(config.model_path)
    data = mujoco.MjData(model)
    qpos = jnp.concatenate([config.p0, config.quat0, config.q0])
    data.qpos = np.asarray(qpos)
    mujoco.mj_kinematics(model, data)
    contact_id_mj = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in config.contact_frame
    ]
    return jnp.array([data.geom_xpos[idx] for idx in contact_id_mj]).flatten()


def solve_task(task_name: str, *, max_iter: int = 100, verbose: bool = True) -> dict:
    """Build the MJX model, generate the reference, run LIPA offline. Returns the result dict."""
    config = _load_config(task_name)

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

    # Wire up dynamics with model + ids.
    dyn = config.dynamics(model, mjx_model, contact_id, body_id)

    # Generate the reference + parameter (contact pattern, foothold targets).
    reference, parameter = config.reference(
        config.N + 1,
        config.dt,
        config.n_joints,
        config.n_contact,
        config.p_legs0,
        config.q0,
    )

    # Initial state: nominal configuration + measured feet from FK.
    qpos0 = jnp.concatenate([config.p0, config.quat0, config.q0])
    qvel0 = jnp.zeros(6 + config.n_joints)
    foot_op = _foot_positions(config)
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

    # Warm start: tile initial_state, then overwrite the kinematic prefix
    # (position/orientation/joint angles) from the reference. Same as mpx
    # does in MPCWrapper.runOffline.
    initial_X0 = jnp.tile(config.initial_state, (config.N + 1, 1))
    X0 = initial_X0.at[:, : 13 + config.n_joints].set(
        reference[:, : 13 + config.n_joints]
    )
    U0 = jnp.tile(config.u_ref, (config.N, 1))
    V0 = jnp.zeros((config.N + 1, config.n))

    lipa_cost, lipa_ineq, lipa_settings, lipa_warmup_cost, lipa_warmup_settings = (
        lipa_pick_cost_and_inequalities(config, config.cost)
    )

    X, U, V, history, stats = run_lipa_offline(
        lipa_cost,
        dyn,
        reference,
        parameter,
        config.W,
        x0,
        X0,
        U0,
        V0,
        settings=lipa_settings,
        inequalities=lipa_ineq,
        warmup_cost=lipa_warmup_cost,
        warmup_settings=lipa_warmup_settings,
        verbose=verbose,
    )

    return {
        "task_name": task_name,
        "config": config,
        "scene_path": getattr(config, "scene_path", config.model_path),
        "X": X,
        "U": U,
        "V": V,
        "reference": reference,
        "history": history,
        "stats": stats,
        "initial_state": {"qpos0": qpos0, "qvel0": qvel0},
    }


def _print_summary(result: dict) -> None:
    stats = result["stats"]
    enforce = getattr(result["config"], "lipa_enforce_inequalities", False)
    enforce_tag = " | enforce-ineq" if enforce else ""
    print(
        f"{result['task_name']} | LIPA{enforce_tag} | "
        f"iterations {stats['n_iterations']} | "
        f"avg iter time {stats['average_iteration_time_ms']:.3f} ms | "
        f"final cost {stats['final_objective']:.4f} | "
        f"defect {stats['final_dynamics_violation']:.2e}"
    )


def run_task(
    task_name: str,
    *,
    headless: bool = True,
    max_iter: int = 100,
    verbose: bool = True,
    video: str | None = None,
    png: str | None = None,
) -> dict:
    result = solve_task(task_name, max_iter=max_iter, verbose=verbose)
    _print_summary(result)

    if png is not None:
        from tests.mpc_examples.viz import record_trajectory_png

        record_trajectory_png(result, png)
    if video is not None:
        from tests.mpc_examples.viz import record_trajectory_video

        record_trajectory_video(result, video)

    if headless and png is None and video is None:
        X = np.asarray(result["X"])
        print(
            f"Solve shapes: X={X.shape}, U={np.asarray(result['U']).shape}, "
            f"reference={np.asarray(result['reference']).shape}"
        )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--task", choices=tuple(TASKS.keys()), required=True)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Render the optimised trajectory to this mp4 path.",
    )
    parser.add_argument(
        "--png",
        type=str,
        default=None,
        help="Render a stroboscopic pose overlay PNG to this path.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_task(
        args.task,
        max_iter=args.max_iter,
        verbose=not args.quiet,
        video=args.video,
        png=args.png,
    )


if __name__ == "__main__":
    main()
