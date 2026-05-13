"""Headless visualization helpers for offline MPC trajectories.

Provenance: lifted from ``mpx/examples/offline_task.py`` —
https://github.com/iit-DLSLab/mpx (BSD-3-Clause). The rendering path
only — the solver bits live in :mod:`tests.mpc_examples.offline_solver`.

Provides:

* ``state_to_mujoco`` — split the LIPA state vector into ``(qpos, qvel)``.
* ``record_trajectory_video`` — one frame per state to mp4.
* ``record_trajectory_png`` — stroboscopic pose overlay PNG.

For interactive viewing in ``mujoco.viewer.launch_passive`` use
``run_offline.py --no-loop`` and point that script at one of the configs.
This module is import-time dependent on ``mujoco`` / ``matplotlib``.
"""

from __future__ import annotations

import os
from typing import Iterable

import jax.numpy as jnp
import mujoco
import numpy as np


def state_to_mujoco(config, state):
    """Split a LIPA state vector into ``(qpos, qvel)`` numpy arrays.

    Configs may override the layout via ``state_to_qpos`` / ``state_to_qvel``
    callables; otherwise we assume the canonical floating-base layout
    ``[qpos | qvel | (rest)]`` with ``qpos`` of size ``7 + n_joints``.
    """
    state = jnp.asarray(state)
    if hasattr(config, "state_to_qpos"):
        return (
            np.asarray(config.state_to_qpos(state)),
            np.asarray(config.state_to_qvel(state)),
        )

    qpos_dim = 7 + config.n_joints
    qvel_start = qpos_dim
    qvel_stop = qvel_start + 6 + config.n_joints
    return (
        np.asarray(state[:qpos_dim]),
        np.asarray(state[qvel_start:qvel_stop]),
    )


class _Recorder:
    """Tiny offscreen mp4 writer wrapping ``mujoco.Renderer`` + imageio.

    Lifted (slimmed) from mpx's VideoRecorder. Tracks ``qpos[:3]`` by
    default — the floating-base world position.
    """

    def __init__(
        self,
        model,
        path,
        fps=30,
        width=640,
        height=480,
        distance=3.0,
        azimuth=90.0,
        elevation=-20.0,
    ):
        import imageio

        self._renderer = mujoco.Renderer(model, height=height, width=width)
        self._writer = imageio.get_writer(
            path,
            format="FFMPEG",
            codec="libx264",
            fps=fps,
            macro_block_size=1,
        )
        self._cam = mujoco.MjvCamera()
        self._cam.distance = float(distance)
        self._cam.azimuth = float(azimuth)
        self._cam.elevation = float(elevation)
        self._cam.lookat[:] = [0.0, 0.0, 0.0]

    def capture(self, data):
        qpos = np.asarray(data.qpos, dtype=np.float64)
        if qpos.size >= 3:
            self._cam.lookat[:] = qpos[:3]
        else:
            ref = data.xpos[1] if data.xpos.shape[0] > 1 else data.xpos[0]
            self._cam.lookat[:] = np.asarray(ref, dtype=np.float64).reshape(3)
        self._renderer.update_scene(data, self._cam)
        self._writer.append_data(self._renderer.render())

    def close(self):
        try:
            self._writer.close()
        finally:
            self._renderer.close()


def record_trajectory_video(result, video_path, fps=None):
    """Render ``result["X"]`` to mp4 with player-safe real-time playback.

    The trajectory is resampled in time to the target ``fps`` so that one
    second of video equals one second of simulated trajectory, regardless
    of the underlying sim ``dt``. Default ``fps`` is ``min(60, round(1/dt))``
    — capping at 60 because many players (browsers, QuickTime) silently
    cap above that and would otherwise render the video in slow-motion.
    """
    config = result["config"]
    scene_path = result["scene_path"]
    X = np.asarray(result["X"])
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)
    parent = os.path.dirname(os.path.abspath(video_path)) or "."
    os.makedirs(parent, exist_ok=True)

    sim_dt = float(config.dt)
    if fps is None:
        fps = min(60, int(round(1.0 / sim_dt)))
    fps = int(fps)

    # Time-based nearest-neighbor resample. Total trajectory length is
    # (T-1) * dt seconds for T sim states; emit ceil(total_time * fps) + 1
    # video frames at that fps so duration matches.
    total_time = max(0.0, (X.shape[0] - 1) * sim_dt)
    n_video_frames = max(1, int(round(total_time * fps)) + 1)

    rec = _Recorder(model, video_path, fps=fps)
    try:
        for i in range(n_video_frames):
            t_sec = i / fps if n_video_frames > 1 else 0.0
            sim_idx = min(X.shape[0] - 1, int(round(t_sec / sim_dt)))
            qpos, qvel = state_to_mujoco(config, X[sim_idx])
            data.qpos = np.asarray(qpos)
            data.qvel = np.asarray(qvel)
            mujoco.mj_forward(model, data)
            rec.capture(data)
    finally:
        rec.close()
        print(f"Wrote video: {video_path} ({n_video_frames} frames @ {fps} fps)")


def _pick_overlay_poses(config, X, default_n=6) -> Iterable[tuple[int, float]]:
    """Pick (frame_idx, alpha) pairs for the stroboscopic overlay.

    For floating-base trajectories that fully flip the base (base z-axis
    dips below 0 — barrel_roll), use just the initial pose plus the apex
    of the flip. Otherwise pick ``default_n`` evenly-spaced poses with
    linearly increasing alpha (earliest faintest, latest solid).
    """
    qpos_all = np.array([state_to_mujoco(config, x)[0] for x in X])
    T = X.shape[0]

    if qpos_all.shape[1] >= 7:
        qx = qpos_all[:, 4]
        qy = qpos_all[:, 5]
        up_z = 1.0 - 2.0 * (qx**2 + qy**2)
        if up_z.min() < -0.5:
            apex = int(np.argmin(up_z))
            return [(0, 1.0), (apex, 0.5)]

    frame_idx = np.linspace(0, T - 1, default_n).astype(int)
    alphas = np.linspace(0.30, 1.00, default_n)
    return list(zip(frame_idx, alphas))


def _render_overlay_pose(config, scene_path, X, pose_specs, width=640, height=480):
    """Stroboscopic composite: one ghost robot per (frame_idx, alpha)."""
    pose_specs = list(pose_specs)
    frame_idx = [int(idx) for idx, _ in pose_specs]
    pose_alphas = [float(a) for _, a in pose_specs]
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)

    robot_geom_ids = np.where(model.geom_bodyid != 0)[0]
    max_geom = 2 * model.ngeom + len(robot_geom_ids) * len(frame_idx) + 100
    renderer = mujoco.Renderer(model, height=height, width=width, max_geom=max_geom)
    cam = mujoco.MjvCamera()

    centers = []
    for idx in frame_idx:
        qpos, _ = state_to_mujoco(config, X[idx])
        qpos_arr = np.asarray(qpos, dtype=np.float64)
        if qpos_arr.size >= 3:
            centers.append(qpos_arr[:3])
        else:
            data.qpos = qpos_arr
            mujoco.mj_forward(model, data)
            centers.append(
                np.asarray(data.xpos[1] if data.xpos.shape[0] > 1 else data.xpos[0])
            )
    centers = np.asarray(centers)
    center = centers.mean(axis=0)
    spread = float(np.linalg.norm(np.ptp(centers, axis=0)))
    cam.distance = float(np.clip(1.5 * (spread + model.stat.extent), 1.5, 8.0))
    cam.azimuth = 90.0
    cam.elevation = -10.0
    cam.lookat[:] = center

    saved_alphas = model.geom_rgba[robot_geom_ids, 3].copy()
    try:
        model.geom_rgba[robot_geom_ids, 3] = 0.0
        qpos, qvel = state_to_mujoco(config, X[frame_idx[0]])
        data.qpos = np.asarray(qpos)
        data.qvel = np.asarray(qvel)
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, cam)
    finally:
        model.geom_rgba[robot_geom_ids, 3] = saved_alphas

    spec_scene = mujoco.MjvScene(model, maxgeom=max(2 * model.ngeom, 200))
    spec_data = mujoco.MjData(model)
    mujoco.mj_forward(model, spec_data)
    mujoco.mjv_updateScene(
        model,
        spec_data,
        mujoco.MjvOption(),
        None,
        mujoco.MjvCamera(),
        mujoco.mjtCatBit.mjCAT_ALL,
        spec_scene,
    )
    geom_specs = {}
    for i in range(spec_scene.ngeom):
        g = spec_scene.geoms[i]
        if g.segid == -1 or model.geom_bodyid[int(g.objid)] == 0:
            continue
        geom_specs[int(g.objid)] = {
            "type": int(g.type),
            "size": np.array(g.size, copy=True),
            "rgba": np.array(g.rgba, copy=True),
            "dataid": int(g.dataid),
            "emission": float(g.emission),
            "specular": float(g.specular),
            "shininess": float(g.shininess),
        }

    scene = renderer.scene
    scratch = mujoco.MjData(model)
    try:
        for alpha, idx in zip(pose_alphas, frame_idx):
            qpos, qvel = state_to_mujoco(config, X[idx])
            scratch.qpos[:] = qpos
            scratch.qvel[:] = qvel
            mujoco.mj_forward(model, scratch)
            for geom_id, spec in geom_specs.items():
                if scene.ngeom >= scene.maxgeom:
                    break
                new_geom = scene.geoms[scene.ngeom]
                scene.ngeom += 1
                rgba = np.array(spec["rgba"], copy=True)
                rgba[3] = alpha
                mujoco.mjv_initGeom(
                    new_geom,
                    type=spec["type"],
                    size=spec["size"],
                    pos=np.asarray(scratch.geom_xpos[geom_id], dtype=np.float64),
                    mat=np.asarray(
                        scratch.geom_xmat[geom_id], dtype=np.float64
                    ).reshape(9),
                    rgba=rgba.astype(np.float32),
                )
                new_geom.dataid = spec["dataid"]
                new_geom.emission = spec["emission"]
                new_geom.specular = spec["specular"]
                new_geom.shininess = spec["shininess"]
                new_geom.category = mujoco.mjtCatBit.mjCAT_DECOR
                new_geom.segid = -1
                new_geom.objid = -1
        return renderer.render().copy()
    finally:
        renderer.close()


def record_trajectory_png(result, png_path, n_overlay=6):
    """Render a stroboscopic pose overlay PNG with summary title."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    config = result["config"]
    scene_path = result["scene_path"]
    X = np.asarray(result["X"])
    stats = result["stats"]
    task_name = result["task_name"]
    dt = float(config.dt)

    parent = os.path.dirname(os.path.abspath(png_path)) or "."
    os.makedirs(parent, exist_ok=True)

    pose_specs = _pick_overlay_poses(config, X, default_n=n_overlay)
    composite = _render_overlay_pose(config, scene_path, X, pose_specs)
    frame_idx = [idx for idx, _ in pose_specs]

    h, w = composite.shape[:2]
    fig = plt.figure(figsize=(w / 80.0, h / 80.0 + 0.6))
    title = (
        f"{task_name} — LIPA, cost={stats['final_objective']:.2f}, "
        f"iters={stats['n_iterations']}, def={stats['final_dynamics_violation']:.2e}"
    )
    fig.suptitle(title, fontsize=11)

    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(composite)
    ax.axis("off")
    overlay_times = ", ".join(f"{idx * dt:.2f}s" for idx in frame_idx)
    ax.set_title(f"Poses at t = {overlay_times}", fontsize=9)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(png_path, dpi=140)
    plt.close(fig)
    print(f"Wrote PNG: {png_path}")
