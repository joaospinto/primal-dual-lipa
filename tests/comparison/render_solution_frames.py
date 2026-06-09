"""Render uniformly sampled MJX solution frames from a saved benchmark archive.

This deliberately reads the ``.npz`` files written by
``tests.comparison.run_benchmark --save-solutions`` so solving and
visualization can be run independently.

Example:

    uv run --extra mpc-examples python -m tests.comparison.render_solution_frames \\
        --solution comparison_results/solutions/backflip__lipa.npz \\
        --problem backflip --out backflip_frames.ppm --frames 6
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

import numpy as np


def _default_mujoco_gl() -> str:
    if sys.platform == "darwin":
        return "cgl"
    if sys.platform.startswith("win"):
        return "wgl"
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        return "glfw"
    return "egl"


PROBLEM_CONFIGS = {
    "barrel_roll": "tests.mpc_examples.configs.config_barrel_roll",
    "backflip": "tests.mpc_examples.configs.config_h1_backflip",
    "jump": "tests.mpc_examples.configs.config_h1_jump_forward",
    "trot": "tests.mpc_examples.configs.config_aliengo_trot",
}


CAMERA_DEFAULTS = {
    "barrel_roll": {"distance": 2.7, "azimuth": 135.0, "elevation": -18.0},
    "backflip": {"distance": 4.2, "azimuth": 90.0, "elevation": -12.0},
    "jump": {"distance": 4.0, "azimuth": 90.0, "elevation": -12.0},
    "trot": {"distance": 3.0, "azimuth": 90.0, "elevation": -15.0},
}


def _load_config(problem_name: str):
    if problem_name not in PROBLEM_CONFIGS:
        choices = ", ".join(sorted(PROBLEM_CONFIGS))
        raise ValueError(f"unknown MJX problem {problem_name!r}; choose one of {choices}")
    return importlib.import_module(PROBLEM_CONFIGS[problem_name])


def _frame_indices(num_states: int, num_frames: int) -> np.ndarray:
    if num_states <= 0:
        raise ValueError("solution archive contains an empty X trajectory")
    if num_frames <= 0:
        raise ValueError("--frames must be positive")
    return np.rint(np.linspace(0, num_states - 1, num_frames)).astype(int)


def _write_ppm(path: Path, image: np.ndarray) -> None:
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected RGB image with shape (H, W, 3), got {image.shape}")
    image = np.clip(image, 0, 255).astype(np.uint8, copy=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(f"P6\n{image.shape[1]} {image.shape[0]}\n255\n".encode("ascii"))
        f.write(image.tobytes())


def _tile_frames(frames: list[np.ndarray], gap_px: int = 8) -> np.ndarray:
    if not frames:
        raise ValueError("no frames to tile")
    heights = {frame.shape[0] for frame in frames}
    widths = {frame.shape[1] for frame in frames}
    if len(heights) != 1 or len(widths) != 1:
        raise ValueError("all frames must have the same size")
    h, w = frames[0].shape[:2]
    gap = np.full((h, gap_px, 3), 255, dtype=np.uint8)
    pieces: list[np.ndarray] = []
    for i, frame in enumerate(frames):
        if i:
            pieces.append(gap)
        pieces.append(frame.astype(np.uint8, copy=False))
    return np.concatenate(pieces, axis=1)


def render_frame_strip(
    solution_path: Path,
    problem_name: str,
    out_path: Path,
    *,
    num_frames: int,
    width: int,
    height: int,
    gap_px: int,
) -> None:
    os.environ.setdefault("MUJOCO_GL", _default_mujoco_gl())
    if os.environ.get("MUJOCO_GL") == "egl":
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    import mujoco

    from tests.mpc_examples.run_offline import _ensure_assets
    from tests.mpc_examples.viz import state_to_mujoco

    config = _load_config(problem_name)
    _ensure_assets(config)

    archive = np.load(solution_path, allow_pickle=False)
    X = np.asarray(archive["X"])
    frame_idx = _frame_indices(X.shape[0], num_frames)

    scene_path = getattr(config, "scene_path", config.model_path)
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)
    cam = mujoco.MjvCamera()
    cam_defaults = CAMERA_DEFAULTS.get(problem_name, CAMERA_DEFAULTS["barrel_roll"])
    cam.distance = float(cam_defaults["distance"])
    cam.azimuth = float(cam_defaults["azimuth"])
    cam.elevation = float(cam_defaults["elevation"])

    frames: list[np.ndarray] = []
    try:
        for idx in frame_idx:
            qpos, qvel = state_to_mujoco(config, X[int(idx)])
            data.qpos = np.asarray(qpos)
            data.qvel = np.asarray(qvel)
            mujoco.mj_forward(model, data)
            if data.qpos.size >= 3:
                cam.lookat[:] = np.asarray(data.qpos[:3], dtype=np.float64)
            else:
                cam.lookat[:] = np.asarray(data.xpos[0], dtype=np.float64)
            renderer.update_scene(data, cam)
            frames.append(renderer.render().copy())
    finally:
        renderer.close()

    strip = _tile_frames(frames, gap_px=gap_px)
    _write_ppm(out_path, strip)
    times = ", ".join(str(i) for i in frame_idx)
    print(f"Wrote {out_path} from frames [{times}]")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--solution", type=Path, required=True)
    parser.add_argument("--problem", choices=tuple(PROBLEM_CONFIGS), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=6)
    parser.add_argument("--width", type=int, default=360)
    parser.add_argument("--height", type=int, default=260)
    parser.add_argument("--gap-px", type=int, default=8)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    render_frame_strip(
        args.solution,
        args.problem,
        args.out,
        num_frames=args.frames,
        width=args.width,
        height=args.height,
        gap_px=args.gap_px,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
