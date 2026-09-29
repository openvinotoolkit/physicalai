# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Headless behaviour benchmark: model sanity for every scene plus conveyor demonstrator success.

Records, per scene: model sizes, the arm's joint/actuator ranges, the pose of each arm after a
seeded reset and 2 s at its home targets, the wrist camera pose in the gripper frame, step cost,
and 224x224 renders of every streamed camera. Then scores the conveyor_sort scripted
demonstrator driving the arm directly, over fixed seeds at two belt speeds.

Compare two runs (for example before and after a model change) with ``--compare``: it prints
the largest numeric difference per scene and the mean absolute pixel difference per render.

Examples (from the repo root):

    uv run python packages/physicalai-mujoco-so101-plugin/scripts/bench_scenes.py --out /tmp/bench/before
    uv run python packages/physicalai-mujoco-so101-plugin/scripts/bench_scenes.py --out /tmp/bench/after \
        --compare /tmp/bench/before/bench.json
"""  # noqa: INP001

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np
from loguru import logger
from physicalai_mujoco_so101_plugin.constants import BIMANUAL_SO101_JOINT_ORDER, SO101_JOINT_ORDER
from physicalai_mujoco_so101_plugin.scene_registry import SceneConfig, get_reset_fn, list_scenes

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_conveyor_demo import run as run_conveyor  # noqa: E402

RENDER_SIZE = 224
HOLD_S = 2.0
STEP_SAMPLES = 1000
CONVEYOR_SPEEDS = (0.03, 0.07)
CONVEYOR_SEEDS = range(8)
CONVEYOR_EPISODES = 2


def _load_model(scene: SceneConfig) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(str(scene.scene_xml_path))


def _arm_prefixes(scene: SceneConfig) -> tuple[str, ...]:
    return ("left_", "right_") if scene.num_arms == 2 else ("",)  # noqa: PLR2004


def _round(values: object) -> object:
    return np.round(np.asarray(values, dtype=np.float64), 6).tolist()


def _bench_scene(scene: SceneConfig, out_dir: Path) -> dict[str, object]:  # noqa: PLR0914
    started = time.perf_counter()
    model = _load_model(scene)
    load_s = time.perf_counter() - started
    data = mujoco.MjData(model)
    joint_order = BIMANUAL_SO101_JOINT_ORDER if scene.num_arms == 2 else SO101_JOINT_ORDER  # noqa: PLR2004
    joints = [model.joint(name) for name in joint_order]
    actuators = [model.actuator(name) for name in joint_order]

    mujoco.mj_forward(model, data)
    arms: dict[str, object] = {}
    for prefix in _arm_prefixes(scene):
        gripper = model.body(f"{prefix}gripper").id
        rot = data.xmat[gripper].reshape(3, 3)
        cam = model.camera(f"{prefix}wrist").id
        arms[prefix or "arm"] = {
            "base_xpos": _round(data.xpos[model.body(f"{prefix}base").id]),
            "base_xquat": _round(data.xquat[model.body(f"{prefix}base").id]),
            "gripper_xpos_at_qpos0": _round(data.xpos[gripper]),
            "gripperframe_xpos_at_qpos0": _round(data.site_xpos[model.site(f"{prefix}gripperframe").id]),
            "wrist_cam_pos_in_gripper": _round(rot.T @ (data.cam_xpos[cam] - data.xpos[gripper])),
            "wrist_cam_rot_in_gripper": _round(rot.T @ data.cam_xmat[cam].reshape(3, 3)),
            "wrist_cam_fovy": float(model.cam_fovy[cam]),
        }

    reset = get_reset_fn(scene.scene_id)
    if reset is not None:
        reset(model, data, np.random.default_rng(0))
    for joint, actuator in zip(joints, actuators, strict=True):
        data.ctrl[actuator.id] = data.qpos[joint.qposadr[0]]
    while data.time < HOLD_S:
        mujoco.mj_step(model, data)
    finite = bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all())
    for prefix in _arm_prefixes(scene):
        arm = arms[prefix or "arm"]
        arm["gripper_xpos_after_hold"] = _round(data.xpos[model.body(f"{prefix}gripper").id])  # type: ignore[index]

    started = time.perf_counter()
    for _ in range(STEP_SAMPLES):
        mujoco.mj_step(model, data)
    step_ms = 1000.0 * (time.perf_counter() - started) / STEP_SAMPLES

    renders: dict[str, object] = {}
    with mujoco.Renderer(model, RENDER_SIZE, RENDER_SIZE) as renderer:
        for camera in ("overview", *(f"{prefix}wrist" for prefix in _arm_prefixes(scene))):
            renderer.update_scene(data, camera=camera)
            pixels = renderer.render()
            path = out_dir / f"{scene.scene_id}_{camera}.png"
            cv2.imwrite(str(path), cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR))
            renders[camera] = {"png": str(path), "mean_pixel": round(float(pixels.mean()), 3)}

    return {
        "load_s": round(load_s, 3),
        "nq": model.nq,
        "nv": model.nv,
        "nu": model.nu,
        "nbody": model.nbody,
        "ngeom": model.ngeom,
        "timestep": float(model.opt.timestep),
        "joint_range": {name: _round(model.jnt_range[j.id]) for name, j in zip(joint_order, joints, strict=True)},
        "ctrlrange": {name: _round(a.ctrlrange) for name, a in zip(joint_order, actuators, strict=True)},
        "forcerange": {name: _round(a.forcerange) for name, a in zip(joint_order, actuators, strict=True)},
        "arms": arms,
        "arm_qpos_after_hold": _round([data.qpos[j.qposadr[0]] for j in joints]),
        "finite_after_hold": finite,
        "step_ms": round(step_ms, 4),
        "renders": renders,
    }


def _bench_conveyor() -> dict[str, object]:
    results: dict[str, object] = {}
    for speed in CONVEYOR_SPEEDS:
        started = time.perf_counter()
        score = {"correct": 0, "wrong": 0, "missed": 0}
        attempts = grasp_failed = timeouts = episodes = 0
        for seed in CONVEYOR_SEEDS:
            res = run_conveyor(speed, seed, CONVEYOR_EPISODES)
            episodes += res.episodes
            for key in score:
                score[key] += res.score[key]
            attempts += res.stats.attempts
            grasp_failed += res.stats.grasp_failed
            timeouts += res.stats.approach_timeouts
        items = sum(score.values())
        results[f"{speed:.2f}"] = {
            "seeds": list(CONVEYOR_SEEDS),
            "episodes": episodes,
            "items": items,
            **score,
            "success_pct": round(100.0 * score["correct"] / max(items, 1), 2),
            "grasp_failed": grasp_failed,
            "attempts": attempts,
            "approach_timeouts": timeouts,
            "wall_s": round(time.perf_counter() - started, 1),
        }
    return results


def _numeric_leaves(value: object, path: str = "") -> dict[str, float]:
    if isinstance(value, dict):
        leaves: dict[str, float] = {}
        for key, item in value.items():
            leaves.update(_numeric_leaves(item, f"{path}/{key}"))
        return leaves
    if isinstance(value, list):
        leaves = {}
        for index, item in enumerate(value):
            leaves.update(_numeric_leaves(item, f"{path}[{index}]"))
        return leaves
    if isinstance(value, bool) or not isinstance(value, int | float):
        return {}
    return {path: float(value)}


def _compare(before: dict[str, object], after: dict[str, object]) -> None:
    timing_keys = ("load_s", "step_ms", "wall_s", "mean_pixel")
    old, new = _numeric_leaves(before), _numeric_leaves(after)
    print("== numeric differences (excluding timings) ==")  # noqa: T201
    for key in sorted(old.keys() | new.keys()):
        if any(t in key for t in timing_keys):
            continue
        a, b = old.get(key), new.get(key)
        if a is None or b is None or abs(a - b) > 1e-5:  # noqa: PLR2004
            print(f"  {key}: {a} -> {b}")  # noqa: T201
    print("== renders: mean |pixel difference| (0-255) ==")  # noqa: T201
    for scene_id, scene in after["scenes"].items():  # type: ignore[union-attr]
        for camera, render in scene["renders"].items():
            previous = before["scenes"].get(scene_id, {}).get("renders", {}).get(camera)  # type: ignore[union-attr]
            if previous is None:
                print(f"  {scene_id}/{camera}: missing before")  # noqa: T201
                continue
            a = cv2.imread(previous["png"]).astype(np.int16)
            b = cv2.imread(render["png"]).astype(np.int16)
            print(f"  {scene_id}/{camera}: {np.abs(a - b).mean():.3f}")  # noqa: T201


def main() -> None:
    """Run the benchmark and write ``bench.json`` plus renders to ``--out``."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True, help="output directory for bench.json and renders")
    parser.add_argument("--compare", type=Path, help="an earlier bench.json to compare against")
    parser.add_argument("--skip-conveyor", action="store_true", help="only run the per-scene checks")
    args = parser.parse_args()
    logger.remove()
    args.out.mkdir(parents=True, exist_ok=True)

    result: dict[str, object] = {"mujoco": mujoco.__version__, "scenes": {}}
    for scene_id, scene in list_scenes().items():
        result["scenes"][scene_id] = _bench_scene(scene, args.out)  # type: ignore[index]
        print(f"{scene_id}: ok")  # noqa: T201
    if not args.skip_conveyor:
        result["conveyor"] = _bench_conveyor()
        for speed, row in result["conveyor"].items():  # type: ignore[union-attr]
            print(  # noqa: T201
                f"conveyor {speed} m/s: {row['success_pct']}% of {row['items']} items, "
                f"grasp fails {row['grasp_failed']}/{row['attempts']}, timeouts {row['approach_timeouts']}"
            )
    (args.out / "bench.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out / 'bench.json'}")  # noqa: T201
    if args.compare is not None:
        _compare(json.loads(args.compare.read_text(encoding="utf-8")), result)


if __name__ == "__main__":
    main()
