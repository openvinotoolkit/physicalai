# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Run the conveyor_sort scripted demonstrator headless: score, sweep belt speeds, or record a video.

Examples (from the package directory):

    uv run python scripts/run_conveyor_demo.py --speed 0.03 --episodes 2
    uv run python scripts/run_conveyor_demo.py --sweep 0.01 0.02 0.03 0.05 0.07 0.10 --seeds 3
    uv run python scripts/run_conveyor_demo.py --speed 0.03 --video /tmp/conveyor_demo.mp4  # overview, wrist, orbit
"""  # noqa: INP001

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
from loguru import logger

from physicalai_mujoco_so101_plugin.constants import SO101_JOINT_ORDER
from physicalai_mujoco_so101_plugin.conveyor import ConveyorSort
from physicalai_mujoco_so101_plugin.conveyor_demo import ConveyorDemonstrator, DemoStats
from physicalai_mujoco_so101_plugin.scene_registry import get_reset_fn, get_scene

SUBSTEPS = 10  # the plugin's default: 50 Hz control with dt = 2 ms


@dataclass
class RunResult:
    """Scores summed over the finished episodes of one run."""

    episodes: int = 0
    score: dict[str, int] = field(default_factory=lambda: {"correct": 0, "wrong": 0, "missed": 0})
    stats: DemoStats = field(default_factory=DemoStats)

    @property
    def items(self) -> int:
        """Items scored."""
        return sum(self.score.values())


def run(  # noqa: PLR0913, PLR0917 - a script entry point with plain knobs
    speed: float,
    seed: int,
    episodes: int,
    video: Path | None = None,
    fps: int = 30,
    action_hz: float | None = None,
    latency_s: float = 0.0,
) -> RunResult:
    """Run `episodes` demonstrator episodes at belt `speed` (m/s).

    `action_hz` and `latency_s` imitate a leader-arm loop such as Studio's
    teleoperation: targets are sampled at `action_hz` and reach the arm
    `latency_s` later. Without them the demonstrator drives the arm every tick.

    Returns:
        The summed scores and demonstrator counters.
    """
    scene = get_scene("conveyor_sort")
    model = scene.load_model()
    data = mujoco.MjData(model)
    reset = get_reset_fn("conveyor_sort")
    conveyor = ConveyorSort.maybe_create(model, rng=np.random.default_rng(seed + 1000), belt_speed=speed)
    if reset is None or conveyor is None:
        msg = "conveyor_sort has no reset function or conveyor belt"
        raise RuntimeError(msg)
    reset(model, data, np.random.default_rng(seed))
    demo = ConveyorDemonstrator(model, conveyor)
    dt = SUBSTEPS * model.opt.timestep
    # Generous cap: every item at the belt's pace plus slack, so a stuck episode cannot hang the run.
    time_limit = episodes * (conveyor.status()["items_per_episode"] * 0.18 / max(speed, 1e-3) + 90.0)

    writer, renderer, orbit = None, None, None
    frame_every = max(1, round(1.0 / (fps * dt)))
    fps_real = 1.0 / (frame_every * dt)  # plays back at real (sim) speed
    if video is not None:
        import cv2  # noqa: PLC0415

        renderer = mujoco.Renderer(model, 480, 640)
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter.fourcc(*"mp4v"), fps_real, (1920, 480))
        orbit = mujoco.MjvCamera()
        orbit.lookat[:] = (0.17, 0.08, 0.06)
        orbit.distance, orbit.elevation = 0.85, -26.0

    result = RunResult(stats=demo.stats)
    tick = 0
    relayed = action_hz is not None or latency_s > 0.0
    in_flight: deque[tuple[float, np.ndarray]] = deque()
    next_sample = 0.0
    actuators = [int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)) for name in SO101_JOINT_ORDER]
    while result.episodes < episodes and data.time < time_limit:
        targets = demo.step(data, dt, apply=not relayed)
        if relayed:
            if data.time >= next_sample:
                in_flight.append((data.time + latency_s, targets))
                next_sample += 1.0 / action_hz if action_hz else dt
            while in_flight and in_flight[0][0] <= data.time:
                data.ctrl[actuators] = in_flight.popleft()[1]
        for _ in range(SUBSTEPS):
            mujoco.mj_step(model, data)
        conveyor.update(model, data)
        status = conveyor.status()
        if status["episode_count"] > result.episodes:
            result.episodes = int(status["episode_count"])
            for key in result.score:
                result.score[key] += int(status["last_episode"][key])  # type: ignore[index]
        if writer is not None and renderer is not None and orbit is not None and tick % frame_every == 0:
            import cv2  # noqa: PLC0415

            renderer.update_scene(data, camera="overview")
            left = renderer.render()
            renderer.update_scene(data, camera="wrist")
            middle = renderer.render()
            orbit.azimuth = 205.0 + 20.0 * np.sin(data.time / 8.0)
            renderer.update_scene(data, camera=orbit)
            right = renderer.render()
            frame = np.hstack([left, middle, right])
            cv2.putText(
                frame,
                f"belt {100 * speed:.0f} cm/s  episode {result.episodes + 1}  {status['score']}",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        tick += 1
    if writer is not None:
        writer.release()
    return result


def main() -> None:
    """Parse arguments and run a single configuration or a belt-speed sweep."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--speed", type=float, default=0.03, help="belt speed in m/s (default 0.03)")
    parser.add_argument("--episodes", type=int, default=3, help="episodes per run (default 3)")
    parser.add_argument("--seed", type=int, default=0, help="first random seed (default 0)")
    parser.add_argument("--seeds", type=int, default=1, help="runs per speed, with consecutive seeds (default 1)")
    parser.add_argument("--sweep", type=float, nargs="+", help="belt speeds (m/s) to sweep instead of --speed")
    parser.add_argument("--action-hz", type=float, help="sample targets at this rate, like a leader-arm loop")
    parser.add_argument("--latency-ms", type=float, default=0.0, help="delay before sampled targets reach the arm")
    parser.add_argument("--video", type=Path, help="write an MP4 (overview, wrist and orbit views) of the first run")
    args = parser.parse_args()
    logger.remove()

    speeds = args.sweep or [args.speed]
    header = f"{'belt':>8} {'items':>6} {'success':>8} {'wrong':>6} {'missed':>7} {'grasp fails':>12} {'timeouts':>9}"
    print(header)  # noqa: T201
    for speed in speeds:
        total = RunResult()
        for seed in range(args.seed, args.seed + args.seeds):
            video = args.video if (args.video is not None and speed == speeds[0] and seed == args.seed) else None
            res = run(
                speed,
                seed,
                args.episodes,
                video=video,
                action_hz=args.action_hz,
                latency_s=args.latency_ms / 1000.0,
            )
            total.episodes += res.episodes
            for key in total.score:
                total.score[key] += res.score[key]
            total.stats.attempts += res.stats.attempts
            total.stats.grasp_failed += res.stats.grasp_failed
            total.stats.approach_timeouts += res.stats.approach_timeouts
        success = 100.0 * total.score["correct"] / max(total.items, 1)
        print(  # noqa: T201
            f"{100 * speed:5.1f} cm/s {total.items:6d} {success:7.1f}% {total.score['wrong']:6d} "
            f"{total.score['missed']:7d} {total.stats.grasp_failed:5d}/{total.stats.attempts:<6d} "
            f"{total.stats.approach_timeouts:9d}"
        )
    if args.video is not None:
        print(f"video: {args.video}")  # noqa: T201


if __name__ == "__main__":
    main()
