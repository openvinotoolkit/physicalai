# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Smoke matrix (TST-5): every supported profile runs stably in its default scene (real MuJoCo, Menagerie download).

The matrix is deselected by default (marker ``mujoco_smoke``); CI runs it in its own step with
``pytest -m mujoco_smoke``. It renders nothing: cameras are off, so it runs on headless hosts.
Thresholds come from the PoC sweep (spec §17): 5 degrees or 5 mm, and a released floating base
keeps at least 80 % of its start height. Every nudge exceeds the tracking tolerance, so a robot
that ignores its actions fails.
"""

# MuJoCo's bindings are supplied by the C extension at runtime.
# pyrefly: ignore-errors [missing-attribute]

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import mujoco
import numpy as np
import pytest

from physicalai.robot.interface import Robot
from physicalai_mujoco_plugin.channels import ArmChannels
from physicalai_mujoco_plugin.profiles import list_profiles
from physicalai_mujoco_plugin.robot import MuJoCoRobot
from physicalai_mujoco_plugin.scene_registry import list_scenes_for

if TYPE_CHECKING:
    from physicalai_mujoco_plugin.channels import Channel

pytestmark = [pytest.mark.mujoco_smoke, pytest.mark.slow, pytest.mark.requires_download]

RATE_HZ = 50.0
IDLE_S = 2.0
"""Idle with no action: a floating base is still held."""
HOLD_S = 3.0
"""The home pose sent as an action, which releases a floating base."""
NUDGE_S = 2.0
"""Fixed bases: every channel with a target span moved toward the middle of its span."""
NUDGE_FRACTION = 0.2
MIN_NUDGE = 2.0
"""Smallest nudge, in tolerances: a robot that drops the action then fails the check."""
MIN_GRIPPER_PROGRESS = 0.5
"""Fraction of a nudge a tendon gripper must cover; the PoC thresholds only cover joints with a range."""
TOLERANCE = {"degrees": 5.0, "metres": 0.005}
"""Largest position error per public unit; the matrix runs every profile in ``degrees``."""
MIN_BASE_HEIGHT_RATIO = 0.8

SUPPORTED_PROFILES = [profile.name for profile in list_profiles() if profile.tier != "unsupported"]


def _run(robot: MuJoCoRobot, seconds: float) -> np.ndarray:
    """Step the robot for *seconds* of simulated time, checking that every state stays finite.

    Returns:
        The last observation's joint positions.
    """
    for _ in range(round(seconds * RATE_HZ)):
        observation = robot.get_observation()
        assert np.isfinite(observation.state).all()
    return observation.joint_positions.astype(np.float64)


def _tolerance(robot: MuJoCoRobot) -> np.ndarray:
    return np.array([TOLERANCE[unit] for arm in robot._sim.channels for unit in arm.units])  # noqa: SLF001


def _assert_tracks(
    robot: MuJoCoRobot, positions: np.ndarray, targets: np.ndarray, phase: str, checked: np.ndarray | None = None
) -> None:
    """Assert that the *checked* channels (default: all) are within tolerance of their targets."""
    error = np.abs(positions - targets) / _tolerance(robot)
    if checked is not None:
        error = np.where(checked, error, 0.0)
    worst = int(np.argmax(error))
    assert error[worst] <= 1.0, f"{phase}: {robot.joint_names[worst]} is off by {error[worst]:.3g} tolerances"


def _span(robot: MuJoCoRobot, channel: Channel) -> tuple[float, float] | None:
    """Return the public span a channel's target may take: its joint range, else its position actuator's ``ctrlrange``.

    Tendon grippers (Panda, xArm7) have no joint range; their ``ctrlrange`` bounds the target.
    """
    model_span = channel.range
    if model_span is None:
        member = channel.first
        model = robot._model  # noqa: SLF001
        if member.kind != "position" or not model.actuator_ctrllimited[member.actuator_id]:
            return None
        factor = member.ctrl_per_unit * member.gear
        model_span = tuple(float(value) / factor for value in model.actuator_ctrlrange[member.actuator_id])
    low, high = sorted(ArmChannels.to_public(channel, value) for value in model_span)
    return low, high


def _nudged(robot: MuJoCoRobot, home: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Move every channel with a span from home toward the middle of its span.

    Each moves ``NUDGE_FRACTION`` of the way, but at least ``MIN_NUDGE`` tolerances (toward the
    upper end from the middle), within the span.

    Returns:
        The nudged targets in public units, which channels were nudged, and which of those have a joint range.
    """
    channels = [channel for arm in robot._sim.channels for channel in arm.channels]  # noqa: SLF001
    tolerance = _tolerance(robot)
    targets = home.copy()
    nudged = np.zeros(len(channels), dtype=bool)
    ranged = np.zeros(len(channels), dtype=bool)
    for index, channel in enumerate(channels):
        span = _span(robot, channel)
        if span is None:
            continue
        toward_middle = (span[0] + span[1]) / 2 - home[index]
        step = NUDGE_FRACTION * toward_middle
        if abs(step) < MIN_NUDGE * tolerance[index]:
            step = math.copysign(MIN_NUDGE * tolerance[index], toward_middle)
        targets[index] = min(max(home[index] + step, span[0]), span[1])
        nudged[index] = True
        ranged[index] = channel.range is not None
    return targets, nudged, ranged


def _assert_nudge_tracks(robot: MuJoCoRobot, before: np.ndarray, home: np.ndarray) -> None:
    """Nudge the channels and check that they moved, and that joints with a range track their targets."""
    targets, nudged, ranged = _nudged(robot, home)
    commanded = targets - before
    assert nudged.any()
    # A target within tolerance of the current position would pass even if the action were dropped.
    assert (np.abs(commanded)[nudged] > _tolerance(robot)[nudged]).all()
    robot.send_action(targets.astype(np.float32))
    after = _run(robot, NUDGE_S)
    _assert_tracks(robot, after, targets, "nudge", ranged)
    progress = (after - before)[nudged] / commanded[nudged]
    names = np.array(robot.joint_names)[nudged]
    assert (progress >= MIN_GRIPPER_PROGRESS).all(), dict(zip(names, progress.round(2), strict=True))


@pytest.mark.parametrize("name", SUPPORTED_PROFILES)
def test_profile_runs_stably_in_its_default_scene(name: str) -> None:
    robot = MuJoCoRobot(profile=name, unit="degrees", rate_hz=RATE_HZ, cameras=[], seed=0)
    assert isinstance(robot, Robot)
    assert robot.profile.default_scene in list_scenes_for(robot.profile)
    robot.connect()
    try:
        start = robot.get_observation()
        home = start.joint_positions.astype(np.float64)
        floating = "base_pos" in start.sensor_data

        _assert_tracks(robot, _run(robot, IDLE_S), home, "idle")

        robot.send_action(home.astype(np.float32))
        positions = _run(robot, HOLD_S)
        if floating:
            assert not any(base["held"] for base in robot._http_status()["bases"])  # noqa: SLF001 - the action released it
            height = robot.get_observation().sensor_data["base_pos"][2]
            assert height >= MIN_BASE_HEIGHT_RATIO * start.sensor_data["base_pos"][2]
        else:
            _assert_tracks(robot, positions, home, "hold")
            _assert_nudge_tracks(robot, positions, home)

        assert robot._data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number == 0  # noqa: SLF001
    finally:
        robot.disconnect()
