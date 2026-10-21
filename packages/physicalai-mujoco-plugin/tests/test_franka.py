# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Franka twins: the FR3 with a Franka Hand and the FR3 Duo, under ``physicalai-franka-plugin``'s names and units."""

# MuJoCo model/data attributes are supplied by the C extension at runtime.
# pyrefly: ignore-errors [missing-attribute]

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest

from physicalai_mujoco_plugin.compose import fetch_profile, load_robot_spec, robot_layout
from physicalai_mujoco_plugin.profiles import get_profile
from physicalai_mujoco_plugin.profiles.franka import (
    FRANKA_DUO_ARM_Y,
    FRANKA_DUO_HEIGHT,
    FRANKA_DUO_HOME_DEG,
    FRANKA_DUO_PROFILE,
    FRANKA_DUO_RPY,
    FRANKA_FR3_HAND_PROFILE,
    FRANKA_HAND_WIDTH_M,
)
from physicalai_mujoco_plugin.scene_registry import supported_arm_counts
from physicalai_mujoco_plugin.studio_catalog import list_catalog_entries

ARM_NAMES = tuple(f"fr3_joint{index}" for index in range(1, 8))
FR3_HAND_NAMES = (*ARM_NAMES, "gripper")
DUO_NAMES = tuple(f"{side}_{name}" for side in ("left", "right") for name in FR3_HAND_NAMES)


def _model(profile: object) -> mujoco.MjModel:
    return load_robot_spec(profile).compile()  # type: ignore[arg-type]


def _relative_pose(model: mujoco.MjModel, data: mujoco.MjData, parent: str, child: str) -> tuple[np.ndarray, np.ndarray]:
    rotation = data.body(parent).xmat.reshape(3, 3)
    return rotation.T @ (data.body(child).xpos - data.body(parent).xpos), rotation.T @ data.body(child).xmat.reshape(
        3, 3
    )


def test_channels_follow_the_franka_plugin() -> None:
    """``FrankaFR3(gripper_enabled=True)``: seven joints in degrees, then the hand width in metres."""
    for profile, names in ((FRANKA_FR3_HAND_PROFILE, FR3_HAND_NAMES), (FRANKA_DUO_PROFILE, DUO_NAMES)):
        channels = robot_layout(profile).channels
        assert tuple(channel.name for channel in channels) == names
        assert [channel.unit for channel in channels] == [("metres" if n.endswith("gripper") else "degrees") for n in names]


def test_channels_match_the_real_driver_when_installed() -> None:
    robot = pytest.importorskip("physicalai_franka_plugin.robot", reason="needs physicalai-franka-plugin")
    assert (*robot.FrankaFR3.JOINT_ORDER, "gripper") == FR3_HAND_NAMES


def test_hand_sits_on_the_flange_as_on_menageries_panda() -> None:
    fr3 = _model(FRANKA_FR3_HAND_PROFILE)
    panda = mujoco.MjModel.from_xml_path(str(fetch_profile(get_profile("franka_panda"))))
    fr3_data, panda_data = mujoco.MjData(fr3), mujoco.MjData(panda)
    mujoco.mj_forward(fr3, fr3_data)
    mujoco.mj_forward(panda, panda_data)
    pos, rot = _relative_pose(fr3, fr3_data, "fr3_link7", "hand")
    panda_pos, panda_rot = _relative_pose(panda, panda_data, "link7", "hand")
    np.testing.assert_allclose(pos, panda_pos, atol=1e-6)
    np.testing.assert_allclose(rot, panda_rot, atol=1e-6)


@pytest.mark.parametrize("width", [0.0, 0.03, FRANKA_HAND_WIDTH_M])
def test_gripper_command_is_the_hand_opening_width(width: float) -> None:
    model = _model(FRANKA_FR3_HAND_PROFILE)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    actuator = model.actuator("gripper")
    np.testing.assert_allclose(model.actuator_ctrlrange[actuator.id], (0.0, FRANKA_HAND_WIDTH_M))
    data.ctrl[actuator.id] = width
    for _ in range(2000):
        mujoco.mj_step(model, data)
    opening = data.joint("finger_joint1").qpos[0] + data.joint("finger_joint2").qpos[0]
    assert opening == pytest.approx(width, abs=1e-3)
    assert data.ten_length[model.tendon("gripper_width").id] == pytest.approx(opening, abs=1e-9)


def test_duo_arms_stand_on_frankas_duo_mount() -> None:
    model = _model(FRANKA_DUO_PROFILE)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    roll, pitch, yaw = FRANKA_DUO_RPY
    for side, sign in (("left", 1.0), ("right", -1.0)):
        base = data.body(f"{side}_base")
        np.testing.assert_allclose(base.xpos, (0.0, sign * FRANKA_DUO_ARM_Y, FRANKA_DUO_HEIGHT), atol=1e-9)
        r, y = (-roll, -yaw) if side == "left" else (roll, yaw)
        rx = np.array([[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]])
        ry = np.array([[math.cos(pitch), 0, math.sin(pitch)], [0, 1, 0], [-math.sin(pitch), 0, math.cos(pitch)]])
        rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
        np.testing.assert_allclose(base.xmat.reshape(3, 3), rz @ ry @ rx, atol=1e-6)


def test_duo_home_is_frankas_home_mirrored_with_hands_open() -> None:
    model = _model(FRANKA_DUO_PROFILE)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    for side, home in FRANKA_DUO_HOME_DEG.items():
        q = [math.degrees(data.joint(f"{side}_fr3_joint{index}").qpos[0]) for index in range(1, 8)]
        np.testing.assert_allclose(q, home, atol=1e-6)
        assert data.ctrl[model.actuator(f"{side}_gripper").id] == FRANKA_HAND_WIDTH_M
    assert np.all(np.isclose(FRANKA_DUO_HOME_DEG["left"], np.array(FRANKA_DUO_HOME_DEG["right"]) * (1, 1, -1, 1, -1, 1, -1)))


def test_duo_home_pose_has_no_contacts() -> None:
    model = _model(FRANKA_DUO_PROFILE)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    assert data.ncon == 0


def test_duo_runs_alone_and_the_fr3_pair_is_the_duo_in_studio() -> None:
    """The Duo is one two-arm model; Studio offers it instead of a side-by-side FR3 pair."""
    assert supported_arm_counts(FRANKA_DUO_PROFILE) == (1,)
    entries = {(entry.profile.name, entry.bimanual) for entry in list_catalog_entries()}
    assert ("franka_duo", False) in entries
    assert ("franka_fr3_hand", False) in entries
    assert ("franka_fr3_hand", True) not in entries
