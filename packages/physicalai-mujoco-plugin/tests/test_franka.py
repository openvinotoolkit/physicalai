# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Franka twins: the FR3 with its Franka Hand and the FR3 Duo, under ``physicalai-franka-plugin``'s names and units."""

# MuJoCo model/data attributes are supplied by the C extension at runtime.
# pyrefly: ignore-errors [missing-attribute]

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest

from physicalai_mujoco_plugin.compose import fetch_profile, load_robot_spec, robot_layout
from physicalai_mujoco_plugin.profiles import get_profile
from physicalai_mujoco_plugin.profiles.dataset import FRANKA_HAND_GRASP_FORCE_N, FRANKA_HAND_SPEED_M_S
from physicalai_mujoco_plugin.profiles.franka import (
    FRANKA_DUO_ARM_Y,
    FRANKA_DUO_HEIGHT,
    FRANKA_DUO_HOME_DEG,
    FRANKA_DUO_PROFILE,
    FRANKA_DUO_RPY,
    FRANKA_FR3_PROFILE,
    FRANKA_HAND_WIDTH_M,
)
from physicalai_mujoco_plugin.scene_registry import supported_arm_counts
from physicalai_mujoco_plugin.studio_catalog import list_catalog_entries

ARM_NAMES = tuple(f"fr3_joint{index}" for index in range(1, 8))
FR3_NAMES = (*ARM_NAMES, "gripper")
DUO_NAMES = tuple(f"{side}_{name}" for side in ("left", "right") for name in FR3_NAMES)


def _model(profile: object) -> mujoco.MjModel:
    return load_robot_spec(profile).compile()  # type: ignore[arg-type]


def _relative_pose(model: mujoco.MjModel, data: mujoco.MjData, parent: str, child: str) -> tuple[np.ndarray, np.ndarray]:
    rotation = data.body(parent).xmat.reshape(3, 3)
    return rotation.T @ (data.body(child).xpos - data.body(parent).xpos), rotation.T @ data.body(child).xmat.reshape(
        3, 3
    )


def test_channels_follow_the_franka_plugin() -> None:
    """``FrankaFR3(gripper_enabled=True)``: seven joints in degrees, then the hand width in metres."""
    for profile, names in ((FRANKA_FR3_PROFILE, FR3_NAMES), (FRANKA_DUO_PROFILE, DUO_NAMES)):
        channels = robot_layout(profile).channels
        assert tuple(channel.name for channel in channels) == names
        assert [channel.unit for channel in channels] == [("metres" if n.endswith("gripper") else "degrees") for n in names]


def test_channels_match_the_real_driver_when_installed() -> None:
    robot = pytest.importorskip("physicalai_franka_plugin.robot", reason="needs physicalai-franka-plugin")
    assert (*robot.FrankaFR3.JOINT_ORDER, "gripper") == FR3_NAMES


def test_hand_sits_on_the_flange_as_on_menageries_panda() -> None:
    fr3 = _model(FRANKA_FR3_PROFILE)
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
    model = _model(FRANKA_FR3_PROFILE)
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


def _with_block(profile: object, hands: tuple[str, ...], width: float) -> mujoco.MjModel:
    """The robot at home with a floating 45 g block of *width* (across the fingers) between each hand's finger tips."""
    spec = load_robot_spec(profile)  # type: ignore[arg-type]
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    for hand in hands:
        rotation = data.body(hand).xmat.reshape(3, 3)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, rotation.ravel())
        pos = data.body(hand).xpos + rotation @ np.array([0.0, 0.0, 0.103])  # the finger pads' centre
        block = spec.worldbody.add_body(name=f"{hand}_block", pos=pos, quat=quat, gravcomp=1.0)
        block.add_freejoint()
        block.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.01, width / 2, 0.006], mass=0.045)
    return spec.compile()


def _squeeze(model: mujoco.MjModel, data: mujoco.MjData, finger: str) -> float:
    """Return the sum of the contact forces on *finger* along its closing axis."""
    body, axis, force, total = model.body(finger).id, data.body(finger).xmat.reshape(3, 3)[:, 1], np.zeros(6), 0.0
    for index, contact in enumerate(data.contact):
        if body in (model.geom_bodyid[contact.geom1], model.geom_bodyid[contact.geom2]):
            mujoco.mj_contactForce(model, data, index, force)
            total += abs(float(contact.frame.reshape(3, 3).T @ force[:3] @ axis))
    return total


@pytest.mark.parametrize(
    ("profile_name", "closed", "prefixes"),
    [("franka_fr3", 0.0, ("",)), ("franka_duo", 0.0, ("left_", "right_")), ("franka_panda", 0.0, ("",))],
)
@pytest.mark.parametrize("width", [0.026, 0.032, 0.038])
def test_closed_hand_squeezes_with_the_real_hands_grasp_force(
    profile_name: str, closed: float, prefixes: tuple[str, ...], width: float
) -> None:
    """The real Franka Hand grasps with 70 N; every simulated one presses each finger with that, without sinking in."""
    model = _with_block(get_profile(profile_name), tuple(f"{prefix}hand" for prefix in prefixes), width)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    grippers = [model.actuator(f"{prefix}gripper" if profile_name != "franka_panda" else "actuator8").id for prefix in prefixes]
    data.ctrl[grippers] = closed
    for _ in range(1000):
        mujoco.mj_step(model, data)
    for prefix in prefixes:
        for side in ("left", "right"):
            assert _squeeze(model, data, f"{prefix}{side}_finger") == pytest.approx(FRANKA_HAND_GRASP_FORCE_N, rel=0.05)
        opening = data.joint(f"{prefix}finger_joint1").qpos[0] + data.joint(f"{prefix}finger_joint2").qpos[0]
        assert width - 0.002 < opening <= width


@pytest.mark.parametrize("profile_name", ["franka_fr3", "franka_panda"])
def test_hand_closes_and_opens_at_the_real_hands_speed(profile_name: str) -> None:
    model = _model(get_profile(profile_name))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    gripper = model.actuator("gripper" if profile_name == "franka_fr3" else "actuator8").id
    fingers = [model.joint(name).qposadr[0] for name in ("finger_joint1", "finger_joint2")]
    for command, target in ((0.0, 0.0), (model.actuator_ctrlrange[gripper, 1], FRANKA_HAND_WIDTH_M)):
        data.ctrl[gripper] = command
        widths = []
        for _ in range(1000):
            mujoco.mj_step(model, data)
            widths.append(data.qpos[fingers].sum())
        speed = np.abs(np.diff(widths)).max() / model.opt.timestep
        assert 0.9 * FRANKA_HAND_SPEED_M_S < speed <= 1.01 * FRANKA_HAND_SPEED_M_S
        assert widths[-1] == pytest.approx(target, abs=1e-4)


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
    assert ("franka_fr3", False) in entries
    assert ("franka_fr3", True) not in entries
