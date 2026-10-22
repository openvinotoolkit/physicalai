# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Franka twins: the FR3 with its Franka Hand and the FR3 Duo, under ``physicalai-franka-plugin``'s names and units.

Also the FR3 with DROID's Robotiq 2F-85, which publishes the same channels but is not a twin of the hardware.
"""

# MuJoCo model/data attributes are supplied by the C extension at runtime.
# pyrefly: ignore-errors [missing-attribute]

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest

from physicalai_mujoco_plugin.compose import fetch_profile, load_robot_spec, robot_layout
from physicalai_mujoco_plugin.profiles import get_profile, validate_profile
from physicalai_mujoco_plugin.profiles.franka import (
    FRANKA_DUO_ARM_Y,
    FRANKA_DUO_HEIGHT,
    FRANKA_DUO_HOME_DEG,
    FRANKA_DUO_PROFILE,
    FRANKA_DUO_RPY,
    FRANKA_FR3_PROFILE,
    FRANKA_FR3_ROBOTIQ_PROFILE,
    FRANKA_HAND_WIDTH_M,
    ROBOTIQ_WIDTH_M,
)
from physicalai_mujoco_plugin.robot import MuJoCoRobot
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


# ---------------------------------------------------------------------------
# FR3 with DROID's Robotiq 2F-85
# ---------------------------------------------------------------------------

ROBOTIQ_WIDTH_TOLERANCE_M = 3e-3
"""The width map is linear in the finger tendon's length; the 2F-85's four-bar linkage departs from it
by up to 2.2 mm (measured between the pad faces), and a closed gripper reads about 2 mm."""


def _robotiq_home(model: mujoco.MjModel) -> mujoco.MjData:
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    return data


def test_robotiq_profile_is_a_dataset_arm_with_the_franka_hands_channels() -> None:
    validate_profile(FRANKA_FR3_ROBOTIQ_PROFILE)
    assert get_profile("franka_fr3_robotiq") is FRANKA_FR3_ROBOTIQ_PROFILE
    assert FRANKA_FR3_ROBOTIQ_PROFILE.tier == "dataset"  # the hardware carries the Franka Hand: not a twin
    assert FRANKA_FR3_ROBOTIQ_PROFILE.default_unit == FRANKA_FR3_PROFILE.default_unit
    assert supported_arm_counts(FRANKA_FR3_ROBOTIQ_PROFILE) == (1, 2)
    assert tuple(channel.name for channel in robot_layout(FRANKA_FR3_ROBOTIQ_PROFILE).channels) == FR3_NAMES
    assert MuJoCoRobot(profile="franka_fr3_robotiq", scene="single_pick_place", cameras=[]).joint_names == list(FR3_NAMES)
    entries = {(entry.profile.name, entry.bimanual) for entry in list_catalog_entries()}
    assert ("franka_fr3_robotiq", False) in entries
    assert ("franka_fr3_robotiq", True) not in entries


def test_robotiq_gripper_is_a_servo_on_the_finger_tendon_in_the_menagerie_range() -> None:
    model = _model(FRANKA_FR3_ROBOTIQ_PROFILE)
    actuator = model.actuator("gripper")
    # Menagerie's drivers span 0..0.8 rad; ctrl is the tendon's target length, the channel converts to a width.
    np.testing.assert_allclose(model.actuator_ctrlrange[actuator.id], (0.0, 0.8))
    np.testing.assert_allclose(model.actuator_forcerange[actuator.id], (-5.0, 5.0))
    assert model.actuator_trnid[actuator.id, 0] == model.tendon("robotiq_split").id


def test_robotiq_mounts_across_the_reach_as_on_droid() -> None:
    """At joint 7 = 0 the pads face each other across the reach (the arm reaches along +x), pointing down."""
    model = _model(FRANKA_FR3_ROBOTIQ_PROFILE)
    data = _robotiq_home(model)
    right, left = model.geom("robotiq_right_pad1").id, model.geom("robotiq_left_pad1").id
    closing = data.geom_xpos[left] - data.geom_xpos[right]
    assert abs(closing[1]) / np.linalg.norm(closing) > 0.99
    assert data.site("robotiq_pinch").xmat.reshape(3, 3)[:, 2] @ np.array([0.0, 0.0, -1.0]) > 0.8


def test_robotiq_home_is_open_with_no_contacts() -> None:
    model = _model(FRANKA_FR3_ROBOTIQ_PROFILE)
    data = _robotiq_home(model)
    assert data.ncon == 0
    assert data.ctrl[model.actuator("gripper").id] == 0.0  # the tendon's open length
    assert model.nexclude > 0  # Franka's disabled self-collision pairs


def test_robotiq_gripper_reads_and_takes_the_opening_width() -> None:
    """Command and state share the Franka Hand's unit and direction: width in metres, 0 closed."""
    robot = MuJoCoRobot(profile="franka_fr3_robotiq", scene="single_pick_place", cameras=[])
    robot.connect()
    try:
        observation = robot.get_observation()
        assert observation.joint_positions[-1] == pytest.approx(ROBOTIQ_WIDTH_M, abs=ROBOTIQ_WIDTH_TOLERANCE_M)
        arm = observation.joint_positions.copy()
        widths = []
        for width in (0.0, 0.03, 0.06, ROBOTIQ_WIDTH_M):
            action = arm.copy()
            action[-1] = width
            robot.send_action(action)
            for _ in range(300):
                observation = robot.get_observation()
            widths.append(float(observation.joint_positions[-1]))
            assert widths[-1] == pytest.approx(width, abs=ROBOTIQ_WIDTH_TOLERANCE_M)
        assert widths == sorted(widths)
        np.testing.assert_allclose(observation.joint_positions[:-1], arm[:-1], atol=0.5)
    finally:
        robot.disconnect()


def test_robotiq_gripper_closes_on_a_block_and_holds_it() -> None:
    probe = _model(FRANKA_FR3_ROBOTIQ_PROFILE)
    home = _robotiq_home(probe)
    spec = load_robot_spec(FRANKA_FR3_ROBOTIQ_PROFILE)
    spec.delete(spec.key("home"))
    block = spec.worldbody.add_body(name="block", pos=list(home.site("robotiq_pinch").xpos))
    block.add_freejoint()
    block.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.02] * 3, mass=0.05, friction=[1.0, 0.005, 0.0001])
    model = spec.compile()
    data = mujoco.MjData(model)
    data.qpos[:7] = home.qpos[:7]
    data.ctrl[:7] = home.ctrl[:7]
    start = data.body("block").xpos.copy()
    mujoco.mj_forward(model, data)

    data.ctrl[model.actuator("gripper").id] = 0.8  # closed
    for _ in range(3000):
        mujoco.mj_step(model, data)

    channel = FRANKA_FR3_ROBOTIQ_PROFILE.channels[-1]
    width = channel.offset + channel.scale * data.actuator_length[model.actuator("gripper").id]
    assert width == pytest.approx(0.04, abs=ROBOTIQ_WIDTH_TOLERANCE_M)  # stopped by the block
    assert data.body("block").xpos[2] > start[2] - 0.01  # still between the pads
