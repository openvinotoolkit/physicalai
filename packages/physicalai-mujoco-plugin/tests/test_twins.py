# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Twin profiles: the WidowX AI and the reBot B601 under their real drivers' names and units (TST-3).

The parity tests build the real driver classes with mocked hardware and compare their joint names,
``state`` length and ``sensor_data`` keys with the simulation's. They skip when the driver's own
dependency (``trossen-arm``, ``motorbridge``) is not installed.
"""

# MuJoCo model/data attributes are supplied by the C extension at runtime.
# pyrefly: ignore-errors [missing-attribute]

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import mujoco
import numpy as np
import pytest
from defusedxml import ElementTree

from physicalai.robot.interface import Robot
from physicalai_mujoco_plugin.compose import fetch_profile, load_scene_model
from physicalai_mujoco_plugin.profiles import get_profile, validate_profile
from physicalai_mujoco_plugin.profiles.rebot_b601 import REBOT_B601_GRIPPER_SCALE, REBOT_B601_PROFILE
from physicalai_mujoco_plugin.profiles.trossen_wxai import TROSSEN_WXAI_PROFILE
from physicalai_mujoco_plugin.robot import MuJoCoRobot
from physicalai_mujoco_plugin.scene_registry import bimanual_scene_id, get_scene, list_scenes_naming

WXAI_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_yaw", "wrist_roll", "gripper")


def _names(profile: object) -> tuple[str, ...]:
    return tuple(channel.name for channel in profile.channels)  # type: ignore[attr-defined]


def _real_widowx(role: str = "follower") -> Robot:
    trossen = pytest.importorskip("physicalai.robot.trossen", reason="needs trossen-arm")
    robot = trossen.WidowXAI("192.0.2.1", role=role)
    driver = MagicMock()
    driver.get_all_positions.return_value = [0.0] * 7
    driver.get_all_velocities.return_value = [0.0] * 7
    driver.get_all_external_efforts.return_value = [0.0] * 7
    robot._driver = driver  # noqa: SLF001 - mocked hardware
    return robot


def _real_rebot(positions: np.ndarray | None = None, velocities: np.ndarray | None = None) -> Robot:
    """Return a ``ReBotB601RS`` whose mocked motors report *positions* and *velocities* in its joint frame.

    The motors get the driver's own inverse: ``motor = radians(joint * direction)``.
    """
    rs = pytest.importorskip("physicalai_rebot_b601_plugin.rs", reason="needs the reBot B601 plugin and motorbridge")
    robot = rs.ReBotB601RS()
    positions = np.zeros(len(robot.JOINT_ORDER)) if positions is None else positions
    velocities = np.zeros(len(robot.JOINT_ORDER)) if velocities is None else velocities
    robot._controller = MagicMock()  # noqa: SLF001 - mocked hardware
    robot._motors = {  # noqa: SLF001
        name: MagicMock(
            get_state=MagicMock(
                return_value=SimpleNamespace(
                    pos=np.radians(position * rs.REBOT_B601_RS_JOINT_DIRECTIONS[name]),
                    vel=np.radians(velocity * rs.REBOT_B601_RS_JOINT_DIRECTIONS[name]),
                    torq=0.0,
                    t_mos=25.0,
                    t_rotor=25.0,
                    status_code=0,
                ),
            ),
        )
        for name, position, velocity in zip(robot.JOINT_ORDER, positions, velocities, strict=True)
    }
    return robot


class TestProfiles:
    def test_widowx_ai_uses_the_real_drivers_names_and_units(self) -> None:
        assert get_profile("trossen_wxai") is TROSSEN_WXAI_PROFILE
        validate_profile(TROSSEN_WXAI_PROFILE)
        assert _names(TROSSEN_WXAI_PROFILE) == WXAI_NAMES
        assert (TROSSEN_WXAI_PROFILE.tier, TROSSEN_WXAI_PROFILE.default_unit) == ("twin", "degrees")
        # Runtime's WidowXAI returns its velocities unconverted (OBS-4).
        assert {channel.velocity_unit for channel in TROSSEN_WXAI_PROFILE.channels} == {"model"}
        assert [camera.name for camera in TROSSEN_WXAI_PROFILE.cameras or ()] == ["wrist"]

    def test_widowx_ai_matches_runtimes_joint_order(self) -> None:
        constants = pytest.importorskip("physicalai.robot.trossen.constants", reason="needs trossen-arm")
        assert constants.WIDOWXAI_JOINT_ORDER == _names(TROSSEN_WXAI_PROFILE)

    def test_rebot_b601_matches_the_rs_drivers_joint_order(self) -> None:
        constants = pytest.importorskip("physicalai_rebot_b601_plugin.constants", reason="needs the reBot B601 plugin")
        assert get_profile("rebot_b601") is REBOT_B601_PROFILE
        validate_profile(REBOT_B601_PROFILE)
        assert _names(REBOT_B601_PROFILE) == constants.REBOT_B601_RS_JOINT_ORDER
        gripper = REBOT_B601_PROFILE.channels[-1]
        assert (gripper.actuators, gripper.unit, gripper.member_scales) == (("joint_left", "joint_right"), "degrees", (1.0,))
        # The driver's 0..270 motor degrees over its direction 6, across Menagerie's 0..0.05 m finger stroke.
        motor_open = constants.REBOT_B601_RS_JOINT_LIMITS_DEG["gripper"][1]
        assert gripper.scale == pytest.approx(motor_open / constants.REBOT_B601_RS_JOINT_DIRECTIONS["gripper"] / 0.05)

    def test_scenes_list_the_twins(self) -> None:
        assert set(list_scenes_naming("trossen_wxai")) == {"single_pick_place", "garment_fold"}
        assert set(list_scenes_naming("rebot_b601")) == {"single_pick_place"}
        assert bimanual_scene_id("so101") == bimanual_scene_id("trossen_wxai") == "garment_fold"
        assert bimanual_scene_id("rebot_b601") is None


@pytest.mark.requires_download
class TestRebotMapping:
    """The provisional reBot mapping follows the B601 plugin's joint-frame URDF; not verified on hardware."""

    def test_signs_follow_the_joint_frame_urdf(self) -> None:
        urdf = pytest.importorskip("physicalai_rebot_b601_plugin", reason="needs the reBot B601 plugin").get_urdf_path()
        root = ElementTree.parse(urdf / "rebot-b601-rs/urdf/00-arm-rs_asm-v3_joint_frame.urdf").getroot()
        urdf_axes = {
            joint.get("name"): np.array([float(v) for v in joint.find("axis").get("xyz").split()])
            for joint in root.iter("joint")
            if joint.get("type") == "revolute"
        }
        model = mujoco.MjModel.from_xml_path(str(fetch_profile(REBOT_B601_PROFILE)))
        for channel in REBOT_B601_PROFILE.channels[:-1]:
            (joint,) = channel.actuators
            # Same link frames (the zero pose), so only the axis direction can differ.
            assert channel.scale == float(np.sign(model.joint(joint).axis @ urdf_axes[joint])), channel.name
            assert channel.offset == 0.0

    def test_gripper_drives_both_fingers_in_driver_units(self) -> None:
        robot = MuJoCoRobot("rebot_b601", scene="single_pick_place", cameras=[])
        robot.connect()
        try:
            np.testing.assert_allclose(robot.get_observation().joint_positions, 0.0, atol=0.5)  # zero = folded rest pose
            target = np.array([20.0, 30.0, -40.0, 10.0, 0.0, 0.0, 45.0], dtype=np.float32)
            robot.send_action(target)
            for _ in range(100):
                observation = robot.get_observation()
            np.testing.assert_allclose(observation.joint_positions, target, atol=1.0)
            data = robot._sim.data  # noqa: SLF001
            for finger in ("joint_left", "joint_right"):
                assert data.joint(finger).qpos[0] == pytest.approx(45.0 / REBOT_B601_GRIPPER_SCALE, abs=1e-3)
        finally:
            robot.disconnect()

    def test_gripper_position_and_velocity_are_in_driver_degrees(self) -> None:
        robot = MuJoCoRobot("rebot_b601", scene="single_pick_place", cameras=[])
        robot.connect()
        try:
            robot.send_action(np.array([0, 0, 0, 0, 0, 0, 45.0], dtype=np.float32))
            observation = robot.get_observation()  # one tick: mid-stroke, still moving
            finger = robot._sim.data.joint("joint_left")  # noqa: SLF001
            gripper, speed = observation.joint_positions[-1], observation.sensor_data["velocities"][-1]
            assert 0.5 < gripper < 45.0
            assert gripper == pytest.approx(REBOT_B601_GRIPPER_SCALE * finger.qpos[0], rel=1e-5)
            assert speed == pytest.approx(REBOT_B601_GRIPPER_SCALE * finger.qvel[0], rel=1e-5)
            assert speed > 0.1
            # The real driver reads the same values from motors at 6x those degrees (motor 0..270).
            real = _real_rebot(observation.joint_positions, observation.sensor_data["velocities"])
            real_observation = real.get_observation()
            np.testing.assert_allclose(real_observation.joint_positions, observation.joint_positions, rtol=1e-5, atol=1e-4)
            np.testing.assert_allclose(
                real_observation.sensor_data["velocities"], observation.sensor_data["velocities"], rtol=1e-5, atol=1e-4
            )
        finally:
            robot.disconnect()


@pytest.mark.requires_download
class TestWidowXParity:
    def test_single_arm_matches_the_real_driver(self) -> None:
        real = _real_widowx()
        robot = MuJoCoRobot("trossen_wxai", scene="single_pick_place", cameras=[])
        assert robot.joint_names == list(real.joint_names)  # before connect (DRV-6)
        robot.connect()
        try:
            real_observation, observation = real.get_observation(), robot.get_observation()
            assert observation.state.shape == real_observation.state.shape
            # The real follower's efforts are not simulated: absent rather than zero (OBS-4).
            assert set(observation.sensor_data) == {"velocities"} < set(real_observation.sensor_data)
            assert robot._http_status()["units"] == ["degrees"] * 6 + ["metres"]  # noqa: SLF001

            robot.send_action(observation.joint_positions + np.array([20, 0, 0, 0, 0, 0, 0], dtype=np.float32))
            observation = robot.get_observation()
            data = robot._sim.data  # noqa: SLF001
            # Velocities in model units, rad/s and m/s, like the real driver's.
            expected = [data.joint(name).qvel[0] for name in ("joint_0", "right_carriage_joint")]
            np.testing.assert_allclose(observation.sensor_data["velocities"][[0, 6]], expected, rtol=1e-6)
            assert abs(expected[0]) > 0.01
        finally:
            robot.disconnect()

    def test_tracks_targets_in_degrees_and_metres(self) -> None:
        robot = MuJoCoRobot("trossen_wxai", scene="single_pick_place", cameras=[])
        robot.connect()
        try:
            target = np.array([20.0, 60.0, 50.0, -20.0, 10.0, 30.0, 0.03], dtype=np.float32)
            robot.send_action(target)
            for _ in range(100):
                observation = robot.get_observation()
            # Menagerie's position servos droop a few degrees under gravity.
            np.testing.assert_allclose(observation.joint_positions[:-1], target[:-1], atol=5.0)
            assert observation.joint_positions[-1] == pytest.approx(0.03, abs=1e-3)
        finally:
            robot.disconnect()

    def test_wrist_camera_replaces_menageries(self) -> None:
        model = load_scene_model(get_scene("single_pick_place").scene_xml_path, TROSSEN_WXAI_PROFILE)
        cameras = {model.camera(i).name for i in range(model.ncam)}
        assert cameras == {"overview", "wrist"}
        wrist = model.camera("wrist")
        assert model.body(wrist.bodyid[0]).name == "camera_link"
        assert wrist.fovy[0] == pytest.approx(58.0)
        assert not model.cam_sensorsize[wrist.id].any()  # a plain field of view: undistorted at any image size

    @pytest.mark.slow
    def test_bimanual_matches_the_real_bimanual_driver(self) -> None:
        trossen = pytest.importorskip("physicalai.robot.trossen", reason="needs trossen-arm")
        real = trossen.BimanualWidowXAI(_real_widowx(), _real_widowx())
        robot = MuJoCoRobot("trossen_wxai", scene="garment_fold", cameras=[])
        assert robot.joint_names == list(real.joint_names)
        robot.connect()
        try:
            observation, real_observation = robot.get_observation(), real.get_observation()
            assert observation.state.shape == real_observation.state.shape == (14,)
            assert set(observation.sensor_data) == {"velocities"} < set(real_observation.sensor_data)
            assert {camera["name"] for camera in robot._http_status()["cameras"]} <= {  # noqa: SLF001
                "left_wrist",
                "right_wrist",
                "overview",
            }
        finally:
            robot.disconnect()


@pytest.mark.requires_download
def test_rebot_wrist_camera_sits_on_the_d405_mount() -> None:
    robot = MuJoCoRobot("rebot_b601", scene="single_pick_place", cameras=[])
    robot.connect()
    try:
        assert robot._sim.camera_sources()["wrist"] == "override"  # noqa: SLF001
        model, data = robot._model, robot._data  # noqa: SLF001
        wrist = model.camera("wrist").id
        link6 = model.body("link6").id
        assert int(model.cam_bodyid[wrist]) == link6  # rolls with the gripper
        assert model.cam_fovy[wrist] == pytest.approx(58.0)  # the D405's vertical field of view
        # In link6 (+z = roll axis toward the fingers): the front face 104 mm off the axis on the
        # assumed +x side, 39 mm along it; looking 30 degrees toward the axis, image up away from it.
        frame = data.xmat[link6].reshape(3, 3)
        camera = frame.T @ data.cam_xmat[wrist].reshape(3, 3)
        np.testing.assert_allclose(frame.T @ (data.cam_xpos[wrist] - data.xpos[link6]), [0.10414, 0, 0.03915], atol=1e-9)
        np.testing.assert_allclose(-camera[:, 2], [-0.5, 0, np.sqrt(3) / 2], atol=1e-9)  # forward
        np.testing.assert_allclose(camera[:, 1], [np.sqrt(3) / 2, 0, 0.5], atol=1e-9)  # image up
        np.testing.assert_allclose(camera[:, 0], [0, 1, 0], atol=1e-9)  # image right
    finally:
        robot.disconnect()


@pytest.mark.requires_download
def test_rebot_matches_the_real_rs_driver() -> None:
    real = _real_rebot()
    robot = MuJoCoRobot("rebot_b601", scene="single_pick_place", cameras=[])
    assert robot.joint_names == list(real.joint_names)
    robot.connect()
    try:
        real_observation, observation = real.get_observation(), robot.get_observation()
        assert observation.state.shape == real_observation.state.shape
        # Torques, temperatures and status codes are not simulated: absent rather than zero.
        assert set(observation.sensor_data) == {"velocities"} < set(real_observation.sensor_data)
        # Every RS channel, the gripper too, is in the driver's degrees.
        assert robot._http_status()["units"] == ["degrees"] * 7  # noqa: SLF001
        assert robot._sim.camera_sources()["wrist"] == "override"  # noqa: SLF001
        assert robot._http_status()["viewer_url"] is None  # noqa: SLF001 - no viewer was opened
    finally:
        robot.disconnect()
