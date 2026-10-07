# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Floating-base robots: spawn anchors, base holds, observations, reset and falls (PR 3)."""

# MuJoCo's bindings are supplied by the C extension at runtime.
# pyrefly: ignore-errors [missing-attribute]

from __future__ import annotations

import dataclasses
import time
from pathlib import Path
from uuid import uuid4

import mujoco
import numpy as np
import pytest

from physicalai.config import Config
from physicalai.robot import SharedRobot
from physicalai.robot.interface import Robot
from physicalai_mujoco_plugin import compose
from physicalai_mujoco_plugin.compose import BASE_HOLD_EQUALITY, RobotBinding, compose_scene
from physicalai_mujoco_plugin.floating import FALL_DURATION_S, FloatingBases, body_frame, tilt_degrees
from physicalai_mujoco_plugin.profiles import RobotProfile
from physicalai_mujoco_plugin.profiles.derive import derive_profile
from physicalai_mujoco_plugin.robot import MuJoCoRobot

_FLOATER = """<mujoco model="floater">
  <compiler angle="radian"/>
  <worldbody>
    <body name="torso" pos="0 0 0.5">
      <freejoint name="root"/>
      <geom type="box" size="0.1 0.1 0.05" mass="2"/>
      <body name="leg" pos="0 0 -0.05">
        <joint name="knee" axis="0 1 0" range="-1 1"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.2" size="0.02" mass="0.2"/>
      </body>
    </body>
  </worldbody>
  <actuator><position name="knee" joint="knee" kp="20"/></actuator>
  <sensor><gyro name="gyro" site="imu"/></sensor>
  <keyframe><key name="home" qpos="5 6 0.3 0.7071068 0 0 0.7071068 0.4" ctrl="0.4"/></keyframe>
</mujoco>"""

_SPAWN_SCENE = """<mujoco><compiler angle="radian"/><option timestep="0.002"/><worldbody>
  <geom name="floor" type="plane" size="0 0 0.05"/>
  <frame name="robot_spawn" pos="1 2 0" euler="0 0 1.5707963267948966"/>
</worldbody></mujoco>"""


def _floater_xml() -> str:
    # The gyro needs a site on the torso.
    return _FLOATER.replace('<geom type="box"', '<site name="imu"/><geom type="box"')


@pytest.fixture
def floater(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> RobotProfile:
    """A tiny floating-base robot served in place of a Menagerie download."""
    robot = tmp_path / "floater.xml"
    robot.write_text(_floater_xml())
    monkeypatch.setattr(compose, "fetch_profile", lambda _profile: robot)
    return RobotProfile(name=f"floater-{uuid4().hex}", display_name="Floater", menagerie_model="floater")


def test_spawn_places_the_base_at_the_frame_at_the_keyframe_height(tmp_path: Path, floater: RobotProfile) -> None:
    """SCN-1/SCN-10: x/y and yaw from the frame, height and orientation from the home keyframe."""
    scene = tmp_path / "scene.xml"
    scene.write_text(_SPAWN_SCENE)
    composed = compose_scene(scene, floater)
    model = composed.model
    (binding,) = composed.robots
    base = binding.layout.base

    assert model.nkey == 0
    assert (base.body, base.joint) == ("torso", "root")
    # Keyframe yaw (90 deg) after the frame's yaw (90 deg) faces -x; the keyframe's x/y (5, 6) is ignored.
    np.testing.assert_allclose(base.home, [1.0, 2.0, 0.3, 0.0, 0.0, 0.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(model.qpos0[base.qpos_adr : base.qpos_adr + 7], base.home)
    assert binding.base_hold == mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, BASE_HOLD_EQUALITY)
    # The weld's reference is the spawn pose: no residual there.
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(data.efc_pos[: data.ne], 0.0, atol=1e-9)
    assert binding.layout.home_qpos == {"knee": (0.4,)}
    assert [sensor.name for sensor in binding.layout.sensors] == ["gyro"]


def test_a_raised_tilted_spawn_frame_gives_only_its_x_y_and_yaw(tmp_path: Path, floater: RobotProfile) -> None:
    """SCN-1: the frame's height, roll and pitch do not lift or tilt the robot; nested frames resolve."""
    scene = tmp_path / "scene.xml"
    scene.write_text(
        _SPAWN_SCENE.replace(
            '<frame name="robot_spawn" pos="1 2 0" euler="0 0 1.5707963267948966"/>',
            '<frame pos="0.5 0 0.4" euler="0.1 0 0"><frame name="robot_spawn" pos="0.5 2 0.5" '
            'euler="0.3 0.2 1.5707963267948966"/></frame>',
        )
    )
    composed = compose_scene(scene, floater)
    (binding,) = composed.robots
    base = binding.layout.base
    probe = mujoco.MjSpec.from_file(str(scene))
    probe.frame("robot_spawn").add_body(name="probe")
    probe_model = probe.compile()
    probe_data = mujoco.MjData(probe_model)
    mujoco.mj_kinematics(probe_model, probe_data)
    frame_x, frame_y = probe_data.xpos[probe_model.body("probe").id][:2]
    rotation = probe_data.xmat[probe_model.body("probe").id].reshape(3, 3)
    yaw = np.arctan2(rotation[1, 0], rotation[0, 0])

    # x/y and yaw from the frame; height 0.3 and the keyframe's own yaw of 90 deg from the keyframe.
    np.testing.assert_allclose(base.home[:3], [frame_x, frame_y, 0.3], atol=1e-9)
    total = yaw + np.pi / 2
    np.testing.assert_allclose(np.abs(base.home[3:]), np.abs([np.cos(total / 2), 0.0, 0.0, np.sin(total / 2)]), atol=1e-9)
    assert tilt_degrees(np.array(base.home[3:])) == pytest.approx(0.0, abs=1e-6)


def test_floating_bases_observe_the_base_in_the_body_frame() -> None:
    model = mujoco.MjModel.from_xml_string(_floater_xml())
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.qvel[:6] = [1.0, 0.0, 0.0, 0.1, 0.2, 0.3]
    mujoco.mj_forward(model, data)
    bases = FloatingBases(data, (RobotBinding("", derive_profile(model)),))

    entries, state = bases.observe()

    # The base faces +y (yaw 90 deg), so world +x is its -y.
    np.testing.assert_allclose(entries["base_linvel"], [0.0, -1.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(entries["base_angvel"], [0.1, 0.2, 0.3], atol=1e-6)
    np.testing.assert_allclose(entries["base_pos"], [5.0, 6.0, 0.3], atol=1e-6)
    assert all(value.dtype == np.float32 for value in entries.values())
    expected = np.concatenate([entries["base_quat"], entries["base_angvel"], entries["base_linvel"], data.sensordata])
    np.testing.assert_array_equal(state, expected.astype(np.float32))


def test_tilt_and_body_frame_helpers() -> None:
    assert tilt_degrees(np.array([1.0, 0.0, 0.0, 0.0])) == pytest.approx(0.0)
    half = np.sqrt(0.5)
    assert tilt_degrees(np.array([half, half, 0.0, 0.0])) == pytest.approx(90.0)
    assert tilt_degrees(np.array([0.0, 1.0, 0.0, 0.0])) == pytest.approx(180.0)
    np.testing.assert_allclose(body_frame(np.array([half, 0.0, 0.0, half]), [0.0, 1.0, 0.0]), [1.0, 0.0, 0.0], atol=1e-12)


def test_a_fall_needs_the_tilt_to_last() -> None:
    """DRV-9: tilted past 60 degrees for 1.5 s of simulated time."""
    model = mujoco.MjModel.from_xml_string(_floater_xml())
    data = mujoco.MjData(model)
    bases = FloatingBases(data, (RobotBinding("", derive_profile(model)),))
    tilted = np.array([np.cos(np.radians(35)), np.sin(np.radians(35)), 0.0, 0.0])  # 70 degrees
    data.qpos[3:7] = tilted

    assert not bases.update_falls(0.0)
    assert not bases.update_falls(FALL_DURATION_S - 0.01)
    assert not bases.status()[0].fallen
    assert bases.update_falls(FALL_DURATION_S)
    assert not bases.update_falls(FALL_DURATION_S + 0.5)  # reported once
    assert bases.status()[0].fallen
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    bases.update_falls(2.0 * FALL_DURATION_S)
    assert not bases.status()[0].fallen


@pytest.mark.requires_download
def test_g1_satisfies_the_robot_protocol_with_the_floating_state() -> None:
    """OBS-2: joints ‖ base_quat ‖ base_angvel ‖ base_linvel ‖ model sensors (G1: 29 + 10 + 4 IMUs x 3)."""
    robot = MuJoCoRobot("unitree_g1", scene="floor_flat", cameras=[])
    assert isinstance(robot, Robot)
    assert len(robot.joint_names) == 29  # before connect (DRV-6)
    assert Config.from_instance(robot).instantiate().profile.name == "unitree_g1"
    robot.connect()
    try:
        observation = robot.get_observation()
        sensors = observation.sensor_data
        assert observation.state.shape == (29 + 10 + 12,)
        assert observation.state.dtype == np.float32
        np.testing.assert_array_equal(observation.state[:29], observation.joint_positions)
        np.testing.assert_array_equal(
            observation.state[29:39], np.concatenate([sensors["base_quat"], sensors["base_angvel"], sensors["base_linvel"]])
        )
        assert set(sensors) == {
            "velocities",
            "base_pos",
            "base_quat",
            "base_angvel",
            "base_linvel",
            "sensor/imu-torso-angular-velocity",
            "sensor/imu-torso-linear-acceleration",
            "sensor/imu-pelvis-angular-velocity",
            "sensor/imu-pelvis-linear-acceleration",
        }
        assert sensors["base_pos"][2] == pytest.approx(0.79, abs=0.01)  # the keyframe's height (SCN-10)
        status = robot._http_status()  # noqa: SLF001
        assert (status["floating_base"], status["fallen"], status["compatible_scenes"]) == (True, False, ["floor_flat"])
        assert status["bases"][0]["held"] is True
        panel = robot._panel_state()  # noqa: SLF001
        assert (panel.profile, panel.tier, panel.units) == ("unitree_g1", "experimental", ("degrees",) * 29)
        assert panel.bases[0].held
        assert robot._switch_to_scene("single_pick_place") is False  # noqa: SLF001
    finally:
        robot.disconnect()


@pytest.mark.requires_download
def test_the_base_is_held_until_the_first_action_and_reset_holds_it_again() -> None:
    """SCN-8 and DRV-8 on the Go2 (torque actuators driven by the software PD)."""
    robot = MuJoCoRobot("unitree_go2", scene="floor_flat", cameras=[])
    robot.connect()
    try:
        start = robot.get_observation()
        assert robot._base_status[0].held  # noqa: SLF001
        robot.send_action(start.joint_positions + 10.0)
        for _ in range(25):
            robot.get_observation()
        assert not robot._base_status[0].held  # noqa: SLF001

        robot.reset()
        observation = robot.get_observation()
        assert robot._base_status[0].held  # noqa: SLF001
        np.testing.assert_allclose(observation.sensor_data["base_pos"], start.sensor_data["base_pos"], atol=2e-3)
        np.testing.assert_allclose(observation.joint_positions, start.joint_positions, atol=0.5)
    finally:
        robot.disconnect()


@pytest.mark.requires_download
@pytest.mark.parametrize("auto_reset", [False, True])
def test_a_fallen_robot_is_reported_and_reset_only_when_the_scene_asks(auto_reset: bool) -> None:  # noqa: FBT001
    robot = MuJoCoRobot("unitree_go2", scene="floor_flat", cameras=[])
    robot.connect()
    try:
        robot._scene_config = dataclasses.replace(robot._scene_config, auto_reset_on_fall=auto_reset)  # noqa: SLF001
        observation = robot.get_observation()
        robot.send_action(observation.joint_positions)
        base = robot._sim.bindings[0].layout.base  # noqa: SLF001
        robot._sim.data.qpos[base.qpos_adr + 3 : base.qpos_adr + 7] = [0.0, 1.0, 0.0, 0.0]  # upside down
        for _ in range(int(FALL_DURATION_S * 50) + 10):
            robot.get_observation()
        status = robot._http_status()  # noqa: SLF001
        if auto_reset:
            assert status["fallen"] is False
            assert status["bases"][0]["held"] is True
            assert status["bases"][0]["tilt_deg"] < 5.0
        else:
            assert status["fallen"] is True
            assert status["bases"][0]["held"] is False
    finally:
        robot.disconnect()


def test_so101_reset_keeps_the_arm_where_it_is() -> None:
    """Fixed bases keep the scene-only reset (SO-101 parity); ``POST /home`` homes them."""
    robot = MuJoCoRobot("so101", scene="single_pick_place", cameras=[], seed=3)
    robot.connect()
    try:
        robot.send_action(np.full(len(robot.joint_names), 20.0, dtype=np.float32))
        for _ in range(10):
            robot.get_observation()
        arm = robot._sim.channels[0]  # noqa: SLF001
        before = arm.read_positions()
        robot.reset()
        np.testing.assert_array_equal(arm.read_positions(), before)
        status = robot._http_status()  # noqa: SLF001
        assert (status["floating_base"], status["fallen"], status["bases"]) == (False, False, [])
        assert robot.get_observation().state.shape == (len(robot.joint_names),)
    finally:
        robot.disconnect()


def test_reset_needs_a_connection() -> None:
    with pytest.raises(ConnectionError):
        MuJoCoRobot("so101", cameras=[]).reset()


@pytest.mark.slow
@pytest.mark.requires_download
@pytest.mark.parametrize("profile", ["unitree_g1", "unitree_go2", "boston_dynamics_spot"])
def test_floating_base_smoke(profile: str, record_property: pytest.RecordProperty) -> None:
    """TST-5 for the experimental tier on floor_flat: finite and stable, held while idle, standing on hold."""
    robot = MuJoCoRobot(profile, scene="floor_flat", cameras=[])
    assert isinstance(robot, Robot)
    robot.connect()
    try:
        data = robot._sim.data  # noqa: SLF001
        start = robot.get_observation()
        height = float(start.sensor_data["base_pos"][2])
        rate = 50  # control ticks per simulated second (rate_hz)
        for _ in range(2 * rate):  # idle 2 s, base held
            observation = robot.get_observation()
            assert np.isfinite(observation.state).all()
        assert robot._base_status[0].held  # noqa: SLF001
        assert observation.sensor_data["base_pos"][2] == pytest.approx(height, abs=5e-3)
        np.testing.assert_allclose(observation.joint_positions, start.joint_positions, atol=5.0)

        robot.send_action(start.joint_positions)  # the hold action releases the base
        lowest = height
        began = time.perf_counter()
        for _ in range(3 * rate):
            observation = robot.get_observation()
            assert np.isfinite(observation.state).all()
            lowest = min(lowest, float(observation.sensor_data["base_pos"][2]))
        real_time_factor = 3.0 / (time.perf_counter() - began)
        assert data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number == 0
        assert not robot._base_status[0].held  # noqa: SLF001
        record_property("lowest_height_ratio", lowest / height)
        record_property("real_time_factor", real_time_factor)
        assert real_time_factor >= 1.5
        assert lowest >= 0.8 * height  # every experimental profile stands on its hold pose
    finally:
        robot.disconnect()


@pytest.mark.slow
@pytest.mark.requires_download
def test_floating_base_round_trips_through_shared_robot() -> None:
    """TST-4, floating base: joint names, the OBS-2 state and the base sensor data cross the transport."""
    robot = MuJoCoRobot("unitree_go2", scene="floor_flat", cameras=[])
    # A short idle timeout can let the owner exit before the first metadata reply (Zenoh connect retry).
    shared = SharedRobot.from_config(
        Config.from_instance(robot),
        name=f"mujoco-go2-floating-test-{uuid4().hex}",
        rate_hz=50.0,
        idle_timeout=10.0,
    )
    try:
        shared.connect()
        assert shared.joint_names == robot.joint_names
        observation = shared.get_observation()
        assert observation.joint_positions.shape == (12,)
        assert observation.state.shape == (12 + 10,)
        assert set(observation.sensor_data) == {"velocities", "base_pos", "base_quat", "base_angvel", "base_linvel"}
        np.testing.assert_allclose(observation.state[12:16], observation.sensor_data["base_quat"])
        shared.send_action(observation.joint_positions)
    finally:
        shared.disconnect()
