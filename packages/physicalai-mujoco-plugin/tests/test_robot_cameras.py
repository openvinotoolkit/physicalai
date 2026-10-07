# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Robot and scene cameras (CAM-2 ... CAM-7): sources, generated defaults, chase, names, headless hosts.

Nothing here renders: CI runners have no OpenGL. A camera's view is checked by casting rays
through a grid of its pixels instead (``_view``), which tells what the frame would show.
"""

# MuJoCo's bindings are supplied by the C extension at runtime.
# pyrefly: ignore-errors [missing-attribute]

from __future__ import annotations

import subprocess
from dataclasses import replace
from unittest.mock import patch

import mujoco
import numpy as np
import pytest
from loguru import logger

from physicalai_mujoco_plugin import compose
from physicalai_mujoco_plugin.cameras import offscreen_rendering_available
from physicalai_mujoco_plugin.compose import (
    ROBOT_SPAWN_FRAME,
    compose_scene,
    compose_scene_spec,
    load_robot_spec,
    load_scene_model,
    robot_layout,
)
from physicalai_mujoco_plugin.profiles import (
    SO101_PROFILE,
    CameraSpec,
    ChannelOverride,
    RobotProfile,
    get_profile,
    validate_profile,
)
from physicalai_mujoco_plugin.profiles.derive import derive_profile
from physicalai_mujoco_plugin.profiles.so101 import SO101_WRIST_CAMERA
from physicalai_mujoco_plugin.robot import MuJoCoRobot
from physicalai_mujoco_plugin.robot_cameras import _look_quat, add_cameras, is_sensor_camera, resolve_cameras
from physicalai_mujoco_plugin.scene_registry import SceneConfig
from physicalai_mujoco_plugin.sim import default_cameras

RENDERING_AVAILABLE = "physicalai_mujoco_plugin.robot.offscreen_rendering_available"
INLINE = RobotProfile(name="inline", display_name="Inline", menagerie_model="inline")
"""A profile without overrides, for robot models written inline (nothing is fetched)."""
VISIBLE_GROUPS = np.array([1, 1, 1, 0, 0, 0], dtype=np.uint8)
"""Geom groups a default ``mujoco.Renderer`` scene shows."""

_ARM = """<mujoco><compiler angle="radian"/><worldbody>
  <body name="base" pos="0 0 0.5"><geom type="cylinder" size="0.05 0.05"/>
    <body name="upper" pos="0 0 0.05"><joint name="shoulder" axis="0 1 0" range="-2 2"/>
      <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.03"/>
      <body name="hand" pos="0.3 0 0" euler="0 3.14159265 0">
        <joint name="wrist" axis="0 0 1" range="-2 2"/>
        <geom type="box" size="0.03 0.06 0.03" pos="0 0 0.03"/>
        {gripper}
      </body>
    </body>
  </body></worldbody>
  <actuator><position name="shoulder" joint="shoulder" kp="20"/><position name="wrist" joint="wrist" kp="5"/>
  {gripper_actuator}</actuator></mujoco>"""
_GRIPPER = """<site name="pinch" pos="0 0 0.12"/>
        <body name="finger_left" pos="0 0.03 0.08"><joint name="finger" type="slide" axis="0 1 0" range="0 0.04"/>
          <geom type="box" size="0.01 0.01 0.03"/></body>"""
_GRIPPER_ACTUATOR = '<position name="finger" joint="finger" kp="50"/>'

_FLOATING = """<mujoco><compiler angle="radian"/><worldbody>
  <body name="torso" pos="0 0 1"><freejoint/><geom type="box" size="0.2 0.1 0.1"/>
    <body name="leg" pos="0 0 -0.1"><joint name="hip" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0 0 0 -0.4" size="0.03"/></body>
    {head}
  </body></worldbody><actuator><motor name="hip" joint="hip"/></actuator></mujoco>"""
_HEAD = '<body name="head_link" pos="0.1 0 0.25"><geom type="sphere" size="{radius}"/></body>'


def _layout_cameras(xml: str, profile: RobotProfile = INLINE) -> tuple[CameraSpec, ...]:
    model = mujoco.MjModel.from_xml_string(xml)
    return resolve_cameras(profile, model, derive_profile(model))


def _with_cameras(xml: str, cameras: tuple[CameraSpec, ...], *, floor_z: float = 0.0) -> tuple[object, object]:
    """Compile *xml* with *cameras* added and a floor, at ``qpos0``.

    Returns:
        The model and its data, after forward kinematics.
    """
    spec = mujoco.MjSpec.from_string(xml)
    add_cameras(spec, cameras, "test")
    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[5, 5, 0.1], pos=[0, 0, floor_z])
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def _view(model: object, data: object, camera: str, columns: int = 16, rows: int = 12) -> list[int]:
    """Cast one ray per cell of a 4:3 pixel grid through *camera*.

    Returns:
        The geom each ray hits first, ``-1`` for sky.
    """
    camera_id = model.camera(camera).id
    half_height = np.tan(np.radians(model.cam_fovy[camera_id]) / 2)
    rotation = data.cam_xmat[camera_id].reshape(3, 3)
    hit = np.zeros(1, dtype=np.int32)
    hits = []
    for v in np.linspace(-1, 1, rows):
        for u in np.linspace(-1, 1, columns):
            direction = rotation @ np.array([u * half_height * 4 / 3, v * half_height, -1.0])
            mujoco.mj_ray(model, data, data.cam_xpos[camera_id], direction, VISIBLE_GROUPS, 1, -1, hit)
            hits.append(int(hit[0]))
    return hits


def _bodies_hit(model: object, hits: list[int]) -> set[str]:
    return {model.body(int(model.geom_bodyid[geom])).name for geom in hits if geom >= 0}


def _forward(model: object, data: object, camera: str) -> np.ndarray:
    """Return the camera's viewing direction in the world (its -z axis)."""
    return -data.cam_xmat[model.camera(camera).id].reshape(3, 3)[:, 2]


class TestSources:
    """CAM-2: the profile's override, else the model's sensor cameras, else a generated default."""

    @pytest.mark.parametrize(
        ("name", "body", "sensor"),
        [
            ("wrist_cam", "link6", True),
            ("egocentric", "head_tilt_link", True),
            ("overview", "world", False),
            ("track", "torso", False),
            ("tracking_cam", "torso", False),
            ("Front", "torso", False),
            ("perspective", "base", False),
        ],
    )
    def test_viewer_and_world_cameras_are_not_sensors(self, name: str, body: str, *, sensor: bool) -> None:
        camera = CameraSpec(name, body, (0, 0, 0), (1, 0, 0, 0), 45.0, source="model")
        assert is_sensor_camera(camera) is sensor

    def test_model_sensor_cameras_beat_generated_defaults(self) -> None:
        xml = _ARM.format(gripper='<camera name="hand_cam" pos="0 0.05 0"/>', gripper_actuator="")
        xml = xml.replace("<worldbody>", '<worldbody><camera name="track" mode="trackcom"/>')

        cameras = _layout_cameras(xml)

        assert [(camera.name, camera.body, camera.source) for camera in cameras] == [("hand_cam", "hand", "model")]

    def test_the_profile_override_wins_even_when_empty(self) -> None:
        xml = _ARM.format(gripper='<camera name="hand_cam" pos="0 0.05 0"/>', gripper_actuator="")
        override = CameraSpec("side_cam", "upper", (0.1, 0.1, 0.0), (1.0, 0.0, 0.0, 0.0), 60.0)

        assert _layout_cameras(xml, replace(INLINE, cameras=(override,))) == (override,)
        assert _layout_cameras(xml, replace(INLINE, cameras=())) == ()


class TestGeneratedDefaults:
    """CAM-3: one ``wrist``, ``head`` or ``front`` camera that sees the robot or its workspace."""

    def test_a_gripper_arm_gets_a_wrist_camera_on_the_gripper_parent(self) -> None:
        xml = _ARM.format(gripper=_GRIPPER, gripper_actuator=_GRIPPER_ACTUATOR)

        (camera,) = _layout_cameras(xml)
        model, data = _with_cameras(xml, (camera,))
        hits = _view(model, data, "wrist")

        assert (camera.name, camera.body, camera.source, camera.fovy) == ("wrist", "hand", "default", 75.0)
        # It looks along the tool axis (the hand's +z points down here) and sees the finger and the floor.
        assert _forward(model, data, "wrist")[2] < -0.9
        assert {"finger_left", "world"} <= _bodies_hit(model, hits)
        # It sits beside the hand, on the side the finger does not open towards (the hand's x).
        assert abs(camera.pos[0]) > 0.04
        assert camera.pos[1] == pytest.approx(0.0, abs=1e-9)

    def test_an_arm_without_gripper_gets_a_wrist_camera_on_its_last_link(self) -> None:
        xml = _ARM.format(gripper="", gripper_actuator="")

        (camera,) = _layout_cameras(xml)
        model, data = _with_cameras(xml, (camera,))
        hits = _view(model, data, "wrist")

        assert (camera.name, camera.body) == ("wrist", "hand")
        assert {"hand", "world"} <= _bodies_hit(model, hits)

    @pytest.mark.parametrize(("radius", "ahead"), [(0.05, 0.08), (0.12, 0.13)])
    def test_a_floating_base_with_a_head_gets_a_head_camera(self, radius: float, ahead: float) -> None:
        """8 cm ahead of the head, or 1 cm ahead of a head larger than that."""
        xml = _FLOATING.format(head=_HEAD.format(radius=radius))

        (camera,) = _layout_cameras(xml)
        model, data = _with_cameras(xml, (camera,))
        pitch = np.radians(20.0)

        assert (camera.name, camera.body, camera.source, camera.fovy) == ("head", "head_link", "default", 80.0)
        np.testing.assert_allclose(data.cam_xpos[model.camera("head").id], [0.1 + ahead, 0.0, 1.25], atol=1e-9)
        np.testing.assert_allclose(_forward(model, data, "head"), [np.cos(pitch), 0.0, -np.sin(pitch)], atol=1e-9)
        assert "world" in _bodies_hit(model, _view(model, data, "head"))

    def test_a_floating_base_without_a_head_gets_a_front_camera(self) -> None:
        xml = _FLOATING.format(head="")

        (camera,) = _layout_cameras(xml)
        model, data = _with_cameras(xml, (camera,))
        pitch = np.radians(15.0)
        hits = _view(model, data, "front")

        assert (camera.name, camera.body, camera.source, camera.fovy) == ("front", "torso", "default", 90.0)
        np.testing.assert_allclose(camera.pos, [0.21, 0.0, 0.0], atol=1e-9)  # the box's +x face, plus 1 cm
        np.testing.assert_allclose(_forward(model, data, "front"), [np.cos(pitch), 0.0, -np.sin(pitch)], atol=1e-9)
        assert -1 in hits
        assert "world" in _bodies_hit(model, hits)


class TestInjection:
    """CAM-7 and CAM-5: profile cameras enter the robot spec; names stay unique."""

    def test_the_so101_wrist_camera_keeps_its_earlier_pose(self) -> None:
        quat = np.zeros(4)
        mujoco.mju_euler2Quat(quat, np.array([0.57, 0.0, np.pi]), "xyz")
        model = load_robot_spec(SO101_PROFILE).compile()
        camera = model.camera("wrist")

        np.testing.assert_array_equal(SO101_WRIST_CAMERA.quat, quat)  # bit-identical: trained policies see it
        assert robot_layout(SO101_PROFILE).cameras == (SO101_WRIST_CAMERA,)
        assert [model.camera(i).name for i in range(model.ncam)] == ["wrist"]  # Menagerie's wrist_cam is gone
        assert model.body(int(model.cam_bodyid[camera.id])).name == "camera_mount"
        np.testing.assert_array_equal(model.cam_pos[camera.id], [0.0, -0.055, -0.045])
        assert model.cam_fovy[camera.id] == 75.0

    def test_a_camera_needs_its_body_and_a_free_name(self) -> None:
        xml = _ARM.format(gripper='<camera name="hand_cam"/>', gripper_actuator="")
        missing = CameraSpec("cam", "nowhere", (0, 0, 0), (1, 0, 0, 0), 60.0)
        taken = CameraSpec("hand_cam", "hand", (0, 0, 0), (1, 0, 0, 0), 60.0)

        with pytest.raises(ValueError, match="missing body 'nowhere'"):
            add_cameras(mujoco.MjSpec.from_string(xml), (missing,), "test")
        with pytest.raises(ValueError, match="clashes with a model camera"):
            add_cameras(mujoco.MjSpec.from_string(xml), (taken,), "test")

    def test_a_profile_model_camera_must_exist(self) -> None:
        """A profile that publishes a model camera by name fails loudly when the model lacks it."""
        xml = _ARM.format(gripper='<camera name="hand_cam"/>', gripper_actuator="")
        present = CameraSpec("hand_cam", "hand", (0, 0, 0), (1, 0, 0, 0), 60.0, source="model")
        missing = CameraSpec("wrist_cam", "hand", (0, 0, 0), (1, 0, 0, 0), 60.0, source="model")
        spec = mujoco.MjSpec.from_string(xml)

        add_cameras(spec, (present,), "test")
        assert [camera.name for camera in spec.cameras] == ["hand_cam"]
        with pytest.raises(ValueError, match="'test' camera 'wrist_cam' names a model camera the robot does not have"):
            add_cameras(spec, (missing,), "test")

    def test_look_quat_leaves_its_inputs_alone(self) -> None:
        forward, up = np.array([2.0, 0.0, -1.0]), np.array([0.0, 0.0, 3.0])

        quat = _look_quat(forward, up)

        np.testing.assert_array_equal(forward, [2.0, 0.0, -1.0])
        np.testing.assert_array_equal(up, [0.0, 0.0, 3.0])
        rotation = np.zeros(9)
        mujoco.mju_quat2Mat(rotation, np.array(quat))
        np.testing.assert_allclose(-rotation.reshape(3, 3)[:, 2], forward / np.linalg.norm(forward), atol=1e-12)

    @pytest.mark.parametrize(
        ("cameras", "message"),
        [
            ((CameraSpec("a", "b", (0, 0, 0), (1, 0, 0, 0), 60.0),) * 2, "unique"),
            ((CameraSpec("a", "b", (0, 0, 0), (0, 0, 0, 0), 60.0),), "quaternion"),
            ((CameraSpec("a", "b", (0, 0, float("nan")), (1, 0, 0, 0), 60.0),), "position"),
            ((CameraSpec("a", "b", (0, 0, 0), (1, 0, 0, 0), 180.0),), "field of view"),
        ],
    )
    def test_profile_cameras_are_validated(self, cameras: tuple[CameraSpec, ...], message: str) -> None:
        with pytest.raises(ValueError, match=message):
            validate_profile(replace(INLINE, cameras=cameras))

    def test_a_robot_camera_that_clashes_with_a_scene_camera_fails_at_connect(self, tmp_path) -> None:
        path = tmp_path / "clash.xml"
        path.write_text('<mujoco><worldbody><camera name="wrist" pos="0 -1 1"/><frame name="robot_mount"/></worldbody></mujoco>')
        robot = MuJoCoRobot("so101", model_path=str(path), cameras=[])

        with pytest.raises(ValueError, match="Camera 'wrist' appears twice"):
            robot.connect()
        assert not robot.is_connected()


class TestChaseCamera:
    """CAM-4: a scene's world camera ``chase`` follows the first robot's floating base."""

    _SCENE = _FLOATING.format(head="").replace(
        "<worldbody>",
        '<worldbody><camera name="overview" pos="0 -3 2" xyaxes="1 0 0 0 0.6 0.8"/>'
        '<camera name="chase" pos="-2 0 1.5" xyaxes="0 -1 0 0.5 0 0.87" fovy="50"/>',
    )

    def test_chase_starts_where_the_scene_put_it_and_follows_the_base(self, tmp_path) -> None:
        path = tmp_path / "floor.xml"
        path.write_text(self._SCENE)

        model = compose_scene(path, INLINE).model
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        chase = model.camera("chase").id
        start_pos, start_mat = data.cam_xpos[chase].copy(), data.cam_xmat[chase].copy()
        data.qpos[:3] += [1.0, 0.5, 0.0]  # walk the free joint
        mujoco.mj_forward(model, data)

        assert model.body(int(model.cam_bodyid[chase])).name == "torso"
        assert model.cam_mode[chase] == mujoco.mjtCamLight.mjCAMLIGHT_TRACKCOM
        assert model.cam_fovy[chase] == 50.0
        np.testing.assert_allclose(start_pos, [-2.0, 0.0, 1.5], atol=1e-9)
        np.testing.assert_allclose(data.cam_xpos[chase], start_pos + [1.0, 0.5, 0.0], atol=1e-9)
        np.testing.assert_allclose(data.cam_xmat[chase], start_mat, atol=1e-9)

    def test_chase_streams_after_the_overview(self, tmp_path) -> None:
        path = tmp_path / "floor.xml"
        path.write_text(self._SCENE)
        robot = MuJoCoRobot("ur5e", model_path=str(path), cameras=[])
        robot.connect()
        try:
            # The body has no camera of its own: a generated ``front`` leads (CAM-2), then the scene's.
            assert [config.name for config in default_cameras(robot._sim)] == ["front", "overview", "chase"]  # noqa: SLF001
        finally:
            robot.disconnect()

    def test_every_compile_path_makes_the_same_chase_camera(self, tmp_path) -> None:
        """``load_scene_model``, ``SceneConfig.load_model`` and ``compose_scene_spec`` match ``compose_scene``."""
        path = tmp_path / "floor.xml"
        path.write_text(self._SCENE)
        scene = SceneConfig(scene_id="floor", display_name="Floor", description="", scene_xml_relpath=str(path))

        expected = compose_scene(path, INLINE).model
        for model in (load_scene_model(path, INLINE), scene.load_model(INLINE), compose_scene_spec(path, INLINE).compile()):
            chase = model.camera("chase").id
            assert model.body(int(model.cam_bodyid[chase])).name == "torso"
            assert model.cam_mode[chase] == mujoco.mjtCamLight.mjCAMLIGHT_TRACKCOM
            np.testing.assert_array_equal(model.cam_pos0[chase], expected.cam_pos0[chase])
            np.testing.assert_array_equal(model.cam_mat0[chase], expected.cam_mat0[chase])

    def test_chase_follows_a_robot_attached_at_an_anchor(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        """The anchored path: the camera moves into the prefixed root body of the first spawned robot."""
        robot_xml = _FLOATING.format(head="").replace("<freejoint/>", '<freejoint name="root"/>')
        monkeypatch.setattr(compose, "_customized_robot_spec", lambda _profile: mujoco.MjSpec.from_string(robot_xml))
        monkeypatch.setattr(compose, "_LAYOUTS", {})
        path = tmp_path / "floor.xml"
        path.write_text(
            '<mujoco><worldbody><camera name="chase" pos="-2 0 1.5" xyaxes="0 -1 0 0.5 0 0.87"/>'
            f'<frame name="left_{ROBOT_SPAWN_FRAME}" pos="1 2 0"/><frame name="right_{ROBOT_SPAWN_FRAME}" pos="1 -2 0"/>'
            "</worldbody></mujoco>",
        )

        composed = compose_scene(path, INLINE)
        model = composed.model
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        chase = model.camera("chase").id
        start_pos, start_mat = data.cam_xpos[chase].copy(), data.cam_xmat[chase].copy()
        root = model.joint(composed.robots[0].layout.base.joint)
        data.qpos[root.qposadr[0] : root.qposadr[0] + 3] += [1.0, 0.5, 0.0]
        mujoco.mj_forward(model, data)

        assert [binding.prefix for binding in composed.robots] == ["left_", "right_"]
        assert model.body(int(model.cam_bodyid[chase])).name == "left_torso"
        assert model.cam_mode[chase] == mujoco.mjtCamLight.mjCAMLIGHT_TRACKCOM
        np.testing.assert_allclose(start_pos, [-2.0, 0.0, 1.5], atol=1e-9)
        np.testing.assert_allclose(data.cam_xpos[chase], start_pos + [1.0, 0.5, 0.0], atol=1e-9)
        np.testing.assert_allclose(data.cam_xmat[chase], start_mat, atol=1e-9)

    def test_a_fixed_base_leaves_the_chase_camera_alone(self, tmp_path) -> None:
        path = tmp_path / "arm.xml"
        path.write_text(_ARM.format(gripper="", gripper_actuator="").replace("<worldbody>", '<worldbody><camera name="chase"/>'))

        model = compose_scene(path, INLINE).model

        assert model.cam_bodyid[model.camera("chase").id] == 0
        assert model.cam_mode[model.camera("chase").id] == mujoco.mjtCamLight.mjCAMLIGHT_FIXED


_SO101_CHANNELS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


def _so101_chain(joint_prefix: str, body_prefix: str, x: float, camera_at: int | None = None, camera: str = "") -> str:
    """An SO-101-named six-joint chain; actuator/joint names carry *joint_prefix*, bodies *body_prefix*."""
    bodies = ""
    for index, name in enumerate(_SO101_CHANNELS):
        pos = f"{x} 0 0.1" if index == 0 else "0 0 0.05"
        bodies += (
            f'<body name="{body_prefix}link{index}" pos="{pos}"><joint name="{joint_prefix}{name}" axis="0 1 0"'
            ' range="-2 2"/><geom type="sphere" size=".02"/>'
        )
        if index == camera_at:
            bodies += f'<camera name="{camera}"/>'
    return bodies + "</body>" * len(_SO101_CHANNELS)


def _so101_actuators(joint_prefix: str) -> str:
    return "".join(f'<position name="{joint_prefix}{n}" joint="{joint_prefix}{n}" kp="20"/>' for n in _SO101_CHANNELS)


class TestCameraOwnership:
    """In a robot-complete model, a sensor camera belongs to the robot whose kinematic subtree holds it."""

    def _cameras(self, tmp_path, worldbody: str, actuators: str) -> list[tuple[str, list[tuple[str, str]]]]:
        path = tmp_path / "robots.xml"
        path.write_text(f"<mujoco><worldbody>{worldbody}</worldbody><actuator>{actuators}</actuator></mujoco>")
        scene = compose_scene(path, SO101_PROFILE)
        return [(robot.prefix, [(c.name, c.source) for c in robot.layout.cameras]) for robot in scene.robots]

    def test_body_names_need_not_carry_the_robot_prefix(self, tmp_path) -> None:
        chain = _so101_chain("left_", "", 0.0, camera_at=2, camera="sensor_cam")
        assert self._cameras(tmp_path, chain, _so101_actuators("left_")) == [("left_", [("sensor_cam", "model")])]

    def test_a_camera_on_a_shared_body_goes_to_the_first_robot(self, tmp_path) -> None:
        """A head camera on the torso both arms hang from is above both arm roots; it must not vanish."""
        arms = _so101_chain("left_", "left_", -0.3) + _so101_chain("right_", "right_", 0.3)
        worldbody = f'<body name="torso" pos="0 0 1"><geom type="box" size=".2 .2 .1"/><camera name="head_cam"/>{arms}</body>'
        cameras = dict(self._cameras(tmp_path, worldbody, _so101_actuators("left_") + _so101_actuators("right_")))
        assert ("head_cam", "model") in cameras["left_"]
        assert all(name != "head_cam" for name, _ in cameras["right_"])

    def test_tendon_driven_robots_own_their_cameras(self, tmp_path) -> None:
        """Tendon channels drive no joint directly; the dofs they move still define each robot."""
        worldbody = tendons = actuators = ""
        for side in ("left_", "right_"):
            worldbody += (
                f'<body name="{side}base"><joint name="{side}joint"/><geom type="sphere" size=".02"/>'
                f'<camera name="{side}cam"/></body>'
            )
            tendons += f'<fixed name="{side}tendon"><joint joint="{side}joint" coef="1"/></fixed>'
            actuators += f'<position name="{side}drive" tendon="{side}tendon" kp="20"/>'
        path = tmp_path / "tendons.xml"
        path.write_text(
            f"<mujoco><worldbody>{worldbody}</worldbody><tendon>{tendons}</tendon><actuator>{actuators}</actuator></mujoco>"
        )
        profile = RobotProfile(
            name="tendons",
            display_name="Tendons",
            menagerie_model="inline",
            channels=(ChannelOverride("drive", ("drive",), unit="metres"),),
        )
        scene = compose_scene(path, profile)
        assert [(r.prefix, [(c.name, c.source) for c in r.layout.cameras]) for r in scene.robots] == [
            ("left_", [("left_cam", "model")]),
            ("right_", [("right_cam", "model")]),
        ]

    def test_overlapping_prefixes_do_not_share_a_camera(self, tmp_path) -> None:
        worldbody = _so101_chain("arm", "arm", -0.3) + _so101_chain("arm2", "arm2", 0.3, camera_at=2, camera="arm2_cam")
        cameras = dict(self._cameras(tmp_path, worldbody, _so101_actuators("arm") + _so101_actuators("arm2")))
        assert cameras["arm2"] == [("arm2_cam", "model")]
        assert ("arm2_cam", "model") not in cameras["arm"]


class TestRobotCompleteModels:
    """A model XML without mount frames defines its own robot; its cameras resolve like a derived profile."""

    _OVERVIEW = '<camera name="overview" pos="0 -1 1" xyaxes="1 0 0 0 0.7 0.7"/>'

    def _robot(self, tmp_path, xml: str) -> MuJoCoRobot:
        path = tmp_path / "arm.xml"
        path.write_text(xml)
        return MuJoCoRobot("ur5e", model_path=str(path))

    def test_sensor_cameras_stream_and_viewer_cameras_do_not(self, tmp_path) -> None:
        arm = _ARM.format(gripper='<camera name="hand_cam" pos="0 0.05 0"/>', gripper_actuator="")
        xml = arm.replace(
            '<body name="upper"',
            '<camera name="tracking" mode="track" pos="0 -1 0"/><body name="upper"',
        ).replace("<worldbody>", f"<worldbody>{self._OVERVIEW}")
        robot = self._robot(tmp_path, xml)
        with patch("mujoco.Renderer"), patch(RENDERING_AVAILABLE, return_value=True):
            robot.connect()
            try:
                streams = [(camera["name"], camera["source"]) for camera in robot._http_status()["cameras"]]  # noqa: SLF001
                labels = robot._sim.camera_sources()  # noqa: SLF001
            finally:
                robot.disconnect()

        assert streams == [("hand_cam", "model"), ("overview", "scene")]
        assert labels == {"overview": "scene", "tracking": "scene", "hand_cam": "model"}

    def test_a_model_without_robot_cameras_gets_a_generated_default(self, tmp_path) -> None:
        xml = _ARM.format(gripper=_GRIPPER, gripper_actuator=_GRIPPER_ACTUATOR)
        xml = xml.replace("<worldbody>", f"<worldbody>{self._OVERVIEW}")
        path = tmp_path / "arm.xml"
        path.write_text(xml)

        composed = compose_scene(path, INLINE)
        model = composed.model

        assert [(camera.name, camera.source) for camera in composed.robots[0].layout.cameras] == [("wrist", "default")]
        assert model.body(int(model.cam_bodyid[model.camera("wrist").id])).name == "hand"
        assert load_scene_model(path, INLINE).camera("wrist").id >= 0
        robot = self._robot(tmp_path, xml)
        robot.connect()
        try:
            assert [config.name for config in default_cameras(robot._sim)] == ["wrist", "overview"]  # noqa: SLF001
        finally:
            robot.disconnect()


class TestGeneratedNames:
    """A generated default never takes a name the model already uses."""

    def test_a_viewer_camera_keeps_its_name_and_the_default_is_skipped(self, tmp_path) -> None:
        """A world viewer camera named ``front`` is not a sensor, and the generated ``front`` steps aside."""
        path = tmp_path / "floor.xml"
        path.write_text(_FLOATING.format(head="").replace("<worldbody>", '<worldbody><camera name="front" pos="2 0 1"/>'))
        messages: list[str] = []
        sink = logger.add(lambda message: messages.append(str(message)), level="DEBUG")
        try:
            composed = compose_scene(path, INLINE)
        finally:
            logger.remove(sink)
        model = composed.model

        assert [binding.layout.cameras for binding in composed.robots] == [()]
        assert [model.camera(i).name for i in range(model.ncam)] == ["front"]
        assert model.cam_bodyid[model.camera("front").id] == 0
        assert any("No default 'front' camera" in message for message in messages)


class TestHeadlessHosts:
    """A host where MuJoCo cannot render streams no default cameras, with one warning."""

    @pytest.fixture(autouse=True)
    def _fresh_probe(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(offscreen_rendering_available, "available", False)
        monkeypatch.setattr(offscreen_rendering_available, "warned", False)

    @pytest.fixture
    def arm_path(self, tmp_path) -> str:
        path = tmp_path / "arm.xml"
        path.write_text(
            _ARM.format(gripper="", gripper_actuator="").replace(
                "<worldbody>",
                '<worldbody><camera name="overview" pos="0 -1 1"/>',
            ),
        )
        return str(path)

    def test_the_probe_renders_in_a_child_process(self) -> None:
        done = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
        with patch("physicalai_mujoco_plugin.cameras.subprocess.run", return_value=done) as run:
            assert offscreen_rendering_available() is True
            assert offscreen_rendering_available() is True

        run.assert_called_once()
        assert run.call_args.kwargs["timeout"] > 0

    @pytest.mark.parametrize(
        "outcome",
        [
            subprocess.TimeoutExpired(cmd="python", timeout=10.0),
            subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="mujoco.FatalError: gladLoadGL error"),
        ],
    )
    def test_a_failed_probe_is_retried_with_one_warning(self, outcome: object) -> None:
        warnings: list[str] = []
        sink = logger.add(lambda message: warnings.append(str(message)), level="WARNING")
        side_effect = outcome if isinstance(outcome, BaseException) else None
        result = None if side_effect else outcome
        done = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
        try:
            with patch("physicalai_mujoco_plugin.cameras.subprocess.run", side_effect=side_effect, return_value=result) as run:
                assert offscreen_rendering_available() is False
                assert offscreen_rendering_available() is False
            with patch("physicalai_mujoco_plugin.cameras.subprocess.run", return_value=done) as recovered:
                assert offscreen_rendering_available() is True
                assert offscreen_rendering_available() is True
        finally:
            logger.remove(sink)

        assert run.call_count == 2  # a failure is not remembered
        recovered.assert_called_once()  # a success is
        assert len(warnings) == 1
        assert "MUJOCO_GL=egl" in warnings[0]

    def test_cameras_come_back_on_the_next_connect(self, arm_path) -> None:
        failed = subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="no GL")
        done = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
        robot = MuJoCoRobot("ur5e", model_path=arm_path)
        with patch("mujoco.Renderer"), patch("physicalai_mujoco_plugin.cameras.subprocess.run", return_value=failed):
            robot.connect()
            assert robot._cameras.configs == []  # noqa: SLF001
            robot.disconnect()
        with patch("mujoco.Renderer"), patch("physicalai_mujoco_plugin.cameras.subprocess.run", return_value=done):
            robot.connect()
            try:
                assert [config.name for config in robot._cameras.configs] == ["wrist", "overview"]  # noqa: SLF001
            finally:
                robot.disconnect()

    def test_default_cameras_are_disabled_without_rendering(self, arm_path) -> None:
        robot = MuJoCoRobot("ur5e", model_path=arm_path)
        with patch("mujoco.Renderer") as renderer, patch(RENDERING_AVAILABLE, return_value=False):
            robot.connect()
            try:
                assert robot._cameras.configs == []  # noqa: SLF001
                assert robot._http_status()["cameras"] == []  # noqa: SLF001
            finally:
                robot.disconnect()

        renderer.assert_not_called()

    def test_explicit_cameras_skip_the_probe(self, arm_path) -> None:
        robot = MuJoCoRobot("ur5e", model_path=arm_path, cameras=[{"name": "overview"}])
        with patch("mujoco.Renderer"), patch(RENDERING_AVAILABLE) as probe:
            robot.connect()
            try:
                assert [config.name for config in robot._cameras.configs] == ["overview"]  # noqa: SLF001
            finally:
                robot.disconnect()

        probe.assert_not_called()


@pytest.mark.requires_download
class TestMenagerieRobots:
    """Real Menagerie models, compiled only (no rendering)."""

    def test_ur5e_streams_a_generated_wrist_camera_and_the_overview(self) -> None:
        robot = MuJoCoRobot("ur5e", scene="single_pick_place")
        with patch("mujoco.Renderer"), patch(RENDERING_AVAILABLE, return_value=True):
            robot.connect()
            try:
                cameras = [(camera["name"], camera["source"]) for camera in robot._http_status()["cameras"]]  # noqa: SLF001
                model, data = robot._model, robot._data  # noqa: SLF001
                hits = _view(model, data, "wrist")
            finally:
                robot.disconnect()

        assert cameras == [("wrist", "default"), ("overview", "scene")]
        assert model.body(int(model.cam_bodyid[model.camera("wrist").id])).name == "wrist_3_link"
        # It sees the flange at the bottom of the image and the workspace beyond it.
        bodies = _bodies_hit(model, hits)
        assert "wrist_3_link" in bodies
        assert bodies - {"wrist_3_link"}

    def test_kinova_gen3_publishes_its_own_wrist_camera(self) -> None:
        cameras = robot_layout(get_profile("kinova_gen3")).cameras

        assert [(camera.name, camera.body, camera.source) for camera in cameras] == [("wrist", "bracelet_link", "model")]

    @pytest.mark.parametrize(("name", "root"), [("unitree_g1", "pelvis"), ("unitree_go2", "base")])
    def test_floating_bases_without_a_head_get_a_front_camera(self, name: str, root: str) -> None:
        profile = replace(get_profile(name), cameras=None)  # the G1 profile overrides it with its head camera
        cameras = robot_layout(profile).cameras
        spec = load_robot_spec(profile)
        spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[5, 5, 0.1])
        model = spec.compile()
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, 0)  # standing on the floor at z = 0
        mujoco.mj_forward(model, data)
        hits = _view(model, data, "front")

        assert [(camera.name, camera.body, camera.source) for camera in cameras] == [("front", root, "default")]
        floor = model.geom("floor").id
        assert 0.2 < hits.count(floor) / len(hits) < 1.0

    @pytest.mark.slow
    @pytest.mark.parametrize(("name", "first", "root"), [("unitree_g1", "head", "pelvis"), ("unitree_go2", "front", "base")])
    def test_floor_flat_streams_a_chase_camera_on_the_spawned_robot(self, name: str, first: str, root: str) -> None:
        robot = MuJoCoRobot(name, scene="floor_flat", cameras=[])
        robot.connect()
        try:
            model = robot._model
            chase = model.camera("chase").id

            assert [camera.name for camera in default_cameras(robot._sim)] == [first, "overview", "chase"]
            assert model.body(int(model.cam_bodyid[chase])).name == root
            assert model.cam_mode[chase] == mujoco.mjtCamLight.mjCAMLIGHT_TRACKCOM
        finally:
            robot.disconnect()

    @pytest.mark.slow
    def test_the_g1_head_camera_looks_ahead_and_down_from_the_torso(self) -> None:
        profile = get_profile("unitree_g1")
        spec = load_robot_spec(profile)
        spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[5, 5, 0.1])
        model = spec.compile()
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, 0)
        mujoco.mj_forward(model, data)
        head = model.camera("head").id
        view = -data.cam_xmat[head].reshape(3, 3)[:, 2]  # cameras look along -z

        cameras = robot_layout(profile).cameras
        assert [(camera.name, camera.body, camera.source) for camera in cameras] == [("head", "torso_link", "override")]
        assert view[0] > 0.6 and view[2] < -0.6  # forward and about 48 degrees down (D435 mount)
        assert abs(view[1]) < 1e-6
        hits = _view(model, data, "head")
        assert hits.count(model.geom("floor").id) / len(hits) > 0.5


def _tabletop_cases() -> list[tuple[str, int]]:
    """Every supported fixed-base profile with each arm count it runs (P7); floating bases have floor scenes only."""
    from physicalai_mujoco_plugin.profiles import list_profiles  # noqa: PLC0415
    from physicalai_mujoco_plugin.scene_registry import supported_arm_counts  # noqa: PLC0415

    return [
        (profile.name, arms)
        for profile in list_profiles()
        if profile.tier != "unsupported" and profile.default_scene not in {None, "floor_flat"}
        for arms in supported_arm_counts(profile)
    ]


@pytest.mark.parametrize(("name", "arms"), _tabletop_cases())
def test_default_cameras_are_the_same_in_every_tabletop_scene(name: str, arms: int) -> None:
    """P7: a dataset's camera features depend only on profile and arm count: the robot cameras and ``overview``.

    Switching scenes must never change them; floor scenes, which add ``chase``, are for floating bases only.
    """
    from physicalai_mujoco_plugin.scene_registry import list_scenes_for  # noqa: PLC0415
    from physicalai_mujoco_plugin.sim import load_sim  # noqa: PLC0415

    profile = get_profile(name)
    scenes = {scene_id: scene for scene_id, scene in list_scenes_for(profile, arms).items() if scene.anchors == "mount"}
    assert scenes, f"{name} runs {arms} arm(s) in no tabletop scene"
    names = {}
    for scene_id, scene in scenes.items():
        sim = load_sim(
            scene.scene_xml_path,
            scene,
            profile,
            unit=profile.default_unit,
            torque_mode="pd",
            rng=np.random.default_rng(0),
            reseed=lambda: None,
            arms=arms,
        )
        robot_cameras = {camera.name for binding in sim.bindings for camera in binding.layout.cameras}
        names[scene_id] = [config.name for config in default_cameras(sim)]
        assert set(names[scene_id]) == robot_cameras | {"overview"}, scene_id
    assert len({tuple(cameras) for cameras in names.values()}) == 1, names
