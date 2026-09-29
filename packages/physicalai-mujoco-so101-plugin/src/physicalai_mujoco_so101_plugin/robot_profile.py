# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Robot profiles and scene composition: scenes hold no robot, arms are attached at mount frames.

A scene XML marks each arm's base pose with a ``<frame name="{prefix}robot_mount">``. Loading the
scene attaches the profile's MJCF at every mount frame with that prefix, so a ``left_robot_mount``
frame yields ``left_shoulder_pan``, ``left_gripper``, ``left_wrist``, and so on. XML files without
mount frames (for example a custom ``--model`` that defines its own arm) compile unchanged.
"""

# MjSpec and the mju_* helpers are supplied by the MuJoCo C extension at runtime.
# pyrefly: ignore-errors [missing-attribute]

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from physicalai_mujoco_so101_plugin._urdf import get_urdf_path
from physicalai_mujoco_so101_plugin.constants import SO101_JOINT_ORDER

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    import mujoco

ROBOT_MOUNT_FRAME = "robot_mount"
"""Suffix of the scene frames that mark an arm's base; the text before it is the arm's name prefix."""


@dataclass(frozen=True)
class RobotProfile:
    """How to load one robot model and adapt it to the plugin's public contract."""

    name: str
    mjcf_relpath: str
    """Robot MJCF, relative to the bundled ``urdf`` directory."""
    joint_order: tuple[str, ...]
    """Public joint (and position actuator) names, unprefixed, in observation/action order."""
    joint_ranges: tuple[tuple[str, float, float], ...]
    """Joint ranges in radians. Normalized units span these ranges, so they are pinned here."""
    actuator_forcerange: tuple[float, float]
    wrist_camera: str
    """Name of the camera on the gripper, unprefixed."""
    customize: Callable[[mujoco.MjSpec], None] | None = None
    """Further edits applied to the freshly loaded robot spec, before it is attached."""

    @property
    def mjcf_path(self) -> Path:
        """Absolute path to the robot's MJCF."""
        return get_urdf_path() / self.mjcf_relpath

    def load_spec(self) -> mujoco.MjSpec:
        """Load the robot MJCF and apply the profile's ranges, force limits, and customization.

        Returns:
            A new robot spec, ready to attach.
        """
        import mujoco  # noqa: PLC0415

        spec = mujoco.MjSpec.from_file(str(self.mjcf_path))
        for joint_name, lower, upper in self.joint_ranges:
            spec.joint(joint_name).range = [lower, upper]
        for joint_name in self.joint_order:
            spec.actuator(joint_name).forcerange = list(self.actuator_forcerange)
        if self.customize is not None:
            self.customize(spec)
        return spec


def _set_quat(element: object, quat: np.ndarray | list[float]) -> None:
    """Give a spec element an explicit quaternion, overriding any authored Euler/axis orientation."""
    import mujoco  # noqa: PLC0415

    element.quat = list(quat)
    element.alt.type = mujoco.mjtOrientation.mjORIENTATION_QUAT


def _euler_xyz_quat(euler: tuple[float, float, float]) -> np.ndarray:
    import mujoco  # noqa: PLC0415

    quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_euler2Quat(quat, np.asarray(euler, dtype=np.float64), "xyz")
    return quat


def _customize_so101(spec: mujoco.MjSpec) -> None:
    """Match Menagerie's SO-101 to the plugin's earlier SO-101 model, which trained policies expect.

    - The wrist camera keeps its name ``wrist``, its pose on the gripper's -y side, and its 75 degree
      field of view; its collision boxes stay on that side too.
    - Menagerie's gripper collision meshes, the extra box on the fixed jaw, and the camera mount's
      visual geoms (mount, PCB, lens) are removed, so grasp contacts and rendered images stay the same.
    - The ``gripperframe`` site keeps its earlier orientation.
    """
    camera = spec.camera("wrist_cam")
    camera.name = "wrist"
    camera.pos = [0.0, -0.055, -0.045]
    _set_quat(camera, _euler_xyz_quat((0.57, 0.0, np.pi)))
    camera.fovy = 75.0

    camera_mount = spec.body("camera_mount")
    for geom in list(camera_mount.geoms):
        if geom.name not in {"camera_box1", "camera_box2"}:
            spec.delete(geom)
    box1, box2 = spec.geom("camera_box1"), spec.geom("camera_box2")
    box1.pos = [0.0025, -0.03, -0.03]
    box2.pos = [0.001, -0.06, -0.04]
    _set_quat(box2, _euler_xyz_quat((0.55, 0.0, 0.0)))

    for geom in list(spec.body("gripper").geoms):
        if geom.classname.name == "collision":
            spec.delete(geom)
    for geom in list(spec.geoms):
        if geom.classname.name == "collision_gripper_mesh":
            spec.delete(geom)
    for mesh_name in (
        "moving_jaw_so101_gripper_v1",
        "wrist_roll_follower_so101_camera_mount",
        "wrist_roll_follower_so101_gripper_part0_v1",
        "moving_jaw_so101_gripper_part0_v1",
        "moving_jaw_so101_gripper_part1_v1",
    ):
        spec.delete(spec.mesh(mesh_name))

    _set_quat(spec.site("gripperframe"), [0.0, 0.0, 1.0, 0.0])


SO101_PROFILE = RobotProfile(
    name="so101",
    mjcf_relpath="robots/so101/so101.xml",
    joint_order=SO101_JOINT_ORDER,
    # The calibrated ranges of urdf/so101/so101_new_calib.urdf. Menagerie rounds them and caps
    # wrist_roll at +2.74385 rad; the plugin's actuators and normalized units reach 2.84121 rad.
    joint_ranges=(
        ("shoulder_pan", -1.9198621771937616, 1.9198621771937634),
        ("shoulder_lift", -1.7453292519943224, 1.7453292519943366),
        ("elbow_flex", -1.69, 1.69),
        ("wrist_flex", -1.6580628494556928, 1.6580627293335335),
        ("wrist_roll", -2.7438472969992493, 2.841206309382605),
        ("gripper", -0.17453297762778586, 1.7453291995659765),
    ),
    # The plugin's servos have always used 3.35 N m; Menagerie's class default is 2.94.
    actuator_forcerange=(-3.35, 3.35),
    wrist_camera="wrist",
    customize=_customize_so101,
)


def robot_mount_prefixes(spec: mujoco.MjSpec) -> tuple[str, ...]:
    """Return the arm prefixes of the scene's mount frames, in document order."""
    return tuple(
        frame.name.removesuffix(ROBOT_MOUNT_FRAME) for frame in spec.frames if frame.name.endswith(ROBOT_MOUNT_FRAME)
    )


def attach_robot(scene: mujoco.MjSpec, profile: RobotProfile, prefix: str) -> None:
    """Attach a fresh copy of the profile's robot at the scene frame ``{prefix}robot_mount``."""
    robot = profile.load_spec()
    # Attaching keeps the scene's physics options; copying them first avoids per-field conflict warnings.
    for field in dir(scene.option):
        if not field.startswith("_"):
            setattr(robot.option, field, getattr(scene.option, field))
    scene.attach(robot, frame=scene.frame(f"{prefix}{ROBOT_MOUNT_FRAME}"), prefix=prefix)


def compose_scene_spec(xml_path: str | Path, profile: RobotProfile = SO101_PROFILE) -> mujoco.MjSpec:
    """Load a scene XML and attach the robot at each of its mount frames.

    Returns:
        The scene spec with its arms attached.
    """
    import mujoco  # noqa: PLC0415

    scene = mujoco.MjSpec.from_file(str(xml_path))
    for prefix in robot_mount_prefixes(scene):
        attach_robot(scene, profile, prefix)
    return scene


def load_scene_model(xml_path: str | Path, profile: RobotProfile = SO101_PROFILE) -> mujoco.MjModel:
    """Compile a scene XML with its arms attached.

    Returns:
        The compiled model.
    """
    return compose_scene_spec(xml_path, profile).compile()
