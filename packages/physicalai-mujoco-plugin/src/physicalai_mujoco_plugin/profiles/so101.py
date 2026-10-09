# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""SO-101 profile: Menagerie's SO-101, adjusted to the plugin's earlier model that trained policies expect."""

# MjSpec and the mju_* helpers are supplied by the MuJoCo C extension at runtime.
# pyrefly: ignore-errors [missing-attribute]

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from physicalai_mujoco_plugin.profiles._types import ChannelOverride, RobotProfile

if TYPE_CHECKING:
    import mujoco

SO101_JOINT_RANGES: tuple[tuple[str, float, float], ...] = (
    ("shoulder_pan", -1.9198621771937616, 1.9198621771937634),
    ("shoulder_lift", -1.7453292519943224, 1.7453292519943366),
    ("elbow_flex", -1.69, 1.69),
    ("wrist_flex", -1.6580628494556928, 1.6580627293335335),
    ("wrist_roll", -2.7438472969992493, 2.841206309382605),
    ("gripper", -0.17453297762778586, 1.7453291995659765),
)
"""The calibrated ranges of ``urdf/so101/so101_new_calib.urdf``, unrounded, in radians.

Normalized units span these ranges, so they are pinned here rather than taken from whichever
Menagerie version is installed.
"""


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


def customize_so101(spec: mujoco.MjSpec) -> None:
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
    display_name="SO-101",
    menagerie_model="robotstudio_so101",
    menagerie_entry="so101",
    tier="twin",
    channels=tuple(
        ChannelOverride(name=name, actuators=(name,), range=(lower, upper), gripper=name == "gripper")
        for name, lower, upper in SO101_JOINT_RANGES
    ),
    default_unit="normalized",
    # The wrist camera is still placed by ``customize``; it becomes a CameraSpec with the camera PR.
    # The plugin's servos have always used 3.35 N m; Menagerie's class default is 2.94.
    actuator_forcerange=(-3.35, 3.35),
    customize=customize_so101,
    default_scene="single_pick_place",
)

__all__ = ["SO101_JOINT_RANGES", "SO101_PROFILE", "customize_so101"]
