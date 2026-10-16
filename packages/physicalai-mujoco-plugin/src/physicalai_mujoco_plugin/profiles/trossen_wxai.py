# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Trossen WidowX AI profile: Menagerie's ``wxai_follower`` under the names and units of Runtime's ``WidowXAI``."""

# MjSpec is supplied by the MuJoCo C extension at runtime.
# pyrefly: ignore-errors [missing-attribute]

from __future__ import annotations

from typing import TYPE_CHECKING

from physicalai_mujoco_plugin.profiles._types import CameraSpec, ChannelOverride, EndEffector, RobotProfile

if TYPE_CHECKING:
    import mujoco

TROSSEN_WXAI_CHANNELS: tuple[tuple[str, str], ...] = (
    ("shoulder_pan", "joint_0"),
    ("shoulder_lift", "joint_1"),
    ("elbow_flex", "joint_2"),
    ("wrist_flex", "joint_3"),
    ("wrist_yaw", "joint_4"),
    ("wrist_roll", "joint_5"),
    # The left carriage follows the right one through the model's joint equality.
    ("gripper", "right_carriage_joint"),
)
"""``(public name, Menagerie actuator)`` in the order of ``physicalai.robot.trossen.WIDOWXAI_JOINT_ORDER``."""

TROSSEN_WXAI_WRIST_CAMERA = CameraSpec(
    name="wrist",
    body="camera_link",
    pos=(0.0, 0.0, 0.0),
    quat=(0.5, 0.5, -0.5, -0.5),
    # The RealSense D405's vertical field of view, as Menagerie's focal length and sensor size give it.
    fovy=58.0,
)
"""The wrist D405 at Menagerie's ``wrist_cam`` pose (Trossen's camera mount).

Menagerie's camera sets a 16:9 sensor size, which MuJoCo squeezes into any other image aspect;
a plain ``fovy`` renders the 640x480 streams undistorted.
"""


def customize_trossen_wxai(spec: mujoco.MjSpec) -> None:
    """Remove Menagerie's ``wrist_cam``; :data:`TROSSEN_WXAI_WRIST_CAMERA` replaces it."""
    spec.delete(spec.camera("wrist_cam"))


TROSSEN_WXAI_PROFILE = RobotProfile(
    name="trossen_wxai",
    display_name="WidowX AI",
    menagerie_model="trossen_wxai",
    menagerie_entry="wxai_follower",
    tier="twin",
    # Runtime's WidowXAI reports rotating joints in degrees and the gripper in metres, and its
    # sensor_data["velocities"] unconverted (rad/s, m/s), so velocities stay in model units (OBS-4).
    channels=tuple(
        ChannelOverride(name=name, actuators=(actuator,), velocity_unit="model")
        for name, actuator in TROSSEN_WXAI_CHANNELS
    ),
    default_unit="degrees",
    cameras=(TROSSEN_WXAI_WRIST_CAMERA,),
    customize=customize_trossen_wxai,
    default_scene="single_pick_place",
    end_effectors=(EndEffector(site="ee_gripper_link", axis=(1.0, 0.0, 0.0)),),
    reach=0.556,
)

__all__ = ["TROSSEN_WXAI_CHANNELS", "TROSSEN_WXAI_PROFILE", "TROSSEN_WXAI_WRIST_CAMERA", "customize_trossen_wxai"]
