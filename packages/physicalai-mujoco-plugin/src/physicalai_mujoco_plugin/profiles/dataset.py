# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Dataset-tier arm profiles: Menagerie arms that public robot-learning datasets use (SUP-1).

Their channels, home pose and cameras are derived from the model. Channel names change only where
LeRobot datasets name the joints: ALOHA's ``left_waist`` ... ``right_gripper``, and the
``shoulder_pan`` ... ``gripper`` names of the SO-100 and Koch arms. Values stay in model units
(degrees, metres) from the model's zero pose; no dataset's calibration is applied.

End effectors are Menagerie's tool sites where the model has one. The SO-ARM100, Koch, PiPER and
Panda have none: theirs is the point between the finger tips (the mean of each finger's mesh
vertices within 5 mm of its tip), in the gripper body's frame. The FR3 has no gripper: its end
effector is the flange.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

from physicalai_mujoco_plugin.profiles._types import ChannelOverride, EndEffector, RobotProfile

if TYPE_CHECKING:
    import mujoco

_HALF_SQRT2 = 0.5**0.5

_ALOHA_JOINTS = ("waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate", "gripper")
_SO_ARM_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


def _turn_base(spec: mujoco.MjSpec, *, body: str, quat: tuple[float, float, float, float]) -> None:
    """Turn an arm about its base body by *quat* (``wxyz``), so that it faces +x like the other arms.

    Menagerie's Koch faces -x and its SO-ARM100 faces -y; their base rotation limits (+-2.2 and
    +-1.92 rad) keep part of the +x side, where the tabletop scenes put the objects, out of reach.
    Only the base body's pose changes: joint names, ranges and zero pose stay those of the datasets.
    """
    import mujoco  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    base = spec.body(body)
    turned = np.zeros(4)
    mujoco.mju_mulQuat(turned, np.asarray(quat, dtype=float), np.asarray(base.quat, dtype=float))
    base.quat = turned


def _renamed(public: tuple[str, ...], actuators: tuple[str, ...]) -> tuple[ChannelOverride, ...]:
    return tuple(
        ChannelOverride(name=name, actuators=(actuator,)) for name, actuator in zip(public, actuators, strict=True)
    )


# Menagerie's ALOHA is one model with both arms at their real spacing, so it attaches at a single
# robot_mount frame; LeRobot's ALOHA datasets (lerobot/aloha_sim_*, lerobot/aloha_static_*) name its
# 14 joints left_waist ... right_gripper.
ALOHA_PROFILE = RobotProfile(
    name="aloha",
    display_name="ALOHA",
    menagerie_model="aloha",
    menagerie_entry="aloha",
    tier="dataset",
    channels=_renamed(
        tuple(f"{side}_{joint}" for side in ("left", "right") for joint in _ALOHA_JOINTS),
        tuple(f"{side}/{joint}" for side in ("left", "right") for joint in _ALOHA_JOINTS),
    ),
    default_scene="single_pick_place",
    # Menagerie's gripper sites, between the finger tips of each arm, with +x out of the fingers.
    end_effectors=(
        EndEffector(site="left/gripper", axis=(1.0, 0.0, 0.0)),
        EndEffector(site="right/gripper", axis=(1.0, 0.0, 0.0)),
    ),
    reach=0.656,
)
SO_ARM100_PROFILE = RobotProfile(
    name="so_arm100",
    display_name="SO-ARM100",
    menagerie_model="trs_so_arm100",
    menagerie_entry="so_arm100",
    tier="dataset",
    channels=_renamed(_SO_ARM_NAMES, ("Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw")),
    default_scene="single_pick_place",
    customize=partial(_turn_base, body="Base", quat=(_HALF_SQRT2, 0.0, 0.0, _HALF_SQRT2)),
    end_effectors=(EndEffector(body="Fixed_Jaw", pos=(0.0, -0.105, 0.0), axis=(0.0, -1.0, 0.0)),),
    reach=0.362,
)


KOCH_PROFILE = RobotProfile(
    name="koch",
    display_name="Koch",
    menagerie_model="low_cost_robot_arm",
    tier="dataset",
    channels=_renamed(_SO_ARM_NAMES, ("base_rotation", "pitch", "elbow", "wrist_pitch", "wrist_roll", "gripper")),
    default_scene="single_pick_place",
    customize=partial(_turn_base, body="base_link", quat=(0.0, 0.0, 0.0, 1.0)),
    end_effectors=(EndEffector(body="gripper_static_finger", pos=(-0.067, 0.007, 0.0), axis=(-1.0, 0.0, 0.0)),),
    reach=0.221,
)
PIPER_PROFILE = RobotProfile(
    name="piper",
    display_name="AgileX PiPER",
    menagerie_model="agilex_piper",
    menagerie_entry="piper",
    tier="dataset",
    default_scene="single_pick_place",
    end_effectors=(EndEffector(body="link6", pos=(0.0, 0.0, 0.138)),),
    reach=0.570,
)
FRANKA_FR3_PROFILE = RobotProfile(
    name="franka_fr3",
    display_name="Franka Research 3",
    menagerie_model="franka_fr3",
    menagerie_entry="fr3",
    tier="dataset",
    default_scene="single_pick_place",
    end_effectors=(EndEffector(site="attachment_site"),),
    reach=0.793,
)
FRANKA_PANDA_PROFILE = RobotProfile(
    name="franka_panda",
    display_name="Franka Emika Panda",
    menagerie_model="franka_emika_panda",
    menagerie_entry="panda",
    tier="dataset",
    default_scene="single_pick_place",
    end_effectors=(EndEffector(body="hand", pos=(0.0, 0.0, 0.110)),),
    reach=0.849,
)
XARM7_PROFILE = RobotProfile(
    name="xarm7",
    display_name="UFACTORY xArm7",
    menagerie_model="ufactory_xarm7",
    menagerie_entry="xarm7",
    tier="dataset",
    default_scene="single_pick_place",
    end_effectors=(EndEffector(site="link_tcp"),),
    reach=0.779,
)
KINOVA_GEN3_PROFILE = RobotProfile(
    name="kinova_gen3",
    display_name="Kinova Gen3",
    menagerie_model="kinova_gen3",
    menagerie_entry="gen3",
    tier="dataset",
    default_scene="single_pick_place",
    end_effectors=(EndEffector(site="pinch_site"),),
    reach=0.771,
)

DATASET_PROFILES: tuple[RobotProfile, ...] = (
    ALOHA_PROFILE,
    SO_ARM100_PROFILE,
    KOCH_PROFILE,
    PIPER_PROFILE,
    FRANKA_FR3_PROFILE,
    FRANKA_PANDA_PROFILE,
    XARM7_PROFILE,
    KINOVA_GEN3_PROFILE,
)
"""The dataset-tier arms besides ``ur5e``, in the order ``physicalai-mujoco profiles`` lists them."""

__all__ = [
    "ALOHA_PROFILE",
    "DATASET_PROFILES",
    "FRANKA_FR3_PROFILE",
    "FRANKA_PANDA_PROFILE",
    "KINOVA_GEN3_PROFILE",
    "KOCH_PROFILE",
    "PIPER_PROFILE",
    "SO_ARM100_PROFILE",
    "XARM7_PROFILE",
]
