# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Dataset-tier arm profiles: Menagerie arms that public robot-learning datasets use (SUP-1).

Their channels, home pose and cameras are derived from the model. Channel names change only where
LeRobot datasets name the joints: ALOHA's ``left_waist`` ... ``right_gripper``, and the
``shoulder_pan`` ... ``gripper`` names of the SO-100 and Koch arms. Values stay in model units
(degrees, metres) from the model's zero pose; no dataset's calibration is applied.

End effectors are Menagerie's tool sites where the model has one. The SO-ARM100, Koch, PiPER and
Panda have none: theirs is the point between the finger tips (the mean of each finger's mesh
vertices within 5 mm of its tip), in the gripper body's frame.
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
# Self-collision pairs Franka disables for its arms (franka_description 2.9.0,
# robots/fr3_duo/fr3_duo.srdf.xacro, missing_self_arm_collisions): their simplified collision hulls
# overlap in valid folded poses, e.g. the Franka GELLO Duo's rest/calibration pose.
_FRANKA_DISABLED_SELF_COLLISIONS = (
    ("link0", "link5"),
    ("link0", "link6"),
    ("link0", "link7"),
    ("link1", "link5"),
    ("link1", "link6"),
    ("link1", "link7"),
    ("link2", "link5"),
    ("link2", "link7"),
    *(
        (link, part)
        for link in ("link0", "link1", "link2", "link5")
        for part in ("hand", "left_finger", "right_finger")
    ),
)


def exclude_franka_self_collisions(spec: mujoco.MjSpec, *, link_prefix: str) -> None:
    """Exclude Franka's disabled self-collision pairs; hand pairs only where the model has a hand."""
    names = {body.name for body in spec.bodies}
    for first, second in _FRANKA_DISABLED_SELF_COLLISIONS:
        a = f"{link_prefix}{first}" if first.startswith("link") else first
        b = f"{link_prefix}{second}" if second.startswith("link") else second
        if a in names and b in names:
            exclude = spec.add_exclude()
            exclude.bodyname1, exclude.bodyname2 = a, b


FRANKA_HAND_GRASP_FORCE_N = 70.0
"""The Franka Hand's continuous grasping force (140 N peak): what each finger presses with."""
FRANKA_HAND_SPEED_M_S = 0.2
"""The Franka Hand's top opening and closing speed, as the rate of change of its width."""
FRANKA_HAND_STIFFNESS_N_M = 7000.0
"""Finger force per metre of width short of the command: the full grasp force from 10 mm short.

A closed command (width 0) on any object 10 mm or wider squeezes it with
:data:`FRANKA_HAND_GRASP_FORCE_N`; an open or intermediate width is held within a fraction of a
millimetre."""
_FINGER_DAMPING = FRANKA_HAND_GRASP_FORCE_N / (FRANKA_HAND_SPEED_M_S / 2)
"""Finger slide damping (N s/m) that caps each finger at half the hand's speed under the full force."""
_FINGER_SOLREF = (0.006, 1.0)
"""Contact time constant of the finger pads: with MuJoCo's default 20 ms a 70 N squeeze sinks
objects 3-6 mm into each finger; with 6 ms about 0.5 mm. 4 ms, twice the scenes' timestep, makes
the fingers chatter on a held 9 V battery. ``solmix`` makes the pads' value win."""
_FINGER_SOLMIX = 100.0


def tune_franka_hand_fingers(spec: mujoco.MjSpec, *, prefix: str = "") -> None:
    """Damp Menagerie's Franka Hand fingers to its top speed and stiffen their contacts.

    The gripper actuator stays the caller's to set: :data:`FRANKA_HAND_STIFFNESS_N_M` per finger up
    to :data:`FRANKA_HAND_GRASP_FORCE_N`. Its force is capped, so the speed limit comes from the
    finger joints' own damping, which acts outside that cap.
    """
    for side, joint in (("left", "finger_joint1"), ("right", "finger_joint2")):
        spec.joint(f"{prefix}{joint}").damping = [_FINGER_DAMPING, 0.0, 0.0]
        for geom in spec.body(f"{prefix}{side}_finger").geoms:
            if geom.contype or geom.conaffinity:
                geom.solref = list(_FINGER_SOLREF)
                geom.solmix = _FINGER_SOLMIX


def customize_franka_panda(spec: mujoco.MjSpec) -> None:
    """Menagerie's Panda with the real Franka Hand's grasp force and speed.

    Its ``actuator8`` keeps Menagerie's command, 0 to 255 for each finger at 0 to 0.04 m; it pulls
    the ``split`` tendon (half the finger sum), so its stiffness and force range are four and two
    times the per-finger values.
    """
    exclude_franka_self_collisions(spec, link_prefix="")
    tune_franka_hand_fingers(spec)
    actuator = spec.actuator("actuator8")
    stiffness = 4.0 * FRANKA_HAND_STIFFNESS_N_M
    gain, bias = list(actuator.gainprm), list(actuator.biasprm)
    gain[0], bias[:3] = stiffness * 0.04 / 255.0, [0.0, -stiffness, 0.0]
    actuator.gainprm, actuator.biasprm = gain, bias
    actuator.forcerange = [-2.0 * FRANKA_HAND_GRASP_FORCE_N, 2.0 * FRANKA_HAND_GRASP_FORCE_N]


FRANKA_PANDA_PROFILE = RobotProfile(
    name="franka_panda",
    display_name="Franka Emika Panda",
    menagerie_model="franka_emika_panda",
    menagerie_entry="panda",
    tier="dataset",
    default_scene="single_pick_place",
    customize=customize_franka_panda,
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
    FRANKA_PANDA_PROFILE,
    XARM7_PROFILE,
    KINOVA_GEN3_PROFILE,
)
"""The dataset-tier arms besides ``ur5e``, in the order ``physicalai-mujoco profiles`` lists them."""

__all__ = [
    "ALOHA_PROFILE",
    "DATASET_PROFILES",
    "FRANKA_HAND_GRASP_FORCE_N",
    "FRANKA_HAND_SPEED_M_S",
    "FRANKA_HAND_STIFFNESS_N_M",
    "FRANKA_PANDA_PROFILE",
    "KINOVA_GEN3_PROFILE",
    "KOCH_PROFILE",
    "PIPER_PROFILE",
    "SO_ARM100_PROFILE",
    "XARM7_PROFILE",
    "customize_franka_panda",
    "exclude_franka_self_collisions",
    "tune_franka_hand_fingers",
]
