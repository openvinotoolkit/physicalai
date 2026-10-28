# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""The arms the bucket_sort demonstrator drives, by profile: what it needs to know of each robot.

Every one carries the Franka Hand (the FR3 and the Duo get the Panda package's hand, see
``profiles/franka.py``), so the hand's frame, its tool point and its fingers are the same for all;
an arm differs in its joint and actuator names, how its gripper actuator takes the opening, its
home pose, its joint speed limits and where its base stands:

- ``franka_fr3``: one FR3 at the scene's mount, its base at the origin facing +x.
- ``franka_panda``: one Panda there; Menagerie names its joints ``joint1`` ... ``joint7`` and its
  actuators ``actuator1`` ... ``actuator8``, and ``actuator8`` drives the finger tendon, which is
  half the opening (Menagerie's 0 to 255 command).
- ``franka_duo``: the FR3 Duo's two arms (``left_``, ``right_``), on Franka's Duo mount. Their bases
  are tilted, so the hand's yaw is no longer the base joint's turn plus joint 7's
  (:attr:`FrankaArm.upright`). One arm works at a time while the other holds its ready pose.

This module does not import MuJoCo: the scene registry reads the home poses from it.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi, radians

FRANKA_GRIP_OPEN = 0.08
"""The Franka Hand's full opening, in metres."""
FINGER_JOINTS = ("finger_joint1", "finger_joint2")
"""The hand's finger joints, 0.04 m each when open."""
FR3_JOINTS = tuple(f"fr3_joint{i}" for i in range(1, 8))
READY_Q = tuple(radians(v) for v in (0.0, -45.0, 0.0, -135.0, 0.0, 90.0, 45.0))
"""Franka's ready pose: the hand 0.49 m above the table in front of the base, fingers down."""
FR3_DQ_MAX = (2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26)
"""FR3 joint velocity limits, rad/s (Franka's FR3 datasheet)."""
FR3_DDQ_MAX = (10.0,) * 7
"""FR3 joint acceleration limits, rad/s^2."""
PANDA_DQ_MAX = (2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61)
"""Panda joint velocity limits, rad/s (Franka's Panda datasheet)."""
PANDA_DDQ_MAX = (15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0)
"""Panda joint acceleration limits, rad/s^2."""


@dataclass(frozen=True)
class FrankaArm:
    """One Franka arm with the Franka Hand, as the bucket_sort demonstrator drives it.

    Names are unprefixed; :meth:`name` adds the arm's prefix (``left_`` on the Duo).
    """

    label: str
    """How messages name the arm."""
    joints: tuple[str, ...] = FR3_JOINTS
    """The seven arm joints, base first."""
    actuators: tuple[str, ...] = FR3_JOINTS
    """Their position actuators, in the same order."""
    gripper: str = "gripper"
    """The hand's actuator."""
    grip_per_width: float = 1.0
    """The gripper actuator's target, in its model unit, per metre of opening: 1 for the FR3's width
    tendon, 0.5 for the Panda's finger tendon (each finger moves half the opening)."""
    gripper_open_ctrl: float = FRANKA_GRIP_OPEN
    """The gripper actuator's ``ctrl`` with the hand open, for the scene's home pose."""
    links: tuple[str, ...] = ("fr3_link5", "fr3_link6", "fr3_link7")
    """The wrist links that pass near the buckets and the objects, besides the hand and fingers."""
    arm_links: tuple[str, ...] = tuple(f"fr3_link{i}" for i in range(8))
    """Every link of the arm, for the clearance from another arm."""
    home: tuple[float, ...] = READY_Q
    """The scene's home pose of the arm, radians: the plans start and end there."""
    home_yaw: float | None = -pi / 2
    """World yaw of the closing axis at :attr:`home`; ``None`` measures it on the model."""
    dq_max: tuple[float, ...] = FR3_DQ_MAX
    ddq_max: tuple[float, ...] = FR3_DDQ_MAX
    upright: bool = True
    """The base stands at the origin facing +x with joint 1 vertical, so joint 7 follows from the
    hand's yaw and position in closed form; otherwise the planner solves for it."""
    prefix: str = ""

    def name(self, element: str) -> str:
        """Return the model name of the arm's *element* (joint, body, actuator).

        Returns:
            The name with the arm's prefix.
        """
        return f"{self.prefix}{element}"

    @property
    def home_qpos(self) -> tuple[tuple[str, float], ...]:
        """The arm's home for the scene's reset: joints, fingers open, and the gripper's ``ctrl``.

        The reset sets the actuator named like each joint; the Panda's ``actuator1`` ... ``actuator7``
        get their joint's value by their own names.
        """
        joints = tuple((self.name(joint), value) for joint, value in zip(self.joints, self.home, strict=True))
        fingers = tuple((self.name(finger), FRANKA_GRIP_OPEN / 2) for finger in FINGER_JOINTS)
        actuators = tuple(
            (self.name(actuator), value)
            for joint, actuator, value in zip(self.joints, self.actuators, self.home, strict=True)
            if actuator != joint
        )
        return (*joints, *fingers, *actuators, (self.name(self.gripper), self.gripper_open_ctrl))


FR3_ARM = FrankaArm(label="FR3")
PANDA_ARM = FrankaArm(
    label="Panda",
    joints=tuple(f"joint{i}" for i in range(1, 8)),
    actuators=tuple(f"actuator{i}" for i in range(1, 8)),
    gripper="actuator8",
    grip_per_width=0.5,
    gripper_open_ctrl=255.0,
    links=("link5", "link6", "link7"),
    arm_links=tuple(f"link{i}" for i in range(8)),
    home_yaw=None,
    dq_max=PANDA_DQ_MAX,
    ddq_max=PANDA_DDQ_MAX,
)
DUO_READY_DEG = {
    "left": (-27.9, -54.7, 42.2, -162.4, 80.5, 83.3, -7.8),
    "right": (27.9, -54.7, -42.2, -162.4, -80.5, 83.3, 7.8),
}
"""Each Duo arm's ready pose, degrees: the hand 0.35 m above the table, 0.40 m in front of the mount
and 0.30 m out to its side, fingers down. The left arm's is the right arm's mirrored (joints 1, 3, 5
and 7 negated). Of the poses tried, from this one the arms plan the most attempts over the table."""
DUO_ARMS = tuple(
    FrankaArm(
        label=f"{side} arm",
        home=tuple(radians(v) for v in DUO_READY_DEG[side]),
        home_yaw=None,
        upright=False,
        prefix=f"{side}_",
    )
    for side in ("left", "right")
)

BUCKET_SORT_ARMS: dict[str, tuple[FrankaArm, ...]] = {
    "franka_fr3": (FR3_ARM,),
    "franka_panda": (PANDA_ARM,),
    "franka_duo": DUO_ARMS,
}
"""The arms of each profile the bucket_sort scene runs, in the profile's channel order."""


def bucket_sort_home(profile_name: str) -> tuple[tuple[str, float], ...]:
    """Return the bucket_sort home pose of *profile_name*'s arms, by full joint (and gripper actuator) name.

    Returns:
        Every arm's :attr:`FrankaArm.home_qpos`.
    """
    return tuple(item for arm in BUCKET_SORT_ARMS[profile_name] for item in arm.home_qpos)


__all__ = [
    "BUCKET_SORT_ARMS",
    "DUO_ARMS",
    "FINGER_JOINTS",
    "FR3_ARM",
    "FRANKA_GRIP_OPEN",
    "PANDA_ARM",
    "FrankaArm",
    "bucket_sort_home",
]
