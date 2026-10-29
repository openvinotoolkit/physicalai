# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""The arms the bucket_sort demonstrator drives, by profile: what it needs to know of each robot.

An arm is its joint and actuator names, its tool (the point between the finger tips, the way the
fingers point and the way they close), how its gripper actuator takes the opening, its home pose,
its joint speed limits, the heights it carries at and how far it may tilt the hand. The Franka arms
carry the Franka Hand (the FR3 and the Duo get the Panda package's hand, see ``profiles/franka.py``)
and keep the fingers down:

- ``franka_fr3``: one FR3 at the scene's mount, its base at the origin facing +x.
- ``franka_panda``: one Panda there; Menagerie names its joints ``joint1`` ... ``joint7`` and its
  actuators ``actuator1`` ... ``actuator8``, and ``actuator8`` drives the finger tendon, which is
  half the opening (Menagerie's 0 to 255 command).
- ``franka_duo``: the FR3 Duo's two arms (``left_``, ``right_``), on Franka's Duo mount. Their bases
  are tilted, so the hand's yaw is no longer the base joint's turn plus joint 7's
  (:attr:`BucketSortArm.upright`). One arm works at a time while the other holds its ready pose.

The six-joint arms reach the buckets with the hand tilted outward (:attr:`BucketSortArm.max_tilt`),
the fingers no longer straight down, and carry lower:

- ``trossen_wxai``: the WidowX AI, alone or two side by side (``left_``, ``right_``, one working at
  a time as on the Duo). Its tool is Menagerie's ``ee_gripper_link`` site at the finger tips; each
  carriage moves half the opening.
- ``rebot_b601``: the reBot B601; its tool point is 0.163 m out along ``link6``, its two fingers move
  half the opening each and both take the gripper's target.

This module does not import MuJoCo: the scene registry reads the home poses from it.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import pi, radians

from physicalai_mujoco_plugin.bucket_sort import TaskLayout

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
class Heights:
    """The TCP heights and offsets an arm's attempts draw from (``DemoParams.sample``), metres."""

    pregrasp_height: tuple[float, float] = (0.06, 0.13)
    lift_height: tuple[float, float] = (0.07, 0.13)
    carry_z: tuple[float, float] = (0.20, 0.27)
    release_z: tuple[float, float] = (0.12, 0.18)
    safe: tuple[tuple[float, float], tuple[float, float]] = ((0.15, 0.25), (0.16, 0.26))
    """``(release_z, carry_z)`` of the two safer variants an attempt falls back to: at least the
    first pair, then exactly the second."""
    retreat_z: float = 0.22
    """The lowest TCP height the hand rises to after the release."""


FRANKA_HEIGHTS = Heights()


@dataclass(frozen=True)
class BucketSortArm:
    """One arm and its gripper, as the bucket_sort demonstrator drives it.

    Names are unprefixed; :meth:`name` adds the arm's prefix (``left_`` on the Duo).
    """

    label: str
    """How messages name the arm."""
    joints: tuple[str, ...] = FR3_JOINTS
    """The arm joints, base first: seven on the Franka arms, six on the others."""
    actuators: tuple[str, ...] = FR3_JOINTS
    """Their position actuators, in the same order."""
    gripper: str = "gripper"
    """The hand's actuator."""
    gripper_followers: tuple[str, ...] = ()
    """More actuators that take the gripper's target (the reBot's second finger)."""
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
    """The base stands upright facing +x with joint 1 vertical, so the wrist joint follows from the
    hand's yaw and position in closed form; otherwise the planner solves for it."""
    tool_body: str = "hand"
    """The body that carries the tool point."""
    tool_site: str | None = None
    """A site at the tool point instead (its body carries it); its frame is the tool frame."""
    tcp: tuple[float, float, float] = (0.0, 0.0, 0.1034)
    """The tool point in :attr:`tool_body`'s frame (without a site): Franka's TCP, 103.4 mm out."""
    approach: tuple[float, float, float] = (0.0, 0.0, 1.0)
    """The axis, in the tool frame, along which the fingers point."""
    closing: tuple[float, float, float] = (0.0, 1.0, 0.0)
    """The axis, in the tool frame, along which the fingers close."""
    fingers: tuple[str, ...] = FINGER_JOINTS
    """The finger joints; each moves half the opening."""
    hand_bodies: tuple[str, ...] = ("hand", "left_finger", "right_finger")
    """The hand's bodies, fingers last, for the clearance checks."""
    finger_bodies: tuple[str, ...] = ("left_finger", "right_finger")
    grip_open: float = FRANKA_GRIP_OPEN
    """The full opening, metres."""
    min_tcp_z: float = 0.013
    """Lowest grasp height of the tool point: the Franka Hand's pads reach 9 mm below it."""
    gravity_offset: bool = False
    """Command each joint as much past its planned angle as its position servo sags under the arm's
    weight there, so the arm stands where planned: for the WidowX AI, whose servos are soft enough
    to sag 2 to 3 cm at the finger tips."""
    squeeze: float | None = None
    """Close the hand this far (m) narrower than the object, not all the way: a stiff gripper with no
    force limit would otherwise crush the object out of its grasp. ``None`` closes fully (the Franka
    Hand, whose force is limited)."""
    wrist: int = 6
    """The joint that turns the hand about its approach axis."""
    shoulder_singularity: bool = True
    """Joint 2 at zero lines joints 1 and 3 up (the Franka arms): the IK pushes it off zero."""
    heights: Heights = FRANKA_HEIGHTS
    max_tilt: float = 0.0
    """How far the hand may lean outward from fingers-down to reach, radians; 0 keeps them down."""
    joint_margin: float | None = None
    """How far inside its limits the IK keeps every joint, radians; ``None`` is the planner's 7 deg."""
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
        fingers = tuple((self.name(finger), self.grip_open / 2) for finger in self.fingers)
        actuators = tuple(
            (self.name(actuator), value)
            for joint, actuator, value in zip(self.joints, self.actuators, self.home, strict=True)
            if actuator != joint
        )
        grippers = tuple((self.name(a), self.gripper_open_ctrl) for a in (self.gripper, *self.gripper_followers))
        return (*joints, *fingers, *actuators, *grippers)


FrankaArm = BucketSortArm
"""The name of :class:`BucketSortArm` while every arm was a Franka."""


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
    "left": (-26.7, -45.0, 28.0, -154.0, 65.8, 83.9, -6.7),
    "right": (26.7, -45.0, -28.0, -154.0, -65.8, 83.9, 6.7),
}
"""Each Duo arm's ready pose, degrees: the hand 0.38 m above the table, 0.40 m in front of the mount
frame and 0.30 m out to its side, fingers down, every joint at least 20 deg inside its limits. The
left arm's is the right arm's mirrored (joints 1, 3, 5 and 7 negated). Of the poses tried, from this
one the arms plan the most attempts over the table."""
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

SIX_JOINT_HEIGHTS = Heights(
    pregrasp_height=(0.05, 0.09),
    lift_height=(0.12, 0.16),
    carry_z=(0.19, 0.24),
    release_z=(0.16, 0.19),
    safe=((0.18, 0.23), (0.19, 0.24)),
    retreat_z=0.17,
)
"""The six-joint arms carry low: the held object clears the 0.12 m rims by about 1.5 cm."""
SIX_JOINT_DQ_MAX = (3.0,) * 6
"""Joint speed limits of the six-joint arms, rad/s: an estimate (no datasheet gives them per joint)."""
SIX_JOINT_DDQ_MAX = (10.0,) * 6

WXAI_ARM = BucketSortArm(
    label="WidowX AI",
    joints=tuple(f"joint_{i}" for i in range(6)),
    actuators=tuple(f"joint_{i}" for i in range(6)),
    gripper="right_carriage_joint",
    grip_per_width=0.5,
    gripper_open_ctrl=0.044,
    links=("link_4", "link_5"),
    arm_links=tuple(f"link_{i}" for i in range(1, 7)),
    home=tuple(radians(v) for v in (0.0, 42.7, 67.2, -84.6, 0.0, 0.0)),
    home_yaw=None,
    dq_max=SIX_JOINT_DQ_MAX,
    ddq_max=SIX_JOINT_DDQ_MAX,
    tool_site="ee_gripper_link",
    approach=(1.0, 0.0, 0.0),
    closing=(0.0, 1.0, 0.0),
    fingers=("left_carriage_joint", "right_carriage_joint"),
    hand_bodies=("link_6", "camera_mount_d405", "carriage_left", "carriage_right", "gripper_left", "gripper_right"),
    finger_bodies=("gripper_left", "gripper_right"),
    grip_open=0.088,
    min_tcp_z=0.006,
    wrist=5,
    shoulder_singularity=False,
    heights=SIX_JOINT_HEIGHTS,
    max_tilt=radians(50.0),
    joint_margin=radians(3.0),
    gravity_offset=True,
)
"""The WidowX AI: Menagerie's ``ee_gripper_link`` site at the finger tips, the fingers along its x,
closing along its y; each carriage moves half the opening, 0 to 0.045 m."""
REBOT_ARM = BucketSortArm(
    label="reBot B601",
    joints=tuple(f"joint{i}" for i in range(1, 7)),
    actuators=tuple(f"joint{i}" for i in range(1, 7)),
    gripper="joint_left",
    gripper_followers=("joint_right",),
    grip_per_width=0.5,
    gripper_open_ctrl=0.05,
    links=("link4", "link5"),
    arm_links=tuple(f"link{i}" for i in range(1, 7)),
    home=tuple(radians(v) for v in (0.0, -50.0, -60.7, 75.7, 0.0, 0.0)),
    home_yaw=None,
    dq_max=SIX_JOINT_DQ_MAX,
    ddq_max=SIX_JOINT_DDQ_MAX,
    tool_body="link6",
    tcp=(0.0, 0.0, 0.163),
    fingers=("joint_left", "joint_right"),
    hand_bodies=("link6", "gripper_end", "gripper_left", "gripper_right"),
    finger_bodies=("gripper_left", "gripper_right"),
    grip_open=0.10,
    min_tcp_z=0.006,
    squeeze=0.006,
    gravity_offset=True,
    wrist=5,
    shoulder_singularity=False,
    heights=SIX_JOINT_HEIGHTS,
    max_tilt=radians(50.0),
    joint_margin=radians(3.0),
)
"""The reBot B601: the profile's tool point 0.163 m out along ``link6``, the fingers along its z,
closing along its y, each moving half the opening (0 to 0.05 m). Its wrist turns only 90 deg either
way, so the joints keep 3 deg inside their limits instead of 7."""

BUCKET_SORT_ARMS: dict[str, tuple[FrankaArm, ...]] = {
    "franka_fr3": (FR3_ARM,),
    "franka_panda": (PANDA_ARM,),
    "franka_duo": DUO_ARMS,
    "trossen_wxai": (WXAI_ARM,),
    "rebot_b601": (REBOT_ARM,),
}
"""The arms of one robot of each profile the bucket_sort scene runs, in the profile's channel order."""


SIX_JOINT_LAYOUT = TaskLayout(
    bucket_x=(0.40, 0.52),
    bucket_y=(-0.36, 0.36),
    object_x=(0.22, 0.34),
    object_y=(-0.24, 0.24),
    reach=(0.22, 0.42),
    bucket_reach=(0.28, 0.60),
    face_bases=True,
    hand_half_size=(0.095, 0.065),
    finger_half_size=(0.05, 0.015),
)
"""One WidowX AI or reBot B601: the objects 0.22 to 0.42 m from the base, where the fingers reach
them nearly straight down, and the buckets 0.28 to 0.60 m (the release points), where the hand
reaches over their rims leaning outward. The buckets' open ends face the base. Full size, the
scene's table; the hands' footprints are the grippers' (camera mount included)."""
WXAI_PAIR_SEPARATION = 0.44
"""Two WidowX AI stand this far apart at the table edge, side by side."""
WXAI_PAIR_TURN = radians(30.0)
"""How far each WidowX AI of a pair turns away from the other at home, so the one waiting there
leaves room for the other reaching across to an object on its side."""
WXAI_PAIR_LAYOUT = replace(
    SIX_JOINT_LAYOUT,
    bucket_y=(-0.44, 0.44),
    object_x=(0.20, 0.36),
    object_y=(-0.20, 0.20),
    reach=(0.20, 0.45),
    bases=((0.0, WXAI_PAIR_SEPARATION / 2), (0.0, -WXAI_PAIR_SEPARATION / 2)),
    objects_reach_all=True,
)
"""Two WidowX AI: every object between them, in reach of both, and the buckets spread across both
sides, each open toward its nearer arm: whichever arm reaches the named bucket reaches the object."""


def bucket_sort_arms(profile_name: str, robots: int = 1) -> tuple[BucketSortArm, ...] | None:
    """Return the arms of *robots* robots of *profile_name* in bucket_sort, in channel order.

    Returns:
        The arms, prefixed ``left_`` and ``right_`` for two robots; ``None`` when the scene does not
        run that many of them (only the WidowX AI runs as a pair).
    """
    arms = BUCKET_SORT_ARMS.get(profile_name)
    if arms is None:
        return None
    if robots == 1:
        return arms
    if robots == 2 and profile_name == "trossen_wxai":  # noqa: PLR2004
        arm = arms[0]
        return tuple(
            replace(arm, label=f"{side} arm", prefix=f"{side}_", home=(arm.home[0] + turn, *arm.home[1:]))
            for side, turn in (("left", WXAI_PAIR_TURN), ("right", -WXAI_PAIR_TURN))
        )
    return None


def bucket_sort_home(profile_name: str, robots: int = 1) -> tuple[tuple[str, float], ...]:
    """Return the bucket_sort home pose of *robots* of *profile_name*'s arms, by full joint (and gripper actuator) name.

    Returns:
        Every arm's :attr:`FrankaArm.home_qpos`, with the prefix of :func:`bucket_sort_arms`.

    Raises:
        ValueError: if bucket_sort does not run that many of them.
    """
    arms = bucket_sort_arms(profile_name, robots)
    if arms is None:
        msg = f"bucket_sort does not run {robots} {profile_name}"
        raise ValueError(msg)
    return tuple(item for arm in arms for item in arm.home_qpos)


__all__ = [
    "BUCKET_SORT_ARMS",
    "DUO_ARMS",
    "FINGER_JOINTS",
    "FR3_ARM",
    "FRANKA_GRIP_OPEN",
    "PANDA_ARM",
    "REBOT_ARM",
    "SIX_JOINT_LAYOUT",
    "WXAI_ARM",
    "WXAI_PAIR_LAYOUT",
    "WXAI_PAIR_SEPARATION",
    "WXAI_PAIR_TURN",
    "BucketSortArm",
    "FrankaArm",
    "Heights",
    "bucket_sort_arms",
    "bucket_sort_home",
]
