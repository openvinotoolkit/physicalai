# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Scene definitions and reset behavior for the MuJoCo simulation."""

# MuJoCo model/data attributes are supplied by the C extension at runtime.
# pyrefly: ignore-errors [missing-attribute]

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal

import numpy as np
from loguru import logger

from physicalai_mujoco_plugin._urdf import get_urdf_path
from physicalai_mujoco_plugin.compose import (
    MountPose,
    OverviewRig,
    OverviewStyle,
    SceneLayout,
    aim_overview_rig,
    load_scene_model,
    mount_positions,
    robot_layout,
)
from physicalai_mujoco_plugin.conveyor import park_items, pool_item_names
from physicalai_mujoco_plugin.profiles import SO101_PROFILE
from physicalai_mujoco_plugin.spawn import (
    ARM_CLEARANCE,
    ARM_FOOTPRINT_HEIGHT,
    SPAWN_ATTEMPTS,
    arm_footprint,
    object_radius,
    place_freejoint,
    read_body_xy,
    sample_clear_position,
    sample_scene_positions,
)

if TYPE_CHECKING:
    from pathlib import Path

    import mujoco

    from physicalai_mujoco_plugin.profiles import RobotProfile

ResetFn = Callable[[object, object, np.random.Generator], None]

BIMANUAL_PREFIXES = ("left_", "right_")
"""Robot prefixes of two arms, left then right as seen facing the workspace from behind the arms."""

_MIN_FRONT_DIRECTION = 1e-3
"""Shortest offset, in metres, from the robots' mounts to the aim point that tells which way they face."""

TwoArmStyle = Literal["side_by_side", "across"]
"""How a scene with one mount derives two: next to each other, or facing each other across the workspace."""


def arm_prefixes(arms: int) -> tuple[str, ...]:
    """Return the robot prefixes of *arms* arms: ``("",)`` for one, :data:`BIMANUAL_PREFIXES` for two.

    Raises:
        ValueError: For any other count.
    """
    if arms == 1:
        return ("",)
    if arms == len(BIMANUAL_PREFIXES):
        return BIMANUAL_PREFIXES
    msg = f"A simulation runs one or two arms, not {arms}"
    raise ValueError(msg)


def _yaw_quat(degrees: float) -> tuple[float, float, float, float]:
    half = np.radians(degrees) / 2
    return (float(np.cos(half)), 0.0, 0.0, float(np.sin(half)))


@dataclass(frozen=True)
class FrontView:
    """Where a scene's ``front`` overview camera stands: across the work area from the robots, facing them.

    The camera looks at the aim point from the side away from the robots (the direction from the
    centre of their mounts to the aim point), ``elevation_deg`` above the table. Distance and aim
    point grow with the profile's reach like the scene's layout.
    """

    distance: float
    """Metres from the aim point to the camera, for the SO-101."""
    aim: tuple[float, float, float] | None = None
    """The point the camera looks at, for the SO-101; ``None`` takes the spawn arc's centre on the floor
    (between the arms with two)."""
    elevation_deg: float = 45.0
    """How far the line of sight points down from level."""


@dataclass(frozen=True)
class FrontOverride:
    """One profile's changes to a scene's :class:`FrontView`; ``None`` keeps the derived value.

    For arms whose home pose the derived pose does not frame, or hides the work area from.
    """

    distance: float | None = None
    """Metres from the aim point to the camera, after scaling."""
    direction: tuple[float, float] | None = None
    """Horizontal direction from the robot toward the camera, for robots whose mounts do not show
    which way they face (ALOHA's two arms face each other across its single mount)."""
    elevation_deg: float | None = None
    """How far the line of sight points down from level."""


@dataclass(frozen=True)
class SceneConfig:
    """Metadata and reset configuration for a simulation scene."""

    scene_id: str
    display_name: str
    description: str
    scene_xml_relpath: str
    free_joints: tuple[str, ...] = ()
    target_bodies: tuple[str, ...] = ()
    spawn_center: tuple[float, float] = (0.22, 0.0)
    spawn_min_r: float = 0.05
    spawn_max_r: float = 0.14
    spawn_angle_half_deg: float = 50.0
    block_min_sep: float = 0.09
    target_min_sep: float = 0.11
    robots: tuple[str, ...] | Literal["*"] = ("so101",)
    """Profiles the scene supports; ``"*"`` accepts any profile whose base type fits ``anchors``."""
    anchors: Literal["mount", "spawn"] = "mount"
    """Anchor frames of the XML: ``robot_mount`` (fixed bases) or ``robot_spawn`` (floating bases)."""
    written_arms: int = 1
    """Number of robots the scene file attaches as written: one per anchor frame."""
    arm_counts: tuple[int, ...] = (1, 2)
    """Arm counts the scene supports. The file's anchor frames lay out ``written_arms``; another count
    takes its ``mounts``, else two arms are derived from the file's single mount (``two_arm_style``)."""
    two_arm_style: TwoArmStyle = "side_by_side"
    """How two arms stand when derived from the single mount, which must be at the origin facing +x.

    ``side_by_side``: both at the mount's table edge, facing +x like the single arm, ``left_`` at +y.
    ``across``: facing each other across the two-arm spawn centre along y, ``left_`` at +y facing -y.
    """
    two_arm_separation: float = 0.20
    """Distance between two derived mounts in metres, for the SO-101; it scales with reach like the layout."""
    two_arm_separations: tuple[tuple[str, float], ...] = ()
    """Per profile: the distance between two derived mounts after scaling, for arms whose home pose
    reaches sideways into the other arm."""
    two_arm_spawn_center: tuple[float, float] | None = None
    """The spawn arc's centre with two arms, before scaling; ``None`` keeps the one-arm centre."""
    mounts: tuple[tuple[int, tuple[MountPose, ...]], ...] = ()
    """Pinned mount poses per arm count, for counts other than ``written_arms``; they replace the
    derived two-arm layout. Prefixes follow :func:`arm_prefixes`."""
    home_qpos: tuple[tuple[str, float], ...] = ()
    """Home joint positions of each arm in radians, without the arm's prefix; unlisted joints use the model default."""
    pinned_home_qpos: tuple[tuple[int, tuple[tuple[str, float], ...]], ...] = ()
    """Per arm count: home joint positions by full (prefixed) name, replacing ``home_qpos`` for that count."""
    auto_reset_on_fall: bool = False
    """Reset when a floating base has fallen (DRV-9); off so a policy can see its fall."""
    layout: Literal["fixed", "reach"] = "fixed"
    """``reach``: the layout is sized for the SO-101 and grows with each profile's reach (SCN-6).

    The spawn arc, the scene's top-level bodies (target, overview camera rig) and so the overview
    camera's distance scale about the robot mount at the origin by ``reach / SO-101 reach``.
    Separations and object sizes stay as they are. The SO-101 gets the scene as written.
    """
    profile_spawn_centers: tuple[tuple[str, tuple[float, float]], ...] = ()
    """Per profile: a spawn centre that replaces ``spawn_center`` before scaling, for robots whose
    arms are not at the mount (ALOHA's two arms face each other across it)."""
    profile_overview_rigs: tuple[tuple[str, OverviewRig], ...] = ()
    """Per profile: the overview rig's pose after scaling, for arms whose home pose the scaled
    camera does not frame (an elbow folded behind the base, ALOHA's second arm)."""
    front_view: FrontView | None = None
    """The ``front`` overview style's camera placement; ``None``: the scene offers only its own (``shoulder``)."""
    profile_front_overrides: tuple[tuple[str, FrontOverride], ...] = ()
    """Per profile: changes to the derived ``front`` camera pose (:class:`FrontOverride`)."""

    @property
    def scene_xml_path(self) -> Path:
        """Absolute path to this scene's XML model."""
        return get_urdf_path() / self.scene_xml_relpath

    def supports(self, profile: RobotProfile) -> bool:
        """Return whether *profile* may run in this scene.

        A scene that accepts any profile still needs the right base type (SCN-5), which comes
        from the profile's model: the first check of a profile compiles its robot model, which
        downloads it on a cold Menagerie cache. Later checks hit the layout cache.

        Returns:
            ``True`` if the scene lists the profile, or accepts any with its base type.
        """
        if self.robots != "*":
            return profile.name in self.robots
        floating = robot_layout(profile).base is not None
        return floating == (self.anchors == "spawn")

    def layout_scale(self, profile: RobotProfile) -> float:
        """Return how much this scene's layout grows for *profile*: ``reach / SO-101 reach`` (SCN-6).

        Returns:
            ``1.0`` for a fixed layout and for the SO-101 (exactly, so nothing is recomputed),
            else the ratio of the reaches.

        Raises:
            ValueError: If the layout scales with reach and the profile has none.
        """
        if self.layout == "fixed" or profile.reach == SO101_PROFILE.reach:
            return 1.0
        if profile.reach is None or SO101_PROFILE.reach is None:
            msg = (
                f"Scene {self.scene_id!r} scales with the robot's reach, but profile {profile.name!r} has none; "
                "set RobotProfile.reach (tests/test_reach.py measures it)"
            )
            raise ValueError(msg)
        return profile.reach / SO101_PROFILE.reach

    @property
    def overview_styles(self) -> tuple[OverviewStyle, ...]:
        """The ``overview`` camera styles this scene offers: ``shoulder`` (as written), and ``front`` if it has one."""
        return ("shoulder", "front") if self.front_view is not None else ("shoulder",)

    def layout_for(
        self, profile: RobotProfile, arms: int | None = None, overview: OverviewStyle = "shoulder"
    ) -> SceneLayout | None:
        """Return how to lay this scene out for *arms* arms of *profile*, for :func:`~.compose.compose_scene`.

        Args:
            profile: The robot profile.
            arms: Number of arms; ``None`` keeps the scene file's mount frames (a custom model).
            overview: Where the ``overview`` camera stands (:data:`~.compose.OverviewStyle`).

        Returns:
            ``None`` when the scene compiles as written (a fixed layout or the SO-101, at the file's
            arm count, with the ``shoulder`` overview). A ``front`` overview the scene does not
            offer raises ``ValueError`` (:meth:`front_overview_rig`).
        """
        scale = self.layout_scale(profile)
        mounts = None if arms is None or arms == self.written_arms else self.mount_poses(profile, arms)
        if overview == "front":
            rig = self.front_overview_rig(profile, arms, mounts)
        else:
            # One rig per profile frames either arm count (tests/test_compose.py checks both).
            rig = dict(self.profile_overview_rigs).get(profile.name)
        if scale == 1.0 and rig is None and mounts is None:  # noqa: RUF069 - layout_scale returns exactly 1.0 for the SO-101
            return None
        return SceneLayout(scale=scale, overview_rig=rig, mounts=mounts)

    def front_overview_rig(
        self,
        profile: RobotProfile,
        arms: int | None = None,
        mounts: tuple[tuple[str, MountPose], ...] | None = None,
    ) -> OverviewRig:
        """Return the overview rig pose of the ``front`` style for *arms* arms of *profile* (:class:`FrontView`).

        Args:
            profile: The robot profile.
            arms: Number of arms; ``None`` takes the scene file's.
            mounts: The mounts of :meth:`layout_for` for that count; ``None`` uses the file's frames.

        Returns:
            The rig's world pose, after scaling.

        Raises:
            ValueError: If the scene has no ``front`` overview, or nothing tells which way the robots face.
        """
        view = self.front_view
        if view is None:
            msg = f"Scene {self.scene_id!r} has no front overview camera; it keeps its own cameras"
            raise ValueError(msg)
        scale = self.layout_scale(profile)
        if view.aim is None:
            x, y = self.for_profile(profile, arms or self.written_arms).spawn_center
            aim = np.array([x, y, 0.0])
        else:
            aim = np.asarray(view.aim) * scale
        override = dict(self.profile_front_overrides).get(profile.name, FrontOverride())
        direction = override.direction
        if direction is None:
            if mounts is not None:
                stands = np.array([pose.pos for _, pose in mounts])
            else:
                stands = np.asarray(mount_positions(self.scene_xml_path)) * scale
            direction = aim[:2] - stands[:, :2].mean(axis=0)
        horizontal = np.asarray(direction, dtype=np.float64)
        length = float(np.linalg.norm(horizontal))
        if length < _MIN_FRONT_DIRECTION:
            msg = (
                f"Scene {self.scene_id!r} cannot tell which way the {profile.display_name} faces for a front "
                "overview: its mounts surround the aim point; set a profile_front_overrides direction"
            )
            raise ValueError(msg)
        distance = override.distance if override.distance is not None else view.distance * scale
        elevation = np.radians(override.elevation_deg if override.elevation_deg is not None else view.elevation_deg)
        offset = np.array([*(horizontal / length * np.cos(elevation)), np.sin(elevation)]) * distance
        eye = aim + offset
        return aim_overview_rig(
            self.scene_xml_path,
            (float(eye[0]), float(eye[1]), float(eye[2])),
            (float(aim[0]), float(aim[1]), float(aim[2])),
        )

    def mount_poses(self, profile: RobotProfile, arms: int) -> tuple[tuple[str, MountPose], ...]:
        """Return where *arms* arms of *profile* stand, by prefix: pinned (``mounts``), else derived.

        Returns:
            One pose per arm, in :func:`arm_prefixes` order.

        Raises:
            ValueError: If the scene neither pins nor derives a layout for *arms* arms.
        """
        pinned = dict(self.mounts).get(arms)
        if pinned is not None:
            return tuple(zip(arm_prefixes(arms), pinned, strict=True))
        if arms != 2 or self.written_arms != 1:  # noqa: PLR2004
            msg = f"Scene {self.scene_id!r} has no layout for {arms} arm(s)"
            raise ValueError(msg)
        separation = dict(self.two_arm_separations).get(
            profile.name, self.two_arm_separation * self.layout_scale(profile)
        )
        half = separation / 2
        if self.two_arm_style == "side_by_side":
            poses = (MountPose((0.0, half, 0.0)), MountPose((0.0, -half, 0.0)))
        else:
            x, y = self.for_profile(profile, arms).spawn_center
            poses = (MountPose((x, y + half, 0.0), _yaw_quat(-90.0)), MountPose((x, y - half, 0.0), _yaw_quat(90.0)))
        return tuple(zip(BIMANUAL_PREFIXES, poses, strict=True))

    def for_profile(self, profile: RobotProfile, arms: int = 1) -> SceneConfig:
        """Return this scene laid out for *arms* arms of *profile*: its spawn centre and radii scaled with its reach.

        With two arms, the spawn arc's centre is ``two_arm_spawn_center`` when the scene sets one;
        either centre lies between the arms of a derived layout.

        Returns:
            This scene itself when nothing changes (a fixed layout, the SO-101), else a copy.
        """
        center = dict(self.profile_spawn_centers).get(profile.name, self.spawn_center)
        if arms == 2 and self.two_arm_spawn_center is not None:  # noqa: PLR2004
            center = self.two_arm_spawn_center
        scale = self.layout_scale(profile)
        if scale == 1.0 and center == self.spawn_center:  # noqa: RUF069 - exactly 1.0, see layout_scale
            return self
        return replace(
            self,
            spawn_center=(center[0] * scale, center[1] * scale),
            spawn_min_r=self.spawn_min_r * scale,
            spawn_max_r=self.spawn_max_r * scale,
        )

    def home_pose(self, arms: int = 1) -> tuple[tuple[str, float], ...]:
        """Return the scene's home pose for *arms* arms, by full joint name.

        Returns:
            ``pinned_home_qpos`` for that count, else ``home_qpos`` for each arm with its prefix.
        """
        pinned = dict(self.pinned_home_qpos).get(arms)
        if pinned is not None:
            return pinned
        return tuple((f"{prefix}{joint}", value) for prefix in arm_prefixes(arms) for joint, value in self.home_qpos)

    def load_model(
        self, profile: RobotProfile = SO101_PROFILE, arms: int | None = None, overview: OverviewStyle = "shoulder"
    ) -> mujoco.MjModel:
        """Compile this scene with *arms* arms of the profile's robot; ``None`` uses the file's mount frames.

        Returns:
            The compiled model, laid out for the profile, with the *overview* camera style.
        """
        return load_scene_model(self.scene_xml_path, profile, scene_layout=self.layout_for(profile, arms, overview))


# One arm reaches over the garment from the far end of the table.
_GARMENT_FOLD_ARM_HOME: tuple[tuple[str, float], ...] = (
    ("shoulder_pan", 0.0),
    ("shoulder_lift", 0.3),
    ("elbow_flex", 0.8),
    ("wrist_flex", 0.3),
)

# Two arms at the table's corners face each other; each pans toward the garment.
_GARMENT_FOLD_HOME: tuple[tuple[str, float], ...] = (
    ("left_shoulder_pan", -1.1),
    ("left_shoulder_lift", 0.3),
    ("left_elbow_flex", 0.8),
    ("left_wrist_flex", 0.3),
    ("right_shoulder_pan", 1.1),
    ("right_shoulder_lift", 0.3),
    ("right_elbow_flex", 0.8),
    ("right_wrist_flex", 0.3),
)

# Gripper hovering ~8 cm above the near belt rail, angled down at the pick zone.
_CONVEYOR_SORT_HOME: tuple[tuple[str, float], ...] = (
    ("shoulder_pan", 0.0),
    ("shoulder_lift", -1.4),
    ("elbow_flex", 0.7),
    ("wrist_flex", 1.6),
    ("wrist_roll", 0.0),
    ("gripper", 0.0),
)


# ---------------------------------------------------------------------------
# Scene reset functions
# ---------------------------------------------------------------------------


def _freejoint_spawn_reset(
    scene: SceneConfig, robot_roots: tuple[int, ...] = (), footprint_height: float = ARM_FOOTPRINT_HEIGHT
) -> ResetFn:
    """Build a reset that respawns a scene's free objects clear of its target and of the arms.

    The target body itself is left where it is; only the freejoints listed in
    the scene's ``free_joints`` are randomized, using that scene's spawn arc.

    Args:
        scene: The scene, laid out for its robot (:meth:`SceneConfig.for_profile`).
        robot_roots: Root body of every robot, whose footprint at reset time the objects avoid
            (:func:`~physicalai_mujoco_plugin.spawn.arm_footprint`); empty ignores the robots.
        footprint_height: Robot geoms higher above the table than this leave the footprint.

    Returns:
        A reset callback for the scene.
    """

    def reset(model: object, data: object, rng: np.random.Generator) -> None:
        import mujoco  # noqa: PLC0415

        target_body = scene.target_bodies[0] if scene.target_bodies else ""
        target_xy = read_body_xy(model, data, target_body, scene.spawn_center)
        footprint = arm_footprint(model, data, robot_roots, footprint_height)
        padding = object_radius(model, scene.free_joints) + ARM_CLEARANCE if footprint is not None else 0.0
        positions = sample_scene_positions(
            scene, len(scene.free_joints), rng=rng, target_xy=target_xy, footprint=footprint, padding=padding
        )

        for joint_name, xy in zip(scene.free_joints, positions, strict=True):
            place_freejoint(model, data, joint_name, xy, rng)

        mujoco.mj_forward(model, data)

    return reset


def _set_arm_pose(model: object, data: object, pose: tuple[tuple[str, float], ...]) -> None:
    """Put the named arm joints (and their position actuators) at `pose`, at rest."""
    import mujoco  # noqa: PLC0415

    for joint_name, val in pose:
        # pyrefly: ignore [missing-attribute]
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if jid < 0:
            continue
        # pyrefly: ignore [missing-attribute]
        qpos_adr = int(model.jnt_qposadr[jid])
        # pyrefly: ignore [missing-attribute]
        dof_adr = int(model.jnt_dofadr[jid])
        # pyrefly: ignore [missing-attribute]
        data.qpos[qpos_adr] = val
        # pyrefly: ignore [missing-attribute]
        data.qvel[dof_adr] = 0.0
        # pyrefly: ignore [missing-attribute]
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name)
        if aid >= 0:
            # pyrefly: ignore [missing-attribute]
            data.ctrl[aid] = val


def _garment_fold_reset(home: tuple[tuple[str, float], ...]) -> ResetFn:
    """Build the garment reset: the arms at *home*, the garment flat.

    Returns:
        A reset callback for the scene.
    """

    def reset(model: object, data: object, rng: np.random.Generator) -> None:  # noqa: ARG001
        _set_arm_pose(model, data, home)
        _flatten_garment(model, data)

    return reset


def _flatten_garment(model: object, data: object) -> None:
    import mujoco  # noqa: PLC0415

    # pyrefly: ignore [missing-attribute]
    if model.nflex > 0:
        # pyrefly: ignore [missing-attribute]
        first_vertex_body = int(model.flex_vertbodyid[0])
        vertex_joints = [j for j in range(model.njnt) if int(model.jnt_bodyid[j]) == first_vertex_body]
        if not vertex_joints:
            # A pinned first vertex has no DOFs, so there is no flex state to restore.
            logger.warning("Flex body {} has no joints; skipping garment reset", first_vertex_body)
        else:
            flex_qpos_adr = min(int(model.jnt_qposadr[j]) for j in vertex_joints)
            flex_dof_adr = min(int(model.jnt_dofadr[j]) for j in vertex_joints)
            data.qpos[flex_qpos_adr : flex_qpos_adr + 3 * model.nflexvert] = model.flex_vert.ravel()
            data.qvel[flex_dof_adr : flex_dof_adr + 3 * model.nflexvert] = 0.0

    # pyrefly: ignore [missing-attribute]
    mujoco.mj_forward(model, data)


def _yahtzee_reset(robot_roots: tuple[int, ...] = ()) -> ResetFn:
    """Build the dice reset: each die dropped around the cup, outside the arms' footprint at reset time.

    Args:
        robot_roots: Root body of every robot, whose footprint the dice's drop points avoid; empty
            ignores the robots.

    Returns:
        A reset callback for the scene.
    """

    def reset(model: object, data: object, rng: np.random.Generator) -> None:
        _drop_dice(model, data, rng, robot_roots)

    return reset


def _drop_dice(  # noqa: PLR0914
    model: object, data: object, rng: np.random.Generator, robot_roots: tuple[int, ...]
) -> None:
    import mujoco  # noqa: PLC0415

    die_joints = [f"die{i}:joint" for i in range(1, 7)]

    cx, cy = 0.30, 0.0
    cup_jitter = 0.005
    footprint = arm_footprint(model, data, robot_roots)
    padding = object_radius(model, tuple(die_joints)) + ARM_CLEARANCE if footprint is not None else 0.0

    def draw() -> tuple[float, float]:
        r = float(rng.uniform(0.06, 0.18))
        theta = float(rng.uniform(0.0, 2.0 * np.pi))
        x = cx + float(rng.uniform(-cup_jitter, cup_jitter)) + r * float(np.cos(theta))
        y = cy + float(rng.uniform(-cup_jitter, cup_jitter)) + r * float(np.sin(theta))
        return x, y

    for joint_name in die_joints:
        # pyrefly: ignore [missing-attribute]
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if jid < 0:
            continue
        # pyrefly: ignore [missing-attribute]
        qpos_adr = int(model.jnt_qposadr[jid])
        # pyrefly: ignore [missing-attribute]
        dof_adr = int(model.jnt_dofadr[jid])

        x, y = sample_clear_position(draw, footprint=footprint, padding=padding, attempts=SPAWN_ATTEMPTS)

        yaw = float(rng.uniform(0.0, 2.0 * np.pi))
        tilt = float(rng.uniform(-0.3, 0.3))
        c = np.cos(yaw / 2.0)
        s = np.sin(yaw / 2.0)
        drop_z = float(rng.uniform(0.12, 0.18))
        # pyrefly: ignore [missing-attribute]
        data.qpos[qpos_adr : qpos_adr + 3] = [x, y, drop_z]
        # pyrefly: ignore [missing-attribute]
        data.qpos[qpos_adr + 3 : qpos_adr + 7] = [
            c * np.sin(tilt / 2.0),
            s * np.sin(tilt / 2.0),
            s * np.cos(tilt / 2.0),
            c * np.cos(tilt / 2.0),
        ]

        vx = float(rng.uniform(-0.5, 0.5))
        vy = float(rng.uniform(-0.5, 0.5))
        vz = float(rng.uniform(-0.4, -0.05))
        wx = float(rng.uniform(-15.0, 15.0))
        wy = float(rng.uniform(-15.0, 15.0))
        wz = float(rng.uniform(-8.0, 8.0))
        # pyrefly: ignore [missing-attribute]
        data.qvel[dof_adr : dof_adr + 6] = [vx, vy, vz, wx, wy, wz]

    # pyrefly: ignore [missing-attribute]
    mujoco.mj_forward(model, data)


def _conveyor_sort_reset(home: tuple[tuple[str, float], ...]) -> ResetFn:
    """Build the conveyor reset: the arms at *home*, every item parked.

    Returns:
        A reset callback for the scene.
    """

    def reset(model: object, data: object, rng: np.random.Generator) -> None:  # noqa: ARG001
        # The belt feed (conveyor.ConveyorSort) randomizes each item as it enters.
        _set_arm_pose(model, data, home)
        park_items(model, data)

    return reset


# ---------------------------------------------------------------------------
# Scene configurations
# ---------------------------------------------------------------------------

_SCENES: dict[str, SceneConfig] = {
    "single_pick_place": SceneConfig(
        scene_id="single_pick_place",
        display_name="Single Pick & Place",
        description="One block and a target disc",
        scene_xml_relpath="scenes/single_pick_place/scene.xml",
        free_joints=("block1:joint",),
        target_bodies=("target",),
        # Keep spawns inside comfortable SO-101 teleop reach (was up to ~0.5 m).
        spawn_center=(0.22, 0.0),
        spawn_min_r=0.05,
        spawn_max_r=0.14,
        spawn_angle_half_deg=50.0,
        target_min_sep=0.11,
        layout="reach",
        # ALOHA's arms sit at x = -0.47 and +0.47 facing each other: its arc spans x = 0.
        profile_spawn_centers=(("aloha", (-0.086, -0.01)),),
        # Two arms stand 10 cm to either side: 2 cm closer, the arc's far edge stays within 15 degrees
        # of straight down for both (tests/test_reach.py).
        two_arm_spawn_center=(0.20, 0.0),
        # The UR5e's home pose reaches 54 cm sideways (+y) from its base, into the other arm.
        two_arm_separations=(("ur5e", 0.65),),
        # Pulled back along the line of sight until every arm's home pose, its geometry included,
        # is in frame (tests/test_compose.py); ALOHA's camera looks across the table along +y, with
        # the left arm on the left of the image. The SO-101, SO-ARM100 and Kinova fit as scaled.
        profile_overview_rigs=(
            ("trossen_wxai", OverviewRig(pos=(-0.416, 0.15, 1.625))),
            ("rebot_b601", OverviewRig(pos=(-0.382, 0.15, 1.554))),
            ("ur5e", OverviewRig(pos=(-0.404, 0.26, 2.294))),
            ("aloha", OverviewRig(pos=(0.0, -0.736, 1.125), quat=(0.0, 0.0, -(0.5**0.5), 0.5**0.5))),
            ("koch", OverviewRig(pos=(-0.136, 0.06, 0.6))),
            ("piper", OverviewRig(pos=(-0.182, 0.155, 1.285))),
            ("franka_fr3", OverviewRig(pos=(-0.313, 0.216, 1.88))),
            ("franka_panda", OverviewRig(pos=(-0.323, 0.231, 1.994))),
            ("xarm7", OverviewRig(pos=(-0.228, 0.212, 1.723))),
        ),
        # Far enough for the tallest home poses (Franka, xArm, Kinova); the WidowX AI's upright
        # forearm needs more (tests/test_compose.py).
        front_view=FrontView(distance=0.85),
        profile_front_overrides=(
            ("trossen_wxai", FrontOverride(distance=1.62)),
            # ALOHA's teleoperator sits at -y, where its shoulder camera stands; front faces its arms
            # from +y, steeper and farther so the target behind the far gripper stays in view.
            ("aloha", FrontOverride(distance=1.6, direction=(0.0, 1.0), elevation_deg=55.0)),
        ),
        robots=(
            "so101",
            "trossen_wxai",
            "rebot_b601",
            "ur5e",
            "aloha",
            "so_arm100",
            "koch",
            "piper",
            "franka_fr3",
            "franka_panda",
            "xarm7",
            "kinova_gen3",
        ),
    ),
    "yahtzee": SceneConfig(
        scene_id="yahtzee",
        display_name="Yahtzee",
        description="Pick up 6 dice and place them in the cup",
        scene_xml_relpath="scenes/yahtzee/scene.xml",
        free_joints=("die1:joint", "die2:joint", "die3:joint", "die4:joint", "die5:joint", "die6:joint"),
        target_bodies=(),
        spawn_center=(0.30, 0.0),
        spawn_min_r=0.06,
        spawn_max_r=0.18,
        spawn_angle_half_deg=160.0,
        block_min_sep=0.018,
        target_min_sep=0.02,
        front_view=FrontView(distance=0.8),
    ),
    "conveyor_sort": SceneConfig(
        scene_id="conveyor_sort",
        display_name="Conveyor Sort",
        description="Sort items off a moving belt by color; cracked or purple items go to reject",
        scene_xml_relpath="scenes/conveyor_sort/scene.xml",
        free_joints=tuple(f"{name}:joint" for name in pool_item_names()),
        # Two arms stand 6 cm behind the single arm's mount, clear of the bins on either side.
        mounts=((2, (MountPose((-0.06, 0.10, 0.0)), MountPose((-0.06, -0.10, 0.0)))),),
        home_qpos=_CONVEYOR_SORT_HOME,
        # The pick zone on the belt.
        front_view=FrontView(distance=0.75, aim=(0.25, 0.0, 0.06)),
    ),
    "garment_fold": SceneConfig(
        scene_id="garment_fold",
        display_name="Garment Fold",
        description="Fold a flexible garment lying flat on a table",
        scene_xml_relpath="scenes/garment_fold/scene.xml",
        written_arms=2,
        # The file pins two arms at the near corners of the table, facing each other. One arm faces the
        # garment from the far end, toward the overview camera, which sees it whole there.
        mounts=((1, (MountPose((0.30, 0.02, 0.40), _yaw_quat(180.0)),)),),
        home_qpos=_GARMENT_FOLD_ARM_HOME,
        pinned_home_qpos=((2, _GARMENT_FOLD_HOME),),
        robots=("so101", "trossen_wxai"),
        # The garment's centre on the table top.
        front_view=FrontView(distance=0.9, aim=(0.0, 0.02, 0.32)),
        profile_front_overrides=(("trossen_wxai", FrontOverride(distance=1.75)),),
    ),
    "floor_flat": SceneConfig(
        scene_id="floor_flat",
        display_name="Flat Floor",
        description="A floating-base robot (humanoid, quadruped) standing on a flat floor",
        scene_xml_relpath="scenes/floor_flat/scene.xml",
        robots="*",
        anchors="spawn",
        arm_counts=(1,),
    ),
}

_FREEJOINT_SPAWN_SCENES = ("single_pick_place",)
"""Scenes whose reset respawns their free objects in the spawn arc laid out for the robot."""

_RESET_FUNCTIONS: dict[str, Callable[[SceneConfig, int, tuple[int, ...]], ResetFn]] = {
    "yahtzee": lambda _scene, _arms, roots: _yahtzee_reset(roots),
    "conveyor_sort": lambda scene, arms, _roots: _conveyor_sort_reset(scene.home_pose(arms)),
    "garment_fold": lambda scene, arms, _roots: _garment_fold_reset(scene.home_pose(arms)),
}
"""Reset builders by scene id, for the scene, its number of arms and the root body of every robot."""


def get_scene(scene_id: str) -> SceneConfig:
    """Return the scene configuration identified by `scene_id`.

    Raises:
        KeyError: If `scene_id` is not registered.
    """
    if scene_id not in _SCENES:
        msg = f"Unknown scene {scene_id!r}. Available: {list(_SCENES)}"
        raise KeyError(msg)
    return _SCENES[scene_id]


def list_scenes() -> dict[str, SceneConfig]:
    """Return all scene configurations by ID."""
    return dict(_SCENES)


def list_scenes_for_arms(num_arms: int) -> dict[str, SceneConfig]:
    """Return the scenes that run `num_arms` SO-101 arms."""
    return list_scenes_for(SO101_PROFILE, num_arms)


def get_reset_fn(
    scene_id: str,
    profile: RobotProfile = SO101_PROFILE,
    arms: int = 1,
    robot_roots: tuple[int, ...] = (),
) -> ResetFn | None:
    """Return the reset callback for `scene_id` with *arms* of *profile*'s robot, if one is registered.

    Args:
        scene_id: The scene.
        profile: The robot profile, for the scene's layout.
        arms: The number of arms.
        robot_roots: Root body of every robot in the compiled scene (``Sim.robot_roots``): resets that
            spawn objects keep them out of these robots' footprint. Empty ignores the robots.
    """
    scene = get_scene(scene_id)
    if scene_id in _FREEJOINT_SPAWN_SCENES:
        height = ARM_FOOTPRINT_HEIGHT * scene.layout_scale(profile)
        return _freejoint_spawn_reset(scene.for_profile(profile, arms), robot_roots, height)
    build = _RESET_FUNCTIONS.get(scene_id)
    return build(scene, arms, robot_roots) if build is not None else None


def list_scenes_naming(profile_name: str) -> dict[str, SceneConfig]:
    """Return the scenes that list *profile_name* explicitly, not through ``"*"``.

    Unlike :func:`list_scenes_for`, this never loads a robot model, so the Studio catalog and the
    CLI can call it before anything is downloaded.

    Returns:
        The scenes by id, in registry order.
    """
    return {
        scene_id: scene for scene_id, scene in _SCENES.items() if scene.robots != "*" and profile_name in scene.robots
    }


def supported_arm_counts(profile: RobotProfile) -> tuple[int, ...]:
    """Return how many copies of *profile*'s robot one simulation can run, without loading its model.

    Returns:
        ``(1,)`` for a model that has several arms already (ALOHA, which lists one end effector per
        arm) and for floating-base robots (a default scene with spawn anchors), else ``(1, 2)``.
    """
    return (1,) if _single_robot_reason(profile) is not None else (1, 2)


def _single_robot_reason(profile: RobotProfile) -> str | None:
    """Return why *profile* runs one robot per simulation, without loading its model.

    Returns:
        The reason for a model that has several arms already (ALOHA) or a floating-base robot
        (a default scene with spawn anchors), else ``None``.
    """
    if len(profile.end_effectors) > 1:
        return f"The {profile.display_name} model has {len(profile.end_effectors)} arms already and runs alone"
    if profile.default_scene is not None and get_scene(profile.default_scene).anchors == "spawn":
        return f"The {profile.display_name} has a floating base; a simulation holds one floating-base robot"
    return None


def check_arm_count(scene: SceneConfig, profile: RobotProfile, arms: int) -> None:
    """Check that *scene* can run *arms* arms of *profile*.

    Raises:
        ValueError: If the profile's model has several arms already, the scene or the profile is for
            floating-base robots, or the scene does not support the count.
    """
    if arms == 1:
        return
    reason = _single_robot_reason(profile)
    if reason is not None:
        msg = f"{reason}; run it without bimanual"
        raise ValueError(msg)
    if scene.anchors == "spawn":
        msg = (
            f"Scene {scene.scene_id!r} holds one floating-base robot; two arms need a tabletop scene "
            "and a fixed-base arm"
        )
        raise ValueError(msg)
    if arms not in scene.arm_counts:
        msg = f"Scene {scene.scene_id!r} runs {' or '.join(map(str, scene.arm_counts))} arm(s), not {arms}"
        raise ValueError(msg)


def check_model_mounts(profile: RobotProfile, mounts: tuple[str, ...], *, bimanual: bool) -> None:
    """Check that a custom model's ``robot_mount`` frames can take *profile* (``model_path``, ``--model``).

    Args:
        profile: The robot profile attached at every mount.
        mounts: The model's mount prefixes, in document order.
        bimanual: Whether two arms were requested; the frames must then be exactly
            ``left_robot_mount`` and ``right_robot_mount``.

    Raises:
        ValueError: If several mounts or ``bimanual`` meet a profile that runs one robot (ALOHA, a
            floating base), or ``bimanual`` meets other mount frames.
    """
    if bimanual or len(mounts) > 1:
        reason = _single_robot_reason(profile)
        if reason is not None:
            msg = f"{reason} (the model has {len(mounts)} robot_mount frame(s))"
            raise ValueError(msg)
    if bimanual and mounts != BIMANUAL_PREFIXES:
        found = ", ".join(f"{prefix}robot_mount" for prefix in mounts) or "none"
        msg = f"bimanual needs a model whose mount frames are left_robot_mount and right_robot_mount; it has {found}"
        raise ValueError(msg)


def list_scenes_for(profile: RobotProfile, num_arms: int | None = None) -> dict[str, SceneConfig]:
    """Return the scenes that support *profile*, optionally only those that run *num_arms* of its robot (SCN-5)."""
    if num_arms is not None and num_arms not in supported_arm_counts(profile):
        return {}
    return {
        scene_id: scene
        for scene_id, scene in _SCENES.items()
        if scene.supports(profile) and (num_arms is None or num_arms in scene.arm_counts)
    }
