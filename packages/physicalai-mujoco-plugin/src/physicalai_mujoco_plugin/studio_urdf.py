# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Studio's 3D model of a simulated robot: a URDF generated from the MuJoCo model the simulation runs.

Studio draws a robot type from a URDF (``RobotAsset``) and moves it with the robot's observations
(``joint_map``), in the robot page, the environment view and dataset playback. Every catalog entry
but the SO-101's, which keep their hand-made URDFs, gets one generated here from the same robot
model the simulation compiles, so the 3D model is the simulated robot.

The URDF is built from the robot alone, attached the way its simulation attaches it: one robot at
the origin, two at the mounts of the profile's default scene. Every body is one link, every hinge
and slide joint one ``revolute`` or ``prismatic`` joint (with its axis, its offset from the body
origin), and every body without one hangs from its parent by a ``fixed`` joint. Its
visual geometry is the geoms MuJoCo draws (groups 0 to 2, not transparent): meshes as binary STL
from the compiled model's vertices, boxes, spheres and cylinders as URDF primitives, capsules as
a cylinder and two spheres, with their colour.

Each public channel (``shoulder_pan``, ``gripper``) is a virtual ``prismatic`` joint
(``shoulder_pan_channel``) that moves nothing; the model's joints ``mimic`` it with the channel's
unit conversion as multiplier and offset. Studio passes a prismatic joint's value through unchanged,
so each joint takes the position the simulation gives it: degrees become radians, a Franka Hand's
opening width becomes each finger's half, and joints that follow another through a joint equality
(the second finger) follow it here too. The passive joints of a closed linkage (the xArm7
gripper's fingers and inner knuckles, held by ``connect`` equalities) follow the channel that moves
them, with the line fitted through their solved positions across its range. ``joint_map`` then
maps each observation Studio records (``shoulder_pan.pos``) to its channel's virtual joint, and is
known from the channel names alone.
The virtual joints are not named after the observations, so Studio's joint panel, which matches
URDF joints to observations by name, does not read their placeholder limits as joint ranges.

MuJoCo's joint limits are soft, so a recorded position can lie just past one, while Studio's
urdf-loader clamps ``revolute`` and ``prismatic`` joints to their ``<limit>``. Hinges are therefore
``continuous`` joints, which it never clamps, with their range in ``<limit>``; slides keep a
``<limit>`` a full range (at least 0.1 m) wider than theirs. Every joint's MuJoCo range is also its
``<safety_controller>`` soft limits.

Generation compiles the model once, when Studio first asks for the URDF (``root_resolver``), and
the result is cached on disk per generator, plugin, Menagerie and MuJoCo version.
"""

# MuJoCo model/data attributes are supplied by the C extension at runtime.
# pyrefly: ignore-errors [missing-attribute]

from __future__ import annotations

import math
import os
import re
import shutil
import struct
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING
from xml.sax.saxutils import quoteattr  # nosec B406 - escapes the attribute values written here; nothing is parsed

import numpy as np
from loguru import logger

from physicalai_mujoco_plugin.profiles.dataset import FRANKA_PANDA_PROFILE

if TYPE_CHECKING:
    import mujoco

    from physicalai_mujoco_plugin.channels import Channel
    from physicalai_mujoco_plugin.compose import MountPose
    from physicalai_mujoco_plugin.profiles import RobotProfile
    from physicalai_mujoco_plugin.profiles.derive import DerivedLayout

GENERATOR_VERSION = 2
"""Bump when the generated files change, so caches of older generators are not served."""
URDF_FILE = "robot.urdf"
"""The URDF's name in its entry directory."""
CACHE_DIR_ENV = "PHYSICALAI_MUJOCO_CACHE_DIR"
"""Overrides the plugin's cache directory (the per-user cache directory)."""
ROOT_LINK = "world"
"""The URDF's root link: the frame the robots are mounted in."""
OBSERVATION_SUFFIX = ".pos"
"""Suffix of the observations Studio maps to URDF joints (``shoulder_pan.pos``)."""
CHANNEL_SUFFIX = "_channel"
"""Suffix of a channel's virtual joint and link (``shoulder_pan_channel``)."""
DEFAULT_PAIR_SEPARATION_M = 0.4
"""Distance between two robots whose default scene lays out no pair."""
_CHANNEL_LIMIT = 1e9
"""Limits of a channel's virtual joint: wide enough never to clamp a public value."""
_SLIDE_LIMIT_MARGIN_M = 0.1
"""Least distance past its range at which a slide joint's URDF limit clamps it."""
_UNLIMITED_SLIDE_M = 10.0
"""Limits of a slide joint without a range; a prismatic URDF joint always clamps to its limits."""
_VISIBLE_GROUPS = 3
"""Geom groups MuJoCo draws by default (0, 1, 2); groups 3 to 5 are collision and helper geoms."""
_DEFAULT_GEOM_RGBA = (0.5, 0.5, 0.5, 1.0)
"""A geom's compiled rgba when it sets none; its material's colour then shows."""
_LINKAGE_ITERATIONS = 200
"""Gauss-Newton steps that place a linkage's passive joints."""
_LINKAGE_TOLERANCE = 1e-10
"""Largest loop-closing residual (metres, radians) a placed linkage keeps."""
_LINKAGE_SAMPLES = 17
_LINKAGE_MAX_STEP = 0.05
"""Largest change of a passive joint (radians, metres) in one Gauss-Newton step."""
"""Channel values a linkage fit is sampled at."""
_LINKAGE_MIN_TRAVEL = 1e-6
"""Least travel of a passive joint over a channel's range that makes it follow the channel."""
_LINKAGE_FIT_WARNING = 1e-3
"""Fit error (radians, metres) above which a linear linkage follower is logged."""
_TEXTURE_ROLE_RGB = 1
"""``mjtTextureRole.mjTEXROLE_RGB``: a material's colour texture."""
_ELLIPSOID_SEGMENTS = (16, 10)
"""Longitude and latitude bands of a generated ellipsoid mesh."""
_EXTRA_MENAGERIE_MODELS: dict[str, tuple[str, ...]] = {
    "franka_fr3": (FRANKA_PANDA_PROFILE.menagerie_model,),
    "franka_duo": (FRANKA_PANDA_PROFILE.menagerie_model,),
}
"""Menagerie models a profile's customization loads besides its own: the FR3s' Franka Hand is the Panda's."""

_GENERATE_LOCK = threading.Lock()
_READY: set[Path] = set()
"""Entry directories this process has found or written."""


@dataclass(frozen=True)
class JointDrive:
    """How one model joint follows a channel: ``value = multiplier * public + offset`` from its zero pose."""

    channel: str
    """The channel's virtual joint (``shoulder_pan_channel``)."""
    multiplier: float
    offset: float


@dataclass(frozen=True)
class GeneratedUrdf:
    """What :func:`write_robot_urdf` wrote."""

    urdf: Path
    joint_map: dict[str, list[str]]
    drives: dict[str, JointDrive]
    """The model joints that follow a channel, by URDF joint name."""
    meshes: tuple[Path, ...]


# ----------------------------------------------------------------------
# Names and paths, without MuJoCo
# ----------------------------------------------------------------------


def entry_key(profile: RobotProfile, arms: int) -> str:
    """Return the directory of one catalog entry's URDF in the cache: ``<profile>-<arms>``."""
    return f"{profile.name}-{arms}"


def package_name(profile: RobotProfile, arms: int) -> str:
    """Return the URDF package its mesh URIs name: ``mujoco_<profile>_<arms>``."""
    return f"mujoco_{profile.name}_{arms}"


def channel_joint(name: str) -> str:
    """Return the virtual joint of a public channel, ``shoulder_pan_channel``."""
    return f"{name}{CHANNEL_SUFFIX}"


def joint_map(names: tuple[str, ...]) -> dict[str, list[str]]:
    """Return Studio's ``joint_map`` for the public channel *names*: each observation drives its virtual joint.

    Returns:
        ``{"shoulder_pan.pos": ["shoulder_pan_channel"], ...}``.
    """
    return {f"{name}{OBSERVATION_SUFFIX}": [channel_joint(name)] for name in names}


def _user_cache_dir() -> Path:
    override = os.environ.get(CACHE_DIR_ENV)
    if override:
        return Path(override)
    if sys.platform == "win32":
        return Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local") / "physicalai_mujoco" / "Cache"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "physicalai_mujoco"
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "physicalai_mujoco"


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


@lru_cache(maxsize=1)
def cache_version() -> str:
    """Return the cache's version directory: generator, plugin, Menagerie and MuJoCo versions.

    Returns:
        E.g. ``v1-0.4.1-menagerie2026.9.4-mujoco3.16.0``, with only ``[A-Za-z0-9._-]``.
    """
    tag = (
        f"v{GENERATOR_VERSION}-{_package_version('physicalai-mujoco-plugin')}"
        f"-menagerie{_package_version('mujoco-menagerie')}-mujoco{_package_version('mujoco')}"
    )
    return re.sub(r"[^A-Za-z0-9._-]", "_", tag)


def cache_root() -> Path:
    """Return the directory that holds every generated URDF of this version (the assets' root)."""
    return _user_cache_dir() / "studio_urdf" / cache_version()


def _menagerie_models(profile: RobotProfile) -> tuple[str, ...]:
    return (profile.menagerie_model, *_EXTRA_MENAGERIE_MODELS.get(profile.name, ()))


def models_available(profile: RobotProfile) -> bool:
    """Return whether the profile's robot model can be loaded without a download.

    Returns:
        ``True`` with a ``MENAGERIE_ROOT`` checkout, or when every Menagerie model the profile
        loads is in the cache.
    """
    import mujoco_menagerie  # noqa: PLC0415

    cache = mujoco_menagerie.Cache()
    if cache.root is not None:
        return True
    try:
        return all(cache.is_cached(mujoco_menagerie.get(name)) for name in _menagerie_models(profile))
    except mujoco_menagerie.MenagerieError:
        return False


_DERIVED_NAMES: dict[RobotProfile, tuple[str, ...]] = {}
"""Channel names read from model files; only found ones, so a model downloaded later is read then."""


def _derived_names(profile: RobotProfile) -> tuple[str, ...] | None:
    """Return the channel names a profile without overrides derives from its model, without compiling it.

    A model that is not downloaded yet is looked for again on the next call, so the names appear
    once ``physicalai-mujoco prefetch`` or a first simulation start has fetched it.

    Returns:
        One name per actuator, as :func:`~.profiles.derive.derive_profile` names it, or ``None``
        when the model would have to be downloaded or does not load.
    """
    cached = _DERIVED_NAMES.get(profile)
    if cached is not None:
        return cached
    if not models_available(profile):
        return None
    import mujoco  # noqa: PLC0415

    from physicalai_mujoco_plugin.compose import _customized_robot_spec  # noqa: PLC0415

    try:
        spec = _customized_robot_spec(profile)
    except (RuntimeError, ValueError) as exc:
        logger.debug("No channel names for {}: {}", profile.name, exc)
        return None
    joint_transmissions = {mujoco.mjtTrn.mjTRN_JOINT, mujoco.mjtTrn.mjTRN_JOINTINPARENT}
    moving = {mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE}
    names = []
    for index, actuator in enumerate(spec.actuators):
        name = actuator.name
        if not name and actuator.trntype in joint_transmissions:
            joint = spec.joint(actuator.target)
            if joint is not None and joint.type in moving:
                name = actuator.target
        names.append(name or f"actuator_{index}")
    _DERIVED_NAMES[profile] = tuple(names)
    return _DERIVED_NAMES[profile]


def public_names(profile: RobotProfile, prefixes: tuple[str, ...]) -> tuple[str, ...] | None:
    """Return the public channel names of *profile* attached with *prefixes*, compiling nothing.

    A profile with channel overrides names them; one without takes them from its model's
    actuators, which needs the model in the Menagerie cache.

    Returns:
        Every robot's names with its prefix, in prefix order, or ``None`` when the model is not
        available without a download.
    """
    names = tuple(channel.name for channel in profile.channels) or _derived_names(profile)
    if names is None:
        return None
    return tuple(f"{prefix}{name}" for prefix in prefixes for name in names)


# ----------------------------------------------------------------------
# Cache
# ----------------------------------------------------------------------


def ensure_robot_urdf(profile: RobotProfile, prefixes: tuple[str, ...]) -> Path:
    """Return the cache root with *profile*'s URDF for *prefixes* in it, generating it on first use.

    Studio calls this (the asset's ``root_resolver``) for the URDF and for every mesh, so a
    generated entry costs one ``stat``. Generation writes into a temporary directory next to the
    entry and renames it into place, so concurrent processes never see half an entry; one that
    loses the race drops its copy. A model that would have to be downloaded, or that fails to
    build, is logged and left out: Studio then answers 404 for the URDF.

    Returns:
        The resolved cache root (:func:`cache_root`).
    """
    root = cache_root()
    key = entry_key(profile, len(prefixes))
    entry = root / key
    if entry in _READY or (entry / URDF_FILE).is_file():
        _READY.add(entry)
        return root.resolve()
    with _GENERATE_LOCK:
        if (entry / URDF_FILE).is_file():
            _READY.add(entry)
            return root.resolve()
        if not models_available(profile):
            logger.warning(
                "No 3D model for {}: its MuJoCo Menagerie model is not downloaded (run `physicalai-mujoco prefetch`)",
                profile.display_name,
            )
            root.mkdir(parents=True, exist_ok=True)
            return root.resolve()
        root.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{key}.", dir=root))
        started = time.perf_counter()
        try:
            write_robot_urdf(profile, prefixes, staging)
        except Exception:  # noqa: BLE001 - any failure only costs the 3D model, which Studio reports
            logger.exception("Could not generate the 3D model of {}", profile.display_name)
            shutil.rmtree(staging, ignore_errors=True)
            return root.resolve()
        try:
            staging.rename(entry)
        except OSError:  # another process put the entry in place first
            shutil.rmtree(staging, ignore_errors=True)
        logger.info(
            "Generated the 3D model of {} in {:.2f} s ({})",
            profile.display_name,
            time.perf_counter() - started,
            entry,
        )
        _READY.add(entry)
    return root.resolve()


# ----------------------------------------------------------------------
# Composing the robot
# ----------------------------------------------------------------------


def mount_poses(profile: RobotProfile, prefixes: tuple[str, ...]) -> tuple[tuple[str, MountPose], ...]:
    """Return where each robot stands: the origin for one, the default scene's mounts for two.

    Returns:
        ``(prefix, pose)`` per robot; a pair the default scene does not lay out stands side by side,
        :data:`DEFAULT_PAIR_SEPARATION_M` apart, facing +x.
    """
    from physicalai_mujoco_plugin.compose import MountPose  # noqa: PLC0415
    from physicalai_mujoco_plugin.scene_registry import get_scene  # noqa: PLC0415

    if len(prefixes) == 1:
        return ((prefixes[0], MountPose((0.0, 0.0, 0.0))),)
    if profile.default_scene is not None:
        try:
            poses = get_scene(profile.default_scene).mount_poses(profile, len(prefixes))
        except (KeyError, ValueError):
            pass
        else:
            if tuple(prefix for prefix, _ in poses) == prefixes:
                return poses
    half = DEFAULT_PAIR_SEPARATION_M / 2
    offsets = np.linspace(half, -half, len(prefixes))
    return tuple((prefix, MountPose((0.0, float(y), 0.0))) for prefix, y in zip(prefixes, offsets, strict=True))


def compose_robot(
    profile: RobotProfile, prefixes: tuple[str, ...]
) -> tuple[mujoco.MjModel, tuple[tuple[str, DerivedLayout], ...]]:
    """Compile *profile*'s robot alone, attached at one mount (or spawn) frame per prefix, as scenes attach it.

    Returns:
        The model, and each robot's prefix and layout bound to it.
    """
    import mujoco  # noqa: PLC0415

    from physicalai_mujoco_plugin.compose import (  # noqa: PLC0415
        ROBOT_MOUNT_FRAME,
        ROBOT_SPAWN_FRAME,
        attach_robot,
        bind_layout,
        robot_layout,
    )

    layout = robot_layout(profile)
    kind = "mount" if layout.base is None else "spawn"
    suffix = ROBOT_MOUNT_FRAME if kind == "mount" else ROBOT_SPAWN_FRAME
    scene = mujoco.MjSpec()
    poses = mount_poses(profile, prefixes)
    for prefix, pose in poses:
        scene.worldbody.add_frame(name=f"{prefix}{suffix}", pos=list(pose.pos), quat=list(pose.quat))
    for prefix, _ in poses:
        attach_robot(scene, profile, prefix, kind)
    model = scene.compile()
    return model, tuple((prefix, bind_layout(layout, model, prefix)) for prefix, _ in poses)


def bind_channels(
    model: mujoco.MjModel, robots: tuple[tuple[str, DerivedLayout], ...], profile: RobotProfile
) -> tuple[Channel, ...]:
    """Return every robot's public channels in the profile's default unit, as the simulation binds them.

    Returns:
        The channels of all robots, in robot order.
    """
    import mujoco  # noqa: PLC0415

    from physicalai_mujoco_plugin.channels import ArmChannels  # noqa: PLC0415

    data = mujoco.MjData(model)
    channels: list[Channel] = []
    for prefix, layout in robots:
        try:
            arm = ArmChannels(model, data, layout, profile, prefix=prefix)
        except ValueError:
            arm = ArmChannels(model, data, layout, profile, prefix=prefix, torque_mode="raw")
        channels.extend(arm.channels)
    return tuple(channels)


def _public_to_model(channel: Channel) -> tuple[float, float]:
    """Return ``(a, b)`` with ``model value = a * public + b`` for *channel* (inside its range, if normalized).

    Returns:
        The slope and intercept of the channel's conversion to model units.
    """
    from physicalai_mujoco_plugin.channels import ArmChannels  # noqa: PLC0415

    if channel.unit == "normalized" and channel.range is not None:
        low, high = channel.range
        public = (ArmChannels.to_public(channel, low), ArmChannels.to_public(channel, high))
        model = (low, high)
    else:
        public = (0.0, 1.0)
        model = (ArmChannels.to_model(channel, 0.0), ArmChannels.to_model(channel, 1.0))
    slope = (model[1] - model[0]) / (public[1] - public[0])
    return slope, model[0] - slope * public[0]


def joint_drives(model: mujoco.MjModel, channels: tuple[Channel, ...]) -> dict[int, JointDrive]:
    """Return how each model joint a channel moves follows it, from its zero pose (``qpos0``).

    A joint member takes the channel's model value times its member scale; the joints of a fixed
    tendon (the Franka Hand's width) share the tendon length in proportion to their coefficients,
    as the hand's equal fingers do. Joints that a joint equality ties to a driven joint follow it
    through the equality's linear term.

    Returns:
        The drives by joint id; the first channel that moves a joint wins.
    """
    import mujoco  # noqa: PLC0415

    drives: dict[int, JointDrive] = {}

    def drive(joint_id: int, name: str, multiplier: float, offset: float) -> None:
        if joint_id not in drives:
            qpos0 = float(model.qpos0[model.jnt_qposadr[joint_id]])
            drives[joint_id] = JointDrive(name, multiplier, offset - qpos0)

    for channel in channels:
        slope, intercept = _public_to_model(channel)
        name = channel_joint(channel.name)
        for position, member in enumerate(channel.members):
            scale = 1.0 if position == 0 or not channel.member_scales else channel.member_scales[position - 1]
            if member.joint_id is not None:
                drive(member.joint_id, name, slope * scale, intercept * scale)
            elif member.transmission == "tendon":
                for joint_id, share in _tendon_shares(model, int(model.actuator_trnid[member.actuator_id, 0])):
                    drive(joint_id, name, slope * scale * share, intercept * scale * share)
    _follow_joint_equalities(model, drives, mujoco.mjtEq.mjEQ_JOINT)
    _fit_linkages(model, drives)
    return drives


def _tendon_shares(model: mujoco.MjModel, tendon_id: int) -> list[tuple[int, float]]:
    """Return each joint of a fixed tendon and its share of the tendon's length (the least-norm split).

    Returns:
        ``(joint id, coef / sum of squared coefs)`` per joint; empty for a spatial tendon.
    """
    import mujoco  # noqa: PLC0415

    start, count = int(model.tendon_adr[tendon_id]), int(model.tendon_num[tendon_id])
    wraps = range(start, start + count)
    if any(int(model.wrap_type[w]) != mujoco.mjtWrap.mjWRAP_JOINT for w in wraps):
        return []
    coefs = [(int(model.wrap_objid[w]), float(model.wrap_prm[w])) for w in wraps]
    norm = sum(coef * coef for _, coef in coefs)
    return [(joint_id, coef / norm) for joint_id, coef in coefs] if norm else []


def _follow_joint_equalities(model: mujoco.MjModel, drives: dict[int, JointDrive], joint_eq: int) -> None:
    """Drive the joints that an active joint equality ties to a driven joint (``y = c0 + c1 x``, from ``qpos0``)."""
    equalities = [
        (int(model.eq_obj1id[e]), int(model.eq_obj2id[e]), float(model.eq_data[e, 0]), float(model.eq_data[e, 1]))
        for e in range(model.neq)
        if int(model.eq_type[e]) == joint_eq and bool(model.eq_active0[e]) and int(model.eq_obj2id[e]) >= 0
    ]
    changed = True
    while changed:
        changed = False
        for follower, leader, c0, c1 in equalities:
            if follower not in drives and leader in drives:
                lead = drives[leader]
                drives[follower] = JointDrive(lead.channel, c1 * lead.multiplier, c0 + c1 * lead.offset)
                changed = True
            elif leader not in drives and follower in drives and c1:
                lead = drives[follower]
                drives[leader] = JointDrive(lead.channel, lead.multiplier / c1, (lead.offset - c0) / c1)
                changed = True


def _linkage_equalities(model: mujoco.MjModel) -> set[int]:
    """Return the active ``connect`` and ``weld`` equalities that close a loop within the robot.

    A weld to the world (a spawned robot's base hold) closes none.

    Returns:
        Their ids.
    """
    import mujoco  # noqa: PLC0415

    linkage = set()
    for equality in range(model.neq):
        kind = int(model.eq_type[equality])
        if kind not in {mujoco.mjtEq.mjEQ_CONNECT, mujoco.mjtEq.mjEQ_WELD} or not bool(model.eq_active0[equality]):
            continue
        bodies = [int(model.eq_obj1id[equality]), int(model.eq_obj2id[equality])]
        if int(model.eq_objtype[equality]) == mujoco.mjtObj.mjOBJ_SITE:
            bodies = [int(model.site_bodyid[site]) if site >= 0 else 0 for site in bodies]
        if 0 not in bodies:
            linkage.add(equality)
    return linkage


def _channel_span(model: mujoco.MjModel, drives: dict[int, JointDrive], channel: str) -> tuple[float, float]:
    """Return the public values that keep every limited joint *channel* drives within its range.

    Returns:
        ``(low, high)``; ``(-1, 1)`` when no driven joint is limited.
    """
    low, high = -math.inf, math.inf
    for joint, drive in drives.items():
        if drive.channel != channel or not drive.multiplier or not bool(model.jnt_limited[joint]):
            continue
        qpos0 = float(model.qpos0[model.jnt_qposadr[joint]])
        ends = sorted((float(end) - qpos0 - drive.offset) / drive.multiplier for end in model.jnt_range[joint])
        low, high = max(low, ends[0]), min(high, ends[1])
    if not (math.isfinite(low) and math.isfinite(high) and low < high):
        return -1.0, 1.0
    return low, high


class _LinkageSolver:
    """Place a model's passive joints where its loop-closing equalities put them, for given channel values.

    The passive joints are solved by Gauss-Newton on the equalities' residuals, with contacts and
    limits off: the pose a settled simulation reaches, without simulating.
    """

    def __init__(self, model: mujoco.MjModel, drives: dict[int, JointDrive], equalities: set[int]) -> None:
        import copy  # noqa: PLC0415

        import mujoco  # noqa: PLC0415

        self.model = copy.copy(model)
        self.model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT) | int(mujoco.mjtDisableBit.mjDSBL_LIMIT)
        self.model.opt.jacobian = mujoco.mjtJacobian.mjJAC_DENSE
        self.data = mujoco.MjData(self.model)
        self.drives = drives
        self.equalities = equalities
        moving = {mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE}
        self.passive = [j for j in range(model.njnt) if int(model.jnt_type[j]) in moving and j not in drives]

    def solve(self, channels: dict[str, float], start: np.ndarray | None = None) -> np.ndarray:
        """Return each passive joint's value from its zero pose, with the driven joints at *channels*.

        The solve starts from *start* (passive values, default the zero pose) and takes short steps,
        so sweeping a channel in small increments stays on one branch of the linkage.
        """
        import mujoco  # noqa: PLC0415

        model, data = self.model, self.data
        data.qpos[:] = model.qpos0
        for joint, drive in self.drives.items():
            address = model.jnt_qposadr[joint]
            data.qpos[address] = (
                model.qpos0[address] + drive.multiplier * channels.get(drive.channel, 0.0) + drive.offset
            )
        addresses = np.array([model.jnt_qposadr[j] for j in self.passive])
        dofs = np.array([model.jnt_dofadr[j] for j in self.passive])
        if start is not None:
            data.qpos[addresses] = model.qpos0[addresses] + start
        for _ in range(_LINKAGE_ITERATIONS):
            mujoco.mj_forward(model, data)
            rows = [
                row
                for row in range(data.nefc)
                if int(data.efc_type[row]) == mujoco.mjtConstraint.mjCNSTR_EQUALITY
                and int(data.efc_id[row]) in self.equalities
            ]
            residual = data.efc_pos[rows]
            if not rows or float(np.abs(residual).max()) < _LINKAGE_TOLERANCE:
                break
            jacobian = data.efc_J.reshape(data.nefc, model.nv)[rows][:, dofs]
            step = np.linalg.lstsq(jacobian, residual, rcond=None)[0]
            largest = float(np.abs(step).max())
            if largest > _LINKAGE_MAX_STEP:
                step *= _LINKAGE_MAX_STEP / largest
            data.qpos[addresses] -= step
        return data.qpos[addresses] - model.qpos0[addresses]


def _fit_linkages(model: mujoco.MjModel, drives: dict[int, JointDrive]) -> None:
    """Drive the passive joints of closed linkages (the xArm7 gripper's fingers and inner knuckles).

    Each channel is swept across its range; a passive joint the sweep moves follows that channel
    with the straight line fitted through the solved positions, the best a URDF ``mimic`` can do.
    """
    equalities = _linkage_equalities(model)
    if not equalities:
        return
    solver = _LinkageSolver(model, drives, equalities)
    if not solver.passive:
        return
    fitted: dict[int, tuple[float, JointDrive]] = {}
    for channel in dict.fromkeys(drive.channel for drive in drives.values()):
        low, high = _channel_span(model, drives, channel)
        samples = np.linspace(low, high, _LINKAGE_SAMPLES)
        # Continue from each solution to the next; start from the zero pose's nearest end.
        order = range(len(samples)) if abs(low) <= abs(high) else range(len(samples) - 1, -1, -1)
        solved = np.zeros((len(samples), len(solver.passive)))
        previous = None
        for index in order:
            previous = solved[index] = solver.solve({channel: float(samples[index])}, previous)
        for index, joint in enumerate(solver.passive):
            travel = float(np.ptp(solved[:, index]))
            if travel < _LINKAGE_MIN_TRAVEL or (joint in fitted and fitted[joint][0] >= travel):
                continue
            slope, intercept = np.polyfit(samples, solved[:, index], 1)
            error = float(np.abs(slope * samples + intercept - solved[:, index]).max())
            if error > _LINKAGE_FIT_WARNING:
                logger.debug("{} follows {} within {:.4f} only", model.joint(joint).name, channel, error)
            fitted[joint] = (travel, JointDrive(channel, float(slope), float(intercept)))
    for joint, (_, drive) in fitted.items():
        drives[joint] = drive


# ----------------------------------------------------------------------
# Writing the URDF
# ----------------------------------------------------------------------


def write_robot_urdf(profile: RobotProfile, prefixes: tuple[str, ...], directory: Path) -> GeneratedUrdf:
    """Compile *profile*'s robot for *prefixes* and write its URDF and meshes into *directory*.

    Returns:
        The files written, the ``joint_map`` and each driven joint's drive.

    Raises:
        ValueError: If the model's channel names differ from :func:`public_names`' (Studio's
            ``joint_map`` would then miss them).
    """
    model, robots = compose_robot(profile, prefixes)
    channels = bind_channels(model, robots, profile)
    names = tuple(channel.name for channel in channels)
    expected = public_names(profile, prefixes)
    if expected is not None and names != expected:
        msg = f"{profile.name}: the model's channels {names} differ from the catalog's {expected}"
        raise ValueError(msg)
    writer = _UrdfWriter(model, package_name(profile, len(prefixes)), directory)
    drives = joint_drives(model, channels)
    text = writer.build(f"mujoco_{profile.name}", names, drives)
    urdf = directory / URDF_FILE
    urdf.write_text(text, encoding="utf-8")
    return GeneratedUrdf(
        urdf=urdf,
        joint_map=joint_map(names),
        drives={writer.joint_names[joint_id]: drive for joint_id, drive in drives.items()},
        meshes=tuple(writer.meshes),
    )


def _fmt(value: float) -> str:
    text = f"{value:.12g}"
    return "0" if text == "-0" else text


def _vec(values: np.ndarray | tuple[float, ...] | list[float]) -> str:
    return " ".join(_fmt(float(v)) for v in values)


def quat_to_matrix(quat: np.ndarray | tuple[float, ...]) -> np.ndarray:
    """Return the rotation matrix of a ``wxyz`` quaternion."""
    w, x, y, z = (float(v) for v in quat)
    norm = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def matrix_to_rpy(rotation: np.ndarray) -> tuple[float, float, float]:
    """Return URDF roll, pitch, yaw (``R = Rz(yaw) Ry(pitch) Rx(roll)``) of a rotation matrix."""
    pitch = math.atan2(-rotation[2, 0], math.hypot(rotation[0, 0], rotation[1, 0]))
    if math.hypot(rotation[0, 0], rotation[1, 0]) < 1e-9:  # noqa: PLR2004 - gimbal lock: roll absorbs yaw
        return 0.0, pitch, math.atan2(-rotation[0, 1], rotation[1, 1])
    return math.atan2(rotation[2, 1], rotation[2, 2]), pitch, math.atan2(rotation[1, 0], rotation[0, 0])


def _origin(pos: np.ndarray | tuple[float, ...], rotation: np.ndarray | None = None) -> str:
    rpy = (0.0, 0.0, 0.0) if rotation is None else matrix_to_rpy(rotation)
    return f'<origin xyz="{_vec(pos)}" rpy="{_vec(rpy)}"/>'


def _linear(channel: float) -> float:
    """Convert an sRGB colour channel to linear: three.js takes URDF colours as linear.

    Returns:
        The linear channel value.
    """
    return channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4  # noqa: PLR2004


def write_binary_stl(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    """Write a triangle mesh as binary STL, with face normals from the vertex winding."""
    triangles = np.asarray(vertices, dtype=np.float64)[np.asarray(faces, dtype=np.int64)]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(normals, lengths, out=np.zeros_like(normals), where=lengths > 0)
    record = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")])
    rows = np.zeros(len(triangles), dtype=record)
    rows["normal"] = normals
    rows["vertices"] = triangles
    header = b"physicalai-mujoco-plugin mesh".ljust(80, b" ")
    with path.open("wb") as stream:
        stream.write(header)
        stream.write(struct.pack("<I", len(rows)))
        stream.write(rows.tobytes())


def ellipsoid_mesh(radii: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the vertices and faces of a UV ellipsoid with half-axes *radii*."""
    longitudes, latitudes = _ELLIPSOID_SEGMENTS
    theta = np.linspace(0.0, np.pi, latitudes + 1)
    phi = np.linspace(0.0, 2 * np.pi, longitudes, endpoint=False)
    grid_theta, grid_phi = np.meshgrid(theta, phi, indexing="ij")
    unit = np.stack(
        [np.sin(grid_theta) * np.cos(grid_phi), np.sin(grid_theta) * np.sin(grid_phi), np.cos(grid_theta)], axis=-1
    ).reshape(-1, 3)
    faces = []
    for row in range(latitudes):
        for col in range(longitudes):
            a = row * longitudes + col
            b = row * longitudes + (col + 1) % longitudes
            c, d = a + longitudes, b + longitudes
            faces += [(a, c, b), (b, c, d)]
    return unit * np.asarray(radii, dtype=np.float64), np.array(faces, dtype=np.int64)


class _UrdfWriter:
    """Turn a compiled robot-only model into URDF text and mesh files."""

    def __init__(self, model: mujoco.MjModel, package: str, directory: Path) -> None:
        self.model = model
        self.package = package
        self.directory = directory
        self.meshes: list[Path] = []
        self.joint_names: dict[int, str] = {}
        """URDF joint name of every model hinge and slide joint, by joint id."""
        self._links: set[str] = set()
        self._joints: set[str] = set()
        self._materials: dict[tuple[float, ...], str] = {}
        self._mesh_files: dict[int, str] = {}
        self._body_links: dict[int, str] = {}

    @staticmethod
    def _unique(taken: set[str], name: str) -> str:
        candidate, index = name, 1
        while candidate in taken:
            index += 1
            candidate = f"{name}_{index}"
        taken.add(candidate)
        return candidate

    def build(self, robot_name: str, channel_names: tuple[str, ...], drives: dict[int, JointDrive]) -> str:
        import mujoco  # noqa: PLC0415

        model = self.model
        for name in (ROOT_LINK, *(channel_joint(n) for n in channel_names)):
            if name in self._links:
                msg = f"Two URDF links would be named {name!r}"
                raise ValueError(msg)
            self._links.add(name)
        self._joints.update(channel_joint(n) for n in channel_names)
        self._body_links[0] = ROOT_LINK
        for body in range(1, model.nbody):
            self._body_links[body] = self._unique(self._links, model.body(body).name or f"body_{body}")
        for joint in range(model.njnt):
            if int(model.jnt_type[joint]) in {mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE}:
                body_name = self._body_links[int(model.jnt_bodyid[joint])]
                self.joint_names[joint] = self._unique(self._joints, model.joint(joint).name or f"{body_name}_joint")
        links: list[str] = []
        joints: list[str] = []
        for name in channel_names:
            virtual = channel_joint(name)
            links.append(f"  <link name={quoteattr(virtual)}/>")
            joints.append(
                f'  <joint name={quoteattr(virtual)} type="prismatic">\n'
                f"    <parent link={quoteattr(ROOT_LINK)}/>\n    <child link={quoteattr(virtual)}/>\n"
                f'    {_origin((0.0, 0.0, 0.0))}\n    <axis xyz="1 0 0"/>\n'
                f'    <limit lower="{_fmt(-_CHANNEL_LIMIT)}" upper="{_fmt(_CHANNEL_LIMIT)}" effort="0" velocity="0"/>\n'
                "  </joint>"
            )
        links.insert(0, self._link(0))
        for body in range(1, model.nbody):
            links.append(self._link(body))
            joints.extend(self._body_joints(body, drives, links))
        materials = [
            f'  <material name="{name}"><color rgba="{_vec(rgba)}"/></material>'
            for rgba, name in self._materials.items()
        ]
        return "\n".join([
            '<?xml version="1.0"?>',
            f"<!-- Generated by physicalai-mujoco-plugin (generator {GENERATOR_VERSION}) from the MuJoCo model. -->",
            f"<robot name={quoteattr(robot_name)}>",
            *materials,
            *links,
            *joints,
            "</robot>",
            "",
        ])

    def _body_joints(self, body: int, drives: dict[int, JointDrive], links: list[str]) -> list[str]:
        """Return the URDF joints from a body's parent link to its link: one per hinge or slide joint.

        A joint whose axis does not pass through the body origin (``jnt_pos``) sits at its anchor,
        and the body hangs from it by the opposite offset.
        """
        model = self.model
        parent = self._body_links[int(model.body_parentid[body])]
        link = self._body_links[body]
        rotation = quat_to_matrix(model.body_quat[body])
        offset = np.asarray(model.body_pos[body], dtype=np.float64)
        moving = [
            joint
            for joint in range(
                int(model.body_jntadr[body]), int(model.body_jntadr[body]) + int(model.body_jntnum[body])
            )
            if joint in self.joint_names
        ]
        out: list[str] = []
        for index, joint in enumerate(moving):
            anchor = np.asarray(model.jnt_pos[joint], dtype=np.float64)
            last = index == len(moving) - 1
            child = link if last and not anchor.any() else self._unique(self._links, f"{self.joint_names[joint]}_frame")
            if child != link:
                links.append(f"  <link name={quoteattr(child)}/>")
            out.append(self._moving_joint(joint, parent, child, offset + rotation @ anchor, rotation, drives))
            parent, offset, rotation = child, -anchor, np.eye(3)
        if parent != link or not moving:
            name = self._unique(self._joints, f"{link}_fixed")
            out.append(
                f'  <joint name={quoteattr(name)} type="fixed">\n'
                f"    <parent link={quoteattr(parent)}/>\n    <child link={quoteattr(link)}/>\n"
                f"    {_origin(offset, rotation)}\n  </joint>"
            )
        return out

    def _moving_joint(
        self,
        joint: int,
        parent: str,
        child: str,
        pos: np.ndarray,
        rotation: np.ndarray,
        drives: dict[int, JointDrive],
    ) -> str:
        import mujoco  # noqa: PLC0415

        model = self.model
        hinge = int(model.jnt_type[joint]) == mujoco.mjtJoint.mjJNT_HINGE
        qpos0 = float(model.qpos0[model.jnt_qposadr[joint]])
        limited = bool(model.jnt_limited[joint])
        # MuJoCo's limits are soft and a recording can sit just past one, but urdf-loader clamps every
        # revolute and prismatic joint to its <limit>. Hinges are continuous (never clamped) and keep
        # their range in <limit>; slides clamp only well outside theirs. Both keep it as soft limits.
        limit = safety = ""
        if limited:
            lower, upper = float(model.jnt_range[joint, 0]) - qpos0, float(model.jnt_range[joint, 1]) - qpos0
            safety = (
                f'\n    <safety_controller soft_lower_limit="{_fmt(lower)}" soft_upper_limit="{_fmt(upper)}"'
                ' k_velocity="0"/>'
            )
            if not hinge:
                margin = max(upper - lower, _SLIDE_LIMIT_MARGIN_M)
                lower, upper = lower - margin, upper + margin
        else:
            lower, upper = -_UNLIMITED_SLIDE_M, _UNLIMITED_SLIDE_M
        if limited or not hinge:
            limit = f'\n    <limit lower="{_fmt(lower)}" upper="{_fmt(upper)}" effort="0" velocity="0"/>'
        kind = "continuous" if hinge else "prismatic"
        drive = drives.get(joint)
        mimic = (
            ""
            if drive is None
            else f'\n    <mimic joint={quoteattr(drive.channel)} multiplier="{_fmt(drive.multiplier)}"'
            f' offset="{_fmt(drive.offset)}"/>'
        )
        return (
            f'  <joint name={quoteattr(self.joint_names[joint])} type="{kind}">\n'
            f"    <parent link={quoteattr(parent)}/>\n    <child link={quoteattr(child)}/>\n"
            f'    {_origin(pos, rotation)}\n    <axis xyz="{_vec(model.jnt_axis[joint])}"/>'
            f"{limit}{safety}{mimic}\n  </joint>"
        )

    def _link(self, body: int) -> str:
        visuals = [visual for geom in self._body_geoms(body) for visual in self._visuals(geom)]
        name = quoteattr(self._body_links[body])
        if not visuals:
            return f"  <link name={name}/>"
        return "\n".join([f"  <link name={name}>", *visuals, "  </link>"])

    def _body_geoms(self, body: int) -> list[int]:
        model = self.model
        start, count = int(model.body_geomadr[body]), int(model.body_geomnum[body])
        if start < 0:
            return []
        return [
            geom
            for geom in range(start, start + count)
            if int(model.geom_group[geom]) < _VISIBLE_GROUPS and self._rgba(geom)[3] > 0
        ]

    def _rgba(self, geom: int) -> tuple[float, ...]:
        model = self.model
        rgba = tuple(float(v) for v in model.geom_rgba[geom])
        material = int(model.geom_matid[geom])
        if material >= 0 and np.allclose(rgba, _DEFAULT_GEOM_RGBA):
            rgba = tuple(float(v) for v in model.mat_rgba[material])
            texture = int(model.mat_texid[material, _TEXTURE_ROLE_RGB])
            if texture >= 0:  # STL has no texture coordinates: tint with the texture's mean colour
                mean = self._texture_colour(texture)
                rgba = (rgba[0] * mean[0], rgba[1] * mean[1], rgba[2] * mean[2], rgba[3])
        return rgba

    def _texture_colour(self, texture: int) -> tuple[float, float, float]:
        model = self.model
        start = int(model.tex_adr[texture])
        channels = int(model.tex_nchannel[texture])
        count = int(model.tex_width[texture]) * int(model.tex_height[texture]) * channels
        pixels = np.asarray(model.tex_data[start : start + count], dtype=np.float64).reshape(-1, channels)
        if channels < 3:  # noqa: PLR2004 - a grey texture
            pixels = np.repeat(pixels[:, :1], 3, axis=1)
        red, green, blue = (float(v) / 255.0 for v in pixels[:, :3].mean(axis=0))
        return red, green, blue

    def _material(self, geom: int) -> str:
        r, g, b, a = self._rgba(geom)
        key = (round(_linear(r), 4), round(_linear(g), 4), round(_linear(b), 4), round(a, 4))
        if key not in self._materials:
            self._materials[key] = f"colour_{len(self._materials)}"
        return self._materials[key]

    def _visuals(self, geom: int) -> list[str]:
        import mujoco  # noqa: PLC0415

        model = self.model
        kind = int(model.geom_type[geom])
        size = np.asarray(model.geom_size[geom], dtype=np.float64)
        pos = np.asarray(model.geom_pos[geom], dtype=np.float64)
        rotation = quat_to_matrix(model.geom_quat[geom])
        shapes: list[tuple[np.ndarray, str]] = []
        if kind == mujoco.mjtGeom.mjGEOM_MESH:
            shapes.append((pos, f'<mesh filename="{self._mesh_file(int(model.geom_dataid[geom]))}"/>'))
        elif kind == mujoco.mjtGeom.mjGEOM_BOX:
            shapes.append((pos, f'<box size="{_vec(2 * size)}"/>'))
        elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
            shapes.append((pos, f'<sphere radius="{_fmt(size[0])}"/>'))
        elif kind in {mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_CAPSULE}:
            shapes.append((pos, f'<cylinder radius="{_fmt(size[0])}" length="{_fmt(2 * size[1])}"/>'))
            if kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
                sphere = f'<sphere radius="{_fmt(size[0])}"/>'
                shapes.extend((pos + rotation @ np.array([0.0, 0.0, end * size[1]]), sphere) for end in (1.0, -1.0))
        elif kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
            file = self.directory / "meshes" / f"ellipsoid_{geom}.stl"
            file.parent.mkdir(parents=True, exist_ok=True)
            write_binary_stl(file, *ellipsoid_mesh(size))
            self.meshes.append(file)
            shapes.append((pos, f'<mesh filename="package://{self.package}/meshes/{file.name}"/>'))
        material = self._material(geom) if shapes else ""
        return [
            f"    <visual>\n      {_origin(at, rotation)}\n      <geometry>{shape}</geometry>\n"
            f'      <material name="{material}"/>\n    </visual>'
            for at, shape in shapes
        ]

    def _mesh_file(self, mesh: int) -> str:
        if mesh not in self._mesh_files:
            model = self.model
            vertex_start, vertex_count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
            face_start, face_count = int(model.mesh_faceadr[mesh]), int(model.mesh_facenum[mesh])
            stem = re.sub(r"[^A-Za-z0-9_-]", "_", model.mesh(mesh).name or "mesh")
            file = self.directory / "meshes" / f"{mesh}_{stem}.stl"
            file.parent.mkdir(parents=True, exist_ok=True)
            write_binary_stl(
                file,
                model.mesh_vert[vertex_start : vertex_start + vertex_count],
                model.mesh_face[face_start : face_start + face_count],
            )
            self.meshes.append(file)
            self._mesh_files[mesh] = f"package://{self.package}/meshes/{file.name}"
        return self._mesh_files[mesh]


__all__ = [
    "CACHE_DIR_ENV",
    "GENERATOR_VERSION",
    "URDF_FILE",
    "GeneratedUrdf",
    "JointDrive",
    "cache_root",
    "channel_joint",
    "compose_robot",
    "ensure_robot_urdf",
    "entry_key",
    "joint_map",
    "models_available",
    "package_name",
    "public_names",
    "write_robot_urdf",
]
