# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Load robot profiles from MuJoCo Menagerie and attach them to robot-free scenes.

Robot models come from MuJoCo Menagerie through the ``mujoco-menagerie`` package, which downloads a
model on first use into a per-user cache (``MENAGERIE_CACHE_DIR`` overrides it). The pinned package
version fixes every model file. ``physicalai-mujoco start`` fetches the robot before it starts the
simulation owner, so the download never runs against the owner's startup timeout. Offline machines
can run ``physicalai-mujoco prefetch`` beforehand, or point ``MENAGERIE_ROOT`` at a checkout.

A scene marks each robot with an anchor frame: ``<frame name="{prefix}robot_mount">`` for a fixed
base (arms), ``<frame name="{prefix}robot_spawn">`` for a floating base (legged robots). Composing
the scene attaches the profile's robot at every anchor with that prefix, so ``left_robot_mount``
yields ``left_shoulder_pan``, ``left_wrist``, and so on. A spawned robot starts at the frame's x, y
and yaw, at the height and orientation of its home keyframe (the frame's height, roll and pitch
are ignored), and a weld holds its base there until the first action (SCN-8). XML files without
anchors (a custom model that defines its own robot) compile unchanged, and their channels are
derived from the whole model.
"""

# MjSpec and the mju_* helpers are supplied by the MuJoCo C extension at runtime.
# pyrefly: ignore-errors [missing-attribute]

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import starmap
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
from loguru import logger

from physicalai_mujoco_plugin.profiles import SO101_PROFILE, RobotProfile
from physicalai_mujoco_plugin.profiles.derive import DerivedLayout, derive_profile
from physicalai_mujoco_plugin.robot_cameras import (
    OVERVIEW_CAMERA,
    add_cameras,
    check_camera_names,
    follow_with_chase_camera,
    model_cameras,
    resolve_cameras,
    robot_bodies,
)

if TYPE_CHECKING:
    import mujoco

ROBOT_MOUNT_FRAME = "robot_mount"
"""Suffix of the scene frames that mark a fixed robot base; the text before it is the robot's prefix."""
ROBOT_SPAWN_FRAME = "robot_spawn"
"""Suffix of the scene frames that mark a floating-base robot's start pose."""
BASE_HOLD_EQUALITY = "mujoco_base_hold_"
"""Name of the weld that holds a spawned robot's base, followed by the robot's prefix (SCN-8)."""

AnchorKind = Literal["mount", "spawn"]
"""``mount``: a fixed base (``robot_mount`` frames). ``spawn``: a floating base (``robot_spawn`` frames)."""


@dataclass(frozen=True)
class RobotBinding:
    """One attached robot: its prefix and its layout, with ids and names of the composed model."""

    prefix: str
    layout: DerivedLayout
    base_hold: int | None = None
    """Id of the weld equality that holds the floating base until the first action, if any."""


@dataclass(frozen=True)
class OverviewRig:
    """A pose for the top-level body that carries a scene's ``overview`` camera, in world coordinates."""

    pos: tuple[float, float, float]
    quat: tuple[float, float, float, float] | None = None
    """Orientation (``wxyz``); ``None`` keeps the scene's."""


@dataclass(frozen=True)
class MountPose:
    """Where a scene puts one robot mount frame, in world coordinates."""

    pos: tuple[float, float, float]
    quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    """Orientation (``wxyz``): the arm faces the frame's +x axis."""


@dataclass(frozen=True)
class SceneLayout:
    """How a scene is laid out for one robot (SCN-6): scaled with its reach, its overview rig placed."""

    scale: float = 1.0
    """Factor for the positions of the scene's top-level bodies, about the robot mount at the origin."""
    overview_rig: OverviewRig | None = None
    """Where the overview camera's rig goes after scaling; ``None`` leaves it scaled."""
    mounts: tuple[tuple[str, MountPose], ...] | None = None
    """The robot mounts by prefix, replacing the scene's ``robot_mount`` frames after scaling, for an arm
    count the scene file does not lay out; ``None`` keeps the scene's frames."""


@dataclass(frozen=True)
class ComposedScene:
    """A compiled scene and the robots attached to it, in anchor order."""

    model: mujoco.MjModel
    robots: tuple[RobotBinding, ...]


def fetch_profile(profile: RobotProfile) -> Path:
    """Put the profile's robot entry in the Menagerie cache, downloading it if needed.

    The download shows a progress bar when stderr is a terminal.

    Returns:
        The robot entry's XML file.

    Raises:
        RuntimeError: If the model is not cached and cannot be obtained.
    """
    import mujoco_menagerie  # noqa: PLC0415

    try:
        robot = mujoco_menagerie.get(profile.menagerie_model)
        entry = robot.entry(profile.menagerie_entry or robot.default_model)
    except mujoco_menagerie.MenagerieError as exc:
        raise RuntimeError(str(exc)) from exc
    cache = mujoco_menagerie.Cache()
    if cache.root is None and not cache.is_cached(robot):
        logger.info(
            "Downloading the {} model from MuJoCo Menagerie ({:.1f} MB) into {}",
            profile.name,
            (robot.download_size or 0) / 2**20,
            cache.dir,
        )
    try:
        return robot.path(cache) / entry.file
    except mujoco_menagerie.DownloadError as exc:
        msg = (
            f"Could not download the MuJoCo Menagerie model {profile.menagerie_model!r}: {exc}. "
            f"Connect to the internet and run `physicalai-mujoco prefetch`, "
            "or set MENAGERIE_ROOT to a mujoco_menagerie checkout."
        )
        raise RuntimeError(msg) from exc
    except mujoco_menagerie.MenagerieError as exc:
        raise RuntimeError(str(exc)) from exc


def load_robot_spec(profile: RobotProfile) -> mujoco.MjSpec:
    """Load the profile's robot with its overrides applied and its cameras added (CAM-7).

    An override that names an element the robot does not have raises ``ValueError``.

    Returns:
        A new robot spec, ready to attach.
    """
    spec = _customized_robot_spec(profile)
    add_cameras(spec, robot_layout(profile).cameras, profile.name)
    return spec


def _customized_robot_spec(profile: RobotProfile) -> mujoco.MjSpec:
    """Load the profile's robot and apply its renames, ranges, force limits and customization.

    Returns:
        A new robot spec without the profile's or generated cameras.

    Raises:
        ValueError: If an override names an element the robot does not have.
    """
    import mujoco  # noqa: PLC0415

    path = fetch_profile(profile)
    spec = mujoco.MjSpec.from_file(str(path))
    for old, new in profile.renames:
        spec = _rename(spec, old, new, path.parent)
    for channel in profile.channels:
        for name in channel.actuators:
            actuator = spec.actuator(name)
            if actuator is None:
                msg = f"Profile {profile.name!r} channel {channel.name!r} names missing actuator {name!r}"
                raise ValueError(msg)
            if channel.range is not None:
                joint = spec.joint(actuator.target)
                if joint is None:
                    msg = f"Profile {profile.name!r} channel {channel.name!r} has a range but no joint"
                    raise ValueError(msg)
                joint.range = list(channel.range)
            if profile.actuator_forcerange is not None:
                actuator.forcerange = list(profile.actuator_forcerange)
    if profile.customize is not None:
        profile.customize(spec)
    return spec


_LAYOUTS: dict[tuple[str, str, str | None, str], tuple[RobotProfile, DerivedLayout]] = {}


def robot_layout(profile: RobotProfile) -> DerivedLayout:
    """Return the derived layout of the profile's robot alone (PRF-4), cached per Menagerie version (PRF-5).

    Returns:
        The robot-only layout, with unprefixed names. Its cameras are the ones the robot
        publishes (CAM-2), not every camera of the model.
    """
    from importlib.metadata import version  # noqa: PLC0415

    key = (profile.name, profile.menagerie_model, profile.menagerie_entry, version("mujoco-menagerie"))
    cached = _LAYOUTS.get(key)
    if cached is not None and cached[0] == profile:
        return cached[1]
    model = _customized_robot_spec(profile).compile()
    layout = derive_profile(model)
    layout = replace(layout, cameras=resolve_cameras(profile, model, layout))
    _LAYOUTS[key] = (profile, layout)
    return layout


def scene_anchors(spec: mujoco.MjSpec) -> tuple[tuple[str, AnchorKind], ...]:
    """Return the robot prefix and anchor kind of each ``robot_mount`` and ``robot_spawn`` frame, in document order.

    Raises:
        ValueError: If one prefix has both a mount and a spawn frame (SCN-1).
    """
    anchors: dict[str, AnchorKind] = {}
    for frame in spec.frames:
        for suffix, kind in ((ROBOT_MOUNT_FRAME, "mount"), (ROBOT_SPAWN_FRAME, "spawn")):
            if not frame.name.endswith(suffix):
                continue
            prefix = frame.name.removesuffix(suffix)
            if anchors.get(prefix, kind) != kind:
                msg = f"The scene has both {prefix}{ROBOT_MOUNT_FRAME} and {prefix}{ROBOT_SPAWN_FRAME}; use one"
                raise ValueError(msg)
            anchors[prefix] = kind
    return tuple(anchors.items())


def anchor_prefixes(spec: mujoco.MjSpec) -> tuple[str, ...]:
    """Return the robot prefixes of the scene's anchor frames, in document order."""
    return tuple(prefix for prefix, _ in scene_anchors(spec))


def mount_prefixes(xml_path: str | Path) -> tuple[str, ...]:
    """Return the robot prefixes of a scene XML's ``robot_mount`` frames, in document order, without compiling it.

    Returns:
        One prefix per mount frame; spawn frames are left out.
    """
    import mujoco  # noqa: PLC0415

    return tuple(prefix for prefix, kind in scene_anchors(mujoco.MjSpec.from_file(str(xml_path))) if kind == "mount")


def scene_needs_robot(xml_path: str | Path) -> bool:
    """Return whether composing the scene XML attaches a robot, so its model must be available.

    Returns:
        ``True`` if the XML has at least one anchor frame.
    """
    import mujoco  # noqa: PLC0415

    return bool(anchor_prefixes(mujoco.MjSpec.from_file(str(xml_path))))


def _option_fields(option: object) -> tuple[str, ...]:
    """Return the settable fields of a spec's ``<option>``, i.e. the binding's writable properties."""
    return tuple(
        name for name, attr in vars(type(option)).items() if isinstance(attr, property) and attr.fset is not None
    )


def attach_robot(scene: mujoco.MjSpec, profile: RobotProfile, prefix: str, kind: AnchorKind = "mount") -> None:
    """Attach a fresh copy of the profile's robot at the scene frame ``{prefix}robot_mount`` or ``robot_spawn``.

    The scene is the single source of physics options (``<option>``): MuJoCo keeps the parent's
    values on attach and warns about every field the robot MJCF sets differently. The robot's
    options are replaced by the scene's before attaching; the overridden values are logged at
    debug level instead. The robot's keyframes are dropped: the layout has read them already.
    The robot's cameras are in its spec before attaching, so they carry the prefix.

    At a spawn frame, the robot takes the frame's world x, y and yaw, and the home keyframe's height
    and orientation (SCN-1, SCN-10); the frame's own height, roll and pitch are ignored. The root
    body moves to that pose before compiling, so ``qpos0`` is the start pose and the base-hold weld
    added here takes it as its reference (SCN-8). The hold always lasts until the first action;
    there is no permanent or disabled hold.

    Raises:
        ValueError: If the anchor does not fit the robot's base: a floating base at a mount frame,
            which would not hold it, or a fixed base at a spawn frame.
    """
    import mujoco  # noqa: PLC0415

    base = robot_layout(profile).base
    if kind == "mount" and base is not None:
        joint = base.joint or base.body
        msg = (
            f"Profile {profile.name!r} has a floating base ({joint!r}), which a robot_mount frame cannot hold: "
            "floating bases need a robot_spawn frame. A robot-complete model XML without anchor frames keeps "
            "its free joint"
        )
        raise ValueError(msg)
    if kind == "spawn" and base is None:
        msg = f"The {profile.name!r} robot has a fixed base; {prefix}{ROBOT_SPAWN_FRAME} needs a floating-base robot"
        raise ValueError(msg)
    robot = load_robot_spec(profile)
    overridden = []
    for field in _option_fields(scene.option):
        scene_value = getattr(scene.option, field)
        if not np.array_equal(np.asarray(getattr(robot.option, field)), np.asarray(scene_value)):
            overridden.append(f"{field}={getattr(robot.option, field)}")
        setattr(robot.option, field, scene_value)
    if overridden:
        logger.debug("{} options overridden by the scene: {}", profile.name, ", ".join(overridden))
    for key in list(robot.keys):
        robot.delete(key)
    if kind == "mount":
        scene.attach(robot, frame=scene.frame(f"{prefix}{ROBOT_MOUNT_FRAME}"), prefix=prefix)
        return
    root = robot.body(base.body)  # type: ignore[union-attr]
    root.pos = [0.0, 0.0, base.home[2]]
    root.quat = list(base.home[3:])
    x, y, yaw = _spawn_frame_xy_yaw(scene, f"{prefix}{ROBOT_SPAWN_FRAME}")
    # Attach at a level copy of the spawn frame: its height, roll and pitch would lift or tilt the robot.
    level = scene.worldbody.add_frame(pos=[x, y, 0.0], quat=[np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
    scene.attach(robot, frame=level, prefix=prefix)
    hold = scene.add_equality()
    hold.name = f"{BASE_HOLD_EQUALITY}{prefix}"
    hold.type = mujoco.mjtEq.mjEQ_WELD
    hold.objtype = mujoco.mjtObj.mjOBJ_BODY
    hold.name1 = f"{prefix}{base.body}"
    # Anchor at the body origin; the all-zero relative pose makes the compiler take it from qpos0.
    hold.data = [0.0] * 10 + [1.0]


def _spawn_frame_xy_yaw(scene: mujoco.MjSpec, name: str) -> tuple[float, float, float]:
    """Return the world x, y and yaw (radians) of a scene frame, wherever it is nested (SCN-1).

    MjSpec resolves frame poses only when compiling, so a probe body is compiled in a copy of the scene.

    Returns:
        The frame's world position x and y, and the heading of its x axis about the world z axis.
    """
    import mujoco  # noqa: PLC0415

    probe = scene.copy()
    probe.frame(name).add_body(name="mujoco_spawn_probe")
    model = probe.compile()
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    body = model.body("mujoco_spawn_probe").id
    rotation = data.xmat[body].reshape(3, 3)
    x, y = (float(v) for v in data.xpos[body][:2])
    return x, y, float(np.arctan2(rotation[1, 0], rotation[0, 0]))


def compose_scene_spec(
    xml_path: str | Path, profile: RobotProfile = SO101_PROFILE, *, scene_layout: SceneLayout | None = None
) -> mujoco.MjSpec:
    """Load a scene XML and attach the profile's robot at each of its anchor frames.

    Returns:
        The scene spec with its robots and cameras in place, as :func:`compose_scene` compiles it.
    """
    return _compose(xml_path, profile, scene_layout).spec


def load_scene_model(
    xml_path: str | Path, profile: RobotProfile = SO101_PROFILE, *, scene_layout: SceneLayout | None = None
) -> mujoco.MjModel:
    """Compile a scene XML with its robots attached.

    Returns:
        The compiled model, identical to :func:`compose_scene`'s.
    """
    return _compose(xml_path, profile, scene_layout).model


def compose_scene(
    xml_path: str | Path, profile: RobotProfile, *, scene_layout: SceneLayout | None = None
) -> ComposedScene:
    """Compile a scene with the profile's robot at each anchor frame, and bind each robot's layout.

    A model without anchor frames defines its own robot: its layout is derived from the whole model,
    and its cameras are resolved like a profile without overrides (CAM-2). In both, a world camera
    named ``chase`` follows the first robot's floating base. Two cameras with one name raise
    ``ValueError`` (CAM-5).

    Args:
        xml_path: Scene XML.
        profile: Robot profile attached at every anchor frame.
        scene_layout: The scene's layout for this robot (:func:`apply_layout`); ``None`` compiles the
            scene as written.

    Returns:
        The compiled model and one binding per robot, in anchor order.
    """
    import mujoco  # noqa: PLC0415

    composed = _compose(xml_path, profile, scene_layout)
    model = composed.model
    if not composed.anchored:
        return ComposedScene(model, tuple(starmap(RobotBinding, composed.layouts)))
    robots = []
    for prefix, layout in composed.layouts:
        hold = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, f"{BASE_HOLD_EQUALITY}{prefix}"))
        robots.append(RobotBinding(prefix, bind_layout(layout, model, prefix), hold if hold >= 0 else None))
    return ComposedScene(model, tuple(robots))


@dataclass(frozen=True)
class _Composed:
    spec: mujoco.MjSpec
    model: mujoco.MjModel
    anchored: bool
    layouts: tuple[tuple[str, DerivedLayout], ...]
    """Per robot prefix: the robot-only layout in anchored scenes; in robot-complete models the whole
    model's, with that robot's cameras."""


def _compose(xml_path: str | Path, profile: RobotProfile, scene_layout: SceneLayout | None = None) -> _Composed:
    """Build and compile a scene: every compile path (spec, model, bound scene) goes through here.

    Returns:
        The final spec, its compiled model, and each robot's prefix and layout.
    """
    import mujoco  # noqa: PLC0415

    scene = mujoco.MjSpec.from_file(str(xml_path))
    if scene_layout is not None:
        apply_layout(scene, scene_layout)
    anchors = scene_anchors(scene)
    if anchors:
        for prefix, kind in anchors:
            attach_robot(scene, profile, prefix, kind)
        check_camera_names(scene, xml_path)
        model = scene.compile()
        layout = robot_layout(profile)
        layouts = tuple((prefix, layout) for prefix, _ in anchors)
        root = None if layout.base is None else f"{anchors[0][0]}{layout.base.body}"
    else:
        model = scene.compile()
        layout = derive_profile(model)
        # Each robot of the model (one per name prefix) gets its own cameras: left_wrist, right_wrist.
        # Which robot owns a sensor camera follows the kinematic tree, not body names.
        robots = model_prefixes(layout, profile)
        owned = robot_bodies(model, layout, profile, robots)
        cameras = {prefix: model_cameras(model, layout, profile, prefix, owned[prefix]) for prefix in robots}
        generated = tuple(camera for robot in cameras.values() for camera in robot if camera.source == "default")
        if generated:
            add_cameras(scene, generated, profile.name)
            model = scene.compile()
        layouts = tuple((prefix, replace(layout, cameras=robot)) for prefix, robot in cameras.items())
        root = None if layout.base is None else layout.base.body
    if root is not None and follow_with_chase_camera(scene, model, root):
        model = scene.compile()
    return _Composed(scene, model, bool(anchors), layouts)


def apply_layout(scene: mujoco.MjSpec, layout: SceneLayout) -> None:
    """Lay a scene out for one robot: scale it with the robot's reach, then place its overview rig.

    Scaling moves the scene's top-level bodies away from the robot mount (SCN-6, CAM-4). Only body
    positions change: a target keeps its size, and the overview camera's rig moves back along its
    line of sight, so it frames the scaled layout as it framed the original. Then the layout's mounts,
    if any, replace the scene's robot mount frames.

    Raises:
        ValueError: If the layout scales the scene and a robot mount frame is not at the origin,
            which the scaling is about, it places mounts and a mount frame is not a top-level frame,
            or it places an overview rig and the scene has no ``overview`` camera.
    """
    import mujoco  # noqa: PLC0415

    if layout.scale != 1.0:  # noqa: RUF069 - exactly 1.0 means as written (SO-101 parity)
        for frame in scene.frames:
            if frame.name.endswith(ROBOT_MOUNT_FRAME) and np.any(np.asarray(frame.pos)):
                msg = (
                    f"Cannot scale the layout about {frame.name!r}: a reach layout needs its robot mount at the origin"
                )
                raise ValueError(msg)
        for body in scene.worldbody.bodies:
            body.pos = np.asarray(body.pos) * layout.scale
    if layout.mounts is not None:
        _place_mounts(scene, layout.mounts)
    rig = layout.overview_rig
    if rig is None:
        return
    camera = scene.camera(OVERVIEW_CAMERA)
    if camera is None:
        msg = f"The scene has no {OVERVIEW_CAMERA!r} camera to place"
        raise ValueError(msg)
    body = camera.parent
    while body.parent.name != "world":
        body = body.parent
    body.pos = list(rig.pos)
    if rig.quat is not None:
        body.quat = list(rig.quat)
        body.alt.type = mujoco.mjtOrientation.mjORIENTATION_QUAT


def _place_mounts(scene: mujoco.MjSpec, mounts: tuple[tuple[str, MountPose], ...]) -> None:
    """Replace the scene's ``robot_mount`` frames with one top-level frame per mount, in order.

    The scene's frames are renamed and moved, and more are added as needed. MjSpec cannot delete a
    frame (MuJoCo 3.14), so a frame left over loses its name and anchors nothing.

    Raises:
        ValueError: If a mount frame is nested in a body: mount poses are world poses.
    """
    import mujoco  # noqa: PLC0415

    frames = [frame for frame in scene.frames if frame.name.endswith(ROBOT_MOUNT_FRAME)]
    for frame in frames:
        if frame.parent.name != "world":
            msg = f"Cannot place {frame.name!r}: a scene laid out for another arm count needs top-level mount frames"
            raise ValueError(msg)
    for index, (prefix, pose) in enumerate(mounts):
        frame = frames[index] if index < len(frames) else scene.worldbody.add_frame()
        frame.name = f"{prefix}{ROBOT_MOUNT_FRAME}"
        frame.pos = list(pose.pos)
        frame.quat = list(pose.quat)
        frame.alt.type = mujoco.mjtOrientation.mjORIENTATION_QUAT
    for frame in frames[len(mounts) :]:
        frame.name = ""


def model_prefixes(layout: DerivedLayout, profile: RobotProfile) -> tuple[str, ...]:
    """Find the robots of a model that defines its own: the prefixes of the profile's first actuator.

    A bimanual model names its arms ``left_shoulder_pan`` and ``right_shoulder_pan``; each prefix
    becomes one robot, in model order. Profiles without channel overrides take the whole model.

    Returns:
        The prefixes, or ``("",)`` when nothing matches (binding then names the missing actuator).
    """
    if not profile.channels:
        return ("",)
    first = profile.channels[0].actuators[0]
    prefixes = []
    for channel in layout.channels:
        # One candidate per actuator, the joint's name first: a model may name the joint
        # ``shoulder_pan`` and its actuator ``motor_shoulder_pan``, which is still one robot.
        name = next((n for n in (channel.joint, channel.actuator) if n and n.endswith(first)), None)
        if name is not None:
            prefixes.append(name.removesuffix(first))
    return tuple(dict.fromkeys(prefixes)) or ("",)


def bind_layout(layout: DerivedLayout, model: mujoco.MjModel, prefix: str) -> DerivedLayout:
    """Re-resolve a robot-only layout in a composed model, where every name carries *prefix*.

    Returns:
        The layout with prefixed names and the composed model's ids and addresses.
    """
    import mujoco  # noqa: PLC0415

    def lookup(kind: int, name: str) -> int:
        element_id = int(mujoco.mj_name2id(model, kind, f"{prefix}{name}"))
        if element_id < 0:
            msg = f"{prefix}{name!r} is missing from the composed model"
            raise ValueError(msg)
        return element_id

    channels = []
    for channel in layout.channels:
        joint_id = lookup(mujoco.mjtObj.mjOBJ_JOINT, channel.joint) if channel.joint is not None else None
        channels.append(
            replace(
                channel,
                name=f"{prefix}{channel.name}",
                actuator=f"{prefix}{channel.actuator}",
                actuator_id=lookup(mujoco.mjtObj.mjOBJ_ACTUATOR, channel.actuator),
                joint=f"{prefix}{channel.joint}" if channel.joint is not None else None,
                joint_id=joint_id,
                qpos_adr=int(model.jnt_qposadr[joint_id]) if joint_id is not None else None,
                dof_adr=int(model.jnt_dofadr[joint_id]) if joint_id is not None else None,
            ),
        )
    sensors = tuple(
        replace(
            sensor,
            name=f"{prefix}{sensor.name}",
            address=int(model.sensor_adr[lookup(mujoco.mjtObj.mjOBJ_SENSOR, sensor.name)]),
        )
        for sensor in layout.sensors
    )
    cameras = tuple(
        replace(camera, name=f"{prefix}{camera.name}", body=f"{prefix}{camera.body}") for camera in layout.cameras
    )
    base = layout.base
    if base is not None:
        body_id = lookup(mujoco.mjtObj.mjOBJ_BODY, base.body)
        joint_id = int(model.body_jntadr[body_id])
        address = int(model.jnt_qposadr[joint_id])
        # The composed qpos0 is where the anchor put the base: the spawn pose, not the keyframe's.
        base = replace(
            base,
            body=f"{prefix}{base.body}",
            body_id=body_id,
            joint=model.joint(joint_id).name,
            qpos_adr=address,
            dof_adr=int(model.jnt_dofadr[joint_id]),
            home=tuple(float(v) for v in model.qpos0[address : address + 7]),
        )
    return replace(
        layout,
        channels=tuple(channels),
        base=base,
        home_qpos={f"{prefix}{name}": value for name, value in layout.home_qpos.items()},
        home_ctrl={f"{prefix}{name}": value for name, value in layout.home_ctrl.items()},
        sensors=sensors,
        cameras=cameras,
    )


def _rename(spec: mujoco.MjSpec, old: str, new: str, asset_root: Path) -> mujoco.MjSpec:
    """Rename the body, joint, actuator, camera or site called *old*, and every reference to it (PRF-6).

    MjSpec neither updates references on rename nor exposes tendon path targets for writing, so the
    spec is round-tripped through its XML, where every reference is a plain attribute.

    Returns:
        A new spec with the element and its references renamed.

    Raises:
        ValueError: If the spec has no renameable element called *old*.
    """
    import mujoco  # noqa: PLC0415
    from defusedxml import ElementTree  # noqa: PLC0415

    root = ElementTree.fromstring(spec.to_xml())
    named_tags = {"body", "joint", "freejoint", "camera", "site"}
    actuator_tags = {"motor", "position", "velocity", "general", "intvelocity", "damper", "adhesion", "muscle"}
    references = {
        "actuator", "body", "body1", "body2", "bodyname1", "bodyname2", "camera", "joint", "joint1", "joint2",
        "name1", "name2", "objname", "refname", "site", "site1", "site2", "sidesite", "target", "tendon",
    }  # fmt: skip
    parents = {child: parent for parent in root.iter() for child in parent}
    renamed = False
    for element in root.iter():
        tag = element.tag.rsplit("}", 1)[-1]
        parent = parents.get(element)
        in_actuator = parent is not None and parent.tag.rsplit("}", 1)[-1] == "actuator"
        if element.get("name") == old and (tag in named_tags or (in_actuator and tag in actuator_tags)):
            element.set("name", new)
            renamed = True
        for attribute in references:
            if element.get(attribute) == old:
                element.set(attribute, new)
    if not renamed:
        msg = f"Cannot rename {old!r}: the robot has no body, joint, actuator, camera or site with that name"
        raise ValueError(msg)
    compiler = root.find("compiler")
    if compiler is not None:
        for attribute in ("meshdir", "texturedir", "assetdir"):
            directory = compiler.get(attribute)
            if directory and not Path(directory).is_absolute():
                compiler.set(attribute, str((asset_root / directory).resolve()))
    return mujoco.MjSpec.from_string(ElementTree.tostring(root, encoding="unicode"))


__all__ = [
    "BASE_HOLD_EQUALITY",
    "ROBOT_MOUNT_FRAME",
    "ROBOT_SPAWN_FRAME",
    "AnchorKind",
    "ComposedScene",
    "MountPose",
    "OverviewRig",
    "RobotBinding",
    "SceneLayout",
    "anchor_prefixes",
    "apply_layout",
    "attach_robot",
    "bind_layout",
    "compose_scene",
    "compose_scene_spec",
    "fetch_profile",
    "load_robot_spec",
    "load_scene_model",
    "model_prefixes",
    "mount_prefixes",
    "robot_layout",
    "scene_anchors",
    "scene_needs_robot",
]
