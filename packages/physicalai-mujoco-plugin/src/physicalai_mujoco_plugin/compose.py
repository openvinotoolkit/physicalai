# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Load robot profiles from MuJoCo Menagerie and attach them to robot-free scenes.

Robot models come from MuJoCo Menagerie through the ``mujoco-menagerie`` package, which downloads a
model on first use into a per-user cache (``MENAGERIE_CACHE_DIR`` overrides it). The pinned package
version fixes every model file. ``physicalai-mujoco start`` fetches the robot before it starts the
simulation owner, so the download never runs against the owner's startup timeout. Offline machines
can run ``physicalai-mujoco prefetch`` beforehand, or point ``MENAGERIE_ROOT`` at a checkout.

A scene marks each robot's base with a ``<frame name="{prefix}robot_mount">``. Composing the scene
attaches the profile's robot at every mount frame with that prefix, so ``left_robot_mount`` yields
``left_shoulder_pan``, ``left_wrist``, and so on. XML files without mount frames (a custom model
that defines its own robot) compile unchanged, and their channels are derived from the whole model.
"""

# MjSpec and the mju_* helpers are supplied by the MuJoCo C extension at runtime.
# pyrefly: ignore-errors [missing-attribute]

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

from physicalai_mujoco_plugin.profiles import SO101_PROFILE, RobotProfile
from physicalai_mujoco_plugin.profiles.derive import DerivedLayout, derive_profile

if TYPE_CHECKING:
    import mujoco

ROBOT_MOUNT_FRAME = "robot_mount"
"""Suffix of the scene frames that mark a fixed robot base; the text before it is the robot's prefix."""
ROBOT_SPAWN_FRAME = "robot_spawn"
"""Suffix of the scene frames that mark a floating robot's start pose (floating bases, not yet supported)."""


@dataclass(frozen=True)
class RobotBinding:
    """One attached robot: its prefix and its layout, with ids and names of the composed model."""

    prefix: str
    layout: DerivedLayout


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
    """Load the profile's robot and apply its renames, ranges, force limits and customization.

    Returns:
        A new robot spec, ready to attach.

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
        The robot-only layout, with unprefixed names.
    """
    from importlib.metadata import version  # noqa: PLC0415

    key = (profile.name, profile.menagerie_model, profile.menagerie_entry, version("mujoco-menagerie"))
    cached = _LAYOUTS.get(key)
    if cached is not None and cached[0] == profile:
        return cached[1]
    layout = derive_profile(load_robot_spec(profile).compile())
    _LAYOUTS[key] = (profile, layout)
    return layout


def anchor_prefixes(spec: mujoco.MjSpec) -> tuple[str, ...]:
    """Return the robot prefixes of the scene's ``robot_mount`` frames, in document order.

    Raises:
        ValueError: If the scene has ``robot_spawn`` frames, which need floating-base support.
    """
    if any(frame.name.endswith(ROBOT_SPAWN_FRAME) for frame in spec.frames):
        msg = "Scenes with robot_spawn frames (floating bases) are not supported yet"
        raise ValueError(msg)
    return tuple(
        frame.name.removesuffix(ROBOT_MOUNT_FRAME) for frame in spec.frames if frame.name.endswith(ROBOT_MOUNT_FRAME)
    )


def scene_needs_robot(xml_path: str | Path) -> bool:
    """Return whether composing the scene XML attaches a robot, so its model must be available.

    Returns:
        ``True`` if the XML has at least one mount frame.
    """
    import mujoco  # noqa: PLC0415

    return bool(anchor_prefixes(mujoco.MjSpec.from_file(str(xml_path))))


def _option_fields(option: object) -> tuple[str, ...]:
    """Return the settable fields of a spec's ``<option>``, i.e. the binding's writable properties."""
    return tuple(
        name for name, attr in vars(type(option)).items() if isinstance(attr, property) and attr.fset is not None
    )


def attach_robot(scene: mujoco.MjSpec, profile: RobotProfile, prefix: str) -> None:
    """Attach a fresh copy of the profile's robot at the scene frame ``{prefix}robot_mount``.

    The scene is the single source of physics options (``<option>``): MuJoCo keeps the parent's
    values on attach and warns about every field the robot MJCF sets differently. The robot's
    options are replaced by the scene's before attaching; the overridden values are logged at
    debug level instead. The robot's keyframes are dropped: the layout has read them already.
    """
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
    scene.attach(robot, frame=scene.frame(f"{prefix}{ROBOT_MOUNT_FRAME}"), prefix=prefix)


def compose_scene_spec(xml_path: str | Path, profile: RobotProfile = SO101_PROFILE) -> mujoco.MjSpec:
    """Load a scene XML and attach the profile's robot at each of its mount frames.

    Returns:
        The scene spec with its robots attached.
    """
    import mujoco  # noqa: PLC0415

    scene = mujoco.MjSpec.from_file(str(xml_path))
    for prefix in anchor_prefixes(scene):
        attach_robot(scene, profile, prefix)
    return scene


def load_scene_model(xml_path: str | Path, profile: RobotProfile = SO101_PROFILE) -> mujoco.MjModel:
    """Compile a scene XML with its robots attached.

    Returns:
        The compiled model.
    """
    return compose_scene_spec(xml_path, profile).compile()


def compose_scene(xml_path: str | Path, profile: RobotProfile) -> ComposedScene:
    """Compile a scene with the profile's robot at each mount frame, and bind each robot's layout.

    A model without mount frames is used as is; its layout is derived from the whole model.

    Returns:
        The compiled model and one binding per robot, in anchor order.
    """
    import mujoco  # noqa: PLC0415

    scene = mujoco.MjSpec.from_file(str(xml_path))
    prefixes = anchor_prefixes(scene)
    if not prefixes:
        model = scene.compile()
        layout = derive_profile(model)
        return ComposedScene(model, tuple(RobotBinding(prefix, layout) for prefix in model_prefixes(layout, profile)))
    for prefix in prefixes:
        attach_robot(scene, profile, prefix)
    model = scene.compile()
    layout = robot_layout(profile)
    return ComposedScene(model, tuple(RobotBinding(prefix, bind_layout(layout, model, prefix)) for prefix in prefixes))


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
    return replace(
        layout,
        channels=tuple(channels),
        floating_base_joint=None if layout.floating_base_joint is None else f"{prefix}{layout.floating_base_joint}",
        base_body=None if layout.base_body is None else f"{prefix}{layout.base_body}",
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
    "ROBOT_MOUNT_FRAME",
    "ComposedScene",
    "RobotBinding",
    "anchor_prefixes",
    "attach_robot",
    "bind_layout",
    "compose_scene",
    "compose_scene_spec",
    "fetch_profile",
    "load_robot_spec",
    "load_scene_model",
    "model_prefixes",
    "robot_layout",
    "scene_needs_robot",
]
