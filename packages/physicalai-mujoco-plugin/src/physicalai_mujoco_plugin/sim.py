# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Load a simulation: compose a scene with the profile's robots, home them and bind their channels.

Everything here is stateless; :class:`~physicalai_mujoco_plugin.robot.MuJoCoRobot` calls it on
connect and on scene switches, and installs the result only once it fully succeeded.
"""

# MuJoCo model/data attributes are supplied by the C extension at runtime.
# pyrefly: ignore-errors [missing-attribute]

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from physicalai_mujoco_plugin.cameras import CameraConfig
from physicalai_mujoco_plugin.channels import ArmChannels

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    import numpy as np

    from physicalai_mujoco_plugin.channels import TorqueMode
    from physicalai_mujoco_plugin.compose import RobotBinding
    from physicalai_mujoco_plugin.profiles import DefaultUnit, RobotProfile
    from physicalai_mujoco_plugin.scene_registry import ResetFn, SceneConfig


@dataclass
class Sim:
    """A compiled scene with its robots bound: everything a scene switch replaces at once."""

    xml_path: Path
    scene: SceneConfig | None
    model: object
    data: object
    bindings: tuple[RobotBinding, ...]
    channels: tuple[ArmChannels, ...]
    on_reset: ResetFn | None

    @property
    def joint_names(self) -> list[str]:
        """Public channel names of every robot, in anchor order."""
        return [name for channels in self.channels for name in channels.names]


def resolve_scene(profile: RobotProfile, scene_id: str | None, model_path: str | None) -> SceneConfig | None:
    """Resolve the scene to start in: the named one, none for a custom model, else the profile's default.

    Returns:
        The scene, or ``None`` for a custom ``model_path`` without a scene id.

    Raises:
        ValueError: If the scene does not support the profile, or nothing names a scene.
    """
    from physicalai_mujoco_plugin.scene_registry import get_scene  # noqa: PLC0415

    if scene_id is None and model_path is not None:
        return None
    scene_id = scene_id or profile.default_scene
    if scene_id is None:
        msg = f"Profile {profile.name!r} has no default scene; pass scene= or model_path="
        raise ValueError(msg)
    scene = get_scene(scene_id)
    if not scene.supports(profile):
        msg = f"Scene {scene_id!r} does not support the {profile.name!r} profile"
        raise ValueError(msg)
    return scene


def load_sim(
    xml_path: Path,
    scene: SceneConfig | None,
    profile: RobotProfile,
    *,
    unit: DefaultUnit,
    torque_mode: TorqueMode,
    rng: np.random.Generator,
    reseed: Callable[[], None],
    robots: int | None = None,
) -> Sim:
    """Compose a scene, put its robots at their home pose, bind their channels and run the scene reset.

    Args:
        xml_path: Scene XML.
        scene: The registered scene, for its reset; ``None`` for a custom model.
        profile: Robot profile attached at every mount frame.
        unit: Robot-wide public unit.
        torque_mode: How torque actuators are driven.
        rng: Generator for the scene reset.
        reseed: Called before the scene reset, so a fixed seed repeats it.
        robots: Required number of robots; ``None`` accepts any.

    Returns:
        The new simulation.

    Raises:
        ValueError: If the scene attaches a different number of robots, or the profile does not bind.
    """
    import mujoco  # noqa: PLC0415

    from physicalai_mujoco_plugin.compose import compose_scene  # noqa: PLC0415
    from physicalai_mujoco_plugin.scene_registry import get_reset_fn  # noqa: PLC0415

    composed = compose_scene(xml_path, profile)
    if robots is not None and len(composed.robots) != robots:
        msg = f"{xml_path} attaches {len(composed.robots)} robot(s), but this simulation drives {robots}"
        raise ValueError(msg)
    model = composed.model
    data = mujoco.MjData(model)
    for binding in composed.robots:
        for joint, values in binding.layout.home_qpos.items():
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
            if joint_id >= 0:
                address = int(model.jnt_qposadr[joint_id])
                data.qpos[address : address + len(values)] = values
        for actuator, value in binding.layout.home_ctrl.items():
            actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator)
            if actuator_id >= 0:
                data.ctrl[actuator_id] = value
    mujoco.mj_forward(model, data)
    channels = tuple(
        ArmChannels(model, data, binding.layout, profile, prefix=binding.prefix, unit=unit, torque_mode=torque_mode)
        for binding in composed.robots
    )
    on_reset = get_reset_fn(scene.scene_id) if scene is not None else None
    if on_reset is not None:
        reseed()
        on_reset(model, data, rng)
    return Sim(xml_path, scene, model, data, composed.robots, channels, on_reset)


def joint_names_before_load(profile: RobotProfile, xml_path: Path) -> list[str]:
    """Derive the public names from the profile and the scene's anchors, without composing it (DRV-6).

    Returns:
        The names the loaded simulation will have.
    """
    import mujoco  # noqa: PLC0415

    from physicalai_mujoco_plugin.compose import anchor_prefixes, robot_layout  # noqa: PLC0415
    from physicalai_mujoco_plugin.profiles.derive import derive_profile  # noqa: PLC0415

    spec = mujoco.MjSpec.from_file(str(xml_path))
    prefixes = anchor_prefixes(spec)
    if profile.channels:
        names = [channel.name for channel in profile.channels]
    elif prefixes:
        names = list(robot_layout(profile).joint_names)
    else:
        return list(derive_profile(spec.compile()).joint_names)
    return [f"{prefix}{name}" for prefix in prefixes or ("",) for name in names]


def default_cameras(sim: Sim) -> list[CameraConfig]:
    """Stream the first robot's cameras, then ``overview``, then the other robots' (CAM-6).

    Returns:
        One 640x480, 30 fps stream per camera that the model has.
    """
    import mujoco  # noqa: PLC0415

    robot_cameras = [[camera.name for camera in binding.layout.cameras] for binding in sim.bindings]
    names = [*robot_cameras[0][:1], "overview", *robot_cameras[0][1:]]
    names += [name for cameras in robot_cameras[1:] for name in cameras]
    return [
        CameraConfig(name=name)
        for name in dict.fromkeys(names)
        if mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_CAMERA, name) >= 0
    ]


__all__ = ["Sim", "default_cameras", "joint_names_before_load", "load_sim", "resolve_scene"]
