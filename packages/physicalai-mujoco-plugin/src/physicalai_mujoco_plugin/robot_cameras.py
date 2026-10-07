# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Robot and scene cameras: which cameras a robot publishes, where generated ones go, and the chase camera.

A robot's cameras come from the first source that defines them (CAM-2):

1. the profile's ``cameras`` override;
2. the model's own sensor cameras: cameras on a robot body whose name is not a viewer camera's
   (``track*``, ``perspective``, ``side``, ...);
3. one generated default (CAM-3), placed from the model's geometry at the home pose:

   - ``wrist`` (fixed base): on the gripper's parent body, or on the last arm link without a
     gripper. The tool axis runs from that body to its tip site (``pinch``, ``tcp``,
     ``attachment``, ...), else along the body's +z to its furthest geometry. The camera sits
     4 cm behind the tip, 2 cm clear of that body's and its parent link's geometry on their
     narrowest side, and looks 10 cm past the tip with the tool at the bottom of the image.
     ``fovy`` 75 degrees.
   - ``head`` (floating base with a ``head`` or ``neck`` body): 8 cm ahead of that body along the
     root's +x (1 cm ahead of a larger head's geometry), looking along +x, pitched down
     20 degrees. ``fovy`` 80 degrees.
   - ``front`` (other floating bases): at the root body's +x geometry extent, looking along +x,
     pitched down 15 degrees. ``fovy`` 90 degrees.

Profile and generated cameras are added to the robot spec before it is attached, so they carry
the robot's prefix (CAM-7). A model XML without mount frames defines its own robot: its cameras
resolve from sources 2 and 3, and its world cameras stay scene cameras. Scenes bring their own
``overview`` camera; a world camera named ``chase`` follows the first robot's floating base (CAM-4).
"""

# MuJoCo model/data attributes are supplied by the C extension at runtime.
# pyrefly: ignore-errors [missing-attribute]

from __future__ import annotations

from dataclasses import replace
from fnmatch import fnmatch
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

from physicalai_mujoco_plugin.profiles import CameraSpec

if TYPE_CHECKING:
    import mujoco

    from physicalai_mujoco_plugin.profiles import RobotProfile
    from physicalai_mujoco_plugin.profiles.derive import DerivedChannel, DerivedLayout

OVERVIEW_CAMERA = "overview"
"""The scene camera every registered scene has."""
CHASE_CAMERA = "chase"
"""A scene's world camera that follows the first robot's floating base once the robot is attached."""

VIEWER_CAMERA_PATTERNS = ("track*", "tracking*", "perspective", "side", "top", "front", "hero", "back", "bottom")
"""Names of Menagerie's viewer cameras, which are not robot sensors."""
_GRIPPER_WORDS = ("gripper", "finger", "jaw")
_TIP_SITE_WORDS = ("pinch", "tcp", "tip", "grasp", "gripper", "end_effector", "attachment")
_HEAD_WORDS = ("head", "neck")

_WRIST_BACK_M = 0.04
_WRIST_AIM_AHEAD_M = 0.10
_WRIST_CLEARANCE_M = 0.02
_WRIST_FOVY = 75.0
_HEAD_AHEAD_M = 0.08
_HEAD_PITCH_DEG = 20.0
_HEAD_FOVY = 80.0
_FRONT_CLEARANCE_M = 0.01
_FRONT_PITCH_DEG = 15.0
_FRONT_FOVY = 90.0
_CAMERA_INTRINSICS = (
    "fovy", "ipd", "proj", "resolution", "sensor_size", "focal_length", "focal_pixel", "principal_length",
    "principal_pixel", "output",
)  # fmt: skip


def is_sensor_camera(camera: CameraSpec) -> bool:
    """Return whether a model camera is a robot sensor rather than a viewer camera (CAM-2).

    Returns:
        ``True`` for cameras on a robot body whose name is not one of :data:`VIEWER_CAMERA_PATTERNS`.
    """
    name = camera.name.lower()
    return camera.body != "world" and not any(fnmatch(name, pattern) for pattern in VIEWER_CAMERA_PATTERNS)


def resolve_cameras(profile: RobotProfile, model: mujoco.MjModel, layout: DerivedLayout) -> tuple[CameraSpec, ...]:
    """Return the robot's cameras from the first source that has any (CAM-2).

    Args:
        profile: The robot profile; its ``cameras`` override wins.
        model: The compiled robot-only model, before any camera was added.
        layout: The model's derived layout, with every model camera.

    Returns:
        The cameras, unprefixed. ``source`` tells where each came from.
    """
    if profile.cameras is not None:
        return profile.cameras
    return model_cameras(model, layout, profile)


def model_cameras(
    model: mujoco.MjModel,
    layout: DerivedLayout,
    profile: RobotProfile,
    prefix: str = "",
    bodies: frozenset[str] | None = None,
) -> tuple[CameraSpec, ...]:
    """Return a robot's sensor cameras, else its generated default: CAM-2 without a profile override.

    A model that defines its own robots (no mount frames) resolves each robot's cameras this way:
    the robot with *prefix* owns the sensor cameras on its *bodies* (see `robot_bodies`; ``None``
    means every body), and world cameras stay scene cameras.

    Returns:
        The robot's sensor cameras (``source="model"``), or one generated camera (``source="default"``).
    """
    sensors = tuple(
        camera for camera in layout.cameras if is_sensor_camera(camera) and (bodies is None or camera.body in bodies)
    )
    return sensors or generated_cameras(model, layout, profile, prefix)


def robot_bodies(
    model: mujoco.MjModel,
    layout: DerivedLayout,
    profile: RobotProfile,
    prefixes: tuple[str, ...],
) -> dict[str, frozenset[str] | None]:
    """Split a robot-complete model's bodies between its robots, by kinematic tree, not by name.

    A robot's root is the deepest body that is an ancestor of every body its actuators move (through
    a joint, tendon or site: the bodies of the dofs in each actuator's moment row); it owns that
    body's subtree. A body under several roots (a robot mounted on another) belongs to the deepest
    root. A body outside every robot (a shared torso or base the arms hang from) belongs to the first
    robot, which streams first (CAM-6). With one robot, it owns every body (``None``).

    Returns:
        The body names of each prefix's robot, or ``None`` for a single robot.
    """
    if len(prefixes) < 2:  # noqa: PLR2004 - several robots need splitting
        return dict.fromkeys(prefixes)
    import mujoco  # noqa: PLC0415

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    moments = np.zeros((model.nu, model.nv))
    mujoco.mju_sparse2dense(moments, data.actuator_moment, data.moment_rownnz, data.moment_rowadr, data.moment_colind)
    depth = [0] * model.nbody
    for body in range(1, model.nbody):
        depth[body] = depth[int(model.body_parentid[body])] + 1
    roots: dict[str, int] = {}
    for prefix in prefixes:
        moved = sorted({
            int(model.dof_bodyid[dof])
            for channel in _robot_channels(layout, profile, prefix, joints_only=False)
            for dof in np.flatnonzero(moments[channel.actuator_id])
        })
        if moved:
            roots[prefix] = _common_ancestor(model, depth, moved)
    owner = dict.fromkeys(range(1, model.nbody), prefixes[0])
    for prefix, root in sorted(roots.items(), key=lambda item: depth[item[1]]):
        for body in _subtree(model, root):
            owner[body] = prefix  # deeper roots come later and win
    names: dict[str, set[str]] = {prefix: set() for prefix in prefixes}
    for body, prefix in owner.items():
        names[prefix].add(model.body(body).name)
    return {prefix: frozenset(bodies) for prefix, bodies in names.items()}


def _common_ancestor(model: mujoco.MjModel, depth: list[int], bodies: list[int]) -> int:
    """Return the deepest body that is an ancestor of (or equal to) every one of *bodies*."""
    common = bodies[0]
    for start in bodies[1:]:
        other = start
        while depth[other] > depth[common]:
            other = int(model.body_parentid[other])
        while depth[common] > depth[other]:
            common = int(model.body_parentid[common])
        while other != common:
            other, common = int(model.body_parentid[other]), int(model.body_parentid[common])
    return common


def generated_cameras(
    model: mujoco.MjModel,
    layout: DerivedLayout,
    profile: RobotProfile,
    prefix: str = "",
) -> tuple[CameraSpec, ...]:
    """Place the default camera of a robot without cameras of its own (CAM-3), named ``{prefix}{name}``.

    The robot is the one whose channels carry *prefix*. A model camera that already has the name
    (a viewer camera such as ``front``) wins: the default is skipped, with a debug log.

    Returns:
        A ``wrist``, ``head`` or ``front`` camera, or nothing for a fixed base without joints.
    """
    import mujoco  # noqa: PLC0415

    camera = _generated_camera(model, layout, profile, prefix)
    if camera is None:
        return ()
    camera = replace(camera, name=f"{prefix}{camera.name}")
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera.name) >= 0:
        logger.debug("No default {!r} camera for {}: the model has a camera of that name", camera.name, profile.name)
        return ()
    return (camera,)


def _generated_camera(
    model: mujoco.MjModel,
    layout: DerivedLayout,
    profile: RobotProfile,
    prefix: str,
) -> CameraSpec | None:
    import mujoco  # noqa: PLC0415

    data = mujoco.MjData(model)
    for joint, values in layout.home_qpos.items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint) if joint else -1
        if joint_id >= 0:  # an unnamed joint (often the free joint) keeps qpos0; poses below are relative
            address = int(model.jnt_qposadr[joint_id])
            data.qpos[address : address + len(values)] = values
    mujoco.mj_kinematics(model, data)
    if layout.base is None:
        body = _end_effector_body(model, layout, profile, prefix)
        return None if body is None else _wrist_camera(model, data, body)
    if not layout.base.body.startswith(prefix):
        return None
    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, layout.base.body)
    head = next(
        (body for body in _subtree(model, root) if any(word in model.body(body).name.lower() for word in _HEAD_WORDS)),
        None,
    )
    if head is not None:
        return _head_camera(model, data, root, head)
    return _front_camera(model, data, root)


def add_cameras(spec: mujoco.MjSpec, cameras: tuple[CameraSpec, ...], profile_name: str) -> None:
    """Add the profile and generated cameras to a robot spec; model cameras are there already (CAM-7).

    Raises:
        ValueError: If a model camera is missing, a camera's body is missing, or a new camera's name
            is taken by a model camera.
    """
    for camera in cameras:
        where = f"Profile {profile_name!r} camera {camera.name!r}"
        if camera.source == "model":
            if spec.camera(camera.name) is None:
                msg = f"{where} names a model camera the robot does not have"
                raise ValueError(msg)
            continue
        body = spec.body(camera.body)
        if body is None:
            msg = f"{where} names missing body {camera.body!r}"
            raise ValueError(msg)
        if spec.camera(camera.name) is not None:
            msg = f"{where} clashes with a model camera of that name"
            raise ValueError(msg)
        body.add_camera(name=camera.name, pos=list(camera.pos), quat=list(camera.quat), fovy=camera.fovy)


def check_camera_names(spec: mujoco.MjSpec, where: object) -> None:
    """Reject a composed scene whose cameras share a name, before MuJoCo's compile does (CAM-5).

    Raises:
        ValueError: Naming the repeated camera.
    """
    seen: set[str] = set()
    for camera in spec.cameras:
        if camera.name and camera.name in seen:
            msg = (
                f"Camera {camera.name!r} appears twice in {where}: a robot camera clashes with a scene "
                "camera or another robot's; camera names must be unique"
            )
            raise ValueError(msg)
        seen.add(camera.name)


def follow_with_chase_camera(spec: mujoco.MjSpec, model: mujoco.MjModel, root_body: str) -> bool:
    """Make the scene's world camera ``chase`` follow *root_body* (CAM-4).

    The camera moves into the root body as a ``trackcom`` camera: it keeps its start pose, then
    follows the robot's centre of mass at a fixed offset without turning with it. The caller
    compiles *spec* again when this returns ``True``.

    Args:
        spec: The composed scene spec that *model* was compiled from.
        model: The compiled scene, whose reference pose places the camera relative to the robot.
        root_body: The floating base body, by its name in the composed model.

    Returns:
        ``True`` when *spec* changed; ``False`` without a world camera ``chase`` or *root_body*.
    """
    import mujoco  # noqa: PLC0415

    camera = spec.camera(CHASE_CAMERA)
    body = spec.body(root_body)
    if camera is None or body is None or camera.parent.name != spec.worldbody.name:
        return False
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CHASE_CAMERA)
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root_body)
    body_rotation = data.xmat[body_id].reshape(3, 3)
    pos = body_rotation.T @ (data.cam_xpos[camera_id] - data.xpos[body_id])
    rotation = body_rotation.T @ data.cam_xmat[camera_id].reshape(3, 3)
    intrinsics = {name: getattr(camera, name) for name in _CAMERA_INTRINSICS}
    spec.delete(camera)
    chase = body.add_camera(name=CHASE_CAMERA, pos=pos.tolist(), quat=_mat_quat(rotation))
    chase.mode = mujoco.mjtCamLight.mjCAMLIGHT_TRACKCOM
    for name, value in intrinsics.items():
        setattr(chase, name, value)
    return True


def _end_effector_body(model: mujoco.MjModel, layout: DerivedLayout, profile: RobotProfile, prefix: str) -> int | None:
    """Return the gripper joint's parent body, else the last arm joint's body, of the robot with *prefix*.

    Returns:
        The body id, or ``None`` when no channel of that robot drives a joint.
    """
    flagged = {f"{prefix}{channel.actuators[0]}" for channel in profile.channels if channel.gripper}
    arm = _robot_channels(layout, profile, prefix)
    if not arm:
        return None
    for channel in arm:
        names = f"{channel.name} {channel.joint}".lower()
        if {channel.actuator, channel.joint} & flagged or any(word in names for word in _GRIPPER_WORDS):
            return int(model.body_parentid[model.jnt_bodyid[channel.joint_id]])
    return int(model.jnt_bodyid[arm[-1].joint_id])


def _robot_channels(
    layout: DerivedLayout, profile: RobotProfile, prefix: str, *, joints_only: bool = True
) -> list[DerivedChannel]:
    """Return the channels of the robot with *prefix*, as `compose.model_prefixes` finds robots.

    Returns:
        The channels whose actuator or joint is a profile actuator with *prefix* (or, for a profile
        without channels, whose actuator name starts with *prefix*), in model order; with
        *joints_only*, only those that drive a joint directly.
    """
    listed = {f"{prefix}{actuator}" for channel in profile.channels for actuator in channel.actuators}
    return [
        channel
        for channel in layout.channels
        if (channel.joint_id is not None or not joints_only)
        and ({channel.actuator, channel.joint} & listed if listed else channel.actuator.startswith(prefix))
    ]


def _wrist_camera(model: mujoco.MjModel, data: mujoco.MjData, body: int) -> CameraSpec:
    subtree = set(_subtree(model, body))
    origin, rotation = data.xpos[body], data.xmat[body].reshape(3, 3)
    points = _geometry_points(model, data, subtree, origin, rotation)
    # The parent link counts for the side too: the camera must not sit inside the arm behind the wrist.
    parent = int(model.body_parentid[body])
    beside = np.concatenate([points, _geometry_points(model, data, {parent} - {0}, origin, rotation)])
    axis, tip = _tool_axis(model, data, subtree, points, origin, rotation)
    # The narrowest side across the tool axis keeps the camera clear of fingers that open sideways.
    first = np.eye(3)[int(np.argmin(np.abs(axis)))]
    first -= (first @ axis) * axis
    first /= np.linalg.norm(first)
    second = np.cross(axis, first)
    sides = (first, -first, second, -second)
    extents = [max(0.0, float((beside @ side).max())) if len(beside) else 0.0 for side in sides]
    side = sides[int(np.argmin(extents))]
    pos = tip - _WRIST_BACK_M * axis + (min(extents) + _WRIST_CLEARANCE_M) * side
    forward = tip + _WRIST_AIM_AHEAD_M * axis - pos
    return CameraSpec(
        name="wrist",
        body=model.body(body).name,
        pos=_vector(pos),
        quat=_look_quat(forward, side),
        fovy=_WRIST_FOVY,
        source="default",
    )


def _tool_axis(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    subtree: set[int],
    points: np.ndarray,
    origin: np.ndarray,
    rotation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the tool axis and tip in the end-effector frame: towards a tip site, else +z to the geometry's end.

    Returns:
        The unit axis and the tip position.
    """
    site = next(
        (
            site
            for site in range(model.nsite)
            if int(model.site_bodyid[site]) in subtree
            and any(word in model.site(site).name.lower() for word in _TIP_SITE_WORDS)
        ),
        None,
    )
    if site is not None:
        tip = rotation.T @ (data.site_xpos[site] - origin)
        length = float(np.linalg.norm(tip))
        if length > 0.01:  # noqa: PLR2004 - a site at the body origin gives its own z axis instead
            return tip / length, tip
        return rotation.T @ data.site_xmat[site].reshape(3, 3)[:, 2], tip
    axis = np.array([0.0, 0.0, 1.0])
    return axis, axis * (max(0.0, float((points @ axis).max())) if len(points) else 0.0)


def _head_camera(model: mujoco.MjModel, data: mujoco.MjData, root: int, head: int) -> CameraSpec:
    root_rotation = data.xmat[root].reshape(3, 3)
    head_rotation = data.xmat[head].reshape(3, 3)
    pitch = np.radians(_HEAD_PITCH_DEG)
    # 8 cm ahead, or just in front of a larger head's geometry.
    points = _geometry_points(model, data, {head}, data.xpos[head], root_rotation)
    ahead = max(_HEAD_AHEAD_M, float(points[:, 0].max()) + _FRONT_CLEARANCE_M if len(points) else 0.0)
    world_pos = data.xpos[head] + ahead * root_rotation[:, 0]
    forward = root_rotation @ np.array([np.cos(pitch), 0.0, -np.sin(pitch)])
    up = root_rotation @ np.array([np.sin(pitch), 0.0, np.cos(pitch)])
    return CameraSpec(
        name="head",
        body=model.body(head).name,
        pos=_vector(head_rotation.T @ (world_pos - data.xpos[head])),
        quat=_look_quat(head_rotation.T @ forward, head_rotation.T @ up),
        fovy=_HEAD_FOVY,
        source="default",
    )


def _front_camera(model: mujoco.MjModel, data: mujoco.MjData, root: int) -> CameraSpec:
    rotation = data.xmat[root].reshape(3, 3)
    points = _geometry_points(model, data, {root}, data.xpos[root], rotation)
    front = max(0.0, float(points[:, 0].max())) if len(points) else 0.0
    pitch = np.radians(_FRONT_PITCH_DEG)
    return CameraSpec(
        name="front",
        body=model.body(root).name,
        pos=(front + _FRONT_CLEARANCE_M, 0.0, 0.0),
        quat=_look_quat(
            np.array([np.cos(pitch), 0.0, -np.sin(pitch)]),
            np.array([np.sin(pitch), 0.0, np.cos(pitch)]),
        ),
        fovy=_FRONT_FOVY,
        source="default",
    )


def _subtree(model: mujoco.MjModel, root: int) -> list[int]:
    """Return *root* and its descendants; MuJoCo orders every body after its parent."""
    bodies = [root]
    members = {root}
    for body in range(root + 1, model.nbody):
        if int(model.body_parentid[body]) in members:
            bodies.append(body)
            members.add(body)
    return bodies


def _geometry_points(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    bodies: set[int],
    origin: np.ndarray,
    rotation: np.ndarray,
) -> np.ndarray:
    """Return the bounding-box corners of the bodies' geoms, in the frame (*origin*, *rotation*).

    Returns:
        An ``(n, 3)`` array; empty when the bodies have no geoms.
    """
    signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=np.float64)
    corners = []
    for geom in range(model.ngeom):
        if int(model.geom_bodyid[geom]) not in bodies:
            continue
        center, half = model.geom_aabb[geom, :3], model.geom_aabb[geom, 3:]
        geom_rotation = data.geom_xmat[geom].reshape(3, 3)
        world = data.geom_xpos[geom] + (center + signs * half) @ geom_rotation.T
        corners.append((world - origin) @ rotation)
    return np.concatenate(corners) if corners else np.zeros((0, 3))


def _look_quat(forward: np.ndarray, up: np.ndarray) -> tuple[float, float, float, float]:
    """Return the orientation of a camera looking along *forward* with *up* at the top of the image.

    Returns:
        A ``wxyz`` quaternion in the frame the vectors are given in.
    """
    # New arrays only: the callers' vectors stay as they were.
    look = np.asarray(forward, dtype=np.float64) / np.linalg.norm(forward)
    top = np.asarray(up, dtype=np.float64) - (np.asarray(up, dtype=np.float64) @ look) * look
    top /= np.linalg.norm(top)
    back = -look
    rotation = np.column_stack([np.cross(top, back), top, back])
    return _mat_quat(rotation)


def _mat_quat(rotation: np.ndarray) -> tuple[float, float, float, float]:
    import mujoco  # noqa: PLC0415

    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.ascontiguousarray(rotation, dtype=np.float64).reshape(9))
    return float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])


def _vector(values: np.ndarray) -> tuple[float, float, float]:
    return float(values[0]), float(values[1]), float(values[2])


__all__ = [
    "CHASE_CAMERA",
    "OVERVIEW_CAMERA",
    "VIEWER_CAMERA_PATTERNS",
    "add_cameras",
    "check_camera_names",
    "follow_with_chase_camera",
    "generated_cameras",
    "is_sensor_camera",
    "model_cameras",
    "resolve_cameras",
]
