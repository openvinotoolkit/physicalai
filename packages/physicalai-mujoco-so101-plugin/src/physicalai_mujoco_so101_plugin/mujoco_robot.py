# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""MuJoCo-backed SO-101 robot implementation."""

# MuJoCo and viser expose runtime-bound attributes exercised by the simulation tests.
# pyrefly: ignore-errors [missing-attribute, not-callable, bad-context-manager]

from __future__ import annotations

import contextlib
import queue
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal, get_args

import numpy as np
from loguru import logger

from physicalai.config import export_config
from physicalai_mujoco_so101_plugin.constants import (
    BIMANUAL_NUM_JOINTS,
    BIMANUAL_SO101_JOINT_ORDER,
    NUM_JOINTS,
    SO101_JOINT_ORDER,
)
from physicalai_mujoco_so101_plugin.spawn import sample_object_positions, write_freejoint_qpos

if TYPE_CHECKING:
    from physicalai.capture.frame import Frame
    from physicalai.robot.interface import RobotObservation
    from physicalai_mujoco_so101_plugin.http_server import FrameBuffer, HttpServer, SimCommand
    from physicalai_mujoco_so101_plugin.viser_controls import ObjectPose, PanelState, SimControlPanel

# Scene XML is polled for live camera edits; walking the include graph is far
# too expensive to do on every control cycle.
_SCENE_XML_POLL_INTERVAL_S = 1.0
_DEFAULT_SUCCESS_DWELL_S = 5.0

JointUnit = Literal["normalized", "degrees"]
"""Units of ``get_observation`` joint positions and ``send_action`` targets."""


def _is_gripper(joint_name: str) -> bool:
    return joint_name == "gripper" or joint_name.endswith("_gripper")


def _normalized_span(joint_names: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Lower bound and width of each joint's normalized range, as the SO101 driver uses them.

    Returns:
        ``(lower, width)``: body joints span ``[-100, 100]``, grippers ``[0, 100]``.
    """
    gripper = np.array([_is_gripper(name) for name in joint_names])
    return np.where(gripper, 0.0, -100.0), np.where(gripper, 100.0, 200.0)


def radians_to_normalized(
    radians: np.ndarray,
    joint_limits: np.ndarray,
    joint_names: tuple[str, ...],
) -> np.ndarray:
    """Map joint angles to the SO101 driver's calibrated normalized units.

    The real driver maps each joint's calibrated tick range linearly onto
    ``[-100, 100]`` (grippers onto ``[0, 100]``) and clamps. The simulation
    uses the model's joint range as the calibrated range.

    Returns:
        Normalized positions, clamped to each joint's normalized range.
    """
    low, high = joint_limits[:, 0], joint_limits[:, 1]
    lower, width = _normalized_span(joint_names)
    fraction = np.clip((radians - low) / (high - low), 0.0, 1.0)
    return lower + fraction * width


def normalized_to_radians(
    normalized: np.ndarray,
    joint_limits: np.ndarray,
    joint_names: tuple[str, ...],
) -> np.ndarray:
    """Map SO101 normalized units back to joint angles within the model's joint range.

    Returns:
        Joint angles in radians, clamped to each joint's range.
    """
    low, high = joint_limits[:, 0], joint_limits[:, 1]
    lower, width = _normalized_span(joint_names)
    fraction = np.clip((normalized - lower) / width, 0.0, 1.0)
    return low + fraction * (high - low)


def _signal_owner_shutdown() -> None:
    """Request a graceful exit of the shared owner loop, if this runs inside one.

    The driver runs inside ``python -m physicalai.robot.transport.
    _owner_worker``, where the worker module is loaded as ``__main__``; check
    both module identities for the loop's shutdown event.
    """
    for module_name in ("__main__", "physicalai.robot.transport._owner_worker"):
        module = sys.modules.get(module_name)
        event = getattr(module, "shutdown", None) if module is not None else None
        if isinstance(event, threading.Event):
            event.set()
            return
    logger.warning("Owner shutdown event not found; use Ctrl+C or stop the process to exit")


@dataclass
class MuJoCoSO101Observation:
    """Observation returned by the MuJoCo SO-101 robot."""

    joint_positions: np.ndarray
    timestamp: float
    sensor_data: dict[str, np.ndarray] | None = None
    images: dict[str, Frame] | None = None

    @property
    def state(self) -> np.ndarray:
        """Joint positions represented as the robot state."""
        return self.joint_positions


@dataclass(frozen=True)
class CameraConfig:
    """Output configuration for a virtual camera.

    When ``device`` is ``None`` the camera is served only over HTTP
    (MJPEG/snapshot). Setting ``device`` to a v4l2loopback node also
    publishes frames there via pyfakewebcam. Frames are streamed as MuJoCo
    renders them; ``mirror_horizontal`` flips them left to right.
    """

    name: str
    device: str | None = None
    width: int = 640
    height: int = 480
    fps: int = 30
    mirror_horizontal: bool = False


@export_config(class_path="physicalai_mujoco_so101_plugin.mujoco_robot.MuJoCoSO101")
class MuJoCoSO101:
    """SO-101 robot simulated with MuJoCo."""

    JOINT_ORDER: ClassVar[tuple[str, ...]] = SO101_JOINT_ORDER
    NUM_JOINTS: ClassVar[int] = NUM_JOINTS
    NUM_ARMS: ClassVar[int] = 1
    DEFAULT_BLOCK_FREEJOINTS: ClassVar[tuple[str, ...]] = ("block1:joint", "block2:joint", "block3:joint")
    DEFAULT_TARGET_BODY_NAME: ClassVar[str] = "target"
    DEFAULT_SPAWN_CENTER: ClassVar[tuple[float, float]] = (0.22, 0.0)
    DEFAULT_SPAWN_MIN_R: ClassVar[float] = 0.05
    DEFAULT_SPAWN_MAX_R: ClassVar[float] = 0.14
    DEFAULT_SPAWN_ANGLE_HALF_DEG: ClassVar[float] = 50.0
    DEFAULT_BLOCK_MIN_SEP: ClassVar[float] = 0.09
    DEFAULT_TARGET_MIN_SEP: ClassVar[float] = 0.11

    def __init__(
        self,
        model_path: str,
        *,
        substeps: int = 1,
        enable_viewer: bool = False,
        cameras: list[CameraConfig | dict] | None = None,
        model: object = None,
        data: object = None,
        scene_config: dict | None = None,
        owner_name: str = "",
        http_host: str = "127.0.0.1",
        http_port: int = 0,
        viser_host: str = "127.0.0.1",
        viser_port: int = 9090,
        unit: JointUnit = "normalized",
    ) -> None:
        """Initialize a disconnected simulation robot.

        ``unit`` selects the units of observed joint positions and action
        targets. ``"normalized"`` matches the calibrated SO101 driver: body
        joints in ``[-100, 100]`` and grippers in ``[0, 100]`` across each
        joint's range in the MuJoCo model, which stands in for a calibrated
        range. ``"degrees"`` uses joint angles. An unsupported ``unit`` raises
        ``ValueError``.
        """
        self._set_joint_unit(unit)
        self._model_path = model_path
        self._substeps = substeps
        self._enable_viewer = enable_viewer
        self._cameras = [cam if isinstance(cam, CameraConfig) else CameraConfig(**cam) for cam in (cameras or [])]
        self._model = model
        self._data = data
        self._viser_server: object | None = None
        self._viser_scene: object | None = None
        self._native_viewer: object | None = None
        self._viser_port = viser_port
        self._camera_devices: dict[str, object] = {}
        self._camera_renderers: dict[str, object] = {}
        self._camera_last_frame_ts: dict[str, float] = {}
        self._frame_buffers: dict[str, FrameBuffer] = {}
        self._commands: queue.Queue[SimCommand] = queue.Queue()
        self._owner_name = owner_name
        self._http_host = http_host
        self._http_port = http_port
        self._http_server: HttpServer | None = None
        self._viser_host = viser_host
        self._ctrl_indices: tuple[int, ...] = ()
        # (NUM_JOINTS, 2) joint ranges in radians, in JOINT_ORDER.
        self._joint_limits: np.ndarray | None = None
        self._block_joint_addrs: list[tuple[int, int]] = []
        self._target_body_id: int | None = None
        self._last_sim_time: float | None = None
        self._rng = np.random.default_rng()
        self._pending_scene_switch: bool = False
        self._current_scene_id: str | None = None
        self._scene_on_reset: object | None = None
        self._scene_xml_paths: list[Path] | None = None
        self._scene_xml_mtimes: dict[str, float] = {}
        self._scene_xml_next_check: float = 0.0
        self._episode_auto_reset: object | None = None
        self._viser_sync_failed: bool = False
        # Guards state the HTTP thread reads (``_http_status``) while the sim
        # thread rebuilds it during a scene switch or disconnect.
        self._state_lock = threading.RLock()
        self._init_control_state()

        self._apply_scene_params(scene_config)

    def _set_joint_unit(self, unit: JointUnit) -> None:
        """Validate and adopt the joint unit.

        Raises:
            ValueError: If ``unit`` is not supported.
        """
        if unit not in get_args(JointUnit):
            msg = f"Unsupported unit {unit!r}; expected one of {get_args(JointUnit)}"
            raise ValueError(msg)
        self._unit: JointUnit = unit

    def _init_control_state(self) -> None:
        """Initialize operator-control state that is not part of the construction recipe."""
        self._seed: int | None = None
        self._auto_reset_active = True
        self._auto_reset_dwell_s = _DEFAULT_SUCCESS_DWELL_S
        self._free_joint_addrs: dict[str, tuple[int, int]] = {}
        # Poses re-applied after every step while a viewer drags an object.
        self._held_objects: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        # Latest free-object poses, published by the sim thread under the lock.
        self._object_poses: dict[str, ObjectPose] = {}
        self._viser_panel: SimControlPanel | None = None
        # Bodies the viewer can follow (name -> id) and each free joint's body name.
        self._follow_body_ids: dict[str, int] = {}
        self._free_joint_bodies: dict[str, str] = {}
        self._viser_panel_failed = False

    def _apply_scene_params(self, scene_config: dict | None) -> None:
        """Adopt a scene's object and spawn parameters, or this class's defaults."""
        if scene_config is None:
            self._free_joints: tuple[str, ...] = self.DEFAULT_BLOCK_FREEJOINTS
            self._target_body_name: str = self.DEFAULT_TARGET_BODY_NAME
            self._spawn_center: tuple[float, float] = self.DEFAULT_SPAWN_CENTER
            self._spawn_min_r: float = self.DEFAULT_SPAWN_MIN_R
            self._spawn_max_r: float = self.DEFAULT_SPAWN_MAX_R
            self._spawn_angle_half_deg: float = self.DEFAULT_SPAWN_ANGLE_HALF_DEG
            self._block_min_sep: float = self.DEFAULT_BLOCK_MIN_SEP
            self._target_min_sep: float = self.DEFAULT_TARGET_MIN_SEP
            return

        self._free_joints = tuple(scene_config["free_joints"])
        self._target_body_name = scene_config["target_bodies"][0] if scene_config["target_bodies"] else ""
        self._spawn_center = tuple(scene_config["spawn_center"])
        self._spawn_min_r = scene_config["spawn_min_r"]
        self._spawn_max_r = scene_config["spawn_max_r"]
        self._spawn_angle_half_deg = scene_config["spawn_angle_half_deg"]
        self._block_min_sep = scene_config["block_min_sep"]
        self._target_min_sep = scene_config["target_min_sep"]
        self._current_scene_id = scene_config.get("scene_id")
        if self._current_scene_id:
            from physicalai_mujoco_so101_plugin.scene_registry import get_reset_fn  # noqa: PLC0415

            self._scene_on_reset = get_reset_fn(self._current_scene_id)

    @property
    def joint_names(self) -> list[str]:
        """Ordered joint names."""
        return list(self.JOINT_ORDER)

    @property
    def device_ids(self) -> tuple[str, ...]:
        """Exclusively owned v4l2 sinks; in-memory simulations own no hardware."""
        return tuple(sorted({f"v4l2:{Path(cam.device).resolve()}" for cam in self._cameras if cam.device}))

    def connect(self) -> None:
        """Load the model and initialize simulation resources.

        Raises:
            ValueError: If the model cannot drive this robot's joints.
        """
        if self.is_connected():
            return
        import mujoco  # noqa: PLC0415

        logger.info("Loading MuJoCo model from {}", self._model_path)
        # pyrefly: ignore [missing-attribute]
        model = mujoco.MjModel.from_xml_path(self._model_path)
        ctrl_indices = self._actuator_indices_for_joint_order(model)
        joint_limits = self._joint_limits_for_joint_order(model)
        if ctrl_indices is None or joint_limits is None:
            msg = f"Model {self._model_path!r} does not provide the joints/actuators for {type(self).__name__}"
            raise ValueError(msg)
        # pyrefly: ignore [missing-attribute]
        data = mujoco.MjData(model)
        # pyrefly: ignore [missing-attribute]
        mujoco.mj_forward(model, data)
        if self._scene_on_reset is not None:
            self._reseed_if_fixed()
            self._scene_on_reset(model, data, self._rng)
        self._model = model
        self._data = data
        self._ctrl_indices = ctrl_indices
        self._joint_limits = joint_limits
        self._last_sim_time = float(self._data.time)
        self._init_block_joint_addrs()
        self._init_episode_auto_reset()
        self._reset_scene_xml_watch()
        self._publish_object_poses()
        logger.info(
            "MuJoCo SO101 connected ({} joints, timestep={})",
            self.NUM_JOINTS,
            self._model.opt.timestep,
        )

        if self._enable_viewer:
            if not self._launch_viser_viewer() and sys.platform != "darwin":
                self._launch_native_viewer()
            if self._viser_scene is None and self._native_viewer is None:
                self._enable_viewer = False

        self._init_cameras()
        self._start_http_server()

    def disconnect(self) -> None:
        """Release simulation resources."""
        self._stop_http_server()
        with self._state_lock:
            for renderer in self._camera_renderers.values():
                with contextlib.suppress(Exception):
                    renderer.close()
            self._camera_renderers.clear()
            self._camera_devices.clear()
            self._camera_last_frame_ts.clear()
            self._frame_buffers.clear()
            self._block_joint_addrs.clear()
            self._free_joint_addrs.clear()
            self._follow_body_ids = {}
            self._free_joint_bodies = {}
            self._held_objects.clear()
            self._object_poses = {}
            self._target_body_id = None
            self._episode_auto_reset = None
            self._last_sim_time = None
            self._ctrl_indices = ()
            self._joint_limits = None

            self._close_viewer()
            self._model = None
            self._data = None
            # Keep the selected scene and its reset recipe for reconnect.
            self._pending_scene_switch = False
            self._commands = queue.Queue()
            self._scene_xml_paths = None
        logger.info("MuJoCo SO101 disconnected")

    def is_connected(self) -> bool:
        """Return whether the simulation model is loaded."""
        return self._model is not None

    def _step_and_sync(self) -> None:
        import mujoco  # noqa: PLC0415

        self._drain_commands()
        self._check_scene_xml_camera()

        for _ in range(self._substeps):
            # pyrefly: ignore [missing-attribute]
            mujoco.mj_step(self._model, self._data)

        if self._episode_auto_reset is not None:
            self._episode_auto_reset.update(self._model, self._data)

        self._apply_held_objects()
        self._publish_object_poses()

        if self._viser_scene is not None:
            self._sync_viser()
            self._refresh_viser_panel()
        elif self._native_viewer is not None:
            if self._native_viewer_is_running():
                self._native_viewer_sync()
                self._handle_viewer_reset()
            else:
                logger.info("MuJoCo viewer closed by user")
                self._enable_viewer = False
                self._native_viewer = None

        self._render_cameras()

    def _init_cameras(self) -> None:
        if not self._cameras:
            return

        import mujoco  # noqa: PLC0415

        for config in self._cameras:
            try:
                renderer = mujoco.Renderer(self._model, config.height, config.width)
            except OSError as exc:
                logger.warning("Camera '{}' renderer unavailable: {}", config.name, exc)
                continue
            self._camera_renderers[config.name] = renderer
            if config.name not in self._frame_buffers:
                from physicalai_mujoco_so101_plugin.http_server import FrameBuffer  # noqa: PLC0415

                self._frame_buffers[config.name] = FrameBuffer(config.name)
            self._camera_last_frame_ts[config.name] = 0.0

            if config.device:
                self._init_v4l2_device(config)
            logger.info(
                "Camera started: {} ({}x{}@{} fps, v4l2={})",
                config.name,
                config.width,
                config.height,
                config.fps,
                config.device or "off",
            )

    def _init_v4l2_device(self, config: CameraConfig) -> None:
        try:
            import pyfakewebcam  # noqa: PLC0415
        except ImportError as exc:
            logger.warning("pyfakewebcam unavailable, camera '{}' v4l2 output disabled: {}", config.name, exc)
            return
        try:
            self._camera_devices[config.name] = pyfakewebcam.FakeWebcam(config.device, config.width, config.height)
        except OSError as exc:
            logger.warning("Camera '{}' v4l2 device '{}' unavailable: {}", config.name, config.device, exc)

    def _init_block_joint_addrs(self) -> None:
        import mujoco  # noqa: PLC0415

        self._block_joint_addrs.clear()
        self._free_joint_addrs.clear()
        for joint_name in self._free_joints:
            # pyrefly: ignore [missing-attribute]
            jid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if jid < 0:
                continue
            # pyrefly: ignore [missing-attribute]
            qpos_addr = int(self._model.jnt_qposadr[jid])
            # pyrefly: ignore [missing-attribute]
            dof_addr = int(self._model.jnt_dofadr[jid])
            self._block_joint_addrs.append((qpos_addr, dof_addr))
            if int(self._model.jnt_type[jid]) == int(mujoco.mjtJoint.mjJNT_FREE):
                self._free_joint_addrs[joint_name] = (qpos_addr, dof_addr)

        # pyrefly: ignore [missing-attribute]
        target_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, self._target_body_name)
        self._target_body_id = int(target_id) if target_id >= 0 else None
        self._init_follow_bodies()

    def _init_follow_bodies(self) -> None:
        """Collect the bodies the viewer can follow: free objects, the target, and the grippers."""
        import mujoco  # noqa: PLC0415

        model = self._model
        follow: dict[str, int] = {}
        joint_bodies: dict[str, str] = {}
        for joint_name in self._free_joint_addrs:
            joint_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name))
            body_id = int(model.jnt_bodyid[joint_id])
            body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"body_{body_id}"
            follow[body_name] = body_id
            joint_bodies[joint_name] = body_name
        candidates = [self._target_body_name] if self._target_body_name else []
        candidates += [f"{prefix}gripper" for prefix in ("", "left_", "right_")]
        for body_name in candidates:
            body_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name))
            if body_id > 0 and body_name not in follow:
                follow[body_name] = body_id
        with self._state_lock:
            self._follow_body_ids = follow
            self._free_joint_bodies = joint_bodies

    def _init_episode_auto_reset(self) -> None:
        from physicalai_mujoco_so101_plugin.episode_auto_reset import EpisodeAutoReset  # noqa: PLC0415

        if self._model is None:
            with self._state_lock:
                self._episode_auto_reset = None
            return
        episode_auto_reset = EpisodeAutoReset.maybe_create(
            self._model,
            free_joints=self._free_joints,
            target_body_name=self._target_body_name,
            spawn_center=self._spawn_center,
            spawn_min_r=self._spawn_min_r,
            spawn_max_r=self._spawn_max_r,
            spawn_angle_half_deg=self._spawn_angle_half_deg,
            target_min_sep=self._target_min_sep,
            rng=self._rng,
            success_dwell_s=self._auto_reset_dwell_s,
            active=self._auto_reset_active,
        )
        with self._state_lock:
            self._episode_auto_reset = episode_auto_reset

    def _key_callback(self, key: int) -> None:
        if key in {ord("n"), ord("N")} and not self._pending_scene_switch:
            self._pending_scene_switch = True

    def _collect_scene_xml_paths(self) -> list[Path]:
        root_path = Path(self._model_path).resolve()
        visited: set[Path] = set()
        ordered_paths: list[Path] = []

        def walk(path: Path) -> None:
            normalized = path.resolve()
            if normalized in visited:
                return
            visited.add(normalized)
            ordered_paths.append(normalized)

            try:
                from defusedxml import ElementTree  # noqa: PLC0415

                root = ElementTree.parse(normalized).getroot()
            # pyrefly: ignore [unbound-name]
            except (ElementTree.ParseError, OSError):
                return

            # pyrefly: ignore [missing-attribute]
            for include in root.findall(".//include"):
                include_file = include.get("file")
                if not include_file:
                    continue
                include_path = (normalized.parent / include_file).resolve()
                walk(include_path)

        walk(root_path)
        return ordered_paths

    def _scene_xml_paths_cached(self) -> list[Path]:
        """Return the scene's include graph, walking the XML only when stale.

        The graph is static between scene switches and XML edits, so parsing it
        on every control cycle would cost a full re-parse of every included file
        at the loop rate.

        Returns:
            Absolute paths of the scene XML and everything it includes.
        """
        if self._scene_xml_paths is None:
            self._scene_xml_paths = self._collect_scene_xml_paths()
        return self._scene_xml_paths

    def _reset_scene_xml_watch(self) -> None:
        """Re-walk the include graph and re-baseline the watched mtimes."""
        self._scene_xml_paths = None
        self._scene_xml_mtimes = self._snapshot_scene_xml_mtimes()
        self._scene_xml_next_check = time.monotonic() + _SCENE_XML_POLL_INTERVAL_S

    def _snapshot_scene_xml_mtimes(self) -> dict[str, float]:
        mtimes: dict[str, float] = {}
        for xml_path in self._scene_xml_paths_cached():
            try:
                mtimes[str(xml_path)] = xml_path.stat().st_mtime
            except OSError:
                continue
        return mtimes

    def _check_scene_xml_camera(self) -> None:
        now = time.monotonic()
        if now < self._scene_xml_next_check:
            return
        self._scene_xml_next_check = now + _SCENE_XML_POLL_INTERVAL_S

        current_mtimes = self._snapshot_scene_xml_mtimes()
        if current_mtimes == self._scene_xml_mtimes:
            return
        logger.info("Scene XML changed, updating camera")
        self._scene_xml_mtimes = current_mtimes
        self._update_camera_from_xml()
        # The edit may have added or removed an <include>; re-walk next poll.
        self._scene_xml_paths = None

    def _update_camera_from_xml(self) -> None:  # noqa: C901, PLR0912, PLR0915
        import mujoco  # noqa: PLC0415
        from defusedxml import ElementTree  # noqa: PLC0415

        roots = []
        for xml_path in self._scene_xml_paths_cached():
            # _collect_scene_xml_paths keeps unreadable includes in the list, so
            # OSError is as survivable here as a parse error.
            try:
                roots.append(ElementTree.parse(xml_path).getroot())
            except (ElementTree.ParseError, OSError) as exc:
                logger.warning("Failed to parse XML {}: {}", xml_path, exc)

        if not roots:
            return

        def find_first(xpath: str) -> object | None:
            for root in roots:
                # pyrefly: ignore [missing-attribute]
                elem = root.find(xpath)
                if elem is not None:
                    return elem
            return None

        cam = find_first(".//camera[@name='overview']")
        if cam is None:
            return

        for body_name in (
            "overview_camera_rig",
            "overview_camera_tilt",
            "camera_mount",
            "camera_mount_wrist",
            "left_camera_mount",
            "right_camera_mount",
        ):
            body_elem = find_first(f".//body[@name='{body_name}']")
            # pyrefly: ignore [missing-attribute]
            body_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_elem is None or body_id < 0:
                continue

            # pyrefly: ignore [missing-attribute]
            pos_str = body_elem.get("pos")
            if pos_str:
                # pyrefly: ignore [missing-attribute]
                self._model.body_pos[body_id] = [float(x) for x in pos_str.split()]

            # pyrefly: ignore [missing-attribute]
            euler_str = body_elem.get("euler")
            if euler_str:
                euler_vals = [float(x) for x in euler_str.split()]
                if len(euler_vals) == 3:  # noqa: PLR2004
                    quat = np.zeros(4, dtype=np.float64)
                    # pyrefly: ignore [missing-attribute]
                    mujoco.mju_euler2Quat(quat, euler_vals, "xyz")
                    # pyrefly: ignore [missing-attribute]
                    self._model.body_quat[body_id] = quat
                    logger.info("Updated {}: {}", body_name, euler_vals)
                else:
                    logger.warning("Invalid {} euler values: {}", body_name, euler_str)

            # pyrefly: ignore [missing-attribute]
            quat_str = body_elem.get("quat")
            if quat_str:
                quat_vals = [float(x) for x in quat_str.split()]
                if len(quat_vals) == 4:  # noqa: PLR2004
                    self._model.body_quat[body_id] = quat_vals  # pyrefly: ignore [missing-attribute]
                    logger.info("Updated {} quat: {}", body_name, quat_vals)
                else:
                    logger.warning("Invalid {} quat values: {}", body_name, quat_str)

        def update_camera_pose(camera_name: str, camera_elem: object) -> None:  # noqa: PLR0912
            camera_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)  # pyrefly: ignore [missing-attribute]
            if camera_id < 0:
                return

            # pyrefly: ignore [missing-attribute]
            pos_str = camera_elem.get("pos")
            if pos_str:
                pos = [float(x) for x in pos_str.split()]
                # pyrefly: ignore [missing-attribute]
                self._model.cam_pos[camera_id] = pos

            # pyrefly: ignore [missing-attribute]
            fovy_str = camera_elem.get("fovy")
            if fovy_str:
                # pyrefly: ignore [missing-attribute]
                self._model.cam_fovy[camera_id] = float(fovy_str)

            # pyrefly: ignore [missing-attribute]
            xyaxes_str = camera_elem.get("xyaxes")
            # pyrefly: ignore [missing-attribute]
            euler_str = camera_elem.get("euler")
            # pyrefly: ignore [missing-attribute]
            quat_str = camera_elem.get("quat")

            if quat_str:
                quat_vals = [float(x) for x in quat_str.split()]
                if len(quat_vals) == 4:  # noqa: PLR2004
                    self._model.cam_quat[camera_id] = quat_vals  # pyrefly: ignore [missing-attribute]
                    logger.info("Updated camera {} quat: {}", camera_name, quat_vals)
                else:
                    logger.warning("Invalid {} camera quat values: {}", camera_name, quat_str)
            elif xyaxes_str:
                vals = [float(x) for x in xyaxes_str.split()]
                if len(vals) == 6:  # noqa: PLR2004
                    # As the MJCF compiler does: normalize x, make y orthogonal
                    # to it, and use the axes as the rotation matrix's columns.
                    x_axis = np.asarray(vals[:3], dtype=np.float64)
                    x_axis /= np.linalg.norm(x_axis)
                    y_axis = np.asarray(vals[3:], dtype=np.float64)
                    y_axis -= (y_axis @ x_axis) * x_axis
                    y_axis /= np.linalg.norm(y_axis)
                    mat = np.column_stack((x_axis, y_axis, np.cross(x_axis, y_axis))).ravel()
                    quat = np.zeros(4, dtype=np.float64)
                    # pyrefly: ignore [missing-attribute]
                    mujoco.mju_mat2Quat(quat, mat)
                    # pyrefly: ignore [missing-attribute]
                    self._model.cam_quat[camera_id] = quat
                    logger.info("Updated camera {} xyaxes: {}", camera_name, vals)
                else:
                    logger.warning("Invalid {} camera xyaxes values: {}", camera_name, xyaxes_str)
            elif euler_str:
                euler_vals = [float(x) for x in euler_str.split()]
                if len(euler_vals) == 3:  # noqa: PLR2004
                    quat = np.zeros(4, dtype=np.float64)
                    # pyrefly: ignore [missing-attribute]
                    mujoco.mju_euler2Quat(quat, euler_vals, "xyz")
                    # pyrefly: ignore [missing-attribute]
                    self._model.cam_quat[camera_id] = quat
                    logger.info("Updated camera {} euler: {} -> quat={}", camera_name, euler_vals, quat.tolist())
                else:
                    logger.warning("Invalid {} camera euler values: {}", camera_name, euler_str)
            else:
                logger.info("No orientation attr (euler/xyaxes/quat) on {} camera", camera_name)

        update_camera_pose("overview", cam)
        for camera_name in ("wrist", "left_wrist", "right_wrist"):
            wrist_cam = find_first(f".//camera[@name='{camera_name}']")
            if wrist_cam is not None:
                update_camera_pose(camera_name, wrist_cam)

        # pyrefly: ignore [missing-attribute]
        mujoco.mj_forward(self._model, self._data)

    def _actuator_indices_for_joint_order(self, model: object) -> tuple[int, ...] | None:
        """Resolve each public joint name to its direct MuJoCo actuator.

        Returns:
            Control indexes in ``JOINT_ORDER``, or ``None`` if the model is
            missing a joint or does not have exactly one direct joint actuator
            for every public joint.
        """
        import mujoco  # noqa: PLC0415

        actuator_indices: list[int] = []
        joint_ids: set[int] = set()
        for name in self.JOINT_ORDER:
            joint_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name))
            if joint_id < 0 or joint_id in joint_ids:
                return None
            joint_ids.add(joint_id)

            matches = [
                actuator_id
                for actuator_id in range(int(model.nu))
                if int(model.actuator_trntype[actuator_id]) == mujoco.mjtTrn.mjTRN_JOINT
                and int(model.actuator_trnid[actuator_id, 0]) == joint_id
            ]
            if len(matches) != 1:
                return None
            actuator_indices.append(matches[0])

        if len(set(actuator_indices)) != self.NUM_JOINTS:
            return None
        return tuple(actuator_indices)

    def _switch_to_scene(self, scene_id: str) -> bool:
        """Hot-swap the simulation to another registered scene.

        Returns:
            ``True`` when the scene was installed, ``False`` when it was missing
            or incompatible with this robot (the current scene is kept).
        """
        import mujoco  # noqa: PLC0415

        from physicalai_mujoco_so101_plugin.scene_registry import get_reset_fn, get_scene  # noqa: PLC0415

        scene = get_scene(scene_id)
        if scene.num_arms != self.NUM_ARMS:
            logger.error(
                "Scene '{}' has {} arm(s) but {} drives {}; keeping scene '{}'",
                scene_id,
                scene.num_arms,
                type(self).__name__,
                self.NUM_ARMS,
                self._current_scene_id,
            )
            return False
        xml_path = scene.scene_xml_path
        if not xml_path.exists():
            logger.error("Scene XML not found: {}", xml_path)
            return False

        # pyrefly: ignore [missing-attribute]
        new_model = mujoco.MjModel.from_xml_path(str(xml_path))
        # pyrefly: ignore [missing-attribute]
        new_data = mujoco.MjData(new_model)
        # pyrefly: ignore [missing-attribute]
        mujoco.mj_forward(new_model, new_data)

        # A scene built for a different arm count would leave get_observation and
        # send_action indexing joints/actuators the model does not have.
        ctrl_indices = self._actuator_indices_for_joint_order(new_model)
        joint_limits = self._joint_limits_for_joint_order(new_model)
        if ctrl_indices is None or joint_limits is None:
            logger.error(
                "Scene '{}' does not provide the {} joints {} drives; keeping scene '{}'",
                scene_id,
                self.NUM_JOINTS,
                type(self).__name__,
                self._current_scene_id,
            )
            return False

        # Prepare the replacement before retiring the current scene. A failed
        # reset must leave the running model and its renderers usable.
        on_reset = get_reset_fn(scene_id)
        if on_reset is not None:
            self._reseed_if_fixed()
            on_reset(new_model, new_data, self._rng)

        with self._state_lock:
            for renderer in self._camera_renderers.values():
                with contextlib.suppress(Exception):
                    renderer.close()
            self._camera_renderers.clear()
            self._camera_devices.clear()

            self._model_path = str(xml_path)
            self._model = new_model
            self._data = new_data
            self._ctrl_indices = ctrl_indices
            self._joint_limits = joint_limits
            self._held_objects.clear()

            self._native_viewer_set_model_data(new_model, new_data)

            self._free_joints = scene.free_joints
            self._target_body_name = scene.target_bodies[0] if scene.target_bodies else ""
            self._spawn_center = scene.spawn_center
            self._spawn_min_r = scene.spawn_min_r
            self._spawn_max_r = scene.spawn_max_r
            self._spawn_angle_half_deg = scene.spawn_angle_half_deg
            self._block_min_sep = scene.block_min_sep
            self._target_min_sep = scene.target_min_sep

            self._last_sim_time = None
            self._init_block_joint_addrs()
            self._init_episode_auto_reset()
            self._init_cameras()

            self._reset_scene_xml_watch()

            self._current_scene_id = scene_id
            self._scene_on_reset = on_reset
            self._publish_object_poses()
            # Rebuild last: the viewer panel renders the new scene's state.
            self._recreate_viser_scene()

        logger.info(
            "Switched to scene '{}' ({} bodies, {} geoms, {} joints)",
            scene_id,
            new_model.nbody,
            new_model.ngeom,
            new_model.njnt,
        )
        return True

    def _close_viewer(self) -> None:
        if self._viser_server is not None:
            stop = getattr(self._viser_server, "stop", None)
            if callable(stop):
                with contextlib.suppress(Exception):
                    stop()
            self._viser_server = None
            self._viser_scene = None
            self._viser_panel = None
        if self._native_viewer is not None:
            with contextlib.suppress(Exception):
                self._native_viewer.close()
            self._native_viewer = None

    def _launch_viser_viewer(self) -> bool:
        """Start a browser-based 3D viewer via mjviser/viser (works on macOS).

        Returns:
            Whether the viewer was launched.
        """
        if self._viser_port <= 0:
            return False
        try:
            import mjviser  # noqa: F401, PLC0415
            import rich  # noqa: PLC0415
            import viser  # noqa: PLC0415
        except ImportError as exc:
            logger.warning("mjviser/viser unavailable, web viewer disabled: {}", exc)
            return False

        # SharedRobot's owner worker closes stdout after its READY handshake.
        # Viser prints from its background thread during stop(), so give Rich a
        # stream that remains open throughout owner teardown.
        rich.get_console().file = sys.stderr

        server = None
        try:
            server = viser.ViserServer(host=self._viser_host, port=self._viser_port, verbose=False)
            self._viser_server = server
            self._build_viser_gui()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to start viser viewer on port {}: {}", self._viser_port, exc)
            if server is not None:
                with contextlib.suppress(Exception):
                    server.stop()
            self._viser_server = None
            self._viser_scene = None
            self._viser_panel = None
            return False
        logger.info("3D viewer: http://{}:{}", self._viser_host, self._viser_port)
        return True

    def _build_viser_gui(self) -> None:
        """Build the viewer's scene and GUI for the current model from scratch.

        Scene switches call this too, so it first clears every GUI element and
        scene node of the previous model (mjviser only ever adds them).

        mjviser's own camera GUI and body tracking are not used: tracking
        follows the model's first moving body (an arbitrary object) by
        shifting the rendered world, and each build registers another
        client-connect hook. The panel's Camera tab moves the viewer cameras
        instead, and the world stays in place.

        Raises:
            RuntimeError: If no Viser server is running.
        """
        import viser  # noqa: PLC0415
        from mjviser import ViserMujocoScene  # noqa: PLC0415

        from physicalai_mujoco_so101_plugin.viser_controls import SimControlPanel  # noqa: PLC0415

        server = self._viser_server
        if server is None:
            msg = "viser server is not running"
            raise RuntimeError(msg)
        if self._viser_panel is None:
            self._viser_panel = SimControlPanel(server, viser, self._submit_command)

        self._viser_scene = None
        server.gui.reset()
        server.scene.reset()

        scene = ViserMujocoScene(server, self._model, num_envs=1)
        scene.camera_tracking_enabled = False
        scene.set_refresh_handler(self._on_viser_visualization_change)
        state = self._panel_state()
        tabs = server.gui.add_tab_group()
        with tabs.add_tab("Simulation", icon=viser.Icon.ROBOT):
            self._viser_panel.build(state)
        with tabs.add_tab("Camera", icon=viser.Icon.VIDEO):
            self._viser_panel.build_camera_tab(state)
        with tabs.add_tab("Visualization", icon=viser.Icon.EYE):
            scene.create_overlay_gui()
        with tabs.add_tab("Groups", icon=viser.Icon.LAYERS_INTERSECT):
            scene.create_groups_gui()
        self._viser_panel.build_scene_nodes(state)

        self._viser_scene = scene
        self._viser_panel_failed = False

    def _on_viser_visualization_change(self) -> None:
        """Let a Visualization/Groups tab change show up on the next sim tick.

        mjviser's default refresh re-renders from the Viser thread while the
        sim thread may be stepping the same data; the sim loop re-renders at
        the control rate anyway.
        """

    def _recreate_viser_scene(self) -> None:
        """Rebuild the viewer after a model hot-swap.

        ``self._model``/``self._data`` are already the new scene's by the time
        this runs, so on failure the old ``_viser_scene`` (built against the
        previous model) must not be left in place: syncing it against the new
        data would feed mismatched geometry and data into the viewer, silently,
        every tick.
        """
        if self._viser_server is None:
            return
        self._viser_scene = None
        try:
            self._build_viser_gui()
        except Exception as exc:  # noqa: BLE001
            self._viser_scene = None
            logger.warning("Failed to recreate viser scene: {}", exc)

    def _panel_state(self) -> PanelState:
        """Snapshot the state rendered by the viewer's Simulation panel.

        Returns:
            The current scene, compatible scenes, seed, episode status, object
            poses and camera frame buffers.
        """
        from physicalai_mujoco_so101_plugin.scene_registry import list_scenes_for_arms  # noqa: PLC0415
        from physicalai_mujoco_so101_plugin.viser_controls import PanelState  # noqa: PLC0415

        with self._state_lock:
            auto_reset = self._episode_auto_reset
            return PanelState(
                scene_id=self._current_scene_id,
                scene_options=tuple(
                    (scene_id, scene.display_name) for scene_id, scene in list_scenes_for_arms(self.NUM_ARMS).items()
                ),
                seed=self._seed,
                episode=self._episode_status(auto_reset),
                objects=dict(self._object_poses),
                # Configured names, not just live buffers: the viewer is built
                # before connect() starts the camera renderers.
                cameras={config.name: self._frame_buffers.get(config.name) for config in self._cameras},
                follow_targets=self._follow_targets(),
                object_bodies=dict(self._free_joint_bodies),
                view_center=self._view_center(),
                view_extent=float(self._model.stat.extent) if self._model is not None else 1.0,
            )

    @staticmethod
    def _episode_status(auto_reset: object | None) -> dict[str, object]:
        return auto_reset.status() if auto_reset is not None else {"enabled": False}

    def _refresh_viser_panel(self) -> None:
        """Mirror sim state into the viewer panel without letting it stop the loop."""
        panel = self._viser_panel
        if panel is None:
            return
        try:
            panel.refresh(self._panel_state(), time.monotonic())
        except Exception as exc:  # noqa: BLE001
            if not self._viser_panel_failed:
                self._viser_panel_failed = True
                logger.warning("viser panel refresh failed, its readouts will be stale: {}", exc)
        else:
            self._viser_panel_failed = False

    def _sync_viser(self) -> None:
        """Push the current sim state into the viser scene.

        Failures are reported once and then muted: this runs at the loop rate,
        and a broken viewer must not stop the simulation.
        """
        try:
            self._viser_scene.update_from_mjdata(self._data)
            self._sync_viser_fixed_bodies()
        except Exception as exc:  # noqa: BLE001
            if not self._viser_sync_failed:
                self._viser_sync_failed = True
                logger.warning("viser sync failed, the 3D viewer will not update: {}", exc)
        else:
            self._viser_sync_failed = False

    def _view_center(self) -> tuple[float, float, float]:
        """Return the model centre, where a free viewer camera looks by default."""
        if self._model is None:
            return (0.0, 0.0, 0.0)
        x, y, z = (float(v) for v in self._model.stat.center)
        return (x, y, z)

    def _follow_targets(self) -> dict[str, tuple[float, float, float]]:
        """Return the current world position of each body the viewer can follow."""
        data = self._data
        if data is None:
            return {}
        return {name: tuple(float(v) for v in data.xpos[body_id]) for name, body_id in self._follow_body_ids.items()}  # type: ignore[misc]

    def _sync_viser_fixed_bodies(self) -> None:
        """Push current ``data.xpos`` into mjviser's fixed-geometry handles.

        The scene XML watch (see ``_reset_scene_xml_watch``) can move fixed
        world bodies — e.g. camera rigs — by writing ``model.body_pos``
        outside a normal sim step. MuJoCo cameras pick that up after
        ``mj_forward``, but mjviser only places fixed meshes at create time,
        so it needs this explicit resync to reflect the change.

        Handles live under ``/fixed_bodies``, which stays at the world origin,
        so store world ``xpos`` as-is.
        """
        scene = self._viser_scene
        data = self._data
        if scene is None or data is None:
            return

        handles = getattr(scene, "_fixed_geom_handles", None) or {}
        for (body_id, *_rest), handle in handles.items():
            handle.position = np.asarray(data.xpos[body_id], dtype=np.float64)
            handle.wxyz = np.asarray(data.xquat[body_id], dtype=np.float64)

        site_handles = getattr(scene, "_fixed_site_handles", None) or {}
        for (body_id, *_rest), handle in site_handles.items():
            handle.position = np.asarray(data.xpos[body_id], dtype=np.float64)
            handle.wxyz = np.asarray(data.xquat[body_id], dtype=np.float64)

    def _launch_native_viewer(self) -> bool:
        """Start the native MuJoCo viewer (Linux/Windows only; broken on macOS).

        Returns:
            Whether the viewer was launched.
        """
        try:
            import mujoco.viewer  # noqa: PLC0415

            viewer = mujoco.viewer.launch_passive(
                self._model,
                self._data,
                key_callback=self._key_callback,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to open MuJoCo viewer: {}", exc)
            return False
        else:
            self._native_viewer = viewer
            logger.info("MuJoCo native viewer opened")
            return True

    def _native_viewer_is_running(self) -> bool:
        if self._native_viewer is None:
            return False
        is_running = getattr(self._native_viewer, "is_running", None)
        if callable(is_running):
            with contextlib.suppress(Exception):
                return bool(is_running())
        return True

    def _native_viewer_sync(self) -> None:
        if self._native_viewer is None:
            return
        sync = getattr(self._native_viewer, "sync", None)
        if callable(sync):
            with contextlib.suppress(Exception):
                sync()

    def _native_viewer_set_model_data(self, model: object, data: object) -> None:
        """Hot-swap scene for native viewer APIs that support it."""
        if self._native_viewer is None or not self._native_viewer_is_running():
            return
        lock = getattr(self._native_viewer, "lock", None)
        get_sim = getattr(self._native_viewer, "_get_sim", None)
        if not callable(lock) or not callable(get_sim):
            return
        with contextlib.suppress(Exception), lock():
            sim = get_sim()
            if sim is not None:
                sim.m = model
                sim.d = data

    def _check_pending_scene_switch(self) -> None:
        if not self._pending_scene_switch:
            return
        self._pending_scene_switch = False

        from physicalai_mujoco_so101_plugin.scene_registry import list_scenes_for_arms  # noqa: PLC0415

        try:
            scene_ids = list(list_scenes_for_arms(self.NUM_ARMS))
            if not scene_ids:
                logger.warning("No scenes available for switching")
                return

            current = self._current_scene_id
            idx = 0 if current is None or current not in scene_ids else scene_ids.index(current)
            for offset in range(1, len(scene_ids) + 1):
                candidate = scene_ids[(idx + offset) % len(scene_ids)]
                if candidate == current:
                    break
                if self._switch_to_scene(candidate):
                    return
            logger.warning("No other scene is compatible with {}", type(self).__name__)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to switch scene: {}", exc)

    def _handle_viewer_reset(self) -> None:
        # pyrefly: ignore [missing-attribute]
        current = float(self._data.time)
        if self._last_sim_time is None:
            self._last_sim_time = current
            return
        if current + 1e-9 < self._last_sim_time:
            logger.info("Viewer reset detected; randomizing")
            self._run_scene_reset()
            if self._native_viewer is not None and self._native_viewer_is_running():
                self._native_viewer_sync()
            current = float(self._data.time)
        self._last_sim_time = current

    def _run_scene_reset(self) -> None:
        """Randomize the current scene without letting a failure stop the loop.

        Reset callbacks read scene-specific model layout (flex vertices, named
        freejoints); a scene that does not match its callback must degrade to a
        logged warning rather than take the owner down.
        """
        try:
            self._held_objects.clear()
            self._reseed_if_fixed()
            if self._scene_on_reset is not None:
                self._scene_on_reset(self._model, self._data, self._rng)
            else:
                self._randomize_blocks()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Scene reset failed: {}", exc)
            return
        if self._episode_auto_reset is not None:
            self._episode_auto_reset.notify_manual_reset()

    def _reseed_if_fixed(self) -> None:
        """Restart the shared RNG from the fixed seed so the next reset repeats.

        The generator is reseeded in place because the episode auto-reset
        helper holds a reference to it.
        """
        if self._seed is not None:
            self._rng.bit_generator.state = np.random.PCG64(self._seed).state

    def _set_seed(self, seed: int | None) -> None:
        self._seed = seed
        self._reseed_if_fixed()
        logger.info("Reset seed {}", "cleared" if seed is None else f"fixed to {seed}")

    def _set_auto_reset(self, *, enabled: bool | None, dwell_s: float | None) -> None:
        if enabled is not None:
            self._auto_reset_active = enabled
        if dwell_s is not None:
            self._auto_reset_dwell_s = float(dwell_s)
        helper = self._episode_auto_reset
        if helper is None:
            logger.warning("Scene '{}' has no episode auto-reset", self._current_scene_id)
            return
        with self._state_lock:
            helper.set_active(self._auto_reset_active)
            helper.set_dwell(self._auto_reset_dwell_s)
        logger.info(
            "Episode auto-reset {} (dwell {:.1f}s)",
            "enabled" if self._auto_reset_active else "disabled",
            self._auto_reset_dwell_s,
        )

    def _home_targets(self) -> dict[str, float]:
        """Return the current scene's home joint positions (radians)."""
        if self._current_scene_id is None:
            return {}
        from physicalai_mujoco_so101_plugin.scene_registry import get_scene  # noqa: PLC0415

        try:
            return dict(get_scene(self._current_scene_id).home_qpos)
        except KeyError:
            return {}

    def _go_home(self) -> None:
        """Place the arm joints at the scene's home pose and target it.

        Joints without a scene-specific home value use the model default
        (``qpos0``). Positions and actuator targets are clipped to the
        joint and control ranges; the next ``send_action`` overrides them.
        """
        import mujoco  # noqa: PLC0415

        model, data = self._model, self._data
        home = self._home_targets()
        for index, name in enumerate(self.JOINT_ORDER):
            joint_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name))
            qpos_addr = int(model.jnt_qposadr[joint_id])
            dof_addr = int(model.jnt_dofadr[joint_id])
            value = float(home.get(name, model.qpos0[qpos_addr]))
            if bool(model.jnt_limited[joint_id]):
                low, high = (float(v) for v in model.jnt_range[joint_id])
                value = min(max(value, low), high)
            data.qpos[qpos_addr] = value
            data.qvel[dof_addr] = 0.0

            actuator_id = self._ctrl_indices[index]
            target = value
            if bool(model.actuator_ctrllimited[actuator_id]):
                low, high = (float(v) for v in model.actuator_ctrlrange[actuator_id])
                target = min(max(target, low), high)
            data.ctrl[actuator_id] = target
        mujoco.mj_forward(model, data)
        logger.info("Arm moved to the home pose")

    def _write_object_pose(self, joint: str, position: np.ndarray, wxyz: np.ndarray) -> None:
        qpos_addr, dof_addr = self._free_joint_addrs[joint]
        self._data.qpos[qpos_addr : qpos_addr + 3] = position
        self._data.qpos[qpos_addr + 3 : qpos_addr + 7] = wxyz
        self._data.qvel[dof_addr : dof_addr + 6] = 0.0

    def _set_object_pose(
        self,
        joint: str,
        position: tuple[float, float, float],
        wxyz: tuple[float, float, float, float] | None,
        *,
        hold: bool,
    ) -> None:
        """Teleport a free object, optionally holding it there every step."""
        import mujoco  # noqa: PLC0415

        if joint not in self._free_joint_addrs:
            logger.warning("Unknown free object joint {!r}", joint)
            return
        qpos_addr, _ = self._free_joint_addrs[joint]
        pos = np.asarray(position, dtype=np.float64)
        quat = np.asarray(
            wxyz if wxyz is not None else self._data.qpos[qpos_addr + 3 : qpos_addr + 7], dtype=np.float64
        )
        norm = float(np.linalg.norm(quat))
        if pos.shape != (3,) or quat.shape != (4,) or not np.isfinite(pos).all() or not np.isfinite(norm) or norm == 0:
            logger.warning("Ignoring invalid pose for {!r}", joint)
            return
        quat /= norm
        self._write_object_pose(joint, pos, quat)
        if hold:
            self._held_objects[joint] = (pos, quat)
        else:
            self._held_objects.pop(joint, None)
        mujoco.mj_forward(self._model, self._data)

    def _apply_held_objects(self) -> None:
        """Pin objects a viewer is dragging back to the dragged pose."""
        if not self._held_objects:
            return
        import mujoco  # noqa: PLC0415

        for joint, (position, wxyz) in self._held_objects.items():
            self._write_object_pose(joint, position, wxyz)
        mujoco.mj_forward(self._model, self._data)

    def _publish_object_poses(self) -> None:
        """Snapshot free-object poses for the HTTP and viewer threads."""
        from physicalai_mujoco_so101_plugin.viser_controls import ObjectPose  # noqa: PLC0415

        poses: dict[str, ObjectPose] = {}
        if self._data is not None:
            qpos = self._data.qpos
            for joint, (qpos_addr, _) in self._free_joint_addrs.items():
                values = [float(v) for v in qpos[qpos_addr : qpos_addr + 7]]
                poses[joint] = ObjectPose(position=tuple(values[:3]), wxyz=tuple(values[3:]))  # type: ignore[arg-type]
        with self._state_lock:
            self._object_poses = poses

    def _sample_target_and_blocks(self, count: int) -> list[tuple[float, float]]:
        """Sample *count* block positions clear of the target and of each other.

        Returns:
            A list of *count* ``(x, y)`` positions.
        """
        if self._target_body_id is not None:
            target_xy = (
                float(self._data.xpos[self._target_body_id][0]),
                float(self._data.xpos[self._target_body_id][1]),
            )
        else:
            target_xy = self._spawn_center

        return sample_object_positions(
            count,
            rng=self._rng,
            center=self._spawn_center,
            min_r=self._spawn_min_r,
            max_r=self._spawn_max_r,
            angle_half_deg=self._spawn_angle_half_deg,
            target_xy=target_xy,
            target_min_sep=self._target_min_sep,
            object_min_sep=self._block_min_sep,
        )

    def _randomize_blocks(self) -> None:
        import mujoco  # noqa: PLC0415

        if not self._block_joint_addrs:
            return

        # Keep the green plate fixed; only respawn free objects.
        positions = self._sample_target_and_blocks(len(self._block_joint_addrs))

        for (qpos_addr, dof_addr), xy in zip(self._block_joint_addrs, positions, strict=True):
            write_freejoint_qpos(
                self._data,
                qpos_addr,
                dof_addr,
                xy,
                yaw=float(self._rng.uniform(0.0, 2.0 * np.pi)),
            )
        mujoco.mj_forward(self._model, self._data)

    def _render_cameras(self) -> None:
        now = time.monotonic()
        for config in self._cameras:
            renderer = self._camera_renderers.get(config.name)
            if renderer is None:
                continue

            period = 1.0 / config.fps
            if now - self._camera_last_frame_ts[config.name] < period:
                continue

            try:
                # pyrefly: ignore [missing-attribute]
                renderer.update_scene(self._data, camera=config.name)
                # mujoco.Renderer already returns the image upright (it flips the
                # OpenGL read-back itself), so the frame is used as rendered.
                # pyrefly: ignore [missing-attribute]
                frame = renderer.render()[:, :, :3]
            except (RuntimeError, ValueError) as exc:
                logger.debug("Camera render error for '{}': {}", config.name, exc)
                continue

            if config.mirror_horizontal:
                frame = frame[:, ::-1, :]
            frame = np.ascontiguousarray(frame)

            buffer = self._frame_buffers.get(config.name)
            if buffer is not None:
                buffer.put(frame)

            cam = self._camera_devices.get(config.name)
            if cam is not None:
                try:
                    # pyrefly: ignore [missing-attribute]
                    cam.schedule_frame(frame)
                except RuntimeError as exc:
                    logger.debug("Camera publish error for '{}': {}", config.name, exc)
                    continue

            self._camera_last_frame_ts[config.name] = now

    def _start_http_server(self) -> None:
        if self._http_port <= 0:
            return

        from physicalai_mujoco_so101_plugin.http_server import HttpServer, build_app  # noqa: PLC0415

        app = build_app(
            service_name=self._owner_name or "mujoco-so101",
            buffers=self._frame_buffers,
            commands=self._commands,
            get_status=self._http_status,
        )
        server = HttpServer(app, self._http_host, self._http_port)
        try:
            server.start()
        except (OSError, RuntimeError) as exc:
            logger.warning("Failed to start HTTP server on {}:{}: {}", self._http_host, self._http_port, exc)
            return
        self._http_server = server
        logger.info("HTTP camera server running at {}", server.url)

    def _stop_http_server(self) -> None:
        if self._http_server is None:
            return
        with contextlib.suppress(Exception):
            self._http_server.stop()
        self._http_server = None

    def _http_status(self) -> dict[str, object]:
        """Return the status payload served to HTTP clients.

        Called on the uvicorn thread, so it takes the state lock and reads each
        field once: a scene switch on the sim thread replaces the renderers, the
        scene id and the auto-reset helper as a group.

        Returns:
            A JSON-friendly snapshot of connection, scene and camera state.
        """
        from physicalai_mujoco_so101_plugin.scene_registry import list_scenes, list_scenes_for_arms  # noqa: PLC0415

        scenes = sorted(list_scenes())
        compatible = sorted(list_scenes_for_arms(self.NUM_ARMS))
        with self._state_lock:
            auto_reset = self._episode_auto_reset
            rendering = set(self._camera_renderers)
            return {
                "connected": self._model is not None,
                "scene": self._current_scene_id,
                "scenes": scenes,
                "compatible_scenes": compatible,
                "seed": self._seed,
                "episode": self._episode_status(auto_reset),
                "objects": [
                    {"joint": joint, "position": list(pose.position), "wxyz": list(pose.wxyz)}
                    for joint, pose in self._object_poses.items()
                ],
                "cameras": [
                    {
                        "name": config.name,
                        "width": config.width,
                        "height": config.height,
                        "fps": config.fps,
                        "device": config.device,
                        "rendering": config.name in rendering,
                    }
                    for config in self._cameras
                ],
            }

    def _submit_command(self, command: SimCommand) -> None:
        """Queue a command for the sim thread.

        Resolves the queue on every call: ``disconnect()`` replaces it, and a
        callback bound to the old queue would silently drop commands.
        """
        self._commands.put(command)

    def _drain_commands(self) -> None:
        while True:
            try:
                command = self._commands.get_nowait()
            except queue.Empty:
                return
            try:
                self._handle_command(command)
            except Exception as exc:  # noqa: BLE001
                # Operator commands are best-effort; a bad one must not stop the loop.
                logger.warning("Command {} failed: {}", type(command).__name__, exc)

    def _handle_command(self, command: SimCommand) -> None:
        from physicalai_mujoco_so101_plugin.http_server import (  # noqa: PLC0415
            HomeCommand,
            ResetCommand,
            SetAutoResetCommand,
            SetObjectPoseCommand,
            SetSeedCommand,
            ShutdownCommand,
            SwitchSceneCommand,
        )

        if isinstance(command, ResetCommand):
            logger.info("Scene reset requested")
            self._run_scene_reset()
        elif isinstance(command, SwitchSceneCommand):
            logger.info("Scene switch requested: {}", command.scene_id)
            try:
                self._switch_to_scene(command.scene_id)
            except Exception as exc:  # noqa: BLE001
                # An unknown id, or a scene XML MuJoCo refuses to compile, is a
                # bad request — not a reason to drop the simulation.
                logger.warning("Scene switch failed: {}", exc)
        elif isinstance(command, HomeCommand):
            self._go_home()
        elif isinstance(command, SetSeedCommand):
            self._set_seed(command.seed)
        elif isinstance(command, SetAutoResetCommand):
            self._set_auto_reset(enabled=command.enabled, dwell_s=command.dwell_s)
        elif isinstance(command, SetObjectPoseCommand):
            self._set_object_pose(command.joint, command.position, command.wxyz, hold=command.hold)
        elif isinstance(command, ShutdownCommand):
            logger.info("Shutdown requested")
            _signal_owner_shutdown()

    def get_observation(self) -> RobotObservation:
        """Return the current simulated joint observation.

        Raises:
            ConnectionError: If the robot is not connected.
        """
        if not self.is_connected():
            msg = "Robot is not connected. Call connect() first."
            raise ConnectionError(msg)

        self._check_pending_scene_switch()
        self._step_and_sync()

        positions = np.empty(self.NUM_JOINTS, dtype=np.float64)
        velocities = np.empty(self.NUM_JOINTS, dtype=np.float64)

        for i, name in enumerate(self.JOINT_ORDER):
            positions[i], velocities[i] = self._read_joint_state(name)

        if self._unit == "normalized":
            limits = self._require_joint_limits()
            _, width = _normalized_span(self.JOINT_ORDER)
            velocities *= width / (limits[:, 1] - limits[:, 0])
            positions = radians_to_normalized(positions, limits, self.JOINT_ORDER)
        else:
            positions = np.degrees(positions)
            velocities = np.degrees(velocities)

        return MuJoCoSO101Observation(
            joint_positions=positions.astype(np.float32),
            timestamp=time.monotonic(),
            sensor_data={"velocities": velocities.astype(np.float32)},
        )

    def send_action(self, action: np.ndarray, *, goal_time: float = 0.1) -> None:  # noqa: ARG002
        """Apply joint targets, in this robot's ``unit``, to the simulation actuators.

        Normalized targets are clamped to the normalized range, as on the
        SO101 driver.

        Raises:
            ConnectionError: If the robot is not connected.
            ValueError: If `action` does not match the robot joint count.
        """
        if not self.is_connected():
            msg = "Robot is not connected. Call connect() first."
            raise ConnectionError(msg)

        if action.shape != (self.NUM_JOINTS,):
            msg = f"Expected action shape ({self.NUM_JOINTS},), got {action.shape}"
            raise ValueError(msg)

        if self._unit == "normalized":
            targets = normalized_to_radians(
                np.asarray(action, dtype=np.float64),
                self._require_joint_limits(),
                self.JOINT_ORDER,
            )
        else:
            targets = np.radians(np.asarray(action, dtype=np.float64))
        for i in range(self.NUM_JOINTS):
            # pyrefly: ignore [missing-attribute]
            self._data.ctrl[self._ctrl_indices[i]] = float(targets[i])

    def _require_joint_limits(self) -> np.ndarray:
        if self._joint_limits is None:
            msg = "Robot is not connected. Call connect() first."
            raise ConnectionError(msg)
        return self._joint_limits

    def _joint_limits_for_joint_order(self, model: object) -> np.ndarray | None:
        """Read each public joint's range, the simulated counterpart of a calibrated range.

        Returns:
            ``(NUM_JOINTS, 2)`` lower/upper limits in radians, or ``None`` if a
            joint is missing or has no range.
        """
        import mujoco  # noqa: PLC0415

        limits = np.empty((self.NUM_JOINTS, 2), dtype=np.float64)
        for i, name in enumerate(self.JOINT_ORDER):
            joint_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name))
            if joint_id < 0:
                return None
            # pyrefly: ignore [missing-attribute]
            low, high = (float(value) for value in model.jnt_range[joint_id])
            if not high > low:
                logger.error("Joint {!r} has no range; normalized units need one", name)
                return None
            limits[i] = (low, high)
        return limits

    def render_camera(self, camera_name: str, width: int, height: int) -> np.ndarray:
        """Render an RGB image from a named camera.

        Returns:
            The rendered RGB image.

        Raises:
            ConnectionError: If the robot is not connected.
        """
        import mujoco  # noqa: PLC0415

        if not self.is_connected():
            msg = "Robot is not connected."
            raise ConnectionError(msg)

        renderer = mujoco.Renderer(self._model, height, width)
        renderer.update_scene(self._data, camera=camera_name)
        rgb = renderer.render()
        renderer.close()
        return rgb

    def _read_joint_state(self, name: str) -> tuple[float, float]:
        import mujoco  # noqa: PLC0415

        # pyrefly: ignore [missing-attribute]
        jnt_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jnt_id < 0:
            msg = f"Joint {name!r} not found in MuJoCo model"
            raise ValueError(msg)
        # pyrefly: ignore [missing-attribute]
        qpos_adr = self._model.jnt_qposadr[jnt_id]
        # pyrefly: ignore [missing-attribute]
        dof_adr = self._model.jnt_dofadr[jnt_id]
        # pyrefly: ignore [missing-attribute]
        return float(self._data.qpos[qpos_adr]), float(self._data.qvel[dof_adr])

    def __getstate__(self) -> dict:
        """Return serializable construction state."""
        return {
            "_model_path": self._model_path,
            "_substeps": self._substeps,
            "_enable_viewer": self._enable_viewer,
            "_cameras": [asdict(cam) for cam in self._cameras],
            "_free_joints": self._free_joints,
            "_target_body_name": self._target_body_name,
            "_spawn_center": self._spawn_center,
            "_spawn_min_r": self._spawn_min_r,
            "_spawn_max_r": self._spawn_max_r,
            "_spawn_angle_half_deg": self._spawn_angle_half_deg,
            "_block_min_sep": self._block_min_sep,
            "_target_min_sep": self._target_min_sep,
            "_current_scene_id": self._current_scene_id,
            "_owner_name": self._owner_name,
            "_http_host": self._http_host,
            "_http_port": self._http_port,
            "_viser_host": self._viser_host,
            "_viser_port": self._viser_port,
            "_unit": self._unit,
        }

    def __setstate__(self, state: dict) -> None:
        """Restore serializable construction state."""
        self._model_path = state["_model_path"]
        self._substeps = state["_substeps"]
        self._set_joint_unit(state.get("_unit", "normalized"))
        self._enable_viewer = state.get("_enable_viewer", False)
        self._cameras = [CameraConfig(**cam) for cam in state.get("_cameras", [])]
        self._free_joints = state.get("_free_joints", self.DEFAULT_BLOCK_FREEJOINTS)
        self._target_body_name = state.get("_target_body_name", self.DEFAULT_TARGET_BODY_NAME)
        self._spawn_center = state.get("_spawn_center", self.DEFAULT_SPAWN_CENTER)
        self._spawn_min_r = state.get("_spawn_min_r", self.DEFAULT_SPAWN_MIN_R)
        self._spawn_max_r = state.get("_spawn_max_r", self.DEFAULT_SPAWN_MAX_R)
        self._spawn_angle_half_deg = state.get("_spawn_angle_half_deg", self.DEFAULT_SPAWN_ANGLE_HALF_DEG)
        self._block_min_sep = state.get("_block_min_sep", self.DEFAULT_BLOCK_MIN_SEP)
        self._target_min_sep = state.get("_target_min_sep", self.DEFAULT_TARGET_MIN_SEP)
        self._current_scene_id = state.get("_current_scene_id")
        if self._current_scene_id:
            from physicalai_mujoco_so101_plugin.scene_registry import get_reset_fn  # noqa: PLC0415

            self._scene_on_reset = get_reset_fn(self._current_scene_id)
        else:
            self._scene_on_reset = None
        self._model = None
        self._data = None
        self._viser_server = None
        self._viser_scene = None
        self._native_viewer = None
        self._viser_port = state.get("_viser_port", 9090)
        self._owner_name = state.get("_owner_name", "")
        self._camera_devices = {}
        self._camera_renderers = {}
        self._camera_last_frame_ts = {}
        self._frame_buffers = {}
        self._commands = queue.Queue()
        self._http_host = state.get("_http_host", "127.0.0.1")
        self._http_port = state.get("_http_port", 0)
        self._http_server = None
        self._viser_host = state.get("_viser_host", "127.0.0.1")
        self._block_joint_addrs = []
        self._target_body_id = None
        self._episode_auto_reset = None
        self._last_sim_time = None
        self._rng = np.random.default_rng()
        self._pending_scene_switch = False
        self._ctrl_indices = ()
        self._joint_limits = None
        self._scene_xml_paths = None
        self._scene_xml_mtimes = {}
        self._scene_xml_next_check = 0.0
        self._viser_sync_failed = False
        self._state_lock = threading.RLock()
        self._init_control_state()


class BiMuJoCoSO101(MuJoCoSO101):
    """Bimanual SO-101 simulated with a single MuJoCo model.

    Runs both arms in one model with ``left_*`` then ``right_*`` joints
    (12 total). ``send_action`` writes into the model actuator array, so the
    dual-arm XML must declare its actuators in ``BIMANUAL_SO101_JOINT_ORDER``.
    """

    JOINT_ORDER: ClassVar[tuple[str, ...]] = BIMANUAL_SO101_JOINT_ORDER
    NUM_JOINTS: ClassVar[int] = BIMANUAL_NUM_JOINTS
    NUM_ARMS: ClassVar[int] = 2
