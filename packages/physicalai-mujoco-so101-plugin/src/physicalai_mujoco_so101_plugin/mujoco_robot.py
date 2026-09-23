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
from typing import TYPE_CHECKING, ClassVar

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

# Scene XML is polled for live camera edits; walking the include graph is far
# too expensive to do on every control cycle.
_SCENE_XML_POLL_INTERVAL_S = 1.0


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
    publishes frames there via pyfakewebcam.
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
        viser_port: int = 9090,
    ) -> None:
        """Initialize a disconnected simulation robot."""
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

        self._apply_scene_params(scene_config)

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
        if not self._model_is_drivable(model):
            msg = f"Model {self._model_path!r} does not provide the joints/actuators for {type(self).__name__}"
            raise ValueError(msg)
        # pyrefly: ignore [missing-attribute]
        data = mujoco.MjData(model)
        # pyrefly: ignore [missing-attribute]
        mujoco.mj_forward(model, data)
        if self._scene_on_reset is not None:
            self._scene_on_reset(model, data, self._rng)
        self._model = model
        self._data = data
        self._last_sim_time = float(self._data.time)
        self._init_block_joint_addrs()
        self._init_episode_auto_reset()
        self._reset_scene_xml_watch()
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
            self._target_body_id = None
            self._episode_auto_reset = None
            self._last_sim_time = None

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

        if self._viser_scene is not None:
            self._sync_viser()
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

        # pyrefly: ignore [missing-attribute]
        target_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, self._target_body_name)
        self._target_body_id = int(target_id) if target_id >= 0 else None

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
            success_dwell_s=5.0,
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
                    mat = np.empty(9, dtype=np.float64)
                    mat[:3] = vals[:3]
                    mat[3:6] = vals[3:]
                    mat[6:] = np.cross(vals[:3], vals[3:])
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

    def _model_is_drivable(self, model: object) -> bool:
        """Return whether *model* exposes every joint and actuator this robot drives.

        Returns:
            ``True`` when the model declares all of ``JOINT_ORDER`` and at least
            ``NUM_JOINTS`` actuators.
        """
        import mujoco  # noqa: PLC0415

        if int(model.nu) < self.NUM_JOINTS:
            return False
        return all(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) >= 0 for name in self.JOINT_ORDER)

    def _switch_to_scene(self, scene_id: str) -> bool:
        """Hot-swap the simulation to another registered scene.

        Returns:
            ``True`` when the scene was installed, ``False`` when it was missing
            or incompatible with this robot (the current scene is kept).
        """
        import mujoco  # noqa: PLC0415

        from physicalai_mujoco_so101_plugin.scene_registry import get_reset_fn, get_scene  # noqa: PLC0415

        scene = get_scene(scene_id)
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
        if not self._model_is_drivable(new_model):
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

            self._recreate_viser_scene()
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
            import rich  # noqa: PLC0415
            import viser  # noqa: PLC0415
            from mjviser import ViserMujocoScene  # noqa: PLC0415
        except ImportError as exc:
            logger.warning("mjviser/viser unavailable, web viewer disabled: {}", exc)
            return False

        # SharedRobot's owner worker closes stdout after its READY handshake.
        # Viser prints from its background thread during stop(), so give Rich a
        # stream that remains open throughout owner teardown.
        rich.get_console().file = sys.stderr

        server = None
        try:
            server = viser.ViserServer(port=self._viser_port, verbose=False)
            scene = ViserMujocoScene(server, self._model, num_envs=1)
            scene.create_visualization_gui()
            self._add_viser_control_gui(server, viser)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to start viser viewer on port {}: {}", self._viser_port, exc)
            if server is not None:
                with contextlib.suppress(Exception):
                    server.stop()
            return False
        else:
            self._viser_server = server
            self._viser_scene = scene
            logger.info("3D viewer: http://127.0.0.1:{}", self._viser_port)
            return True

    def _add_viser_control_gui(self, server: object, viser_module: object) -> None:
        """Add Reset/Shutdown controls, independent of the per-scene GUI tabs.

        Placed outside the tab group created by ``create_visualization_gui``
        so a scene hot-swap (which rebuilds that tab group from scratch) does
        not duplicate these controls.
        """
        from physicalai_mujoco_so101_plugin.http_server import ResetCommand, ShutdownCommand  # noqa: PLC0415

        reset_button = server.gui.add_button("Reset Scene", icon=viser_module.Icon.REFRESH)

        @reset_button.on_click
        def _on_reset(_: object) -> None:
            self._commands.put(ResetCommand())

        shutdown_button = server.gui.add_button("Shutdown", icon=viser_module.Icon.POWER, color="red")

        @shutdown_button.on_click
        def _on_shutdown_click(event: object) -> None:
            client = event.client
            if client is None:
                return
            with client.gui.add_modal("Confirm shutdown") as modal:
                client.gui.add_markdown("Stop the simulation owner? This disconnects all viewers.")
                confirm_button = client.gui.add_button("Shutdown", color="red")
                cancel_button = client.gui.add_button("Cancel")

                @confirm_button.on_click
                def _on_confirm(_: object) -> None:
                    self._commands.put(ShutdownCommand())
                    modal.close()

                @cancel_button.on_click
                def _on_cancel(_: object) -> None:
                    modal.close()

    def _recreate_viser_scene(self) -> None:
        """Rebuild the viser scene after a model hot-swap.

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
            from mjviser import ViserMujocoScene  # noqa: PLC0415

            scene = ViserMujocoScene(self._viser_server, self._model, num_envs=1)
            scene.create_visualization_gui()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to recreate viser scene: {}", exc)
            return
        self._viser_scene = scene

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

    def _sync_viser_fixed_bodies(self) -> None:
        """Push current ``data.xpos`` into mjviser's fixed-geometry handles.

        The scene XML watch (see ``_reset_scene_xml_watch``) can move fixed
        world bodies — e.g. camera rigs — by writing ``model.body_pos``
        outside a normal sim step. MuJoCo cameras pick that up after
        ``mj_forward``, but mjviser only places fixed meshes at create time,
        so it needs this explicit resync to reflect the change.

        Handles live under ``/fixed_bodies``, whose frame already carries
        mjviser's camera-tracking offset, so store world ``xpos`` as-is.
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

        from physicalai_mujoco_so101_plugin.scene_registry import list_scenes  # noqa: PLC0415

        try:
            scene_ids = list(list_scenes().keys())
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
            if self._scene_on_reset is not None:
                self._scene_on_reset(self._model, self._data, self._rng)
            else:
                self._randomize_blocks()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Scene reset failed: {}", exc)
            return
        if self._episode_auto_reset is not None:
            self._episode_auto_reset.notify_manual_reset()

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
                # pyrefly: ignore [missing-attribute]
                frame = renderer.render()[:, :, :3][::-1, :, :]
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
        from physicalai_mujoco_so101_plugin.scene_registry import list_scenes  # noqa: PLC0415

        scenes = sorted(list_scenes())
        with self._state_lock:
            auto_reset = self._episode_auto_reset
            rendering = set(self._camera_renderers)
            return {
                "connected": self._model is not None,
                "scene": self._current_scene_id,
                "scenes": scenes,
                "episode": auto_reset.status() if auto_reset is not None else {"enabled": False},
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

    def _drain_commands(self) -> None:
        while True:
            try:
                command = self._commands.get_nowait()
            except queue.Empty:
                return
            self._handle_command(command)

    def _handle_command(self, command: SimCommand) -> None:
        from physicalai_mujoco_so101_plugin.http_server import (  # noqa: PLC0415
            ResetCommand,
            ShutdownCommand,
            SwitchSceneCommand,
        )

        if isinstance(command, ResetCommand):
            logger.info("Scene reset requested via HTTP")
            self._run_scene_reset()
        elif isinstance(command, SwitchSceneCommand):
            logger.info("Scene switch requested via HTTP: {}", command.scene_id)
            try:
                self._switch_to_scene(command.scene_id)
            except Exception as exc:  # noqa: BLE001
                # An unknown id, or a scene XML MuJoCo refuses to compile, is a
                # bad request — not a reason to drop the simulation.
                logger.warning("Scene switch failed: {}", exc)
        elif isinstance(command, ShutdownCommand):
            logger.info("Shutdown requested via HTTP")
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
            pos, vel = self._read_joint_state(name)
            positions[i] = np.degrees(pos)
            velocities[i] = np.degrees(vel)

        return MuJoCoSO101Observation(
            joint_positions=positions.astype(np.float32),
            timestamp=time.monotonic(),
            sensor_data={"velocities": velocities.astype(np.float32)},
        )

    def send_action(self, action: np.ndarray, *, goal_time: float = 0.1) -> None:  # noqa: ARG002
        """Apply joint-angle targets to the simulation actuators.

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

        for i in range(self.NUM_JOINTS):
            # pyrefly: ignore [missing-attribute]
            self._data.ctrl[i] = float(np.radians(action[i]))

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
            "_viser_port": self._viser_port,
        }

    def __setstate__(self, state: dict) -> None:
        """Restore serializable construction state."""
        self._model_path = state["_model_path"]
        self._substeps = state["_substeps"]
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
        self._block_joint_addrs = []
        self._target_body_id = None
        self._episode_auto_reset = None
        self._last_sim_time = None
        self._rng = np.random.default_rng()
        self._pending_scene_switch = False
        self._scene_xml_paths = None
        self._scene_xml_mtimes = {}
        self._scene_xml_next_check = 0.0
        self._viser_sync_failed = False
        self._state_lock = threading.RLock()


class BiMuJoCoSO101(MuJoCoSO101):
    """Bimanual SO-101 simulated with a single MuJoCo model.

    Runs both arms in one model with ``left_*`` then ``right_*`` joints
    (12 total). ``send_action`` writes into the model actuator array, so the
    dual-arm XML must declare its actuators in ``BIMANUAL_SO101_JOINT_ORDER``.
    """

    JOINT_ORDER: ClassVar[tuple[str, ...]] = BIMANUAL_SO101_JOINT_ORDER
    NUM_JOINTS: ClassVar[int] = BIMANUAL_NUM_JOINTS
