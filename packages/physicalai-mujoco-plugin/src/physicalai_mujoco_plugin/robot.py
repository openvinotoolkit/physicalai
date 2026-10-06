# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""``MuJoCoRobot``: any MuJoCo Menagerie robot, simulated through the Runtime ``Robot`` protocol.

The driver composes the profile's robot into a scene, converts actions and observations through
:class:`~physicalai_mujoco_plugin.channels.ArmChannels`, and steps physics on the owner's control
tick inside :meth:`MuJoCoRobot.get_observation`; there is no physics thread. Cameras, the viewer
and the HTTP control server are services the driver owns but does not implement
(``cameras.py``, ``viewer.py``, ``http_server.py``). Operator commands from those services are
queued and applied on the sim thread at the start of the next tick.
"""

# MuJoCo model/data attributes are supplied by the C extension at runtime.
# pyrefly: ignore-errors [missing-attribute]

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, get_args

import numpy as np
from loguru import logger

from physicalai.config import export_config
from physicalai_mujoco_plugin.camera_thread import RateMeter
from physicalai_mujoco_plugin.cameras import CameraConfig, CameraService
from physicalai_mujoco_plugin.channels import TorqueMode
from physicalai_mujoco_plugin.control import OperatorControls
from physicalai_mujoco_plugin.profiles import DefaultUnit, RobotProfile, get_profile
from physicalai_mujoco_plugin.scene_objects import SceneObjects, SpawnArea
from physicalai_mujoco_plugin.sim import Sim, default_cameras, joint_names_before_load, load_sim, resolve_scene
from physicalai_mujoco_plugin.studio_recorder import DEFAULT_STUDIO_URL

if TYPE_CHECKING:
    from physicalai.capture.frame import Frame
    from physicalai_mujoco_plugin.conveyor_automation import ConveyorAutomation
    from physicalai_mujoco_plugin.http_server import HttpServer, SimCommand
    from physicalai_mujoco_plugin.scene_registry import SceneConfig
    from physicalai_mujoco_plugin.scene_watch import SceneXmlWatcher
    from physicalai_mujoco_plugin.viewer import ViewerService

_DEFAULT_SUCCESS_DWELL_S = 5.0


@dataclass
class MuJoCoObservation:
    """Observation returned by :class:`MuJoCoRobot`."""

    joint_positions: np.ndarray
    timestamp: float
    sensor_data: dict[str, np.ndarray] | None = None
    images: dict[str, Frame] | None = None

    @property
    def state(self) -> np.ndarray:
        """Joint positions, as on the real fixed-base drivers (OBS-2)."""
        return self.joint_positions


@export_config(class_path="physicalai_mujoco_plugin.robot.MuJoCoRobot")
class MuJoCoRobot(OperatorControls):
    """A MuJoCo Menagerie robot in a scene, with cameras, a viewer and an HTTP control server."""

    def __init__(  # noqa: PLR0913, PLR0915
        self,
        profile: str = "so101",
        *,
        scene: str | None = None,
        model_path: str | None = None,
        unit: DefaultUnit | None = None,
        torque_mode: TorqueMode = "pd",
        substeps: int | None = None,
        rate_hz: float = 50.0,
        cameras: list[CameraConfig | dict] | None = None,
        enable_viewer: bool = False,
        viser_host: str = "127.0.0.1",
        viser_port: int = 9090,
        http_host: str = "127.0.0.1",
        http_port: int = 0,
        owner_name: str = "",
        studio_url: str = DEFAULT_STUDIO_URL,
        seed: int | None = None,
    ) -> None:
        """Create a disconnected simulation.

        Args:
            profile: Robot profile (``so101``, ``ur5e``) or any MuJoCo Menagerie model name.
            scene: Registered scene id; ``None`` uses ``model_path``, else the profile's default scene.
            model_path: Scene XML to load instead of the registered scene's. Its ``robot_mount``
                frames get the profile's robot; an XML without them is used as is.
            unit: ``normalized`` or ``degrees``; ``None`` uses the profile's default.
            torque_mode: ``pd`` tracks position targets on torque actuators; ``raw`` passes ``ctrl``.
            substeps: Physics steps per control tick; ``None`` keeps real time at ``rate_hz``.
            rate_hz: The owner's control rate, used to derive ``substeps``.
            cameras: Camera streams; ``None`` streams the robot cameras and ``overview``.
            enable_viewer: Open the viser viewer (or the native viewer where viser is unavailable).
            viser_host: viser bind address.
            viser_port: viser port.
            http_host: HTTP camera and control server bind address.
            http_port: HTTP port; ``0`` disables the server.
            owner_name: The shared-robot owner name, used to find its Studio session.
            studio_url: Studio backend that automatic episode recording attaches to.
            seed: Fixed seed for scene resets; ``None`` randomizes.

        Raises:
            ValueError: If ``unit``, ``torque_mode``, ``rate_hz`` or ``substeps`` is invalid.
        """
        if unit is not None and unit not in get_args(DefaultUnit):
            msg = f"Unsupported unit {unit!r}; expected one of {get_args(DefaultUnit)}"
            raise ValueError(msg)
        if torque_mode not in get_args(TorqueMode):
            msg = f"Unsupported torque_mode {torque_mode!r}; expected one of {get_args(TorqueMode)}"
            raise ValueError(msg)
        if not rate_hz > 0 or (substeps is not None and substeps < 1):
            msg = f"rate_hz must be positive and substeps at least 1, got {rate_hz!r} and {substeps!r}"
            raise ValueError(msg)
        self._recipe: dict[str, object] = {
            "profile": profile, "scene": scene, "model_path": model_path, "unit": unit, "torque_mode": torque_mode,
            "substeps": substeps, "rate_hz": rate_hz, "cameras": cameras, "enable_viewer": enable_viewer,
            "viser_host": viser_host, "viser_port": viser_port, "http_host": http_host, "http_port": http_port,
            "owner_name": owner_name, "studio_url": studio_url, "seed": seed,
        }  # fmt: skip
        """Constructor arguments, for pickling (``__getstate__``)."""
        self._profile: RobotProfile = get_profile(profile)
        self._unit: DefaultUnit = unit or self._profile.default_unit
        self._torque_mode: TorqueMode = torque_mode
        self._scene_arg = scene
        self._model_path = model_path
        self._substeps_arg = substeps
        self._substeps = substeps or 1
        self._rate_hz = float(rate_hz)
        self._camera_arg = (
            None if cameras is None else [c if isinstance(c, CameraConfig) else CameraConfig(**c) for c in cameras]
        )
        self._enable_viewer = enable_viewer
        self._viser_host, self._viser_port = viser_host, viser_port
        self._http_host, self._http_port = http_host, http_port
        self._owner_name = owner_name
        self._studio_url = studio_url
        self._seed = seed
        self._rng = np.random.default_rng()
        self._reseed_if_fixed()
        self._state_lock = threading.RLock()
        """Guards state the HTTP and viewer threads read while the sim thread replaces it."""
        self._commands: queue.Queue[SimCommand] = queue.Queue()
        self._sim: Sim | None = None
        self._scene_config = self._initial_scene = resolve_scene(self._profile, scene, model_path)
        self._area = SpawnArea() if self._scene_config is None else SpawnArea.from_scene(self._scene_config)
        self._objects = SceneObjects(self._area, self._state_lock)
        self._cameras = CameraService(self._camera_arg or [])
        self._viewer: ViewerService | None = None
        self._http_server: HttpServer | None = None
        self._watcher: SceneXmlWatcher | None = None
        self._automation: ConveyorAutomation | None = None
        self._episode_auto_reset: object | None = None
        self._auto_reset_active = True
        self._auto_reset_dwell_s = _DEFAULT_SUCCESS_DWELL_S
        self._pending_scene_switch = False
        self._last_sim_time: float | None = None
        self._last_tick_sim_time: float | None = None
        self._tick_meter = RateMeter()
        self._ignored_action_logged = False
        self._invalid_action_logged_at = 0.0
        self._names: list[str] | None = None

    # ------------------------------------------------------------------
    # Robot protocol
    # ------------------------------------------------------------------

    @property
    def profile(self) -> RobotProfile:
        """The robot profile."""
        return self._profile

    @property
    def joint_names(self) -> list[str]:
        """Public channel names in action order; known before ``connect()`` (DRV-6)."""
        if self._sim is not None:
            return self._sim.joint_names
        if self._names is None:
            self._names = joint_names_before_load(self._profile, self._xml_path(self._scene_config))
        return list(self._names)

    @property
    def device_ids(self) -> tuple[str, ...]:
        """Exclusively owned hardware; a simulation owns none."""
        return ()

    def is_connected(self) -> bool:
        """Return whether the simulation is loaded."""
        return self._sim is not None

    def connect(self) -> None:
        """Compose the scene, bind the channels, home the robot and start the services."""
        if self._sim is not None:
            return
        from physicalai_mujoco_plugin.conveyor_automation import ConveyorAutomation  # noqa: PLC0415

        xml_path = self._xml_path(self._scene_config)
        logger.info("Loading MuJoCo scene {} with the {} robot", xml_path, self._profile.name)
        sim = self._load(xml_path, self._scene_config)
        if self._automation is None:
            self._automation = ConveyorAutomation(
                self._studio_url,
                joint_names=tuple(sim.joint_names),
                unit=self._unit,
                to_units=self._targets_to_units,
                owner_name=lambda: self._owner_name,
            )
        self._sim = sim
        if self._substeps_arg is None:
            self._substeps = max(1, round(1.0 / (self._rate_hz * float(sim.model.opt.timestep))))
        self._last_sim_time = float(sim.data.time)
        self._bind_scene()
        logger.info(
            "MuJoCo {} connected ({} joints, timestep={})",
            self._profile.display_name,
            len(self.joint_names),
            sim.model.opt.timestep,
        )
        if self._enable_viewer:
            from physicalai_mujoco_plugin.viewer import ViewerService  # noqa: PLC0415

            self._viewer = ViewerService(
                host=self._viser_host,
                port=self._viser_port,
                submit_command=self._submit_command,
                panel_state=self._panel_state,
                key_callback=self._key_callback,
            )
            if not self._viewer.open(sim.model, sim.data):
                self._viewer, self._enable_viewer = None, False
        if self._camera_arg is None:
            self._cameras.configs = default_cameras(sim)
        self._cameras.start(sim.model, sim.data)
        self._start_http_server()

    def disconnect(self) -> None:
        """Stop the services and release the simulation; repeated calls are harmless."""
        self._stop_http_server()
        self._cameras.close()
        if self._automation is not None:
            self._automation.close()
        with self._state_lock:
            self._objects.clear()
            self._episode_auto_reset = None
            self._last_sim_time = None
            if self._viewer is not None:
                self._viewer.close()
                self._viewer = None
            self._sim = None
            # Keep the selected scene and its reset recipe for reconnect.
            self._pending_scene_switch = False
            self._commands = queue.Queue()
            self._watcher = None
        logger.info("MuJoCo {} disconnected", self._profile.display_name)

    def get_observation(self) -> MuJoCoObservation:
        """Advance physics by one control tick and return the robot's joints.

        Returns:
            Joint positions and velocities in this robot's units.

        Raises:
            ConnectionError: If the robot is not connected.
        """
        if self._sim is None:
            msg = "Robot is not connected. Call connect() first."
            raise ConnectionError(msg)
        self._check_pending_scene_switch()
        self._step_and_sync()
        sim = self._sim
        positions = np.concatenate([channels.read_positions() for channels in sim.channels])
        velocities = np.concatenate([channels.read_velocities() for channels in sim.channels])
        return MuJoCoObservation(
            joint_positions=positions.astype(np.float32),
            timestamp=time.monotonic(),
            sensor_data={"velocities": velocities.astype(np.float32)},
        )

    def send_action(self, action: np.ndarray, *, goal_time: float = 0.1) -> None:  # noqa: ARG002
        """Set joint targets, in this robot's units, for the next ticks.

        A non-finite action is discarded and logged at most once per second (CHN-7). While the
        conveyor autopilot drives the arm, actions are ignored.

        Raises:
            ConnectionError: If the robot is not connected.
            ValueError: If `action` does not have one value per joint.
        """
        sim = self._sim
        if sim is None:
            msg = "Robot is not connected. Call connect() first."
            raise ConnectionError(msg)
        values = np.asarray(action, dtype=np.float64)
        count = sum(len(channels) for channels in sim.channels)
        if values.shape != (count,):
            msg = f"Expected action shape ({count},), got {values.shape}"
            raise ValueError(msg)
        if not np.isfinite(values).all():
            now = time.monotonic()
            if now - self._invalid_action_logged_at >= 1.0:
                self._invalid_action_logged_at = now
                logger.warning("Discarding an action with non-finite values")
            return
        if self._automation is not None and self._automation.drives_arm:
            if not self._ignored_action_logged:
                logger.info("The autopilot drives the arm; ignoring actions until it is switched off")
                self._ignored_action_logged = True
            return
        self._ignored_action_logged = False
        start = 0
        for channels in sim.channels:
            channels.write(values[start : start + len(channels)])
            start += len(channels)

    def render_camera(self, camera_name: str, width: int, height: int) -> np.ndarray:
        """Render an RGB image from a named camera.

        Returns:
            The rendered RGB image.

        Raises:
            ConnectionError: If the robot is not connected.
        """
        import mujoco  # noqa: PLC0415

        if self._sim is None:
            msg = "Robot is not connected."
            raise ConnectionError(msg)
        with mujoco.Renderer(self._sim.model, height, width) as renderer:
            renderer.update_scene(self._sim.data, camera=camera_name)
            return renderer.render()

    # ------------------------------------------------------------------
    # Scene loading (sim.py) and switching
    # ------------------------------------------------------------------

    def _xml_path(self, scene: SceneConfig | None) -> Path:
        """Return the XML to load for *scene*.

        ``model_path`` replaces the XML of the constructor's scene only, not of scenes switched to.

        Returns:
            The scene XML path.
        """
        if self._model_path is not None and scene is self._initial_scene:
            return Path(self._model_path)
        return scene.scene_xml_path  # type: ignore[union-attr]

    def _load(self, xml_path: Path, scene: SceneConfig | None) -> Sim:
        """Load a simulation with this driver's settings, without touching the running one.

        Returns:
            The new simulation.
        """
        # Scene switches keep the robot count: the transport advertised the joint names once.
        robots = len(self._sim.bindings) if self._sim is not None else None
        return load_sim(
            xml_path,
            scene,
            self._profile,
            unit=self._unit,
            torque_mode=self._torque_mode,
            rng=self._rng,
            reseed=self._reseed_if_fixed,
            robots=robots,
        )

    def _bind_scene(self) -> None:
        """Bind the scene objects, episode automation and live XML watch to the current simulation."""
        from physicalai_mujoco_plugin.scene_watch import SceneXmlWatcher  # noqa: PLC0415

        sim = self._sim
        grippers = [f"{binding.prefix}gripper" for binding in sim.bindings]  # type: ignore[union-attr]
        self._objects.bind(sim.model, follow_bodies=grippers)  # type: ignore[union-attr]
        self._init_episode_auto_reset()
        self._watcher = SceneXmlWatcher(sim.xml_path)  # type: ignore[union-attr]
        self._objects.publish(sim.data)  # type: ignore[union-attr]

    def _switch_to_scene(self, scene_id: str) -> bool:
        """Hot-swap the simulation to another registered scene (DRV-7).

        Returns:
            ``True`` when the scene was installed, ``False`` when it was missing or incompatible
            (the current scene is kept).
        """
        from physicalai_mujoco_plugin.scene_registry import get_scene  # noqa: PLC0415

        scene = get_scene(scene_id)
        arms = self._robot_count()
        if not scene.supports(self._profile) or (arms is not None and scene.num_arms != arms):
            logger.error(
                "Scene '{}' does not support {} with {} robot(s); keeping scene '{}'",
                scene_id,
                self._profile.name,
                arms,
                self._current_scene_id,
            )
            return False
        if not scene.scene_xml_path.exists():
            logger.error("Scene XML not found: {}", scene.scene_xml_path)
            return False
        try:
            # Prepare the replacement before retiring the current scene: a failure must leave
            # the running model and its renderers usable.
            sim = self._load(scene.scene_xml_path, scene)
        except ValueError as exc:
            logger.error(
                "Scene '{}' cannot run {}: {}; keeping scene '{}'",
                scene_id,
                self._profile.name,
                exc,
                self._current_scene_id,
            )
            return False
        running = self._sim.joint_names if self._sim is not None else None
        if running is not None and sim.joint_names != running:
            # The transport advertised the joint names on connect; actions and observations must keep them.
            logger.error(
                "Scene '{}' would change the joint names from {} to {}; keeping scene '{}'",
                scene_id,
                running,
                sim.joint_names,
                self._current_scene_id,
            )
            return False

        # The camera thread renders the old model; stop it before taking the lock.
        self._cameras.stop()
        with self._state_lock:
            self._sim = sim
            self._scene_config = scene
            self._area = SpawnArea.from_scene(scene)
            self._objects.area = self._area
            self._last_sim_time = None
            self._bind_scene()
            if self._camera_arg is None:
                self._cameras.configs = default_cameras(sim)
            self._cameras.start(sim.model, sim.data)
            # Rebuild last: the viewer panel renders the new scene's state.
            if self._viewer is not None:
                self._viewer.rebuild(sim.model, sim.data)
        logger.info(
            "Switched to scene '{}' ({} bodies, {} geoms, {} joints)",
            scene_id,
            sim.model.nbody,
            sim.model.ngeom,
            sim.model.njnt,
        )
        return True

    def _check_pending_scene_switch(self) -> None:
        """Switch to the next compatible scene after the native viewer's ``N`` key."""
        if not self._pending_scene_switch:
            return
        self._pending_scene_switch = False
        try:
            scene_ids = list(self._compatible_scenes())
            current = self._current_scene_id
            index = scene_ids.index(current) if current is not None and current in scene_ids else 0
            for offset in range(1, len(scene_ids) + 1):
                candidate = scene_ids[(index + offset) % len(scene_ids)]
                if candidate == current:
                    break
                if self._switch_to_scene(candidate):
                    return
            logger.warning("No other scene is compatible with {}", self._profile.name)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to switch scene: {}", exc)

    def _compatible_scenes(self) -> dict[str, SceneConfig]:
        from physicalai_mujoco_plugin.scene_registry import list_scenes_for  # noqa: PLC0415

        arms = self._robot_count()
        return list_scenes_for(self._profile, arms)

    def _robot_count(self) -> int | None:
        """Return the number of attached robots: the running scene's, else the configured scene's.

        Returns:
            The robot count, or ``None`` for a custom model that is not loaded yet.
        """
        if self._sim is not None:
            return len(self._sim.bindings)
        return self._scene_config.num_arms if self._scene_config is not None else None

    def _key_callback(self, key: int) -> None:
        if key in {ord("n"), ord("N")}:
            self._pending_scene_switch = True

    def __getstate__(self) -> dict[str, object]:
        """Pickle the constructor recipe only; an unpickled robot is disconnected.

        Returns:
            The constructor arguments.
        """
        return dict(self._recipe)

    def __setstate__(self, state: dict[str, object]) -> None:
        """Rebuild a disconnected robot from its constructor arguments."""
        type(self).__init__(self, **state)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # The control tick
    # ------------------------------------------------------------------

    def _step_and_sync(self) -> None:
        import mujoco  # noqa: PLC0415

        self._drain_commands()
        sim = self._sim
        model, data = sim.model, sim.data  # type: ignore[union-attr]
        if self._watcher is not None and self._watcher.changed():
            self._apply_scene_edit(model, data)
        control_dt = self._substeps * float(model.opt.timestep)
        self._automation.tick(model, data, control_dt, self._arm_targets())  # type: ignore[union-attr]
        for _ in range(self._substeps):
            for channels in sim.channels:  # type: ignore[union-attr]
                channels.apply_pd()
            mujoco.mj_step(model, data)
        self._record_tick(float(data.time))
        if self._episode_auto_reset is not None:
            self._episode_auto_reset.update(model, data)
        self._objects.apply_held(model, data)
        self._objects.publish(data)
        if self._viewer is not None:
            if not self._viewer.sync(data):
                self._viewer, self._enable_viewer = None, False
            elif self._viewer.native is not None:
                self._handle_viewer_reset()
        self._cameras.publish(data)

    def _apply_scene_edit(self, model: object, data: object) -> None:
        """Apply live scene XML camera edits with no camera thread reading the model."""
        # MuJoCo shares a model between threads only while it is read-only; the camera thread
        # runs mj_fwdPosition and renders from this model, so it pauses for the edit.
        restart = self._cameras.on_thread
        if restart:
            self._cameras.stop()
        try:
            self._watcher.apply(model, data)  # type: ignore[union-attr]
        finally:
            if restart:
                self._cameras.start(model, data)

    def _record_tick(self, sim_time: float) -> None:
        if self._last_tick_sim_time is not None and sim_time < self._last_tick_sim_time:
            self._tick_meter.reset()  # the clock restarted (new model)
        self._last_tick_sim_time = sim_time
        self._tick_meter.add(sim_time)

    def _arm_targets(self) -> np.ndarray:
        """Return the current target of every channel in model units (radians for hinges).

        Returns:
            One target per public channel.
        """
        return np.concatenate([channels.model_targets() for channels in self._sim.channels])  # type: ignore[union-attr]

    def _targets_to_units(self, model_values: np.ndarray) -> np.ndarray:
        """Convert model-unit targets of every channel to public units (the virtual leader's pose).

        Returns:
            The targets in public units.
        """
        parts, start = [], 0
        for channels in self._sim.channels:  # type: ignore[union-attr]
            parts.append(channels.targets_to_public(model_values[start : start + len(channels)]))
            start += len(channels)
        return np.concatenate(parts)

    def _handle_viewer_reset(self) -> None:
        """Randomize the scene when the native viewer's reset button rewound the clock."""
        current = float(self._sim.data.time)  # type: ignore[union-attr]
        if self._last_sim_time is not None and current + 1e-9 < self._last_sim_time:
            logger.info("Viewer reset detected; randomizing")
            self._run_scene_reset()
            if self._viewer is not None and self._viewer.native_running():
                self._viewer.native_sync()
            current = float(self._sim.data.time)  # type: ignore[union-attr]
        self._last_sim_time = current


__all__ = ["MuJoCoObservation", "MuJoCoRobot"]
