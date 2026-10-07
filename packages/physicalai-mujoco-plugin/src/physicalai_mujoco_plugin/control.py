# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Operator controls of ``MuJoCoRobot``: commands from the viewer panel and HTTP, and their status.

The viewer and HTTP threads only enqueue a ``SimCommand``; :meth:`OperatorControls._drain_commands`
applies them on the sim thread at the start of the next tick. Readouts for those threads
(``_http_status``, ``_panel_state``) are snapshots taken under ``_state_lock``.

A replay (``POST /replay``) sets the robot's joints from recorded positions on each tick instead
of stepping physics, and ignores actions. ``POST /replay/stop`` restores the physics state from
before the replay, so the simulation continues as if the replay never ran. A reset, home or object
pose command ends the replay the same way before it applies; a scene switch drops it with the old
model.
"""

# MuJoCo model/data attributes are supplied by the C extension at runtime.
# pyrefly: ignore-errors [missing-attribute]

from __future__ import annotations

import contextlib
import queue
import sys
import threading
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from physicalai_mujoco_plugin.camera_thread import RateMeter
    from physicalai_mujoco_plugin.cameras import CameraService
    from physicalai_mujoco_plugin.conveyor_automation import ConveyorAutomation
    from physicalai_mujoco_plugin.floating import BaseStatus
    from physicalai_mujoco_plugin.http_server import HttpServer, ReplayCommand, SimCommand
    from physicalai_mujoco_plugin.profiles import RobotProfile
    from physicalai_mujoco_plugin.replay import Replay
    from physicalai_mujoco_plugin.scene_objects import SceneObjects, SpawnArea
    from physicalai_mujoco_plugin.scene_registry import SceneConfig
    from physicalai_mujoco_plugin.sim import Sim
    from physicalai_mujoco_plugin.viewer import ViewerService
    from physicalai_mujoco_plugin.viser_controls import PanelState


def _signal_owner_shutdown() -> None:
    """Request a graceful exit of the shared owner loop, if this runs inside one.

    The driver runs inside ``python -m physicalai.robot.transport._owner_worker``, where the
    worker module is loaded as ``__main__``; check both module identities for its shutdown event.
    """
    for module_name in ("__main__", "physicalai.robot.transport._owner_worker"):
        module = sys.modules.get(module_name)
        event = getattr(module, "shutdown", None) if module is not None else None
        if isinstance(event, threading.Event):
            event.set()
            return
    logger.warning("Owner shutdown event not found; use Ctrl+C or stop the process to exit")


class OperatorControls:
    """Commands and status of the simulation's operator surfaces (viewer panel, HTTP).

    A base class of :class:`~physicalai_mujoco_plugin.robot.MuJoCoRobot`; it owns no state of its
    own and works on the driver's attributes declared below.
    """

    _profile: RobotProfile
    _sim: Sim | None
    _scene_config: SceneConfig | None
    _area: SpawnArea
    _objects: SceneObjects
    _cameras: CameraService
    _automation: ConveyorAutomation | None
    _episode_auto_reset: object | None
    _auto_reset_active: bool
    _auto_reset_dwell_s: float
    _commands: queue.Queue[SimCommand]
    _state_lock: threading.RLock
    _rng: np.random.Generator
    _seed: int | None
    _tick_meter: RateMeter
    _http_server: HttpServer | None
    _http_host: str
    _http_port: int
    _owner_name: str
    _base_status: tuple[BaseStatus, ...]
    _viewer: ViewerService | None
    _replay: Replay | None

    @property
    def joint_names(self) -> list[str]:
        """Public joint names; implemented by the driver."""
        raise NotImplementedError

    @property
    def _model(self) -> object | None:
        return self._sim.model if self._sim is not None else None

    @property
    def _data(self) -> object | None:
        return self._sim.data if self._sim is not None else None

    @property
    def _current_scene_id(self) -> str | None:
        return self._scene_config.scene_id if self._scene_config is not None else None

    def _compatible_scenes(self) -> dict[str, SceneConfig]:
        """Scenes the running robot can switch to; implemented by the driver."""
        raise NotImplementedError

    def _switch_to_scene(self, scene_id: str) -> bool:
        """Install another scene; implemented by the driver."""
        raise NotImplementedError

    def _submit_command(self, command: SimCommand) -> None:
        """Queue a command for the sim thread; ``disconnect()`` replaces the queue."""
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
        from physicalai_mujoco_plugin import http_server as cmd  # noqa: PLC0415

        if self._handle_replay_command(command):
            return
        sim = self._sim
        if isinstance(command, cmd.ResetCommand):
            logger.info("Reset requested")
            self._reset()
        elif isinstance(command, cmd.SwitchSceneCommand):
            logger.info("Scene switch requested: {}", command.scene_id)
            try:
                self._switch_to_scene(command.scene_id)
            except Exception as exc:  # noqa: BLE001
                # An unknown id, or a scene XML MuJoCo refuses to compile, is a bad request,
                # not a reason to drop the simulation.
                logger.warning("Scene switch failed: {}", exc)
        elif isinstance(command, cmd.HomeCommand):
            self._go_home()
        elif isinstance(command, cmd.SetSeedCommand):
            self._set_seed(command.seed)
        elif isinstance(command, cmd.SetAutoResetCommand):
            self._set_auto_reset(enabled=command.enabled, dwell_s=command.dwell_s)
        elif isinstance(command, cmd.SetBeltSpeedCommand):
            self._set_belt_speed(command.speed)
        elif isinstance(command, cmd.SetAutopilotCommand):
            self._automation.set_autopilot(command.mode)  # type: ignore[union-attr]
        elif isinstance(command, cmd.SetStudioRecordingCommand):
            self._automation.set_studio_recording(command.options)  # type: ignore[union-attr]
        elif isinstance(command, cmd.SetObjectPoseCommand) and sim is not None:
            self._objects.set_pose(
                sim.model, sim.data, command.joint, command.position, command.wxyz, hold=command.hold
            )
        elif isinstance(command, cmd.ShutdownCommand):
            logger.info("Shutdown requested")
            _signal_owner_shutdown()

    def _reset(self) -> None:
        """Restore the start state without changing the scene (DRV-8; ``reset()``, ``POST /reset``).

        Floating-base robots go back to their home pose, at rest, and their base holds are armed
        again; then the scene's reset runs. Fixed-base robots keep the scene reset's handling
        (SO-101 parity), so only scenes that set an arm pose move them.
        """
        import mujoco  # noqa: PLC0415

        from physicalai_mujoco_plugin.sim import place_home  # noqa: PLC0415

        sim = self._sim
        if sim is not None and sim.bases:
            floating = [
                (binding, channels)
                for binding, channels in zip(sim.bindings, sim.channels, strict=True)
                if binding.layout.base is not None
            ]
            for binding, _ in floating:
                place_home(sim.model, sim.data, binding)
            mujoco.mj_forward(sim.model, sim.data)
            for _, channels in floating:
                channels.hold_current()
            sim.bases.arm_holds()
            self._base_status = sim.bases.status()
            logger.info("Floating bases back at their start pose and held")
        self._run_scene_reset()

    def _run_scene_reset(self) -> None:
        """Randomize the current scene without letting a failure stop the loop.

        Reset callbacks read scene-specific model layout (flex vertices, named free joints); a
        scene that does not match its callback degrades to a logged warning.
        """
        sim = self._sim
        try:
            self._objects.held.clear()
            self._reseed_if_fixed()
            if sim.on_reset is not None:  # type: ignore[union-attr]
                sim.on_reset(sim.model, sim.data, self._rng)  # type: ignore[union-attr]
            else:
                self._objects.randomize(sim.model, sim.data, self._rng)  # type: ignore[union-attr]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Scene reset failed: {}", exc)
            return
        if self._episode_auto_reset is not None:
            self._episode_auto_reset.notify_manual_reset()
        self._automation.reset()  # type: ignore[union-attr]

    def _handle_replay_command(self, command: SimCommand) -> bool:
        """Start or stop a replay, and end it before a command that edits the physics state.

        Returns:
            Whether *command* was a replay command, with nothing left to do.
        """
        from physicalai_mujoco_plugin import http_server as cmd  # noqa: PLC0415

        if isinstance(command, cmd.ReplayCommand):
            self._start_replay(command)
            return True
        if isinstance(command, cmd.StopReplayCommand):
            self._stop_replay()
            return True
        if isinstance(command, (cmd.ResetCommand, cmd.HomeCommand, cmd.SetObjectPoseCommand)):
            # These edit the live physics state, which the replay's end would overwrite.
            self._stop_replay()
        return False

    def _start_replay(self, command: ReplayCommand) -> None:
        """Start playing *command*'s frames; a replay already running is replaced.

        Raises:
            ValueError: If the frames do not have one value per joint, or one cannot be shown
                unchanged (see :meth:`ArmChannels.replay_placements`).
        """
        import mujoco  # noqa: PLC0415

        from physicalai_mujoco_plugin.replay import Replay  # noqa: PLC0415

        sim = self._sim
        count = sum(len(channels) for channels in sim.channels)  # type: ignore[union-attr]
        if command.joint_positions.shape[1:] != (count,) or not len(command.joint_positions):
            msg = f"Replay frames have shape {command.joint_positions.shape}; expected (frames >= 1, {count})"
            raise ValueError(msg)
        # Every frame of every robot is checked before the first one is shown, so no frame can fail
        # half-way through placing the robots.
        self._replay_preflight(command.joint_positions)
        if self._replay is not None:
            # Keep the state from before the first replay; the current one is a replayed frame.
            saved = self._replay.saved_state
        else:
            spec = mujoco.mjtState.mjSTATE_INTEGRATION
            saved = np.empty(mujoco.mj_stateSize(sim.model, spec))  # type: ignore[union-attr]
            mujoco.mj_getState(sim.model, sim.data, saved, spec)  # type: ignore[union-attr]
        with self._state_lock:
            self._replay = Replay(command.joint_positions, command.fps, saved)
        logger.info(
            "Replaying {} frames at {:g} fps; actions are ignored until the replay is stopped",
            len(command.joint_positions),
            command.fps,
        )

    def _stop_replay(self) -> None:
        """End the replay, if any, and restore the physics state from before it."""
        import mujoco  # noqa: PLC0415

        replay = self._replay
        if replay is None:
            return
        sim = self._sim
        mujoco.mj_setState(sim.model, sim.data, replay.saved_state, mujoco.mjtState.mjSTATE_INTEGRATION)  # type: ignore[union-attr]
        mujoco.mj_forward(sim.model, sim.data)  # type: ignore[union-attr]
        with self._state_lock:
            self._replay = None
        logger.info("Replay stopped; the simulation continues from where it was before the replay")

    def _show_replay_frame(self, control_dt: float) -> None:
        """Set the joints to the replay's current frame, without stepping physics."""
        import mujoco  # noqa: PLC0415

        sim = self._sim
        row = self._replay.next_frame(control_dt)  # type: ignore[union-attr]
        start = 0
        for channels in sim.channels:  # type: ignore[union-attr]
            channels.set_positions(row[start : start + len(channels)])
            start += len(channels)
        mujoco.mj_forward(sim.model, sim.data)  # type: ignore[union-attr]

    def _unreplayable_joints(self) -> list[str]:
        """Return the public channels a replay cannot place (tendon or site transmissions).

        Returns:
            Their names; empty when every channel can be replayed or nothing is loaded.
        """
        sim = self._sim
        return [name for channels in sim.channels for name in channels.unpositionable] if sim is not None else []

    def _replay_preflight(self, frames: np.ndarray) -> None:
        """Check that every robot can show every replay frame unchanged.

        Raises:
            ValueError: Naming the first joint and frame that cannot be shown, or a shape mismatch.
        """
        sim = self._sim
        if sim is None:
            msg = "The simulation is not connected"
            raise ValueError(msg)
        count = sum(len(channels) for channels in sim.channels)
        if frames.ndim != 2 or frames.shape[1] != count:  # noqa: PLR2004
            msg = f"Replay frames have shape {frames.shape}; expected (frames, {count})"
            raise ValueError(msg)
        start = 0
        for channels in sim.channels:
            channels.replay_placements(frames[:, start : start + len(channels)])
            start += len(channels)

    def _replay_problem(self, frames: np.ndarray) -> str | None:
        """Check replay frames for ``POST /replay``: the same check the sim thread runs before playing.

        It reads only the channel conversions, which never change while a scene is loaded.

        Returns:
            Why the frames cannot be replayed, or ``None``.
        """
        try:
            self._replay_preflight(frames)
        except ValueError as exc:
            return str(exc)
        return None

    def _replay_status(self) -> dict[str, object]:
        status = self._replay.status() if self._replay is not None else {"active": False}
        return {**status, "unsupported_joints": self._unreplayable_joints()}

    def _go_home(self) -> None:
        """Place every robot at the scene's home pose (else the profile's) and hold it there."""
        import mujoco  # noqa: PLC0415

        sim = self._sim
        scene = self._scene_config
        scene_home = dict(scene.home_pose(len(sim.bindings))) if scene is not None else {}  # type: ignore[union-attr]
        for binding, channels in zip(sim.bindings, sim.channels, strict=True):  # type: ignore[union-attr]
            home = {joint: values[0] for joint, values in binding.layout.home_qpos.items() if len(values) == 1}
            home.update({joint: value for joint, value in scene_home.items() if joint in home})
            channels.place(home)
        mujoco.mj_forward(sim.model, sim.data)  # type: ignore[union-attr]
        logger.info("Robot moved to the home pose")

    def _reseed_if_fixed(self) -> None:
        """Restart the shared RNG from the fixed seed so the next reset repeats.

        The generator is reseeded in place because the episode auto-reset helper holds it.
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

    def _set_belt_speed(self, speed: float) -> None:
        with self._state_lock:
            has_belt = self._automation.set_belt_speed(speed)  # type: ignore[union-attr]
        if not has_belt:
            logger.warning("Scene '{}' has no conveyor belt", self._current_scene_id)

    def _init_episode_auto_reset(self) -> None:
        """Attach the scene's episode controller: the conveyor's, else the cube-on-plate auto-reset."""
        from physicalai_mujoco_plugin.episode_auto_reset import EpisodeAutoReset  # noqa: PLC0415

        model = self._sim.model  # type: ignore[union-attr]
        helper = self._automation.attach_scene(  # type: ignore[union-attr]
            model,
            rng=self._rng,
            active=self._auto_reset_active,
            robots=len(self._sim.bindings),  # type: ignore[union-attr]
            robot_roots=self._sim.robot_roots,  # type: ignore[union-attr]
        )
        if helper is None:
            area = self._area
            helper = EpisodeAutoReset.maybe_create(
                model,
                free_joints=area.free_joints,
                target_body_name=area.target_body_name,
                spawn_center=area.spawn_center,
                spawn_min_r=area.spawn_min_r,
                spawn_max_r=area.spawn_max_r,
                spawn_angle_half_deg=area.spawn_angle_half_deg,
                target_min_sep=area.target_min_sep,
                rng=self._rng,
                success_dwell_s=self._auto_reset_dwell_s,
                active=self._auto_reset_active,
            )
        with self._state_lock:
            self._episode_auto_reset = helper

    # ------------------------------------------------------------------
    # Status for the HTTP server and the viewer panel (other threads)
    # ------------------------------------------------------------------

    def _start_http_server(self) -> None:
        if self._http_port <= 0:
            return
        from physicalai_mujoco_plugin.http_server import HttpServer, build_app  # noqa: PLC0415

        app = build_app(
            service_name=self._owner_name or f"mujoco-{self._profile.name}",
            buffers=self._cameras.frame_buffers,
            commands=self._commands,
            get_status=self._http_status,
            check_replay=self._replay_problem,
            get_leader=self._automation.leader_snapshot,  # type: ignore[union-attr]
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
        if self._http_server is not None:
            with contextlib.suppress(Exception):
                self._http_server.stop()
            self._http_server = None

    def _timing_status(self) -> dict[str, object]:
        """Real-time factor, control rate and per-camera frame rates over the last ~2 s.

        Returns:
            A JSON-friendly dict; rates are ``None`` until enough ticks or frames were seen.
        """
        return {
            "real_time_factor": self._tick_meter.value_rate(),
            "control_hz": self._tick_meter.rate(),
            "camera_fps": self._cameras.fps(),
            "cameras_on_thread": self._cameras.on_thread,
        }

    @staticmethod
    def _episode_status(helper: object | None) -> dict[str, object]:
        return helper.status() if helper is not None else {"enabled": False}

    def _http_status(self) -> dict[str, object]:
        """Return the status served at ``GET /`` (uvicorn thread).

        Returns:
            A JSON-friendly snapshot of the robot, scene, episode and cameras.
        """
        from physicalai_mujoco_plugin.scene_registry import list_scenes  # noqa: PLC0415

        rendering = self._cameras.rendering()
        compatible = sorted(self._compatible_scenes())
        with self._state_lock:
            sim = self._sim
            sources = sim.camera_sources() if sim is not None else {}
            return {
                "connected": sim is not None,
                "profile": self._profile.name,
                "tier": self._profile.tier,
                "joint_names": self.joint_names if sim is not None else [],
                "units": [unit for channels in sim.channels for unit in channels.units] if sim is not None else [],
                "floating_base": bool(sim.bases) if sim is not None else False,
                "fallen": any(base.fallen for base in self._base_status),
                "bases": [base.as_dict() for base in self._base_status],
                "viewer_url": self._viewer.url if self._viewer is not None else None,
                "scene": self._current_scene_id,
                "scenes": sorted(list_scenes()),
                "compatible_scenes": compatible,
                "seed": self._seed,
                "episode": self._episode_status(self._episode_auto_reset),
                "replay": self._replay_status(),
                "timing": self._timing_status(),
                **(self._automation.status() if self._automation is not None else {}),
                "objects": [
                    {"joint": joint, "position": list(pose.position), "wxyz": list(pose.wxyz)}
                    for joint, pose in self._objects.poses.items()
                ],
                "cameras": [
                    {
                        "name": config.name,
                        "width": config.width,
                        "height": config.height,
                        "fps": config.fps,
                        "rendering": config.name in rendering,
                        # override, model or default for robot cameras (CAM-3), scene otherwise.
                        "source": sources.get(config.name),
                    }
                    for config in self._cameras.configs
                ],
            }

    def _panel_state(self) -> PanelState:
        """Snapshot the state rendered by the viewer's Simulation panel.

        Returns:
            The current scene, compatible scenes, seed, episode status, object poses, cameras, and
            the robot's profile, tier, units and floating bases.
        """
        from physicalai_mujoco_plugin.viser_controls import PanelState  # noqa: PLC0415

        scenes = self._compatible_scenes()
        with self._state_lock:
            model, sim = self._model, self._sim
            return PanelState(
                scene_id=self._current_scene_id,
                scene_options=tuple((scene_id, scene.display_name) for scene_id, scene in scenes.items()),
                seed=self._seed,
                episode=self._episode_status(self._episode_auto_reset),
                objects=dict(self._objects.poses),
                # Configured names, not just live buffers: the viewer is built before the cameras start.
                cameras={config.name: self._cameras.frame_buffers.get(config.name) for config in self._cameras.configs},
                follow_targets=self._objects.follow_targets(self._data),
                object_bodies=dict(self._objects.joint_bodies),
                view_center=tuple(float(v) for v in model.stat.center) if model is not None else (0.0, 0.0, 0.0),  # type: ignore[arg-type]
                view_extent=float(model.stat.extent) if model is not None else 1.0,
                timing=self._timing_status(),
                profile=self._profile.name,
                tier=self._profile.tier,
                units=tuple(unit for channels in sim.channels for unit in channels.units) if sim is not None else (),
                bases=self._base_status,
                replay=self._replay_status(),
                **(self._automation.status() if self._automation is not None else {}),
            )


__all__ = ["OperatorControls"]
