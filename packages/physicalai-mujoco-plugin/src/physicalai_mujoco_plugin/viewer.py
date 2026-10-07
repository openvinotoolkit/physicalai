# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Viewer service: the viser browser viewer with its control panel, or MuJoCo's native viewer.

The driver owns one :class:`ViewerService` and syncs it once per tick; the viewer never steps
physics. viser works in a browser on macOS without ``mjpython``, so it is the default; the native
viewer is a fallback on Linux.
"""

# MuJoCo and viser expose runtime-bound attributes.
# pyrefly: ignore-errors [missing-attribute, not-callable, bad-context-manager]

from __future__ import annotations

import contextlib
import sys
import time
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from physicalai_mujoco_plugin.http_server import SimCommand
    from physicalai_mujoco_plugin.viser_controls import PanelState, SimControlPanel


class ViewerService:
    """Show the simulation in viser (with the control panel) or in MuJoCo's native viewer."""

    def __init__(
        self,
        *,
        host: str,
        port: int,
        submit_command: Callable[[SimCommand], None],
        panel_state: Callable[[], PanelState],
        key_callback: Callable[[int], None] | None = None,
    ) -> None:
        """Create a closed viewer.

        Args:
            host: viser bind address.
            port: viser port; ``0`` or less disables viser.
            submit_command: Queues a panel command for the sim thread.
            panel_state: Snapshots the state the panel shows.
            key_callback: Native viewer key handler.
        """
        self.host = host
        self.port = port
        self._submit_command = submit_command
        self._panel_state = panel_state
        self._key_callback = key_callback
        self.server: object | None = None
        self.scene: object | None = None
        self.panel: SimControlPanel | None = None
        self.native: object | None = None
        self._model: object | None = None
        self._sync_failed = False
        self._panel_failed = False

    @property
    def active(self) -> bool:
        """Whether a viewer is showing the simulation."""
        return self.scene is not None or self.native is not None

    @property
    def url(self) -> str | None:
        """The viser page, when viser runs; an IPv6 host is bracketed."""
        if self.server is None:
            return None
        host = f"[{self.host}]" if ":" in self.host and not self.host.startswith("[") else self.host
        return f"http://{host}:{self.port}"

    def open(self, model: object, data: object) -> bool:
        """Open viser, or the native viewer where viser is unavailable (not on macOS).

        Returns:
            Whether a viewer opened.
        """
        self._model = model
        if not self._launch_viser() and sys.platform != "darwin":
            self._launch_native(model, data)
        return self.active

    def close(self) -> None:
        """Close whichever viewer is open."""
        if self.server is not None:
            stop = getattr(self.server, "stop", None)
            if callable(stop):
                with contextlib.suppress(Exception):
                    stop()
            self.server = None
            self.scene = None
            self.panel = None
        if self.native is not None:
            with contextlib.suppress(Exception):
                self.native.close()
            self.native = None

    def sync(self, data: object) -> bool:
        """Push this tick's state to the viewer (sim thread).

        Returns:
            ``False`` once the user closed the native viewer, else ``True``.
        """
        if self.scene is not None:
            self._sync_viser(data)
            self.refresh_panel()
        elif self.native is not None:
            if not self.native_running():
                logger.info("MuJoCo viewer closed by user")
                self.native = None
                return False
            self.native_sync()
        return True

    def rebuild(self, model: object, data: object) -> None:
        """Show a new model after a scene switch.

        On failure the viser scene is dropped rather than left in place: syncing a scene built
        for the previous model against the new data would feed it mismatched geometry, silently,
        every tick.
        """
        self._model = model
        self._set_native_model_data(model, data)
        if self.server is None:
            return
        self.scene = None
        try:
            self.build_gui()
        except Exception as exc:  # noqa: BLE001
            self.scene = None
            logger.warning("Failed to recreate viser scene: {}", exc)

    def native_running(self) -> bool:
        """Return whether the native viewer window is still open.

        Returns:
            ``True`` while the window is open.
        """
        if self.native is None:
            return False
        is_running = getattr(self.native, "is_running", None)
        if callable(is_running):
            with contextlib.suppress(Exception):
                return bool(is_running())
        return True

    @contextlib.contextmanager
    def native_lock(self) -> Iterator[None]:
        """Hold the native viewer's lock, so its render thread does not read the model meanwhile.

        A no-op without a native viewer. Never call ``set_texts`` inside: it takes the lock too.

        Yields:
            Nothing; the lock is held until the block exits.
        """
        lock = getattr(self.native, "lock", None)
        if not callable(lock):
            yield
            return
        with lock():
            yield

    def native_sync(self) -> None:
        """Copy the simulation state into the native viewer."""
        sync = getattr(self.native, "sync", None)
        if callable(sync):
            with contextlib.suppress(Exception):
                sync()

    def refresh_panel(self) -> None:
        """Mirror sim state into the viewer panel without letting it stop the loop."""
        panel = self.panel
        if panel is None:
            return
        try:
            panel.refresh(self._panel_state(), time.monotonic())
        except Exception as exc:  # noqa: BLE001
            if not self._panel_failed:
                self._panel_failed = True
                logger.warning("viser panel refresh failed, its readouts will be stale: {}", exc)
        else:
            self._panel_failed = False

    def build_gui(self) -> None:
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

        from physicalai_mujoco_plugin.viser_controls import SimControlPanel  # noqa: PLC0415

        server = self.server
        if server is None:
            msg = "viser server is not running"
            raise RuntimeError(msg)
        if self.panel is None:
            self.panel = SimControlPanel(server, viser, self._submit_command)

        self.scene = None
        server.gui.reset()
        server.scene.reset()

        scene = ViserMujocoScene(server, self._model, num_envs=1)
        scene.camera_tracking_enabled = False
        # mjviser's default refresh re-renders from the viser thread while the sim thread may be
        # stepping the same data; the sim loop re-renders at the control rate anyway.
        scene.set_refresh_handler(lambda: None)
        state = self._panel_state()
        tabs = server.gui.add_tab_group()
        with tabs.add_tab("Simulation", icon=viser.Icon.ROBOT):
            self.panel.build(state)
        with tabs.add_tab("Camera", icon=viser.Icon.VIDEO):
            self.panel.build_camera_tab(state)
        with tabs.add_tab("Visualization", icon=viser.Icon.EYE):
            scene.create_overlay_gui()
        with tabs.add_tab("Groups", icon=viser.Icon.LAYERS_INTERSECT):
            scene.create_groups_gui()
        self.panel.build_scene_nodes(state)

        self.scene = scene
        self._panel_failed = False

    def _launch_viser(self) -> bool:
        """Start the browser-based 3D viewer via mjviser/viser (works on macOS).

        Returns:
            Whether the viewer was launched.
        """
        if self.port <= 0:
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
            server = viser.ViserServer(host=self.host, port=self.port, verbose=False)
            self.server = server
            self.build_gui()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to start viser viewer on port {}: {}", self.port, exc)
            if server is not None:
                with contextlib.suppress(Exception):
                    server.stop()
            self.server = None
            self.scene = None
            self.panel = None
            return False
        logger.info("3D viewer: http://{}:{}", self.host, self.port)
        return True

    def _launch_native(self, model: object, data: object) -> bool:
        """Start the native MuJoCo viewer (Linux/Windows only; broken on macOS).

        Returns:
            Whether the viewer was launched.
        """
        try:
            import mujoco.viewer  # noqa: PLC0415

            viewer = mujoco.viewer.launch_passive(model, data, key_callback=self._key_callback)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to open MuJoCo viewer: {}", exc)
            return False
        self.native = viewer
        logger.info("MuJoCo native viewer opened")
        return True

    def _set_native_model_data(self, model: object, data: object) -> None:
        """Hot-swap the native viewer's model, for viewer APIs that support it."""
        if self.native is None or not self.native_running():
            return
        lock = getattr(self.native, "lock", None)
        get_sim = getattr(self.native, "_get_sim", None)
        if not callable(lock) or not callable(get_sim):
            return
        with contextlib.suppress(Exception), lock():
            sim = get_sim()
            if sim is not None:
                sim.m = model
                sim.d = data

    def _sync_viser(self, data: object) -> None:
        """Push the current sim state into the viser scene.

        Failures are reported once and then muted: this runs at the loop rate,
        and a broken viewer must not stop the simulation.
        """
        try:
            self.scene.update_from_mjdata(data)
            self._sync_fixed_bodies(data)
        except Exception as exc:  # noqa: BLE001
            if not self._sync_failed:
                self._sync_failed = True
                logger.warning("viser sync failed, the 3D viewer will not update: {}", exc)
        else:
            self._sync_failed = False

    def _sync_fixed_bodies(self, data: object) -> None:
        """Push current ``data.xpos`` into mjviser's fixed-geometry handles.

        Live camera edits (``scene_watch``) can move fixed world bodies, e.g. camera rigs, by
        writing ``model.body_pos`` outside a normal sim step. MuJoCo cameras pick that up after
        ``mj_forward``, but mjviser only places fixed meshes at create time, so it needs this
        explicit resync to reflect the change.

        Handles live under ``/fixed_bodies``, which stays at the world origin,
        so store world ``xpos`` as-is.
        """
        scene = self.scene
        if scene is None:
            return
        for attribute in ("_fixed_geom_handles", "_fixed_site_handles"):
            for (body_id, *_rest), handle in (getattr(scene, attribute, None) or {}).items():
                handle.position = np.asarray(data.xpos[body_id], dtype=np.float64)
                handle.wxyz = np.asarray(data.xquat[body_id], dtype=np.float64)


__all__ = ["ViewerService"]
