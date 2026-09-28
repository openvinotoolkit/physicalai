# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Simulation control panel for the Viser browser viewer.

The panel never touches MuJoCo state. Viser callbacks only enqueue
:data:`~physicalai_mujoco_so101_plugin.http_server.SimCommand` objects (the
same commands the HTTP API sends), and the simulation thread pushes
:class:`PanelState` snapshots back through :meth:`SimControlPanel.refresh`.

Assigning a Viser input's ``value`` from the server fires its ``on_update``
callbacks with ``client=None``; every input callback ignores those events so
that syncing the panel from sim state never echoes a command back.

The Camera tab is the exception: following a body and the field of view are
viewer presentation, not simulation state, so they live on the panel and act
on the viewer cameras directly. The rendered world is never shifted.
"""

from __future__ import annotations

import contextlib
import itertools
import re
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
from loguru import logger

from physicalai_mujoco_so101_plugin.http_server import (
    MAX_DWELL_S,
    MAX_SEED,
    MIN_DWELL_S,
    HomeCommand,
    ResetCommand,
    SetAutoResetCommand,
    SetObjectPoseCommand,
    SetSeedCommand,
    ShutdownCommand,
    SwitchSceneCommand,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from physicalai_mujoco_so101_plugin.http_server import FrameBuffer, SimCommand

CUSTOM_MODEL_LABEL = "Custom model"
"""Scene dropdown entry shown when the loaded model is not a registered scene."""

_STATUS_PERIOD_S = 0.5
_OBJECT_PERIOD_S = 0.1
_PREVIEW_PERIOD_S = 0.2
_PREVIEW_WIDTH = 320
_PREVIEW_JPEG_QUALITY = 70
_GIZMO_SCALE = 0.08
_GIZMO_PARENT = "/fixed_bodies"
"""mjviser's frame for fixed geometry; it sits at the world origin."""

NO_FOLLOW_LABEL = "None"
"""Follow dropdown entry for a free camera."""
_FOV_DEFAULT_DEG = 60.0
_FOV_MIN_DEG = 20.0
_FOV_MAX_DEG = 150.0
_CAMERA_AZIMUTH_DEG = 120.0
_CAMERA_ELEVATION_DEG = 20.0
_CAMERA_DISTANCE_EXTENTS = 3.0
_FOLLOW_BLEND_S = 0.3
_FOLLOW_EPS_M = 1e-5
_PAN_RELEASE_M = 0.015
"""How far a viewer's look-at may stray from the steered path before it counts as a pan."""
_FOLLOW_HISTORY_S = 1.0
"""How long commanded look-at points are kept, to tolerate lagging camera reports."""

CameraView = tuple[np.ndarray, np.ndarray]
"""A viewer camera's ``(position, look_at)`` in world coordinates."""


def _smoothstep(progress: float) -> float:
    return progress * progress * (3.0 - 2.0 * progress)


def _distance_to_path(point: np.ndarray, path: list[np.ndarray]) -> float:
    """Return the distance from *point* to the polyline through *path*."""
    if len(path) == 1:
        return float(np.linalg.norm(point - path[0]))
    best = float("inf")
    for start, end in itertools.pairwise(path):
        segment = end - start
        length_sq = float(segment @ segment)
        t = 0.0 if length_sq < _FOLLOW_EPS_M**2 else min(max(float((point - start) @ segment) / length_sq, 0.0), 1.0)
        best = min(best, float(np.linalg.norm(point - (start + t * segment))))
    return best


@dataclass(frozen=True)
class FollowStep:
    """Result of one :meth:`CameraFollow.step`."""

    shifts: dict[int, np.ndarray] = field(default_factory=dict)
    """Client id -> world translation for that client's camera."""
    released: bool = False
    """A viewer panned away, so following stopped."""


class CameraFollow:
    """Keeps every viewer camera looking at the followed body.

    Following moves the viewer cameras, never the world: each step shifts a
    camera (position and look-at together) so its look-at point sits on the
    body. With nothing followed the cameras are left alone.

    Orbiting and zooming keep a camera's look-at point where it is, so they
    work around the body. Panning moves the look-at point; when a viewer's
    look-at leaves the path this class steered it along, the user panned, and
    following stops. Browsers report their camera with some lag, so the check
    is against the recently commanded path rather than the latest point.

    Choosing a body, and letting go of it after a drag, glides each camera
    onto it instead of jumping. While the followed object is being dragged the
    cameras hold still, so the drag handle stays under the pointer.

    Changed from Viser callbacks and stepped on the sim thread, so it locks.
    """

    def __init__(
        self,
        blend_s: float = _FOLLOW_BLEND_S,
        pan_release_m: float = _PAN_RELEASE_M,
        history_s: float = _FOLLOW_HISTORY_S,
    ) -> None:
        """Start with a free camera."""
        self._lock = threading.Lock()
        self._blend_s = blend_s
        self._pan_release_m = pan_release_m
        self._history_s = history_s
        self._name: str | None = None
        # Client id -> (look-at when the glide started, start time).
        self._glides: dict[int, tuple[np.ndarray, float]] = {}
        # Client id -> recent (time, commanded look-at), oldest first.
        self._commanded: dict[int, list[tuple[float, np.ndarray]]] = {}
        self._start_glides = False
        self._held = False
        self._last_target: np.ndarray | None = None

    @property
    def name(self) -> str | None:
        """Followed body name, or ``None`` for a free camera."""
        with self._lock:
            return self._name

    @property
    def last_target(self) -> np.ndarray | None:
        """Last position of the followed body, if one is followed."""
        with self._lock:
            return None if self._last_target is None else self._last_target.copy()

    def select(self, name: str | None) -> None:
        """Follow body *name*, or stop following with ``None``."""
        with self._lock:
            if name == self._name:
                return
            self._select_locked(name)

    def _select_locked(self, name: str | None) -> None:
        self._name = name
        self._glides.clear()
        self._commanded.clear()
        self._start_glides = name is not None
        self._held = False
        self._last_target = None

    def step(
        self,
        views: Mapping[int, CameraView],
        target: np.ndarray | None,
        *,
        hold: bool,
        now: float,
    ) -> FollowStep:
        """Advance following by one tick.

        Args:
            views: Current ``(position, look_at)`` of each open viewer.
            target: World position of the followed body, or ``None``.
            hold: Keep cameras still (the followed object is being dragged).
            now: Monotonic time, for gliding.

        Returns:
            The camera shifts to apply, or ``released`` if a viewer panned away.
        """
        with self._lock:
            if self._name is None or target is None:
                self._glides.clear()
                self._commanded.clear()
                return FollowStep()
            target = np.asarray(target, dtype=np.float64).reshape(3)
            self._last_target = target.copy()
            looks = {cid: np.asarray(look_at, dtype=np.float64).reshape(3) for cid, (_, look_at) in views.items()}
            for cid in [cid for cid in self._commanded if cid not in looks]:
                del self._commanded[cid]
            for cid in [cid for cid in self._glides if cid not in looks]:
                del self._glides[cid]

            if self._panned(looks):
                self._select_locked(None)
                return FollowStep(released=True)

            if hold:
                self._held = True
                for cid, look_at in looks.items():
                    self._commanded.setdefault(cid, [(now, look_at)])
                return FollowStep()
            if self._held or self._start_glides:
                self._held = False
                self._start_glides = False
                self._glides = {cid: (look_at.copy(), now) for cid, look_at in looks.items()}

            shifts: dict[int, np.ndarray] = {}
            for cid, look_at in looks.items():
                shift = self._steer_locked(cid, look_at, target, now)
                if float(np.linalg.norm(shift)) > _FOLLOW_EPS_M:
                    shifts[cid] = shift
            return FollowStep(shifts=shifts)

    def _steer_locked(self, cid: int, look_at: np.ndarray, target: np.ndarray, now: float) -> np.ndarray:
        """Return the shift that puts client *cid*'s look-at on the (gliding) target, and record it."""
        history = self._commanded.setdefault(cid, [(now, look_at.copy())])
        desired = target
        glide = self._glides.get(cid)
        if glide is not None:
            start, started = glide
            progress = (now - started) / self._blend_s
            if progress < 1.0:
                desired = start + (target - start) * _smoothstep(max(progress, 0.0))
            else:
                del self._glides[cid]
        history.append((now, desired.copy()))
        # Keep the recent path plus the last point before it, so the path
        # still reaches where the camera sat before a quiet spell.
        while len(history) > 1 and history[1][0] < now - self._history_s:
            history.pop(0)
        return desired - look_at

    def _panned(self, looks: Mapping[int, np.ndarray]) -> bool:
        """Return whether any viewer's look-at left the path it was steered along."""
        for cid, look_at in looks.items():
            history = self._commanded.get(cid)
            if history and _distance_to_path(look_at, [point for _, point in history]) > self._pan_release_m:
                return True
        return False


def default_camera_position(extent: float) -> np.ndarray:
    """Return mjviser's default viewpoint for a model of size *extent*, relative to what it looks at."""
    azimuth = np.deg2rad(_CAMERA_AZIMUTH_DEG)
    elevation = np.deg2rad(_CAMERA_ELEVATION_DEG)
    direction = np.array(
        [
            -np.cos(elevation) * np.cos(azimuth),
            -np.cos(elevation) * np.sin(azimuth),
            np.sin(elevation),
        ],
    )
    return direction * _CAMERA_DISTANCE_EXTENTS * max(float(extent), 1e-3)


@dataclass(frozen=True)
class ObjectPose:
    """World pose of one free object."""

    position: tuple[float, float, float]
    wxyz: tuple[float, float, float, float]


@dataclass(frozen=True)
class PanelState:
    """Sim-thread snapshot rendered by the panel."""

    scene_id: str | None
    scene_options: tuple[tuple[str, str], ...]
    """Compatible scenes as ``(scene_id, display_name)`` pairs."""
    seed: int | None
    episode: Mapping[str, Any]
    objects: Mapping[str, ObjectPose]
    cameras: Mapping[str, FrameBuffer | None]
    """Configured cameras; ``None`` until a camera's renderer starts."""
    follow_targets: Mapping[str, tuple[float, float, float]] = field(default_factory=dict)
    """World positions of the bodies the viewer can follow, in dropdown order."""
    object_bodies: Mapping[str, str] = field(default_factory=dict)
    """Free object joint -> the body it moves."""
    view_center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Point a free camera looks at by default."""
    view_extent: float = 1.0
    """Model size, used for the default camera distance."""


@dataclass
class _Handles:
    """Handles for one build of the panel; replaced on every rebuild."""

    scene_dropdown: Any = None
    scene_by_label: dict[str, str] = field(default_factory=dict)
    fixed_seed: Any = None
    seed_number: Any = None
    auto_reset: Any = None
    dwell: Any = None
    episode_status: Any = None
    drag_toggle: Any = None
    gizmos: dict[str, Any] = field(default_factory=dict)
    preview_toggle: Any = None
    previews: dict[str, Any] = field(default_factory=dict)
    preview_seq: dict[str, int] = field(default_factory=dict)
    follow_dropdown: Any = None


def _is_server_event(event: object) -> bool:
    """Return whether an input event came from the server, not a browser."""
    return getattr(event, "client", None) is None


def _node_name(joint: str) -> str:
    return f"{_GIZMO_PARENT}/drag_{re.sub(r'[^A-Za-z0-9_-]', '_', joint)}"


def _scene_label(state: PanelState) -> str:
    for scene_id, label in state.scene_options:
        if scene_id == state.scene_id:
            return label
    return CUSTOM_MODEL_LABEL


def _downscale(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    if width <= _PREVIEW_WIDTH:
        return frame
    size = (_PREVIEW_WIDTH, max(1, round(height * _PREVIEW_WIDTH / width)))
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)


def _episode_markdown(episode: Mapping[str, Any]) -> str:
    if not episode.get("active", True):
        phase = "paused"
    elif episode.get("phase") == "success_hold" and episode.get("countdown_s") is not None:
        phase = f"success, respawning in {float(episode['countdown_s']):.1f} s"
    else:
        phase = "waiting for success"
    return f"**Status:** {phase}  \n**Episodes:** {int(episode.get('episode_count', 0))}"


class SimControlPanel:
    """Builds and refreshes the "Simulation" and "Camera" tabs of the Viser viewer.

    One panel lives as long as its Viser server. Scene switches rebuild the
    whole GUI, so :meth:`build`, :meth:`build_camera_tab` and
    :meth:`build_scene_nodes` run again for every scene while viewer
    preferences (followed body, FOV, previews, object dragging) persist on
    the panel.
    """

    def __init__(self, server: Any, viser_module: Any, submit: Callable[[SimCommand], None]) -> None:  # noqa: ANN401
        """Attach to *server*; *submit* enqueues commands for the sim thread."""
        self._server = server
        self._viser = viser_module
        self._submit = submit
        self._handles = _Handles()
        self._show_previews = False
        self._drag_objects = False
        # Object joint -> (client id, last pose) for gizmos being dragged. Written
        # on Viser's event loop, read on the sim thread.
        self._drags: dict[str, tuple[int | None, ObjectPose]] = {}
        self._drags_lock = threading.Lock()
        self._next_status = 0.0
        self._next_objects = 0.0
        self._next_preview = 0.0
        self.camera = CameraFollow()
        self._fov_deg = _FOV_DEFAULT_DEG
        self._view_center = np.zeros(3)
        self._view_extent = 1.0
        server.on_client_disconnect(self._on_client_disconnect)
        # Registered once for the server's lifetime (mjviser's scene GUI would
        # add one per scene build), so it always reads the current settings.
        server.on_client_connect(self._on_client_connect)

    @property
    def show_previews(self) -> bool:
        """Whether camera previews are shown (persists across rebuilds)."""
        return self._show_previews

    @property
    def drag_objects(self) -> bool:
        """Whether object drag handles are shown (persists across rebuilds)."""
        return self._drag_objects

    def dragged_joints(self) -> frozenset[str]:
        """Return the object joints a browser is currently dragging."""
        with self._drags_lock:
            return frozenset(self._drags)

    # ------------------------------------------------------------------
    # Build (sim thread, inside a Viser GUI container)
    # ------------------------------------------------------------------

    def build(self, state: PanelState) -> None:
        """Add the panel's inputs to the current GUI container."""
        self._handles = _Handles()
        with self._drags_lock:
            self._drags.clear()
        self._next_status = self._next_objects = self._next_preview = 0.0
        self._build_scene_controls(state)
        self._build_seed_controls(state)
        if state.episode.get("enabled"):
            self._build_episode_controls(state)
        if state.objects:
            self._build_object_controls()
        if state.cameras:
            self._build_camera_previews(state)
        self._build_shutdown()

    def build_camera_tab(self, state: PanelState) -> None:
        """Add the follow, field-of-view and reset-view inputs to the current GUI container."""
        gui = self._server.gui
        self._view_center = np.asarray(state.view_center, dtype=np.float64)
        self._view_extent = float(state.view_extent)
        followed = self.camera.name
        if followed is not None and followed not in state.follow_targets:
            followed = None
            self.camera.select(None)

        dropdown = gui.add_dropdown(
            "Follow",
            options=[NO_FOLLOW_LABEL, *state.follow_targets],
            initial_value=followed if followed is not None else NO_FOLLOW_LABEL,
            hint="Keep the view on this while you orbit and zoom around it; panning stops following",
        )
        self._handles.follow_dropdown = dropdown

        @dropdown.on_update
        def _on_follow(event: object) -> None:
            if _is_server_event(event):
                return
            self.camera.select(None if dropdown.value == NO_FOLLOW_LABEL else str(dropdown.value))

        fov = gui.add_slider(
            "FOV (°)",
            min=_FOV_MIN_DEG,
            max=_FOV_MAX_DEG,
            step=1.0,
            initial_value=self._fov_deg,
            hint="Vertical field of view for every open viewer",
        )

        @fov.on_update
        def _on_fov(event: object) -> None:
            if _is_server_event(event):
                return
            self._fov_deg = min(max(float(fov.value), _FOV_MIN_DEG), _FOV_MAX_DEG)
            for client in self._server.get_clients().values():
                client.camera.fov = np.deg2rad(self._fov_deg)

        reset_view = gui.add_button("Reset View", icon=self._viser.Icon.FOCUS_CENTERED)

        @reset_view.on_click
        def _on_reset_view(_: object) -> None:
            for client in self._server.get_clients().values():
                self._aim_default(client)

    def _aim_default(self, client: Any) -> None:  # noqa: ANN401
        """Point *client* at the followed body, or the model centre, from the default angle."""
        target = self.camera.last_target
        look_at = target if target is not None else self._view_center
        with client.atomic() if hasattr(client, "atomic") else contextlib.nullcontext():
            client.camera.fov = np.deg2rad(self._fov_deg)
            client.camera.position = look_at + default_camera_position(self._view_extent)
            client.camera.look_at = look_at

    def _on_client_connect(self, client: object) -> None:
        self._aim_default(client)

    def build_scene_nodes(self, state: PanelState) -> None:
        """Add drag gizmos for free objects (after the MuJoCo scene exists)."""
        for joint, pose in state.objects.items():
            gizmo = self._server.scene.add_transform_controls(
                _node_name(joint),
                scale=_GIZMO_SCALE,
                depth_test=False,
                position=pose.position,
                wxyz=pose.wxyz,
                visible=self._drag_objects,
            )
            gizmo.on_update(self._make_gizmo_callback(joint))
            self._handles.gizmos[joint] = gizmo

    def _build_scene_controls(self, state: PanelState) -> None:
        gui = self._server.gui
        handles = self._handles
        with gui.add_folder("Scene"):
            handles.scene_by_label = {label: scene_id for scene_id, label in state.scene_options}
            options = list(handles.scene_by_label)
            current = _scene_label(state)
            if current == CUSTOM_MODEL_LABEL:
                options.insert(0, CUSTOM_MODEL_LABEL)
            if options:
                dropdown = gui.add_dropdown("Scene", options=options, initial_value=current)
                handles.scene_dropdown = dropdown

                @dropdown.on_update
                def _on_scene(event: object) -> None:
                    if _is_server_event(event):
                        return
                    scene_id = handles.scene_by_label.get(dropdown.value)
                    if scene_id is not None and scene_id != state.scene_id:
                        self._submit(SwitchSceneCommand(scene_id=scene_id))

            reset_button = gui.add_button("Reset Scene", icon=self._viser.Icon.REFRESH)

            @reset_button.on_click
            def _on_reset(_: object) -> None:
                self._submit(ResetCommand())

            home_button = gui.add_button("Home Arm", icon=self._viser.Icon.HOME)

            @home_button.on_click
            def _on_home(_: object) -> None:
                self._submit(HomeCommand())

    def _build_seed_controls(self, state: PanelState) -> None:
        gui = self._server.gui
        with gui.add_folder("Randomization"):
            fixed = gui.add_checkbox(
                "Fixed seed",
                initial_value=state.seed is not None,
                hint="Reseed before every reset so layouts repeat",
            )
            number = gui.add_number(
                "Seed",
                initial_value=state.seed if state.seed is not None else 0,
                min=0,
                max=MAX_SEED,
                step=1,
                disabled=state.seed is None,
            )
        self._handles.fixed_seed = fixed
        self._handles.seed_number = number

        def _seed_value() -> int:
            return min(max(int(number.value), 0), MAX_SEED)

        @fixed.on_update
        def _on_fixed(event: object) -> None:
            if _is_server_event(event):
                return
            number.disabled = not fixed.value
            self._submit(SetSeedCommand(seed=_seed_value() if fixed.value else None))

        @number.on_update
        def _on_seed(event: object) -> None:
            if _is_server_event(event) or not fixed.value:
                return
            self._submit(SetSeedCommand(seed=_seed_value()))

    def _build_episode_controls(self, state: PanelState) -> None:
        gui = self._server.gui
        episode = state.episode
        with gui.add_folder("Episode"):
            auto_reset = gui.add_checkbox(
                "Auto-reset",
                initial_value=bool(episode.get("active", True)),
                hint="Respawn the cube after it rests on the target",
            )
            dwell = gui.add_number(
                "Success dwell (s)",
                initial_value=float(episode.get("success_dwell_s", 5.0)),
                min=MIN_DWELL_S,
                max=MAX_DWELL_S,
                step=0.5,
            )
            status = gui.add_markdown(_episode_markdown(episode))
        self._handles.auto_reset = auto_reset
        self._handles.dwell = dwell
        self._handles.episode_status = status

        @auto_reset.on_update
        def _on_auto_reset(event: object) -> None:
            if not _is_server_event(event):
                self._submit(SetAutoResetCommand(enabled=bool(auto_reset.value)))

        @dwell.on_update
        def _on_dwell(event: object) -> None:
            if not _is_server_event(event):
                value = min(max(float(dwell.value), MIN_DWELL_S), MAX_DWELL_S)
                self._submit(SetAutoResetCommand(dwell_s=value))

    def _build_object_controls(self) -> None:
        gui = self._server.gui
        with gui.add_folder("Objects"):
            toggle = gui.add_checkbox(
                "Drag objects",
                initial_value=self._drag_objects,
                hint="Show handles to move free objects; the target stays fixed",
            )
        self._handles.drag_toggle = toggle

        @toggle.on_update
        def _on_toggle(event: object) -> None:
            if _is_server_event(event):
                return
            self._drag_objects = bool(toggle.value)
            for gizmo in self._handles.gizmos.values():
                gizmo.visible = self._drag_objects

    def _build_camera_previews(self, state: PanelState) -> None:
        gui = self._server.gui
        with gui.add_folder("Cameras"):
            toggle = gui.add_checkbox(
                "Show previews",
                initial_value=self._show_previews,
                hint="Stream low-rate camera thumbnails to every open viewer",
            )
            for name in state.cameras:
                self._handles.previews[name] = gui.add_image(
                    np.zeros((240, _PREVIEW_WIDTH, 3), dtype=np.uint8),
                    label=name,
                    format="jpeg",
                    jpeg_quality=_PREVIEW_JPEG_QUALITY,
                    visible=self._show_previews,
                )
        self._handles.preview_toggle = toggle

        @toggle.on_update
        def _on_toggle(event: object) -> None:
            if _is_server_event(event):
                return
            self._show_previews = bool(toggle.value)
            self._next_preview = 0.0
            for image in self._handles.previews.values():
                image.visible = self._show_previews

    def _build_shutdown(self) -> None:
        shutdown_button = self._server.gui.add_button("Shutdown", icon=self._viser.Icon.POWER, color="red")

        @shutdown_button.on_click
        def _on_shutdown_click(event: object) -> None:
            client = getattr(event, "client", None)
            if client is None:
                return
            with client.gui.add_modal("Confirm shutdown") as modal:
                client.gui.add_markdown("Stop the simulation owner? This disconnects all viewers.")
                confirm_button = client.gui.add_button("Shutdown", color="red")
                cancel_button = client.gui.add_button("Cancel")

                @confirm_button.on_click
                def _on_confirm(_: object) -> None:
                    self._submit(ShutdownCommand())
                    modal.close()

                @cancel_button.on_click
                def _on_cancel(_: object) -> None:
                    modal.close()

    # ------------------------------------------------------------------
    # Object dragging (Viser event loop)
    # ------------------------------------------------------------------

    def _make_gizmo_callback(self, joint: str) -> Callable[[object], Any]:
        # An async callback keeps start/update/end in order; threadpool
        # callbacks may run out of order and release a hold too early.
        async def _on_gizmo(event: object) -> None:  # noqa: RUF029
            target = event.target  # type: ignore[attr-defined]
            pose = ObjectPose(
                position=tuple(float(v) for v in target.position),  # type: ignore[arg-type]
                wxyz=tuple(float(v) for v in target.wxyz),  # type: ignore[arg-type]
            )
            phase = getattr(event, "phase", "update")
            with self._drags_lock:
                if phase == "end":
                    self._drags.pop(joint, None)
                else:
                    self._drags[joint] = (getattr(event, "client_id", None), pose)
            self._submit(
                SetObjectPoseCommand(joint=joint, position=pose.position, wxyz=pose.wxyz, hold=phase != "end"),
            )

        return _on_gizmo

    def _on_client_disconnect(self, client: object) -> None:
        """Release objects held by a browser that went away mid-drag."""
        client_id = getattr(client, "client_id", None)
        with self._drags_lock:
            released = [(joint, pose) for joint, (owner, pose) in self._drags.items() if owner == client_id]
            for joint, _ in released:
                del self._drags[joint]
        for joint, pose in released:
            self._submit(SetObjectPoseCommand(joint=joint, position=pose.position, wxyz=pose.wxyz, hold=False))

    # ------------------------------------------------------------------
    # Refresh (sim thread)
    # ------------------------------------------------------------------

    def refresh(self, state: PanelState, now: float) -> None:
        """Mirror sim state into the panel, each part at its own rate."""
        self._follow(state, now)
        if now >= self._next_status:
            self._next_status = now + _STATUS_PERIOD_S
            self._refresh_status(state)
        if now >= self._next_objects:
            self._next_objects = now + _OBJECT_PERIOD_S
            self._refresh_gizmos(state)
        if self._show_previews and now >= self._next_preview:
            self._next_preview = now + _PREVIEW_PERIOD_S
            self._refresh_previews(state)

    def _follow(self, state: PanelState, now: float) -> None:
        """Move every viewer camera along with the followed body (every tick)."""
        followed = self.camera.name
        if followed is None:
            return
        target = state.follow_targets.get(followed)
        dragged = self.dragged_joints()
        hold = any(state.object_bodies.get(joint) == followed for joint in dragged)
        clients = self._server.get_clients()
        views: dict[int, CameraView] = {}
        for client_id, client in clients.items():
            # A client that has not reported its camera yet has no view to move.
            with contextlib.suppress(AssertionError, AttributeError):
                views[client_id] = (np.asarray(client.camera.position), np.asarray(client.camera.look_at))
        target_array = None if target is None else np.asarray(target, dtype=np.float64)
        result = self.camera.step(views, target_array, hold=hold, now=now)
        if result.released:
            logger.info("Viewer panned away from '{}'; camera follow stopped", followed)
            if self._handles.follow_dropdown is not None:
                self._handles.follow_dropdown.value = NO_FOLLOW_LABEL
            return
        for client_id, shift in result.shifts.items():
            position, _ = views[client_id]
            # Setting the position moves look_at by the same amount.
            clients[client_id].camera.position = position + shift

    def _refresh_status(self, state: PanelState) -> None:
        handles = self._handles
        if handles.scene_dropdown is not None:
            label = _scene_label(state)
            if label in handles.scene_dropdown.options:
                handles.scene_dropdown.value = label
        if handles.fixed_seed is not None:
            handles.fixed_seed.value = state.seed is not None
            handles.seed_number.disabled = state.seed is None
            if state.seed is not None:
                handles.seed_number.value = state.seed
        if handles.auto_reset is not None:
            episode = state.episode
            handles.auto_reset.value = bool(episode.get("active", True))
            with contextlib.suppress(TypeError, ValueError):
                handles.dwell.value = float(episode.get("success_dwell_s", handles.dwell.value))
            handles.episode_status.content = _episode_markdown(episode)

    def _refresh_gizmos(self, state: PanelState) -> None:
        if not self._drag_objects or not self._handles.gizmos:
            return
        dragged = self.dragged_joints()
        for joint, gizmo in self._handles.gizmos.items():
            pose = state.objects.get(joint)
            if pose is None or joint in dragged:
                continue
            gizmo.position = pose.position
            gizmo.wxyz = pose.wxyz

    def _refresh_previews(self, state: PanelState) -> None:
        handles = self._handles
        for name, image in handles.previews.items():
            buffer = state.cameras.get(name)
            sample = buffer.snapshot() if buffer is not None else None
            if sample is None or handles.preview_seq.get(name) == sample.seq:
                continue
            handles.preview_seq[name] = sample.seq
            try:
                image.image = _downscale(sample.frame)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Camera preview update failed for '{}': {}", name, exc)
