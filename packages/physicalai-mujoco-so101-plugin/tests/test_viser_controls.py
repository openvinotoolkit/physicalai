"""Tests for the Viser simulation control panel, using fake Viser handles."""

from __future__ import annotations

import asyncio
import contextlib
from types import SimpleNamespace
from typing import Any, ClassVar
from unittest.mock import MagicMock

import numpy as np
import pytest

from physicalai_mujoco_so101_plugin.http_server import (
    MAX_DWELL_S,
    MAX_SEED,
    MIN_DWELL_S,
    FrameBuffer,
    HomeCommand,
    ResetCommand,
    SetAutoResetCommand,
    SetObjectPoseCommand,
    SetSeedCommand,
    ShutdownCommand,
    SwitchSceneCommand,
)
from physicalai_mujoco_so101_plugin.viser_controls import (
    CUSTOM_MODEL_LABEL,
    NO_FOLLOW_LABEL,
    CameraFollow,
    CameraView,
    FollowStep,
    ObjectPose,
    PanelState,
    SimControlPanel,
    default_camera_position,
)

_BROWSER = object()


class FakeHandle:
    """Stand-in for a Viser GUI/scene handle that records callbacks."""

    def __init__(self, label: str | None = None, value: Any = None, **kwargs: Any) -> None:  # noqa: ANN401
        self.label = label
        self.value = value
        self.kwargs = kwargs
        self.callbacks: list[Any] = []
        self.disabled = kwargs.get("disabled", False)
        self.visible = kwargs.get("visible", True)
        self.options = tuple(kwargs.get("options", ()))
        self.content = kwargs.get("content")
        self.image = kwargs.get("image")
        self.position = kwargs.get("position")
        self.wxyz = kwargs.get("wxyz")

    def on_update(self, func: Any) -> Any:  # noqa: ANN401
        self.callbacks.append(func)
        return func

    on_click = on_update

    def fire(self, value: Any = ..., *, client: object | None = _BROWSER) -> None:  # noqa: ANN401
        """Deliver an input event, from a browser unless ``client`` is ``None``."""
        if value is not ...:
            self.value = value
        for callback in self.callbacks:
            callback(SimpleNamespace(client=client, target=self))


class FakeGui:
    def __init__(self) -> None:
        self.handles: dict[str, FakeHandle] = {}
        self.folders: list[str] = []

    def _add(self, label: str, handle: FakeHandle) -> FakeHandle:
        self.handles[label] = handle
        return handle

    def add_folder(self, label: str) -> contextlib.AbstractContextManager[None]:
        self.folders.append(label)
        return contextlib.nullcontext()

    def add_dropdown(self, label: str, *, options: list[str], initial_value: str, hint: str | None = None) -> FakeHandle:
        return self._add(label, FakeHandle(label, initial_value, options=options, hint=hint))

    def add_button(self, label: str, **kwargs: Any) -> FakeHandle:  # noqa: ANN401
        return self._add(label, FakeHandle(label, **kwargs))

    def add_checkbox(self, label: str, *, initial_value: bool, hint: str | None = None) -> FakeHandle:
        return self._add(label, FakeHandle(label, initial_value, hint=hint))

    def add_number(self, label: str, *, initial_value: float, **kwargs: Any) -> FakeHandle:  # noqa: ANN401
        return self._add(label, FakeHandle(label, initial_value, **kwargs))

    def add_slider(self, label: str, *, initial_value: float, **kwargs: Any) -> FakeHandle:  # noqa: ANN401
        return self._add(label, FakeHandle(label, initial_value, **kwargs))

    def add_markdown(self, content: str) -> FakeHandle:
        return self._add("markdown", FakeHandle("markdown", content=content))

    def add_image(self, image: np.ndarray, *, label: str, **kwargs: Any) -> FakeHandle:  # noqa: ANN401
        return self._add(f"image:{label}", FakeHandle(label, image=image, **kwargs))


class FakeScene:
    def __init__(self) -> None:
        self.nodes: dict[str, FakeHandle] = {}

    def add_transform_controls(self, name: str, **kwargs: Any) -> FakeHandle:  # noqa: ANN401
        handle = FakeHandle(name, **kwargs)
        self.nodes[name] = handle
        return handle


class FakeServer:
    def __init__(self) -> None:
        self.gui = FakeGui()
        self.scene = FakeScene()
        self.disconnect_callbacks: list[Any] = []
        self.connect_callbacks: list[Any] = []
        self.clients: dict[int, Any] = {}

    def on_client_disconnect(self, callback: Any) -> Any:  # noqa: ANN401
        self.disconnect_callbacks.append(callback)
        return callback

    def on_client_connect(self, callback: Any) -> Any:  # noqa: ANN401
        self.connect_callbacks.append(callback)
        return callback

    def get_clients(self) -> dict[int, Any]:
        return dict(self.clients)


class FakeCamera:
    """Mimics viser: setting ``position`` moves ``look_at`` by the same amount."""

    def __init__(self, look_at: tuple[float, float, float]) -> None:
        self.fov: float | None = None
        self.look_at = np.asarray(look_at, dtype=np.float64)
        self._position = self.look_at + np.array([1.0, 0.0, 0.0])

    @property
    def position(self) -> np.ndarray:
        return self._position

    @position.setter
    def position(self, value: np.ndarray) -> None:
        value = np.asarray(value, dtype=np.float64)
        self.look_at = self.look_at + (value - self._position)
        self._position = value


def make_client(look_at: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> SimpleNamespace:
    return SimpleNamespace(camera=FakeCamera(look_at))


def make_state(**overrides: Any) -> PanelState:  # noqa: ANN401
    values: dict[str, Any] = {
        "scene_id": "single_pick_place",
        "scene_options": (("pick_lift", "Pick & Lift"), ("single_pick_place", "Single Pick & Place")),
        "seed": None,
        "episode": {
            "enabled": True,
            "active": True,
            "phase": "idle",
            "episode_count": 0,
            "success_dwell_s": 5.0,
            "countdown_s": None,
        },
        "objects": {"block1:joint": ObjectPose(position=(0.2, 0.0, 0.02), wxyz=(1.0, 0.0, 0.0, 0.0))},
        "cameras": {},
        "follow_targets": {"block1": (0.2, 0.0, 0.02), "target": (0.3, 0.1, 0.0), "gripper": (0.1, 0.0, 0.2)},
        "object_bodies": {"block1:joint": "block1"},
        "view_extent": 0.5,
    }
    values.update(overrides)
    return PanelState(**values)


def build_panel(state: PanelState | None = None) -> tuple[SimControlPanel, FakeServer, list[object]]:
    commands: list[object] = []
    server = FakeServer()
    panel = SimControlPanel(server, SimpleNamespace(Icon=MagicMock()), commands.append)
    state = state or make_state()
    panel.build(state)
    panel.build_camera_tab(state)
    panel.build_scene_nodes(state)
    return panel, server, commands


def run_gizmo(gizmo: FakeHandle, phase: str, position: tuple[float, float, float], client_id: int = 1) -> None:
    gizmo.position = position
    gizmo.wxyz = (1.0, 0.0, 0.0, 0.0)
    asyncio.run(gizmo.callbacks[0](SimpleNamespace(target=gizmo, phase=phase, client_id=client_id)))


class TestSceneControls:
    def test_dropdown_lists_compatible_scenes_and_selects_the_current_one(self) -> None:
        _, server, _ = build_panel()
        dropdown = server.gui.handles["Scene"]
        assert dropdown.options == ("Pick & Lift", "Single Pick & Place")
        assert dropdown.value == "Single Pick & Place"

    def test_selecting_another_scene_enqueues_a_switch(self) -> None:
        _, server, commands = build_panel()
        server.gui.handles["Scene"].fire("Pick & Lift")
        assert commands == [SwitchSceneCommand(scene_id="pick_lift")]

    def test_reselecting_the_current_scene_does_nothing(self) -> None:
        _, server, commands = build_panel()
        server.gui.handles["Scene"].fire("Single Pick & Place")
        assert commands == []

    def test_server_side_value_sync_is_not_echoed(self) -> None:
        _, server, commands = build_panel()
        server.gui.handles["Scene"].fire("Pick & Lift", client=None)
        assert commands == []

    def test_custom_model_is_listed_but_not_switchable(self) -> None:
        _, server, commands = build_panel(make_state(scene_id=None))
        dropdown = server.gui.handles["Scene"]
        assert dropdown.options[0] == CUSTOM_MODEL_LABEL
        assert dropdown.value == CUSTOM_MODEL_LABEL
        dropdown.fire(CUSTOM_MODEL_LABEL)
        assert commands == []

    def test_reset_and_home_buttons(self) -> None:
        _, server, commands = build_panel()
        server.gui.handles["Reset Scene"].fire()
        server.gui.handles["Home Arm"].fire()
        assert commands == [ResetCommand(), HomeCommand()]


class TestSeedControls:
    def test_seed_input_starts_disabled_without_a_fixed_seed(self) -> None:
        _, server, _ = build_panel()
        assert server.gui.handles["Fixed seed"].value is False
        assert server.gui.handles["Seed"].disabled is True

    def test_fixing_and_clearing_the_seed(self) -> None:
        _, server, commands = build_panel()
        server.gui.handles["Seed"].value = 42
        server.gui.handles["Fixed seed"].fire(True)
        assert server.gui.handles["Seed"].disabled is False
        server.gui.handles["Fixed seed"].fire(False)
        assert commands == [SetSeedCommand(seed=42), SetSeedCommand(seed=None)]

    def test_seed_edits_apply_only_while_fixed(self) -> None:
        _, server, commands = build_panel(make_state(seed=3))
        server.gui.handles["Seed"].fire(9)
        server.gui.handles["Fixed seed"].value = False
        server.gui.handles["Seed"].fire(10)
        assert commands == [SetSeedCommand(seed=9)]

    def test_seed_is_clamped(self) -> None:
        _, server, commands = build_panel(make_state(seed=3))
        server.gui.handles["Seed"].fire(MAX_SEED + 10)
        server.gui.handles["Seed"].fire(-5)
        assert commands == [SetSeedCommand(seed=MAX_SEED), SetSeedCommand(seed=0)]


class TestEpisodeControls:
    def test_hidden_when_the_scene_has_no_auto_reset(self) -> None:
        _, server, _ = build_panel(make_state(episode={"enabled": False}))
        assert "Episode" not in server.gui.folders
        assert "Auto-reset" not in server.gui.handles

    def test_toggle_and_dwell_enqueue_commands(self) -> None:
        _, server, commands = build_panel()
        server.gui.handles["Auto-reset"].fire(False)
        server.gui.handles["Success dwell (s)"].fire(2.5)
        server.gui.handles["Success dwell (s)"].fire(MAX_DWELL_S * 2)
        server.gui.handles["Success dwell (s)"].fire(0.0)
        assert commands == [
            SetAutoResetCommand(enabled=False),
            SetAutoResetCommand(dwell_s=2.5),
            SetAutoResetCommand(dwell_s=MAX_DWELL_S),
            SetAutoResetCommand(dwell_s=MIN_DWELL_S),
        ]

    def test_status_shows_countdown_and_count(self) -> None:
        panel, server, _ = build_panel()
        episode = {
            "enabled": True,
            "active": True,
            "phase": "success_hold",
            "countdown_s": 3.25,
            "episode_count": 4,
            "success_dwell_s": 5.0,
        }
        panel.refresh(make_state(episode=episode), now=1.0)
        content = server.gui.handles["markdown"].content
        assert "respawning in 3.2 s" in content
        assert "**Episodes:** 4" in content

    def test_status_shows_paused(self) -> None:
        panel, server, _ = build_panel()
        episode = {"enabled": True, "active": False, "phase": "idle", "episode_count": 0, "success_dwell_s": 5.0}
        panel.refresh(make_state(episode=episode), now=1.0)
        assert "paused" in server.gui.handles["markdown"].content
        assert server.gui.handles["Auto-reset"].value is False


class TestRefresh:
    def test_mirrors_scene_seed_and_episode_settings(self) -> None:
        panel, server, commands = build_panel()
        episode = {"enabled": True, "active": True, "phase": "idle", "episode_count": 0, "success_dwell_s": 8.0}
        panel.refresh(make_state(scene_id="pick_lift", seed=5, episode=episode), now=1.0)

        assert server.gui.handles["Scene"].value == "Pick & Lift"
        assert server.gui.handles["Fixed seed"].value is True
        assert server.gui.handles["Seed"].value == 5
        assert server.gui.handles["Seed"].disabled is False
        assert server.gui.handles["Success dwell (s)"].value == 8.0
        assert commands == []

    def test_status_is_rate_limited(self) -> None:
        panel, server, _ = build_panel()
        panel.refresh(make_state(seed=1), now=10.0)
        panel.refresh(make_state(seed=2), now=10.1)
        assert server.gui.handles["Seed"].value == 1
        panel.refresh(make_state(seed=2), now=10.6)
        assert server.gui.handles["Seed"].value == 2


class TestObjectDragging:
    def test_gizmos_start_hidden_at_the_object_pose(self) -> None:
        _, server, _ = build_panel()
        gizmo = server.scene.nodes["/fixed_bodies/drag_block1_joint"]
        assert gizmo.visible is False
        assert gizmo.position == (0.2, 0.0, 0.02)

    def test_no_object_controls_without_free_objects(self) -> None:
        _, server, _ = build_panel(make_state(objects={}))
        assert "Drag objects" not in server.gui.handles
        assert server.scene.nodes == {}

    def test_toggle_shows_gizmos_and_persists_across_rebuilds(self) -> None:
        panel, server, _ = build_panel()
        server.gui.handles["Drag objects"].fire(True)
        assert server.scene.nodes["/fixed_bodies/drag_block1_joint"].visible is True

        server.gui = FakeGui()
        server.scene = FakeScene()
        panel.build(make_state())
        panel.build_scene_nodes(make_state())
        assert server.gui.handles["Drag objects"].value is True
        assert server.scene.nodes["/fixed_bodies/drag_block1_joint"].visible is True

    def test_drag_holds_until_release(self) -> None:
        panel, server, commands = build_panel()
        gizmo = server.scene.nodes["/fixed_bodies/drag_block1_joint"]

        run_gizmo(gizmo, "start", (0.2, 0.0, 0.1))
        run_gizmo(gizmo, "update", (0.25, 0.05, 0.1))
        assert panel.dragged_joints() == {"block1:joint"}
        run_gizmo(gizmo, "end", (0.25, 0.05, 0.1))

        assert [command.hold for command in commands] == [True, True, False]
        assert commands[-1] == SetObjectPoseCommand(
            joint="block1:joint",
            position=(0.25, 0.05, 0.1),
            wxyz=(1.0, 0.0, 0.0, 0.0),
            hold=False,
        )
        assert panel.dragged_joints() == frozenset()

    def test_refresh_follows_objects_except_while_dragged(self) -> None:
        objects = {
            "a:joint": ObjectPose(position=(0.1, 0.0, 0.02), wxyz=(1.0, 0.0, 0.0, 0.0)),
            "b:joint": ObjectPose(position=(0.2, 0.0, 0.02), wxyz=(1.0, 0.0, 0.0, 0.0)),
        }
        panel, server, _ = build_panel(make_state(objects=objects))
        server.gui.handles["Drag objects"].fire(True)
        run_gizmo(server.scene.nodes["/fixed_bodies/drag_a_joint"], "start", (0.3, 0.3, 0.3))

        moved = {name: ObjectPose(position=(0.0, 0.1, 0.02), wxyz=(1.0, 0.0, 0.0, 0.0)) for name in objects}
        panel.refresh(make_state(objects=moved), now=1.0)

        assert server.scene.nodes["/fixed_bodies/drag_a_joint"].position == (0.3, 0.3, 0.3)
        assert server.scene.nodes["/fixed_bodies/drag_b_joint"].position == (0.0, 0.1, 0.02)

    def test_disconnect_releases_that_clients_holds(self) -> None:
        panel, server, commands = build_panel()
        run_gizmo(server.scene.nodes["/fixed_bodies/drag_block1_joint"], "start", (0.2, 0.0, 0.1), client_id=7)
        commands.clear()

        server.disconnect_callbacks[0](SimpleNamespace(client_id=8))
        assert commands == []
        server.disconnect_callbacks[0](SimpleNamespace(client_id=7))

        assert commands == [
            SetObjectPoseCommand(joint="block1:joint", position=(0.2, 0.0, 0.1), wxyz=(1.0, 0.0, 0.0, 0.0), hold=False),
        ]
        assert panel.dragged_joints() == frozenset()


class TestCameraPreviews:
    @staticmethod
    def _state_with_frame(frame: np.ndarray) -> tuple[PanelState, FrameBuffer]:
        buffer = FrameBuffer("overview")
        buffer.put(frame)
        return make_state(cameras={"overview": buffer, "wrist": None}), buffer

    def test_previews_are_hidden_and_not_streamed_by_default(self) -> None:
        state, _ = self._state_with_frame(np.zeros((480, 640, 3), dtype=np.uint8))
        panel, server, _ = build_panel(state)
        image = server.gui.handles["image:overview"]
        placeholder = image.image
        assert image.visible is False

        panel.refresh(state, now=1.0)
        assert image.image is placeholder

    def test_enabled_previews_stream_downscaled_new_frames(self) -> None:
        state, buffer = self._state_with_frame(np.full((480, 640, 3), 200, dtype=np.uint8))
        panel, server, _ = build_panel(state)
        server.gui.handles["Show previews"].fire(True)
        image = server.gui.handles["image:overview"]
        assert image.visible is True
        assert server.gui.handles["image:wrist"].visible is True

        panel.refresh(state, now=1.0)
        assert image.image.shape == (240, 320, 3)

        sentinel = object()
        image.image = sentinel
        panel.refresh(state, now=2.0)
        assert image.image is sentinel  # unchanged frame is not re-sent

        buffer.put(np.zeros((480, 640, 3), dtype=np.uint8))
        panel.refresh(state, now=3.0)
        assert image.image is not sentinel

    def test_no_camera_section_without_cameras(self) -> None:
        _, server, _ = build_panel(make_state(cameras={}))
        assert "Show previews" not in server.gui.handles


def view(look_at: tuple[float, float, float], back: float = 1.0) -> CameraView:
    target = np.asarray(look_at, dtype=np.float64)
    return target + np.array([back, 0.0, 0.0]), target


def settled(follow: CameraFollow, look_at: tuple[float, float, float], now: float = 0.0) -> None:
    """Start following with the camera already on the body and the glide finished."""
    target = np.asarray(look_at, dtype=np.float64)
    follow.step({1: view(look_at)}, target, hold=False, now=now)
    follow.step({1: view(look_at)}, target, hold=False, now=now + 10.0)


class TestCameraFollow:
    def test_free_camera_by_default_moves_nothing(self) -> None:
        follow = CameraFollow()
        assert follow.name is None
        assert follow.step({1: view((0, 0, 0))}, np.array([1.0, 0, 0]), hold=False, now=0.0) == FollowStep()

    def test_follow_glides_each_camera_onto_the_body(self) -> None:
        follow = CameraFollow(blend_s=1.0)
        follow.select("block1")
        views = {1: view((0, 0, 0)), 2: view((0, 2, 0))}
        target = np.array([1.0, 0.0, 0.0])

        assert follow.step(views, target, hold=False, now=0.0).shifts == {}  # starts where each camera looks

        halfway = follow.step(views, target, hold=False, now=0.5).shifts
        np.testing.assert_allclose(halfway[1], [0.5, 0, 0])
        np.testing.assert_allclose(halfway[2], [0.5, -1.0, 0])

        # The cameras moved as told (look-at now on the glide path).
        views = {1: view((0.5, 0, 0)), 2: view((0.5, 1.0, 0))}
        done = follow.step(views, target, hold=False, now=1.0).shifts
        np.testing.assert_allclose(done[1], [0.5, 0, 0])
        np.testing.assert_allclose(done[2], [0.5, -1.0, 0])

    def test_cameras_track_a_moving_body(self) -> None:
        follow = CameraFollow(blend_s=0.1)
        follow.select("block1")
        settled(follow, (0, 0, 0))

        shifts = follow.step({1: view((0, 0, 0))}, np.array([0.1, 0.2, 0.0]), hold=False, now=20.0).shifts
        np.testing.assert_allclose(shifts[1], [0.1, 0.2, 0.0])

    def test_orbit_and_zoom_keep_following(self) -> None:
        follow = CameraFollow(blend_s=0.1)
        follow.select("block1")
        settled(follow, (0.1, 0.2, 0.0))

        # Orbiting/zooming moves the camera but not its look-at point.
        result = follow.step({1: view((0.1, 0.2, 0.0), back=5.0)}, np.array([0.1, 0.2, 0.0]), hold=False, now=20.0)
        assert result == FollowStep()
        assert follow.name == "block1"

    def test_a_pan_stops_following(self) -> None:
        follow = CameraFollow(blend_s=0.1)
        follow.select("block1")
        settled(follow, (0, 0, 0))

        result = follow.step({1: view((0.3, 0, 0))}, np.zeros(3), hold=False, now=20.0)
        assert result.released is True
        assert result.shifts == {}
        assert follow.name is None
        assert follow.step({1: view((0.3, 0, 0))}, np.zeros(3), hold=False, now=21.0) == FollowStep()

    def test_a_small_drift_below_the_threshold_is_not_a_pan(self) -> None:
        follow = CameraFollow(blend_s=0.1, pan_release_m=0.02)
        follow.select("block1")
        settled(follow, (0, 0, 0))
        result = follow.step({1: view((0.01, 0, 0))}, np.zeros(3), hold=False, now=20.0)
        assert result.released is False
        np.testing.assert_allclose(result.shifts[1], [-0.01, 0, 0])

    def test_a_lagging_report_along_the_steered_path_is_not_a_pan(self) -> None:
        follow = CameraFollow(blend_s=0.1, pan_release_m=0.01)
        follow.select("block1")
        settled(follow, (0, 0, 0))
        # The body moves quickly and the browser lags: each report is a point it
        # was sent a few ticks ago, or partway between two of them.
        reported = 0.0
        for index, x in enumerate((0.1, 0.2, 0.3, 0.4)):
            follow.step({1: view((reported, 0, 0))}, np.array([x, 0.0, 0.0]), hold=False, now=20.0 + 0.02 * index)
            reported = max(0.0, x - 0.2)
        result = follow.step({1: view((0.15, 0, 0))}, np.array([0.5, 0.0, 0.0]), hold=False, now=20.1)
        assert result.released is False
        assert follow.name == "block1"

    def test_old_path_points_expire(self) -> None:
        follow = CameraFollow(blend_s=0.1, pan_release_m=0.01, history_s=0.5)
        follow.select("block1")
        settled(follow, (0, 0, 0))
        follow.step({1: view((0, 0, 0))}, np.array([1.0, 0, 0]), hold=False, now=30.0)
        follow.step({1: view((1.0, 0, 0))}, np.array([1.0, 0, 0]), hold=False, now=30.8)
        # The origin has aged out of the path, so a look-at back there is a pan.
        assert follow.step({1: view((0, 0, 0))}, np.array([1.0, 0, 0]), hold=False, now=31.0).released is True

    def test_a_viewer_opened_mid_follow_is_not_a_pan(self) -> None:
        follow = CameraFollow(blend_s=0.1)
        follow.select("block1")
        settled(follow, (0, 0, 0))
        views = {1: view((0, 0, 0)), 2: view((5.0, 5.0, 5.0))}
        result = follow.step(views, np.zeros(3), hold=False, now=20.0)
        assert result.released is False
        np.testing.assert_allclose(result.shifts[2], [-5.0, -5.0, -5.0])

    def test_hold_keeps_cameras_still_then_glides(self) -> None:
        follow = CameraFollow(blend_s=1.0)
        follow.select("block1")
        settled(follow, (0, 0, 0))

        moved = np.array([1.0, 0.0, 0.0])
        assert follow.step({1: view((0, 0, 0))}, moved, hold=True, now=16.0) == FollowStep()
        assert follow.step({1: view((0, 0, 0))}, moved, hold=False, now=17.0).shifts == {}  # glide starts
        np.testing.assert_allclose(follow.step({1: view((0, 0, 0))}, moved, hold=False, now=17.5).shifts[1], [0.5, 0, 0])

    def test_panning_while_the_followed_object_is_dragged_stops_following(self) -> None:
        follow = CameraFollow(blend_s=0.1)
        follow.select("block1")
        settled(follow, (0, 0, 0))
        assert follow.step({1: view((0.5, 0, 0))}, np.ones(3), hold=True, now=20.0).released is True

    def test_stopping_follow_leaves_cameras_where_they_are(self) -> None:
        follow = CameraFollow()
        follow.select("block1")
        follow.step({1: view((0, 0, 0))}, np.ones(3), hold=False, now=0.0)
        follow.select(None)
        assert follow.step({1: view((0, 0, 0))}, np.ones(3), hold=False, now=1.0) == FollowStep()
        assert follow.last_target is None

    def test_reselecting_does_not_restart_the_glide(self) -> None:
        follow = CameraFollow(blend_s=1.0)
        follow.select("block1")
        follow.step({1: view((0, 0, 0))}, np.array([2.0, 0, 0]), hold=False, now=0.0)
        follow.select("block1")
        shifts = follow.step({1: view((0, 0, 0))}, np.array([2.0, 0, 0]), hold=False, now=0.5).shifts
        np.testing.assert_allclose(shifts[1], [1.0, 0, 0])

    def test_last_target_tracks_the_body(self) -> None:
        follow = CameraFollow()
        follow.select("block1")
        follow.step({}, np.array([0.1, 0.2, 0.3]), hold=False, now=0.0)
        np.testing.assert_allclose(follow.last_target, [0.1, 0.2, 0.3])


class TestCameraTab:
    TARGETS: ClassVar[dict[str, tuple[float, float, float]]] = {
        "block1": (0.2, 0.0, 0.02),
        "target": (0.3, 0.1, 0.0),
        "gripper": (0.1, 0.0, 0.2),
    }

    def test_follow_lists_none_then_bodies(self) -> None:
        _, server, _ = build_panel()
        dropdown = server.gui.handles["Follow"]
        assert dropdown.options == (NO_FOLLOW_LABEL, "block1", "target", "gripper")
        assert dropdown.value == NO_FOLLOW_LABEL

    def test_choosing_a_body_is_viewer_state_not_a_command(self) -> None:
        panel, server, commands = build_panel()
        server.gui.handles["Follow"].fire("block1")
        assert panel.camera.name == "block1"
        server.gui.handles["Follow"].fire(NO_FOLLOW_LABEL)
        assert panel.camera.name is None
        assert commands == []

    def test_server_side_sync_is_ignored(self) -> None:
        panel, server, _ = build_panel()
        server.gui.handles["Follow"].fire("block1", client=None)
        assert panel.camera.name is None

    def test_refresh_moves_viewer_cameras_not_the_world(self) -> None:
        panel, server, _ = build_panel()
        client = make_client(look_at=(0.2, 0.0, 0.02))
        server.clients = {1: client}
        server.gui.handles["Follow"].fire("block1")
        panel.refresh(make_state(), now=0.0)

        moved = dict(self.TARGETS, block1=(0.25, 0.05, 0.02))
        panel.refresh(make_state(follow_targets=moved), now=10.0)
        np.testing.assert_allclose(client.camera.look_at, [0.25, 0.05, 0.02])
        np.testing.assert_allclose(client.camera.position, [1.25, 0.05, 0.02])

    def test_dragging_the_followed_object_holds_the_camera(self) -> None:
        panel, server, _ = build_panel()
        client = make_client(look_at=(0.2, 0.0, 0.02))
        server.clients = {1: client}
        server.gui.handles["Follow"].fire("block1")
        panel.refresh(make_state(), now=0.0)
        run_gizmo(server.scene.nodes["/fixed_bodies/drag_block1_joint"], "start", (0.4, 0.0, 0.1))

        panel.refresh(make_state(follow_targets=dict(self.TARGETS, block1=(0.4, 0.0, 0.1))), now=10.0)
        np.testing.assert_allclose(client.camera.look_at, [0.2, 0.0, 0.02])

    def test_panning_a_viewer_stops_following_and_resets_the_dropdown(self) -> None:
        panel, server, commands = build_panel()
        client = make_client(look_at=(0.2, 0.0, 0.02))
        server.clients = {1: client}
        server.gui.handles["Follow"].fire("block1")
        panel.refresh(make_state(), now=0.0)
        panel.refresh(make_state(), now=10.0)

        client.camera.look_at = np.array([0.6, 0.0, 0.02])  # the user panned
        panel.refresh(make_state(), now=11.0)

        assert panel.camera.name is None
        assert server.gui.handles["Follow"].value == NO_FOLLOW_LABEL
        np.testing.assert_allclose(client.camera.look_at, [0.6, 0.0, 0.02])  # left where the user put it
        assert commands == []

    def test_orbiting_a_viewer_keeps_following(self) -> None:
        panel, server, _ = build_panel()
        client = make_client(look_at=(0.2, 0.0, 0.02))
        server.clients = {1: client}
        server.gui.handles["Follow"].fire("block1")
        panel.refresh(make_state(), now=0.0)
        panel.refresh(make_state(), now=10.0)

        client.camera._position = np.array([0.2, 1.0, 0.5])  # orbit: look-at unchanged
        panel.refresh(make_state(follow_targets=dict(self.TARGETS, block1=(0.25, 0.0, 0.02))), now=11.0)

        assert panel.camera.name == "block1"
        np.testing.assert_allclose(client.camera.look_at, [0.25, 0.0, 0.02])

    def test_clients_without_a_camera_yet_are_skipped(self) -> None:
        panel, server, _ = build_panel()
        broken = SimpleNamespace(camera=SimpleNamespace())
        server.clients = {1: broken}
        server.gui.handles["Follow"].fire("block1")
        panel.refresh(make_state(), now=0.0)

    def test_follow_persists_across_rebuilds_when_the_body_exists(self) -> None:
        panel, server, _ = build_panel()
        server.gui.handles["Follow"].fire("target")
        server.gui = FakeGui()
        panel.build_camera_tab(make_state(follow_targets={"block1": (0, 0, 0), "target": (0, 0, 0)}))
        assert panel.camera.name == "target"
        assert server.gui.handles["Follow"].value == "target"

    def test_follow_stops_when_the_new_scene_lacks_the_body(self) -> None:
        panel, server, _ = build_panel()
        server.gui.handles["Follow"].fire("block1")
        server.gui = FakeGui()
        panel.build_camera_tab(make_state(follow_targets={"obj1": (0, 0, 0)}))
        assert panel.camera.name is None
        assert server.gui.handles["Follow"].value == NO_FOLLOW_LABEL

    def test_new_clients_look_at_the_model_centre_with_the_chosen_fov(self) -> None:
        _, server, _ = build_panel(make_state(view_center=(0.1, 0.0, 0.2), view_extent=2.0))
        server.gui.handles["FOV (°)"].fire(90.0)
        client = make_client()

        assert len(server.connect_callbacks) == 1
        server.connect_callbacks[0](client)

        assert client.camera.fov == pytest.approx(np.deg2rad(90.0))
        np.testing.assert_allclose(client.camera.look_at, [0.1, 0.0, 0.2])
        np.testing.assert_allclose(client.camera.position, np.array([0.1, 0.0, 0.2]) + default_camera_position(2.0))

    def test_new_clients_look_at_the_followed_body(self) -> None:
        panel, server, _ = build_panel()
        server.gui.handles["Follow"].fire("gripper")
        panel.refresh(make_state(), now=0.0)
        client = make_client()
        server.connect_callbacks[0](client)
        np.testing.assert_allclose(client.camera.look_at, self.TARGETS["gripper"])

    def test_rebuilds_do_not_add_connect_hooks(self) -> None:
        panel, server, _ = build_panel()
        panel.build_camera_tab(make_state())
        panel.build_camera_tab(make_state())
        assert len(server.connect_callbacks) == 1

    def test_fov_applies_to_open_viewers_and_is_clamped(self) -> None:
        _, server, _ = build_panel()
        server.clients = {1: make_client(), 2: make_client()}
        server.gui.handles["FOV (°)"].fire(500.0)
        for client in server.clients.values():
            assert client.camera.fov == pytest.approx(np.deg2rad(150.0))

    def test_reset_view_reaims_open_viewers(self) -> None:
        _, server, _ = build_panel()
        client = make_client(look_at=(1.0, 1.0, 1.0))
        server.clients = {1: client}
        server.gui.handles["Reset View"].fire()
        np.testing.assert_allclose(client.camera.look_at, np.zeros(3))
        np.testing.assert_allclose(client.camera.position, default_camera_position(0.5))

    def test_default_camera_position_matches_mjviser(self) -> None:
        position = default_camera_position(1.0)
        assert np.linalg.norm(position) == pytest.approx(3.0)
        assert position[2] == pytest.approx(3.0 * np.sin(np.deg2rad(20.0)))


class TestShutdown:
    @staticmethod
    def _open_modal() -> tuple[list[object], MagicMock, MagicMock, MagicMock]:
        _, server, commands = build_panel()
        client = MagicMock()
        confirm_button, cancel_button = MagicMock(), MagicMock()
        client.gui.add_button.side_effect = [confirm_button, cancel_button]
        server.gui.handles["Shutdown"].fire(client=client)
        return commands, client, confirm_button, cancel_button

    def test_confirm_enqueues_shutdown_and_closes_modal(self) -> None:
        commands, client, confirm_button, _ = self._open_modal()
        confirm_button.on_click.call_args.args[0](MagicMock())
        assert commands == [ShutdownCommand()]
        client.gui.add_modal.return_value.__enter__.return_value.close.assert_called_once()

    def test_cancel_closes_modal_without_enqueueing(self) -> None:
        commands, client, _, cancel_button = self._open_modal()
        cancel_button.on_click.call_args.args[0](MagicMock())
        assert commands == []
        client.gui.add_modal.return_value.__enter__.return_value.close.assert_called_once()

    def test_click_without_client_is_a_noop(self) -> None:
        _, server, commands = build_panel()
        server.gui.handles["Shutdown"].fire(client=None)
        assert commands == []


@pytest.mark.parametrize(
    ("scene_id", "expected"),
    [("single_pick_place", "Single Pick & Place"), ("gone", CUSTOM_MODEL_LABEL)],
)
def test_scene_label(scene_id: str, expected: str) -> None:
    from physicalai_mujoco_so101_plugin.viser_controls import _scene_label  # noqa: PLC0415

    assert _scene_label(make_state(scene_id=scene_id)) == expected
