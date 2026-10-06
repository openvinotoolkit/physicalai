from __future__ import annotations

import queue
import sys
import threading
from unittest.mock import MagicMock, patch

import mujoco
import numpy as np
import pytest

from physicalai_mujoco_plugin.cameras import CameraConfig, CameraService
from physicalai_mujoco_plugin.http_server import ResetCommand
from physicalai_mujoco_plugin.scene_watch import SceneXmlWatcher
from physicalai_mujoco_plugin.viewer import ViewerService

_XML = """<mujoco><worldbody><camera name="overview" pos="0 -1 0.5" xyaxes="1 0 0 0 0.5 1"/>
<body name="b"><freejoint/><geom size="0.1"/></body></worldbody></mujoco>"""


@pytest.fixture
def model_data() -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = mujoco.MjModel.from_xml_string(_XML)
    return model, mujoco.MjData(model)


def _viewer(commands: queue.Queue | None = None, **kwargs) -> ViewerService:
    commands = commands if commands is not None else queue.Queue()
    return ViewerService(
        host=kwargs.pop("host", "127.0.0.1"),
        port=kwargs.pop("port", 9090),
        submit_command=commands.put,
        panel_state=kwargs.pop("panel_state", MagicMock()),
    )


class TestViewerLaunch:
    def test_viser_console_does_not_use_owner_stdout(self, model_data) -> None:
        """Viser's teardown message must survive the owner closing stdout after READY."""
        import rich

        console = rich.get_console()
        original_file = console.file
        try:
            with (
                patch("viser.ViserServer", return_value=MagicMock()) as server_factory,
                patch("mjviser.ViserMujocoScene", return_value=MagicMock()),
            ):
                assert _viewer().open(*model_data) is True
            server_factory.assert_called_once_with(host="127.0.0.1", port=9090, verbose=False)
            assert console.file is sys.stderr
        finally:
            console.file = original_file

    def test_configured_host_is_used(self, model_data) -> None:
        with (
            patch("viser.ViserServer", return_value=MagicMock()) as server_factory,
            patch("mjviser.ViserMujocoScene", return_value=MagicMock()),
        ):
            viewer = _viewer(host="0.0.0.0")  # noqa: S104
            viewer.open(*model_data)

        server_factory.assert_called_once_with(host="0.0.0.0", port=9090, verbose=False)  # noqa: S104
        assert viewer.url == "http://0.0.0.0:9090"

    def test_stops_partially_created_server_on_scene_failure(self, model_data) -> None:
        """A server created before the scene build fails must not leak the port/thread."""
        server = MagicMock()
        viewer = _viewer()
        with (
            patch("viser.ViserServer", return_value=server),
            patch("mjviser.ViserMujocoScene", side_effect=RuntimeError("boom")),
            patch.object(sys, "platform", "darwin"),  # no native fallback
        ):
            assert viewer.open(*model_data) is False

        server.stop.assert_called_once()
        assert viewer.server is None
        assert not viewer.active


class TestViewerRebuild:
    def test_failed_rebuild_clears_the_stale_scene(self, model_data) -> None:
        """A rebuild failure must not leave the old (now-mismatched) scene wired up."""
        viewer = _viewer()
        viewer.server = MagicMock()
        viewer.scene = MagicMock()  # stale scene from before the hot-swap

        with patch("mjviser.ViserMujocoScene", side_effect=RuntimeError("boom")):
            viewer.rebuild(*model_data)

        assert viewer.scene is None

    def test_successful_rebuild_replaces_the_scene(self, model_data) -> None:
        viewer = _viewer()
        viewer.server = MagicMock()
        new_scene = MagicMock()

        with patch("mjviser.ViserMujocoScene", return_value=new_scene):
            viewer.rebuild(*model_data)

        assert viewer.scene is new_scene

    def test_no_viser_server_is_a_noop(self, model_data) -> None:
        viewer = _viewer()
        viewer.rebuild(*model_data)
        assert viewer.scene is None


class TestViewerGui:
    @staticmethod
    def _built(model_data, server: MagicMock | None = None, commands: queue.Queue | None = None) -> ViewerService:
        viewer = _viewer(commands)
        viewer.server = server or MagicMock()
        viewer._model = model_data[0]  # noqa: SLF001
        return viewer

    def test_rebuild_clears_the_previous_gui_and_scene(self, model_data) -> None:
        server = MagicMock()
        viewer = self._built(model_data, server)

        with patch("mjviser.ViserMujocoScene", return_value=MagicMock()):
            for _ in range(3):
                viewer.build_gui()

        assert server.gui.reset.call_count == 3
        assert server.scene.reset.call_count == 3
        # The panel's one camera hook, registered when it was created.
        assert server.on_client_connect.call_count == 1

    def test_mjviser_body_tracking_and_camera_gui_are_replaced(self, model_data) -> None:
        viewer = self._built(model_data)
        scene = MagicMock()

        with patch("mjviser.ViserMujocoScene", return_value=scene):
            viewer.build_gui()

        assert scene.camera_tracking_enabled is False
        scene.create_scene_gui.assert_not_called()
        scene.set_refresh_handler.assert_called_once()

    def test_tab_order(self, model_data) -> None:
        server = MagicMock()
        viewer = self._built(model_data, server)

        with patch("mjviser.ViserMujocoScene", return_value=MagicMock()):
            viewer.build_gui()

        tabs = server.gui.add_tab_group.return_value
        assert [call.args[0] for call in tabs.add_tab.call_args_list] == ["Simulation", "Camera", "Visualization", "Groups"]
        assert viewer.panel is not None

    def test_panel_survives_rebuilds_and_close_drops_it(self, model_data) -> None:
        viewer = self._built(model_data)

        with patch("mjviser.ViserMujocoScene", return_value=MagicMock()):
            viewer.build_gui()
            panel = viewer.panel
            viewer.rebuild(*model_data)

        assert viewer.panel is panel
        viewer.close()
        assert viewer.panel is None

    def test_panel_submits_commands(self, model_data) -> None:
        commands: queue.Queue = queue.Queue()
        viewer = self._built(model_data, commands=commands)
        with patch("mjviser.ViserMujocoScene", return_value=MagicMock()):
            viewer.build_gui()

        viewer.panel._submit(ResetCommand())  # noqa: SLF001

        assert isinstance(commands.get_nowait(), ResetCommand)

    def test_panel_refresh_failure_does_not_stop_the_loop(self, model_data) -> None:
        viewer = _viewer()
        viewer.scene = MagicMock()
        viewer.panel = MagicMock()
        viewer.panel.refresh.side_effect = RuntimeError("boom")

        assert viewer.sync(model_data[1]) is True
        assert viewer.sync(model_data[1]) is True
        assert viewer._panel_failed is True  # noqa: SLF001

    def test_closed_native_viewer_is_reported_once(self, model_data) -> None:
        viewer = _viewer()
        viewer.native = MagicMock()
        viewer.native.is_running.return_value = False

        assert viewer.sync(model_data[1]) is False
        assert viewer.native is None


@pytest.mark.parametrize("size", [{"fps": 0}, {"fps": -5}, {"width": 0}])
def test_camera_config_rejects_non_positive_sizes_and_rates(size: dict[str, int]) -> None:
    with pytest.raises(ValueError, match="positive size and frame rate"):
        CameraConfig("overview", **size)


class TestCameraFrames:
    @staticmethod
    def _stream_one_frame(model_data, rendered: np.ndarray, *, mirror: bool) -> np.ndarray:
        service = CameraService([CameraConfig("overview", mirror_horizontal=mirror)], render_in_thread=False)
        renderer = MagicMock()
        renderer.render.return_value = rendered
        with patch("mujoco.Renderer", return_value=renderer):
            service.start(*model_data)
        service.publish(model_data[1])
        frame = service.frame_buffers["overview"].snapshot().frame
        service.close()
        return frame

    def test_frames_are_streamed_as_rendered(self, model_data) -> None:
        """mujoco.Renderer already returns upright images; flipping again turns them upside down."""
        rendered = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
        np.testing.assert_array_equal(self._stream_one_frame(model_data, rendered, mirror=False), rendered)

    def test_mirror_flips_left_to_right_only(self, model_data) -> None:
        rendered = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
        np.testing.assert_array_equal(self._stream_one_frame(model_data, rendered, mirror=True), rendered[:, ::-1])

    def test_stop_keeps_the_buffers_and_close_drops_them(self, model_data) -> None:
        service = CameraService([CameraConfig("overview")], render_in_thread=False)
        with patch("mujoco.Renderer", return_value=MagicMock()):
            service.start(*model_data)
        assert service.rendering() == {"overview"}

        service.stop()
        assert service.rendering() == set()
        assert "overview" in service.frame_buffers
        service.close()
        assert service.frame_buffers == {}

    def test_stop_reports_a_camera_thread_that_is_still_rendering(self, model_data) -> None:
        """A render stuck past the stop timeout still reads the model; a later stop waits for it again."""
        rendering, release = threading.Event(), threading.Event()

        def hang() -> np.ndarray:
            rendering.set()
            release.wait(10.0)
            return np.zeros((4, 6, 3), dtype=np.uint8)

        renderer = MagicMock()
        renderer.render.side_effect = hang
        service = CameraService([CameraConfig("overview", fps=100)])
        with patch("mujoco.Renderer", return_value=renderer):
            service.start(*model_data)
            try:
                assert rendering.wait(5.0)
                assert service.stop(0.05) is False
                assert service.on_thread
                release.set()
                assert service.stop() is True
                assert not service.on_thread
            finally:
                release.set()
                service.close()
        renderer.close.assert_called_once()


class TestSceneXmlWatch:
    def test_include_graph_is_walked_once_per_poll(self, tmp_path) -> None:
        path = tmp_path / "scene.xml"
        path.write_text(_XML)
        watcher = SceneXmlWatcher(path)

        with patch.object(watcher, "_collect_paths", return_value=[]) as collect:
            watcher._cached_paths = None  # noqa: SLF001
            for _ in range(5):
                watcher._paths()  # noqa: SLF001

        collect.assert_called_once()

    def test_xml_is_not_polled_on_every_tick(self, tmp_path, model_data) -> None:
        path = tmp_path / "scene.xml"
        path.write_text(_XML)
        watcher = SceneXmlWatcher(path)

        with patch.object(watcher, "_snapshot") as snapshot:
            for _ in range(10):
                watcher.poll(*model_data)

        snapshot.assert_not_called()

    def test_unreadable_include_does_not_escape(self, tmp_path, model_data) -> None:
        path = tmp_path / "scene.xml"
        path.write_text(_XML)
        watcher = SceneXmlWatcher(path)
        watcher._cached_paths = [tmp_path / "deleted-include.xml"]  # noqa: SLF001

        watcher._apply_camera_edits(*model_data)  # noqa: SLF001

    def test_a_half_typed_edit_does_not_escape(self, tmp_path, model_data) -> None:
        path = tmp_path / "scene.xml"
        path.write_text(_XML)
        watcher = SceneXmlWatcher(path)
        path.write_text(_XML.replace('pos="0 -1 0.5"', 'pos="0 -1"'))  # valid XML, wrong shape
        watcher._next_check = 0.0  # noqa: SLF001
        watcher._mtimes = {}  # noqa: SLF001

        watcher.poll(*model_data)

        np.testing.assert_allclose(model_data[0].cam_pos[0], (0.0, -1.0, 0.5))

    def test_an_edit_moves_the_camera(self, tmp_path, model_data) -> None:
        path = tmp_path / "scene.xml"
        path.write_text(_XML)
        watcher = SceneXmlWatcher(path)
        path.write_text(_XML.replace('pos="0 -1 0.5"', 'pos="0 -2 0.5"'))
        watcher._next_check = 0.0  # noqa: SLF001
        watcher._mtimes = {}  # noqa: SLF001

        watcher.poll(*model_data)

        np.testing.assert_allclose(model_data[0].cam_pos[0], (0.0, -2.0, 0.5))

    def test_an_edit_is_offered_again_until_it_is_applied(self, tmp_path, model_data) -> None:
        """A caller that cannot apply an edit yet (the camera thread did not stop) gets it at the next check."""
        path = tmp_path / "scene.xml"
        path.write_text(_XML)
        watcher = SceneXmlWatcher(path)
        watcher._mtimes = {}  # noqa: SLF001

        watcher._next_check = 0.0  # noqa: SLF001
        assert watcher.changed()
        watcher._next_check = 0.0  # noqa: SLF001
        assert watcher.changed()
        watcher.apply(*model_data)
        watcher._next_check = 0.0  # noqa: SLF001
        assert not watcher.changed()

    @pytest.mark.parametrize(
        "orientation",
        [
            'xyaxes="0 0 0 0 1 0"',  # zero x axis
            'xyaxes="1 0 0 2 0 0"',  # y collinear with x
            'xyaxes="nan 0 0 0 1 0"',
            'xyaxes="1 0 0 0 inf 1"',
            'quat="0 0 0 0"',
            'quat="nan 1 0 0"',
            'quat="1e308 1e308 0 0"',  # finite, but its norm overflows
            'euler="0 nan 0"',
        ],
    )
    def test_an_unusable_orientation_leaves_the_camera_alone(self, tmp_path, model_data, orientation) -> None:
        """A half-typed orientation must not install a NaN or zero quaternion in the live model."""
        model = model_data[0]
        path = tmp_path / "scene.xml"
        path.write_text(_XML.replace('xyaxes="1 0 0 0 0.5 1"', orientation))
        watcher = SceneXmlWatcher(path)
        before = model.cam_quat[0].copy()

        watcher.apply(*model_data)

        np.testing.assert_array_equal(model.cam_quat[0], before)

    def test_an_unusable_rig_orientation_leaves_the_rig_alone(self, tmp_path) -> None:
        xml = """<mujoco><worldbody><camera name="overview" pos="0 -1 0.5"/>
<body name="camera_mount" euler="0 0 0"><geom size="0.1"/></body></worldbody></mujoco>"""
        model = mujoco.MjModel.from_xml_string(xml)
        path = tmp_path / "scene.xml"
        path.write_text(xml.replace('euler="0 0 0"', 'quat="0 0 0 0"'))

        SceneXmlWatcher(path).apply(model, mujoco.MjData(model))

        np.testing.assert_array_equal(model.body_quat[model.body("camera_mount").id], (1.0, 0.0, 0.0, 0.0))

    def test_an_xyaxes_edit_turns_the_camera(self, tmp_path, model_data) -> None:
        model = model_data[0]
        path = tmp_path / "scene.xml"
        path.write_text(_XML.replace('xyaxes="1 0 0 0 0.5 1"', 'xyaxes="0 1 0 -1 0 0"'))

        SceneXmlWatcher(path).apply(*model_data)

        expected = mujoco.MjModel.from_xml_string(_XML.replace('xyaxes="1 0 0 0 0.5 1"', 'xyaxes="0 1 0 -1 0 0"'))
        np.testing.assert_allclose(model.cam_quat[0], expected.cam_quat[0], atol=1e-12)
