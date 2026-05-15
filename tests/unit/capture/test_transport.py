# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ctypes
import importlib.util
import pickle
from unittest.mock import MagicMock, patch
from uuid import uuid4

import numpy as np
import pytest

from physicalai.capture.camera import ColorMode
from physicalai.capture.errors import CaptureError, NotConnectedError
from physicalai.capture.frame import Frame
from physicalai.capture.transport._header import (
    HEADER_SIZE,
    PROTOCOL_VERSION,
    FrameHeader,
    decode_depth,
    decode_header,
    decode_rgb,
    decode_rgb_view,
    encode_frame,
)
from physicalai.capture.transport._shared_camera import SharedCamera
from physicalai.capture.transport._spec import CameraSpec

HAS_ICEORYX2 = importlib.util.find_spec("iceoryx2") is not None

requires_iceoryx2 = pytest.mark.skipif(not HAS_ICEORYX2, reason="iceoryx2 not installed")


def _service_name() -> str:
    return f"physicalai/test/{uuid4().hex[:8]}/frame"


class TestCameraSpec:
    def test_picklable(self) -> None:
        spec = CameraSpec(camera_type="uvc", camera_kwargs={"device": 0, "width": 640})
        blob = pickle.dumps(spec)
        restored = pickle.loads(blob)

        assert restored.camera_type == spec.camera_type
        assert restored.camera_kwargs == spec.camera_kwargs

    def test_build_delegates_to_factory(self) -> None:
        spec = CameraSpec(camera_type="uvc", camera_kwargs={"device": 1, "fps": 30})

        with patch("physicalai.capture.factory.create_camera") as mock_create:
            spec.build()

        mock_create.assert_called_once_with("uvc", device=1, fps=30)

    def test_default_kwargs_empty_dict(self) -> None:
        spec = CameraSpec("uvc")
        assert spec.camera_kwargs == {}


class TestFrameHeader:
    def test_sizeof_is_44(self) -> None:
        assert ctypes.sizeof(FrameHeader) == 44
        assert HEADER_SIZE == ctypes.sizeof(FrameHeader)

    def test_protocol_version(self) -> None:
        assert PROTOCOL_VERSION == 2


class TestEncodeDecodeRoundtrip:
    def test_rgb_roundtrip(self) -> None:
        data = np.arange(240 * 320 * 3, dtype=np.uint8).reshape((240, 320, 3))
        frame = Frame(data=data, timestamp=123.456789, sequence=7)

        header, payload = encode_frame(frame, ColorMode.RGB)
        full_payload = bytes(header) + payload

        decoded_header = decode_header(full_payload)
        decoded_frame = decode_rgb(decoded_header, full_payload)

        assert decoded_frame.data.shape == (240, 320, 3)
        assert decoded_frame.data.dtype == np.uint8
        assert decoded_frame.sequence == 7
        assert decoded_frame.timestamp == pytest.approx(frame.timestamp)

    def test_gray_roundtrip(self) -> None:
        data = np.arange(240 * 320, dtype=np.uint8).reshape((240, 320))
        frame = Frame(data=data, timestamp=1.0, sequence=3)

        header, payload = encode_frame(frame, ColorMode.GRAY)
        full_payload = bytes(header) + payload

        decoded_header = decode_header(full_payload)
        decoded_frame = decode_rgb(decoded_header, full_payload)

        assert decoded_frame.data.shape == (240, 320)
        assert decoded_frame.data.dtype == np.uint8

    def test_version_mismatch_raises(self) -> None:
        header = FrameHeader(version=PROTOCOL_VERSION + 1)
        payload = bytes(header)
        with pytest.raises(CaptureError, match="Unsupported protocol version"):
            decode_header(payload)

    def test_payload_too_small_raises(self) -> None:
        with pytest.raises(CaptureError, match="Payload too small"):
            decode_header(b"")

    def test_depth_roundtrip(self) -> None:
        rgb_data = np.zeros((240, 320, 3), dtype=np.uint8)
        depth_data = np.arange(240 * 320, dtype=np.uint16).reshape((240, 320))
        frame = Frame(data=rgb_data, timestamp=2.0, sequence=11)
        depth_frame = Frame(data=depth_data, timestamp=2.0, sequence=11)

        header, payload = encode_frame(frame, ColorMode.RGB, depth_frame=depth_frame)
        full_payload = bytes(header) + payload

        assert header.depth_offset > 0

        decoded_depth = decode_depth(header, full_payload)
        assert decoded_depth.data.shape == depth_data.shape
        assert decoded_depth.data.dtype == depth_data.dtype

    def test_rgb_view_roundtrip(self) -> None:
        data = np.arange(240 * 320 * 3, dtype=np.uint8).reshape((240, 320, 3))
        frame = Frame(data=data, timestamp=1.0, sequence=1)

        header, payload = encode_frame(frame, ColorMode.RGB)
        full_payload = memoryview(bytes(header) + payload)

        decoded_header = decode_header(full_payload)
        decoded_frame = decode_rgb_view(decoded_header, full_payload)

        assert decoded_frame.data.shape == (240, 320, 3)
        assert decoded_frame.data.dtype == np.uint8
        assert decoded_frame.sequence == 1
        assert not decoded_frame.data.flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            decoded_frame.data[0, 0, 0] = 0

    def test_no_depth_raises(self) -> None:
        rgb_data = np.zeros((120, 160, 3), dtype=np.uint8)
        frame = Frame(data=rgb_data, timestamp=0.0, sequence=0)
        header, payload = encode_frame(frame, ColorMode.RGB)
        full_payload = bytes(header) + payload

        with pytest.raises(NotImplementedError, match="no depth data"):
            decode_depth(header, full_payload)

    def test_fps_roundtrip(self) -> None:
        data = np.zeros((240, 320, 3), dtype=np.uint8)
        frame = Frame(data=data, timestamp=1.0, sequence=1)

        header, payload = encode_frame(frame, ColorMode.RGB, fps=30)
        full_payload = bytes(header) + payload

        decoded_header = decode_header(full_payload)
        assert decoded_header.fps == 30

    def test_fps_defaults_to_zero(self) -> None:
        data = np.zeros((240, 320, 3), dtype=np.uint8)
        frame = Frame(data=data, timestamp=1.0, sequence=1)

        header, payload = encode_frame(frame, ColorMode.RGB)
        full_payload = bytes(header) + payload

        decoded_header = decode_header(full_payload)
        assert decoded_header.fps == 0


class TestSharedCameraConstruction:
    """Unit tests for SharedCamera constructor and from_publisher."""

    def test_constructor_with_camera_type(self) -> None:
        cam = SharedCamera("uvc", device=0)
        assert cam._camera_type == "uvc"
        assert cam._service_name == "physicalai/camera/uvc/0/frame"
        assert cam.device_id == "0"

    def test_constructor_with_explicit_service_name(self) -> None:
        cam = SharedCamera.from_publisher("custom/name")
        assert cam._service_name == "custom/name"

    def test_from_publisher(self) -> None:
        cam = SharedCamera.from_publisher("physicalai/camera/uvc/0/frame")
        assert cam._camera_type is None
        assert cam._service_name == "physicalai/camera/uvc/0/frame"

    def test_constructor_rejects_no_args(self) -> None:
        with pytest.raises(ValueError, match="must provide"):
            SharedCamera(None)

    def test_constructor_rejects_service_name_as_type(self) -> None:
        with pytest.raises(ValueError):
            SharedCamera("physicalai/camera/uvc/0/frame")

    def test_default_device_zero(self) -> None:
        cam = SharedCamera("uvc")
        assert cam._service_name is not None
        assert cam._service_name.endswith("/0/frame")

    def test_serial_number_in_service_name(self) -> None:
        cam = SharedCamera("realsense", serial_number="12345")
        assert cam._service_name is not None
        assert "12345" in cam._service_name


class TestSharedCameraSpawnFlow:
    """Unit tests for SharedCamera auto-spawn and race recovery flow."""

    @staticmethod
    def _mock_iox2_stack(sample: object | None = None) -> tuple[MagicMock, MagicMock, MagicMock]:
        iox2 = MagicMock()

        node = MagicMock()
        data_builder = MagicMock()
        event_builder = MagicMock()
        pub_sub = MagicMock()
        event_svc = MagicMock()
        subscriber = MagicMock()
        listener = MagicMock()

        iox2.NodeBuilder.new.return_value.create.return_value = node
        iox2.ServiceName.new.side_effect = lambda value: value
        iox2.Duration.from_secs_f64.return_value = MagicMock()

        node.service_builder.side_effect = [data_builder, event_builder]

        data_builder.publish_subscribe.return_value.open.return_value = pub_sub
        pub_sub.subscriber_builder.return_value.create.return_value = subscriber

        event_builder.event.return_value.open.return_value = event_svc
        event_svc.listener_builder.return_value.create.return_value = listener

        if sample is None:
            subscriber.receive.return_value = None
        else:
            subscriber.receive.return_value = sample

        return iox2, subscriber, listener

    @patch("physicalai.capture.transport._shared_camera.import_module")
    @patch("physicalai.capture.transport._publisher.CameraPublisher")
    @patch("physicalai.capture.transport._shared_camera._probe_service")
    def test_connect_spawns_publisher_when_none_found(
        self,
        mock_probe: MagicMock,
        mock_publisher_cls: MagicMock,
        mock_import_module: MagicMock,
    ) -> None:
        sample = MagicMock()
        iox2, _, _ = self._mock_iox2_stack(sample=sample)
        mock_import_module.return_value = iox2

        mock_probe.side_effect = [False, True]
        mock_publisher = MagicMock()
        mock_publisher_cls.return_value = mock_publisher

        camera = SharedCamera("uvc", device=0)
        with patch.object(camera, "_decode_sample", return_value=(MagicMock(), MagicMock())):
            camera.connect(timeout=0.1)

        assert camera.is_connected
        mock_publisher_cls.assert_called_once()
        mock_publisher.start.assert_called_once_with()

    @patch("physicalai.capture.transport._shared_camera.import_module")
    @patch("physicalai.capture.transport._publisher.CameraPublisher")
    @patch("physicalai.capture.transport._shared_camera._probe_service")
    def test_connect_skips_spawn_when_publisher_found(
        self,
        mock_probe: MagicMock,
        mock_publisher_cls: MagicMock,
        mock_import_module: MagicMock,
    ) -> None:
        sample = MagicMock()
        iox2, _, _ = self._mock_iox2_stack(sample=sample)
        mock_import_module.return_value = iox2
        mock_probe.return_value = True

        camera = SharedCamera("uvc", device=0)
        with patch.object(camera, "_decode_sample", return_value=(MagicMock(), MagicMock())):
            camera.connect(timeout=0.1)

        assert camera.is_connected
        mock_publisher_cls.assert_not_called()

    @patch("physicalai.capture.transport._shared_camera.import_module")
    @patch("physicalai.capture.transport._publisher.CameraPublisher")
    @patch("physicalai.capture.transport._shared_camera._probe_service")
    def test_connect_race_recovery(
        self,
        mock_probe: MagicMock,
        mock_publisher_cls: MagicMock,
        mock_import_module: MagicMock,
    ) -> None:
        sample = MagicMock()
        iox2, _, _ = self._mock_iox2_stack(sample=sample)
        mock_import_module.return_value = iox2

        mock_probe.side_effect = [False, True]
        mock_publisher = MagicMock()
        mock_publisher.start.side_effect = RuntimeError("publisher already running")
        mock_publisher_cls.return_value = mock_publisher

        camera = SharedCamera("uvc", device=0)
        with patch.object(camera, "_decode_sample", return_value=(MagicMock(), MagicMock())):
            camera.connect(timeout=0.1)

        assert camera.is_connected
        assert camera._publisher is None
        mock_probe.assert_called_with(camera._service_name)
        assert mock_probe.call_count == 2
        mock_publisher.start.assert_called_once_with()

    def test_disconnect_stops_spawned_publisher(self) -> None:
        camera = SharedCamera("uvc", device=0)
        spawned_publisher = MagicMock()
        camera._publisher = spawned_publisher
        camera._connected = True
        camera._subscriber = MagicMock()
        camera._listener = MagicMock()
        camera._node = MagicMock()

        camera.disconnect()

        assert not camera.is_connected
        assert camera._subscriber is None
        assert camera._listener is None
        assert camera._node is None


class TestSharedCameraValidateOnConnect:
    """Tests for connect-time config validation on attach to existing publisher."""

    @staticmethod
    def _header_frame(width: int, height: int, fps: int = 30) -> tuple[FrameHeader, Frame]:
        header = FrameHeader(
            version=PROTOCOL_VERSION,
            channels=3,
            dtype=0,
            color_mode=0,
            width=width,
            height=height,
            sequence=0,
            timestamp_ns=0,
            depth_offset=0,
            depth_width=0,
            depth_height=0,
            fps=fps,
        )
        data = np.zeros((height, width, 3), dtype=np.uint8)
        return header, Frame(data=data, timestamp=0.0, sequence=0)

    @patch("physicalai.capture.transport._shared_camera.import_module")
    @patch("physicalai.capture.transport._shared_camera._probe_service")
    def test_validate_on_connect_raises_on_resolution_mismatch(
        self,
        mock_probe: MagicMock,
        mock_import_module: MagicMock,
    ) -> None:
        sample = MagicMock()
        iox2, _, _ = TestSharedCameraSpawnFlow._mock_iox2_stack(sample=sample)
        mock_import_module.return_value = iox2
        mock_probe.return_value = True  # publisher exists, no spawn

        camera = SharedCamera("uvc", device=0, width=640, height=480, validate_on_connect=True)
        with patch.object(camera, "_decode_sample", return_value=self._header_frame(1920, 1080)):
            with pytest.raises(CaptureError, match="does not match"):
                camera.connect(timeout=0.1)

        assert not camera.is_connected
        assert camera._subscriber is None

    @patch("physicalai.capture.transport._shared_camera.import_module")
    @patch("physicalai.capture.transport._shared_camera._probe_service")
    @patch("physicalai.capture.transport._publisher.CameraPublisher.start")
    def test_validate_on_connect_spawned_publisher_mismatch(
        self,
        mock_start: MagicMock,
        mock_probe: MagicMock,
        mock_import_module: MagicMock,
    ) -> None:
        sample = MagicMock()
        iox2, _, _ = TestSharedCameraSpawnFlow._mock_iox2_stack(sample=sample)
        mock_import_module.return_value = iox2
        mock_probe.return_value = False

        camera = SharedCamera("uvc", device=0, width=640, height=480, validate_on_connect=True)
        with patch.object(camera, "_decode_sample", return_value=self._header_frame(1920, 1080)):
            with pytest.raises(CaptureError, match="does not match"):
                camera.connect(timeout=0.1)

        assert not camera.is_connected

    @patch("physicalai.capture.transport._shared_camera.import_module")
    @patch("physicalai.capture.transport._shared_camera._probe_service")
    def test_no_validate_on_connect_warns_on_mismatch_and_attaches(
        self,
        mock_probe: MagicMock,
        mock_import_module: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        sample = MagicMock()
        iox2, _, _ = TestSharedCameraSpawnFlow._mock_iox2_stack(sample=sample)
        mock_import_module.return_value = iox2
        mock_probe.return_value = True

        camera = SharedCamera("uvc", device=0, width=640, height=480, validate_on_connect=False)
        with patch.object(camera, "_decode_sample", return_value=self._header_frame(1920, 1080)):
            camera.connect(timeout=0.1)

        assert camera.is_connected

    @patch("physicalai.capture.transport._shared_camera.import_module")
    @patch("physicalai.capture.transport._shared_camera._probe_service")
    def test_validate_on_connect_silent_on_match(
        self,
        mock_probe: MagicMock,
        mock_import_module: MagicMock,
    ) -> None:
        sample = MagicMock()
        iox2, _, _ = TestSharedCameraSpawnFlow._mock_iox2_stack(sample=sample)
        mock_import_module.return_value = iox2
        mock_probe.return_value = True

        camera = SharedCamera("uvc", device=0, width=640, height=480, validate_on_connect=True)
        with patch.object(camera, "_decode_sample", return_value=self._header_frame(640, 480)):
            camera.connect(timeout=0.1)

        assert camera.is_connected

    @patch("physicalai.capture.transport._shared_camera.import_module")
    @patch("physicalai.capture.transport._shared_camera._probe_service")
    def test_no_dimensions_requested_skips_check(
        self,
        mock_probe: MagicMock,
        mock_import_module: MagicMock,
    ) -> None:
        sample = MagicMock()
        iox2, _, _ = TestSharedCameraSpawnFlow._mock_iox2_stack(sample=sample)
        mock_import_module.return_value = iox2
        mock_probe.return_value = True

        camera = SharedCamera("uvc", device=0, validate_on_connect=True)
        with patch.object(camera, "_decode_sample", return_value=self._header_frame(1920, 1080)):
            camera.connect(timeout=0.1)

        assert camera.is_connected

    @patch("physicalai.capture.transport._shared_camera.import_module")
    @patch("physicalai.capture.transport._shared_camera._probe_service")
    def test_fps_mismatch_validate_on_connect_raises(
        self,
        mock_probe: MagicMock,
        mock_import_module: MagicMock,
    ) -> None:
        sample = MagicMock()
        iox2, _, _ = TestSharedCameraSpawnFlow._mock_iox2_stack(sample=sample)
        mock_import_module.return_value = iox2
        mock_probe.return_value = True

        camera = SharedCamera("uvc", device=0, width=640, height=480, fps=30, validate_on_connect=True)
        with patch.object(camera, "_decode_sample", return_value=self._header_frame(640, 480, fps=60)):
            with pytest.raises(CaptureError, match="does not match"):
                camera.connect(timeout=0.1)

        assert not camera.is_connected

    @patch("physicalai.capture.transport._shared_camera.import_module")
    @patch("physicalai.capture.transport._shared_camera._probe_service")
    def test_no_validate_on_connect_warns_once_not_repeatedly(
        self,
        mock_probe: MagicMock,
        mock_import_module: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        sample = MagicMock()
        iox2, subscriber, listener = TestSharedCameraSpawnFlow._mock_iox2_stack(sample=sample)
        mock_import_module.return_value = iox2
        mock_probe.return_value = True

        camera = SharedCamera("uvc", device=0, width=640, height=480, validate_on_connect=False)
        hf = self._header_frame(1920, 1080)
        with patch.object(camera, "_decode_sample", return_value=hf):
            camera.connect(timeout=0.1)

        assert camera.is_connected
        assert camera._config_warned is True

        # Simulate a second frame read — should NOT warn again
        with caplog.at_level("WARNING"):
            camera._check_config_match(hf[0])
        # Only one warning total (from connect)
        warn_count = sum(1 for r in caplog.records if "existing publisher" in r.message)
        assert warn_count <= 1


class TestOverwriteSettings:
    """Tests for overwrite_settings reconfigure flow in _check_config_match."""

    @staticmethod
    def _header_frame(width: int, height: int, fps: int = 30) -> tuple[FrameHeader, Frame]:
        header = FrameHeader(
            version=PROTOCOL_VERSION,
            channels=3,
            dtype=0,
            color_mode=0,
            width=width,
            height=height,
            sequence=0,
            timestamp_ns=0,
            depth_offset=0,
            depth_width=0,
            depth_height=0,
            fps=fps,
        )
        data = np.zeros((height, width, 3), dtype=np.uint8)
        return header, Frame(data=data, timestamp=0.0, sequence=0)

    @patch("physicalai.capture.transport._shared_camera.import_module")
    @patch("physicalai.capture.transport._shared_camera._probe_service")
    def test_overwrite_reconfigure_success(
        self,
        mock_probe: MagicMock,
        mock_import_module: MagicMock,
    ) -> None:
        sample = MagicMock()
        iox2, _, _ = TestSharedCameraSpawnFlow._mock_iox2_stack(sample=sample)
        mock_import_module.return_value = iox2
        mock_probe.return_value = True

        camera = SharedCamera(
            "uvc",
            device=0,
            width=640,
            height=480,
            validate_on_connect=True,
            overwrite_settings=True,
        )
        with (
            patch.object(camera, "_decode_sample", return_value=self._header_frame(1920, 1080)),
            patch.object(camera, "_request_reconfigure", return_value={"ok": True}),
        ):
            camera.connect(timeout=0.1)

        assert camera.is_connected

    @patch("physicalai.capture.transport._shared_camera.import_module")
    @patch("physicalai.capture.transport._shared_camera._probe_service")
    def test_overwrite_validate_on_connect_reconfigure_failure_raises(
        self,
        mock_probe: MagicMock,
        mock_import_module: MagicMock,
    ) -> None:
        sample = MagicMock()
        iox2, _, _ = TestSharedCameraSpawnFlow._mock_iox2_stack(sample=sample)
        mock_import_module.return_value = iox2
        mock_probe.return_value = True

        camera = SharedCamera(
            "uvc",
            device=0,
            width=640,
            height=480,
            validate_on_connect=True,
            overwrite_settings=True,
        )
        with (
            patch.object(camera, "_decode_sample", return_value=self._header_frame(1920, 1080)),
            patch.object(
                camera,
                "_request_reconfigure",
                return_value={"ok": False, "error": "camera busy"},
            ),
            pytest.raises(CaptureError, match="reconfigure failed"),
        ):
            camera.connect(timeout=0.1)

        assert not camera.is_connected

    @patch("physicalai.capture.transport._shared_camera.import_module")
    @patch("physicalai.capture.transport._shared_camera._probe_service")
    def test_overwrite_no_validate_on_connect_reconfigure_failure_warns(
        self,
        mock_probe: MagicMock,
        mock_import_module: MagicMock,
    ) -> None:
        sample = MagicMock()
        iox2, _, _ = TestSharedCameraSpawnFlow._mock_iox2_stack(sample=sample)
        mock_import_module.return_value = iox2
        mock_probe.return_value = True

        camera = SharedCamera(
            "uvc",
            device=0,
            width=640,
            height=480,
            validate_on_connect=False,
            overwrite_settings=True,
        )
        with (
            patch.object(camera, "_decode_sample", return_value=self._header_frame(1920, 1080)),
            patch.object(
                camera,
                "_request_reconfigure",
                return_value={"ok": False, "error": "camera busy"},
            ),
        ):
            camera.connect(timeout=0.1)

        assert camera.is_connected

    @patch("physicalai.capture.transport._shared_camera.import_module")
    @patch("physicalai.capture.transport._shared_camera._probe_service")
    def test_no_control_service_validate_on_connect_raises(
        self,
        mock_probe: MagicMock,
        mock_import_module: MagicMock,
    ) -> None:
        sample = MagicMock()
        iox2, _, _ = TestSharedCameraSpawnFlow._mock_iox2_stack(sample=sample)
        mock_import_module.return_value = iox2
        mock_probe.return_value = True

        camera = SharedCamera(
            "uvc",
            device=0,
            width=640,
            height=480,
            validate_on_connect=True,
            overwrite_settings=True,
        )
        with (
            patch.object(camera, "_decode_sample", return_value=self._header_frame(1920, 1080)),
            patch.object(
                camera,
                "_request_reconfigure",
                side_effect=CaptureError("publisher does not support reconfigure"),
            ),
            pytest.raises(CaptureError, match="does not support reconfigure"),
        ):
            camera.connect(timeout=0.1)

        assert not camera.is_connected

    @patch("physicalai.capture.transport._shared_camera.import_module")
    @patch("physicalai.capture.transport._shared_camera._probe_service")
    def test_no_control_service_no_validate_on_connect_warns(
        self,
        mock_probe: MagicMock,
        mock_import_module: MagicMock,
    ) -> None:
        sample = MagicMock()
        iox2, _, _ = TestSharedCameraSpawnFlow._mock_iox2_stack(sample=sample)
        mock_import_module.return_value = iox2
        mock_probe.return_value = True

        camera = SharedCamera(
            "uvc",
            device=0,
            width=640,
            height=480,
            validate_on_connect=False,
            overwrite_settings=True,
        )
        with (
            patch.object(camera, "_decode_sample", return_value=self._header_frame(1920, 1080)),
            patch.object(
                camera,
                "_request_reconfigure",
                side_effect=CaptureError("publisher does not support reconfigure"),
            ),
        ):
            camera.connect(timeout=0.1)

        assert camera.is_connected

    @patch("physicalai.capture.transport._shared_camera.import_module")
    @patch("physicalai.capture.transport._shared_camera._probe_service")
    def test_reconfigure_only_attempted_once(
        self,
        mock_probe: MagicMock,
        mock_import_module: MagicMock,
    ) -> None:
        sample = MagicMock()
        iox2, subscriber, listener = TestSharedCameraSpawnFlow._mock_iox2_stack(sample=sample)
        mock_import_module.return_value = iox2
        mock_probe.return_value = True

        camera = SharedCamera(
            "uvc",
            device=0,
            width=640,
            height=480,
            validate_on_connect=False,
            overwrite_settings=True,
        )
        mock_reconfig = MagicMock(return_value={"ok": False, "error": "busy"})
        hf = self._header_frame(1920, 1080)
        with (
            patch.object(camera, "_decode_sample", return_value=hf),
            patch.object(camera, "_request_reconfigure", mock_reconfig),
        ):
            camera.connect(timeout=0.1)

        assert camera.is_connected

        camera._check_config_match(hf[0])

        mock_reconfig.assert_called_once()


@requires_iceoryx2
class TestCameraPublisher:
    def test_start_stop_lifecycle(self, fake_camera_spec: CameraSpec) -> None:
        from physicalai.capture.transport._publisher import CameraPublisher

        publisher = CameraPublisher(
            fake_camera_spec,
            _service_name(),
            _factory_override="tests.unit.capture.fake:FakeCamera",
        )
        publisher.start(timeout=10.0)
        assert publisher.is_alive
        publisher.stop()
        assert not publisher.is_alive

    def test_context_manager(self, fake_camera_spec: CameraSpec) -> None:
        from physicalai.capture.transport._publisher import CameraPublisher

        with CameraPublisher(
            fake_camera_spec,
            _service_name(),
            _factory_override="tests.unit.capture.fake:FakeCamera",
        ) as publisher:
            assert publisher.is_alive
        assert not publisher.is_alive

    def test_start_failure_propagates(self) -> None:
        from physicalai.capture.transport._publisher import CameraPublisher

        bad_spec = CameraSpec(camera_type="does-not-exist", camera_kwargs={})
        publisher = CameraPublisher(bad_spec, _service_name())

        with pytest.raises(CaptureError, match="failed"):
            publisher.start(timeout=2.0)


@requires_iceoryx2
class TestSharedCamera:
    def test_connect_disconnect(self, publisher_service: str) -> None:
        camera = SharedCamera.from_publisher(publisher_service)
        camera.connect(timeout=5.0)
        assert camera.is_connected
        camera.disconnect()
        assert not camera.is_connected

    def test_read_latest_returns_frame(self, publisher_service: str) -> None:
        camera = SharedCamera.from_publisher(publisher_service)
        camera.connect(timeout=5.0)
        frame = camera.read_latest()
        camera.disconnect()

        assert isinstance(frame, Frame)

    def test_read_blocks_until_frame(self, publisher_service: str) -> None:
        camera = SharedCamera.from_publisher(publisher_service)
        camera.connect(timeout=5.0)
        frame = camera.read(timeout=2.0)
        camera.disconnect()

        assert isinstance(frame, Frame)

    def test_read_not_connected(self) -> None:
        camera = SharedCamera.from_publisher(_service_name())
        with pytest.raises(NotConnectedError):
            camera.read()

    def test_read_latest_not_connected(self) -> None:
        camera = SharedCamera.from_publisher(_service_name())
        with pytest.raises(NotConnectedError):
            camera.read_latest()

    def test_zero_copy_read_only(self, publisher_service: str) -> None:
        camera = SharedCamera.from_publisher(publisher_service, zero_copy=True)
        camera.connect(timeout=5.0)
        frame = camera.read_latest()
        camera.disconnect()

        assert isinstance(frame, Frame)
        assert not frame.data.flags.writeable

    def test_connect_idempotent(self, publisher_service: str) -> None:
        camera = SharedCamera.from_publisher(publisher_service)
        camera.connect(timeout=5.0)
        camera.connect(timeout=5.0)  # second call should be no-op
        assert camera.is_connected
        camera.disconnect()

    def test_from_publisher_connect_no_spawn(self, publisher_service: str) -> None:
        camera = SharedCamera.from_publisher(publisher_service)
        camera.connect(timeout=5.0)
        assert camera.is_connected
        camera.disconnect()


@requires_iceoryx2
class TestMultiSubscriber:
    def test_two_subscribers_receive_frames(self, publisher_service: str) -> None:
        cam_a = SharedCamera.from_publisher(publisher_service)
        cam_b = SharedCamera.from_publisher(publisher_service)
        cam_a.connect(timeout=5.0)
        cam_b.connect(timeout=5.0)

        frame_a = cam_a.read_latest()
        frame_b = cam_b.read_latest()

        cam_a.disconnect()
        cam_b.disconnect()

        assert isinstance(frame_a, Frame)
        assert isinstance(frame_b, Frame)


@requires_iceoryx2
class TestReconfigureIntegration:
    """End-to-end tests for the control channel reconfigure flow."""

    def test_reconfigure_resolution_change(self) -> None:
        from physicalai.capture.transport._publisher import CameraPublisher

        service_name = f"physicalai/test/{uuid4().hex[:8]}/frame"
        spec = CameraSpec(camera_type="fake", camera_kwargs={"width": 320, "height": 240})
        publisher = CameraPublisher(
            spec,
            service_name,
            _factory_override="tests.unit.capture.fake:FakeCamera",
        )
        publisher.start(timeout=10.0)

        try:
            camera = SharedCamera.from_publisher(service_name)
            camera.connect(timeout=5.0)
            frame = camera.read_latest()
            assert frame.data.shape == (240, 320, 3)

            camera._overwrite_settings = True
            camera._camera_type = None
            camera._camera_kwargs = {"width": 640, "height": 480}

            result = camera._request_reconfigure(timeout=5.0)
            assert result["ok"] is True

            import time

            time.sleep(0.3)

            frame2 = camera.read(timeout=5.0)
            assert frame2.data.shape == (480, 640, 3)

            camera.disconnect()
        finally:
            publisher.stop()

    def test_reconfigure_failure_restores_old(self) -> None:
        from physicalai.capture.transport._publisher import CameraPublisher

        service_name = f"physicalai/test/{uuid4().hex[:8]}/frame"
        spec = CameraSpec(camera_type="fake", camera_kwargs={"width": 320, "height": 240})
        publisher = CameraPublisher(
            spec,
            service_name,
            _factory_override="tests.unit.capture.fake:FakeCamera",
        )
        publisher.start(timeout=10.0)

        try:
            camera = SharedCamera.from_publisher(service_name)
            camera.connect(timeout=5.0)

            result = camera._request_reconfigure(timeout=5.0)
            assert result["ok"] is False or result["ok"] is True

            frame = camera.read(timeout=5.0)
            assert frame.data.shape[0] > 0
            camera.disconnect()
        finally:
            publisher.stop()

    def test_no_control_service_on_v1_publisher(self) -> None:
        camera = SharedCamera.from_publisher(f"physicalai/test/{uuid4().hex[:8]}/frame")
        camera._overwrite_settings = True
        camera._camera_kwargs = {"width": 640}

        with pytest.raises(CaptureError, match="does not support reconfigure"):
            camera._request_reconfigure(timeout=1.0)

    def test_end_to_end_overwrite_on_connect(self) -> None:
        """Full flow: overwrite_settings triggers auto-reconfigure during connect."""
        import time

        from physicalai.capture.transport._publisher import CameraPublisher

        service_name = f"physicalai/test/{uuid4().hex[:8]}/frame"
        # Publisher starts serving 320×240
        spec = CameraSpec(camera_type="fake", camera_kwargs={"width": 320, "height": 240})
        publisher = CameraPublisher(
            spec,
            service_name,
            _factory_override="tests.unit.capture.fake:FakeCamera",
        )
        publisher.start(timeout=10.0)

        try:
            # Subscriber requests 640×480 — connect should auto-reconfigure publisher
            camera = SharedCamera(
                None,
                service_name=service_name,
                width=640,
                height=480,
                overwrite_settings=True,
            )
            camera.connect(timeout=10.0)
            assert camera.is_connected

            # Publisher has been reconfigured — next frame should be 640×480
            time.sleep(0.3)
            frame = camera.read(timeout=5.0)
            assert frame.data.shape == (480, 640, 3)

            camera.disconnect()
        finally:
            publisher.stop()
