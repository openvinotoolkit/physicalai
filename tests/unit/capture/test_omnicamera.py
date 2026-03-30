# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: S101, PLR2004

"""Tests for OmniCameraBackend."""

from __future__ import annotations

import importlib
import sys
from unittest import mock

import numpy as np
import pytest

from physicalai.capture.camera import ColorMode
from physicalai.capture.discovery import DeviceInfo
from physicalai.capture.errors import CaptureError, CaptureTimeoutError, MissingDependencyError, NotConnectedError
from physicalai.capture.frame import Frame


@pytest.fixture
def omnicamera_cls():  # noqa: ANN201
    """Inject a mock omni_camera module and reload OmniCameraBackend with it.

    Yields:
        Tuple of (OmniCameraBackend class, omni_camera mock object).
    """
    mock_omni_camera = mock.MagicMock()

    mock_camera_info = mock.MagicMock()
    mock_camera_info.index = 0
    mock_camera_info.name = "Test OmniCamera"
    mock_camera_info.description = "Test Camera Description"
    mock_camera_info.misc = ""
    mock_camera_info.can_open.return_value = True

    mock_omni_camera.query.return_value = [mock_camera_info]

    mock_cam = mock.MagicMock()
    mock_omni_camera.Camera.return_value = mock_cam

    mock_fmt_opts = mock.MagicMock()
    mock_cam.get_format_options.return_value = mock_fmt_opts
    mock_fmt_opts.prefer_width_range.return_value = mock_fmt_opts
    mock_fmt_opts.prefer_height_range.return_value = mock_fmt_opts
    mock_fmt_opts.prefer_fps_range.return_value = mock_fmt_opts

    mock_fmt = mock.MagicMock()
    mock_fmt.width = 640
    mock_fmt.height = 480
    mock_fmt.frame_rate = 30
    mock_fmt_opts.resolve.return_value = mock_fmt

    mock_cam.poll_frame_np.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_cam.open.return_value = None
    mock_cam.close.return_value = None

    sys.modules["omni_camera"] = mock_omni_camera
    sys.modules.pop("physicalai.capture.cameras.uvc._omnicamera", None)

    module = importlib.import_module("physicalai.capture.cameras.uvc._omnicamera")
    camera_cls = module.OmniCameraBackend

    yield camera_cls, mock_omni_camera

    sys.modules.pop("omni_camera", None)
    sys.modules.pop("physicalai.capture.cameras.uvc._omnicamera", None)


def test_import_raises_missing_dependency_without_omni_camera() -> None:
    """MissingDependencyError raised when omni_camera is absent at module import."""
    sys.modules.pop("physicalai.capture.cameras.uvc._omnicamera", None)
    omni_camera_backup = sys.modules.pop("omni_camera", None)
    sys.modules["omni_camera"] = None  # type: ignore[assignment]
    try:
        with pytest.raises(MissingDependencyError):
            importlib.import_module("physicalai.capture.cameras.uvc._omnicamera")
    finally:
        sys.modules.pop("physicalai.capture.cameras.uvc._omnicamera", None)
        if omni_camera_backup is not None:
            sys.modules["omni_camera"] = omni_camera_backup
        else:
            sys.modules.pop("omni_camera", None)


def test_constructor_defaults(omnicamera_cls: tuple) -> None:
    """OmniCameraBackend has expected default parameter values."""
    camera_cls, _ = omnicamera_cls
    cam = camera_cls()
    assert cam.device_id == "0"
    assert cam._width == 640  # noqa: SLF001
    assert cam._height == 480  # noqa: SLF001
    assert cam._fps == 30  # noqa: SLF001


def test_device_id_property_int_input(omnicamera_cls: tuple) -> None:
    """device_id returns string when constructed with an integer."""
    camera_cls, _ = omnicamera_cls
    cam = camera_cls(device_id=1)
    assert cam.device_id == "1"


def test_device_id_property_str_input(omnicamera_cls: tuple) -> None:
    """device_id returns same string when constructed with a string."""
    camera_cls, _ = omnicamera_cls
    cam = camera_cls(device_id="2")
    assert cam.device_id == "2"


def test_not_connected_initially(omnicamera_cls: tuple) -> None:
    """Camera is not connected before connect() is called."""
    camera_cls, _ = omnicamera_cls
    cam = camera_cls()
    assert not cam.is_connected


def test_connect_queries_cameras(omnicamera_cls: tuple) -> None:
    """connect() calls omni_camera.query(only_usable=True)."""
    camera_cls, mock_omni_camera = omnicamera_cls
    cam = camera_cls()
    cam.connect()
    mock_omni_camera.query.assert_called_once_with(only_usable=True)


def test_connect_creates_camera_without_suggested_fps(omnicamera_cls: tuple) -> None:
    """connect() calls omni_camera.Camera with CameraInfo only, not suggested_fps."""
    camera_cls, mock_omni_camera = omnicamera_cls
    cam = camera_cls()
    cam.connect()
    # Camera must be called once
    assert mock_omni_camera.Camera.call_count == 1
    call_kwargs = mock_omni_camera.Camera.call_args.kwargs
    assert "suggested_fps" not in call_kwargs


def test_connect_calls_open_with_resolved_format(omnicamera_cls: tuple) -> None:
    """connect() calls camera.open() with the resolved format object."""
    camera_cls, mock_omni_camera = omnicamera_cls
    mock_cam = mock_omni_camera.Camera.return_value
    mock_fmt = mock_cam.get_format_options.return_value.resolve.return_value

    cam = camera_cls()
    cam.connect()

    mock_cam.open.assert_called_once_with(mock_fmt)


def test_connect_sets_connected_true(omnicamera_cls: tuple) -> None:
    """is_connected is True after successful connect()."""
    camera_cls, _ = omnicamera_cls
    cam = camera_cls()
    cam.connect()
    assert cam.is_connected


def test_connect_raises_capture_error_when_no_camera_found(omnicamera_cls: tuple) -> None:
    """connect() raises CaptureError when query returns an empty list."""
    camera_cls, mock_omni_camera = omnicamera_cls
    mock_omni_camera.query.return_value = []

    cam = camera_cls()
    with pytest.raises(CaptureError):
        cam.connect()


def test_connect_raises_capture_error_when_camera_cant_open(omnicamera_cls: tuple) -> None:
    """connect() raises CaptureError when can_open() returns False."""
    camera_cls, mock_omni_camera = omnicamera_cls
    mock_camera_info = mock_omni_camera.query.return_value[0]
    mock_camera_info.can_open.return_value = False

    cam = camera_cls()
    with pytest.raises(CaptureError):
        cam.connect()


def test_connect_timeout_raises_when_poll_always_none(omnicamera_cls: tuple) -> None:
    """connect() raises CaptureTimeoutError when poll_frame_np always returns None."""
    camera_cls, mock_omni_camera = omnicamera_cls
    mock_cam = mock_omni_camera.Camera.return_value
    mock_cam.poll_frame_np.return_value = None

    cam = camera_cls()
    with pytest.raises(CaptureTimeoutError):
        cam.connect(timeout=0.01)


def test_connect_format_warning_on_mismatch(omnicamera_cls: tuple) -> None:
    """connect() emits a loguru warning when resolved format differs from requested."""
    camera_cls, mock_omni_camera = omnicamera_cls
    mock_cam = mock_omni_camera.Camera.return_value
    mock_fmt_opts = mock_cam.get_format_options.return_value
    mock_fmt = mock.MagicMock()
    mock_fmt.width = 1280
    mock_fmt.height = 480
    mock_fmt.frame_rate = 30
    mock_fmt_opts.resolve.return_value = mock_fmt

    cam = camera_cls(width=640, height=480, fps=30)
    with mock.patch("loguru.logger.warning") as mock_warning:
        cam.connect()
    mock_warning.assert_called_once()
    warning_msg = mock_warning.call_args[0][0]
    assert "640" in warning_msg or "1280" in warning_msg


def test_read_returns_frame(omnicamera_cls: tuple) -> None:
    """read() returns a Frame with correct shape after connect()."""
    camera_cls, _ = omnicamera_cls
    cam = camera_cls()
    cam.connect()
    frame = cam.read()
    assert isinstance(frame, Frame)
    assert frame.data.shape == (480, 640, 3)


def test_read_rgb_mode(omnicamera_cls: tuple) -> None:
    """read() with ColorMode.RGB returns data unchanged from raw array."""
    camera_cls, mock_omni_camera = omnicamera_cls
    raw = np.zeros((480, 640, 3), dtype=np.uint8)
    raw[:, :, 0] = 100
    mock_cam = mock_omni_camera.Camera.return_value
    mock_cam.poll_frame_np.return_value = raw

    cam = camera_cls(color_mode=ColorMode.RGB)
    cam.connect()
    mock_cam.poll_frame_np.return_value = raw
    frame = cam.read()

    assert isinstance(frame, Frame)
    np.testing.assert_array_equal(frame.data, raw)


def test_read_bgr_mode(omnicamera_cls: tuple) -> None:
    """read() with ColorMode.BGR returns data with swapped channels."""
    camera_cls, mock_omni_camera = omnicamera_cls
    raw = np.zeros((480, 640, 3), dtype=np.uint8)
    raw[:, :, 0] = 100
    mock_cam = mock_omni_camera.Camera.return_value
    mock_cam.poll_frame_np.return_value = raw

    cam = camera_cls(color_mode=ColorMode.BGR)
    cam.connect()
    mock_cam.poll_frame_np.return_value = raw
    frame = cam.read()

    assert isinstance(frame, Frame)
    assert frame.data.shape == (480, 640, 3)
    np.testing.assert_array_equal(frame.data[:, :, 2], raw[:, :, 0])


def test_read_gray_mode(omnicamera_cls: tuple) -> None:
    """read() with ColorMode.GRAY returns a 2D (H, W) array."""
    camera_cls, mock_omni_camera = omnicamera_cls
    raw = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_cam = mock_omni_camera.Camera.return_value
    mock_cam.poll_frame_np.return_value = raw

    cam = camera_cls(color_mode=ColorMode.GRAY)
    cam.connect()
    mock_cam.poll_frame_np.return_value = raw
    frame = cam.read()

    assert isinstance(frame, Frame)
    assert frame.data.shape == (480, 640)


def test_read_not_connected_raises(omnicamera_cls: tuple) -> None:
    """read() raises NotConnectedError when called before connect()."""
    camera_cls, _ = omnicamera_cls
    cam = camera_cls()
    with pytest.raises(NotConnectedError):
        cam.read()


def test_read_timeout_raises(omnicamera_cls: tuple) -> None:
    """read() raises CaptureTimeoutError when poll always returns None within timeout."""
    camera_cls, mock_omni_camera = omnicamera_cls
    mock_cam = mock_omni_camera.Camera.return_value

    cam = camera_cls()
    cam.connect()

    mock_cam.poll_frame_np.return_value = None
    with pytest.raises(CaptureTimeoutError):
        cam.read(timeout=0.01)


def test_read_sequence_increments(omnicamera_cls: tuple) -> None:
    """read() increments sequence on each call (1, 2, 3 …)."""
    camera_cls, mock_omni_camera = omnicamera_cls
    raw = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_cam = mock_omni_camera.Camera.return_value
    mock_cam.poll_frame_np.return_value = raw

    cam = camera_cls()
    cam.connect()

    f1 = cam.read()
    f2 = cam.read()
    f3 = cam.read()
    assert f1.sequence == 1
    assert f2.sequence == 2
    assert f3.sequence == 3


def test_read_latest_returns_frame(omnicamera_cls: tuple) -> None:
    """read_latest() returns a Frame when poll_frame_np returns a frame."""
    camera_cls, mock_omni_camera = omnicamera_cls
    raw = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_cam = mock_omni_camera.Camera.return_value
    mock_cam.poll_frame_np.return_value = raw

    cam = camera_cls()
    cam.connect()
    mock_cam.poll_frame_np.return_value = raw

    frame = cam.read_latest()
    assert isinstance(frame, Frame)
    assert frame.data.shape == (480, 640, 3)


def test_read_latest_not_connected_raises(omnicamera_cls: tuple) -> None:
    """read_latest() raises NotConnectedError before connect()."""
    camera_cls, _ = omnicamera_cls
    cam = camera_cls()
    with pytest.raises(NotConnectedError):
        cam.read_latest()


def test_read_latest_returns_cached_frame_when_poll_none(omnicamera_cls: tuple) -> None:
    """read_latest() returns cached frame when poll_frame_np returns None."""
    camera_cls, mock_omni_camera = omnicamera_cls
    raw = np.zeros((480, 640, 3), dtype=np.uint8)
    raw[:, :, 0] = 42
    mock_cam = mock_omni_camera.Camera.return_value
    mock_cam.poll_frame_np.return_value = raw

    cam = camera_cls()
    cam.connect()

    mock_cam.poll_frame_np.return_value = None
    seq_before = cam._sequence  # noqa: SLF001
    frame = cam.read_latest()

    assert isinstance(frame, Frame)
    assert frame.sequence == seq_before
    assert frame.data[0, 0, 0] == 42


def test_read_latest_raises_when_no_cache_and_poll_none(omnicamera_cls: tuple) -> None:
    """read_latest() raises CaptureError if no cache and poll returns None."""
    camera_cls, mock_omni_camera = omnicamera_cls
    mock_cam = mock_omni_camera.Camera.return_value

    cam = camera_cls()
    cam.connect()

    cam._last_frame = None  # noqa: SLF001
    mock_cam.poll_frame_np.return_value = None

    with pytest.raises(CaptureError, match="No frame available"):
        cam.read_latest()


def test_disconnect_closes_camera(omnicamera_cls: tuple) -> None:
    """disconnect() calls cam.close() and sets is_connected to False."""
    camera_cls, mock_omni_camera = omnicamera_cls
    mock_cam = mock_omni_camera.Camera.return_value

    cam = camera_cls()
    cam.connect()
    assert cam.is_connected

    cam.disconnect()
    mock_cam.close.assert_called_once()
    assert not cam.is_connected


def test_disconnect_idempotent(omnicamera_cls: tuple) -> None:
    """Calling disconnect() twice does not raise."""
    camera_cls, _ = omnicamera_cls
    cam = camera_cls()
    cam.connect()
    cam.disconnect()
    cam.disconnect()


def test_context_manager_connects_and_disconnects(omnicamera_cls: tuple) -> None:
    """Context manager calls connect on enter and disconnect on exit."""
    camera_cls, mock_omni_camera = omnicamera_cls
    mock_cam = mock_omni_camera.Camera.return_value

    with camera_cls(device_id=0) as cam:
        assert cam.is_connected

    mock_cam.close.assert_called_once()
    assert not cam.is_connected


def test_discover_returns_device_info(omnicamera_cls: tuple) -> None:
    """discover() returns a list of DeviceInfo with driver='uvc'."""
    camera_cls, mock_omni_camera = omnicamera_cls
    mock_camera_info = mock_omni_camera.query.return_value[0]
    mock_camera_info.index = 0
    mock_camera_info.name = "Test Camera"
    mock_camera_info.description = "USB Camera"
    mock_camera_info.misc = ""
    mock_camera_info.can_open.return_value = True

    devices = camera_cls.discover()

    assert len(devices) == 1
    assert isinstance(devices[0], DeviceInfo)
    assert devices[0].device_id == "0"
    assert devices[0].name == "Test Camera"
    assert devices[0].driver == "uvc"


def test_discover_filters_unopenable(omnicamera_cls: tuple) -> None:
    """discover() excludes cameras where can_open() returns False."""
    camera_cls, mock_omni_camera = omnicamera_cls

    cam_info_0 = mock.MagicMock()
    cam_info_0.index = 0
    cam_info_0.name = "Camera 0"
    cam_info_0.description = ""
    cam_info_0.misc = ""
    cam_info_0.can_open.return_value = True

    cam_info_1 = mock.MagicMock()
    cam_info_1.index = 1
    cam_info_1.name = "Camera 1"
    cam_info_1.description = ""
    cam_info_1.misc = ""
    cam_info_1.can_open.return_value = False

    mock_omni_camera.query.return_value = [cam_info_0, cam_info_1]

    devices = camera_cls.discover()

    assert len(devices) == 1
    assert devices[0].device_id == "0"


def test_device_selector_path_string_raises_value_error(omnicamera_cls: tuple) -> None:
    """connect() with a path-style string device_id raises ValueError."""
    camera_cls, _ = omnicamera_cls
    cam = camera_cls(device_id="/dev/video0")
    with pytest.raises(ValueError, match="integer camera index"):
        cam.connect()
