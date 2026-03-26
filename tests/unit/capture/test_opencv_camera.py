# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for OpenCVCamera backend."""

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
def mock_cv2():  # noqa: ANN201
    """Inject a mock cv2 module and reload OpenCVCamera with it.

    Yields:
        Tuple of (OpenCVCamera class, cv2 mock object).
    """
    cv2_mock = mock.MagicMock()
    cv2_mock.CAP_PROP_FRAME_WIDTH = 3
    cv2_mock.CAP_PROP_FRAME_HEIGHT = 4
    cv2_mock.CAP_PROP_FPS = 5
    cv2_mock.CAP_PROP_BUFFERSIZE = 38
    cv2_mock.COLOR_BGR2RGB = 4
    cv2_mock.COLOR_BGR2GRAY = 6
    cv2_mock.VideoCapture = mock.MagicMock()

    sys.modules["cv2"] = cv2_mock

    mods_to_remove = [k for k in sys.modules if "physicalai.capture.cameras.opencv" in k]
    for m in mods_to_remove:
        del sys.modules[m]

    module = importlib.import_module("physicalai.capture.cameras.opencv._camera")
    camera_cls = module.OpenCVCamera

    yield camera_cls, cv2_mock

    sys.modules.pop("cv2", None)
    for k in list(sys.modules):
        if "physicalai.capture.cameras.opencv" in k:
            del sys.modules[k]


def test_import_raises_missing_dependency_without_cv2() -> None:
    """MissingDependencyError raised when cv2 is absent at module import."""
    mods_to_remove = [k for k in sys.modules if "physicalai.capture.cameras.opencv" in k]
    for m in mods_to_remove:
        del sys.modules[m]
    assert "cv2" not in sys.modules, "cv2 should not be installed in test env"
    with pytest.raises(MissingDependencyError):
        importlib.import_module("physicalai.capture.cameras.opencv._camera")


def test_constructor_defaults(mock_cv2: tuple) -> None:
    """OpenCVCamera has expected default parameter values."""
    camera_cls, _ = mock_cv2
    cam = camera_cls()
    assert cam.device_id == "0"
    assert cam._width == 640  # noqa: SLF001
    assert cam._height == 480  # noqa: SLF001
    assert cam._fps == 30  # noqa: SLF001


def test_device_id_returns_string(mock_cv2: tuple) -> None:
    """device_id property returns a string for both int and str inputs."""
    camera_cls, _ = mock_cv2
    assert camera_cls(device_id=0).device_id == "0"
    assert camera_cls(device_id="/dev/video0").device_id == "/dev/video0"


def test_not_connected_initially(mock_cv2: tuple) -> None:
    """Camera is not connected before connect() is called."""
    camera_cls, _ = mock_cv2
    cam = camera_cls()
    assert not cam.is_connected


def test_connect_opens_capture_and_sets_props(mock_cv2: tuple) -> None:
    """connect() creates VideoCapture, sets 4 properties, and marks connected."""
    camera_cls, cv2_mock = mock_cv2
    cap_instance = mock.MagicMock()
    cap_instance.isOpened.return_value = True
    cap_instance.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
    cv2_mock.VideoCapture.return_value = cap_instance

    cam = camera_cls()
    cam.connect()

    assert cam.is_connected
    cv2_mock.VideoCapture.assert_called_once_with(0)
    assert cap_instance.set.call_count >= 4  # noqa: PLR2004


def test_connect_raises_capture_error_on_open_failure(mock_cv2: tuple) -> None:
    """connect() raises CaptureError when VideoCapture fails to open."""
    camera_cls, cv2_mock = mock_cv2
    cap_instance = mock.MagicMock()
    cap_instance.isOpened.return_value = False
    cv2_mock.VideoCapture.return_value = cap_instance

    cam = camera_cls()
    with pytest.raises(CaptureError):
        cam.connect()


def test_connect_timeout_no_frame(mock_cv2: tuple) -> None:
    """connect() raises CaptureTimeoutError when no frame arrives within timeout."""
    camera_cls, cv2_mock = mock_cv2
    cap_instance = mock.MagicMock()
    cap_instance.isOpened.return_value = True
    cap_instance.read.return_value = (False, None)
    cv2_mock.VideoCapture.return_value = cap_instance

    cam = camera_cls()
    with pytest.raises(CaptureTimeoutError):
        cam.connect(timeout=0.01)


def test_read_returns_frame_rgb(mock_cv2: tuple) -> None:
    """read() returns a Frame with cvtColor called for RGB mode."""
    camera_cls, cv2_mock = mock_cv2
    bgr_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    bgr_frame[:, :, 0] = 100
    cap_instance = mock.MagicMock()
    cap_instance.isOpened.return_value = True
    cap_instance.read.return_value = (True, bgr_frame)
    cv2_mock.VideoCapture.return_value = cap_instance

    rgb_frame = bgr_frame.copy()
    rgb_frame[:, :, 2] = 100
    cv2_mock.cvtColor.return_value = rgb_frame

    cam = camera_cls(color_mode=ColorMode.RGB)
    cam.connect()
    frame = cam.read()

    assert isinstance(frame, Frame)
    assert frame.sequence == 0
    cv2_mock.cvtColor.assert_called()


def test_read_returns_frame_bgr(mock_cv2: tuple) -> None:
    """read() returns frame data as-is for BGR mode (no cvtColor)."""
    camera_cls, cv2_mock = mock_cv2
    bgr_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cap_instance = mock.MagicMock()
    cap_instance.isOpened.return_value = True
    cap_instance.read.return_value = (True, bgr_frame)
    cv2_mock.VideoCapture.return_value = cap_instance

    cam = camera_cls(color_mode=ColorMode.BGR)
    cam.connect()
    frame = cam.read()
    assert frame.data is bgr_frame
    cv2_mock.cvtColor.assert_not_called()


def test_read_returns_frame_gray(mock_cv2: tuple) -> None:
    """read() returns a grayscale (H, W) array for GRAY mode."""
    camera_cls, cv2_mock = mock_cv2
    bgr_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    gray_frame = np.zeros((480, 640), dtype=np.uint8)
    cap_instance = mock.MagicMock()
    cap_instance.isOpened.return_value = True
    cap_instance.read.return_value = (True, bgr_frame)
    cv2_mock.VideoCapture.return_value = cap_instance
    cv2_mock.cvtColor.return_value = gray_frame

    cam = camera_cls(color_mode=ColorMode.GRAY)
    cam.connect()
    frame = cam.read()
    assert frame.data.shape == (480, 640)


def test_read_not_connected_raises(mock_cv2: tuple) -> None:
    """read() raises NotConnectedError when called before connect()."""
    camera_cls, _ = mock_cv2
    cam = camera_cls()
    with pytest.raises(NotConnectedError):
        cam.read()


def test_read_sequence_increments(mock_cv2: tuple) -> None:
    """read() increments sequence number on each call (0, 1, 2, ...)."""
    camera_cls, cv2_mock = mock_cv2
    bgr_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cap_instance = mock.MagicMock()
    cap_instance.isOpened.return_value = True
    cap_instance.read.return_value = (True, bgr_frame)
    cv2_mock.VideoCapture.return_value = cap_instance
    cv2_mock.cvtColor.return_value = bgr_frame

    cam = camera_cls()
    cam.connect()
    f1 = cam.read()
    f2 = cam.read()
    assert f1.sequence == 0
    assert f2.sequence == 1


def test_disconnect_releases_capture(mock_cv2: tuple) -> None:
    """disconnect() calls cap.release() and marks camera as not connected."""
    camera_cls, cv2_mock = mock_cv2
    cap_instance = mock.MagicMock()
    cap_instance.isOpened.return_value = True
    cap_instance.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
    cv2_mock.VideoCapture.return_value = cap_instance

    cam = camera_cls()
    cam.connect()
    cam.disconnect()

    cap_instance.release.assert_called_once()
    assert not cam.is_connected


def test_context_manager(mock_cv2: tuple) -> None:
    """Context manager connects on enter and disconnects on exit."""
    camera_cls, cv2_mock = mock_cv2
    cap_instance = mock.MagicMock()
    cap_instance.isOpened.return_value = True
    cap_instance.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
    cv2_mock.VideoCapture.return_value = cap_instance

    with camera_cls() as cam:
        assert cam.is_connected
    assert not cam.is_connected


def test_discover_with_cv2_enumerate(mock_cv2: tuple) -> None:
    """discover() uses cv2_enumerate_cameras when available."""
    camera_cls, _ = mock_cv2

    mock_cam_info = mock.MagicMock()
    mock_cam_info.index = 0
    mock_cam_info.name = "Test Camera"
    mock_cam_info.backend = "DSHOW"
    mock_enumerate_cameras_module = mock.MagicMock()
    mock_enumerate_cameras_module.enumerate_cameras = mock.MagicMock(return_value=[mock_cam_info])

    with mock.patch.dict(sys.modules, {"cv2_enumerate_cameras": mock_enumerate_cameras_module}):
        devices = camera_cls.discover()

    assert len(devices) == 1
    assert isinstance(devices[0], DeviceInfo)
    assert devices[0].name == "Test Camera"
    assert devices[0].driver == "opencv"


def test_discover_fallback_probes_indices(mock_cv2: tuple) -> None:
    """discover() probes indices 0-9 when cv2_enumerate_cameras is absent."""
    camera_cls, cv2_mock = mock_cv2

    def make_cap(idx: int) -> mock.MagicMock:
        cap = mock.MagicMock()
        cap.isOpened.return_value = idx == 0
        return cap

    cv2_mock.VideoCapture.side_effect = make_cap

    with mock.patch.dict(sys.modules, {"cv2_enumerate_cameras": None}):
        devices = camera_cls.discover()

    assert len(devices) == 1
    assert isinstance(devices[0], DeviceInfo)
    assert devices[0].device_id == "0"
    assert devices[0].driver == "opencv"
