# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: S101, PLR2004

"""Tests for OmniCamera."""

from __future__ import annotations

import importlib
import pathlib
import sys
from unittest import mock

import numpy as np
import pytest

from physicalai.capture.camera import ColorMode
from physicalai.capture.cameras.uvc._camera_setting import CameraSetting
from physicalai.capture.discovery import DeviceInfo
from physicalai.capture.errors import CaptureError, CaptureTimeoutError, NotConnectedError
from physicalai.capture.frame import Frame


@pytest.fixture
def omnicamera_cls(monkeypatch: pytest.MonkeyPatch):  # noqa: ANN201
    """Inject a mock omni_camera module and reload OmniCamera with it.

    Yields:
        Tuple of (OmniCamera class, omni_camera mock object).
    """
    mock_omni_camera = mock.MagicMock()

    mock_camera_info = mock.MagicMock()
    mock_camera_info.index = 0
    mock_camera_info.name = "Test OmniCamera"
    mock_camera_info.description = "Test Camera Description"
    mock_camera_info.misc = ""
    mock_camera_info.can_open.return_value = True
    mock_camera_info.unique_id = ""
    mock_camera_info.id_stable = False

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
    mock_fmt_opts.__iter__ = mock.Mock(return_value=iter([mock_fmt]))
    mock_fmt_opts.__bool__ = mock.Mock(return_value=True)

    mock_cam.poll_frame_np.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_cam.poll_frame_np_with_seq.return_value = (np.zeros((480, 640, 3), dtype=np.uint8), 0)
    mock_cam.open.return_value = None
    mock_cam.close.return_value = None

    sys.modules["pynokhwa"] = mock_omni_camera
    sys.modules.pop("physicalai.capture.cameras.uvc._omnicamera", None)

    module = importlib.import_module("physicalai.capture.cameras.uvc._omnicamera")
    # Keep discovery hermetic: the real helper reads /sys and would otherwise
    # report whatever cameras happen to be plugged into the test machine.
    monkeypatch.setattr(module, "_usb_identity", lambda _index: None)
    camera_cls = module.OmniCamera

    yield camera_cls, mock_omni_camera

    sys.modules.pop("pynokhwa", None)
    sys.modules.pop("physicalai.capture.cameras.uvc._omnicamera", None)


def test_constructor_defaults(omnicamera_cls: tuple) -> None:
    """OmniCamera has expected default parameter values."""
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
    """connect() calls query(only_usable=False) to match discover() indices."""
    camera_cls, mock_omni_camera = omnicamera_cls
    cam = camera_cls()
    cam.connect()
    assert any(call.kwargs.get("only_usable") is False for call in mock_omni_camera.query.call_args_list)


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
    """connect() raises CaptureError when Camera() fails to open an unusable device."""
    camera_cls, mock_omni_camera = omnicamera_cls
    mock_omni_camera.Camera.side_effect = RuntimeError("cannot open")

    cam = camera_cls()
    with pytest.raises(RuntimeError):
        cam.connect()


def test_connect_rejects_bgra_only_virtual_camera(omnicamera_cls: tuple) -> None:
    """connect() raises CaptureError when camera reports no supported formats (e.g. BGRA-only)."""
    camera_cls, mock_omni_camera = omnicamera_cls
    mock_cam = mock_omni_camera.Camera.return_value
    mock_cam.get_format_options.return_value = []  # no decodable formats

    cam = camera_cls()
    with pytest.raises(CaptureError, match="virtual camera"):
        cam.connect()


def test_connect_raises_capture_error_on_fourcharcode(omnicamera_cls: tuple) -> None:
    """connect() raises CaptureError when open() fails with an unsupported FourCharCode."""
    camera_cls, mock_omni_camera = omnicamera_cls
    mock_cam = mock_omni_camera.Camera.return_value
    mock_cam.open.side_effect = RuntimeError(
        "Could not generate required structure FourCharCode: Unknown FourCharCode 1111970369"
    )

    cam = camera_cls()
    with pytest.raises(CaptureError, match="unsupported pixel format"):
        cam.connect()


def test_connect_timeout_raises_when_poll_always_none(omnicamera_cls: tuple) -> None:
    """connect() raises CaptureTimeoutError when poll_frame_np always returns None."""
    camera_cls, mock_omni_camera = omnicamera_cls
    mock_cam = mock_omni_camera.Camera.return_value
    mock_cam.poll_frame_np_with_seq.return_value = None

    cam = camera_cls()
    with pytest.raises(CaptureTimeoutError):
        cam.connect(timeout=0.01)


def test_connect_format_mismatch_raises(omnicamera_cls: tuple) -> None:
    """connect() raises CaptureError when no exact format match exists."""
    camera_cls, mock_omni_camera = omnicamera_cls
    mock_cam = mock_omni_camera.Camera.return_value
    mock_fmt_opts = mock_cam.get_format_options.return_value
    mock_fmt = mock.MagicMock()
    mock_fmt.width = 1280
    mock_fmt.height = 720
    mock_fmt.frame_rate = 30
    mock_fmt_opts.__iter__ = mock.Mock(return_value=iter([mock_fmt]))
    mock_fmt_opts.__bool__ = mock.Mock(return_value=True)

    cam = camera_cls(width=640, height=480, fps=30)
    with pytest.raises(CaptureError, match="No camera format matching"):
        cam.connect()


def test_read_returns_frame(omnicamera_cls: tuple) -> None:
    """read() returns a Frame with correct shape after connect()."""
    camera_cls, mock_omni_camera = omnicamera_cls
    mock_cam = mock_omni_camera.Camera.return_value
    cam = camera_cls()
    cam.connect()
    mock_cam.poll_frame_np_with_seq.return_value = (np.zeros((480, 640, 3), dtype=np.uint8), 1)
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
    mock_cam.poll_frame_np_with_seq.return_value = (raw, 1)
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
    mock_cam.poll_frame_np_with_seq.return_value = (raw, 1)
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
    mock_cam.poll_frame_np_with_seq.return_value = (raw, 1)
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

    mock_cam.poll_frame_np_with_seq.return_value = (None, 0)
    with pytest.raises(CaptureTimeoutError):
        cam.read(timeout=0.01)


def test_read_sequence_increments(omnicamera_cls: tuple) -> None:
    """read() reports sequence from the hardware counter."""
    camera_cls, mock_omni_camera = omnicamera_cls
    raw = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_cam = mock_omni_camera.Camera.return_value
    mock_cam.poll_frame_np.return_value = raw

    cam = camera_cls()
    cam.connect()

    mock_cam.poll_frame_np_with_seq.side_effect = [(raw, 1), (raw, 2), (raw, 3)]

    f1 = cam.read()
    f2 = cam.read()
    f3 = cam.read()
    assert f1.sequence == 1
    assert f2.sequence == 2
    assert f3.sequence == 3


def test_read_latest_returns_frame(omnicamera_cls: tuple) -> None:
    """read_latest() returns a Frame when poll_frame_np_with_seq returns a new frame."""
    camera_cls, mock_omni_camera = omnicamera_cls
    raw = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_cam = mock_omni_camera.Camera.return_value
    mock_cam.poll_frame_np.return_value = raw

    cam = camera_cls()
    cam.connect()
    mock_cam.poll_frame_np_with_seq.return_value = (raw, 1)

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
    """read_latest() returns cached frame when poll_frame_np_with_seq returns no new frame."""
    camera_cls, mock_omni_camera = omnicamera_cls
    raw = np.zeros((480, 640, 3), dtype=np.uint8)
    raw[:, :, 0] = 42
    mock_cam = mock_omni_camera.Camera.return_value

    cam = camera_cls()
    # Connect stores _last_frame from connect's poll
    cam.connect()

    # First read_latest with a real frame to populate _last_frame with our marked data
    mock_cam.poll_frame_np_with_seq.return_value = (raw, 1)
    cam.read_latest()

    # Now poll returns None — should return the cached frame
    mock_cam.poll_frame_np_with_seq.return_value = (None, 0)
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
    mock_cam.poll_frame_np_with_seq.return_value = (None, 0)

    with pytest.raises(CaptureError, match="No frame available"):
        cam.read_latest()


def test_read_latest_sequence_from_hw_counter(omnicamera_cls: tuple) -> None:
    """read_latest() uses the hw sequence counter — repeated same seq does not inflate."""
    camera_cls, mock_omni_camera = omnicamera_cls
    raw = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_cam = mock_omni_camera.Camera.return_value
    mock_cam.poll_frame_np.return_value = raw

    cam = camera_cls()
    cam.connect()

    mock_cam.poll_frame_np_with_seq.return_value = (raw, 5)
    f1 = cam.read_latest()
    f2 = cam.read_latest()
    assert f1.sequence == 5
    assert f2.sequence == 5

    mock_cam.poll_frame_np_with_seq.return_value = (raw, 6)
    f3 = cam.read_latest()
    assert f3.sequence == 6


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
    """discover() returns a list of DeviceInfo with index and backend metadata."""
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
    assert devices[0].index == 0
    assert devices[0].name == "Test Camera"
    assert devices[0].driver == "uvc"
    assert devices[0].model == "Test Camera"


def test_device_selector_path_string_maps_to_index(omnicamera_cls: tuple) -> None:
    """connect() with /dev/videoN extracts N and opens that explicit index."""
    camera_cls, mock_omni_camera = omnicamera_cls

    cam_info_2 = mock.MagicMock()
    cam_info_2.index = 2
    cam_info_2.name = "Camera Two"
    cam_info_2.description = ""
    cam_info_2.misc = ""
    cam_info_2.can_open.return_value = True

    mock_omni_camera.query.return_value = [
        mock_omni_camera.query.return_value[0],  # index 0
        cam_info_2,  # index 2
    ]

    cam = camera_cls(device_id="/dev/video2")
    cam.connect()
    assert cam.is_connected
    mock_omni_camera.Camera.assert_called_with(2)


def test_device_selector_invalid_path_raises_value_error(omnicamera_cls: tuple) -> None:
    """connect() with a non-video path string raises ValueError."""
    camera_cls, _ = omnicamera_cls
    cam = camera_cls(device_id="/dev/sda1")
    with pytest.raises(ValueError, match="integer camera index"):
        cam.connect()


# ------------------------------------------------------------------
# Stable ID tests
# ------------------------------------------------------------------


def test_discover_uses_unique_id_when_stable(omnicamera_cls: tuple) -> None:
    """discover() uses unique_id as device_id when id_stable and unique_id are truthy."""
    camera_cls, mock_omni_camera = omnicamera_cls
    mock_camera_info = mock_omni_camera.query.return_value[0]
    mock_camera_info.index = 0
    mock_camera_info.name = "Stable Camera"
    mock_camera_info.description = ""
    mock_camera_info.misc = ""
    mock_camera_info.unique_id = "abc-123-stable"
    mock_camera_info.id_stable = True

    devices = camera_cls.discover()

    assert len(devices) == 1
    assert devices[0].device_id == "abc-123-stable"
    assert devices[0].hardware_id == "abc-123-stable"
    assert devices[0].id_stable is True
    assert devices[0].metadata["unique_id"] == "abc-123-stable"
    assert devices[0].metadata["serial_collision"] is False


def _make_cam_info(index: int, unique_id: str, *, name: str = "UVC Camera", id_stable: bool = True):  # noqa: ANN202
    """Build a mock omni_camera CameraInfo."""
    info = mock.MagicMock()
    info.index = index
    info.name = name
    info.description = ""
    info.misc = ""
    info.unique_id = unique_id
    info.id_stable = id_stable
    info.can_open.return_value = True
    return info


def _patch_usb_identity(
    monkeypatch: pytest.MonkeyPatch,
    camera_cls: type,
    identities: dict[int, tuple[str, tuple[str, str, str]]],
) -> None:
    """Install a fake sysfs ``node index -> (devpath, model_key)`` map."""
    module = sys.modules[camera_cls.__module__]
    fake = {index: module._UsbIdentity(devpath, key) for index, (devpath, key) in identities.items()}  # noqa: SLF001
    monkeypatch.setattr(module, "_usb_identity", fake.get)


# Real values. The ``-video-indexN`` suffix numbers a camera's nodes *within*
# that camera, so every unit's capture node wants the same ``-video-index0``
# name -- which is why two units of one model collide. sysfs identity is
# ``(idVendor, idProduct, serial)``: two *different* models ship the same
# placeholder serial, so the serial alone cannot be the key.
_INNOMAKER_BY_ID = "/dev/v4l/by-id/usb-Innomaker_Innomaker-U20CAM-1080p-S1_SN0001-video-index0"
_INNOMAKER_META_BY_ID = _INNOMAKER_BY_ID.replace("-video-index0", "-video-index1")
_UGREEN_BY_ID = "/dev/v4l/by-id/usb-UGREEN_Camera_2K_UGREEN_Camera_2K_SN0001-video-index0"
_INNOMAKER_USB = ("0c45", "6366", "SN0001")
_UGREEN_USB = ("0c45", "636f", "SN0001")


def test_discover_demotes_by_id_when_a_same_model_twin_has_none(
    omnicamera_cls: tuple,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """discover() must not hand out a by-id path that udev could not disambiguate.

    "One symlink wins": two units sharing an iSerial claim the same
    ``/dev/v4l/by-id`` name and udev materialises it once, so only the winner
    reports a by-id and the loser falls back to a synthetic ``index:N``. No
    value is duplicated, so counting duplicate by-ids finds no collision and
    the winner keeps ``id_stable=True`` -- though its symlink denotes either
    camera and can point at the other one after a reboot.
    """
    camera_cls, mock_omni_camera = omnicamera_cls
    monkeypatch.setattr(sys, "platform", "linux")
    mock_omni_camera.query.return_value = [
        _make_cam_info(40, _INNOMAKER_BY_ID, name="Innomaker-U20CAM-1080p-S1"),
        _make_cam_info(42, "index:42", name="Innomaker-U20CAM-1080p-S1", id_stable=False),
    ]
    _patch_usb_identity(monkeypatch, camera_cls, {40: ("3-6.1", _INNOMAKER_USB), 42: ("3-6.2", _INNOMAKER_USB)})

    devices = camera_cls.discover()

    assert [d.index for d in devices] == [40, 42]
    # Neither camera may claim a stable fingerprint: the lone by-id cannot tell
    # the two units apart, so both fall back to index-based ids.
    assert [d.device_id for d in devices] == ["40", "42"]
    assert all(d.id_stable is False for d in devices)
    assert all(d.metadata["serial_collision"] is True for d in devices)
    # The ambiguous identity stays available for diagnostics.
    assert devices[0].hardware_id == _INNOMAKER_BY_ID


def test_discover_keeps_distinct_models_sharing_a_generic_serial(
    omnicamera_cls: tuple,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two different models with the same generic iSerial stay two stable devices.

    Their by-id paths differ (vendor and product are baked in), so there is no
    real collision and neither entry may be dropped or demoted. Regression guard
    against a collision check that keys on the serial alone.
    """
    camera_cls, mock_omni_camera = omnicamera_cls
    monkeypatch.setattr(sys, "platform", "linux")
    mock_omni_camera.query.return_value = [
        _make_cam_info(0, _UGREEN_BY_ID, name="UGREEN Camera 2K"),
        _make_cam_info(40, _INNOMAKER_BY_ID, name="Innomaker-U20CAM-1080p-S1"),
    ]
    _patch_usb_identity(monkeypatch, camera_cls, {0: ("3-5", _UGREEN_USB), 40: ("3-6.1", _INNOMAKER_USB)})

    devices = camera_cls.discover()

    assert [d.device_id for d in devices] == [_UGREEN_BY_ID, _INNOMAKER_BY_ID]
    assert all(d.id_stable is True for d in devices)
    assert all(d.metadata["serial_collision"] is False for d in devices)


def test_discover_collapses_multi_node_single_camera(
    omnicamera_cls: tuple,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The capture and metadata nodes of one camera collapse to one entry.

    udev names them ``-video-index0`` and ``-video-index1``, so grouping on the
    identity keeps both and lists a metadata node as a camera. The USB device
    path is the sole discriminator: one camera's nodes share it, two cameras
    never do. The lowest-index node survives, and the pair must not be counted
    as two units of the same model.
    """
    camera_cls, mock_omni_camera = omnicamera_cls
    monkeypatch.setattr(sys, "platform", "linux")
    mock_omni_camera.query.return_value = [
        _make_cam_info(40, _INNOMAKER_BY_ID, name="Innomaker-U20CAM-1080p-S1"),
        _make_cam_info(41, _INNOMAKER_META_BY_ID, name="Innomaker-U20CAM-1080p-S1"),
    ]
    _patch_usb_identity(monkeypatch, camera_cls, {40: ("3-6.1", _INNOMAKER_USB), 41: ("3-6.1", _INNOMAKER_USB)})

    devices = camera_cls.discover()

    assert len(devices) == 1
    assert devices[0].index == 40
    assert devices[0].device_id == _INNOMAKER_BY_ID
    assert devices[0].id_stable is True
    assert devices[0].metadata["serial_collision"] is False


def test_discover_demotes_twins_that_report_no_serial(
    omnicamera_cls: tuple,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two units of a model that reports no iSerial at all also collide.

    udev simply omits the serial from the by-id name, so both units claim
    ``usb-Vendor_Model-video-index0``. Regression guard against skipping
    devices with an empty serial when counting units per model.
    """
    camera_cls, mock_omni_camera = omnicamera_cls
    monkeypatch.setattr(sys, "platform", "linux")
    by_id = "/dev/v4l/by-id/usb-Innomaker_Innomaker-U20CAM-1080p-S1-video-index0"
    mock_omni_camera.query.return_value = [
        _make_cam_info(40, by_id, name="Innomaker-U20CAM-1080p-S1"),
        _make_cam_info(42, "index:42", name="Innomaker-U20CAM-1080p-S1", id_stable=False),
    ]
    no_serial = ("0c45", "6366", "")
    _patch_usb_identity(monkeypatch, camera_cls, {40: ("3-6.1", no_serial), 42: ("3-6.2", no_serial)})

    devices = camera_cls.discover()

    assert [d.device_id for d in devices] == ["40", "42"]
    assert all(d.id_stable is False for d in devices)
    assert all(d.metadata["serial_collision"] is True for d in devices)


def test_discover_demotes_duplicate_ids_without_sysfs(
    omnicamera_cls: tuple,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Off Linux there is no sysfs, so duplicate ids remain the only signal.

    Regression guard against the sysfs evidence *replacing* rather than
    augmenting the duplicate check, which would silently hand macOS and
    Windows two stable devices sharing one identity.
    """
    camera_cls, mock_omni_camera = omnicamera_cls
    monkeypatch.setattr(sys, "platform", "darwin")
    mock_omni_camera.query.return_value = [
        _make_cam_info(0, "shared-uid"),
        _make_cam_info(1, "shared-uid"),
    ]

    devices = camera_cls.discover()

    assert [d.device_id for d in devices] == ["0", "1"]
    assert all(d.id_stable is False for d in devices)
    assert all(d.metadata["serial_collision"] is True for d in devices)


@pytest.mark.parametrize(
    ("attrs", "expected"),
    [
        ({"idVendor": "0c45", "idProduct": "6366", "serial": "SN0001"}, ("0c45", "6366", "SN0001")),
        # udev leaves an unreported serial out of the by-id name, so an empty
        # serial has to survive as a collision key of its own.
        ({"idVendor": "0c45", "idProduct": "6366"}, ("0c45", "6366", "")),
        # A PCI capture device (an IPU ISYS node, say) has no USB ancestor and
        # must not be grouped or flagged with anything.
        ({}, None),
    ],
    ids=["with-serial", "without-serial", "no-usb-parent"],
)
def test_usb_identity_reads_the_owning_usb_device(
    omnicamera_cls: tuple,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
    attrs: dict[str, str],
    expected: tuple[str, str, str] | None,
) -> None:
    """The sysfs read walks from a node's device link up to the USB descriptors.

    Everything else trusts this: it is the only evidence that two cameras are
    the same model, and the by-id paths cannot supply it.
    """
    camera_cls, _ = omnicamera_cls
    # The fixture patches this helper away to keep discovery off the host's
    # /sys; reload restores the real one for this test.
    module = importlib.reload(sys.modules[camera_cls.__module__])
    monkeypatch.setattr(sys, "platform", "linux")

    usb_device = tmp_path / "3-6.1"
    interface = usb_device / "3-6.1:1.0"  # the UVC interface the node links to
    interface.mkdir(parents=True)
    for name, value in attrs.items():
        (usb_device / name).write_text(f"{value}\n")
    class_dir = tmp_path / "video4linux"
    (class_dir / "video40").mkdir(parents=True)
    (class_dir / "video40" / "device").symlink_to(interface)
    monkeypatch.setattr(module, "_SYSFS_V4L2", class_dir)

    identity = module._usb_identity(40)  # noqa: SLF001

    if expected is None:
        assert identity is None
    else:
        assert identity.devpath == "3-6.1"
        assert identity.model_key == expected


# ------------------------------------------------------------------
# Open-target tests: which camera a resolved id actually opens
# ------------------------------------------------------------------


def _install_innomaker_twins(omnicamera_cls: tuple, monkeypatch: pytest.MonkeyPatch) -> None:
    """Fake two units of one model sharing an iSerial, as query() reports them.

    Four V4L2 nodes over two USB ports. Both units claim the same by-id names,
    udev materialises them for the winner only, so the loser's nodes fall back
    to a synthetic ``index:N``.

    Args:
        omnicamera_cls: The ``(class, omni_camera mock)`` fixture value.
        monkeypatch: Test monkeypatch.
    """
    camera_cls, mock_omni_camera = omnicamera_cls
    monkeypatch.setattr(sys, "platform", "linux")
    mock_omni_camera.query.return_value = [
        _make_cam_info(40, _INNOMAKER_BY_ID, name="Innomaker-U20CAM-1080p-S1"),
        _make_cam_info(41, _INNOMAKER_META_BY_ID, name="Innomaker-U20CAM-1080p-S1"),
        _make_cam_info(42, "index:42", name="Innomaker-U20CAM-1080p-S1", id_stable=False),
        _make_cam_info(43, "index:43", name="Innomaker-U20CAM-1080p-S1", id_stable=False),
    ]
    _patch_usb_identity(
        monkeypatch,
        camera_cls,
        {
            40: ("3-6.1", _INNOMAKER_USB),
            41: ("3-6.1", _INNOMAKER_USB),
            42: ("3-6.2", _INNOMAKER_USB),
            43: ("3-6.2", _INNOMAKER_USB),
        },
    )


def test_connect_by_index_opens_that_index_verbatim(
    omnicamera_cls: tuple,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An index request opens the named node, never a by-id symlink.

    Handed a CameraInfo the backend prefers ``unique_id`` and resolves the by-id
    symlink, which for a colliding pair exists once and may point at the *other*
    unit -- so the index that discover() demotes a colliding device to would open
    the wrong camera. A bare index is opened directly.
    """
    camera_cls, mock_omni_camera = omnicamera_cls
    _install_innomaker_twins(omnicamera_cls, monkeypatch)

    cam = camera_cls(device_id=42)
    cam.connect()

    assert cam.is_connected
    mock_omni_camera.Camera.assert_called_with(42)


def test_connect_by_index_bypasses_a_shared_unique_id_without_sysfs(
    omnicamera_cls: tuple,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Off Linux an index request also bypasses the shared identity.

    macOS and Windows have no by-id symlink to resolve, but the backend still
    prefers ``unique_id`` and asks the platform to look it up, which for two
    devices advertising one id can land on either.
    """
    camera_cls, mock_omni_camera = omnicamera_cls
    monkeypatch.setattr(sys, "platform", "darwin")
    mock_omni_camera.query.return_value = [_make_cam_info(0, "shared-uid"), _make_cam_info(1, "shared-uid")]

    cam = camera_cls(device_id=1)
    cam.connect()

    assert cam.is_connected
    mock_omni_camera.Camera.assert_called_with(1)


def test_connect_by_unique_id_passes_camera_info(omnicamera_cls: tuple, monkeypatch: pytest.MonkeyPatch) -> None:
    """An unambiguous unique_id request still opens through the CameraInfo.

    That is the point of a stable id: the backend re-resolves it, so the camera
    is found again after its video index has changed.
    """
    camera_cls, mock_omni_camera = omnicamera_cls
    monkeypatch.setattr(sys, "platform", "linux")
    infos = [
        _make_cam_info(0, _UGREEN_BY_ID, name="UGREEN Camera 2K"),
        _make_cam_info(40, _INNOMAKER_BY_ID, name="Innomaker-U20CAM-1080p-S1"),
    ]
    mock_omni_camera.query.return_value = infos
    _patch_usb_identity(monkeypatch, camera_cls, {0: ("3-5", _UGREEN_USB), 40: ("3-6.1", _INNOMAKER_USB)})

    cam = camera_cls(device_id=_INNOMAKER_BY_ID)
    cam.connect()

    assert cam.is_connected
    mock_omni_camera.Camera.assert_called_with(infos[1])


@pytest.mark.parametrize(("device_id", "expected"), [("index:42", 42), ("index:40", 40)], ids=["reported", "stale"])
def test_connect_by_synthetic_index_id_is_an_index_request(
    omnicamera_cls: tuple,
    monkeypatch: pytest.MonkeyPatch,
    device_id: str,
    expected: int,
) -> None:
    """An ``index:N`` identity opens node N and is never refused as ambiguous.

    That is what the backend reports for a node with no by-id name of its own,
    so it reaches us through ``hardware_id``. It names exactly one node, unlike
    the by-id its twin also claims -- and it keeps naming that node after udev
    has since given it a by-id, when it no longer matches any reported id.
    """
    camera_cls, mock_omni_camera = omnicamera_cls
    _install_innomaker_twins(omnicamera_cls, monkeypatch)

    cam = camera_cls(device_id=device_id)
    cam.connect()

    assert cam.is_connected
    mock_omni_camera.Camera.assert_called_with(expected)


@pytest.mark.parametrize("by_id", [_INNOMAKER_BY_ID, _INNOMAKER_META_BY_ID], ids=["capture", "metadata"])
def test_connect_refuses_a_by_id_shared_by_two_units(
    omnicamera_cls: tuple,
    monkeypatch: pytest.MonkeyPatch,
    by_id: str,
) -> None:
    """connect() refuses a by-id that udev could not disambiguate.

    discover() stops offering it, but a config written before the second unit
    was plugged in still names it. Opening it would silently stream whichever
    unit currently owns the symlink -- and pairing it with the other unit's
    index yields the same camera twice. A camera's metadata node is refused
    too: it carries its own by-id that the twin claims just as well.
    """
    camera_cls, _ = omnicamera_cls
    _install_innomaker_twins(omnicamera_cls, monkeypatch)

    cam = camera_cls(device_id=by_id)
    with pytest.raises(CaptureError, match="video index"):
        cam.connect()

    assert not cam.is_connected


def test_query_formats_by_index_uses_the_index_token(
    omnicamera_cls: tuple,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """query_formats() probes the named index, not the by-id symlink.

    Otherwise a colliding device reports the other unit's capabilities.
    """
    camera_cls, mock_omni_camera = omnicamera_cls
    _install_innomaker_twins(omnicamera_cls, monkeypatch)

    assert camera_cls.query_formats("42") == [(640, 480, 30)]
    mock_omni_camera.Camera.assert_called_with(42)


def test_discover_falls_back_to_index_when_id_unstable(omnicamera_cls: tuple) -> None:
    """discover() falls back to str(index) when id_stable is False."""
    camera_cls, mock_omni_camera = omnicamera_cls
    mock_camera_info = mock_omni_camera.query.return_value[0]
    mock_camera_info.index = 3
    mock_camera_info.name = "Unstable Camera"
    mock_camera_info.description = ""
    mock_camera_info.misc = ""
    mock_camera_info.unique_id = "some-uid"
    mock_camera_info.id_stable = False

    devices = camera_cls.discover()

    assert len(devices) == 1
    assert devices[0].device_id == "3"
    assert devices[0].hardware_id == "some-uid"
    assert devices[0].id_stable is False


def test_connect_resolves_by_unique_id(omnicamera_cls: tuple) -> None:
    """connect() resolves device by unique_id string when it matches a camera."""
    camera_cls, mock_omni_camera = omnicamera_cls

    cam_info_0 = mock.MagicMock()
    cam_info_0.index = 0
    cam_info_0.name = "Camera Zero"
    cam_info_0.description = ""
    cam_info_0.misc = ""
    cam_info_0.unique_id = "uid-zero"
    cam_info_0.id_stable = True
    cam_info_0.can_open.return_value = True

    cam_info_1 = mock.MagicMock()
    cam_info_1.index = 1
    cam_info_1.name = "Camera One"
    cam_info_1.description = ""
    cam_info_1.misc = ""
    cam_info_1.unique_id = "uid-one"
    cam_info_1.id_stable = True
    cam_info_1.can_open.return_value = True

    mock_omni_camera.query.return_value = [cam_info_0, cam_info_1]

    cam = camera_cls(device_id="uid-one")
    cam.connect()
    assert cam.is_connected
    mock_omni_camera.Camera.assert_called_with(cam_info_1)


# ------------------------------------------------------------------
# get_settings tests
# ------------------------------------------------------------------


def _make_mock_control(*, value_range: range, is_active: bool = True) -> mock.MagicMock:
    """Create a mock omni_camera CameraControl."""
    ctrl = mock.MagicMock()
    ctrl.value_range = value_range
    ctrl.is_active = is_active
    return ctrl


def test_get_settings_parses_dict(omnicamera_cls: tuple) -> None:
    """get_settings() correctly parses Dict[str, CameraControl] from get_controls()."""
    camera_cls, mock_omni_camera = omnicamera_cls
    mock_cam = mock_omni_camera.Camera.return_value

    mock_cam.get_controls.return_value = {
        "Brightness": _make_mock_control(value_range=range(0, 256, 1), is_active=True),
        "Exposure": _make_mock_control(value_range=range(0, 0), is_active=True),
        "Gain": _make_mock_control(value_range=range(0, 128, 2), is_active=False),
    }

    cam = camera_cls()
    cam.connect()
    controls = cam.get_settings()

    assert len(controls) == 3

    brightness = next(c for c in controls if c.name == "Brightness")
    assert brightness.id == "Brightness"
    assert brightness.setting_type == "integer"
    assert brightness.min == 0
    assert brightness.max == 255
    assert brightness.step == 1
    assert brightness.default is None
    assert brightness.value is None
    assert brightness.inactive is False

    exposure = next(c for c in controls if c.name == "Exposure")
    assert exposure.id == "Exposure"
    assert exposure.min is None
    assert exposure.max is None
    assert exposure.step is None
    assert exposure.inactive is False

    gain = next(c for c in controls if c.name == "Gain")
    assert gain.id == "Gain"
    assert gain.min == 0
    assert gain.max == 126
    assert gain.step == 2
    assert gain.inactive is True


def test_get_settings_empty_dict(omnicamera_cls: tuple) -> None:
    """get_settings() returns empty list when get_controls() returns empty dict."""
    camera_cls, mock_omni_camera = omnicamera_cls
    mock_cam = mock_omni_camera.Camera.return_value
    mock_cam.get_controls.return_value = {}

    cam = camera_cls()
    cam.connect()
    controls = cam.get_settings()
    assert controls == []


def test_get_settings_not_connected_raises(omnicamera_cls: tuple) -> None:
    """get_settings() raises NotConnectedError before connect()."""
    camera_cls, _ = omnicamera_cls
    cam = camera_cls()
    with pytest.raises(NotConnectedError):
        cam.get_settings()


def test_get_settings_no_get_controls_raises(omnicamera_cls: tuple) -> None:
    """get_settings() raises NotImplementedError when get_controls is unavailable."""
    camera_cls, mock_omni_camera = omnicamera_cls
    mock_cam = mock_omni_camera.Camera.return_value
    del mock_cam.get_controls

    cam = camera_cls()
    cam.connect()
    with pytest.raises(NotImplementedError, match="not available"):
        cam.get_settings()


def test_apply_settings_raises_not_implemented(omnicamera_cls: tuple) -> None:
    """apply_settings() raises NotImplementedError on OmniCamera backend."""
    camera_cls, _ = omnicamera_cls
    cam = camera_cls()
    cam.connect()
    with pytest.raises(NotImplementedError):
        cam.apply_settings(CameraSetting(id="Brightness", name="Brightness", setting_type="integer", value=128))
