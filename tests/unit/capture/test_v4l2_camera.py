# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for V4L2Camera backend."""

from __future__ import annotations

import ctypes
import mmap as mmap_mod
import sys
from contextlib import contextmanager
from unittest import mock

import numpy as np
import pytest

from physicalai.capture.camera import ColorMode
from physicalai.capture.cameras.uvc.v4l2._camera import V4L2Camera

from physicalai.capture.cameras.uvc.v4l2._ioctl import (
    V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC,
    V4L2_BUF_TYPE_VIDEO_CAPTURE,
    V4L2_CAP_STREAMING,
    V4L2_CAP_VIDEO_CAPTURE,
    V4L2_MEMORY_MMAP,
    VIDIOC_DQBUF,
    VIDIOC_G_FMT,
    VIDIOC_QBUF,
    VIDIOC_QUERYCAP,
    VIDIOC_REQBUFS,
    VIDIOC_STREAMOFF,
    VIDIOC_STREAMON,
    v4l2_buffer,
    v4l2_capability,
    v4l2_requestbuffers,
    v4l2_timecode,
)
from physicalai.capture.discovery import DeviceInfo
from physicalai.capture.errors import CaptureError, CaptureTimeoutError, NotConnectedError
from physicalai.capture.frame import Frame
from physicalai.capture.transport import _publisher_worker as worker

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FAKE_FD = 5


def _make_xioctl(num_buffers: int = 1, dqbuf_frame_data: bytes | None = None):
    """Return a mock xioctl side-effect that handles core requests."""
    sequence = 0

    def _xioctl(fd: int, request: int, arg: object) -> int:  # noqa: ARG001
        nonlocal sequence
        if request == VIDIOC_QUERYCAP:
            cap = v4l2_capability()
            cap.capabilities = V4L2_CAP_VIDEO_CAPTURE | V4L2_CAP_STREAMING
            ctypes.memmove(
                ctypes.addressof(arg),  # type: ignore[arg-type]
                ctypes.addressof(cap),
                ctypes.sizeof(v4l2_capability),
            )
        elif request == VIDIOC_REQBUFS:
            # Mutate the count field of the passed v4l2_requestbuffers struct.
            req_ptr = ctypes.cast(ctypes.addressof(arg), ctypes.POINTER(ctypes.c_uint32))  # type: ignore[arg-type]
            req_ptr[0] = num_buffers
        elif request == VIDIOC_DQBUF and dqbuf_frame_data is not None:
            buf = v4l2_buffer()
            buf.index = 0
            buf.bytesused = len(dqbuf_frame_data)
            buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
            buf.memory = V4L2_MEMORY_MMAP
            buf.timestamp.tv_sec = 1000
            buf.timestamp.tv_usec = 500000
            buf.flags = V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC
            buf.sequence = sequence
            sequence += 1
            ctypes.memmove(
                ctypes.addressof(arg),  # type: ignore[arg-type]
                ctypes.addressof(buf),
                ctypes.sizeof(v4l2_buffer),
            )
        return 0

    return _xioctl


def _make_mmap_mock(frame_data: bytes | None = None) -> mock.MagicMock:
    """Return a MagicMock mmap whose slicing returns *frame_data*."""
    mm = mock.MagicMock(spec=mmap_mod.mmap)
    mm.__getitem__ = mock.Mock(return_value=frame_data or b"\x00" * 100)
    return mm


def _make_turbojpeg_mock(output_array: np.ndarray | None = None) -> dict:
    """Return a sys.modules patch dict for turbojpeg."""
    if output_array is None:
        output_array = np.zeros((480, 640, 3), dtype=np.uint8)
    tj_instance = mock.MagicMock()
    tj_instance.decode.return_value = output_array
    tj_cls = mock.MagicMock(return_value=tj_instance)
    tj_module = mock.MagicMock()
    tj_module.TurboJPEG = tj_cls
    tj_module.TJPF_RGB = 0
    tj_module.TJPF_BGR = 1
    tj_module.TJPF_GRAY = 6
    return {"turbojpeg": tj_module}


@contextmanager
def _mock_v4l2_device(
    num_buffers: int = 1,
    frame_data: bytes | None = None,
    select_returns: tuple | None = None,
    turbojpeg_output: np.ndarray | None = None,
):
    """Context manager providing a fully-mocked V4L2 hardware stack.

    Yields:
        (mock_mm, mock_xioctl, mock_os_close)
    """
    if select_returns is None:
        select_returns = ([_FAKE_FD], [], [])

    mock_mm = _make_mmap_mock(frame_data)
    xioctl_fn = _make_xioctl(num_buffers=num_buffers, dqbuf_frame_data=frame_data)
    tj_patch = _make_turbojpeg_mock(turbojpeg_output)

    with mock.patch("physicalai.capture.cameras.uvc.v4l2._camera.os.open", return_value=_FAKE_FD):
        with mock.patch("physicalai.capture.cameras.uvc.v4l2._camera.os.close") as mock_close:
            with mock.patch(
                "physicalai.capture.cameras.uvc.v4l2._camera.xioctl",
                side_effect=xioctl_fn,
            ) as mock_xi:
                with mock.patch(
                    "physicalai.capture.cameras.uvc.v4l2._camera.mmap.mmap",
                    return_value=mock_mm,
                ):
                    with mock.patch(
                        "physicalai.capture.cameras.uvc.v4l2._camera.select.select",
                        return_value=select_returns,
                    ):
                        with mock.patch.dict(sys.modules, tj_patch):
                            yield mock_mm, mock_xi, mock_close


def _connected_cam(
    num_buffers: int = 1,
    frame_data: bytes | None = None,
    select_returns: tuple | None = None,
    turbojpeg_output: np.ndarray | None = None,
    **kwargs,
) -> tuple[V4L2Camera, object, object, object]:
    """Helper that returns (cam, mock_mm, mock_xi, mock_close) with cam already connected.

    Returns contextmanager tuple for use in tests that need post-connect mocking.
    """
    raise NotImplementedError("Use _mock_v4l2_device context manager directly")


# ---------------------------------------------------------------------------
# Test 1
# ---------------------------------------------------------------------------


def test_constructor_defaults() -> None:
    """V4L2Camera has expected default parameter values."""
    cam = V4L2Camera()
    assert cam._device_path == "/dev/video0"  # noqa: SLF001
    assert cam._width == 640  # noqa: SLF001
    assert cam._height == 480  # noqa: SLF001
    assert cam._fps == 30  # noqa: SLF001
    assert cam._num_buffers == 4  # noqa: SLF001
    assert cam._pixel_format == "mjpeg"  # noqa: SLF001
    assert not cam.is_connected


# ---------------------------------------------------------------------------
# Test 2
# ---------------------------------------------------------------------------


def test_constructor_invalid_pixel_format() -> None:
    """ValueError raised for unsupported pixel_format."""
    with pytest.raises(ValueError, match="h264"):
        V4L2Camera(pixel_format="h264")


# ---------------------------------------------------------------------------
# Test 3
# ---------------------------------------------------------------------------


def test_device_id_returns_device_path() -> None:
    """device_id property returns the configured device path."""
    cam = V4L2Camera(device_path="/dev/video2")
    assert cam.device_id == "/dev/video2"


# ---------------------------------------------------------------------------
# Test 4
# ---------------------------------------------------------------------------


def test_connect_opens_device_and_starts_streaming() -> None:
    """connect() opens fd, runs ioctls, mmaps buffers, and marks connected."""
    with _mock_v4l2_device(num_buffers=2):
        cam = V4L2Camera(num_buffers=2)
        cam.connect()

    assert cam.is_connected


# ---------------------------------------------------------------------------
# Test 6
# ---------------------------------------------------------------------------


def test_connect_timeout_raises() -> None:
    """connect() raises CaptureTimeoutError when select() returns no ready fds."""
    with _mock_v4l2_device(num_buffers=1, select_returns=([], [], [])):
        cam = V4L2Camera(num_buffers=1)
        with pytest.raises(CaptureTimeoutError):
            cam.connect()


# ---------------------------------------------------------------------------
# Test 7
# ---------------------------------------------------------------------------


def test_read_returns_frame() -> None:
    """read() returns a Frame with decoded data and sequence=0."""
    frame_data = b"\xff\xd8\xff" + b"\x00" * 97  # fake JPEG bytes
    expected_arr = np.zeros((480, 640, 3), dtype=np.uint8)

    with _mock_v4l2_device(
        num_buffers=1,
        frame_data=frame_data,
        turbojpeg_output=expected_arr,
    ) as (_, _, _):
        cam = V4L2Camera(num_buffers=1)
        cam.connect()
        frame = cam.read()

    assert isinstance(frame, Frame)
    assert frame.sequence == 0
    assert frame.data.shape == (480, 640, 3)


# ---------------------------------------------------------------------------
# Test 8
# ---------------------------------------------------------------------------


def test_read_not_connected_raises() -> None:
    """read() raises NotConnectedError when camera is not connected."""
    cam = V4L2Camera()
    with pytest.raises(NotConnectedError):
        cam.read()


# ---------------------------------------------------------------------------
# Test 9
# ---------------------------------------------------------------------------


def test_read_timeout_raises() -> None:
    """read() raises CaptureTimeoutError when select() returns no ready fds."""
    frame_data = b"\x00" * 100

    # Connect succeeds (select returns ready fd), then read times out.
    connect_select = ([_FAKE_FD], [], [])
    read_select = ([], [], [])

    with _mock_v4l2_device(num_buffers=1, frame_data=frame_data):
        cam = V4L2Camera(num_buffers=1)
        cam.connect()

    # Now patch select only for the read() call.
    xioctl_fn = _make_xioctl(num_buffers=1, dqbuf_frame_data=frame_data)
    mm = _make_mmap_mock(frame_data)
    tj_patch = _make_turbojpeg_mock()

    with mock.patch(
        "physicalai.capture.cameras.uvc.v4l2._camera.select.select",
        return_value=read_select,
    ):
        with mock.patch.dict(sys.modules, tj_patch):
            with pytest.raises(CaptureTimeoutError):
                cam.read(timeout=0.01)


# ---------------------------------------------------------------------------
# Test 10
# ---------------------------------------------------------------------------


def test_read_sequence_increments() -> None:
    """read() increments sequence number on each call (0, 1, 2 …)."""
    frame_data = b"\xff\xd8\xff" + b"\x00" * 97
    expected_arr = np.zeros((480, 640, 3), dtype=np.uint8)

    with _mock_v4l2_device(
        num_buffers=1,
        frame_data=frame_data,
        turbojpeg_output=expected_arr,
    ):
        cam = V4L2Camera(num_buffers=1)
        cam.connect()
        f1 = cam.read()
        f2 = cam.read()

    assert f1.sequence == 0
    assert f2.sequence == 1


# ---------------------------------------------------------------------------
# Test 11
# ---------------------------------------------------------------------------


def test_read_latest_drains_buffers() -> None:
    """read_latest() drains the allocated ready buffers and returns the last one."""
    frame_data = b"\xff\xd8\xff" + b"\x00" * 97
    expected_arr = np.zeros((480, 640, 3), dtype=np.uint8)
    with _mock_v4l2_device(num_buffers=2, frame_data=frame_data, turbojpeg_output=expected_arr):
        camera = V4L2Camera(num_buffers=2)
        camera.connect()
        with mock.patch(
            "physicalai.capture.cameras.uvc.v4l2._camera.select.select",
            side_effect=[([_FAKE_FD], [], []), ([_FAKE_FD], [], [])],
        ):
            frame = camera.read_latest()
    assert isinstance(frame, Frame)
    assert frame.data.shape == (480, 640, 3)
    assert frame.sequence == 1


# ---------------------------------------------------------------------------
# Test 12
# ---------------------------------------------------------------------------


def test_disconnect_releases_resources() -> None:
    """disconnect() calls mmap.close() for each buffer and os.close()."""
    with _mock_v4l2_device(num_buffers=2) as (mock_mm, _, mock_close):
        cam = V4L2Camera(num_buffers=2)
        cam.connect()
        cam.disconnect()

    assert not cam.is_connected
    mock_mm.close.assert_called()
    mock_close.assert_called_once_with(_FAKE_FD)


# ---------------------------------------------------------------------------
# Test 13
# ---------------------------------------------------------------------------


def test_context_manager() -> None:
    """Context manager connects on enter and disconnects on exit."""
    with _mock_v4l2_device(num_buffers=1):
        cam = V4L2Camera(num_buffers=1)
        with cam:
            assert cam.is_connected
    assert not cam.is_connected


# ---------------------------------------------------------------------------
# Test 14
# ---------------------------------------------------------------------------


def test_discover_delegates_to_discover_v4l2() -> None:
    """V4L2Camera.discover() delegates to discover_v4l2 and returns its result."""
    mock_devices = [mock.MagicMock(spec=DeviceInfo)]
    with mock.patch("physicalai.capture.cameras.uvc.v4l2._discover.discover_v4l2", return_value=mock_devices):
        result = V4L2Camera.discover()
    assert result == mock_devices


# ---------------------------------------------------------------------------
# Test 15
# ---------------------------------------------------------------------------


def test_color_mode_gray_produces_2d_array() -> None:
    """MJPEG decode with GRAY ColorMode squeezes (H, W, 1) → (H, W)."""
    frame_data = b"\xff\xd8\xff" + b"\x00" * 97
    # turbojpeg returns (H, W, 1) for GRAY pixel format
    gray_array = np.zeros((480, 640, 1), dtype=np.uint8)

    with _mock_v4l2_device(
        num_buffers=1,
        frame_data=frame_data,
        turbojpeg_output=gray_array,
    ):
        cam = V4L2Camera(num_buffers=1, color_mode=ColorMode.GRAY)
        cam.connect()
        frame = cam.read()

    assert frame.data.shape == (480, 640)


_IS_64BIT = sys.maxsize > 2**32


@pytest.mark.skipif(not _IS_64BIT, reason="struct sizes differ on 32-bit")
def test_v4l2_format_struct_size() -> None:
    from physicalai.capture.cameras.uvc.v4l2._ioctl import v4l2_format

    assert ctypes.sizeof(v4l2_format) == 208, (
        f"Expected 208, got {ctypes.sizeof(v4l2_format)}. "
        "v4l2_format union must include v4l2_window (pointer alignment=8 forces 4-byte padding)."
    )


@pytest.mark.skipif(not _IS_64BIT, reason="struct sizes differ on 32-bit")
def test_v4l2_buffer_struct_size() -> None:
    assert ctypes.sizeof(v4l2_buffer) == 88, (
        f"Expected 88, got {ctypes.sizeof(v4l2_buffer)}. "
        "v4l2_buffer.m must be a Union with c_ulong userptr (8 bytes on 64-bit)."
    )


@pytest.mark.skipif(not _IS_64BIT, reason="struct offset differs on 32-bit")
def test_v4l2_format_fmt_offset() -> None:
    from physicalai.capture.cameras.uvc.v4l2._ioctl import v4l2_format

    assert v4l2_format.fmt.offset == 8, (
        f"Expected fmt at offset 8, got {v4l2_format.fmt.offset}. "
        "Missing 4-byte alignment padding between type (uint32) and fmt (union with pointer)."
    )


@pytest.mark.skipif(not _IS_64BIT, reason="struct sizes differ on 32-bit")
def test_v4l2_timecode_struct_size() -> None:
    assert ctypes.sizeof(v4l2_timecode) == 16, f"Expected 16, got {ctypes.sizeof(v4l2_timecode)}."


@pytest.mark.skipif(not _IS_64BIT, reason="ioctl numbers differ on 32-bit")
def test_ioctl_numbers_match_kernel() -> None:
    from physicalai.capture.cameras.uvc.v4l2._ioctl import (
        VIDIOC_DQBUF,
        VIDIOC_QBUF,
        VIDIOC_QUERYCAP,
        VIDIOC_QUERYBUF,
        VIDIOC_S_FMT,
    )

    assert (VIDIOC_QUERYCAP & 0xFFFFFFFF) == 0x80685600, f"VIDIOC_QUERYCAP: got 0x{VIDIOC_QUERYCAP & 0xFFFFFFFF:08X}"
    assert (VIDIOC_S_FMT & 0xFFFFFFFF) == 0xC0D05605, (
        f"VIDIOC_S_FMT: got 0x{VIDIOC_S_FMT & 0xFFFFFFFF:08X}, expected 0xC0D05605"
    )
    assert (VIDIOC_QUERYBUF & 0xFFFFFFFF) == 0xC0585609, f"VIDIOC_QUERYBUF: got 0x{VIDIOC_QUERYBUF & 0xFFFFFFFF:08X}"
    assert (VIDIOC_QBUF & 0xFFFFFFFF) == 0xC058560F, f"VIDIOC_QBUF: got 0x{VIDIOC_QBUF & 0xFFFFFFFF:08X}"
    assert (VIDIOC_DQBUF & 0xFFFFFFFF) == 0xC0585611, f"VIDIOC_DQBUF: got 0x{VIDIOC_DQBUF & 0xFFFFFFFF:08X}"
    assert (VIDIOC_G_FMT & 0xFFFFFFFF) == 0xC0D05604, f"VIDIOC_G_FMT: got 0x{VIDIOC_G_FMT & 0xFFFFFFFF:08X}"


# ---------------------------------------------------------------------------
# Buffer timestamp tests
# ---------------------------------------------------------------------------


def test_read_uses_kernel_buffer_timestamp() -> None:
    """read() uses buf.timestamp from kernel instead of time.monotonic()."""
    frame_data = b"\xff\xd8\xff" + b"\x00" * 97
    expected_arr = np.zeros((480, 640, 3), dtype=np.uint8)

    with _mock_v4l2_device(
        num_buffers=1,
        frame_data=frame_data,
        turbojpeg_output=expected_arr,
    ):
        cam = V4L2Camera(num_buffers=1)
        cam.connect()
        frame = cam.read()

    # The mock sets tv_sec=1000, tv_usec=500000 → 1000.5
    assert frame.timestamp == pytest.approx(1000.5)


def test_read_latest_without_cache_never_calls_blocking_read() -> None:
    with _mock_v4l2_device(frame_data=b"frame"):
        camera = V4L2Camera()
        camera.connect()
        with (
            mock.patch.object(camera, "read", side_effect=AssertionError("blocking read must not run")),
            mock.patch(
                "physicalai.capture.cameras.uvc.v4l2._camera.select.select", return_value=([], [], [])
            ) as select,
            pytest.raises(CaptureError, match="No frame available"),
        ):
            camera.read_latest()
        assert all(call.args[3] == 0 for call in select.call_args_list)


def test_read_latest_returns_original_cached_frame_without_waiting() -> None:
    with _mock_v4l2_device(frame_data=b"frame"):
        camera = V4L2Camera()
        camera.connect()
        first = camera.read()
        with mock.patch("physicalai.capture.cameras.uvc.v4l2._camera.select.select", return_value=([], [], [])):
            assert camera.read_latest() is first
            assert camera.read_latest().timestamp == first.timestamp


def test_read_latest_bounds_continuously_refilled_queue() -> None:
    with _mock_v4l2_device(num_buffers=2, frame_data=b"frame") as (_, ioctl, _):
        camera = V4L2Camera(num_buffers=2)
        camera.connect()
        original = ioctl.side_effect
        dequeues = 0

        def bounded(fd: int, request: int, arg: object) -> int:
            nonlocal dequeues
            if request == VIDIOC_DQBUF:
                dequeues += 1
                if dequeues > 2:
                    raise AssertionError("latest read exceeded allocated buffer count")
            return original(fd, request, arg)

        ioctl.side_effect = bounded
        frame = camera.read_latest()
        assert dequeues == 2
        assert frame.sequence == 1


def test_read_latest_handles_readiness_race_with_cached_frame() -> None:
    with _mock_v4l2_device(frame_data=b"frame") as (_, ioctl, _):
        camera = V4L2Camera()
        camera.connect()
        first = camera.read()
        ioctl.side_effect = BlockingIOError(11, "buffer no longer ready")
        assert camera.read_latest() is first


def test_read_preserves_kernel_sequence_gaps() -> None:
    with _mock_v4l2_device(frame_data=b"frame") as (_, ioctl, _):
        camera = V4L2Camera()
        camera.connect()
        original = ioctl.side_effect
        sequences = iter([11, 15])

        def sequenced(fd: int, request: int, arg: object) -> int:
            result = original(fd, request, arg)
            if request == VIDIOC_DQBUF:
                assert isinstance(arg, v4l2_buffer)
                arg.sequence = next(sequences)
            return result

        ioctl.side_effect = sequenced
        assert camera.read().sequence == 11
        assert camera.read().sequence == 15


@pytest.mark.parametrize("clock_flags", [0, 0x4000])
def test_unknown_timestamp_clock_uses_local_time_and_warns_once(clock_flags: int) -> None:
    with _mock_v4l2_device(frame_data=b"frame") as (_, ioctl, _):
        camera = V4L2Camera()
        camera.connect()
        original = ioctl.side_effect

        def unknown_clock(fd: int, request: int, arg: object) -> int:
            result = original(fd, request, arg)
            if request == VIDIOC_DQBUF:
                assert isinstance(arg, v4l2_buffer)
                arg.flags = clock_flags
                arg.timestamp.tv_sec = 1_700_000_000
            return result

        ioctl.side_effect = unknown_clock
        with (
            mock.patch("physicalai.capture.cameras.uvc.v4l2._camera.time.monotonic", return_value=42.0),
            mock.patch("physicalai.capture.cameras.uvc.v4l2._camera.logger") as logger,
        ):
            assert camera.read().timestamp == 42.0
            assert camera.read().timestamp == 42.0
            logger.warning.assert_called_once()
            assert "source capture age" in logger.warning.call_args.args[0]


def test_live_publisher_drains_pending_v4l2_frames() -> None:
    with _mock_v4l2_device(num_buffers=3, frame_data=b"frame"):
        camera = V4L2Camera(num_buffers=3)
        camera.connect()
        with mock.patch(
            "physicalai.capture.cameras.uvc.v4l2._camera.select.select",
            side_effect=[([_FAKE_FD], [], []), ([_FAKE_FD], [], []), ([_FAKE_FD], [], []), ([], [], [])],
        ):
            frame = worker._read_live_frame(camera)
        assert frame.sequence == 2


# ---------------------------------------------------------------------------
# Controls tests
# ---------------------------------------------------------------------------


def test_controls_applied_at_connect() -> None:
    """Controls dict passed to constructor is applied during connect()."""
    from physicalai.capture.cameras.uvc.v4l2._ioctl import VIDIOC_S_CTRL  # noqa: PLC0415

    s_ctrl_count = 0

    def _counting_xioctl(fd: int, request: int, arg: object) -> int:
        nonlocal s_ctrl_count
        base = _make_xioctl(num_buffers=1)
        result = base(fd, request, arg)
        if request == VIDIOC_S_CTRL:
            s_ctrl_count += 1
        return result

    with mock.patch("physicalai.capture.cameras.uvc.v4l2._camera.os.open", return_value=_FAKE_FD):
        with mock.patch("physicalai.capture.cameras.uvc.v4l2._camera.os.close"):
            with mock.patch(
                "physicalai.capture.cameras.uvc.v4l2._camera.xioctl",
                side_effect=_counting_xioctl,
            ):
                with mock.patch(
                    "physicalai.capture.cameras.uvc.v4l2._controls.xioctl",
                    side_effect=_counting_xioctl,
                ):
                    with mock.patch(
                        "physicalai.capture.cameras.uvc.v4l2._camera.mmap.mmap",
                        return_value=_make_mmap_mock(),
                    ):
                        with mock.patch(
                            "physicalai.capture.cameras.uvc.v4l2._camera.select.select",
                            return_value=([_FAKE_FD], [], []),
                        ):
                            cam = V4L2Camera(
                                num_buffers=1,
                                controls={0x009A0901: 1, 0x009A0902: 100},
                            )
                            cam.connect()

    assert s_ctrl_count == 2  # noqa: PLR2004
