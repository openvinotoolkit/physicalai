# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: S101, PLR2004

from __future__ import annotations

import os
import time
from unittest import mock

import cv2
import numpy as np
import pytest

from physicalai.capture.camera import ColorMode
from physicalai.capture.cameras.ip import IPCamera
from physicalai.capture.errors import CaptureError, CaptureTimeoutError, NotConnectedError
from physicalai.capture.frame import Frame

URL = "rtsp://cam.example.internal:554/stream"


def _bgr_frame(height: int = 480, width: int = 640) -> np.ndarray:
    """A frame with distinct B/G/R channel values so conversion is checkable."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[..., 0] = 10  # B
    frame[..., 1] = 20  # G
    frame[..., 2] = 30  # R
    return frame


@pytest.fixture
def mock_cap():  # noqa: ANN201
    """A MagicMock standing in for cv2.VideoCapture — the only network I/O boundary.

    cv2.cvtColor / cv2.resize are the real OpenCV implementations (a core
    dependency here, unlike the vendor SDKs used by other camera backends),
    so conversion/resize correctness is verified against the real math.
    """
    cap = mock.MagicMock()
    cap.isOpened.return_value = True
    cap.read.return_value = (True, _bgr_frame())
    cap.set.return_value = True
    # Patch the cv2 reference already bound inside the module under test, not
    # the global "cv2" import — other tests in this suite that fake out cv2
    # entirely (e.g. test_basler.py) pop it from sys.modules in teardown,
    # which would otherwise leave a stale/mismatched patch target depending
    # on test execution order.
    with mock.patch(
        "physicalai.capture.cameras.ip._camera.cv2.VideoCapture",
        return_value=cap,
    ) as video_capture:
        cap.video_capture_mock = video_capture
        try:
            yield cap
        finally:
            # connect() sets this real process-global env var for rtsp://
            # URLs (see IPCamera's "RTSP transport" docstring section); don't
            # leak it into other tests/files in this process.
            os.environ.pop("OPENCV_FFMPEG_CAPTURE_OPTIONS", None)


def test_connect_opens_stream_and_reads_first_frame(mock_cap: mock.MagicMock) -> None:
    camera = IPCamera(url=URL)
    camera.connect()
    assert camera.is_connected
    mock_cap.video_capture_mock.assert_called_once()
    call_args = mock_cap.video_capture_mock.call_args
    assert call_args[0][0] == URL
    assert call_args[0][1] == cv2.CAP_FFMPEG


def test_connect_constructor_params_only_include_open_timeout(mock_cap: mock.MagicMock) -> None:
    """Regression test for a real (non-mocked) OpenCV incompatibility: passing
    CAP_PROP_READ_TIMEOUT_MSEC and/or CAP_PROP_BUFFERSIZE in the VideoCapture
    constructor's params array made a real opencv-python-headless build (5.x,
    FFmpeg backend) reject the whole array and fail to open *any* stream,
    valid or not — "unsupported parameters in .open(), ... Bailout". Verified
    against the real library, not just this mock. Only
    CAP_PROP_OPEN_TIMEOUT_MSEC may go in the constructor; the others must be
    applied via cap.set() after a successful open.
    """
    camera = IPCamera(url=URL)
    camera.connect(timeout=5.0)
    call_args = mock_cap.video_capture_mock.call_args
    params = call_args[0][2]
    assert params == [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000]
    mock_cap.set.assert_any_call(cv2.CAP_PROP_BUFFERSIZE, 1)


def test_connect_forces_tcp_transport_for_rtsp_urls(
    mock_cap: mock.MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression test for a real-camera failure (Reolink): without forcing
    TCP, FFmpeg's default RTSP media transport is UDP on many builds, which
    routers/NAT frequently drop — the session opens fine (isOpened() True)
    but no frame data ever arrives, and every read blocks until it gives up.
    Forcing TCP (matching the documented `ffprobe -rtsp_transport tcp`
    pre-flight check) fixed this against the real camera.
    """
    monkeypatch.delenv("OPENCV_FFMPEG_CAPTURE_OPTIONS", raising=False)
    seen_options: list[str | None] = []

    def _capture_env(*_args: object, **_kwargs: object) -> mock.MagicMock:
        seen_options.append(os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS"))
        return mock_cap

    mock_cap.video_capture_mock.side_effect = _capture_env

    camera = IPCamera(url=URL)
    camera.connect(timeout=4.0)

    assert len(seen_options) == 1
    options = seen_options[0]
    assert options is not None
    assert "rtsp_transport;tcp" in options
    assert "stimeout;4000000" in options


def test_connect_restores_previous_env_var_after_opening(
    mock_cap: mock.MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """connect() must not leak its temporary env var past the VideoCapture()
    call — otherwise it silently affects unrelated cv2.VideoCapture() opens
    elsewhere in the process for the rest of its lifetime.
    """
    monkeypatch.setenv("OPENCV_FFMPEG_CAPTURE_OPTIONS", "some_other_option;value")
    camera = IPCamera(url=URL)
    camera.connect(timeout=4.0)
    assert os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] == "some_other_option;value"


def test_connect_clears_env_var_after_opening_when_previously_unset(
    mock_cap: mock.MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENCV_FFMPEG_CAPTURE_OPTIONS", raising=False)
    camera = IPCamera(url=URL)
    camera.connect(timeout=4.0)
    assert "OPENCV_FFMPEG_CAPTURE_OPTIONS" not in os.environ


def test_connect_does_not_set_rtsp_options_for_non_rtsp_urls(
    mock_cap: mock.MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENCV_FFMPEG_CAPTURE_OPTIONS", raising=False)
    camera = IPCamera(url="http://cam.example.internal/stream.mjpg")
    camera.connect()
    assert "OPENCV_FFMPEG_CAPTURE_OPTIONS" not in os.environ


def test_connect_raises_capture_error_when_not_opened(mock_cap: mock.MagicMock) -> None:
    mock_cap.isOpened.return_value = False
    camera = IPCamera(url=URL)
    with pytest.raises(CaptureError, match="Failed to open"):
        camera.connect()
    mock_cap.release.assert_called_once()


def test_connect_timeout_raises_capture_timeout(mock_cap: mock.MagicMock) -> None:
    mock_cap.read.return_value = (False, None)
    camera = IPCamera(url=URL)
    with pytest.raises(CaptureTimeoutError):
        camera.connect(timeout=0.05)


def test_connect_releases_capture_when_first_frame_conversion_fails(
    mock_cap: mock.MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If reading/converting the first frame raises unexpectedly, the
    VideoCapture handle assigned to self._cap before this point must still
    be released — otherwise it leaks and orphans on a subsequent connect().
    """
    camera = IPCamera(url=URL)
    monkeypatch.setattr(camera, "_convert", mock.Mock(side_effect=RuntimeError("bad channel layout")))
    with pytest.raises(CaptureError, match="Failed to read first frame"):
        camera.connect()
    mock_cap.release.assert_called_once()
    assert not camera.is_connected


def test_connect_total_timeout_is_not_doubled(mock_cap: mock.MagicMock, monkeypatch: pytest.MonkeyPatch) -> None:
    """connect()'s overall wait must not exceed ~timeout, even if opening the
    stream itself consumes most of the budget (CAP_PROP_OPEN_TIMEOUT_MSEC
    only bounds the constructor call, not the first-frame wait after it).
    """
    times = iter([0.0, 10.0])
    monkeypatch.setattr(
        "physicalai.capture.cameras.ip._camera.time.monotonic",
        lambda: next(times, 10.0),
    )
    camera = IPCamera(url=URL)
    with pytest.raises(CaptureTimeoutError):
        camera.connect(timeout=5.0)
    # The single monotonic() check inside _wait_first_frame's loop condition
    # already saw a time past the deadline computed at the top of connect(),
    # so the read loop body never ran — proving both phases share one budget.
    mock_cap.read.assert_not_called()


def test_connect_clamps_read_timeout_to_remaining_budget(
    mock_cap: mock.MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each cap.read() attempt inside the first-frame wait must be bounded by
    whatever time is actually left before connect()'s deadline, not the full
    original timeout configured at construction — otherwise a slow open
    followed by one blocked read could overrun the caller's timeout.
    """
    # deadline = 0.0 + 5.0 = 5.0; the loop then sees 4.0s elapsed, i.e. 1.0s
    # left, and must clamp the read timeout to 1000ms, not the original 5000ms.
    times = iter([0.0, 4.0])
    monkeypatch.setattr(
        "physicalai.capture.cameras.ip._camera.time.monotonic",
        lambda: next(times, 4.0),
    )
    camera = IPCamera(url=URL)
    camera.connect(timeout=5.0)
    mock_cap.set.assert_any_call(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 1000.0)


def test_connect_sets_fps_best_effort(mock_cap: mock.MagicMock) -> None:
    camera = IPCamera(url=URL, fps=15)
    camera.connect()
    mock_cap.set.assert_any_call(cv2.CAP_PROP_FPS, 15)


def test_disconnect_releases_capture(mock_cap: mock.MagicMock) -> None:
    camera = IPCamera(url=URL)
    camera.connect()
    camera.disconnect()
    mock_cap.release.assert_called_once()
    assert not camera.is_connected


def test_disconnect_without_connect_is_safe() -> None:
    camera = IPCamera(url=URL)
    camera.disconnect()


def test_read_returns_rgb_frame_with_correct_shape_and_dtype(mock_cap: mock.MagicMock) -> None:
    camera = IPCamera(url=URL)
    camera.connect()
    frame = camera.read()
    assert isinstance(frame, Frame)
    assert frame.data.shape == (480, 640, 3)
    assert frame.data.dtype == np.uint8
    # BGR (10, 20, 30) -> RGB (30, 20, 10)
    assert tuple(frame.data[0, 0]) == (30, 20, 10)


def test_read_increments_sequence(mock_cap: mock.MagicMock) -> None:
    camera = IPCamera(url=URL)
    camera.connect()
    f1 = camera.read()
    f2 = camera.read()
    f3 = camera.read()
    assert f1.sequence == 1
    assert f2.sequence == 2
    assert f3.sequence == 3


def test_read_timeout_reports_capture_timeout(mock_cap: mock.MagicMock) -> None:
    camera = IPCamera(url=URL)
    camera.connect()
    mock_cap.read.return_value = (False, None)
    with pytest.raises(CaptureTimeoutError):
        camera.read(timeout=0.05)


def test_read_clamps_timeout_to_remaining_budget(
    mock_cap: mock.MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed first attempt must shrink the read timeout used for the next
    one to whatever's actually left, not keep re-applying the original full
    timeout on every retry.
    """
    camera = IPCamera(url=URL)
    camera.connect()
    mock_cap.set.reset_mock()
    mock_cap.read.side_effect = [(False, None), (True, _bgr_frame())]
    times = iter([0.0, 0.0, 1.0, 1.0])
    monkeypatch.setattr(
        "physicalai.capture.cameras.ip._camera.time.monotonic",
        lambda: next(times, 1.0),
    )
    camera.read(timeout=2.0)
    mock_cap.set.assert_any_call(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 2000.0)
    mock_cap.set.assert_any_call(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 1000.0)


def test_read_sleeps_between_failed_attempts_instead_of_busy_spinning(
    mock_cap: mock.MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dropped connection where cap.read() fails immediately (no network
    block) must not pin a CPU core in a tight loop — each failed attempt
    should back off by _POLL_INTERVAL_S.
    """
    camera = IPCamera(url=URL)
    camera.connect()
    mock_cap.read.side_effect = [(False, None), (False, None), (True, _bgr_frame())]
    sleep_mock = mock.MagicMock()
    monkeypatch.setattr("physicalai.capture.cameras.ip._camera.time.sleep", sleep_mock)
    camera.read()
    assert sleep_mock.call_args_list == [mock.call(IPCamera._POLL_INTERVAL_S)] * 2  # noqa: SLF001


def test_read_not_connected_raises() -> None:
    camera = IPCamera(url=URL)
    with pytest.raises(NotConnectedError):
        camera.read()


def test_read_latest_returns_cached_when_no_new_frame(mock_cap: mock.MagicMock) -> None:
    camera = IPCamera(url=URL)
    camera.connect()
    first = camera.read()
    mock_cap.read.return_value = (False, None)
    latest = camera.read_latest()
    assert latest.sequence == first.sequence
    np.testing.assert_array_equal(latest.data, first.data)


def test_read_latest_falls_back_to_connect_time_frame_with_valid_timestamp(
    mock_cap: mock.MagicMock,
) -> None:
    """The frame captured during connect() (cached as the read_latest()
    fallback) must have a real timestamp, not the 0.0 sentinel default —
    otherwise calling read_latest() right after connect(), before any
    read()/read_latest() has ever succeeded live, returns a Frame that looks
    like it was captured at process start rather than just now.
    """
    before = time.monotonic()
    camera = IPCamera(url=URL)
    camera.connect()
    mock_cap.read.return_value = (False, None)
    latest = camera.read_latest()
    assert latest.timestamp >= before
    assert latest.timestamp > 0.0


def test_read_latest_uses_minimal_read_timeout_not_read_timeout(mock_cap: mock.MagicMock) -> None:
    """read_latest() must not inherit whatever multi-second timeout a prior
    read()/connect() call configured on the capture object — otherwise a
    stalled stream blocks the control loop instead of returning the cached
    frame promptly.
    """
    camera = IPCamera(url=URL)
    camera.connect()
    camera.read(timeout=5.0)  # leaves a 5s read timeout configured on cap
    mock_cap.set.reset_mock()
    camera.read_latest()
    mock_cap.set.assert_called_once_with(cv2.CAP_PROP_READ_TIMEOUT_MSEC, IPCamera._LATEST_READ_TIMEOUT_MS)  # noqa: SLF001


def test_read_latest_not_connected_raises() -> None:
    camera = IPCamera(url=URL)
    with pytest.raises(NotConnectedError):
        camera.read_latest()


def test_read_latest_raises_when_never_connected_frame(mock_cap: mock.MagicMock) -> None:
    mock_cap.read.side_effect = [(True, _bgr_frame())] + [(False, None)] * 5
    camera = IPCamera(url=URL)
    camera.connect()
    camera._last_frame_data = None  # simulate no cached frame available  # noqa: SLF001
    with pytest.raises(CaptureError, match="No frame available"):
        camera.read_latest()


def test_color_mode_bgr_skips_conversion(mock_cap: mock.MagicMock) -> None:
    camera = IPCamera(url=URL, color_mode=ColorMode.BGR)
    camera.connect()
    frame = camera.read()
    assert tuple(frame.data[0, 0]) == (10, 20, 30)


def test_color_mode_gray(mock_cap: mock.MagicMock) -> None:
    camera = IPCamera(url=URL, color_mode=ColorMode.GRAY)
    camera.connect()
    frame = camera.read()
    assert frame.data.shape == (480, 640)
    assert frame.data.dtype == np.uint8


def test_resize_applied_when_dimensions_requested(mock_cap: mock.MagicMock) -> None:
    camera = IPCamera(url=URL, width=320, height=240)
    camera.connect()
    frame = camera.read()
    assert frame.data.shape == (240, 320, 3)


def test_resize_applied_when_only_width_requested(mock_cap: mock.MagicMock) -> None:
    """Requesting only one dimension must still resize that axis, not
    silently no-op both (native frame is 480x640)."""
    camera = IPCamera(url=URL, width=320)
    camera.connect()
    frame = camera.read()
    assert frame.data.shape == (480, 320, 3)


def test_resize_applied_when_only_height_requested(mock_cap: mock.MagicMock) -> None:
    camera = IPCamera(url=URL, height=240)
    camera.connect()
    frame = camera.read()
    assert frame.data.shape == (240, 640, 3)


def test_resize_skipped_when_dimensions_not_requested(mock_cap: mock.MagicMock) -> None:
    camera = IPCamera(url=URL)
    camera.connect()
    frame = camera.read()
    assert frame.data.shape == (480, 640, 3)


def test_device_id_redacts_credentials(mock_cap: mock.MagicMock) -> None:
    camera = IPCamera(url="rtsp://admin:s3cr3t@10.0.0.5:554/stream")
    assert "s3cr3t" not in camera.device_id
    assert "admin" not in camera.device_id
    assert camera.device_id == "ip:rtsp://***@10.0.0.5:554/stream"


def test_device_id_without_credentials_unchanged(mock_cap: mock.MagicMock) -> None:
    camera = IPCamera(url=URL)
    assert camera.device_id == f"ip:{URL}"


def test_discover_returns_empty_list() -> None:
    assert IPCamera.discover() == []
