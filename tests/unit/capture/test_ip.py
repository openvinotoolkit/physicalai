# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: S101, SLF001, PLR2004

from __future__ import annotations

import os
import queue
import threading
import time
from collections.abc import Iterator
from unittest import mock

import cv2
import numpy as np
import pytest

from physicalai.capture.camera import ColorMode
from physicalai.capture.cameras.ip import IPCamera
from physicalai.capture.errors import CaptureError, CaptureTimeoutError, NotConnectedError

URL = "rtsp://cam.example.internal:554/stream"


def _bgr_frame(height: int = 480, width: int = 640) -> np.ndarray:
    frame = np.empty((height, width, 3), dtype=np.uint8)
    frame[:] = (10, 20, 30)
    return frame


@pytest.fixture
def mock_cap() -> Iterator[mock.MagicMock]:
    """Only fake the native I/O boundary; use real conversion/resize."""
    cap = mock.MagicMock()
    cap.isOpened.return_value = True

    def read() -> tuple[bool, np.ndarray]:
        time.sleep(0.005)
        return True, _bgr_frame()

    cap.read.side_effect = read
    with mock.patch("physicalai.capture.cameras.ip._camera.cv2.VideoCapture", return_value=cap) as constructor:
        cap.video_capture_mock = constructor
        yield cap


@pytest.fixture(autouse=True)
def close_cameras(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    cameras: list[IPCamera] = []
    init = IPCamera.__init__

    def tracked_init(
        camera: IPCamera,
        *,
        url: str,
        fps: int = 30,
        width: int | None = None,
        height: int | None = None,
        color_mode: ColorMode = ColorMode.RGB,
    ) -> None:
        init(camera, url=url, fps=fps, width=width, height=height, color_mode=color_mode)
        cameras.append(camera)

    monkeypatch.setattr(IPCamera, "__init__", tracked_init)
    yield
    for camera in cameras:
        camera.disconnect()


def test_open_parameters_and_fps(mock_cap: mock.MagicMock) -> None:
    camera = IPCamera(url=URL, fps=15)
    camera.connect(timeout=4.0)
    assert camera.is_connected
    mock_cap.video_capture_mock.assert_called_once_with(
        URL,
        cv2.CAP_FFMPEG,
        [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 4000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 1000],
    )
    mock_cap.set.assert_any_call(cv2.CAP_PROP_BUFFERSIZE, 1)
    mock_cap.set.assert_any_call(cv2.CAP_PROP_FPS, 15)
    # READ_TIMEOUT is open-only on real FFmpeg, not a cap.set() property.
    assert all(call.args[0] != cv2.CAP_PROP_READ_TIMEOUT_MSEC for call in mock_cap.set.call_args_list)


@pytest.mark.parametrize("previous", [None, "some_other_option;value"])
def test_rtsp_options_are_scoped_to_open(
    mock_cap: mock.MagicMock,
    monkeypatch: pytest.MonkeyPatch,
    previous: str | None,
) -> None:
    if previous is None:
        monkeypatch.delenv("OPENCV_FFMPEG_CAPTURE_OPTIONS", raising=False)
    else:
        monkeypatch.setenv("OPENCV_FFMPEG_CAPTURE_OPTIONS", previous)
    seen: list[str | None] = []

    def capture(*_args: object) -> mock.MagicMock:
        seen.append(os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS"))
        return mock_cap

    mock_cap.video_capture_mock.side_effect = capture
    IPCamera(url=URL).connect(timeout=4.0)
    assert seen == ["rtsp_transport;tcp|stimeout;4000000"]
    assert os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS") == previous


def test_http_does_not_set_rtsp_options(mock_cap: mock.MagicMock, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENCV_FFMPEG_CAPTURE_OPTIONS", raising=False)
    IPCamera(url="http://cam.example.internal/stream.mjpg").connect()
    assert "OPENCV_FFMPEG_CAPTURE_OPTIONS" not in os.environ


def test_open_failure_releases_handle(mock_cap: mock.MagicMock) -> None:
    mock_cap.isOpened.return_value = False
    with pytest.raises(CaptureError, match="Failed to open"):
        IPCamera(url=URL).connect()
    mock_cap.release.assert_called_once()


def test_open_exception_redacts_url(mock_cap: mock.MagicMock) -> None:
    mock_cap.video_capture_mock.side_effect = RuntimeError("native failure")
    with pytest.raises(CaptureError, match="Failed to open") as exc:
        IPCamera(url="rtsp://user:secret@cam.example.internal/stream").connect()
    assert "secret" not in str(exc.value)


def test_connect_timeout_releases_and_joins(mock_cap: mock.MagicMock) -> None:
    mock_cap.read.side_effect = None
    mock_cap.read.return_value = (False, None)
    camera = IPCamera(url=URL)
    with pytest.raises(CaptureTimeoutError):
        camera.connect(timeout=0.03)
    assert camera._reader is None
    assert not camera.is_connected
    mock_cap.release.assert_called_once()
    assert mock_cap.read.call_count < 100  # failed reads back off, not busy-spin


def test_reader_start_failure_releases_handle(mock_cap: mock.MagicMock, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(threading.Thread, "start", mock.Mock(side_effect=RuntimeError("cannot start thread")))
    camera = IPCamera(url=URL)
    with pytest.raises(CaptureError, match="Failed to start"):
        camera.connect()
    mock_cap.release.assert_called_once()
    assert camera._reader is None
    assert not camera.is_connected


def test_conversion_failure_releases_and_joins(mock_cap: mock.MagicMock, monkeypatch: pytest.MonkeyPatch) -> None:
    camera = IPCamera(url=URL)
    monkeypatch.setattr(camera, "_convert", mock.Mock(side_effect=RuntimeError("bad channel layout")))
    with pytest.raises(CaptureError, match="acquisition failed"):
        camera.connect()
    mock_cap.release.assert_called_once()
    assert camera._reader is None


def test_connect_uses_remaining_deadline(mock_cap: mock.MagicMock, monkeypatch: pytest.MonkeyPatch) -> None:
    times = iter([0.0, 10.0])
    monkeypatch.setattr("physicalai.capture.cameras.ip._camera.time.monotonic", lambda: next(times, 10.0))
    with pytest.raises(CaptureTimeoutError):
        IPCamera(url=URL).connect(timeout=5.0)


@pytest.mark.parametrize("first_timestamp", [0.99, 1.02])
def test_first_frame_deadline_uses_first_acquisition_not_latest(
    monkeypatch: pytest.MonkeyPatch,
    first_timestamp: float,
) -> None:
    camera = IPCamera(url=URL)
    camera._first_timestamp = first_timestamp
    camera._last_timestamp = 1.02  # reader overwrote the slot before waiter resumed
    monkeypatch.setattr("physicalai.capture.cameras.ip._camera.time.monotonic", lambda: 1.03)
    if first_timestamp <= 1.0:
        camera._wait_first_frame(deadline=1.0)
    else:
        with pytest.raises(CaptureTimeoutError):
            camera._wait_first_frame(deadline=1.0)


def test_disconnect_and_reconnect(mock_cap: mock.MagicMock) -> None:
    camera = IPCamera(url=URL)
    camera.connect()
    reader = camera._reader
    camera.connect()  # idempotent, no second native owner
    assert camera._reader is reader
    camera.disconnect()
    assert reader is not None and not reader.is_alive()
    mock_cap.release.assert_called_once()
    assert not camera.is_connected
    camera.connect()
    assert camera._reader is not reader
    assert camera.read_latest().sequence < 3
    camera.disconnect()
    assert mock_cap.release.call_count == 2


def test_disconnect_without_connect() -> None:
    IPCamera(url=URL).disconnect()


@pytest.mark.parametrize("method", ["read", "read_latest"])
def test_not_connected(method: str) -> None:
    with pytest.raises(NotConnectedError):
        getattr(IPCamera(url=URL), method)()


@pytest.mark.parametrize(
    ("mode", "width", "height", "shape", "pixel"),
    [
        (ColorMode.RGB, None, None, (480, 640, 3), (30, 20, 10)),
        (ColorMode.BGR, None, None, (480, 640, 3), (10, 20, 30)),
        (ColorMode.GRAY, None, None, (480, 640), None),
        (ColorMode.RGB, 320, 240, (240, 320, 3), (30, 20, 10)),
        (ColorMode.RGB, 320, None, (480, 320, 3), (30, 20, 10)),
        (ColorMode.RGB, None, 240, (240, 640, 3), (30, 20, 10)),
    ],
)
def test_conversion(
    mock_cap: mock.MagicMock,
    mode: ColorMode,
    width: int | None,
    height: int | None,
    shape: tuple[int, ...],
    pixel: tuple[int, ...] | None,
) -> None:
    camera = IPCamera(url=URL, color_mode=mode, width=width, height=height)
    camera.connect()
    frame = camera.read()
    assert frame.data.shape == shape
    assert frame.data.dtype == np.uint8
    if pixel is not None:
        assert tuple(frame.data[0, 0]) == pixel


def test_read_sequence_advances(mock_cap: mock.MagicMock) -> None:
    camera = IPCamera(url=URL)
    camera.connect()
    first_timestamp = camera._first_timestamp
    frames = [camera.read() for _ in range(3)]
    assert frames[0].sequence < frames[1].sequence < frames[2].sequence
    assert camera._first_timestamp == first_timestamp
    assert first_timestamp is not None
    assert first_timestamp < frames[-1].timestamp


def test_stall_keeps_metadata_and_consumer_timeout_bounded(mock_cap: mock.MagicMock) -> None:
    # Block native acquisition independently of consumer calls.
    frames: queue.Queue[np.ndarray | Exception] = queue.Queue()
    frames.put(_bgr_frame())

    def read() -> tuple[bool, np.ndarray | None]:
        try:
            frame = frames.get(timeout=0.05)
        except queue.Empty:
            return False, None
        if isinstance(frame, Exception):
            raise frame
        return True, frame

    mock_cap.read.side_effect = read
    camera = IPCamera(url=URL)
    camera.connect()
    first = camera.read_latest()
    before_calls = mock_cap.read.call_count
    for _ in range(100):
        latest = camera.read_latest()
        assert latest.sequence == first.sequence
        assert latest.timestamp == first.timestamp
    assert mock_cap.read.call_count <= before_calls + 1
    with pytest.raises(CaptureTimeoutError):
        camera.read(timeout=0.01)
    frames.put(_bgr_frame())
    assert camera.read().sequence > first.sequence
    frames.put(RuntimeError("native error containing credentials"))
    with pytest.raises(CaptureError, match="acquisition failed"):
        camera.read()
    with pytest.raises(CaptureError, match="acquisition failed"):
        camera.read_latest()
    assert not camera.is_connected
    frames.put(_bgr_frame())
    camera.connect()
    assert camera.is_connected
    assert camera.read_latest().sequence == 0


def test_release_failure_never_escapes_reader(
    mock_cap: mock.MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    escaped: list[object] = []
    monkeypatch.setattr(threading, "excepthook", escaped.append)
    mock_cap.release.side_effect = RuntimeError("rtsp://user:secret@cam.example.internal/stream")
    camera = IPCamera(url=URL)
    camera.connect()
    camera.disconnect()
    assert escaped == []
    assert camera._reader is None


def test_disconnect_wakes_waiting_consumer(mock_cap: mock.MagicMock) -> None:
    camera = IPCamera(url=URL)
    camera.connect()
    mock_cap.read.side_effect = None
    mock_cap.read.return_value = (False, None)
    # Consume any frame already in flight before starting the blocked read.
    time.sleep(0.02)
    camera.read_latest()
    errors: list[Exception] = []
    started = threading.Event()

    def consume() -> None:
        started.set()
        try:
            camera.read(timeout=10)
        except CaptureError as exc:
            errors.append(exc)

    consumer = threading.Thread(target=consume)
    consumer.start()
    assert started.wait(1)
    camera.disconnect()
    consumer.join(timeout=1)
    assert not consumer.is_alive()
    assert len(errors) == 1 and isinstance(errors[0], NotConnectedError)


def test_identity_redaction_and_discovery() -> None:
    assert IPCamera(url="rtsp://user:secret@cam.example.internal/stream").device_id == (
        "ip:rtsp://***@cam.example.internal/stream"
    )
    assert IPCamera(url=URL).device_id == f"ip:{URL}"
    assert IPCamera.discover() == []
