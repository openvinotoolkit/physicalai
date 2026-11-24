# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: S101, SLF001, PLR2004

"""Freshness comes from source pixels, never receiver-generated timestamps."""

from __future__ import annotations

import contextlib
import ctypes
import importlib.util
import threading
import time
from collections import deque
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import MagicMock
from uuid import uuid4

import cv2
import numpy as np
import pytest

from physicalai.capture.camera import ColorMode
from physicalai.capture.cameras.ip import IPCamera
from physicalai.capture.errors import CaptureError, CaptureTimeoutError
from physicalai.capture.frame import Frame
from physicalai.capture.transport import _publisher_worker as worker
from physicalai.capture.transport._header import encode_frame
from physicalai.capture.transport._shared_camera import SharedCamera


class PixelStream:
    """A paced, real JPEG/HTTP source on a private ephemeral loopback port."""

    def __init__(self) -> None:
        self.stop = threading.Event()
        self.stalled = threading.Event()
        self.origin = time.monotonic()
        self.sequence = 0
        source = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *_args: object) -> None:
                pass

            def do_GET(self) -> None:  # noqa: N802
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                deadline = time.monotonic()
                try:
                    while not source.stop.is_set():
                        if not source.stalled.is_set():
                            source.sequence += 1
                            elapsed_ms = int((time.monotonic() - source.origin) * 1000)
                            values = np.array([source.sequence, elapsed_ms], dtype=">u4")
                            bits = np.unpackbits(values.view(np.uint8))
                            image = np.full((480, 640, 3), 80, dtype=np.uint8)
                            # 64 eight-pixel cells survive real lossy JPEG decoding.
                            image[:16, :512] = np.repeat(bits * 255, 8)[None, :, None]
                            ok, jpeg = cv2.imencode(".jpg", image)
                            assert ok
                            payload = jpeg.tobytes()
                            self.wfile.write(
                                b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                + str(len(payload)).encode()
                                + b"\r\n\r\n"
                                + payload
                                + b"\r\n",
                            )
                            self.wfile.flush()
                        deadline += 1 / 30
                        source.stop.wait(max(0, deadline - time.monotonic()))
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, name="test-mjpeg-server")
        self.thread.start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}/stream.mjpg"

    def close(self) -> None:
        self.stop.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        assert not self.thread.is_alive()


@contextlib.contextmanager
def pixel_stream() -> Iterator[PixelStream]:
    stream = PixelStream()
    try:
        yield stream
    finally:
        stream.close()


def pixel_metadata(image: np.ndarray) -> tuple[int, int]:
    bits = image[:16, :512].reshape(16, 64, 8, 3).mean(axis=(0, 2, 3)) > 128
    values = np.packbits(bits).view(">u4")
    return int(values[0]), int(values[1])


def measure_stream(camera_class: type[IPCamera], monkeypatch: pytest.MonkeyPatch) -> dict:
    """Exercise decode/conversion and publication cost plus consumer stalls."""
    with pixel_stream() as source:
        camera = camera_class(url=source.url)
        convert = camera._convert

        def costly_convert(image: np.ndarray) -> np.ndarray:
            time.sleep(0.010)
            return convert(image)

        monkeypatch.setattr(camera, "_convert", costly_convert)
        ages: list[float] = []
        ids: list[int] = []
        lags: list[int] = []
        try:
            camera.connect()
            started = time.monotonic()
            initial_source_sequence = source.sequence
            for index in range(60):
                if index in (20, 40):
                    time.sleep(0.2)  # consumer stalls must not stall acquisition
                iteration_start = time.monotonic()
                frame = camera.read()
                # Preserve the real transport's image copy/header work.
                encode_frame(frame, camera.color_mode, fps=30)
                time.sleep(0.014)
                frame_id, source_ms = pixel_metadata(frame.data)
                ids.append(frame_id)
                lags.append(source.sequence - frame_id)
                ages.append((time.monotonic() - source.origin) * 1000 - source_ms)
                worker._pace_capture(iteration_start, 30)
            elapsed = time.monotonic() - started
        finally:
            camera.disconnect()
        return {
            "age_ms_p50": float(np.median(ages)),
            "age_ms_max": max(ages),
            "early_age_ms_p50": float(np.median(ages[:20])),
            "late_age_ms_p50": float(np.median(ages[-20:])),
            "max_source_frame_lag": max(lags),
            "stall_recovery_id_jumps": [ids[20] - ids[19], ids[40] - ids[39]],
            "delivered_hz_including_stalls": len(ids) / elapsed,
            "source_requested_hz": 30,
            "source_actual_hz": (source.sequence - initial_source_sequence) / elapsed,
            "delivered_hz_excluding_stalls": len(ids) / (elapsed - 0.4),
        }


def test_real_mjpeg_freshness_under_load_and_stalls(monkeypatch: pytest.MonkeyPatch) -> None:
    result = measure_stream(IPCamera, monkeypatch)
    assert result["max_source_frame_lag"] <= 4
    assert min(result["stall_recovery_id_jumps"]) >= 4
    assert result["late_age_ms_p50"] <= result["early_age_ms_p50"] + 70
    assert result["age_ms_max"] < 180


@pytest.mark.skipif(importlib.util.find_spec("iceoryx2") is None, reason="iceoryx2 not installed")
@pytest.mark.parametrize("zero_copy", [False, True])
def test_real_shared_mjpeg_latest_after_consumer_stalls(zero_copy: bool) -> None:
    with pixel_stream() as source:
        camera = SharedCamera.from_config(
            {"class_path": "physicalai.capture.IPCamera", "init_args": {"url": source.url, "fps": 30}},
            service_name=f"physicalai/test/freshness-{uuid4().hex}/frame",
            zero_copy=zero_copy,
        )
        publisher = None
        try:
            camera.connect()
            publisher = camera._publisher
            assert publisher is not None
            previous_id, _ = pixel_metadata(camera.read_latest().data)
            for _ in range(5):
                time.sleep(0.2)
                frame = camera.read_latest()
                frame_id, source_ms = pixel_metadata(frame.data)
                assert frame_id >= previous_id + 3
                assert source.sequence - frame_id <= 4
                assert (time.monotonic() - source.origin) * 1000 - source_ms < 180
                previous_id = frame_id
        finally:
            publisher = publisher or camera._publisher
            camera.disconnect()
            if publisher is not None:
                publisher.stop()
                assert not publisher.is_alive


def test_real_mjpeg_stall_timeout_and_shutdown() -> None:
    with pixel_stream() as source:
        camera = IPCamera(url=source.url)
        try:
            camera.connect()
            reader = camera._reader
            source.stalled.set()
            # Allow already-decoded/network-buffered frames to settle.
            time.sleep(0.15)
            cached = camera.read_latest()
            with pytest.raises(CaptureTimeoutError):
                camera.read(timeout=0.02)
            assert camera.read_latest().sequence == cached.sequence
            assert camera.read_latest().timestamp == cached.timestamp
            started = time.monotonic()
            camera.disconnect()
            assert time.monotonic() - started < 2.2
            assert reader is not None and not reader.is_alive()
        finally:
            camera.disconnect()


@pytest.mark.parametrize(
    ("work", "fps", "expected"),
    [
        (0.0, 30, 1 / 30),
        (0.012, 30, 1 / 30 - 0.012),
        (1 / 30, 30, 0.0),
        (0.050, 30, 0.0),
        (0.0, 0, None),
    ],
)
def test_publisher_pacing_accounts_for_elapsed_work(
    monkeypatch: pytest.MonkeyPatch,
    work: float,
    fps: int,
    expected: float | None,
) -> None:
    shutdown = MagicMock()
    monkeypatch.setattr(worker, "shutdown", shutdown)
    monkeypatch.setattr(worker.time, "monotonic", lambda: work)
    worker._pace_capture(0.0, fps)
    if expected is None:
        shutdown.wait.assert_not_called()
    else:
        shutdown.wait.assert_called_once_with(pytest.approx(expected))


@pytest.mark.parametrize("zero_copy", [False, True])
def test_shared_latest_drains_after_stalls_and_preserves_source_metadata(zero_copy: bool) -> None:
    camera = SharedCamera.from_publisher("physicalai/test/pixel-freshness/frame", zero_copy=zero_copy)
    camera._connected = True
    queue: deque = deque(maxlen=2)
    buffers: list[ctypes.Array] = []

    def receive() -> MagicMock | None:
        return queue.popleft() if queue else None

    camera._subscriber = MagicMock()
    camera._subscriber.buffer_size = 2
    camera._subscriber.receive.side_effect = receive
    for source_id in range(20):
        image = np.full((4, 4, 3), source_id, dtype=np.uint8)
        header, pixels = encode_frame(Frame(image, source_id / 30, source_id), ColorMode.RGB)
        payload = bytes(header) + pixels
        buffer = ctypes.create_string_buffer(payload)
        buffers.append(buffer)
        sample = MagicMock()
        sample.payload.return_value.data_ptr = ctypes.addressof(buffer)
        sample.payload.return_value.number_of_elements = len(payload)
        queue.append(sample)
        if source_id in (0, 9, 19):
            frame = camera.read_latest()
            assert int(frame.data[0, 0, 0]) == source_id
            assert frame.sequence == source_id
            assert frame.timestamp == pytest.approx(source_id / 30, abs=1e-8)
            assert camera.read_latest() is frame
            if zero_copy:
                assert camera._held_sample is sample
    # An always-refilled queue must not trap read_latest in its drain loop.
    camera._subscriber.receive.side_effect = None
    camera._subscriber.receive.return_value = sample
    camera._subscriber.receive.reset_mock()
    assert camera.read_latest().sequence == 19
    assert camera._subscriber.receive.call_count == 2
    camera.disconnect()


def test_live_publisher_waits_then_returns_latest_pixels() -> None:
    camera = MagicMock()
    camera.read.return_value = Frame(np.full((2, 2, 3), 1, dtype=np.uint8), 1.0, 1)
    newest = Frame(np.full((2, 2, 3), 3, dtype=np.uint8), 3.0, 3)
    camera.read_latest.return_value = newest
    assert worker._read_live_frame(camera) is newest
    camera.read.assert_called_once_with(timeout=1.0)
    camera.read_latest.assert_called_once_with()


@pytest.mark.parametrize("method", ["read", "read_latest"])
def test_live_publisher_preserves_capture_errors(method: str) -> None:
    camera = MagicMock()
    error = CaptureError("camera disconnected")
    getattr(camera, method).side_effect = error
    with pytest.raises(CaptureError) as raised:
        worker._read_live_frame(camera)
    assert raised.value is error
    if method == "read":
        camera.read_latest.assert_not_called()


def test_shared_read_ignores_stale_notifications(monkeypatch: pytest.MonkeyPatch) -> None:
    camera = SharedCamera.from_publisher("physicalai/test/stale-notify/frame")
    camera._connected = True
    camera._subscriber = MagicMock()
    camera._subscriber.buffer_size = 2
    camera._subscriber.receive.return_value = None
    camera._listener = MagicMock()
    camera._listener.timed_wait_one.return_value = object()
    camera._latest = Frame(np.zeros((4, 4, 3), dtype=np.uint8), 1.0, 1)
    monkeypatch.setattr("physicalai.capture.transport._shared_camera.import_module", lambda _: MagicMock())
    times = iter([0, 0, 0.02])
    monkeypatch.setattr("physicalai.capture.transport._shared_camera.time.monotonic", lambda: next(times))
    with pytest.raises(CaptureTimeoutError):
        camera.read(timeout=0.01)
    camera._listener.timed_wait_one.assert_called_once()
    camera.disconnect()
