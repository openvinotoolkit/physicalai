# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import multiprocessing
from types import SimpleNamespace
from typing import Any, cast

import pytest

from physicalai.diagnostics.models import Inventory
from physicalai.diagnostics import streaming


def _fake_worker(request, connection, start, stop, duration):
    connection.send({"ready": True})
    assert start.wait(2)
    connection.send({"metrics": {"received_fps": request.fps, "frames": 10}})
    connection.close()


def _hanging_worker(request, connection, start, stop, duration):
    start.wait(30)
    import time
    time.sleep(30)


def test_isolated_fake_workers_synchronize_and_close(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(streaming, "_worker", _fake_worker)
    requests = [streaming.StreamRequest(selector=f"fake:{i}", width=640, height=480, fps=30, pixel_format="mjpeg") for i in range(2)]
    before = {p.pid for p in multiprocessing.active_children()}
    results = streaming._trial(requests, duration=0.01)
    assert all(result["metrics"]["received_fps"] == 30 for result in results)
    assert {p.pid for p in multiprocessing.active_children()} == before


def test_unsupported_platform_does_not_start_workers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(streaming, "_trial", lambda *a, **k: pytest.fail("must not start"))
    with pytest.raises(ValueError, match="Linux V4L2 only"):
        streaming.test_streams(Inventory(platform="darwin"), ["camera"], duration=1)


def test_hung_worker_is_stopped_and_reaped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(streaming, "_worker", _hanging_worker)
    monkeypatch.setattr(streaming, "_STARTUP_TIMEOUT", 0.1)
    before = {p.pid for p in multiprocessing.active_children()}
    result = streaming._trial([
        streaming.StreamRequest(selector="fake", width=640, height=480, fps=30, pixel_format="mjpeg"),
    ], duration=0.01)
    assert "deadline" in result[0]["error"]
    assert {p.pid for p in multiprocessing.active_children()} == before


def test_measure_received_frames_with_a_fake_camera(monkeypatch: pytest.MonkeyPatch) -> None:
    now = [0.0]
    class Camera:
        def read(self, **kwargs):
            now[0] += 0.1
            return SimpleNamespace(timestamp=now[0])
    monkeypatch.setattr(streaming.time, "monotonic", lambda: now[0])
    metrics = streaming.measure(cast(Any, Camera()), 1)
    assert metrics["received_fps"] == pytest.approx(10)
    assert metrics["largest_gap_s"] == pytest.approx(0.1)
    assert metrics["timestamp_fps"] == pytest.approx(10)


def test_slow_individual_rate_is_not_blamed_on_added_cameras() -> None:
    request = streaming.StreamRequest(selector="fake", width=640, height=480, fps=30, pixel_format="mjpeg")
    finding = streaming._comparison(request, {"metrics": {"received_fps": 10.4}}, {"metrics": {"received_fps": 10.2}})
    assert finding is not None and finding.severity == "INFO"
    assert "already present alone" in finding.summary
    assert streaming._comparison(request, {"error": "busy"}, {}) is None


def test_visible_owner_prevents_any_camera_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    from physicalai.diagnostics.models import Device, Endpoint, ProcessInfo, VideoMode
    monkeypatch.setattr(streaming, "_trial", lambda *a, **k: pytest.fail("must not touch an occupied device"))
    snapshot = Inventory(platform="linux", devices=[Device(id="camera", name="camera", endpoints=[
        Endpoint(id="video", path="/dev/video40", kind="video", role="capture", holders=[ProcessInfo(pid=42, name="owner")],
                 modes=[VideoMode(width=640, height=480, fps=30, pixel_format="MJPG")]),
    ])])
    with pytest.raises(ValueError, match="release the existing owner"):
        streaming.test_streams(snapshot, ["/dev/video40"], duration=1)


def test_incomplete_concurrent_workload_never_reports_a_successful_comparison(monkeypatch: pytest.MonkeyPatch) -> None:
    from physicalai.diagnostics.models import Device, Endpoint, VideoMode
    snapshot = Inventory(platform="linux", devices=[Device(id=f"camera:{i}", name="camera", endpoints=[
        Endpoint(id=f"video:{i}", path=f"/dev/video{i}", kind="video", role="capture",
                 modes=[VideoMode(width=640, height=480, fps=30, pixel_format="MJPG")]),
    ]) for i in range(2)])
    def trial(requests, duration):
        if len(requests) == 1:
            return [{"metrics": {"received_fps": 30}}]
        return [{"metrics": {"received_fps": 30}}, {"error": "camera stopped"}]
    monkeypatch.setattr(streaming, "_trial", trial)
    findings = streaming.test_streams(snapshot, ["/dev/video0", "/dev/video1"], duration=1)
    comparisons = [finding for finding in findings if finding.check_id == "stream.comparison"]
    assert len(comparisons) == 1 and comparisons[0].severity == "SKIP"


def test_cooperative_stop_does_not_read_another_frame() -> None:
    class Camera:
        def read(self, **kwargs):
            pytest.fail("must stop before another camera read")
    class Stop:
        def is_set(self):
            return True
    metrics = streaming.measure(cast(Any, Camera()), 1, stop=cast(Any, Stop()))
    assert metrics["frames"] == 0
