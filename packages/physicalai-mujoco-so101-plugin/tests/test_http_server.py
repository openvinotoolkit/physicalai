from __future__ import annotations

import asyncio
import http.client
import json
import queue
import socket
import time
import urllib.error
import urllib.request
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from physicalai_mujoco_so101_plugin.http_server import (
    FrameBuffer,
    HttpServer,
    ResetCommand,
    ShutdownCommand,
    SwitchSceneCommand,
    _mjpeg_stream,
    build_app,
    encode_jpeg,
)


@pytest.fixture
def frame() -> np.ndarray:
    return np.full((48, 64, 3), [10, 120, 230], dtype=np.uint8)


@pytest.fixture
def app_context(frame: np.ndarray) -> dict:
    buffers = {"overview": FrameBuffer("overview")}
    buffers["overview"].put(frame)
    commands: queue.Queue = queue.Queue()
    status = {
        "connected": True,
        "scene": "single_pick_place",
        "scenes": ["pick_lift", "single_pick_place"],
        "cameras": [
            {
                "name": "overview",
                "width": 64,
                "height": 48,
                "fps": 30,
                "device": None,
                "rendering": True,
            },
        ],
    }
    app = build_app(
        service_name="mujoco-so101",
        buffers=buffers,
        commands=commands,
        get_status=lambda: status,
    )
    return {"app": app, "buffers": buffers, "commands": commands}


@pytest.fixture
def client(app_context: dict) -> TestClient:
    return TestClient(app_context["app"])


class TestFrameBuffer:
    def test_snapshot_empty(self) -> None:
        assert FrameBuffer("cam").snapshot() is None

    def test_put_and_snapshot(self, frame: np.ndarray) -> None:
        buffer = FrameBuffer("cam")
        buffer.put(frame)
        sample = buffer.snapshot()
        assert sample is not None
        assert sample.seq == 1
        assert np.array_equal(sample.frame, frame)

    def test_seq_increments(self, frame: np.ndarray) -> None:
        buffer = FrameBuffer("cam")
        buffer.put(frame)
        buffer.put(frame)
        sample = buffer.snapshot()
        assert sample is not None
        assert sample.seq == 2

    def test_wait_newer_than_returns_new_frame(self, frame: np.ndarray) -> None:
        buffer = FrameBuffer("cam")
        buffer.put(frame)
        sample = buffer.wait_newer_than(0, timeout=1.0)
        assert sample is not None
        assert sample.seq == 1

    def test_wait_newer_than_times_out_with_stale_sample(self, frame: np.ndarray) -> None:
        buffer = FrameBuffer("cam")
        buffer.put(frame)
        start = time.monotonic()
        sample = buffer.wait_newer_than(1, timeout=0.1)
        assert sample is not None
        assert sample.seq == 1
        assert time.monotonic() - start >= 0.1

    def test_wait_newer_than_times_out_empty(self) -> None:
        buffer = FrameBuffer("cam")
        assert buffer.wait_newer_than(0, timeout=0.05) is None


class TestEncodeJpeg:
    def test_jpeg_magic_bytes(self, frame: np.ndarray) -> None:
        data = encode_jpeg(frame, quality=85)
        assert data[:2] == b"\xff\xd8"


class TestAppEndpoints:
    def test_root(self, client: TestClient) -> None:
        response = client.get("/")
        assert response.status_code == 200
        body = response.json()
        assert body["service"] == "mujoco-so101"
        assert body["cameras"] == ["overview"]
        assert body["scene"] == "single_pick_place"

    def test_health(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["connected"] is True

    def test_cameras_lists_urls(self, client: TestClient) -> None:
        response = client.get("/cameras")
        assert response.status_code == 200
        cameras = response.json()
        assert cameras[0]["name"] == "overview"
        assert cameras[0]["stream_url"] == "/cameras/overview/mjpeg"
        assert cameras[0]["snapshot_url"] == "/cameras/overview/frame.jpg"
        assert cameras[0]["streaming"] is True

    def test_snapshot(self, client: TestClient) -> None:
        response = client.get("/cameras/overview/frame.jpg")
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/jpeg"
        assert response.content[:2] == b"\xff\xd8"

    def test_snapshot_unknown_camera(self, client: TestClient) -> None:
        assert client.get("/cameras/nope/frame.jpg").status_code == 404

    def test_snapshot_before_first_frame(self, app_context: dict) -> None:
        app_context["buffers"]["wrist"] = FrameBuffer("wrist")
        client = TestClient(app_context["app"])
        assert client.get("/cameras/wrist/frame.jpg").status_code == 503

    def test_mjpeg_unknown_camera(self, client: TestClient) -> None:
        assert client.get("/cameras/nope/mjpeg").status_code == 404

    def test_scenes(self, client: TestClient) -> None:
        response = client.get("/scenes")
        assert response.status_code == 200
        assert response.json() == {
            "current": "single_pick_place",
            "available": ["pick_lift", "single_pick_place"],
        }

    def test_switch_scene_enqueues_command(self, client: TestClient, app_context: dict) -> None:
        response = client.post("/scenes/pick_lift")
        assert response.status_code == 200
        assert response.json() == {"status": "queued", "scene": "pick_lift"}
        command = app_context["commands"].get_nowait()
        assert command == SwitchSceneCommand(scene_id="pick_lift")

    def test_switch_scene_unknown(self, client: TestClient) -> None:
        assert client.post("/scenes/nope").status_code == 404

    def test_reset_enqueues_command(self, client: TestClient, app_context: dict) -> None:
        response = client.post("/reset")
        assert response.status_code == 200
        assert response.json() == {"status": "queued"}
        assert app_context["commands"].get_nowait() == ResetCommand()

    def test_shutdown_enqueues_command(self, client: TestClient, app_context: dict) -> None:
        response = client.post("/shutdown")
        assert response.status_code == 200
        assert response.json() == {"status": "shutting_down"}
        assert app_context["commands"].get_nowait() == ShutdownCommand()


class TestMjpegGenerator:
    @pytest.mark.anyio
    async def test_yields_boundary_and_jpeg(self, frame: np.ndarray) -> None:
        buffer = FrameBuffer("overview")
        buffer.put(frame)
        stream = _mjpeg_stream(buffer, quality=85)
        chunk = await anext(stream)
        await stream.aclose()
        assert chunk.startswith(b"--mujoco-frame\r\n")
        assert b"Content-Type: image/jpeg" in chunk
        assert b"\xff\xd8" in chunk
        assert chunk.endswith(b"\r\n")

    @pytest.mark.anyio
    async def test_waits_for_first_frame(self, frame: np.ndarray) -> None:
        buffer = FrameBuffer("overview")
        stream = _mjpeg_stream(buffer, quality=85)
        iterator = anext(stream)

        async def publish_later() -> None:
            await asyncio.sleep(0.05)
            buffer.put(frame)

        chunk, _ = await asyncio.gather(iterator, publish_later())
        await stream.aclose()
        assert b"\xff\xd8" in chunk

    @pytest.mark.anyio
    async def test_streams_do_not_borrow_the_default_executor(self, frame: np.ndarray) -> None:
        buffer = FrameBuffer("overview")
        buffer.put(frame)
        stream = _mjpeg_stream(buffer, quality=85)

        with patch("asyncio.to_thread", side_effect=AssertionError("stream used a pool thread")):
            chunk = await anext(stream)
            await stream.aclose()

        assert b"\xff\xd8" in chunk

    @pytest.mark.anyio
    async def test_waiter_is_never_held_across_a_yield(self, frame: np.ndarray) -> None:
        buffer = FrameBuffer("overview")
        buffer.put(frame)
        stream = _mjpeg_stream(buffer, quality=85)
        await anext(stream)

        # Nothing stays registered while the client consumes a chunk, so an
        # abandoned generator cannot leak a waiter into the buffer.
        assert buffer._async_waiters == set()  # noqa: SLF001

        await stream.aclose()

        assert buffer._async_waiters == set()  # noqa: SLF001

    @pytest.mark.anyio
    async def test_publish_from_another_thread_wakes_the_stream(self, frame: np.ndarray) -> None:
        buffer = FrameBuffer("overview")
        stream = _mjpeg_stream(buffer, quality=85)
        iterator = anext(stream)

        def publish_from_sim_thread() -> None:
            time.sleep(0.05)
            buffer.put(frame)

        chunk, _ = await asyncio.gather(
            iterator,
            asyncio.get_running_loop().run_in_executor(None, publish_from_sim_thread),
        )
        await stream.aclose()
        assert b"\xff\xd8" in chunk


class TestFrameBufferAsyncWaiter:
    @pytest.mark.anyio
    async def test_waiter_is_registered_and_removed(self) -> None:
        buffer = FrameBuffer("cam")
        with buffer.async_waiter() as event:
            assert not event.is_set()
            assert len(buffer._async_waiters) == 1  # noqa: SLF001
        assert buffer._async_waiters == set()  # noqa: SLF001

    @pytest.mark.anyio
    async def test_put_sets_every_waiter(self, frame: np.ndarray) -> None:
        buffer = FrameBuffer("cam")
        with buffer.async_waiter() as first, buffer.async_waiter() as second:
            buffer.put(frame)
            await asyncio.sleep(0)
            assert first.is_set()
            assert second.is_set()

    @pytest.mark.anyio
    async def test_put_without_waiters_is_fine(self, frame: np.ndarray) -> None:
        buffer = FrameBuffer("cam")
        buffer.put(frame)
        assert buffer.snapshot() is not None


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class TestHttpServerLifecycle:
    def test_start_serve_stop(self, app_context: dict) -> None:
        port = _free_port()
        server = HttpServer(app_context["app"], "127.0.0.1", port)
        server.start()
        try:
            assert server.url == f"http://127.0.0.1:{port}"
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as response:
                assert json.loads(response.read())["connected"] is True
        finally:
            server.stop()

    def test_stop_closes_socket(self, app_context: dict) -> None:
        port = _free_port()
        server = HttpServer(app_context["app"], "127.0.0.1", port)
        server.start()
        server.stop()
        with pytest.raises(urllib.error.URLError):
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)

    def test_start_fails_on_used_port(self, app_context: dict) -> None:
        port = _free_port()
        first = HttpServer(app_context["app"], "127.0.0.1", port)
        first.start()
        try:
            second = HttpServer(app_context["app"], "127.0.0.1", port)
            with pytest.raises(RuntimeError, match="not available"):
                second.start()
        finally:
            first.stop()

    def test_mjpeg_over_http(self, app_context: dict) -> None:
        port = _free_port()
        server = HttpServer(app_context["app"], "127.0.0.1", port)
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        server.start()
        try:
            conn.request("GET", "/cameras/overview/mjpeg")
            response = conn.getresponse()
            assert response.status == 200
            assert "multipart/x-mixed-replace" in response.getheader("content-type", "")
            chunk = response.read1(65536)
        finally:
            conn.close()
            server.stop()
        assert b"--mujoco-frame" in chunk
        assert b"\xff\xd8" in chunk
