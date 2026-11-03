# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Teleoperation from a VR headset (Meta Quest) over WebXR.

The simulation serves a page over HTTPS on the local network (``start --xr-port``); the headset's
browser opens it and starts an immersive session, over the room where the headset has passthrough.
The page (``xr_page/``) draws the simulation in 3D, life-size: it fetches the geometry once and
then gets the bodies' poses over a WebSocket (:mod:`~physicalai_mujoco_plugin.xr_scene`), and it
shows the camera streams, fetched as JPEG frames, above the robot.

Every frame the page sends the headset's and the controllers' poses, their triggers and grip
buttons, and where it shows the robot: the point of the room where the operator's eye point
(``teleop.OperatorView``) lands, and the robot's heading there. :class:`XRServer` keeps the newest
:class:`XRSample`, and :func:`robot_controllers` turns its controllers into the robot's world frame
for ``teleop.py``, so a controller moved toward a part of the robot as the operator sees it moves
the gripper toward that part.

WebXR runs only on secure pages. The server makes a self-signed certificate with ``openssl`` on
first use, for this host's ``.local`` name and its local network addresses, and keeps it in the
Menagerie cache directory; the headset's browser asks once to accept it.

The certificate proves nothing about the headset, so the page pairs with a code: six digits made
on first use and kept beside the certificate, shown in the log and the viewer's **Teleop** folder.
The page asks for it once and keeps it; every API request and the WebSocket carry it, and the
WebSocket also refuses pages from another origin. One page at a time drives the arms: the first to
send controllers holds the session until it ends it, goes silent for ``XR_TIMEOUT_S`` or
disconnects; other pages watch. Anyone with the code on the network can drive the robot, so serve
the page only on a network you trust, as the robot's own transport.

The page's axes are WebXR's ``local-floor`` space: +y up, -z ahead of the operator at the start of
the session. :data:`XR_TO_ROBOT` turns them into the robot's: ahead is +x, left is +y.
"""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import importlib.resources as ir  # nosemgrep: python.lang.compatibility.python37.python37-compatibility-importlib2
import json
import math
import secrets
import shutil
import socket
import subprocess  # noqa: S404 - runs openssl and scutil with fixed arguments  # nosec B404
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated, Any

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from loguru import logger

from physicalai_mujoco_plugin.teleop import Controller

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path

    from physicalai_mujoco_plugin.http_server import FrameBuffer, HttpServer
    from physicalai_mujoco_plugin.teleop import OperatorView
    from physicalai_mujoco_plugin.xr_scene import SceneStream

XR_TO_ROBOT = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
"""WebXR axes (x right, y up, z back) to the robot's (x ahead, y left, z up): a proper rotation."""
CAMERA_JPEG_QUALITY = 75
XR_TIMEOUT_S = 0.5
"""A headset silent for longer has left its session."""
POSE_INTERVAL_S = 1 / 60
"""How often the page gets the bodies' poses, at most."""
SIDES = ("left", "right")
PAIRING_CODE_DIGITS = 6
_PAGE_FILES = {"app.js": "text/javascript"}


@dataclass(frozen=True)
class XRPose:
    """A pose in WebXR's ``local-floor`` space."""

    position: np.ndarray
    rotation: np.ndarray
    """3x3 rotation matrix."""


@dataclass(frozen=True)
class XRController:
    """One controller in WebXR's space: its grip pose, trigger and grip button (0 to 1)."""

    pose: XRPose
    trigger: float
    squeeze: float


@dataclass(frozen=True)
class XRSample:
    """The headset, the controllers and where the page shows the robot, in one WebXR frame."""

    t: float
    """``time.monotonic()`` when the frame arrived."""
    head: XRPose | None
    controllers: dict[str, XRController] = field(default_factory=dict)
    frame: XRPose | None = None
    """Where the page puts the operator's eye point, and the rotation about the vertical that turns
    WebXR's -z onto the robot's ahead there; ``None`` before the page places the robot."""
    session: int = 0
    """The WebSocket connection the sample came from: a new number is a new operator."""


def _rotation(wxyz: Any) -> np.ndarray | None:  # noqa: ANN401 - untrusted JSON
    try:
        w, x, y, z = (float(v) for v in wxyz)
    except (TypeError, ValueError):
        return None
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if not math.isfinite(norm) or norm < 1e-6:  # noqa: PLR2004
        return None
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def _pose(message: Any) -> XRPose | None:  # noqa: ANN401 - untrusted JSON
    if not isinstance(message, dict):
        return None
    try:
        position = np.asarray(message.get("p"), dtype=np.float64)
    except (TypeError, ValueError):
        return None
    rotation = _rotation(message.get("q"))
    if position.shape != (3,) or not np.all(np.isfinite(position)) or rotation is None:
        return None
    return XRPose(position, rotation)


def _unit(value: Any) -> float:  # noqa: ANN401 - untrusted JSON
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return min(max(number, 0.0), 1.0) if math.isfinite(number) else 0.0


def parse_frame(message: dict[str, Any], now: float) -> XRSample:
    """Return the sample of one page message: ``head``, ``left``, ``right`` and ``frame``.

    Returns:
        The sample; controllers without a valid pose are left out.
    """
    controllers = {}
    for side in SIDES:
        entry = message.get(side)
        pose = _pose(entry)
        if pose is not None:
            controllers[side] = XRController(pose, _unit(entry.get("trigger")), _unit(entry.get("squeeze")))  # type: ignore[union-attr]
    return XRSample(now, _pose(message.get("head")), controllers, _pose(message.get("frame")))


def xr_to_robot(frame: XRPose, view: OperatorView) -> tuple[np.ndarray, np.ndarray]:
    """Return the rotation and translation that turn the page's WebXR space into the robot's world frame.

    The page shows the robot so that the operator's eye point ``view.eye`` lands at
    ``frame.position``, with the robot's heading ``view.yaw`` along ``frame.rotation``'s -z.

    Returns:
        ``(rotation, translation)``: a WebXR point ``p`` is ``rotation @ p + translation`` in the world.
    """
    c, s = math.cos(view.yaw), math.sin(view.yaw)
    heading = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    rotation = heading @ XR_TO_ROBOT @ frame.rotation.T
    return rotation, np.asarray(view.eye, dtype=np.float64) - rotation @ frame.position


def robot_controllers(sample: XRSample, view: OperatorView) -> dict[str, Controller]:
    """Return the sample's controllers in the robot's world frame, by side.

    Returns:
        The controllers; none before the page places the robot.
    """
    if sample.frame is None:
        return {}
    rotation, translation = xr_to_robot(sample.frame, view)
    return {
        side: Controller(
            rotation @ controller.pose.position + translation,
            rotation @ controller.pose.rotation,
            controller.trigger,
            controller.squeeze,
        )
        for side, controller in sample.controllers.items()
    }


def _local_host_name() -> str:
    """Return the name this host answers to as ``<name>.local`` (Bonjour's on macOS, which can differ)."""
    scutil = shutil.which("scutil")
    if scutil is not None:
        result = subprocess.run([scutil, "--get", "LocalHostName"], capture_output=True, text=True, check=False)  # noqa: S603  # nosec B603
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    return socket.gethostname().split(".")[0]


def lan_names() -> list[str]:
    """Return the names the headset may reach this host by: its ``.local`` name and its LAN address.

    Returns:
        Host names and IP addresses, best first.
    """
    names = []
    host = _local_host_name()
    if host:
        names.append(f"{host}.local")
    with contextlib.suppress(OSError), socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.connect(("192.0.2.1", 9))  # no packet is sent: this only picks the outgoing interface
        address = probe.getsockname()[0]
        if not address.startswith("127."):
            names.append(address)
    return names


def certificate(directory: Path, names: list[str]) -> tuple[str, str]:
    """Return a self-signed certificate and key for *names*, made with ``openssl`` when missing or stale.

    Returns:
        The certificate and key files.

    Raises:
        RuntimeError: If ``openssl`` is missing or fails.
    """
    directory.mkdir(parents=True, exist_ok=True)
    cert, key, record = directory / "xr.crt", directory / "xr.key", directory / "xr.names"
    wanted = ",".join([*names, "localhost", "127.0.0.1"])
    if cert.is_file() and key.is_file() and record.is_file() and record.read_text() == wanted:
        return str(cert), str(key)
    openssl = shutil.which("openssl")
    if openssl is None:
        msg = "Teleoperation from a headset needs openssl to make its HTTPS certificate"
        raise RuntimeError(msg)
    alt = ",".join(f"IP:{name}" if name.replace(".", "").isdigit() else f"DNS:{name}" for name in wanted.split(","))
    command = [
        openssl, "req", "-x509", "-newkey", "ec", "-pkeyopt", "ec_paramgen_curve:prime256v1", "-nodes",
        "-days", "825", "-subj", "/CN=physicalai-mujoco", "-addext", f"subjectAltName={alt}",
        "-keyout", str(key), "-out", str(cert),
    ]  # fmt: skip
    result = subprocess.run(command, capture_output=True, text=True, check=False)  # noqa: S603  # nosec B603
    if result.returncode != 0:
        msg = f"openssl could not make the headset's HTTPS certificate: {result.stderr.strip()}"
        raise RuntimeError(msg)
    key.chmod(0o600)
    record.write_text(wanted)
    return str(cert), str(key)


def pairing_code(directory: Path) -> str:
    """Return this machine's pairing code for the headset page: six digits, made once and kept.

    Returns:
        The code.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "xr.code"
    code = path.read_text().strip() if path.is_file() else ""
    if len(code) != PAIRING_CODE_DIGITS or not code.isdigit():
        code = f"{secrets.randbelow(10**PAIRING_CODE_DIGITS):0{PAIRING_CODE_DIGITS}d}"
        path.write_text(code)
        path.chmod(0o600)
    return code


def _host(origin: str) -> str:
    """Return an Origin header's ``host:port``, as the Host header gives it."""
    return origin.split("://", 1)[-1].rstrip("/")


def _page_file(name: str) -> str:
    return ir.files("physicalai_mujoco_plugin").joinpath("xr_page", name).read_text(encoding="utf-8")


class XRServer:
    """The HTTPS page and WebSocket a headset teleoperates through, and its newest sample."""

    def __init__(
        self,
        host: str,
        port: int,
        cameras: Callable[[], Mapping[str, FrameBuffer]] | None = None,
        scene: SceneStream | None = None,
    ) -> None:
        """Configure the server on *host*:*port* (not yet started).

        Args:
            host: The address to listen on.
            port: The HTTPS port.
            cameras: Returns the camera streams to show.
            scene: The simulation's geometry and poses to draw.
        """
        self.host = host
        self.port = port
        self._cameras = cameras or dict
        self._scene = scene
        self.names = lan_names()
        self._sample: XRSample | None = None
        self._owner: int | None = None
        """The connection that drives the arms; ``None`` while none does."""
        self._connections = 0
        self.code: str | None = None
        """The pairing code, once :meth:`start` read it."""
        self._grabs: str | None = None
        """The newest grab state for the page, as its JSON message."""
        self._clients = 0
        self._lock = threading.Lock()
        self._server: HttpServer | None = None

    @property
    def urls(self) -> list[str]:
        """Addresses to open on the headset."""
        return [f"https://{name}:{self.port}/" for name in self.names]

    def check_code(self, code: str | None) -> bool:
        """Return whether *code* is the pairing code (compared in constant time)."""
        return self.code is not None and code is not None and hmac.compare_digest(code, self.code)

    def latest(self, max_age: float = XR_TIMEOUT_S) -> XRSample | None:
        """Return the newest sample, or ``None`` if it is older than *max_age* seconds."""
        sample = self._sample
        if sample is None or time.monotonic() - sample.t > max_age:
            return None
        return sample

    def status(self) -> dict[str, Any]:
        """Return the addresses, connections and the controllers seen, for the panel and ``GET /health``.

        Returns:
            A JSON-friendly snapshot.
        """
        sample = self.latest()
        return {
            "urls": self.urls,
            "code": self.code,
            "connected": self._clients > 0,
            "session": sample is not None,
            "controllers": sorted(sample.controllers) if sample is not None else [],
        }

    def start(self) -> None:
        """Make the certificate and start serving."""
        import mujoco_menagerie  # noqa: PLC0415

        from physicalai_mujoco_plugin.http_server import HttpServer  # noqa: PLC0415

        directory = mujoco_menagerie.Cache().dir / "xr"
        tls = certificate(directory, self.names)
        self.code = pairing_code(directory)
        self._server = HttpServer(self._app(), self.host, self.port, tls=tls, name="mujoco-xr-server")
        self._server.start()
        logger.info(
            "Headset teleoperation: open {} in the headset's browser; pairing code {}",
            " or ".join(self.urls),
            self.code,
        )

    def stop(self) -> None:
        """Stop serving."""
        if self._server is not None:
            self._server.stop()
            self._server = None

    def set_grab_state(self, state: Mapping[str, Mapping[str, int]]) -> None:
        """Tell the page which grab points the controllers hover and hold (``Teleop.grab_state``)."""
        self._grabs = json.dumps({"type": "grabs", **state}, sort_keys=True)

    def receive(self, message: Any, connection: int = 0) -> None:  # noqa: ANN401 - untrusted JSON
        """Keep a page message's sample, or end the session, if *connection* drives the arms or may take over.

        A connection takes over when none drives the arms, or the one that does went silent for
        ``XR_TIMEOUT_S``; the messages of the others are ignored.
        """
        if not isinstance(message, dict) or message.get("type") not in {"frame", "end"}:
            return
        with self._lock:
            if self._owner not in {None, connection} and self.latest() is not None:
                return  # another page drives the arms
            if message["type"] == "end":
                self._owner, self._sample = None, None
                return
            sample = parse_frame(message, time.monotonic())
            self._owner = connection
            self._sample = XRSample(sample.t, sample.head, sample.controllers, sample.frame, connection)

    def _drop(self, connection: int) -> None:
        with self._lock:
            self._clients -= 1
            if self._owner == connection:
                self._owner, self._sample = None, None

    def _app(self) -> FastAPI:
        # FastAPI reads the endpoints' annotations: WebSocket is imported at module level for it.
        app = FastAPI(title="MuJoCo headset teleoperation")

        @app.get("/", response_class=HTMLResponse)
        def page() -> str:
            return _page_file("index.html")

        @app.get("/{name}")
        def page_file(name: str) -> Response:
            if name not in _PAGE_FILES:
                raise HTTPException(status_code=404, detail=f"No {name!r}")
            return Response(_page_file(name), media_type=_PAGE_FILES[name])

        self._add_api_routes(app)
        self._add_websocket(app)
        return app

    def _add_api_routes(self, app: FastAPI) -> None:
        def paired(code: str | None) -> None:
            if not self.check_code(code):
                raise HTTPException(status_code=403, detail="Wrong or missing pairing code")

        @app.get("/api/check")
        def check(code: Annotated[str | None, Query()] = None) -> dict[str, bool]:
            paired(code)
            return {"paired": True}

        @app.get("/api/cameras")
        def cameras(code: Annotated[str | None, Query()] = None) -> list[str]:
            paired(code)
            return sorted(self._cameras(), key=lambda name: (name != "overview", name))

        @app.get("/api/camera/{name}.jpg")
        def camera(name: str, code: Annotated[str | None, Query()] = None) -> Response:
            from physicalai_mujoco_plugin.http_server import encode_jpeg  # noqa: PLC0415

            paired(code)
            buffer = self._cameras().get(name)
            sample = buffer.snapshot() if buffer is not None else None
            if sample is None:
                raise HTTPException(status_code=404, detail=f"No frame from camera {name!r}")
            return Response(
                encode_jpeg(sample.frame, CAMERA_JPEG_QUALITY),
                media_type="image/jpeg",
                headers={"Cache-Control": "no-store"},
            )

        @app.get("/api/scene")
        def scene(code: Annotated[str | None, Query()] = None) -> Response:
            paired(code)
            blob = self._scene.blob() if self._scene is not None else None
            if blob is None:
                raise HTTPException(status_code=404, detail="No scene yet")
            headers = {"Content-Encoding": "gzip", "Cache-Control": "no-store"}
            return Response(blob, media_type="application/octet-stream", headers=headers)

    def _add_websocket(self, app: FastAPI) -> None:
        @app.websocket("/ws")
        async def stream(websocket: WebSocket, code: str | None = Query(default=None)) -> None:
            origin = websocket.headers.get("origin")
            if not self.check_code(code) or (origin is not None and _host(origin) != websocket.headers.get("host")):
                await websocket.close(code=1008)  # policy violation: unpaired, or a page from another site
                return
            await websocket.accept()
            with self._lock:
                self._clients += 1
                self._connections += 1
                connection = self._connections
            pusher = asyncio.create_task(self._push_poses(websocket))
            try:
                while True:
                    with contextlib.suppress(ValueError):
                        self.receive(json.loads(await websocket.receive_text()), connection)
            except WebSocketDisconnect:
                pass
            finally:
                pusher.cancel()
                self._drop(connection)

    async def _push_poses(self, websocket: WebSocket) -> None:
        """Send the bodies' poses whenever the simulation has new ones, and the grab state when it changes."""
        sent, grabs = None, None
        with contextlib.suppress(WebSocketDisconnect, RuntimeError):
            while True:
                poses = self._scene.poses() if self._scene is not None else None
                if poses is not None and poses is not sent:
                    await websocket.send_bytes(poses)
                    sent = poses
                if self._grabs is not None and self._grabs != grabs:
                    grabs = self._grabs
                    await websocket.send_text(grabs)
                await asyncio.sleep(POSE_INTERVAL_S)


__all__ = [
    "XR_TIMEOUT_S",
    "XR_TO_ROBOT",
    "XRController",
    "XRPose",
    "XRSample",
    "XRServer",
    "certificate",
    "lan_names",
    "pairing_code",
    "parse_frame",
    "robot_controllers",
    "xr_to_robot",
]
