# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""The headset's page and server: its messages, the robot's frame in the room, and the 3D scene it draws."""

# MuJoCo model/data attributes are supplied by the C extension at runtime.
# pyrefly: ignore-errors [missing-attribute]

from __future__ import annotations

import asyncio
import gzip
import json
import math
import re
import shutil
import socket
import ssl
import subprocess  # noqa: S404 - runs node on the page's script
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np
import pytest

from physicalai_mujoco_plugin import xr
from physicalai_mujoco_plugin.http_server import FrameBuffer
from physicalai_mujoco_plugin.teleop import OperatorView
from physicalai_mujoco_plugin.xr import XR_TO_ROBOT, XRPose, XRServer, parse_frame, robot_controllers, xr_to_robot
from physicalai_mujoco_plugin.xr_scene import VERTEX, SceneStream, scene_blob


def _yaw(degrees: float) -> np.ndarray:
    """A turn about WebXR's vertical (+y): positive turns -z (ahead) toward -x (left)."""
    c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _yaw_quat(degrees: float) -> list[float]:
    half = math.radians(degrees) / 2
    return [math.cos(half), 0.0, math.sin(half), 0.0]


def test_webxr_ahead_left_and_up_are_the_robots_x_y_and_z() -> None:
    np.testing.assert_allclose(XR_TO_ROBOT @ (0.0, 0.0, -1.0), (1.0, 0.0, 0.0))
    np.testing.assert_allclose(XR_TO_ROBOT @ (-1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    np.testing.assert_allclose(XR_TO_ROBOT @ (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    assert np.linalg.det(XR_TO_ROBOT) == pytest.approx(1.0)


def test_a_frame_parses_poses_buttons_and_where_the_robot_is() -> None:
    message = {
        "type": "frame",
        "head": {"p": [0.0, 1.6, 0.0], "q": _yaw_quat(90.0)},
        "left": {"p": [-0.2, 1.1, -0.3], "q": [1.0, 0.0, 0.0, 0.0], "trigger": 0.25, "squeeze": 2.0},
        "right": {"trigger": 1.0},
        "frame": {"p": [0.0, 1.0, -0.5], "q": _yaw_quat(30.0)},
    }
    sample = parse_frame(message, 5.0)
    assert sample.t == 5.0
    assert list(sample.controllers) == ["left"]  # the right one has no pose
    left = sample.controllers["left"]
    assert (left.trigger, left.squeeze) == (0.25, 1.0)
    np.testing.assert_allclose(sample.head.rotation, _yaw(90.0), atol=1e-12)  # type: ignore[union-attr]
    np.testing.assert_allclose(sample.frame.rotation, _yaw(30.0), atol=1e-12)  # type: ignore[union-attr]


@pytest.mark.parametrize(
    "entry",
    [
        {"p": [0, 1, "x"], "q": [1, 0, 0, 0]},
        {"p": [0, 1, float("nan")], "q": [1, 0, 0, 0]},
        {"p": [0, 1], "q": [1, 0, 0, 0]},
        {"p": [0, 1, 0], "q": [0, 0, 0, 0]},
        {"p": [0, 1, 0], "q": [1, 0]},
        {"p": [0, 1, 0], "q": None},
        "nonsense",
    ],
)
def test_a_malformed_pose_is_left_out_instead_of_failing(entry: object) -> None:
    sample = parse_frame({"type": "frame", "left": entry, "frame": entry}, 0.0)
    assert sample.controllers == {}
    assert sample.frame is None


def test_the_robot_frame_puts_the_eye_point_where_the_page_shows_it() -> None:
    view = OperatorView(eye=(0.1, -0.2, 0.6), yaw=math.radians(40.0))
    frame = XRPose(np.array([0.3, 1.5, -0.7]), _yaw(25.0))
    rotation, translation = xr_to_robot(frame, view)
    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(rotation @ frame.position + translation, view.eye, atol=1e-12)
    # Ahead on the page (WebXR's -z turned by the frame) is the robot's heading in the world.
    ahead = frame.rotation @ (0.0, 0.0, -1.0)
    np.testing.assert_allclose(rotation @ ahead, (math.cos(view.yaw), math.sin(view.yaw), 0.0), atol=1e-12)
    np.testing.assert_allclose(rotation @ (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), atol=1e-12)


def test_controllers_are_in_the_robots_frame_only_once_the_page_shows_the_robot() -> None:
    view = OperatorView(eye=(0.0, 0.0, 0.6), yaw=0.0)
    message = {"type": "frame", "left": {"p": [0.0, 1.0, -0.3], "q": [1, 0, 0, 0], "squeeze": 1.0}}
    assert robot_controllers(parse_frame(message, 0.0), view) == {}
    message["frame"] = {"p": [0.0, 1.5, 0.0], "q": [1, 0, 0, 0]}
    left = robot_controllers(parse_frame(message, 0.0), view)["left"]
    np.testing.assert_allclose(left.position, (0.3, 0.0, 0.1), atol=1e-12)  # 0.3 m ahead of the eye, 0.5 m below it
    assert left.squeeze == 1.0


_PRIMITIVES = """<mujoco>
<asset><material name="window_glass" rgba="1 1 1 0.2"/><material name="collision" rgba="1 0 0 0.5"/></asset>
<worldbody>
  <geom name="floor" type="plane" size="0 0 0.1"/>
  <geom name="hidden" type="box" size="0.1 0.1 0.1" group="3"/>
  <geom name="glass" type="box" size="0.1 0.1 0.1" group="3" material="window_glass"/>
  <geom name="fixed_collision" type="box" size="0.1 0.1 0.1" group="3" material="collision"/>
  <body name="link" pos="1 0 0">
    <freejoint/>
    <geom name="collision" type="box" size="0.1 0.1 0.1" group="3" material="collision"/>
    <geom type="box" size="0.1 0.2 0.3" pos="0 0 1" rgba="1 0 0 1"/>
    <geom type="sphere" size="0.1"/>
    <geom type="capsule" size="0.05 0.2"/>
    <geom type="cylinder" size="0.05 0.2"/>
    <geom type="ellipsoid" size="0.1 0.2 0.3"/>
  </body>
</worldbody></mujoco>"""
_VIEW = OperatorView(eye=(0.0, 0.0, 0.5), yaw=0.0)


def _decode(blob: bytes) -> tuple[dict, np.ndarray, np.ndarray]:
    length = int(np.frombuffer(blob[:4], np.uint32)[0])
    header = json.loads(blob[4 : 4 + length])
    first = 4 + length
    vertices = np.frombuffer(blob[first : first + 16 * header["vertices"]], dtype=VERTEX)
    indices = np.frombuffer(blob[first + 16 * header["vertices"] :], dtype=np.uint32)
    return header, vertices, indices


def test_the_scene_holds_the_shown_geoms_in_their_bodies_frames() -> None:
    model = mujoco.MjModel.from_xml_string(_PRIMITIVES)
    header, vertices, indices = _decode(scene_blob(model, _VIEW, 3))
    assert (header["version"], header["bodies"]) == (3, 2)
    assert header["view"] == {"eye": [0.0, 0.0, 0.5], "yaw": 0.0}
    bodies = [geom[0] for geom in header["geoms"]]
    assert bodies == [0, 1, 1, 1, 1, 1]  # the glass, then the link's five; not the floor, the hidden or collision boxes
    assert header["geoms"][0][4] == pytest.approx(0.2)
    assert len(indices) == header["indices"] == sum(geom[6] for geom in header["geoms"])
    assert indices.max() < len(vertices)
    first, count = header["geoms"][1][5:]
    box = vertices["position"][indices[first : first + count]]
    np.testing.assert_allclose(box.min(axis=0), (-0.1, -0.2, 0.7), atol=1e-6)  # in the body's frame, at the geom's place
    np.testing.assert_allclose(box.max(axis=0), (0.1, 0.2, 1.3), atol=1e-6)
    normals = vertices["normal"][:, :3].astype(float) / 127
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0, atol=0.02)
    first, count = header["geoms"][3][5:]  # the capsule: 0.05 + 0.2 tall on each side
    extent = np.ptp(vertices["position"][indices[first : first + count]], axis=0)
    np.testing.assert_allclose(extent, (0.1, 0.1, 0.5), atol=1e-6)


def test_the_stream_versions_the_model_and_the_view_and_carries_the_poses() -> None:
    model = mujoco.MjModel.from_xml_string(_PRIMITIVES)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    stream = SceneStream()
    assert stream.blob() is None
    stream.update(model, data, _VIEW)
    poses = np.frombuffer(stream.poses(), dtype=np.float32)  # type: ignore[arg-type]
    assert poses[0] == 1
    np.testing.assert_allclose(poses[1:].reshape(-1, 7)[1], (1, 0, 0, 1, 0, 0, 0))
    assert _decode(gzip.decompress(stream.blob()))[0]["version"] == 1  # type: ignore[arg-type]
    assert stream.blob() is stream.blob()  # built once
    stream.update(model, data, _VIEW)
    assert np.frombuffer(stream.poses(), dtype=np.float32)[0] == 1  # type: ignore[arg-type]
    stream.update(mujoco.MjModel.from_xml_string(_PRIMITIVES), data, _VIEW)  # a scene switch brings a new model
    assert np.frombuffer(stream.poses(), dtype=np.float32)[0] == 2  # type: ignore[arg-type]
    stream.update(model, data, OperatorView(eye=(0.0, 0.0, 0.6), yaw=0.0))
    assert _decode(gzip.decompress(stream.blob()))[0]["version"] == 3  # type: ignore[arg-type]


@pytest.mark.skipif(shutil.which("openssl") is None, reason="needs openssl for the certificate")
def test_the_page_its_files_and_its_websocket_run_over_https(tmp_path: Path) -> None:
    websockets = pytest.importorskip("websockets")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    overview = FrameBuffer("overview")
    overview.put(np.full((48, 64, 3), 200, dtype=np.uint8))
    buffers = {"left_wrist": FrameBuffer("left_wrist"), "overview": overview}
    model = mujoco.MjModel.from_xml_string(_PRIMITIVES)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    scene = SceneStream()
    scene.update(model, data, _VIEW)
    server = XRServer("127.0.0.1", port, cameras=lambda: buffers, scene=scene)
    with patch.dict("os.environ", {"MENAGERIE_CACHE_DIR": str(tmp_path)}):
        server.start()
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    base = f"https://127.0.0.1:{port}"

    code = server.code
    assert code is not None
    assert len(code) == 6
    assert code.isdigit()
    assert (tmp_path / "xr" / "xr.code").read_text() == code  # kept, so the headset pairs once

    def get(path: str, paired: bool = True) -> bytes:  # noqa: FBT001, FBT002
        query = ("&" if "?" in path else "?") + f"code={code}" if paired else ""
        return urllib.request.urlopen(base + path + query, context=context).read()  # noqa: S310

    try:
        assert b'src="app.js"' in get("/", paired=False)
        assert b"immersive-ar" in get("/app.js", paired=False)
        with pytest.raises(urllib.error.HTTPError, match="404"):
            get("/secret.txt")
        for path in ("/api/check", "/api/cameras", "/api/camera/overview.jpg", "/api/scene", "/api/check?code=000000x"):
            with pytest.raises(urllib.error.HTTPError, match="403"):
                get(path, paired=False)  # nothing of the simulation without the pairing code
        assert json.loads(get("/api/check")) == {"paired": True}
        assert json.loads(get("/api/cameras")) == ["overview", "left_wrist"]  # the overview first: the large panel
        assert get("/api/camera/overview.jpg")[:2] == b"\xff\xd8"
        with pytest.raises(urllib.error.HTTPError, match="404"):
            get("/api/camera/left_wrist.jpg")  # no frame yet
        geometry = urllib.request.urlopen(f"{base}/api/scene?code={code}", context=context)  # noqa: S310
        assert geometry.headers["Content-Encoding"] == "gzip"
        assert _decode(gzip.decompress(geometry.read()))[0]["bodies"] == 2
        url = f"wss://127.0.0.1:{port}/ws"
        frame = {"type": "frame", "right": {"p": [0.2, 1.0, -0.3], "q": [1, 0, 0, 0], "squeeze": 1.0}}

        async def refused(address: str, **options: object) -> None:
            with pytest.raises(websockets.exceptions.WebSocketException):
                async with websockets.connect(address, ssl=context, **options) as connection:
                    await asyncio.wait_for(connection.recv(), timeout=2)

        async def stream() -> None:
            await refused(url)
            await refused(f"{url}?code=123")
            await refused(f"{url}?code={code}", origin="https://unrelated.invalid")
            async with websockets.connect(f"{url}?code={code}", ssl=context) as connection:
                poses = await asyncio.wait_for(connection.recv(), timeout=2)
                assert np.frombuffer(poses, dtype=np.float32).shape == (1 + 2 * 7,)
                frame = {"type": "frame", "right": {"p": [0.2, 1.0, -0.3], "q": [1, 0, 0, 0], "squeeze": 1.0}}
                await connection.send(json.dumps(frame))
                await asyncio.sleep(0.2)
                assert server.status()["controllers"] == ["right"]
                await connection.send("not json")
                await connection.send(json.dumps([1, 2]))
                await connection.send(json.dumps({"type": "end"}))
                await asyncio.sleep(0.2)
                assert server.latest() is None

        async def two_pages() -> None:
            async with (
                websockets.connect(f"{url}?code={code}", ssl=context, origin=f"https://127.0.0.1:{port}") as first,
                websockets.connect(f"{url}?code={code}", ssl=context) as second,
            ):
                await first.send(json.dumps(frame))
                await asyncio.sleep(0.2)
                owner = server.latest().session  # type: ignore[union-attr]
                await second.send(json.dumps({**frame, "right": {**frame["right"], "p": [9, 9, 9]}}))
                await second.send(json.dumps({"type": "end"}))
                await asyncio.sleep(0.2)
                sample = server.latest()
                assert sample is not None, "the other page cannot end the session"
                assert sample.session == owner
                np.testing.assert_allclose(sample.controllers["right"].pose.position, (0.2, 1.0, -0.3))
                await first.close()
                await asyncio.sleep(0.2)
                assert server.latest() is None, "the session ends with its page"
                await second.send(json.dumps(frame))
                await asyncio.sleep(0.2)
                assert server.latest().session != owner  # type: ignore[union-attr]  # the other page takes over

        asyncio.run(stream())
        asyncio.run(two_pages())
    finally:
        server.stop()
    assert (tmp_path / "xr" / "xr.crt").is_file()


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node to run the page's script")
def test_the_page_places_the_robot_where_the_server_expects_it(tmp_path: Path) -> None:
    """The page's ``placement`` (robot to room) and :func:`xr_to_robot` (room to robot) are inverses."""
    source = (Path(xr.__file__).parent / "xr_page" / "app.js").read_text()
    functions = "".join(re.search(rf"(function {name}\(.*?\n}}\n)", source, re.S).group(1) for name in ("anchor", "placement", "yaw"))  # type: ignore[union-attr]
    place = {"ax": math.sin(-0.4), "az": -math.cos(0.4), "hx": 0.3, "hy": 1.5, "hz": -0.2, "distance": 0.2, "height": -0.1}
    view = OperatorView(eye=(0.1, -0.2, 0.6), yaw=0.7)
    script = tmp_path / "placement.js"
    script.write_text(
        functions
        + f"const place = {json.dumps(place)}, view = {json.dumps({'eye': view.eye, 'yaw': view.yaw})};\n"
        + "console.log(JSON.stringify({m: Array.from(placement(place, view)), p: anchor(place), q: yaw(place)}));\n"
    )
    result = json.loads(subprocess.check_output([shutil.which("node"), str(script)]))  # type: ignore[list-item]  # noqa: S603
    room = np.array(result["m"]).reshape(4, 4).T  # column-major
    frame = parse_frame({"type": "frame", "frame": {"p": result["p"], "q": result["q"]}}, 0.0).frame
    rotation, translation = xr_to_robot(frame, view)  # type: ignore[arg-type]
    point = np.array([0.4, 0.1, 0.2])
    np.testing.assert_allclose(rotation @ (room[:3, :3] @ point + room[:3, 3]) + translation, point, atol=1e-6)


_PAGE_HARNESS = """
const vm = require("vm");
const fs = require("fs");
const element = () => ({ textContent: "", hidden: true, style: {}, value: "", clientWidth: 0, clientHeight: 0,
  getContext: () => null });
const elements = {};
const sockets = [];
let cameraList = ["overview", "left_wrist"];
const requests = [];
class FakeSocket { constructor(url) { this.url = url; this.readyState = 0; sockets.push(this); } send() {} }
FakeSocket.OPEN = 1;
const context = {
  console, URLSearchParams, JSON, Math, Float32Array, Uint8Array, Uint32Array, TextDecoder, Promise, Object, Set,
  setTimeout: () => 0, requestAnimationFrame: () => 0, performance: { now: () => 0 },
  location: { search: "?code=123456", host: "mac.local:8443" },
  localStorage: { getItem: () => null, setItem: () => {} },
  document: { getElementById: (id) => (elements[id] = elements[id] || element()) },
  navigator: {}, WebSocket: FakeSocket, URL: { revokeObjectURL: () => {}, createObjectURL: () => "" },
  fetch: async (path) => {
    requests.push(path);
    if (path.startsWith("api/cameras")) return { ok: true, json: async () => cameraList };
    if (path.startsWith("api/check")) return { ok: true };
    return { ok: false, status: 404 };
  },
};
vm.createContext(context);
vm.runInContext(fs.readFileSync(process.argv[2], "utf8") + `
globalThis.hook = {
  connect, bodyMatrices,
  setUploaded: (value) => { uploaded = value; },
  setPoses: (value) => { poses = value; },
  epoch: () => epoch, cameras: () => cameras,
};`, context);
(async () => {
  const flush = () => new Promise((resolve) => setImmediate(resolve));
  await flush(); await flush();
  const first = sockets.at(-1);
  first.onopen();
  await flush(); await flush();
  const place = { ax: 0, az: -1, hx: 0, hy: 1.6, hz: 0, distance: 0, height: 0 };
  const pose = new Float32Array([1, 0, 0, 0, 1, 0, 0, 0]);
  context.hook.setUploaded({ epoch: context.hook.epoch(), version: 1, view: { eye: [0, 0, 0], yaw: 0 } });
  context.hook.setPoses(pose);
  const before = context.hook.bodyMatrices(place) !== null;
  const camerasBefore = context.hook.cameras().slice();
  cameraList = ["overview", "wrist"];  // the simulation restarts with another robot
  first.onclose = null;
  context.hook.connect();
  sockets.at(-1).onopen();
  context.hook.setPoses(pose);  // its first poses carry the same version number
  const during = context.hook.bodyMatrices(place) !== null;
  await flush(); await flush();
  console.log(JSON.stringify({ before, during, camerasBefore, camerasAfter: context.hook.cameras(),
    socketUrl: first.url, listings: requests.filter((path) => path.startsWith("api/cameras")).length }));
})();
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node to run the page's script")
def test_the_page_forgets_the_old_simulation_when_it_reconnects(tmp_path: Path) -> None:
    """A simulation restarted on the same port may run another robot: no old meshes on new poses, new cameras."""
    harness = tmp_path / "harness.js"
    harness.write_text(_PAGE_HARNESS)
    page = Path(xr.__file__).parent / "xr_page" / "app.js"
    result = json.loads(subprocess.check_output([shutil.which("node"), str(harness), str(page)]))  # type: ignore[list-item]  # noqa: S603
    assert result["socketUrl"] == "wss://mac.local:8443/ws?code=123456"
    assert result["before"]
    assert not result["during"], "the old geometry is not drawn on the new robot's poses"
    assert result["camerasBefore"] == ["overview", "left_wrist"]
    assert result["camerasAfter"] == ["overview", "wrist"]
    assert result["listings"] == 2
