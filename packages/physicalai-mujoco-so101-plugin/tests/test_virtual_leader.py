"""The virtual leader reads the simulation's ``GET /leader`` over real HTTP."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from physicalai_mujoco_so101_plugin.constants import SO101_JOINT_ORDER
from physicalai_mujoco_so101_plugin.virtual_leader import MuJoCoVirtualLeader

if TYPE_CHECKING:
    from collections.abc import Iterator


class _Leader:
    def __init__(self) -> None:
        self.payload: dict[str, Any] = {
            "seq": 1,
            "mode": "leader",
            "unit": "normalized",
            "joint_names": list(SO101_JOINT_ORDER),
            "joint_positions": [1.0, 2.0, 3.0, 4.0, 5.0, 60.0],
        }
        self.requests = 0


@pytest.fixture
def served() -> Iterator[tuple[_Leader, int]]:
    leader = _Leader()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"  # keep-alive, like uvicorn

        def do_GET(self) -> None:
            leader.requests += 1
            if self.path != "/leader":
                self.send_error(404)
                return
            body = json.dumps(leader.payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield leader, int(server.server_address[1])
    server.shutdown()
    server.server_close()


def test_reads_positions_and_advances_only_with_the_simulation(served: tuple[_Leader, int]) -> None:
    leader, port = served
    robot = MuJoCoVirtualLeader(http_port=port)
    robot.connect()
    try:
        first = robot.get_observation()
        np.testing.assert_allclose(first.joint_positions, [1, 2, 3, 4, 5, 60])
        assert first.joint_positions.dtype == np.float32
        assert robot.get_observation().timestamp == first.timestamp  # same tick: a stalled leader
        leader.payload = {**leader.payload, "seq": 2, "joint_positions": [0.0] * 6}
        second = robot.get_observation()
        assert second.timestamp > first.timestamp
        np.testing.assert_allclose(second.joint_positions, np.zeros(6))
        robot.send_action(np.ones(6))  # ignored
    finally:
        robot.disconnect()
    assert not robot.is_connected()


@pytest.mark.parametrize(
    "payload",
    [
        {"seq": 5, "joint_names": list(SO101_JOINT_ORDER)},  # no pose yet
        {"seq": 5, "joint_names": list(SO101_JOINT_ORDER), "joint_positions": ["a"] * 6},
        {"seq": 5, "joint_names": list(SO101_JOINT_ORDER), "joint_positions": [1.0, 2.0]},
    ],
    ids=["missing", "not-numbers", "wrong-length"],
)
def test_an_incomplete_pose_is_a_connection_error(served: tuple[_Leader, int], payload: dict[str, Any]) -> None:
    leader, port = served
    robot = MuJoCoVirtualLeader(http_port=port)
    robot.connect()
    try:
        leader.payload = payload
        with pytest.raises(ConnectionError, match="Leader pose"):
            robot.get_observation()
    finally:
        robot.disconnect()


def test_rejects_a_simulation_with_another_joint_layout(served: tuple[_Leader, int]) -> None:
    leader, port = served
    leader.payload = {**leader.payload, "joint_names": ["left_shoulder_pan"]}
    robot = MuJoCoVirtualLeader(http_port=port)
    with pytest.raises(ConnectionError, match="expected"):
        robot.connect()
    assert not robot.is_connected()


def test_unreachable_simulation_raises_connection_error() -> None:
    robot = MuJoCoVirtualLeader(http_port=1, timeout_s=0.5)
    with pytest.raises(ConnectionError, match="not answering"):
        robot.connect()
    with pytest.raises(ConnectionError, match="not connected"):
        robot.get_observation()


@pytest.mark.parametrize(("kwargs", "match"), [({"host": "10.0.0.2"}, "this machine"), ({"http_port": 0}, "1..65535")])
def test_only_reads_from_this_machine(kwargs: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        MuJoCoVirtualLeader(**kwargs)


def test_does_not_import_mujoco() -> None:
    # Studio builds the leader in its own environment, which may carry another MuJoCo.
    import ast  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    import physicalai_mujoco_so101_plugin.virtual_leader as module  # noqa: PLC0415

    tree = ast.parse(Path(module.__file__).read_text())
    imported = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    imported |= {node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert not any(name.startswith("mujoco") for name in imported)
