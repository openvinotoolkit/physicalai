from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from physicalai.robot import Robot

from physicalai_franka_plugin.gello import BimanualGelloLeader, GelloCalibration


class _FakePortHandler:
    def __init__(self, port: str) -> None:
        self.port = port
        self.closed = False

    def openPort(self) -> bool:  # noqa: N802 - mirrors Dynamixel SDK
        return True

    def setBaudRate(self, baudrate: int) -> bool:  # noqa: N802 - mirrors Dynamixel SDK
        _ = baudrate
        return True

    def closePort(self) -> None:  # noqa: N802 - mirrors Dynamixel SDK
        self.closed = True


class _FakePacketHandler:
    def ping(self, port: _FakePortHandler, motor_id: int) -> tuple[int, int, int]:
        _ = port, motor_id
        return 1200, 0, 0

    def write1ByteTxRx(self, port: _FakePortHandler, motor_id: int, address: int, value: int) -> tuple[int, int, int]:  # noqa: N802
        _ = port, motor_id, address, value
        return 0, 0, 0

    def read4ByteTxRx(self, port: _FakePortHandler, motor_id: int, address: int) -> tuple[int, int, int]:  # noqa: N802
        _ = port, address
        return 1000 + motor_id, 0, 0


@pytest.fixture
def calibration_path(tmp_path: Path) -> Path:
    path = tmp_path / "gello.json"
    path.write_text(
        json.dumps(
            {
                "joint_offsets": {f"joint_{index}": 1000 + index + 1 for index in range(6)},
                "gripper_open_position": 1007,
                "gripper_closed_position": 432,
            },
        ),
    )
    return path


@pytest.fixture
def fake_dynamixel() -> object:
    module = SimpleNamespace(PortHandler=_FakePortHandler, PacketHandler=lambda protocol: _FakePacketHandler())
    with patch.dict(sys.modules, {"dynamixel_sdk": module}):
        yield module


def test_calibration_loads_reference_plugin_dict_shape(calibration_path: Path) -> None:
    calibration = GelloCalibration.from_file(calibration_path)
    assert calibration.joint_offsets == (1001, 1002, 1003, 1004, 1005, 1006)
    assert calibration.gripper_closed_position == 432


def test_bimanual_gello_is_read_only_robot(calibration_path: Path, fake_dynamixel: object) -> None:
    _ = fake_dynamixel
    leader = BimanualGelloLeader(
        "/dev/ttyUSB0",
        "/dev/ttyUSB1",
        left_calibration_path=str(calibration_path),
        right_calibration_path=str(calibration_path),
        smoothing=1.0,
    )
    assert isinstance(leader, Robot)
    assert leader.device_ids == ("serial:/dev/ttyUSB0", "serial:/dev/ttyUSB1")
    leader.connect()
    observation = leader.get_observation()
    assert observation.joint_positions.shape == (14,)
    np.testing.assert_allclose(observation.joint_positions[:6], [0, -89.95, 89.95, -89.95, -89.95, -89.95], atol=0.1)
    assert observation.joint_positions[6] == pytest.approx(0.0)
    with pytest.raises(RuntimeError, match="read-only"):
        leader.send_action(np.zeros(14))
    leader.disconnect()
    assert not leader.is_connected()


def test_bimanual_gello_requires_distinct_ports(calibration_path: Path) -> None:
    with pytest.raises(ValueError, match="distinct"):
        BimanualGelloLeader(
            "/dev/ttyUSB0",
            "/dev/ttyUSB0",
            left_calibration_path=str(calibration_path),
            right_calibration_path=str(calibration_path),
        )
