from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from physicalai.robot import Robot

from physicalai_franka_plugin.gello import BimanualGelloLeader, GelloCalibration, GelloLeader
from physicalai_franka_plugin.gello_calibrate import calculate_calibration


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

    def write1ByteTxRx(self, port: _FakePortHandler, motor_id: int, address: int, value: int) -> tuple[int, int]:  # noqa: N802
        _ = port, motor_id, address, value
        return 0, 0

    def read4ByteTxRx(self, port: _FakePortHandler, motor_id: int, address: int) -> tuple[int, int, int]:  # noqa: N802
        _ = port, address
        return 1000 + motor_id, 0, 0


@pytest.fixture
def fake_dynamixel() -> object:
    module = SimpleNamespace(PortHandler=_FakePortHandler, PacketHandler=lambda protocol: _FakePacketHandler())
    with patch.dict(sys.modules, {"dynamixel_sdk": module}):
        yield module


def test_default_calibration_matches_fr3_gello() -> None:
    calibration = GelloCalibration.default()
    assert calibration.joint_signs == (1, -1, 1, -1, 1, 1, 1)
    assert calibration.gripper_range_rad == (2.0, 3.22)


def test_calibration_parses_advanced_json_string() -> None:
    calibration = GelloCalibration.from_json(
        '{"joint_signs":[1,-1,1,-1,1,1,1],"assembly_offsets_rad":[0,0,3.142,3.142,3.142,4.712,0],"gripper_range_rad":[2,3.22]}'
    )
    assert calibration.joint_signs == (1, -1, 1, -1, 1, 1, 1)


def test_calibration_rounds_offsets_to_motor_flange_increments() -> None:
    calibration = calculate_calibration(
        np.array([-1.57, 0.77, 4.94, 0.14, 4.54, 6.21, -2.10]),
        reference_pose_rad=(-1.57, -0.80, 1.80, -3.00, 1.40, 1.50, -2.10),
        gripper_closed_rad=2.0,
        gripper_open_rad=3.22,
        baudrate=57_600,
    )
    assert calibration.joint_signs == (1, -1, 1, -1, 1, 1, 1)
    assert all(0.0 <= value < 2 * np.pi for value in calibration.assembly_offsets_rad)
    assert calibration.gripper_range_rad == (2.0, 3.22)


def test_bimanual_gello_is_read_only_robot(fake_dynamixel: object) -> None:
    _ = fake_dynamixel
    leader = BimanualGelloLeader("/dev/ttyUSB0", "/dev/ttyUSB1", smoothing=1.0)
    assert isinstance(leader, Robot)
    assert leader.device_ids == ("serial:/dev/ttyUSB0", "serial:/dev/ttyUSB1")
    leader.connect()
    observation = leader.get_observation()
    assert observation.joint_positions.shape == (16,)
    with pytest.raises(RuntimeError, match="read-only"):
        leader.send_action(np.zeros(14))
    leader.disconnect()
    assert not leader.is_connected()


def test_bimanual_gello_requires_distinct_ports() -> None:
    with pytest.raises(ValueError, match="distinct"):
        BimanualGelloLeader(
            "/dev/ttyUSB0",
            "/dev/ttyUSB0",
        )


def test_single_gello_uses_default_calibration(fake_dynamixel: object) -> None:
    _ = fake_dynamixel
    leader = GelloLeader("/dev/ttyUSB0")
    leader.connect()
    assert leader.get_observation().joint_positions.shape == (8,)
    leader.disconnect()
