from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from physicalai.robot import Robot

from physicalai_franka_plugin.gello import (
    BimanualGelloLeader,
    GelloCalibration,
    GelloLeader,
    present_position_counts,
)
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


def test_present_position_is_signed_int32() -> None:
    assert present_position_counts(914) == 914
    assert present_position_counts((1 << 32) - 921) == -921


def test_initial_pose_maps_duo_reference_pose(fake_dynamixel: object) -> None:
    # Raw counts read from a real, pre-assembled Franka GELLO Duo right arm resting in its calibration
    # pose; J5 and J7 report negative positions. The factory calibration must map it to the documented
    # reference pose (upstream convention, gello_software ros2 README).
    counts = [3073, 2490, 2890, 1992, (1 << 32) - 921, 2076, (1 << 32) - 710, 2184]
    _ = fake_dynamixel
    with patch.object(
        _FakePacketHandler, "read4ByteTxRx", lambda self, port, motor_id, address: (counts[motor_id - 1], 0, 0)
    ):
        leader = GelloLeader("/dev/ttyUSB0")
        leader.connect()
        degrees = leader.get_observation().joint_positions[:7]
    reference = np.degrees([1.57, -0.80, -1.80, -3.00, -1.40, 1.50, 2.10])
    limits = np.degrees(GelloLeader._LIMITS_RAD)
    assert np.all(np.abs(degrees - np.clip(reference, limits[:, 0], limits[:, 1])) < 15.0)


def test_calibration_rejects_degenerate_trigger_range() -> None:
    with pytest.raises(ValueError, match="recalibrate the trigger"):
        GelloCalibration.from_json(
            '{"joint_signs":[1,-1,1,-1,1,1,1],"assembly_offsets_rad":[0,0,0,0,0,0,0],"gripper_range_rad":[3.3,3.3]}'
        )


def test_trigger_wraps_whole_turns_into_the_calibrated_window(fake_dynamixel: object) -> None:
    # Released trigger (3.30 rad) reported one full turn low, as multi-turn XL330 counts can be.
    counts = [2048] * 7 + [round((3.30 - 2 * np.pi) * 4095 / (2 * np.pi)) & 0xFFFFFFFF]
    _ = fake_dynamixel
    with patch.object(
        _FakePacketHandler, "read4ByteTxRx", lambda self, port, motor_id, address: (counts[motor_id - 1], 0, 0)
    ):
        leader = GelloLeader(
            "/dev/ttyUSB0",
            calibration=GelloCalibration(
                joint_signs=(1, -1, 1, -1, 1, 1, 1),
                assembly_offsets_rad=(0.0,) * 7,
                gripper_range_rad=(2.24, 3.32),
            ),
        )
        leader.connect()
        trigger = leader.get_observation().joint_positions[7]
    assert trigger > 0.95


def test_no_drift_after_passing_a_joint_limit(fake_dynamixel: object) -> None:
    # J4 rests beyond the FR3 limit (-157 deg) at the Duo rest pose (~ -175 deg). Moving past the limit
    # and back must return the same reading; the old clamped-delta integrator kept the clamp error.
    _ = fake_dynamixel
    rest = [2048, 2048, 2048, 2048, 2048, 2048, 2048, 2700]
    counts = list(rest)
    with patch.object(
        _FakePacketHandler, "read4ByteTxRx", lambda self, port, motor_id, address: (counts[motor_id - 1], 0, 0)
    ):
        leader = GelloLeader(
            "/dev/ttyUSB0",
            calibration=GelloCalibration(
                joint_signs=(1, -1, 1, -1, 1, 1, 1),
                assembly_offsets_rad=(np.pi, np.pi, np.pi, np.pi, np.pi, np.pi / 2, np.pi),
                gripper_range_rad=(2.24, 3.32),
            ),
        )
        counts[3] = round(1.4 * 4095 / (2 * np.pi))  # J4 = -1.4 rad, inside the range
        leader.connect()
        before = leader.get_observation().joint_positions[3]
        counts[3] = round(3.05 * 4095 / (2 * np.pi))  # J4 = -3.05 rad, past the -2.75 rad limit
        assert leader.get_observation().joint_positions[3] == pytest.approx(np.degrees(-2.7478), abs=0.1)
        counts[3] = round(1.4 * 4095 / (2 * np.pi))
        after = leader.get_observation().joint_positions[3]
    assert after == pytest.approx(before, abs=0.1)


class _FakeGroupSyncRead:
    def __init__(self, port: _FakePortHandler, packet: _FakePacketHandler, address: int, length: int) -> None:
        self._packet, self._port, self._address, self._ids = packet, port, address, []
        _ = length

    def addParam(self, motor_id: int) -> bool:  # noqa: N802 - mirrors Dynamixel SDK
        self._ids.append(motor_id)
        return True

    def txRxPacket(self) -> int:  # noqa: N802 - mirrors Dynamixel SDK
        self.data = {i: self._packet.read4ByteTxRx(self._port, i, self._address)[0] for i in self._ids}
        return 0

    def isAvailable(self, motor_id: int, address: int, length: int) -> bool:  # noqa: N802
        _ = address, length
        return motor_id in self.data

    def getData(self, motor_id: int, address: int, length: int) -> int:  # noqa: N802
        _ = address, length
        return self.data[motor_id]


def test_bimanual_reads_both_buses_with_group_sync_read() -> None:
    module = SimpleNamespace(
        PortHandler=_FakePortHandler,
        PacketHandler=lambda protocol: _FakePacketHandler(),
        GroupSyncRead=_FakeGroupSyncRead,
    )
    with patch.dict(sys.modules, {"dynamixel_sdk": module}):
        leader = BimanualGelloLeader("/dev/ttyUSB0", "/dev/ttyUSB1")
        leader.connect()
        assert leader._left._sync_read is not None
        assert leader.get_observation().joint_positions.shape == (16,)
        leader.disconnect()


def test_calibrator_reproduces_factory_offsets_from_real_rest_pose() -> None:
    # Raw radians read from a pre-assembled Franka GELLO Duo (left and right) resting in the calibration
    # pose; the calibrator must return Franka's factory offsets (franka_gello_duo.yaml).
    factory = (0.0, 0.0, np.pi, np.pi, np.pi, 3 * np.pi / 2, 0.0)
    raws = {
        "left": ((1.402, 3.827, 1.811, 3.055, 1.335, 3.187, 0.957), (-1.57, -0.80, 1.80, -3.00, 1.40, 1.50, -2.10)),
        "right": (
            (4.715, 3.821, 4.434, 3.056, -1.413, 3.185, -1.089),
            (1.57, -0.80, -1.80, -3.00, -1.40, 1.50, 2.10),
        ),
    }
    for raw, reference in raws.values():
        calibration = calculate_calibration(
            np.array(raw), reference_pose_rad=reference, gripper_closed_rad=2.24, gripper_open_rad=3.32, baudrate=57_600
        )
        assert np.allclose(np.mod(np.subtract(calibration.assembly_offsets_rad, factory) + 0.1, 2 * np.pi), 0.1)


def test_single_gello_uses_default_calibration(fake_dynamixel: object) -> None:
    _ = fake_dynamixel
    leader = GelloLeader("/dev/ttyUSB0")
    leader.connect()
    assert leader.get_observation().joint_positions.shape == (8,)
    leader.disconnect()
