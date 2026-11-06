# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Stateful hardware fakes for the public, torque-off setup contract."""

from __future__ import annotations

import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from physicalai_so101_plugin import SO101Calibration, SO101JointCalibration, SO101SetupSession
from physicalai_so101_plugin import setup as session_module
from physicalai_so101_plugin.constants import SO101_JOINT_ORDER, STS3215Addr as A, STS3215_BAUDRATES


class FakeIO:
    def __init__(self, port):
        self.baudrate = 1_000_000
        self.closed = False
        self.log = []
        self.fail_open = None
        self.fail_write = None
        self.mismatch = None
        self.motors = {}
        self.positions_entered = None
        self.positions_release = None
        for servo_id in range(1, 7):
            self.add_motor(servo_id)

    def add_motor(self, servo_id, baudrate=1_000_000, model=777):
        self.motors[(servo_id, baudrate)] = {
            A.MODEL_NUMBER: model,
            A.ID: servo_id,
            A.BAUD_RATE: STS3215_BAUDRATES[baudrate],
            A.TORQUE_ENABLE: 1,
            A.EEPROM_LOCK: 1,
            A.PHASE: 0x35,
            A.HOMING_OFFSET: 0,
            A.MIN_POSITION_LIMIT: 0,
            A.MAX_POSITION_LIMIT: 4095,
            A.PRESENT_VOLTAGE: 120,
            A.PRESENT_POSITION: 2047,
        }

    def open(self, baudrate):
        self.log.append(("open", baudrate))
        if self.fail_open:
            raise self.fail_open
        self.set_baudrate(baudrate)

    def set_baudrate(self, baudrate):
        self.log.append(("baud", baudrate))
        self.baudrate = baudrate

    def close(self):
        self.log.append(("close",))
        self.closed = True

    def motor(self, servo_id):
        try:
            return self.motors[(servo_id, self.baudrate)]
        except KeyError:
            raise ConnectionError(f"ID {servo_id} absent") from None

    def ping(self, servo_id):
        self.log.append(("ping", servo_id))
        self.motor(servo_id)
        return True

    def discover(self, baudrate):
        self.log.append(("discover", baudrate))
        return [servo_id for servo_id, baud in self.motors if baud == baudrate]

    def read(self, servo_id, address, width):
        self.log.append(("read", servo_id, address))
        motor = self.motor(servo_id)
        if self.mismatch == (servo_id, address):
            return 999
        return motor.get(address, 0)

    def write(self, servo_id, address, width, value, *, new_id=None):
        self.log.append(("write", servo_id, address, value))
        if self.fail_write == (servo_id, address):
            raise RuntimeError("Injected servo status error")
        motor = self.motor(servo_id)
        if address not in {A.TORQUE_ENABLE, A.EEPROM_LOCK, A.ACCELERATION}:
            assert motor[A.TORQUE_ENABLE] == 0, "Mutating with torque enabled"
            assert motor[A.EEPROM_LOCK] == 0, "Mutating locked EEPROM"
        motor[address] = value
        if address == A.ID:
            self.motors[(value, self.baudrate)] = self.motors.pop((servo_id, self.baudrate))
        if address == A.BAUD_RATE:
            baud = next(baud for baud, encoded in STS3215_BAUDRATES.items() if encoded == value)
            self.motors[(servo_id, baud)] = self.motors.pop((servo_id, self.baudrate))

    def positions(self, servo_ids, address):
        self.log.append(("positions",))
        if self.positions_entered is not None:
            self.positions_entered.set()
            assert self.positions_release.wait(5)
        result = []
        for servo_id in servo_ids:
            motor = self.motor(servo_id)
            encoded = motor[A.HOMING_OFFSET]
            offset = -(encoded & ~2048) if encoded & 2048 else encoded
            raw = motor[A.PRESENT_POSITION]
            position = raw - offset
            result.append(abs(position) | (0x8000 if position < 0 else 0))
        return result


@pytest.fixture
def hardware(monkeypatch):
    io = FakeIO("fake")
    monkeypatch.setattr(session_module, "_SetupIO", lambda port: io)
    return io


@pytest.fixture
def session(hardware):
    session = SO101SetupSession("fake", role="follower")
    session.connect()
    return session


def calibration(offset=-100):
    return SO101Calibration.from_dict({
        name: {"id": servo_id, "drive_mode": 0, "homing_offset": offset, "range_min": 100, "range_max": 3100}
        for servo_id, name in enumerate(SO101_JOINT_ORDER, 1)
    })


def test_import_is_lazy_in_isolated_process():
    code = """
import sys
from physicalai_so101_plugin import SO101Calibration, SO101SetupSession, get_urdf_path
SO101SetupSession('fake', role='follower')
assert 'scservo_sdk' not in sys.modules
assert not any(name.startswith('lerobot') for name in sys.modules)
"""
    subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).parents[1] / "src", check=True)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"port": "", "role": "leader"},
        {"port": "fake", "role": "wrong"},
        {"port": "fake", "role": "leader", "baudrate": 9600},
        {"port": "fake", "role": "leader", "baudrate": True},
    ],
)
def test_invalid_constructor_does_no_io(hardware, kwargs):
    with pytest.raises(ValueError):
        SO101SetupSession(**kwargs)
    assert hardware.log == []


def test_empty_arm_connect_and_idempotent_cleanup(hardware):
    hardware.motors.clear()
    session = SO101SetupSession("fake", role="follower")
    session.connect()
    session.connect()
    assert hardware.log == [("open", 1_000_000), ("baud", 1_000_000)]
    assert list(session.servo_ids) == list(SO101_JOINT_ORDER)
    with pytest.raises(TypeError):
        session.servo_ids["gripper"] = 99
    assert session.ping("gripper") is None
    session.disconnect()
    session.disconnect()
    assert hardware.closed
    assert len([entry for entry in hardware.log if entry[0] == "close"]) == 1
    with pytest.raises(ConnectionError):
        session.ping("gripper")


@pytest.mark.parametrize("error", [PermissionError("denied"), ConnectionError("missing"), RuntimeError("init")])
def test_failed_open_closes_partial_resources(hardware, error):
    hardware.fail_open = error
    session = SO101SetupSession("fake", role="leader")
    with pytest.raises(type(error)):
        session.connect()
    assert hardware.closed
    with pytest.raises(ConnectionError):
        session.read_positions()


def test_probe_voltage_and_invalid_name(session, hardware):
    assert session.ping("gripper") == 777
    hardware.motor(6)[A.MODEL_NUMBER] = 2825
    assert session.ping("gripper") == 2825
    hardware.motor(6)[A.PRESENT_VOLTAGE] = 0
    assert session.read_voltage("gripper") == 0
    del hardware.motors[(6, hardware.baudrate)]
    assert session.ping("gripper") is None
    with pytest.raises(ConnectionError):
        session.read_voltage("gripper")
    before = list(hardware.log)
    for operation in [session.ping, session.read_voltage, session.setup_motor]:
        with pytest.raises(ValueError):
            operation("unknown")
    assert hardware.log == before


@pytest.mark.parametrize("baudrate", list(STS3215_BAUDRATES))
def test_single_motor_setup_across_supported_baudrates(session, hardware, baudrate):
    hardware.motors.clear()
    hardware.add_motor(42, baudrate)
    session.setup_motor("elbow_flex")
    assert hardware.baudrate == 1_000_000
    assert set(hardware.motors) == {(3, 1_000_000)}
    motor = hardware.motor(3)
    assert motor[A.EEPROM_LOCK] == 1
    assert motor[A.TORQUE_ENABLE] == 0
    assert len([entry for entry in hardware.log if entry[0] == "discover"]) == 8


@pytest.mark.parametrize("kind", ["none", "multiple", "different_bauds", "wrong_model"])
def test_discovery_rejects_before_mutation(session, hardware, kind):
    hardware.motors.clear()
    if kind != "none":
        hardware.add_motor(42, model=2825 if kind == "wrong_model" else 777)
    if kind == "multiple":
        hardware.add_motor(43)
    if kind == "different_bauds":
        hardware.add_motor(43, 57_600)
    with pytest.raises((ConnectionError, RuntimeError)):
        session.setup_motor("gripper")
    assert hardware.baudrate == 1_000_000
    assert not any(entry[0] == "write" for entry in hardware.log)


def test_setup_partial_id_failure_reports_context_and_relocks(session, hardware):
    hardware.motors.clear()
    hardware.add_motor(42, 57_600)
    hardware.fail_write = (42, A.ID)
    with pytest.raises(RuntimeError, match="42.*57600"):
        session.setup_motor("gripper")
    assert hardware.baudrate == 1_000_000
    assert hardware.motors[(42, 57_600)][A.EEPROM_LOCK] == 1
    assert hardware.motors[(42, 57_600)][A.TORQUE_ENABLE] == 0


def test_homing_resets_offsets_before_read_and_uses_exact_center(session, hardware):
    actual = [0, 2046, 2047, 2048, 3000, 4094]
    for servo_id, position in enumerate(actual, 1):
        motor = hardware.motor(servo_id)
        motor.update({
            A.PRESENT_POSITION: position,
            A.HOMING_OFFSET: 100,
            A.MIN_POSITION_LIMIT: 100,
            A.MAX_POSITION_LIMIT: 3000,
        })
    session.load_calibration(calibration())
    offsets = session.set_half_turn_homings()
    assert list(offsets.values()) == [-2047, -1, 0, 1, 953, 2047]
    assert session.calibration is None
    assert list(session.read_positions().values()) == [2047] * 6
    first_read = next(i for i, entry in enumerate(hardware.log) if entry[0] == "positions")
    for servo_id in range(1, 7):
        assert ("write", servo_id, A.HOMING_OFFSET, 0) in hardware.log[:first_read]
        motor = hardware.motor(servo_id)
        assert motor[A.PHASE] == 0x25
        assert motor[A.MIN_POSITION_LIMIT] == 0
        assert motor[A.MAX_POSITION_LIMIT] == 4095
        assert motor[A.EEPROM_LOCK] == 1
        assert motor[A.TORQUE_ENABLE] == 0
    assert session.set_half_turn_homings() == offsets


def test_homing_endpoint_failure_does_not_wrap_or_write_calculated_offsets(session, hardware):
    hardware.motor(6)[A.PRESENT_POSITION] = 4095
    with pytest.raises(ValueError, match="center"):
        session.set_half_turn_homings()
    assert all(motor[A.HOMING_OFFSET] == 0 for motor in hardware.motors.values())
    assert all(motor[A.EEPROM_LOCK] == 1 for motor in hardware.motors.values())


def test_calibration_read_load_write_are_distinct_and_defensive(session, hardware):
    default = session.read_calibration()
    assert session.calibration is None
    assert default.joints["gripper"].range_max == 4095
    cal = calibration()
    before = list(hardware.log)
    session.load_calibration(cal)
    assert hardware.log == before
    cal.joints.clear()
    assert session.calibration == calibration()
    exposed = session.calibration
    exposed.joints.clear()
    assert session.calibration == calibration()
    session.write_calibration(calibration())
    assert session.calibration == session.read_calibration() == calibration()
    session.clear_calibration()
    assert session.calibration is None


@pytest.mark.parametrize(
    "change",
    [
        {"drive_mode": 1},
        {"id": 42},
        {"homing_offset": 2048},
        {"homing_offset": -2048},
        {"range_min": 3100},
        {"range_max": 4096},
        {"range_min": -1},
        {"id": True},
    ],
)
def test_setup_specific_validation_before_io(session, hardware, change):
    fields = calibration().to_dict()
    fields["gripper"].update(change)
    # Direct dataclass construction is deliberately tested: session validation
    # must not trust a frozen object's constructor as a validation boundary.
    cal = SO101Calibration({name: SO101JointCalibration(**joint) for name, joint in fields.items()})
    before = list(hardware.log)
    for operation in [session.load_calibration, session.write_calibration]:
        with pytest.raises(ValueError):
            operation(cal)
    assert hardware.log == before


@pytest.mark.parametrize("failure", ["write", "mismatch", "relock"])
def test_partial_calibration_has_no_cached_success(session, hardware, failure):
    session.load_calibration(calibration())
    if failure == "write":
        hardware.fail_write = (3, A.MIN_POSITION_LIMIT)
    elif failure == "mismatch":
        hardware.mismatch = (3, A.MIN_POSITION_LIMIT)
    else:
        original = hardware.write

        def fail_relock(servo_id, address, width, value, **kwargs):
            if servo_id == 3 and address == A.EEPROM_LOCK and value == 1:
                raise RuntimeError("relock failed")
            return original(servo_id, address, width, value, **kwargs)

        hardware.write = fail_relock
    with pytest.raises(RuntimeError):
        session.write_calibration(calibration())
    assert session.calibration is None
    assert all(motor[A.TORQUE_ENABLE] == 0 for motor in hardware.motors.values())


@pytest.mark.parametrize("role", ["leader", "follower"])
def test_exact_role_configuration_and_no_motion(hardware, role):
    session = SO101SetupSession("fake", role=role)
    session.connect()
    session.configure_motors()
    writes = [entry for entry in hardware.log if entry[0] == "write"]
    for servo_id in range(1, 7):
        for address, value in [
            (A.RETURN_DELAY_TIME, 0),
            (A.MAXIMUM_ACCELERATION, 254),
            (A.ACCELERATION, 254),
            (A.OPERATING_MODE, 0),
            (A.PHASE, 0x25),
        ]:
            assert ("write", servo_id, address, value) in writes
        for address, value in [(A.P_COEFFICIENT, 16), (A.I_COEFFICIENT, 0), (A.D_COEFFICIENT, 32)]:
            assert (("write", servo_id, address, value) in writes) == (role == "follower")
    protection = [(A.MAX_TORQUE_LIMIT, 500), (A.PROTECTION_CURRENT, 250), (A.OVERLOAD_TORQUE, 25)]
    for address, value in protection:
        assert (("write", 6, address, value) in writes) == (role == "follower")
        assert not any(entry[2] == address and entry[1] != 6 for entry in writes)
    session.disconnect()
    assert not any(
        entry[2] == A.GOAL_POSITION or (entry[2] == A.TORQUE_ENABLE and entry[3] != 0)
        for entry in hardware.log
        if entry[0] == "write"
    )


@pytest.mark.parametrize(
    "tick,body,gripper", [(0, -100, 0), (100, -100, 0), (1600, 0, 50), (3100, 100, 100), (4095, 100, 100)]
)
def test_normalized_golden_values(session, hardware, tick, body, gripper):
    session.load_calibration(calibration(offset=0))
    for motor in hardware.motors.values():
        motor[A.PRESENT_POSITION] = tick
    values = session.read_positions(normalized=True)
    assert list(values) == list(SO101_JOINT_ORDER)
    assert list(values.values()) == [float(body)] * 5 + [float(gripper)]
    from physicalai_so101_plugin._conversion import ticks_to_normalized

    assert ticks_to_normalized(np.full(6, tick), calibration()).dtype == np.float32


def test_signed_positions_and_normalized_cache_requirement(session, hardware):
    hardware.motor(1)[A.PRESENT_POSITION] = -20
    assert session.read_positions()["shoulder_pan"] == -20
    with pytest.raises(RuntimeError, match="calibration"):
        session.read_positions(normalized=True)


def test_disconnect_serializes_with_active_read(session, hardware):
    hardware.positions_entered = threading.Event()
    hardware.positions_release = threading.Event()
    with ThreadPoolExecutor(max_workers=2) as executor:
        read = executor.submit(session.read_positions)
        assert hardware.positions_entered.wait(5)
        disconnect_started = threading.Event()

        def disconnect():
            disconnect_started.set()
            session.disconnect()

        close = executor.submit(disconnect)
        assert disconnect_started.wait(5)
        assert not hardware.closed
        hardware.positions_release.set()
        read.result(5)
        close.result(5)
    assert hardware.closed
    with pytest.raises(ConnectionError):
        session.read_positions()
