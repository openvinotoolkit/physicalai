# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Noninteractive, torque-off SO101 setup, separate from deployment drivers."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal

import numpy as np
from loguru import logger

from physicalai_so101_plugin._conversion import ticks_to_normalized
from physicalai_so101_plugin._setup_io import _SetupIO
from physicalai_so101_plugin.calibration import SO101Calibration
from physicalai_so101_plugin.constants import (
    SO101_JOINT_ORDER,
    STS3215_BAUDRATES,
    STS3215_MODEL_NUMBER,
    VALID_ROLES,
    STS3215Addr,
    STS3215Len,
)

_OFFSET_SIGN_BIT = 11
_POSITION_SIGN_BIT = 15
_ENCODER_MAX = 4095
_HALF_TURN = 2047

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping


def _decode_signed(value: int, sign_bit: int) -> int:
    magnitude = value & ~(1 << sign_bit)
    return -magnitude if value & (1 << sign_bit) else magnitude


def _encode_offset(value: int) -> int:
    if abs(value) > _HALF_TURN:
        msg = f"Homing offset {value} exceeds ±2047; move the joint away from the encoder endpoint and center it again"
        raise ValueError(msg)
    return abs(value) | (1 << _OFFSET_SIGN_BIT if value < 0 else 0)


class SO101SetupSession:
    """One exclusive serial owner for diagnostics, calibration and verification.

    Construction does no I/O. Connection opens the port without pinging or
    writing registers, so incomplete arms remain accessible. Explicit setup
    operations disable torque, including for followers; none enables it or
    writes goal positions. Call ``disable_torque`` before manual positioning.

    Warning:
        Support the arm throughout setup and disconnect: torque-off joints can
        fall under gravity. Homing destructively resets previous calibration,
        even if a later step fails. Hardware writes are not rolled back.

    Args:
        port: Operator-selected serial device path.
        role: Leader or follower; only followers receive PID/protection tuning.
        baudrate: Supported STS3215 baudrate, defaulting to 1,000,000.

    Raises:
        ValueError: If port, role or baudrate is invalid.
    """

    def __init__(self, port: str, *, role: Literal["leader", "follower"], baudrate: int = 1_000_000) -> None:
        """Validate session arguments without loading the SDK or opening hardware.

        Raises:
            ValueError: If port, role or baudrate is invalid.
        """
        if not isinstance(port, str) or not port.strip() or "\x00" in port:
            msg = "Setup port must be a nonempty serial device path"
            raise ValueError(msg)
        if role not in VALID_ROLES:
            msg = f"Invalid setup role {role!r}"
            raise ValueError(msg)
        if type(baudrate) is not int or baudrate not in STS3215_BAUDRATES:
            msg = f"Unsupported STS3215 baudrate {baudrate!r}"
            raise ValueError(msg)
        self._port = port
        self._role = role
        self._baudrate = baudrate
        self._servo_ids = MappingProxyType(dict(zip(SO101_JOINT_ORDER, range(1, 7), strict=True)))
        self._calibration: SO101Calibration | None = None
        self._io: _SetupIO | None = None
        self._lock = threading.RLock()

    @property
    def servo_ids(self) -> Mapping[str, int]:
        """Read-only canonical joint names and IDs, in observation order."""
        return self._servo_ids

    @property
    def calibration(self) -> SO101Calibration | None:
        """Validated cache copy, not necessarily the current EEPROM contents."""
        with self._lock:
            return None if self._calibration is None else SO101Calibration.from_dict(self._calibration.to_dict())

    def clear_calibration(self) -> None:
        """Invalidate software calibration without touching hardware.

        Use when entering a new setup/homing flow, so stale verification cannot
        reuse a previous successful calibration. This operation performs no I/O.
        """
        with self._lock:
            self._calibration = None

    @contextmanager
    def _operation(self) -> Iterator[_SetupIO]:
        with self._lock:
            if self._io is None:
                msg = "SO101 setup session is disconnected"
                raise ConnectionError(msg)
            try:
                yield self._io
            except PermissionError:
                raise
            except OSError as exc:
                if isinstance(exc, ConnectionError):
                    raise
                msg = f"SO101 setup communication failed on {self._port}: {exc}"
                raise ConnectionError(msg) from exc

    def connect(self) -> None:
        """Open the configured port with no motor probes or register writes.

        Idempotent. Permission failures remain PermissionError; device and
        communication failures raise ConnectionError. Partial resources close.

        Raises:
            ConnectionError: If serial-port setup fails.
        """
        with self._lock:
            if self._io is not None:
                return
            io = _SetupIO(self._port)
            try:
                io.open(self._baudrate)
            except BaseException as exc:
                try:
                    io.close()
                except Exception:  # noqa: BLE001 - cleanup must preserve the original open failure
                    logger.exception("Failed to close partially opened SO101 setup port")
                if isinstance(exc, OSError) and not isinstance(exc, (PermissionError, ConnectionError)):
                    msg = f"Could not open SO101 setup port {self._port}: {exc}"
                    raise ConnectionError(msg) from exc
                raise
            self._io = io

    def disconnect(self) -> None:
        """Best-effort torque off at every expected ID, then always close.

        Idempotent and serialized with active operations. Support the arm: this
        does not hold position, including for followers with missing motors.
        """
        with self._lock:
            io = self._io
            if io is None:
                return
            try:
                for servo_id in self._servo_ids.values():
                    try:
                        io.write(servo_id, STS3215Addr.TORQUE_ENABLE, 1, 0)
                    except Exception:  # noqa: BLE001 - attempt every expected motor before closing
                        logger.warning("Could not disable setup torque on ID {} during disconnect", servo_id)
            finally:
                self._io = None
                io.close()

    def _motor_id(self, name: str) -> int:
        if name not in self._servo_ids:
            msg = f"Unknown SO101 motor {name!r}"
            raise ValueError(msg)
        return self._servo_ids[name]

    def _read(self, servo_id: int, address: STS3215Addr) -> int:
        io = self._io
        if io is None:
            msg = "SO101 setup session is disconnected"
            raise ConnectionError(msg)
        try:
            return io.read(servo_id, address, STS3215Len[address.name])
        except (ConnectionError, RuntimeError) as exc:
            msg = f"ID {servo_id} read {address.name}: {exc}"
            raise type(exc)(msg) from exc

    def _write(self, servo_id: int, address: STS3215Addr, value: int, *, verify: bool = True) -> None:
        io = self._io
        if io is None:
            msg = "SO101 setup session is disconnected"
            raise ConnectionError(msg)
        try:
            io.write(servo_id, address, STS3215Len[address.name], value)
            self._verify_write(servo_id, address, value, verify=verify)
        except (ConnectionError, RuntimeError) as exc:
            msg = f"ID {servo_id} write {address.name}: {exc}"
            raise type(exc)(msg) from exc

    def _verify_write(self, servo_id: int, address: STS3215Addr, value: int, *, verify: bool) -> None:
        if verify and self._read(servo_id, address) != value:
            msg = f"ID {servo_id} write {address.name}: read-back mismatch"
            raise RuntimeError(msg)

    def ping(self, motor_name: str) -> int | None:
        """Return actual model number, or None for a failed expected-motor probe.

        Unknown names raise ValueError before bus traffic. A disconnected session
        raises ConnectionError rather than treating all motors as missing.
        """
        servo_id = self._motor_id(motor_name)
        with self._operation() as io:
            try:
                if not io.ping(servo_id):
                    return None
                return self._read(servo_id, STS3215Addr.MODEL_NUMBER)
            except (ConnectionError, RuntimeError):
                return None

    def read_voltage(self, motor_name: str) -> int:
        """Read raw 0.1 V register units; unsuccessful reads raise, not return zero.

        Returns:
            Raw voltage integer, including genuine zero.
        """
        servo_id = self._motor_id(motor_name)
        with self._operation():
            return self._read(servo_id, STS3215Addr.PRESENT_VOLTAGE)

    def _validate_motors(self) -> None:
        for name, servo_id in self._servo_ids.items():
            model = self._read(servo_id, STS3215Addr.MODEL_NUMBER)
            if model != STS3215_MODEL_NUMBER:
                msg = f"Motor {name} (ID {servo_id}) has model {model}; expected {STS3215_MODEL_NUMBER}"
                raise RuntimeError(msg)

    def disable_torque(self) -> None:
        """Disable and verify torque for all six motors; never unlock EEPROM.

        Attempt every motor even if one fails, then raise the first failure.
        """
        with self._operation():
            failure: Exception | None = None
            for servo_id in self._servo_ids.values():
                try:
                    self._write(servo_id, STS3215Addr.TORQUE_ENABLE, 0)
                except (OSError, RuntimeError) as exc:
                    if failure is None:
                        failure = exc
            if failure is not None:
                raise failure

    @contextmanager
    def _unlocked(self, ids: list[int]) -> Iterator[None]:
        unlocked: list[int] = []
        failed = False
        try:
            for servo_id in ids:
                self._write(servo_id, STS3215Addr.TORQUE_ENABLE, 0)
            for servo_id in ids:
                # Track BEFORE the write: a lost response can still mean EEPROM
                # unlocked, so cleanup must attempt to relock that motor.
                unlocked.append(servo_id)
                self._write(servo_id, STS3215Addr.EEPROM_LOCK, 0)
            yield
        except BaseException:
            failed = True
            raise
        finally:
            cleanup_failure: Exception | None = None
            for servo_id in unlocked:
                try:
                    self._write(servo_id, STS3215Addr.EEPROM_LOCK, 1)
                except (OSError, RuntimeError) as exc:
                    logger.warning("Could not relock setup EEPROM on ID {}: {}", servo_id, exc)
                    if cleanup_failure is None:
                        cleanup_failure = exc
            if cleanup_failure is not None and not failed:
                raise cleanup_failure

    def setup_motor(self, motor_name: str) -> None:
        """Discover and assign a single physically isolated STS3215.

        Scan all eight supported baudrates before mutation. Multiple, corrupt or
        wrong-model responders are rejected. Duplicate physical IDs cannot be
        proven absent; connect ONLY the selected motor. Failure reports known
        old/new ID and baudrate context without promising hardware rollback.

        Raises:
            PermissionError: If access to the serial device is denied.
            ConnectionError: If discovery or communication fails.
            RuntimeError: If discovery is ambiguous or a write does not verify.
        """
        target_id = self._motor_id(motor_name)
        with self._operation() as io:
            self._calibration = None
            old_address: tuple[int, int] | None = None
            try:
                old_address = self._discover_single(io, motor_name)
                self._assign_motor(io, old_address, target_id)
            except PermissionError:
                raise
            except RuntimeError as exc:
                msg = self._assignment_failure(motor_name, old_address, target_id, exc)
                raise RuntimeError(msg) from exc
            except OSError as exc:
                msg = self._assignment_failure(motor_name, old_address, target_id, exc)
                raise ConnectionError(msg) from exc
            finally:
                self._restore_host_baudrate(io)

    def _assignment_failure(
        self, name: str, old_address: tuple[int, int] | None, target_id: int, exc: Exception
    ) -> str:
        return (
            f"Setup {name} failed: old ID/baudrate={old_address}, "
            f"target ID/baudrate={target_id}/{self._baudrate}: {exc}. "
            "Hardware writes may have partially applied; rescan the isolated motor."
        )

    def _discover_single(self, io: _SetupIO, motor_name: str) -> tuple[int, int]:
        found: list[tuple[int, int]] = []
        baudrates = [self._baudrate, *(baud for baud in STS3215_BAUDRATES if baud != self._baudrate)]
        for baud in baudrates:
            io.set_baudrate(baud)
            for servo_id in io.discover(baud):
                found.append((servo_id, baud))
                model = self._read(servo_id, STS3215Addr.MODEL_NUMBER)
                if model != STS3215_MODEL_NUMBER:
                    msg = f"ID {servo_id} at {baud} has model {model}, expected {STS3215_MODEL_NUMBER}"
                    raise RuntimeError(msg)
                if len(found) > 1:
                    msg = f"Multiple motors detected at ID/baudrate {found}; connect only {motor_name}"
                    raise RuntimeError(msg)
        if not found:
            msg = f"No STS3215 found; connect only {motor_name} and check its power/cabling"
            raise ConnectionError(msg)
        return found[0]

    def _assign_motor(self, io: _SetupIO, old_address: tuple[int, int], target_id: int) -> None:
        old_id, old_baud = old_address
        io.set_baudrate(old_baud)
        try:
            self._write(old_id, STS3215Addr.TORQUE_ENABLE, 0)
            self._write(old_id, STS3215Addr.EEPROM_LOCK, 0)
            self._assign_id(io, old_id, target_id)
            self._assign_baudrate(io, target_id)
            self._write(target_id, STS3215Addr.EEPROM_LOCK, 1)
        except BaseException:
            self._relock_candidates(io, old_address, target_id)
            raise

    def _assign_id(self, io: _SetupIO, old_id: int, target_id: int) -> None:
        if old_id != target_id:
            try:
                io.write(old_id, STS3215Addr.ID, 1, target_id, new_id=target_id)
            except ConnectionError:
                # A missed acknowledgement is not proof the ID stayed unchanged.
                if self._read(target_id, STS3215Addr.ID) != target_id:
                    raise
        if self._read(target_id, STS3215Addr.ID) != target_id:
            msg = f"Motor ID assignment to {target_id} did not verify"
            raise RuntimeError(msg)

    def _assign_baudrate(self, io: _SetupIO, target_id: int) -> None:
        baud_error: ConnectionError | None = None
        try:
            io.write(target_id, STS3215Addr.BAUD_RATE, 1, STS3215_BAUDRATES[self._baudrate])
        except ConnectionError as exc:
            baud_error = exc
        # An acknowledgement may occur at old settings. Verify at the target;
        # never retry the baud-changing write blindly after losing its response.
        io.set_baudrate(self._baudrate)
        try:
            self._validate_assigned_motor(target_id)
        except (ConnectionError, RuntimeError) as exc:
            if baud_error is not None:
                raise exc from baud_error
            raise

    def _relock_candidates(self, io: _SetupIO, old_address: tuple[int, int], target_id: int) -> None:
        old_id, old_baud = old_address
        cleanup_ok = False
        for baud in dict.fromkeys([self._baudrate, old_baud]):
            try:
                io.set_baudrate(baud)
            except OSError:
                continue
            for servo_id in dict.fromkeys([target_id, old_id]):
                try:
                    self._write(servo_id, STS3215Addr.TORQUE_ENABLE, 0)
                    self._write(servo_id, STS3215Addr.EEPROM_LOCK, 1)
                    cleanup_ok = True
                except (OSError, RuntimeError):
                    continue
        if not cleanup_ok:
            logger.error("Could not verify torque-off/EEPROM relock after failed SO101 ID setup")

    def _restore_host_baudrate(self, io: _SetupIO) -> None:
        try:
            io.set_baudrate(self._baudrate)
        except OSError:
            # Do not leave a seemingly connected session at unknown host settings.
            self._io = None
            io.close()
            raise

    def _validate_assigned_motor(self, servo_id: int) -> None:
        expected = {
            STS3215Addr.MODEL_NUMBER: STS3215_MODEL_NUMBER,
            STS3215Addr.ID: servo_id,
            STS3215Addr.BAUD_RATE: STS3215_BAUDRATES[self._baudrate],
        }
        for address, value in expected.items():
            if self._read(servo_id, address) != value:
                msg = f"Assigned ID {servo_id} register {address.name} failed verification"
                raise RuntimeError(msg)

    def _position_mode(self, servo_id: int) -> None:
        self._write(servo_id, STS3215Addr.OPERATING_MODE, 0)
        phase = self._read(servo_id, STS3215Addr.PHASE)
        self._write(servo_id, STS3215Addr.PHASE, phase & ~0x10)

    def set_half_turn_homings(self) -> dict[str, int]:
        """Destructively reset calibration and write verified half-turn offsets.

        Returns:
            Written offsets around center 2047, keyed by canonical joint name.

        Raises:
            ValueError: If centering produces an unencodable offset or position.
        """
        with self._operation():
            self._validate_motors()
            self._calibration = None
            with self._unlocked(list(self._servo_ids.values())):
                for servo_id in self._servo_ids.values():
                    self._position_mode(servo_id)
                    self._write(servo_id, STS3215Addr.HOMING_OFFSET, 0)
                    self._write(servo_id, STS3215Addr.MIN_POSITION_LIMIT, 0)
                    self._write(servo_id, STS3215Addr.MAX_POSITION_LIMIT, _ENCODER_MAX)
                positions = self.read_positions()
                offsets = {name: int(position) - _HALF_TURN for name, position in positions.items()}
                for name, position in positions.items():
                    if not 0 <= position <= _ENCODER_MAX:
                        msg = f"Motor {name} has invalid encoder position {position}; center it and retry homing"
                        raise ValueError(msg)
                encoded = {name: _encode_offset(offset) for name, offset in offsets.items()}
                for name, value in encoded.items():
                    self._write(self._servo_ids[name], STS3215Addr.HOMING_OFFSET, value)
            return offsets

    def read_positions(self, *, normalized: bool = False) -> dict[str, int | float]:
        """Read signed already-homed positions, optionally using cached ranges.

        Returns:
            Canonically ordered joint positions. Raw signed values remain signed;
            callers recording encoder ranges must reject values outside 0..4095.

        Raises:
            RuntimeError: If normalized reads have no validated calibration.
        """
        with self._operation() as io:
            if normalized and self._calibration is None:
                msg = "Load or write calibration before requesting normalized setup positions"
                raise RuntimeError(msg)
            raw = io.positions(list(self._servo_ids.values()), STS3215Addr.PRESENT_POSITION)
            positions = np.array([_decode_signed(value, _POSITION_SIGN_BIT) for value in raw], dtype=np.int32)
            if normalized:
                calibration = self._calibration
                if calibration is None:
                    msg = "Setup calibration is unavailable"
                    raise RuntimeError(msg)
                values = ticks_to_normalized(positions, calibration)
                return {name: float(value) for name, value in zip(self._servo_ids, values, strict=True)}
            return {name: int(value) for name, value in zip(self._servo_ids, positions, strict=True)}

    def _validated_calibration(self, calibration: SO101Calibration) -> SO101Calibration:
        data = calibration.to_dict()
        if set(data) != set(self._servo_ids):
            msg = "Setup calibration must contain exactly the six canonical joints"
            raise ValueError(msg)
        for name, fields in data.items():
            if any(type(value) is not int for value in fields.values()):
                msg = f"Motor {name} setup calibration fields must be integers"
                raise ValueError(msg)
            if fields["id"] != self._servo_ids[name] or fields["drive_mode"] != 0:
                msg = f"Motor {name} requires canonical ID {self._servo_ids[name]} and drive_mode=0 for setup"
                raise ValueError(msg)
            _encode_offset(fields["homing_offset"])
        return SO101Calibration.from_dict(data)

    def read_calibration(self) -> SO101Calibration:
        """Read and validate all six EEPROM calibrations without updating cache.

        Returns:
            Hardware offsets and limits, with software drive_mode fixed to zero.
        """
        with self._operation():
            data = {}
            for name, servo_id in self._servo_ids.items():
                data[name] = {
                    "id": servo_id,
                    "drive_mode": 0,
                    "homing_offset": _decode_signed(self._read(servo_id, STS3215Addr.HOMING_OFFSET), _OFFSET_SIGN_BIT),
                    "range_min": self._read(servo_id, STS3215Addr.MIN_POSITION_LIMIT),
                    "range_max": self._read(servo_id, STS3215Addr.MAX_POSITION_LIMIT),
                }
            return self._validated_calibration(SO101Calibration.from_dict(data))

    def load_calibration(self, calibration: SO101Calibration) -> None:
        """Validate and defensively cache calibration, without hardware writes."""
        with self._lock:
            self._calibration = self._validated_calibration(calibration)

    def write_calibration(self, calibration: SO101Calibration) -> None:
        """Write, relock and verify calibration; cache only complete success.

        Torque remains disabled. Partial EEPROM writes are possible on failure;
        cached calibration is invalidated before the first mutation.

        Raises:
            RuntimeError: If calibration writes or read-back verification fail.
        """
        with self._operation():
            validated = self._validated_calibration(calibration)
            self._validate_motors()
            self._calibration = None
            with self._unlocked(list(self._servo_ids.values())):
                for name, cal in validated.joints.items():
                    servo_id = self._servo_ids[name]
                    self._write(servo_id, STS3215Addr.HOMING_OFFSET, _encode_offset(cal.homing_offset))
                    self._write(servo_id, STS3215Addr.MIN_POSITION_LIMIT, cal.range_min)
                    self._write(servo_id, STS3215Addr.MAX_POSITION_LIMIT, cal.range_max)
                if self.read_calibration() != validated:
                    msg = "SO101 setup calibration read-back mismatch"
                    raise RuntimeError(msg)
            self._calibration = validated

    def configure_motors(self) -> None:
        """Apply verified role-specific setup configuration with torque disabled.

        Leaders receive no follower PID/protection settings. No goal position or
        torque-enable command is sent, including during follower verification.
        """
        with self._operation():
            self._validate_motors()
            with self._unlocked(list(self._servo_ids.values())):
                for name, servo_id in self._servo_ids.items():
                    self._position_mode(servo_id)
                    for address, value in self._configuration(name):
                        self._write(servo_id, address, value)

    def _configuration(self, name: str) -> list[tuple[STS3215Addr, int]]:
        settings = [
            (STS3215Addr.RETURN_DELAY_TIME, 0),
            (STS3215Addr.MAXIMUM_ACCELERATION, 254),
            (STS3215Addr.ACCELERATION, 254),
        ]
        if self._role == "follower":
            settings.extend([
                (STS3215Addr.P_COEFFICIENT, 16),
                (STS3215Addr.I_COEFFICIENT, 0),
                (STS3215Addr.D_COEFFICIENT, 32),
            ])
            if name == "gripper":
                settings.extend([
                    (STS3215Addr.MAX_TORQUE_LIMIT, 500),
                    (STS3215Addr.PROTECTION_CURRENT, 250),
                    (STS3215Addr.OVERLOAD_TORQUE, 25),
                ])
        return settings
