# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Setup-only protocol-0 I/O with bounded, status-preserving packet reception.

The PyPI SDK's receive loops use wall time, busy-poll, and can bypass timeout
checks on malformed packets. Its sync reader also discards servo error status.
Use its transmit primitive, but receive first-party bounded status packets.
No SDK classes or normal Runtime connections are patched.
"""

from __future__ import annotations

import time
from importlib import import_module
from typing import TYPE_CHECKING, Protocol

_PACKET_TIMEOUT = 0.1
_WRITE_TIMEOUT = 0.1
_MAX_BUFFER = 4096
_MAX_ID = 252
_BROADCAST_ID = 254
_HEADER_LENGTH = 4
_MIN_STATUS_LENGTH = 2
_MAX_STATUS_LENGTH = 250
_CHECKSUM_MASK = 0xFF
_POSITION_WIDTH = 2

if TYPE_CHECKING:
    from serial import Serial


class _SDKPort(Protocol):
    ser: Serial
    is_using: bool


class _SetupPort:
    """Private SDK transmit adapter; never call the SDK's unbounded flush."""

    def __init__(self, port: _SDKPort) -> None:
        self.port = port

    @property
    def is_using(self) -> bool:
        return bool(self.port.is_using)

    @is_using.setter
    def is_using(self, value: bool) -> None:
        self.port.is_using = value

    def clearPort(self) -> None:  # noqa: N802 - SDK transmit interface
        self.port.ser.reset_input_buffer()

    def writePort(self, packet: list[int]) -> int:  # noqa: N802 - SDK transmit interface
        return int(self.port.ser.write(bytes(packet)))


class _SetupIO:
    def __init__(self, port: str) -> None:
        # Loading the setup API and constructing a session do not load the SDK.
        sdk = import_module("scservo_sdk")
        self.port = sdk.PortHandler(port)
        self._tx_port = _SetupPort(self.port)
        self._packet = sdk.PacketHandler(0)

    def open(self, baudrate: int) -> None:
        if not self.port.openPort():
            msg = "Could not open setup serial port"
            raise ConnectionError(msg)
        self.set_baudrate(baudrate)

    def set_baudrate(self, baudrate: int) -> None:
        # SDK setBaudRate reopens the device: apply serial bounds on EVERY reopen.
        if not self.port.setBaudRate(baudrate):
            msg = f"Could not set setup baudrate {baudrate}"
            raise ConnectionError(msg)
        self.port.ser.timeout = 0
        self.port.ser.write_timeout = _WRITE_TIMEOUT

    def close(self) -> None:
        # openPort can fail after creating a serial object but before is_open.
        if self.port.ser is not None:
            self.port.closePort()

    def _transmit(self, servo_id: int, instruction: int, parameters: list[int]) -> None:
        packet = [0, 0, servo_id, len(parameters) + 2, instruction, *parameters, 0]
        result = self._packet.txPacket(self._tx_port, packet)
        if result != 0:
            msg = f"Setup transmit failed for ID {servo_id}: communication status {result}"
            raise ConnectionError(msg)

    def _receive(self, deadline: float, *, discovery: bool = False) -> list[tuple[int, int, bytes]]:
        buffer = bytearray()
        packets: list[tuple[int, int, bytes]] = []
        while time.monotonic() < deadline:
            chunk = self.port.ser.read(_MAX_BUFFER)
            if not chunk:
                time.sleep(min(0.001, max(0.0, deadline - time.monotonic())))
                continue
            buffer.extend(chunk)
            if len(buffer) > _MAX_BUFFER:
                msg = "Ambiguous setup response: receive buffer overflow"
                raise ConnectionError(msg)
            while len(buffer) >= _HEADER_LENGTH:
                if buffer[:2] != b"\xff\xff" or not _MIN_STATUS_LENGTH <= buffer[3] <= _MAX_STATUS_LENGTH:
                    msg = "Corrupt or ambiguous setup response; connect only one motor for ID setup"
                    raise ConnectionError(msg)
                length = buffer[3] + 4
                if len(buffer) < length:
                    break
                frame = bytes(buffer[:length])
                del buffer[:length]
                if frame[2] > _MAX_ID or sum(frame[2:]) & _CHECKSUM_MASK != _CHECKSUM_MASK:
                    msg = "Corrupt setup response ID/checksum; check cabling and duplicate motor IDs"
                    raise ConnectionError(msg)
                packets.append((frame[2], frame[4], frame[5:-1]))
                if len(packets) > _MAX_ID + 1:
                    msg = "Ambiguous setup response: too many status packets"
                    raise ConnectionError(msg)
            if packets and not discovery:
                if buffer:
                    # Preserve a partial subsequent frame for a sync read.
                    continue
                return packets
        if buffer:
            msg = "Incomplete setup response; check cabling and duplicate motor IDs"
            raise ConnectionError(msg)
        return packets

    def exchange(
        self,
        servo_id: int,
        instruction: int,
        parameters: list[int],
        *,
        response_ids: set[int] | None = None,
    ) -> bytes:
        try:
            self._transmit(servo_id, instruction, parameters)
            packets = self._receive(time.monotonic() + _PACKET_TIMEOUT)
            expected = response_ids if response_ids is not None else {servo_id}
            if len(packets) != 1 or packets[0][0] not in expected:
                msg = f"Missing or ambiguous setup response for ID {servo_id}"
                raise ConnectionError(msg)
            _, error, data = packets[0]
            if error:
                msg = f"Servo ID {servo_id} rejected setup operation {instruction}: status {error}"
                raise RuntimeError(msg)
            return data
        finally:
            self._tx_port.is_using = False

    def read(self, servo_id: int, address: int, width: int) -> int:
        data = self.exchange(servo_id, 2, [address, width])
        if len(data) != width:
            msg = f"Invalid setup read length for ID {servo_id}, register {address}"
            raise ConnectionError(msg)
        return int.from_bytes(data, "little")

    def write(self, servo_id: int, address: int, width: int, value: int, *, new_id: int | None = None) -> None:
        response_ids = {servo_id} if new_id is None else {servo_id, new_id}
        data = self.exchange(servo_id, 3, [address, *value.to_bytes(width, "little")], response_ids=response_ids)
        if data:
            msg = f"Unexpected setup write response for ID {servo_id}, register {address}"
            raise ConnectionError(msg)

    def ping(self, servo_id: int) -> bool:
        return not self.exchange(servo_id, 1, [])

    def discover(self, baudrate: int) -> list[int]:
        # Conservatively allow all 253 status slots, 3 ms servo response spacing,
        # and full wire time at the slowest supported baudrate. Never stop at the
        # first responder: it could hide another motor at this/another baudrate.
        timeout = 0.8 + 253 * 6 * 10 / baudrate
        try:
            self._transmit(_BROADCAST_ID, 1, [])
            packets = self._receive(time.monotonic() + timeout, discovery=True)
            ids = [servo_id for servo_id, _, _ in packets]
            if len(set(ids)) != len(ids) or any(error or data for _, error, data in packets):
                msg = "Ambiguous motor discovery; connect exactly one healthy STS3215"
                raise RuntimeError(msg)
            return ids
        finally:
            self._tx_port.is_using = False

    def positions(self, servo_ids: list[int], address: int) -> list[int]:
        try:
            self._transmit(_BROADCAST_ID, 130, [address, 2, *servo_ids])
            deadline = time.monotonic() + _PACKET_TIMEOUT
            found: dict[int, int] = {}
            while len(found) < len(servo_ids):
                packets = self._receive(deadline)
                if not packets:
                    msg = f"Missing setup position responses for IDs {set(servo_ids) - found.keys()}"
                    raise ConnectionError(msg)
                for servo_id, error, data in packets:
                    if servo_id not in servo_ids or servo_id in found or len(data) != _POSITION_WIDTH:
                        msg = f"Ambiguous setup position response for ID {servo_id}"
                        raise ConnectionError(msg)
                    if error:
                        msg = f"Servo ID {servo_id} position read failed: status {error}"
                        raise RuntimeError(msg)
                    found[servo_id] = int.from_bytes(data, "little")
            return [found[servo_id] for servo_id in servo_ids]
        finally:
            self._tx_port.is_using = False
