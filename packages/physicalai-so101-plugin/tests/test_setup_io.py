# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Packet-level tests, including the deployed SDK transmit primitive."""

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from physicalai_so101_plugin import _setup_io as io_module
from physicalai_so101_plugin._setup_io import _SetupIO, _SetupPort


def frame(servo_id=1, error=0, data=b""):
    body = bytes([servo_id, len(data) + 2, error]) + data
    return b"\xff\xff" + body + bytes([~sum(body) & 255])


class FakeSerial:
    def __init__(self):
        self.chunks = []
        self.response = []
        self.writes = []

    def read(self, size):
        return self.chunks.pop(0) if self.chunks else b""

    def reset_input_buffer(self):
        self.chunks.clear()

    def write(self, packet):
        self.writes.append(packet)
        self.chunks.extend(self.response)
        return len(packet)


class FakePacket:
    def txPacket(self, port, packet):
        if port.is_using:
            return -1
        port.is_using = True
        packet[:2] = [255, 255]
        packet[-1] = ~sum(packet[2:-1]) & 255
        port.clearPort()
        return 0 if port.writePort(packet) == len(packet) else -2


@pytest.fixture
def io(monkeypatch):
    # Deterministic deadlines, with each empty read yielding a clock advance.
    now = [0.0]
    monkeypatch.setattr(io_module.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(io_module.time, "sleep", lambda seconds: now.__setitem__(0, now[0] + seconds))
    io = _SetupIO.__new__(_SetupIO)
    io.port = SimpleNamespace(ser=FakeSerial(), is_using=False)
    io._tx_port = _SetupPort(io.port)
    io._packet = FakePacket()
    return io


def test_explicit_little_endian_model_and_register_encoding(io):
    io.port.ser.response = [frame(data=b"\x09\x03")]
    assert io.read(1, 3, 2) == 777
    assert io.port.ser.writes[-1][2:7] == bytes([1, 4, 2, 3, 2])
    io.port.ser.response = [frame()]
    io.write(1, 31, 2, 0x0801)
    assert io.port.ser.writes[-1][5:-1] == bytes([31, 1, 8])
    assert not io.port.is_using


@pytest.mark.parametrize("response,error", [
    ([], ConnectionError), ([frame(error=1)], RuntimeError),
    ([b"garbage"], ConnectionError), ([frame()[:-1] + b"\x00"], ConnectionError),
    ([frame()[:-1]], ConnectionError), ([frame(2)], ConnectionError),
    ([frame() + frame()], ConnectionError), ([b"\xff\xff\x01\xff"], ConnectionError),
])
def test_bounded_packet_failures_release_sdk_busy_flag(io, response, error):
    io.port.ser.response = response
    with pytest.raises(error):
        io.ping(1)
    assert not io.port.is_using


def test_fragmented_response_and_old_or_new_id_ack(io):
    packet = frame(6)
    io.port.ser.response = [packet[:3], packet[3:]]
    io.write(42, 5, 1, 6, new_id=6)
    io.port.ser.response = [frame(42)]
    io.write(42, 5, 1, 6, new_id=6)


def test_sync_positions_preserve_errors_and_never_return_stale_data(io):
    io.port.ser.response = [frame(2, data=b"\xff\x7f"), frame(1, data=b"\x14\x80")]
    assert io.positions([1, 2], 56) == [0x8014, 0x7FFF]
    io.port.ser.response = [frame(1, error=32, data=b"\x00\x00"), frame(2, data=b"\x00\x00")]
    with pytest.raises(RuntimeError, match="status 32"):
        io.positions([1, 2], 56)
    io.port.ser.response = [frame(1, data=b"\x00\x00")]
    with pytest.raises(ConnectionError, match="Missing"):
        io.positions([1, 2], 56)


def test_discovery_collects_all_responders_and_rejects_duplicate_or_corrupt(io):
    io.port.ser.response = [frame(1), frame(42)]
    assert io.discover(19_200) == [1, 42]
    io.port.ser.response = [frame(1), frame(1)]
    with pytest.raises(RuntimeError, match="Ambiguous"):
        io.discover(1_000_000)
    io.port.ser.response = [frame(1), b"bad"]
    with pytest.raises(ConnectionError):
        io.discover(1_000_000)
    io.port.ser.response = []
    assert io.discover(1_000_000) == []


def test_every_sdk_reopen_reapplies_serial_timeouts(io):
    io.port.setBaudRate = Mock(return_value=True)
    io.set_baudrate(57_600)
    assert io.port.ser.timeout == 0
    assert io.port.ser.write_timeout == 0.1
    io.port.ser.timeout = None
    io.port.ser.write_timeout = None
    io.set_baudrate(1_000_000)
    assert io.port.ser.timeout == 0
    assert io.port.ser.write_timeout == 0.1
    io.port.setBaudRate.return_value = False
    with pytest.raises(ConnectionError):
        io.set_baudrate(57_600)


def test_missing_serial_write_count_is_a_communication_failure(io):
    io.port.ser.write = lambda packet: None
    with pytest.raises(ConnectionError, match="transmit"):
        io.ping(1)
    assert not io.port.is_using


def test_real_sdk_transmit_conformance_in_isolated_process():
    # The driver test suite installs module mocks, so exercise the real SDK in
    # a fresh process. Serial remains fake: this test never opens a device.
    code = """
from types import SimpleNamespace
from scservo_sdk import PacketHandler
from physicalai_so101_plugin._setup_io import _SetupPort
class Serial:
    packet = None
    def reset_input_buffer(self): pass
    def write(self, packet):
        self.packet = packet
        return len(packet)
serial = Serial()
port = SimpleNamespace(ser=serial, is_using=False)
tx = _SetupPort(port)
result = PacketHandler(0).txPacket(tx, [0, 0, 1, 4, 2, 3, 2, 0])
assert result == 0
assert serial.packet == bytes([255, 255, 1, 4, 2, 3, 2, 243])
"""
    subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).parents[1] / "src", check=True)
