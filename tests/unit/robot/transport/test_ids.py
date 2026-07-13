# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest

from physicalai.robot.transport._ids import (
    action_key,
    derive_device_id,
    derive_endpoint_port,
    derive_robot_id,
    meta_key,
    state_key,
)


class TestDeriveDeviceId:
    def test_serial_port_basename(self) -> None:
        assert derive_device_id({"port": "/dev/ttyUSB0"}) == "ttyUSB0"

    def test_symlink_resolved(self, tmp_path: Path) -> None:
        real = tmp_path / "ttyUSB7"
        real.touch()
        link = tmp_path / "by-id-alias"
        link.symlink_to(real)
        # Non-/dev paths resolve only for basename; use /dev-style check via tmp symlink
        assert Path(str(link)).resolve().name == "ttyUSB7"

    def test_dev_symlink_same_id(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # Simulate /dev/serial/by-id -> /dev/ttyUSB0 style aliasing.
        real = tmp_path / "ttyACM3"
        real.touch()
        link = tmp_path / "usb-alias"
        link.symlink_to(real)

        # Patch Path.resolve interception is unnecessary: derive_device_id
        # resolves anything starting with /dev/; feed it the tmp link via
        # a /dev/-prefixed fake by monkeypatching Path.resolve.
        original_resolve = Path.resolve

        def fake_resolve(self: Path) -> Path:
            if str(self) == "/dev/serial/by-id/usb-alias":
                return real
            return original_resolve(self)

        monkeypatch.setattr(Path, "resolve", fake_resolve)
        assert derive_device_id({"port": "/dev/serial/by-id/usb-alias"}) == "ttyACM3"

    def test_trossen_ip(self) -> None:
        assert derive_device_id({"ip": "192.168.1.2"}) == "192.168.1.2"

    def test_port_wins_over_ip(self) -> None:
        assert derive_device_id({"port": "/dev/ttyUSB0", "ip": "10.0.0.1"}) == "ttyUSB0"

    def test_neither_raises(self) -> None:
        with pytest.raises(ValueError, match="neither 'port' nor 'ip'"):
            derive_device_id({"role": "follower"})


class TestDeriveRobotId:
    def test_deterministic(self) -> None:
        kwargs = {"port": "/dev/ttyUSB0", "role": "follower"}
        a = derive_robot_id("so101", kwargs, host="hostA")
        b = derive_robot_id("so101", kwargs, host="hostA")
        assert a == b == "physicalai/robot/so101/hostA/ttyUSB0"

    def test_role_excluded(self) -> None:
        leader = derive_robot_id("so101", {"port": "/dev/ttyUSB0", "role": "leader"}, host="h")
        follower = derive_robot_id("so101", {"port": "/dev/ttyUSB0", "role": "follower"}, host="h")
        assert leader == follower

    def test_override_passthrough(self) -> None:
        rid = derive_robot_id("so101", {"port": "/dev/ttyUSB0"}, robot_id="left_arm")
        assert rid == "physicalai/robot/left_arm"

    def test_override_full_prefix_not_doubled(self) -> None:
        rid = derive_robot_id("so101", {}, robot_id="physicalai/robot/custom/x")
        assert rid == "physicalai/robot/custom/x"

    def test_default_host_is_hostname(self) -> None:
        import socket

        rid = derive_robot_id("so101", {"port": "/dev/ttyUSB0"})
        assert socket.gethostname() in rid


class TestKeys:
    def test_key_builders(self) -> None:
        rid = "physicalai/robot/so101/h/ttyUSB0"
        assert state_key(rid) == f"{rid}/state"
        assert action_key(rid) == f"{rid}/action"
        assert meta_key(rid) == f"{rid}/meta"


class TestEndpointPort:
    def test_deterministic_and_in_range(self) -> None:
        port = derive_endpoint_port("physicalai/robot/so101/h/ttyUSB0")
        assert port == derive_endpoint_port("physicalai/robot/so101/h/ttyUSB0")
        assert 17000 <= port <= 17999

    def test_different_ids_usually_differ(self) -> None:
        a = derive_endpoint_port("physicalai/robot/so101/h/ttyUSB0")
        b = derive_endpoint_port("physicalai/robot/so101/h/ttyUSB1")
        assert a != b
