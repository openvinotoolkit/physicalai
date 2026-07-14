# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import socket

import pytest

from physicalai.robot.transport._ids import (
    action_key,
    derive_endpoint_port,
    metadata_key,
    robot_prefix,
    state_key,
    validate_name,
)


class TestValidateName:
    def test_valid_name_passthrough(self) -> None:
        assert validate_name("left-arm") == "left-arm"

    def test_underscore_and_digits_allowed(self) -> None:
        assert validate_name("arm_7") == "arm_7"

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="invalid robot name"):
            validate_name("")

    def test_slash_raises(self) -> None:
        with pytest.raises(ValueError, match="invalid robot name"):
            validate_name("left/arm")

    def test_wildcard_raises(self) -> None:
        with pytest.raises(ValueError, match="invalid robot name"):
            validate_name("left*")


class TestRobotPrefix:
    def test_prefix(self) -> None:
        assert robot_prefix("left-arm") == "physicalai/robot/left-arm"

    def test_invalid_name_raises(self) -> None:
        with pytest.raises(ValueError, match="invalid robot name"):
            robot_prefix("bad/name")


class TestKeys:
    def test_key_builders(self) -> None:
        name = "left-arm"
        prefix = "physicalai/robot/left-arm"
        assert state_key(name) == f"{prefix}/state"
        assert action_key(name) == f"{prefix}/action"
        assert metadata_key(name) == f"{prefix}/metadata"


class TestEndpointPort:
    def test_deterministic_and_in_range(self) -> None:
        port = derive_endpoint_port("left-arm")
        assert port == derive_endpoint_port("left-arm")
        assert 20000 <= port <= 59999

    def test_different_names_usually_differ(self) -> None:
        a = derive_endpoint_port("left-arm")
        b = derive_endpoint_port("right-arm")
        assert a != b


def test_default_host_is_hostname() -> None:
    from physicalai.robot.transport._ids import default_host

    assert default_host() == socket.gethostname()
