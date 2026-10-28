# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Shared, hardware-independent configuration validation."""

from __future__ import annotations

from ipaddress import IPv4Address
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Pydantic needs this alias at runtime, not just for static typing.
from .profiles import Model  # ruff: ignore[TC001]

_BROADCAST_ADDRESS = IPv4Address("255.255.255.255")


class RobotSettings(BaseModel):
    """Validated RTDE connection and software motion bounds.

    These bounds are not safety-rated. Configure the robot's safety system and
    operate only on a trusted, isolated robot-cell network.
    """

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)

    ip: str = Field(description="Robot controller IPv4 address (not a URL or hostname).")
    model: Model = "ur5e"
    control_mode: Literal["remote", "external_control"] = "remote"
    external_control_port: int = Field(default=50002, ge=1024, le=65535)
    frequency: float = Field(default=500.0, ge=50.0, le=500.0)
    command_timeout: float = Field(default=0.5, ge=0.1, le=5.0)
    max_velocity: float = Field(default=30.0, gt=0.0, le=180.0, description="Maximum setpoint speed in degrees/s.")
    max_relative_target: float = Field(default=10.0, gt=0.0, le=45.0, description="Maximum target offset in degrees.")
    lookahead_time: float = Field(default=0.1, ge=0.03, le=0.2)
    gain: float = Field(default=300.0, ge=100.0, le=2000.0)

    @field_validator("ip")
    @classmethod
    def validate_ip(cls, value: str) -> str:
        """Normalize literal addresses without performing DNS or network I/O.

        Returns:
            The canonical dotted-decimal address.

        Raises:
            ValueError: The address is not a literal unicast IPv4 address.
        """
        address = IPv4Address(value)
        if address.is_unspecified or address.is_multicast or address == _BROADCAST_ADDRESS:
            msg = "Use a unicast controller IPv4 address."
            raise ValueError(msg)
        return str(address)
