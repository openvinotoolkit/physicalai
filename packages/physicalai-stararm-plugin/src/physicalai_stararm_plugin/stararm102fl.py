# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Star Arm 102-FL follower driver."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from physicalai.config import export_config
from physicalai_stararm_plugin._stararm102 import _StarArm102, _StarArm102Observation


@dataclass
class StarArm102FLFollowerObservation(_StarArm102Observation):
    """Observation data for the Star Arm 102-FL follower."""


@export_config
class StarArm102FLFollower(_StarArm102):
    """FashionStar UART follower arm driver with position control."""

    MODEL_NAME: ClassVar[str] = "Star Arm 102-FL"
    DEVICE_PREFIX: ClassVar[str] = "stararm102-fl"
    OBSERVATION_CLASS: ClassVar[type[_StarArm102Observation]] = StarArm102FLFollowerObservation

    def __init__(
        self,
        port: str = "/dev/ttyUSB0",
        *,
        baudrate: int = 1_000_000,
        unlock_on_connect: bool = True,
        reset_multi_turn_on_connect: bool = True,
        zero_on_connect: bool = False,
        command_interval_ms: int = 10,
    ) -> None:
        """Initialize an FL follower that reports and accepts native Star Arm joint positions.

        Args:
            port: UART serial device connected to the follower.
            baudrate: UART communication speed in bits per second.
            unlock_on_connect: Unlock every servo when connecting.
            reset_multi_turn_on_connect: Reset each servo's accumulated turn count when connecting.
            zero_on_connect: Store the current servo positions as their origins when connecting.
            command_interval_ms: Minimum duration passed to servo position commands.

        Raises:
            ValueError: If the baud rate is not positive or the command interval is negative.
        """  # noqa: DOC502 — validation is delegated to the inherited constructor.
        super().__init__(
            port=port,
            baudrate=baudrate,
            unlock_on_connect=unlock_on_connect,
            reset_multi_turn_on_connect=reset_multi_turn_on_connect,
            zero_on_connect=zero_on_connect,
            command_interval_ms=command_interval_ms,
        )
