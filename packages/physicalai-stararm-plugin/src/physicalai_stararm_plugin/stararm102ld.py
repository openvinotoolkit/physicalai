# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Star Arm 102-LD leader driver."""

from __future__ import annotations

from typing import ClassVar, Literal

from physicalai.config import export_config
from physicalai_stararm_plugin.stararm102hd import StarArm102HDLeader


@export_config
class StarArm102LDLeader(StarArm102HDLeader):
    """FashionStar UART leader arm driver for Star Arm 102-LD."""

    MODEL_NAME: ClassVar[str] = "Star Arm 102-LD"
    DEVICE_PREFIX: ClassVar[str] = "stararm102-ld"

    def __init__(
        self,
        port: str = "/dev/ttyUSB0",
        *,
        baudrate: int = 1_000_000,
        unlock_on_connect: bool = True,
        reset_multi_turn_on_connect: bool = True,
        zero_on_connect: bool = False,
        follower_profile: Literal["b601"] | None = None,
    ) -> None:
        """Initialize a passive LD leader with optional follower-aware observations.

        Args:
            port: UART serial device connected to the leader.
            baudrate: UART communication speed in bits per second.
            unlock_on_connect: Unlock every servo when connecting.
            reset_multi_turn_on_connect: Reset each servo's accumulated turn count when connecting.
            zero_on_connect: Store the current servo positions as their origins when connecting.
            follower_profile: Optional follower whose reachable ranges constrain observations. The
                ``"b601"`` profile clips the gripper to 45 degrees without changing signs or units;
                ``None`` preserves native Star Arm ranges.

        Raises:
            ValueError: If the baud rate or follower profile is invalid.
        """  # noqa: DOC502 — validation is delegated to the inherited constructor.
        super().__init__(
            port=port,
            baudrate=baudrate,
            unlock_on_connect=unlock_on_connect,
            reset_multi_turn_on_connect=reset_multi_turn_on_connect,
            zero_on_connect=zero_on_connect,
            control_mode="passive",
            command_interval_ms=10,
            follower_profile=follower_profile,
        )
