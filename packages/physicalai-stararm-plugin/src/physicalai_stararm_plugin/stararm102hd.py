# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Star Arm 102-HD leader driver."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal

import numpy as np

from physicalai.config import export_config
from physicalai_stararm_plugin._stararm102 import _StarArm102, _StarArm102Observation
from physicalai_stararm_plugin.constants import STAR_ARM_102_B601_FOLLOWER_RANGES_DEG


@dataclass
class StarArm102HDLeaderObservation(_StarArm102Observation):
    """Observation data for the Star Arm 102-HD leader."""


@export_config
class StarArm102HDLeader(_StarArm102):
    """FashionStar UART leader arm driver with passive and assist modes."""

    MODEL_NAME: ClassVar[str] = "Star Arm 102-HD"
    DEVICE_PREFIX: ClassVar[str] = "stararm102-hd"
    OBSERVATION_CLASS: ClassVar[type[_StarArm102Observation]] = StarArm102HDLeaderObservation
    VALID_CONTROL_MODES: ClassVar[frozenset[str]] = frozenset({"passive", "assist"})
    VALID_FOLLOWER_PROFILES: ClassVar[frozenset[str]] = frozenset({"b601"})

    def __init__(
        self,
        port: str = "/dev/ttyUSB0",
        *,
        baudrate: int = 1_000_000,
        unlock_on_connect: bool = True,
        reset_multi_turn_on_connect: bool = True,
        zero_on_connect: bool = False,
        control_mode: Literal["passive", "assist"] = "passive",
        command_interval_ms: int = 10,
        follower_profile: Literal["b601"] | None = None,
    ) -> None:
        """Initialize an HD leader with optional assist control and follower-aware observations.

        Args:
            port: UART serial device connected to the leader.
            baudrate: UART communication speed in bits per second.
            unlock_on_connect: Unlock every servo when connecting.
            reset_multi_turn_on_connect: Reset each servo's accumulated turn count when connecting.
            zero_on_connect: Store the current servo positions as their origins when connecting.
            control_mode: ``"passive"`` for read-only guidance or ``"assist"`` to accept commands.
            command_interval_ms: Minimum duration passed to servo position commands.
            follower_profile: Optional follower whose reachable ranges constrain observations. The
                ``"b601"`` profile clips the gripper to 45 degrees without changing signs or units;
                ``None`` preserves native Star Arm ranges.

        Raises:
            ValueError: If the baud rate, command interval, control mode, or follower profile is invalid.
        """
        if control_mode not in self.VALID_CONTROL_MODES:
            msg = f"Invalid control_mode {control_mode!r}. Must be one of {sorted(self.VALID_CONTROL_MODES)}."
            raise ValueError(msg)
        if follower_profile is not None and follower_profile not in self.VALID_FOLLOWER_PROFILES:
            msg = (
                f"Invalid follower_profile {follower_profile!r}. "
                f"Must be one of {sorted(self.VALID_FOLLOWER_PROFILES)} or None."
            )
            raise ValueError(msg)
        super().__init__(
            port=port,
            baudrate=baudrate,
            unlock_on_connect=unlock_on_connect,
            reset_multi_turn_on_connect=reset_multi_turn_on_connect,
            zero_on_connect=zero_on_connect,
            command_interval_ms=command_interval_ms,
        )
        self._control_mode = control_mode
        self._follower_profile = follower_profile
        self._holding = False

    @property
    def control_mode(self) -> Literal["passive", "assist"]:
        """Current control mode for this leader."""
        return self._control_mode

    @property
    def follower_profile(self) -> Literal["b601"] | None:
        """Follower whose reachable ranges constrain leader observations."""
        return self._follower_profile

    @property
    def is_holding(self) -> bool:
        """Whether hold mode is currently active."""
        return self._holding

    def disconnect(self) -> None:
        """Close the UART bus and end any active hold."""
        self._holding = False
        super().disconnect()

    def disable_torque(self) -> None:
        """Unlock every servo so the arm can be moved by hand, ending any hold."""
        super().disable_torque()
        self._holding = False

    def _observation_positions(self, positions: np.ndarray) -> np.ndarray:
        profiled = positions.copy()
        if self._follower_profile == "b601":
            for i, name in enumerate(self.JOINT_ORDER):
                follower_range = STAR_ARM_102_B601_FOLLOWER_RANGES_DEG.get(name)
                if follower_range is not None:
                    profiled[i] = float(np.clip(profiled[i], *follower_range))
        return profiled

    def send_action(self, action: np.ndarray, *, goal_time: float = 0.1) -> None:
        """Optionally command the HD leader in assist mode."""
        if self._control_mode != "assist":
            return
        self._send_action_internal(action, goal_time=goal_time)

    def hold_position(self, *, goal_time: float = 0.2) -> None:
        """Capture the native pose and command the HD leader to hold it.

        Raises:
            ConnectionError: If no native joint sample is available.
        """
        self.get_observation()
        if self._last_positions is None:  # pragma: no cover - get_observation either populates or raises
            msg = "No native joint sample is available to hold."
            raise ConnectionError(msg)
        self._send_action_internal(self._last_positions.copy(), goal_time=goal_time)
        self._holding = True

    def release_hold(self) -> None:
        """Release hold mode and return to manual guidance."""
        self._holding = False
