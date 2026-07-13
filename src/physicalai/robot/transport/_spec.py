# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Serializable robot construction spec for transport endpoints.

The owner subprocess must construct the robot itself — a live serial/socket
handle cannot cross a process boundary. All spec kwargs must therefore be
JSON-serializable: SO-101 ``calibration`` is passed as a file path, ``role``
as a string.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from physicalai.robot.interface import Robot

_DEFAULT_RATE_HZ: dict[str, float] = {
    # Serial bus (1 Mbaud) — read+write comfortably fits at 100 Hz.
    "so101": 100.0,
    # TCP to the arm controller — higher realistic ceiling.
    "widowxai": 200.0,
}

_FALLBACK_RATE_HZ = 100.0


def default_rate_hz(robot_type: str) -> float:
    """Robot-appropriate default owner loop rate.

    Serial (SO-101) and TCP (WidowXAI) transports have different realistic
    ceilings, so there is no single global constant.

    Args:
        robot_type: Logical robot type.

    Returns:
        Loop rate in Hz.
    """
    return _DEFAULT_RATE_HZ.get(robot_type, _FALLBACK_RATE_HZ)


@dataclass(frozen=True)
class RobotSpec:
    """Config payload describing how to construct a robot driver.

    Attributes:
        robot_type: Logical robot type (``"so101"`` or ``"widowxai"``).
        robot_kwargs: JSON-serializable keyword arguments forwarded to the
            driver constructor.
    """

    robot_type: str
    robot_kwargs: dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dictionary.

        Returns:
            Dictionary with ``robot_type`` and ``robot_kwargs`` keys.
        """
        return {"robot_type": self.robot_type, "robot_kwargs": self.robot_kwargs}

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> RobotSpec:
        """Deserialize from a JSON dictionary.

        Args:
            data: Dictionary with ``robot_type`` and optional
                ``robot_kwargs`` keys.

        Returns:
            A new :class:`RobotSpec` instance.
        """
        return cls(
            robot_type=data["robot_type"],
            robot_kwargs=data.get("robot_kwargs", {}),
        )

    def build(self) -> Robot:
        """Instantiate the robot driver described by this spec.

        Returns:
            A new, not-yet-connected driver instance.

        Raises:
            ValueError: If ``robot_type`` is unknown.
        """
        if self.robot_type == "so101":
            from physicalai.robot.so101 import SO101  # noqa: PLC0415

            return SO101(**self.robot_kwargs)
        if self.robot_type == "widowxai":
            from physicalai.robot.trossen import WidowXAI  # noqa: PLC0415

            return WidowXAI(**self.robot_kwargs)
        msg = f"unknown robot_type {self.robot_type!r}; expected 'so101' or 'widowxai'"
        raise ValueError(msg)
