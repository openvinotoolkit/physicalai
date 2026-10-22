# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Optional, plugin-owned joint-position handover capabilities."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator

_PositiveFinite = Annotated[float, Field(gt=0, allow_inf_nan=False, strict=True)]


class HandoverCapabilities(BaseModel):
    """Declare coordinates and limits without exposing calibration data.

    Equal ``position_space`` identifiers promise the same units, directions,
    zero convention, and gripper mapping for equal ordered joint names. Plugins
    must use different identifiers for incompatible conventions. Limits are in
    those coordinates, one value per joint. ``max_position_step`` bounds extra
    follower catch-up per tick, not leader motion or absolute speed.
    ``supports_alignment`` additionally
    promises position commands and acknowledged ``set_torque(enabled=...)``.
    Buffered drivers must implement ``get_fresh_observation()`` to request a new
    capture; unbuffered drivers can use their synchronous ``get_observation()``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    position_space: str = Field(min_length=1, pattern=r"\S", strict=True)
    position_tolerance: tuple[_PositiveFinite, ...] = Field(min_length=1)
    max_position_step: tuple[_PositiveFinite, ...] = Field(min_length=1)
    supports_alignment: bool = Field(default=False, strict=True)

    @model_validator(mode="after")
    def validate_limits(self) -> HandoverCapabilities:
        """Require matching nonempty per-joint limits.

        Returns:
            The validated capabilities.

        Raises:
            ValueError: If the limit vectors have different lengths.
        """
        if len(self.position_tolerance) != len(self.max_position_step):
            msg = "Handover limit vectors must have the same length"
            raise ValueError(msg)
        return self


def validate_handover_capabilities(value: object, *, num_joints: int) -> HandoverCapabilities | None:
    """Validate optional driver declarations or transport metadata.

    Args:
        value: A capability object, its JSON-compatible mapping, or None.
        num_joints: Number of ordered joints declared by the robot.

    Returns:
        Validated capabilities, or None for an undeclared coordinate space.

    Raises:
        ValueError: If capabilities are malformed or do not match the robot.
    """
    if value is None:
        return None
    capabilities = HandoverCapabilities.model_validate(value, from_attributes=True)
    if len(capabilities.position_tolerance) != num_joints:
        msg = "Handover limits must match the robot's joint count"
        raise ValueError(msg)
    return capabilities
