# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Declarative, non-executable topic mappings."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:
    from pathlib import Path

PositiveInt = Annotated[int, Field(strict=True, gt=0)]
Name = Annotated[str, Field(min_length=1, pattern=r"^[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)*$")]
Topic = Annotated[str, Field(min_length=2, pattern=r"^/")]
NumericType = Literal[
    "sensor_msgs/msg/JointState",
    "std_msgs/msg/Float32",
    "std_msgs/msg/Float64",
    "std_msgs/msg/Float32MultiArray",
    "std_msgs/msg/Float64MultiArray",
]


class StrictModel(BaseModel):
    """Reject misspelled options and nonfinite numeric configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)


class ResourceLimits(StrictModel):
    """Operational bounds, checked before decoding or frame generation."""

    image_pixels: PositiveInt = 16_000_000
    message_bytes: PositiveInt = 64 * 1024 * 1024
    chunk_bytes: PositiveInt = 256 * 1024 * 1024
    episode_frames: PositiveInt = 100_000
    total_frames: PositiveInt = 1_000_000


class Component(StrictModel):
    """One named dimension selected from a ROS numeric message."""

    name: Name
    topic: Topic
    message_type: NumericType
    field: Literal["position", "velocity", "effort", "data"]
    joint: str | None = None
    index: Annotated[int, Field(strict=True, ge=0)] | None = None
    max_age_ms: PositiveInt | None = None
    conversion: Literal["identity", "rad_to_deg", "deg_to_rad"] = "identity"
    scale: float = 1.0
    offset: float = 0.0

    @model_validator(mode="after")
    def validate_selection(self) -> Self:
        """Require a selector appropriate to the message type.

        Returns:
            The validated component.

        Raises:
            ValueError: Selectors or conversion settings conflict.
        """
        if self.conversion != "identity" and (self.scale != 1.0 or self.offset != 0.0):  # noqa: RUF069 - exact identity sentinels
            msg = "Named conversion cannot be combined with scale/offset"
            raise ValueError(msg)
        if self.message_type == "sensor_msgs/msg/JointState":
            if not self.joint or self.index is not None or self.field == "data":
                msg = "JointState requires joint and position/velocity/effort, without index"
                raise ValueError(msg)
        elif self.field != "data" or self.joint is not None:
            msg = "Numeric commands require data and cannot select a joint"
            raise ValueError(msg)
        elif self.message_type.endswith("MultiArray") != (self.index is not None):
            msg = "Only MultiArray requires an index"
            raise ValueError(msg)
        return self


class Camera(StrictModel):
    """A required RGB camera topic."""

    name: Name
    topic: Topic
    message_type: Literal["sensor_msgs/msg/Image", "sensor_msgs/msg/CompressedImage"]
    max_age_ms: PositiveInt | None = None


class MappingConfig(StrictModel):
    """Versioned mapping from required ROS topics to LeRobot features."""

    schema_version: Annotated[int, Field(strict=True, ge=1, le=1)]
    fps: Annotated[int, Field(strict=True, ge=1, le=1000)]
    task: Annotated[str, Field(min_length=1)]
    robot_type: Annotated[str, Field(min_length=1)]
    max_age_ms: PositiveInt
    state: Annotated[list[Component], Field(min_length=1)]
    action: Annotated[list[Component], Field(min_length=1)]
    cameras: Annotated[list[Camera], Field(min_length=1)]
    limits: ResourceLimits = Field(default_factory=ResourceLimits)

    @model_validator(mode="after")
    def validate_topics(self) -> Self:
        """Require unique feature names and consistent per-topic declarations.

        Returns:
            The validated mapping.

        Raises:
            ValueError: Names or declarations conflict.
        """
        for group in (self.state, self.action, self.cameras):
            if len({entry.name for entry in group}) != len(group):
                msg = "Duplicate output names within a feature"
                raise ValueError(msg)
        topics: dict[str, tuple[str, int]] = {}
        for entry in (*self.state, *self.action, *self.cameras):
            spec = (entry.message_type, entry.max_age_ms or self.max_age_ms)
            if entry.topic in topics and topics[entry.topic] != spec:
                msg = f"Conflicting type/freshness settings for {entry.topic}"
                raise ValueError(msg)
            topics[entry.topic] = spec
        return self

    def topics(self) -> dict[str, tuple[str, int]]:
        """Return required message types and maximum age in nanoseconds."""
        return {
            entry.topic: (entry.message_type, (entry.max_age_ms or self.max_age_ms) * 1_000_000)
            for entry in (*self.state, *self.action, *self.cameras)
        }

    @classmethod
    def from_yaml(cls, path: Path) -> Self:
        """Load an operator-selected YAML mapping, limited to one MiB.

        Returns:
            The validated mapping.

        Raises:
            ValueError: The document exceeds the size limit or violates the schema.
        """
        with path.open("rb") as stream:
            content = stream.read(1024 * 1024 + 1)
        if len(content) > 1024 * 1024:
            msg = "Mapping exceeds one MiB"
            raise ValueError(msg)
        return cls.model_validate(yaml.safe_load(content))
