# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Model-specific Studio entries backed by one standalone Runtime driver."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypedDict, cast

from physicalai_studio_plugin import (
    CatalogRobot,
    CatalogRobotFactory,
    RobotAsset,
    RobotCatalogDefinition,
    robot_field_ui,
    robot_payload_ui,
)
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .driver import UniversalRobot
from .profiles import JOINT_NAMES, SUPPORTED_MODELS, Model, get_asset_root
from .settings import RobotSettings

if TYPE_CHECKING:
    from typing import Self

    from physicalai_studio_plugin.catalog import RobotCatalogRegistry
    from pydantic import JsonValue


class _DriverArgs(TypedDict):
    ip: str
    control_mode: Literal["remote", "external_control"]
    external_control_port: int
    frequency: float
    command_timeout: float
    max_velocity: float
    max_relative_target: float
    lookahead_time: float
    gain: float


# SDK TypedDicts describe valid JSON but are not assignable to Pydantic's recursive dict alias.
_ADVANCED_SCHEMA = cast("dict[str, JsonValue]", robot_field_ui({"advanced_configuration": True}))


class URPayload(BaseModel):
    """Persisted network connection and optional RTDE tuning, not model selection."""

    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
        json_schema_extra=cast(
            "dict[str, JsonValue]",
            robot_payload_ui([
                {
                    "kind": "info",
                    "variant": "warning",
                    "text": "Follower-only, six joints, no gripper. Configure safety on the teach pendant. "
                    "Connection starts a control script but does not power on, release brakes, or home. "
                    "Use an isolated robot-cell network; RTDE is not authenticated.",
                },
                {"kind": "ip_address", "name": "connection_string", "label": "Robot IPv4 address"},
                *[
                    {"kind": "field", "name": name}
                    for name in RobotSettings.model_fields
                    if name not in {"ip", "model"}
                ],
            ]),
        ),
    )

    connection_string: str = Field(
        title="Robot IPv4 address", description="Literal IPv4 address; no DNS lookup or scan."
    )
    control_mode: Literal["remote", "external_control"] = Field(
        default="remote",
        title="Control mode",
        description="Remote: PolyScope 5 remote mode. External Control: install the URCap and start it on the pendant.",
    )
    external_control_port: int = Field(
        default=50002,
        ge=1024,
        le=65535,
        title="External Control port",
        json_schema_extra=_ADVANCED_SCHEMA,
    )
    frequency: float = Field(
        default=500.0,
        ge=50.0,
        le=500.0,
        title="Servo frequency (Hz)",
        json_schema_extra=_ADVANCED_SCHEMA,
    )
    command_timeout: float = Field(
        default=0.5,
        ge=0.1,
        le=5.0,
        title="Command timeout (seconds)",
        description="Stop and latch a fault if commands or RTDE state become stale.",
        json_schema_extra=_ADVANCED_SCHEMA,
    )
    max_velocity: float = Field(
        default=30.0,
        gt=0.0,
        le=180.0,
        title="Setpoint speed limit (degrees/s)",
        json_schema_extra=_ADVANCED_SCHEMA,
    )
    max_relative_target: float = Field(
        default=10.0,
        gt=0.0,
        le=45.0,
        title="Maximum target offset (degrees)",
        json_schema_extra=_ADVANCED_SCHEMA,
    )
    lookahead_time: float = Field(
        default=0.1,
        ge=0.03,
        le=0.2,
        title="Servo lookahead (seconds)",
        json_schema_extra=_ADVANCED_SCHEMA,
    )
    gain: float = Field(
        default=300.0,
        ge=100.0,
        le=2000.0,
        title="Servo gain",
        json_schema_extra=_ADVANCED_SCHEMA,
    )

    @model_validator(mode="after")
    def validate_settings(self) -> Self:
        """Keep direct Runtime construction and Studio network validation identical.

        Returns:
            This payload with its canonical controller address.
        """
        self.connection_string = RobotSettings.model_validate(self.driver_args()).ip
        return self

    def driver_args(self) -> _DriverArgs:
        """Return Runtime keyword arguments without any hardware access."""
        return {
            "ip": self.connection_string,
            "control_mode": self.control_mode,
            "external_control_port": self.external_control_port,
            "frequency": self.frequency,
            "command_timeout": self.command_timeout,
            "max_velocity": self.max_velocity,
            "max_relative_target": self.max_relative_target,
            "lookahead_time": self.lookahead_time,
            "gain": self.gain,
        }


_TYPE_TO_MODEL: dict[str, Model] = {
    f"UniversalRobots_{model.replace('ur', 'UR')}_Follower": model for model in SUPPORTED_MODELS
}


async def _build_robot(  # ruff: ignore[RUF029] -- the Studio builder contract is asynchronous.
    robot: CatalogRobot[URPayload],
    _factory: CatalogRobotFactory,
) -> UniversalRobot:
    if not isinstance(robot.payload, URPayload):
        msg = "Expected a validated URPayload."
        raise TypeError(msg)
    if robot.type not in _TYPE_TO_MODEL:
        msg = f"Unknown Universal Robots catalog type: {robot.type}"
        raise ValueError(msg)
    return UniversalRobot(model=_TYPE_TO_MODEL[robot.type], **robot.payload.driver_args())


def register_physicalai_studio_plugin(registry: RobotCatalogRegistry) -> None:
    """Register six stable follower types, each with its own official URDF."""
    for catalog_type, model in _TYPE_TO_MODEL.items():
        registry.register_robot(
            RobotCatalogDefinition(
                type=catalog_type,
                display_name=f"Universal Robots {model.replace('ur', 'UR')} Follower",
                role="follower",
                robot_payload=URPayload,
                robot_builder=_build_robot,
                asset=RobotAsset(
                    urdf_relative_path=Path(f"{model}.urdf"),
                    packages={"ur_description": Path()},
                    joint_map={f"{name}.pos": [name] for name in JOINT_NAMES},
                    root_resolver=get_asset_root,
                ),
            )
        )
