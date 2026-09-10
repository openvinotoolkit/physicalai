# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Studio catalog plugin for Star Arm 102 robots."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal, Self

from physicalai_studio_plugin import (
    CatalogRobotFactory,
    PayloadContainer,
    PortScanner,
    RobotAdapterOptions,
    RobotAsset,
    RobotCatalogDefinition,
    RobotProbe,
    SerialPortInfo,
    robot_field_ui,
    robot_payload_ui,
)
from pydantic import BaseModel, ConfigDict, Field, model_validator

from physicalai_stararm_plugin import StarArm102FLFollower, StarArm102HDLeader, StarArm102LDLeader, get_urdf_path

if TYPE_CHECKING:
    from typing import Protocol

    from physicalai.robot.interface import Robot as PhysicalAIRobot

    class _RobotCatalogRegistry(Protocol):
        def register_robot(self, definition: RobotCatalogDefinition) -> None: ...


_STAR_ARM_102_TO_URDF: dict[str, list[str]] = {
    "shoulder_pan.pos": ["joint1"],
    "shoulder_lift.pos": ["joint2"],
    "elbow_flex.pos": ["joint3"],
    "wrist_flex.pos": ["joint4"],
    "wrist_yaw.pos": ["joint5"],
    "wrist_roll.pos": ["joint6"],
    "gripper.pos": ["joint7_left", "joint7_right"],
}


def _get_stararm_urdf_root() -> Path:
    return get_urdf_path()


_STAR_ARM_102_LD_ASSET = RobotAsset(
    urdf_relative_path=Path("stararm102/urdf/stararm102_ld_description.urdf"),
    packages={"stararm102": Path("stararm102")},
    joint_map=_STAR_ARM_102_TO_URDF,
    root_resolver=_get_stararm_urdf_root,
)

_STAR_ARM_102_HD_ASSET = RobotAsset(
    urdf_relative_path=Path("stararm102/urdf/stararm102_hd_description.urdf"),
    packages={"stararm102": Path("stararm102")},
    joint_map=_STAR_ARM_102_TO_URDF,
    root_resolver=_get_stararm_urdf_root,
)

_STAR_ARM_102_FL_ASSET = RobotAsset(
    urdf_relative_path=Path("stararm102/urdf/stararm102_fl_description.urdf"),
    packages={"stararm102": Path("stararm102")},
    joint_map=_STAR_ARM_102_TO_URDF,
    root_resolver=_get_stararm_urdf_root,
)


class StarArm102LDPayload(BaseModel):
    """Connection payload for a Star Arm 102-LD leader arm."""

    connection_string: str = ""
    serial_number: str = ""
    baudrate: int = Field(  # pyrefly: ignore [no-matching-overload]
        default=1_000_000,
        json_schema_extra=robot_field_ui({"advanced_configuration": True}),
    )
    unlock_on_connect: bool = Field(  # pyrefly: ignore [no-matching-overload]
        default=True,
        json_schema_extra=robot_field_ui({"advanced_configuration": True}),
    )
    reset_multi_turn_on_connect: bool = Field(  # pyrefly: ignore [no-matching-overload]
        default=True,
        json_schema_extra=robot_field_ui({"advanced_configuration": True}),
    )
    zero_on_connect: bool = Field(  # pyrefly: ignore [no-matching-overload]
        default=False,
        json_schema_extra=robot_field_ui({"advanced_configuration": True}),
    )

    model_config = ConfigDict(
        json_schema_extra=robot_payload_ui(  # pyrefly: ignore [bad-argument-type]
            [
                {
                    "kind": "connection",
                    "label": "Select robot",
                    "device_discovery": True,
                    "bind": {"connection": "connection_string", "serial_number": "serial_number"},
                },
            ],
        ),
    )

    @model_validator(mode="after")
    def _require_connection_identifier(self) -> Self:
        if not self.connection_string and not self.serial_number:
            msg = "At least one of connection_string or serial_number must be provided"
            raise ValueError(msg)
        return self


class StarArm102FLPayload(BaseModel):
    """Connection payload for a Star Arm 102-FL follower arm."""

    connection_string: str = ""
    serial_number: str = ""
    baudrate: int = Field(  # pyrefly: ignore [no-matching-overload]
        default=1_000_000,
        json_schema_extra=robot_field_ui({"advanced_configuration": True}),
    )
    unlock_on_connect: bool = Field(  # pyrefly: ignore [no-matching-overload]
        default=True,
        json_schema_extra=robot_field_ui({"advanced_configuration": True}),
    )
    reset_multi_turn_on_connect: bool = Field(  # pyrefly: ignore [no-matching-overload]
        default=True,
        json_schema_extra=robot_field_ui({"advanced_configuration": True}),
    )
    zero_on_connect: bool = Field(  # pyrefly: ignore [no-matching-overload]
        default=False,
        json_schema_extra=robot_field_ui({"advanced_configuration": True}),
    )
    command_interval_ms: int = Field(  # pyrefly: ignore [no-matching-overload]
        default=10,
        json_schema_extra=robot_field_ui({"advanced_configuration": True}),
    )

    model_config = ConfigDict(
        json_schema_extra=robot_payload_ui(  # pyrefly: ignore [bad-argument-type]
            [
                {
                    "kind": "connection",
                    "label": "Select robot",
                    "device_discovery": True,
                    "bind": {"connection": "connection_string", "serial_number": "serial_number"},
                },
            ],
        ),
    )

    @model_validator(mode="after")
    def _require_connection_identifier(self) -> Self:
        if not self.connection_string and not self.serial_number:
            msg = "At least one of connection_string or serial_number must be provided"
            raise ValueError(msg)
        return self


class StarArm102HDPayload(BaseModel):
    """Connection payload for a Star Arm 102-HD leader arm."""

    connection_string: str = ""
    serial_number: str = ""
    baudrate: int = Field(  # pyrefly: ignore [no-matching-overload]
        default=1_000_000,
        json_schema_extra=robot_field_ui({"advanced_configuration": True}),
    )
    unlock_on_connect: bool = Field(  # pyrefly: ignore [no-matching-overload]
        default=True,
        json_schema_extra=robot_field_ui({"advanced_configuration": True}),
    )
    reset_multi_turn_on_connect: bool = Field(  # pyrefly: ignore [no-matching-overload]
        default=True,
        json_schema_extra=robot_field_ui({"advanced_configuration": True}),
    )
    zero_on_connect: bool = Field(  # pyrefly: ignore [no-matching-overload]
        default=False,
        json_schema_extra=robot_field_ui({"advanced_configuration": True}),
    )
    control_mode: Literal["passive", "assist"] = Field(  # pyrefly: ignore [no-matching-overload]
        default="passive",
        json_schema_extra=robot_field_ui({"advanced_configuration": True}),
        description="HD leader mode: passive (read-only) or assist (accepts actions and hold).",
    )
    command_interval_ms: int = Field(  # pyrefly: ignore [no-matching-overload]
        default=10,
        json_schema_extra=robot_field_ui({"advanced_configuration": True}),
    )

    model_config = ConfigDict(
        json_schema_extra=robot_payload_ui(  # pyrefly: ignore [bad-argument-type]
            [
                {
                    "kind": "connection",
                    "label": "Select robot",
                    "device_discovery": True,
                    "bind": {"connection": "connection_string", "serial_number": "serial_number"},
                },
            ],
        ),
    )

    @model_validator(mode="after")
    def _require_connection_identifier(self) -> Self:
        if not self.connection_string and not self.serial_number:
            msg = "At least one of connection_string or serial_number must be provided"
            raise ValueError(msg)
        return self


StarArmPayload = StarArm102LDPayload | StarArm102HDPayload | StarArm102FLPayload


class StarArmProbe(RobotProbe[StarArmPayload]):
    """Probe implementation for Star Arm serial devices."""

    async def discover(self, manager: PortScanner) -> list[SerialPortInfo]:
        """Discover available Star Arm devices via the active port scanner.

        Returns:
            list[SerialPortInfo]: Detected serial devices.
        """
        _ = self
        await manager.find_robots()
        return manager.robots

    async def identify(
        self,
        payload: StarArmPayload,
        manager: PortScanner | None = None,
        joint: str | None = None,
    ) -> None:
        """Request a visual identify action on a specific joint, if supported."""
        _ = self, payload, manager, joint

    async def is_online(self, payload: StarArmPayload, manager: PortScanner | None = None) -> bool:
        """Report whether a Star Arm device is currently online.

        Returns:
            bool: ``True`` if the device is present in discovered serial ports.
        """
        _ = self
        if manager is None:
            return False

        ports = manager.robots
        if payload.connection_string and any(port.connection_string == payload.connection_string for port in ports):
            return True

        return bool(payload.serial_number) and any(port.serial_number == payload.serial_number for port in ports)


_STAR_ARM_PROBE = StarArmProbe()


async def _build_stararm_102_hd_driver(
    robot: PayloadContainer[StarArm102HDPayload],
    factory: CatalogRobotFactory,
) -> PhysicalAIRobot:
    raw = robot.payload
    if isinstance(raw, BaseModel) and type(raw) is not StarArm102HDPayload:
        raw = raw.model_dump()
    validated = raw if isinstance(raw, StarArm102HDPayload) else StarArm102HDPayload.model_validate(raw)
    connection_string = validated.connection_string or None
    serial_number = validated.serial_number or None
    port = await factory.find_port(
        SerialPortInfo(
            connection_string=connection_string,
            serial_number=serial_number,
        ),
    )

    if port is None:
        msg = f"Robot not found: {serial_number or connection_string}"
        raise RuntimeError(msg)
    return StarArm102HDLeader(
        port=port,
        baudrate=validated.baudrate,
        unlock_on_connect=validated.unlock_on_connect,
        reset_multi_turn_on_connect=validated.reset_multi_turn_on_connect,
        zero_on_connect=validated.zero_on_connect,
        control_mode=validated.control_mode,
        command_interval_ms=validated.command_interval_ms,
    )


async def _build_stararm_102_fl_driver(
    robot: PayloadContainer[StarArm102FLPayload],
    factory: CatalogRobotFactory,
) -> PhysicalAIRobot:
    raw = robot.payload
    if isinstance(raw, BaseModel) and type(raw) is not StarArm102FLPayload:
        raw = raw.model_dump()
    validated = raw if isinstance(raw, StarArm102FLPayload) else StarArm102FLPayload.model_validate(raw)
    connection_string = validated.connection_string or None
    serial_number = validated.serial_number or None
    port = await factory.find_port(
        SerialPortInfo(
            connection_string=connection_string,
            serial_number=serial_number,
        ),
    )

    if port is None:
        msg = f"Robot not found: {serial_number or connection_string}"
        raise RuntimeError(msg)
    return StarArm102FLFollower(
        port=port,
        baudrate=validated.baudrate,
        unlock_on_connect=validated.unlock_on_connect,
        reset_multi_turn_on_connect=validated.reset_multi_turn_on_connect,
        zero_on_connect=validated.zero_on_connect,
        command_interval_ms=validated.command_interval_ms,
    )


async def _build_stararm_102_ld_driver(
    robot: PayloadContainer[StarArm102LDPayload],
    factory: CatalogRobotFactory,
) -> PhysicalAIRobot:
    raw = robot.payload
    if isinstance(raw, BaseModel) and type(raw) is not StarArm102LDPayload:
        raw = raw.model_dump()
    validated = raw if isinstance(raw, StarArm102LDPayload) else StarArm102LDPayload.model_validate(raw)
    connection_string = validated.connection_string or None
    serial_number = validated.serial_number or None
    port = await factory.find_port(
        SerialPortInfo(
            connection_string=connection_string,
            serial_number=serial_number,
        ),
    )

    if port is None:
        msg = f"Robot not found: {serial_number or connection_string}"
        raise RuntimeError(msg)
    return StarArm102LDLeader(
        port=port,
        baudrate=validated.baudrate,
        unlock_on_connect=validated.unlock_on_connect,
        reset_multi_turn_on_connect=validated.reset_multi_turn_on_connect,
        zero_on_connect=validated.zero_on_connect,
    )


def _definitions() -> list[RobotCatalogDefinition]:
    return [
        RobotCatalogDefinition(
            type="StarArm_102_LD_Leader",
            display_name="Star Arm 102-LD Leader",
            role="leader",
            robot_builder=_build_stararm_102_ld_driver,
            robot_payload=StarArm102LDPayload,
            asset=_STAR_ARM_102_LD_ASSET,
            adapter_options=RobotAdapterOptions(include_velocities=False, external_effort_gain=None),
            probe=_STAR_ARM_PROBE,
        ),
        RobotCatalogDefinition(
            type="StarArm_102_HD_Leader",
            display_name="Star Arm 102-HD Leader",
            role="leader",
            robot_builder=_build_stararm_102_hd_driver,
            robot_payload=StarArm102HDPayload,
            asset=_STAR_ARM_102_HD_ASSET,
            adapter_options=RobotAdapterOptions(include_velocities=False, external_effort_gain=None),
            probe=_STAR_ARM_PROBE,
        ),
        RobotCatalogDefinition(
            type="StarArm_102_FL_Follower",
            display_name="Star Arm 102-FL Follower",
            role="follower",
            robot_builder=_build_stararm_102_fl_driver,
            robot_payload=StarArm102FLPayload,
            asset=_STAR_ARM_102_FL_ASSET,
            adapter_options=RobotAdapterOptions(include_velocities=True, external_effort_gain=None),
            probe=_STAR_ARM_PROBE,
        ),
    ]


def _assert_payload_model_resolvable(model: type[BaseModel]) -> None:
    model.model_rebuild(_types_namespace=globals(), raise_errors=True)


def register_physicalai_studio_plugin(registry: _RobotCatalogRegistry) -> None:
    """Register Star Arm catalog entries with the Physical AI Studio registry."""
    for definition in _definitions():
        payload_model = definition.robot_payload
        if isinstance(payload_model, type) and issubclass(payload_model, BaseModel):
            _assert_payload_model_resolvable(payload_model)
        registry.register_robot(definition)
