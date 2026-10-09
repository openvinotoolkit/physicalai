# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# ruff: file-ignore[undocumented-public-method, f-string-in-exception, import-outside-top-level, no-self-use, raise-vanilla-args, too-many-statements-in-try-clause]

"""Studio catalog plugin for single-arm and bimanual SO-101 robots."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from loguru import logger
from physicalai_studio_plugin import (
    CatalogRobot,
    CatalogRobotFactory,
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

from physicalai_so101_plugin import SO101, SO101Calibration, SO101JointCalibration, get_urdf_path
from physicalai_so101_plugin.constants import TICKS_PER_REVOLUTION

if TYPE_CHECKING:
    from collections.abc import Sequence


class _RobotCatalogRegistry(Protocol):
    def register_robot(self, definition: RobotCatalogDefinition) -> None: ...


class SO101RobotPayload(BaseModel):
    """Connection configuration for SO-101 serial robots."""

    connection_string: str = Field(
        default="",
        description="Serial port path; leave empty to auto-discover via serial_number",
    )
    serial_number: str = Field(
        default="",
        description="USB serial number of the robot (when available)",
    )
    calibration: dict[str, SO101JointCalibration] | None = Field(
        default=None,
        description=(
            "Provide SO101 calibration values. Studio uses these values as-is: it does not overwrite "
            "calibration on the control board and it skips the guided manual calibration flow."
        ),
        json_schema_extra=robot_field_ui({
            "advanced_configuration": True,
            "info": {
                "title": "Calibration values",
                "description": (
                    "Upload a calibration JSON exported for this SO101. If provided, Studio uses these "
                    "values as-is, does not overwrite control-board calibration, and skips guided manual "
                    "calibration."
                ),
                "variant": "help",
            },
        }),
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {"connection_string": "", "serial_number": "SO101-2024-001", "calibration": None},
            **robot_payload_ui([
                {
                    "kind": "connection",
                    "label": "Connection",
                    "device_discovery": True,
                    "identify": True,
                    "manual_entry": True,
                    "bind": {"connection": "connection_string", "serial_number": "serial_number"},
                },
                {"kind": "calibration", "name": "calibration"},
            ]),
        },
    )

    @model_validator(mode="after")
    def validate_identifier(self) -> SO101RobotPayload:
        """Require a serial number or an explicit serial-device path.

        Returns:
            The validated payload.

        Raises:
            ValueError: If neither a serial number nor a device path is supplied.
        """
        if not self.connection_string and not self.serial_number:
            msg = "Either serial_number or connection_string is required for SO101 robots"
            raise ValueError(msg)
        return self


_SO101_TO_URDF = {
    "shoulder_pan.pos": ["shoulder_pan"],
    "shoulder_lift.pos": ["shoulder_lift"],
    "elbow_flex.pos": ["elbow_flex"],
    "wrist_flex.pos": ["wrist_flex"],
    "wrist_roll.pos": ["wrist_roll"],
    "gripper.pos": ["gripper"],
}

_SO101_FOLLOWER_ASSET = RobotAsset(
    urdf_relative_path=Path("so101-follower/so101_follower.urdf"),
    packages={"SO101": Path("so101-follower")},
    joint_map=_SO101_TO_URDF,
    root_resolver=get_urdf_path,
)

_SO101_LEADER_ASSET = RobotAsset(
    urdf_relative_path=Path("so101-leader/urdf/so101_leader.urdf"),
    packages={"SO101": Path("so101-leader/urdf")},
    joint_map=_SO101_TO_URDF,
    root_resolver=get_urdf_path,
)


async def _build_so101_driver(robot: CatalogRobot[SO101RobotPayload], factory: CatalogRobotFactory) -> SO101:
    payload = robot.payload
    if not isinstance(payload, SO101RobotPayload):
        raise TypeError(f"Expected SO101RobotPayload, got {type(payload).__name__}")
    port = await factory.find_port(
        SerialPortInfo(
            connection_string=payload.connection_string or None,
            serial_number=payload.serial_number or None,
        )
    )
    if port is None:
        resource_key = payload.serial_number or payload.connection_string
        raise ValueError(f"Could not resolve a serial port for {resource_key}")

    role = "follower" if robot.type == "SO101_Follower" else "leader"
    if payload.calibration is None:
        logger.warning(
            "SO101 {} has no calibration; building in raw-ticks mode. "
            "Positions are not normalized and must not be used for policy inference.",
            robot.type,
        )
        return SO101.uncalibrated(port=port, role=role)

    return SO101(
        port=port,
        calibration=SO101Calibration(joints=payload.calibration),
        role=role,
        unit="normalized",
    )


def _resolve_serial_port(discovered: Sequence[SerialPortInfo], target: SerialPortInfo) -> str | None:
    if target.serial_number is not None:
        for serial_port in discovered:
            if serial_port.serial_number == target.serial_number:
                return serial_port.connection_string
        return None

    for serial_port in discovered:
        if serial_port.connection_string == target.connection_string:
            return serial_port.connection_string
    return None


async def find_so101_port(manager: PortScanner, serial_port: SerialPortInfo) -> str | None:
    """Find the current port for an SO101 robot by serial number or configured port.

    Returns:
        The resolved serial device path, or ``None`` when it is unavailable.
    """
    port = _resolve_serial_port(manager.robots, serial_port)
    if port is not None:
        return port
    await manager.find_robots()
    return _resolve_serial_port(manager.robots, serial_port)


_IDENTIFY_WIGGLE = 0.08


def identify_so101_robot_visually(connection_string: str) -> None:
    """Move the gripper a small way and back to identify a connected robot.

    Raises:
        RuntimeError: If the gripper stops responding during identification.
    """
    joint = "gripper"
    joint_index = SO101.JOINT_ORDER.index(joint)
    driver = SO101.uncalibrated(port=connection_string, role="follower")
    driver.torque_on_disconnect = False
    driver.connect()
    try:
        observation = driver.get_observation()
        current = float(observation.joint_positions[joint_index])
        step = TICKS_PER_REVOLUTION * _IDENTIFY_WIGGLE
        lo, hi = 0.0, float(TICKS_PER_REVOLUTION - 1)
        targets = (min(max(current + step, lo), hi), min(max(current - step, lo), hi), current)
        for target in targets:
            action = observation.joint_positions.copy()
            action[joint_index] = target
            driver.send_action(action)
            time.sleep(1.0)
            driver.get_observation()
    except ConnectionError as exc:
        msg = (
            f"Robot identify failed: {joint} stopped responding during motion. "
            "The servo may have tripped overload protection. Power-cycle the robot and try again."
        )
        raise RuntimeError(msg) from exc
    finally:
        driver.set_torque(enabled=False)
        driver.disconnect()


class SO101Probe(RobotProbe[SO101RobotPayload]):
    """Probe for SO101 robots — serial discovery and gripper identification."""

    async def discover(self, manager: PortScanner) -> list[SerialPortInfo]:
        await manager.find_robots()
        return manager.robots

    async def identify(
        self,
        payload: SO101RobotPayload,
        manager: PortScanner | None,
        joint: str | None = None,
    ) -> None:
        _ = joint
        if manager is None:
            msg = "PortScanner required for SO101 identification"
            raise ValueError(msg)
        connection_string = await find_so101_port(
            manager,
            SerialPortInfo(
                connection_string=payload.connection_string or None,
                serial_number=payload.serial_number or None,
            ),
        )
        if connection_string is None:
            if payload.serial_number:
                msg = f"Could not find the serial port for serial number {payload.serial_number}"
                raise ValueError(msg)
            msg = "Could not resolve a serial port from connection_string"
            raise ValueError(msg)
        await asyncio.to_thread(identify_so101_robot_visually, connection_string)

    async def is_online(self, payload: SO101RobotPayload, manager: PortScanner | None = None) -> bool:
        serial_port = SerialPortInfo(
            connection_string=payload.connection_string or None,
            serial_number=payload.serial_number or None,
        )
        if manager is not None:
            return await find_so101_port(manager, serial_port) is not None

        from serial.tools import list_ports

        discovered = [
            SerialPortInfo(
                connection_string=port.device,
                serial_number=getattr(port, "serial_number", None) or None,
            )
            for port in list_ports.comports()
        ]
        return _resolve_serial_port(discovered, serial_port) is not None


_SO101_PROBE = SO101Probe()


def _definitions() -> list[RobotCatalogDefinition]:
    """Return all SO-101 catalog definitions."""
    from physicalai_so101_plugin.bimanual_studio_catalog import (
        ensure_bimanual_payload_models_resolvable,
        get_bimanual_definitions,
    )

    ensure_bimanual_payload_models_resolvable()
    return [
        RobotCatalogDefinition(
            type="SO101_Follower",
            display_name="SO101 Follower",
            role="follower",
            robot_builder=_build_so101_driver,
            robot_payload=SO101RobotPayload,
            asset=_SO101_FOLLOWER_ASSET,
            adapter_options=RobotAdapterOptions(goal_time_scale=1.0, external_effort_gain=None),
            probe=_SO101_PROBE,
        ),
        RobotCatalogDefinition(
            type="SO101_Leader",
            display_name="SO101 Leader",
            role="leader",
            robot_builder=_build_so101_driver,
            robot_payload=SO101RobotPayload,
            asset=_SO101_LEADER_ASSET,
            adapter_options=RobotAdapterOptions(goal_time_scale=1.0, external_effort_gain=None),
            probe=_SO101_PROBE,
        ),
        *get_bimanual_definitions(),
    ]


def register_physicalai_studio_plugin(registry: _RobotCatalogRegistry) -> None:
    """Register all SO-101 catalog entries with Physical AI Studio."""
    for definition in _definitions():
        registry.register_robot(definition)
