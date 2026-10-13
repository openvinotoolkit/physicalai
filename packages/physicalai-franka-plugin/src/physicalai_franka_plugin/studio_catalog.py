# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Studio catalog plugin for Physical AI Studio.

Exposes :func:`register_physicalai_studio_plugin` as the entry-point callable
for the ``physicalai.studio.catalog_plugins`` group.
"""

from __future__ import annotations

import socket
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Self

from loguru import logger
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

import physicalai_franka_plugin
from physicalai_franka_plugin import BimanualFrankaFR3, FrankaFR3, get_urdf_path

if TYPE_CHECKING:
    from typing import Protocol

    from physicalai.robot.interface import Robot as PhysicalAIRobot

    class _RobotCatalogRegistry(Protocol):
        def register_robot(self, definition: RobotCatalogDefinition) -> None: ...


_FCI_PORT = 50000

_FRANKA_FR3_TO_URDF: dict[str, list[str]] = {
    "fr3_joint1.pos": ["fr3_joint1"],
    "fr3_joint2.pos": ["fr3_joint2"],
    "fr3_joint3.pos": ["fr3_joint3"],
    "fr3_joint4.pos": ["fr3_joint4"],
    "fr3_joint5.pos": ["fr3_joint5"],
    "fr3_joint6.pos": ["fr3_joint6"],
    "fr3_joint7.pos": ["fr3_joint7"],
    "gripper.pos": [],
}

_BIMANUAL_FRANKA_FR3_TO_URDF: dict[str, list[str]] = {
    f"left_{driver_name}": [f"left_{urdf_name}" for urdf_name in urdf_names]
    for driver_name, urdf_names in _FRANKA_FR3_TO_URDF.items()
} | {
    f"right_{driver_name}": [f"right_{urdf_name}" for urdf_name in urdf_names]
    for driver_name, urdf_names in _FRANKA_FR3_TO_URDF.items()
}


def _get_franka_urdf_root() -> Path:
    configured_root = get_urdf_path()
    if configured_root.exists():
        return configured_root
    plugin_package_root = Path(physicalai_franka_plugin.__file__).resolve().parent
    site_packages_urdf_root = plugin_package_root.parent / "urdf"
    if site_packages_urdf_root.exists():
        logger.warning(
            "Franka plugin get_urdf_path() returned missing path={}; falling back to {}",
            configured_root,
            site_packages_urdf_root,
        )
        return site_packages_urdf_root
    return configured_root


_FRANKA_FR3_ASSET = RobotAsset(
    urdf_relative_path=Path("franka_description/urdf/fr3.urdf"),
    packages={"franka_description": Path("franka_description")},
    joint_map=_FRANKA_FR3_TO_URDF,
    root_resolver=_get_franka_urdf_root,
)

_BIMANUAL_FRANKA_FR3_ASSET = RobotAsset(
    urdf_relative_path=Path("franka_description/urdf/fr3_bimanual.urdf"),
    packages={"franka_description": Path("franka_description")},
    joint_map=_BIMANUAL_FRANKA_FR3_TO_URDF,
    root_resolver=_get_franka_urdf_root,
)


class FrankaFR3Payload(BaseModel):
    """Connection payload for a Franka Research 3 follower arm."""

    connection_string: str = Field(
        default="",
        title="Robot host",
        description="IP address or hostname of the Franka Research 3 control unit.",
    )
    gripper_enabled: bool = Field(
        default=True,
        description=(
            "Include the Franka Hand gripper in the action/observation layout. Requires a "
            "Franka Hand to be attached; arm-only control continues with a warning when it "
            "cannot be reached."
        ),
    )
    realtime_config: Literal["kEnforce", "kIgnore"] = Field(  # pyrefly: ignore[no-matching-overload]
        default="kEnforce",
        json_schema_extra=robot_field_ui({"advanced_configuration": True}),  # pyrefly: ignore[bad-typed-dict-key]
    )
    goal_time: float = Field(  # pyrefly: ignore[no-matching-overload]
        default=0.1,
        json_schema_extra=robot_field_ui({"advanced_configuration": True}),  # pyrefly: ignore[bad-typed-dict-key]
    )
    auto_recover_on_error: bool = Field(  # pyrefly: ignore[no-matching-overload]
        default=True,
        json_schema_extra=robot_field_ui({"advanced_configuration": True}),  # pyrefly: ignore[bad-typed-dict-key]
    )
    gripper_speed_m_s: float = Field(  # pyrefly: ignore[no-matching-overload]
        default=0.05,
        json_schema_extra=robot_field_ui({"advanced_configuration": True}),  # pyrefly: ignore[bad-typed-dict-key]
    )

    model_config = ConfigDict(
        json_schema_extra=robot_payload_ui(  # pyrefly: ignore[bad-argument-type]
            [
                {
                    "kind": "connection",
                    "label": "Robot control unit",
                    "description": "Enter the FCI hostname or IP address of the control unit.",
                    "device_discovery": False,
                    "manual_entry": True,
                    "bind": {"connection": "connection_string"},
                },
            ],
        ),
    )

    @model_validator(mode="after")
    def _require_host(self) -> Self:
        if not self.connection_string:
            msg = "connection_string (robot host) must be provided"
            raise ValueError(msg)
        return self


class BimanualFrankaFR3Payload(BaseModel):
    """Connection payload for a left/right pair of Franka FR3 followers."""

    left_host: str = Field(..., title="Left robot host")
    right_host: str = Field(..., title="Right robot host")
    gripper_enabled: bool = Field(default=True)
    realtime_config: Literal["kEnforce", "kIgnore"] = Field(  # pyrefly: ignore[no-matching-overload]
        default="kEnforce", json_schema_extra=robot_field_ui({"advanced_configuration": True})
    )
    goal_time: float = Field(  # pyrefly: ignore[no-matching-overload]
        default=0.1, json_schema_extra=robot_field_ui({"advanced_configuration": True})
    )
    auto_recover_on_error: bool = Field(  # pyrefly: ignore[no-matching-overload]
        default=True, json_schema_extra=robot_field_ui({"advanced_configuration": True})
    )
    gripper_speed_m_s: float = Field(  # pyrefly: ignore[no-matching-overload]
        default=0.05, json_schema_extra=robot_field_ui({"advanced_configuration": True})
    )

    model_config = ConfigDict(
        json_schema_extra=robot_payload_ui(  # pyrefly: ignore[bad-argument-type]
            [
                {
                    "kind": "connection",
                    "label": "Left robot control unit",
                    "device_discovery": False,
                    "manual_entry": True,
                    "bind": {"connection": "left_host"},
                },
                {
                    "kind": "connection",
                    "label": "Right robot control unit",
                    "device_discovery": False,
                    "manual_entry": True,
                    "bind": {"connection": "right_host"},
                },
            ],
        ),
    )

    @model_validator(mode="after")
    def _require_distinct_hosts(self) -> Self:
        if not self.left_host or not self.right_host:
            msg = "Both left_host and right_host must be provided"
            raise ValueError(msg)
        if self.left_host == self.right_host:
            msg = "left_host and right_host must be different"
            raise ValueError(msg)
        return self


class FrankaFR3Probe(RobotProbe[FrankaFR3Payload]):
    """Probe implementation for Franka Research 3 control units over the FCI."""

    async def discover(self, manager: PortScanner) -> list[SerialPortInfo]:
        """Report no discoverable devices; the FCI is addressed by hostname.

        Returns:
            An empty list — network robot discovery is not supported.
        """
        _ = self, manager
        return []

    async def identify(
        self,
        payload: FrankaFR3Payload,
        manager: PortScanner | None = None,
        joint: str | None = None,
    ) -> None:
        """No visual identify action is supported for the FR3."""
        _ = self, payload, manager, joint

    async def is_online(self, payload: FrankaFR3Payload, manager: PortScanner | None = None) -> bool:
        """Best-effort reachability check against the FCI port of the host.

        The FCI command interface is UDP-based, so a TCP connect may report the
        robot offline even when it is reachable. This is only a hint.

        Returns:
            ``True`` if the host accepts a TCP connection on the FCI port.
        """
        _ = self, manager
        host = payload.connection_string
        if not host:
            return False
        try:
            with socket.create_connection((host, _FCI_PORT), timeout=2.0):
                return True
        except OSError:
            return False


_FRANKA_FR3_PROBE = FrankaFR3Probe()


async def _build_franka_fr3_driver(
    robot: PayloadContainer[FrankaFR3Payload],
    factory: CatalogRobotFactory,
) -> PhysicalAIRobot:
    raw = robot.payload
    if isinstance(raw, BaseModel) and type(raw) is not FrankaFR3Payload:
        raw = raw.model_dump()
    validated = raw if isinstance(raw, FrankaFR3Payload) else FrankaFR3Payload.model_validate(raw)
    host = validated.connection_string
    resolved = await factory.find_port(SerialPortInfo(connection_string=host, serial_number=None))
    if resolved:
        host = resolved
    return FrankaFR3(
        host=host,
        gripper_enabled=validated.gripper_enabled,
        realtime_config=validated.realtime_config,
        goal_time=validated.goal_time,
        auto_recover_on_error=validated.auto_recover_on_error,
        gripper_speed_m_s=validated.gripper_speed_m_s,
    )


async def _build_bimanual_franka_fr3_driver(
    robot: PayloadContainer[BimanualFrankaFR3Payload],
    factory: CatalogRobotFactory,
) -> PhysicalAIRobot:
    raw = robot.payload
    if isinstance(raw, BaseModel) and type(raw) is not BimanualFrankaFR3Payload:
        raw = raw.model_dump()
    validated = raw if isinstance(raw, BimanualFrankaFR3Payload) else BimanualFrankaFR3Payload.model_validate(raw)
    left_host = await factory.find_port(SerialPortInfo(connection_string=validated.left_host, serial_number=None))
    right_host = await factory.find_port(SerialPortInfo(connection_string=validated.right_host, serial_number=None))
    common_args = {
        "gripper_enabled": validated.gripper_enabled,
        "realtime_config": validated.realtime_config,
        "goal_time": validated.goal_time,
        "auto_recover_on_error": validated.auto_recover_on_error,
        "gripper_speed_m_s": validated.gripper_speed_m_s,
    }
    return BimanualFrankaFR3(
        left=FrankaFR3(host=left_host or validated.left_host, **common_args),
        right=FrankaFR3(host=right_host or validated.right_host, **common_args),
    )


def _definitions() -> list[RobotCatalogDefinition]:
    return [
        RobotCatalogDefinition(
            type="Franka_FR3_Follower",
            display_name="Franka FR3 Follower",
            role="follower",
            robot_builder=_build_franka_fr3_driver,
            robot_payload=FrankaFR3Payload,
            asset=_FRANKA_FR3_ASSET,
            adapter_options=RobotAdapterOptions(include_velocities=True, external_effort_gain=None),
            probe=_FRANKA_FR3_PROBE,
        ),
        RobotCatalogDefinition(
            type="Bimanual_Franka_FR3_Follower",
            display_name="Bimanual Franka FR3 Follower",
            role="follower",
            robot_builder=_build_bimanual_franka_fr3_driver,
            robot_payload=BimanualFrankaFR3Payload,
            asset=_BIMANUAL_FRANKA_FR3_ASSET,
            adapter_options=RobotAdapterOptions(include_velocities=True, external_effort_gain=None),
            probe=_FRANKA_FR3_PROBE,
        ),
    ]


def _assert_payload_model_resolvable(model: type[BaseModel]) -> None:
    model.model_rebuild(_types_namespace=globals(), raise_errors=True)


def register_physicalai_studio_plugin(registry: _RobotCatalogRegistry) -> None:
    """Register Franka catalog entries with the Physical AI Studio registry."""
    for definition in _definitions():
        payload_model = definition.robot_payload
        if isinstance(payload_model, type) and issubclass(payload_model, BaseModel):
            _assert_payload_model_resolvable(payload_model)
        registry.register_robot(definition)
