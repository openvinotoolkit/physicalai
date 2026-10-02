# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Studio catalog plugin for Physical AI Studio.

Exposes :func:`register_physicalai_studio_plugin` as the entry-point callable
for the ``physicalai.studio.catalog_plugins`` group.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal, Self

from loguru import logger
from physicalai_studio_plugin import (
    CatalogRobotFactory,
    PayloadContainer,
    PortScanner,
    RobotAdapterOptions,
    RobotAsset,
    RobotCatalogDefinition,
    RobotProbe,
    RobotZeroCalibration,
    SerialPortInfo,
    robot_field_ui,
    robot_payload_ui,
)
from pydantic import BaseModel, ConfigDict, Field, model_validator

import physicalai_rebot_b601_plugin
from physicalai_rebot_b601_plugin import ReBotB601DM, ReBotB601RS, get_urdf_path
from physicalai_rebot_b601_plugin.constants import REBOT_B601_RS_MIT_KD, REBOT_B601_RS_MIT_KP

if TYPE_CHECKING:
    from typing import Protocol

    from physicalai.robot.interface import Robot as PhysicalAIRobot

    class _RobotCatalogRegistry(Protocol):
        def register_robot(self, definition: RobotCatalogDefinition) -> None: ...


_REBOT_B601_DM_TO_URDF: dict[str, list[str]] = {
    "shoulder_pan.pos": ["joint1"],
    "shoulder_lift.pos": ["joint2"],
    "elbow_flex.pos": ["joint3"],
    "wrist_flex.pos": ["joint4"],
    "wrist_yaw.pos": ["joint5"],
    "wrist_roll.pos": ["joint6"],
    "gripper.pos": [],
}


def _get_rebot_urdf_root() -> Path:
    configured_root = get_urdf_path()
    if configured_root.exists():
        return configured_root
    plugin_package_root = Path(physicalai_rebot_b601_plugin.__file__).resolve().parent
    site_packages_urdf_root = plugin_package_root.parent / "urdf"
    if site_packages_urdf_root.exists():
        logger.warning(
            "ReBot plugin get_urdf_path() returned missing path={}; falling back to {}",
            configured_root,
            site_packages_urdf_root,
        )
        return site_packages_urdf_root
    return configured_root


_REBOT_B601_DM_ASSET = RobotAsset(
    urdf_relative_path=Path("rebot-b601-dm/urdf/reBot-DevArm_fixend.urdf"),
    packages={"rebot-b601-dm": Path("rebot-b601-dm")},
    joint_map=_REBOT_B601_DM_TO_URDF,
    root_resolver=_get_rebot_urdf_root,
)

# The RS preview uses a joint-frame copy of the URDF: the original follows the motor frame, so it would
# mirror elbow_flex, wrist_flex and wrist_yaw (direction -1 in REBOT_B601_RS_JOINT_DIRECTIONS).
_REBOT_B601_RS_ASSET = RobotAsset(
    urdf_relative_path=Path("rebot-b601-rs/urdf/00-arm-rs_asm-v3_joint_frame.urdf"),
    packages={"rebot-b601-rs": Path("rebot-b601-rs")},
    joint_map=_REBOT_B601_DM_TO_URDF,
    root_resolver=_get_rebot_urdf_root,
)


class ReBotB601DMPayload(BaseModel):
    """Connection payload for a ReBot B601 DM follower arm."""

    connection_string: str = ""
    serial_number: str = ""
    can_adapter: Literal["damiao", "socketcan"] = Field(  # pyrefly: ignore [bad-argument-type, no-matching-overload]
        default="damiao",
        json_schema_extra=robot_field_ui({"advanced_configuration": True}),
    )
    dm_serial_baud: int = Field(  # pyrefly: ignore [bad-argument-type, no-matching-overload]
        default=921600,
        json_schema_extra=robot_field_ui({"advanced_configuration": True}),
    )
    disable_torque_on_disconnect: bool = Field(  # pyrefly: ignore [bad-argument-type, no-matching-overload]
        default=True,
        json_schema_extra=robot_field_ui({"advanced_configuration": True}),
    )
    force_pos_torque_ratio: float = Field(  # pyrefly: ignore [bad-argument-type, no-matching-overload]
        default=0.1,
        json_schema_extra=robot_field_ui({"advanced_configuration": True}),
    )
    control_mode: Literal["pos_vel", "mit"] = Field(
        default="pos_vel",
        description=(
            "Position-joint control strategy. 'mit' commands a target position with "
            "per-joint stiffness/damping gains, giving fast, responsive motion. "
            "'pos_vel' uses velocity-capped position control, which is gentler "
            "and more predictable but slower."
        ),
    )
    gripper_control_mode: Literal["force_pos", "mit"] = Field(
        default="force_pos",
        description=(
            "Gripper control strategy. 'mit' uses impedance (stiffness/damping) control, "
            "letting the gripper comply if it meets resistance. 'force_pos' uses force-limited "
            "position control with a fixed torque ratio."
        ),
    )

    model_config = ConfigDict(
        json_schema_extra=robot_payload_ui(  # pyrefly: ignore[bad-argument-type]
            [
                {
                    "kind": "connection",
                    "label": "Select robot",
                    "device_discovery": True,
                    "bind": {"connection": "connection_string", "serial_number": "serial_number"},
                },
                {
                    "kind": "section",
                    "id": "control",
                    "title": "Control mode",
                    "description": "MIT mode drives joints with torque/stiffness gains for fast, responsive motion; "
                    "POS_VEL / FORCE_POS use velocity-capped position control.",
                    "items": [
                        {"kind": "field", "name": "control_mode"},
                        {"kind": "field", "name": "gripper_control_mode"},
                    ],
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


class ReBotProbe(RobotProbe[ReBotB601DMPayload]):
    """Probe implementation for ReBot serial devices."""

    async def discover(self, manager: PortScanner) -> list[SerialPortInfo]:
        """Discover available ReBot devices via the active port scanner.

        Returns:
            list[SerialPortInfo]: Detected serial devices.
        """
        _ = self
        await manager.find_robots()
        return manager.robots

    async def identify(
        self,
        payload: ReBotB601DMPayload,
        manager: PortScanner | None = None,
        joint: str | None = None,
    ) -> None:
        """Request a visual identify action on a specific joint, if supported."""
        _ = self, payload, manager, joint

    async def is_online(self, payload: ReBotB601DMPayload, manager: PortScanner | None = None) -> bool:
        """Report whether a ReBot device is currently online.

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


_REBOT_PROBE = ReBotProbe()


async def _build_rebot_b601_dm_driver(
    robot: PayloadContainer[ReBotB601DMPayload],
    factory: CatalogRobotFactory,
) -> PhysicalAIRobot:
    raw = robot.payload
    if isinstance(raw, BaseModel) and type(raw) is not ReBotB601DMPayload:
        raw = raw.model_dump()
    validated = raw if isinstance(raw, ReBotB601DMPayload) else ReBotB601DMPayload.model_validate(raw)
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
    return ReBotB601DM(
        port=port,
        can_adapter=validated.can_adapter,
        dm_serial_baud=validated.dm_serial_baud,
        role="follower",
        disable_torque_on_disconnect=validated.disable_torque_on_disconnect,
        force_pos_torque_ratio=validated.force_pos_torque_ratio,
        control_mode=validated.control_mode,
        gripper_control_mode=validated.gripper_control_mode,
    )


_SYSFS_NET_ROOT = Path("/sys/class/net")
# Linux interface names are at most 15 characters; starting with an alphanumeric excludes "." and "..".
_SOCKETCAN_INTERFACE_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,14}$"


_Gain = Annotated[float, Field(ge=0.0, allow_inf_nan=False)]


class ReBotB601RSJointGains(BaseModel):
    """MIT gains for the six ReBot B601 RS position joints."""

    shoulder_pan: _Gain
    shoulder_lift: _Gain
    elbow_flex: _Gain
    wrist_flex: _Gain
    wrist_yaw: _Gain
    wrist_roll: _Gain


class ReBotB601RSPayload(BaseModel):
    """Connection payload for a ReBot B601 RS follower arm on a SocketCAN interface."""

    connection_string: str = Field(
        default="can0",
        title="CAN interface",
        description="SocketCAN interface wired to the arm, for example can0. Bring it up at 1 Mbit/s first.",
        pattern=_SOCKETCAN_INTERFACE_PATTERN,
    )
    max_relative_target: float = Field(
        default=10.0,
        gt=0.0,
        allow_inf_nan=False,
        title="Max step per command (degrees)",
        description=(
            "Largest move, in joint degrees, the arm makes toward a new target in one control step. "
            "Stops the arm lunging when the leader or a policy jumps."
        ),
    )
    mit_kp: ReBotB601RSJointGains = Field(  # pyrefly: ignore [bad-argument-type, no-matching-overload]
        default=ReBotB601RSJointGains(**REBOT_B601_RS_MIT_KP),
        title="Joint stiffness (kp)",
        description="MIT stiffness per position joint. Lower values hold more softly but sag more under load.",
        json_schema_extra=robot_field_ui({"advanced_configuration": True}),
    )
    mit_kd: ReBotB601RSJointGains = Field(  # pyrefly: ignore [bad-argument-type, no-matching-overload]
        default=ReBotB601RSJointGains(**REBOT_B601_RS_MIT_KD),
        title="Joint damping (kd)",
        description="MIT damping per position joint. Higher values reduce overshoot but slow the response.",
        json_schema_extra=robot_field_ui({"advanced_configuration": True}),
    )
    gripper_mit_kp: float = Field(  # pyrefly: ignore [bad-argument-type, no-matching-overload]
        default=12.0,
        ge=0.0,
        allow_inf_nan=False,
        title="Gripper stiffness (kp)",
        json_schema_extra=robot_field_ui({"advanced_configuration": True}),
    )
    gripper_mit_kd: float = Field(  # pyrefly: ignore [bad-argument-type, no-matching-overload]
        default=0.05,
        ge=0.0,
        allow_inf_nan=False,
        title="Gripper damping (kd)",
        json_schema_extra=robot_field_ui({"advanced_configuration": True}),
    )
    gripper_mit_torque_limit: float = Field(  # pyrefly: ignore [bad-argument-type, no-matching-overload]
        default=3.5,
        ge=0.0,
        allow_inf_nan=False,
        title="Gripper torque limit while moving (N·m)",
        json_schema_extra=robot_field_ui({"advanced_configuration": True}),
    )
    gripper_mit_hold_torque_limit: float = Field(  # pyrefly: ignore [bad-argument-type, no-matching-overload]
        default=1.0,
        ge=0.0,
        allow_inf_nan=False,
        title="Gripper torque limit while holding (N·m)",
        description="Torque limit once the gripper stalls, for example on a grasped object.",
        json_schema_extra=robot_field_ui({"advanced_configuration": True}),
    )


class ReBotRSProbe(RobotProbe[ReBotB601RSPayload]):
    """Online check for ReBot RS arms, based on the SocketCAN interface state."""

    async def discover(self, manager: PortScanner) -> list[SerialPortInfo]:
        """Return no devices: SocketCAN interfaces are not serial ports.

        Returns:
            list[SerialPortInfo]: Always empty.
        """
        _ = self, manager
        return []

    async def identify(
        self,
        payload: ReBotB601RSPayload,
        manager: PortScanner | None = None,
        joint: str | None = None,
    ) -> None:
        """Request a visual identify action; not supported for ReBot RS arms."""
        _ = self, payload, manager, joint

    async def is_online(self, payload: ReBotB601RSPayload, manager: PortScanner | None = None) -> bool:
        """Report whether the configured SocketCAN interface is up.

        Returns:
            bool: ``True`` if the interface exists and its operational state is ``up``.
        """
        _ = self, manager
        operstate = _SYSFS_NET_ROOT / payload.connection_string / "operstate"
        try:
            return operstate.read_text(encoding="utf-8").strip() == "up"
        except OSError:
            return False


_REBOT_RS_PROBE = ReBotRSProbe()


async def _build_rebot_b601_rs_driver(  # noqa: RUF029 - Studio awaits every robot builder
    robot: PayloadContainer[ReBotB601RSPayload],
    factory: CatalogRobotFactory,
) -> PhysicalAIRobot:
    _ = factory  # SocketCAN interfaces are network devices, so there is no serial port to resolve.
    raw = robot.payload
    if isinstance(raw, BaseModel) and type(raw) is not ReBotB601RSPayload:
        raw = raw.model_dump()
    validated = raw if isinstance(raw, ReBotB601RSPayload) else ReBotB601RSPayload.model_validate(raw)
    return ReBotB601RS(
        port=validated.connection_string,
        can_adapter="socketcan",
        role="follower",
        max_relative_target=validated.max_relative_target,
        mit_kp=validated.mit_kp.model_dump(),
        mit_kd=validated.mit_kd.model_dump(),
        gripper_mit_kp=validated.gripper_mit_kp,
        gripper_mit_kd=validated.gripper_mit_kd,
        gripper_mit_torque_limit=validated.gripper_mit_torque_limit,
        gripper_mit_hold_torque_limit=validated.gripper_mit_hold_torque_limit,
    )


async def _release_rs(robot: ReBotB601RS) -> None:
    await asyncio.to_thread(robot.disable_torque)


async def _set_rs_zero(robot: ReBotB601RS) -> None:
    await asyncio.to_thread(robot.set_zero_position)


_REBOT_B601_RS_ZERO_CALIBRATION = RobotZeroCalibration[ReBotB601RS](
    instructions=(
        "Motor torque is off, so the arm can be moved by hand. Move it into its zero pose: the folded rest "
        "pose it sits in when powered off, with the gripper fully closed. Hold it still, then set zero."
    ),
    release=_release_rs,
    set_zero=_set_rs_zero,
)


def _definitions() -> list[RobotCatalogDefinition]:
    return [
        RobotCatalogDefinition(
            type="ReBot_B601_DM_Follower",
            display_name="ReBot B601 DM Follower",
            role="follower",
            robot_builder=_build_rebot_b601_dm_driver,
            robot_payload=ReBotB601DMPayload,
            asset=_REBOT_B601_DM_ASSET,
            adapter_options=RobotAdapterOptions(include_velocities=True, external_effort_gain=None),
            probe=_REBOT_PROBE,
        ),
        RobotCatalogDefinition(
            type="ReBot_B601_RS_Follower",
            display_name="ReBot B601 RS Follower",
            role="follower",
            robot_builder=_build_rebot_b601_rs_driver,
            robot_payload=ReBotB601RSPayload,
            asset=_REBOT_B601_RS_ASSET,
            adapter_options=RobotAdapterOptions(include_velocities=True, external_effort_gain=None),
            probe=_REBOT_RS_PROBE,
            zero_calibration=_REBOT_B601_RS_ZERO_CALIBRATION,
        ),
    ]


def _assert_payload_model_resolvable(model: type[BaseModel]) -> None:
    model.model_rebuild(_types_namespace=globals(), raise_errors=True)


def register_physicalai_studio_plugin(registry: _RobotCatalogRegistry) -> None:
    """Register ReBot catalog entries with the Physical AI Studio registry."""
    for definition in _definitions():
        payload_model = definition.robot_payload
        if isinstance(payload_model, type) and issubclass(payload_model, BaseModel):
            _assert_payload_model_resolvable(payload_model)
        registry.register_robot(definition)
