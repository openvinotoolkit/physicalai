"""Studio catalog plugin for bimanual SO-101 robots."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

from physicalai.robot.so101 import SO101, SO101Calibration, SO101JointCalibration
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
from pydantic import BaseModel, ConfigDict, Field
from serial.tools import list_ports

import physicalai_bimanual_so101_plugin
from physicalai_bimanual_so101_plugin import BimanualSO101, get_urdf_path

if TYPE_CHECKING:
    from typing import Protocol

    from physicalai.robot.interface import Robot as PhysicalAIRobot

    class _RobotCatalogRegistry(Protocol):
        def register_robot(self, definition: RobotCatalogDefinition) -> None: ...


_BIMANUAL_SO101_TO_URDF: dict[str, list[str]] = {
    "left_shoulder_pan.pos": ["left_shoulder_pan"],
    "left_shoulder_lift.pos": ["left_shoulder_lift"],
    "left_elbow_flex.pos": ["left_elbow_flex"],
    "left_wrist_flex.pos": ["left_wrist_flex"],
    "left_wrist_roll.pos": ["left_wrist_roll"],
    "left_gripper.pos": ["left_gripper"],
    "right_shoulder_pan.pos": ["right_shoulder_pan"],
    "right_shoulder_lift.pos": ["right_shoulder_lift"],
    "right_elbow_flex.pos": ["right_elbow_flex"],
    "right_wrist_flex.pos": ["right_wrist_flex"],
    "right_wrist_roll.pos": ["right_wrist_roll"],
    "right_gripper.pos": ["right_gripper"],
}


def _get_bimanual_urdf_root() -> Path:
    configured_root = get_urdf_path()
    if configured_root.exists():
        return configured_root
    plugin_package_root = Path(physicalai_bimanual_so101_plugin.__file__).resolve().parent
    site_packages_urdf_root = plugin_package_root.parent / "urdf"
    if site_packages_urdf_root.exists():
        return site_packages_urdf_root
    return configured_root


_BIMANUAL_SO101_ASSET = RobotAsset(
    urdf_relative_path=Path("so101_dual/so101_dual.urdf"),
    packages={"so101_dual": Path("so101_dual")},
    joint_map=_BIMANUAL_SO101_TO_URDF,
    root_resolver=_get_bimanual_urdf_root,
)


class BimanualSO101Payload(BaseModel):
    """Connection payload for a bimanual SO-101 configuration."""

    left_serial_number: str = Field(...)
    right_serial_number: str = Field(...)
    left_calibration: dict[str, SO101JointCalibration] | None = Field(  # pyrefly: ignore [no-matching-overload]
        default=None,
        json_schema_extra=robot_field_ui({"advanced_configuration": True}),
    )
    right_calibration: dict[str, SO101JointCalibration] | None = Field(  # pyrefly: ignore [no-matching-overload]
        default=None,
        json_schema_extra=robot_field_ui({"advanced_configuration": True}),
    )
    baudrate: int = Field(  # pyrefly: ignore [no-matching-overload]
        default=1_000_000,
        json_schema_extra=robot_field_ui({"advanced_configuration": True}),
    )
    role: Literal["follower", "leader"] = "follower"
    disable_torque_on_disconnect: bool = Field(  # pyrefly: ignore [no-matching-overload]
        default=True,
        json_schema_extra=robot_field_ui({"advanced_configuration": True}),
    )

    model_config = ConfigDict(
        # pyrefly: ignore [bad-argument-type]
        json_schema_extra=robot_payload_ui(
            [
                {
                    "kind": "calibration",
                    "name": "left_calibration",
                    "info": {
                        "description": "Upload left-arm SO101 calibration JSON (joint-name keyed).",
                    },
                },
                {
                    "kind": "calibration",
                    "name": "right_calibration",
                    "info": {
                        "description": "Upload right-arm SO101 calibration JSON (joint-name keyed).",
                    },
                },
            ],
        ),
    )


class BimanualSO101Probe(RobotProbe[BimanualSO101Payload]):
    """Probe implementation for bimanual SO-101 robots."""

    async def discover(self, manager: PortScanner) -> list[SerialPortInfo]:
        """Discover available serial devices.

        Returns:
            list[SerialPortInfo]: Detected serial devices.
        """
        _ = self
        await manager.find_robots()
        return manager.robots

    async def identify(
        self,
        payload: BimanualSO101Payload,
        manager: PortScanner | None = None,
        joint: str | None = None,
    ) -> None:
        """Request a visual identify action, if supported."""
        _ = self, payload, manager, joint

    async def is_online(self, payload: BimanualSO101Payload, manager: PortScanner | None = None) -> bool:
        """Check whether both arms are online.

        Returns:
            bool: ``True`` when both configured serial numbers are present.
        """
        _ = self
        if manager is not None:
            ports_list = manager.robots
            left_match = any(p.serial_number == payload.left_serial_number for p in ports_list)
            right_match = any(p.serial_number == payload.right_serial_number for p in ports_list)
            return left_match and right_match

        all_ports = list_ports.comports()
        left_match = any(p.serial_number == payload.left_serial_number for p in all_ports)
        right_match = any(p.serial_number == payload.right_serial_number for p in all_ports)
        return left_match and right_match


_BIMANUAL_PROBE = BimanualSO101Probe()


async def _build_bimanual_driver(
    robot: PayloadContainer[BimanualSO101Payload],
    factory: CatalogRobotFactory,
    *,
    role: Literal["follower", "leader"] | None = None,
) -> PhysicalAIRobot:
    raw = robot.payload
    if isinstance(raw, BaseModel) and type(raw) is not BimanualSO101Payload:
        raw = raw.model_dump()
    validated = raw if isinstance(raw, BimanualSO101Payload) else BimanualSO101Payload.model_validate(raw)

    driver_role = validated.role if role is None else role
    baudrate = validated.baudrate

    left_port = await factory.find_port(
        SerialPortInfo(connection_string=None, serial_number=validated.left_serial_number),
    )
    if left_port is None:
        msg = f"Left arm not found: {validated.left_serial_number}"
        raise RuntimeError(msg)

    right_port = await factory.find_port(
        SerialPortInfo(connection_string=None, serial_number=validated.right_serial_number),
    )
    if right_port is None:
        msg = f"Right arm not found: {validated.right_serial_number}"
        raise RuntimeError(msg)

    has_left_cal = validated.left_calibration is not None
    has_right_cal = validated.right_calibration is not None

    if has_left_cal != has_right_cal:
        msg = (
            "Both arms must have calibrations or both must be uncalibrated. "
            f"Got left_calibration={validated.left_calibration!r}, "
            f"right_calibration={validated.right_calibration!r}."
        )
        raise ValueError(msg)

    if has_left_cal and has_right_cal:
        left_calibration = validated.left_calibration
        right_calibration = validated.right_calibration
        if left_calibration is None or right_calibration is None:
            msg = "Both calibrations are required when calibration is enabled"
            raise ValueError(msg)

        left_arm = SO101(
            port=left_port,
            baudrate=baudrate,
            role=driver_role,
            calibration=SO101Calibration(joints=left_calibration),
            unit="normalized",
        )
        right_arm = SO101(
            port=right_port,
            baudrate=baudrate,
            role=driver_role,
            calibration=SO101Calibration(joints=right_calibration),
            unit="normalized",
        )
    else:
        left_arm = SO101.uncalibrated(port=left_port, baudrate=baudrate, role=driver_role, unit="ticks")
        right_arm = SO101.uncalibrated(port=right_port, baudrate=baudrate, role=driver_role, unit="ticks")

    return BimanualSO101(left=left_arm, right=right_arm)


async def _build_bimanual_follower(
    robot: PayloadContainer[BimanualSO101Payload],
    factory: CatalogRobotFactory,
) -> PhysicalAIRobot:
    return await _build_bimanual_driver(robot, factory, role="follower")


async def _build_bimanual_leader(
    robot: PayloadContainer[BimanualSO101Payload],
    factory: CatalogRobotFactory,
) -> PhysicalAIRobot:
    return await _build_bimanual_driver(robot, factory, role="leader")


def _definitions() -> list[RobotCatalogDefinition]:
    return [
        RobotCatalogDefinition(
            type="BimanualSO101_Follower",
            display_name="Bimanual SO-101 Follower",
            role="follower",
            robot_builder=_build_bimanual_follower,
            robot_payload=BimanualSO101Payload,
            asset=_BIMANUAL_SO101_ASSET,
            adapter_options=RobotAdapterOptions(include_velocities=False, external_effort_gain=None),
            probe=_BIMANUAL_PROBE,
        ),
        RobotCatalogDefinition(
            type="BimanualSO101_Leader",
            display_name="Bimanual SO-101 Leader",
            role="leader",
            robot_builder=_build_bimanual_leader,
            robot_payload=BimanualSO101Payload,
            asset=_BIMANUAL_SO101_ASSET,
            adapter_options=RobotAdapterOptions(include_velocities=False, external_effort_gain=None),
            probe=_BIMANUAL_PROBE,
        ),
    ]


def _assert_payload_model_resolvable(model: type[BaseModel]) -> None:
    model.model_rebuild(_types_namespace=globals(), raise_errors=True)


def register_physicalai_studio_plugin(registry: _RobotCatalogRegistry) -> None:
    """Register bimanual SO-101 catalog entries with the Studio registry."""
    for definition in _definitions():
        payload_model = definition.robot_payload
        if isinstance(payload_model, type) and issubclass(payload_model, BaseModel):
            _assert_payload_model_resolvable(payload_model)
        registry.register_robot(definition)
