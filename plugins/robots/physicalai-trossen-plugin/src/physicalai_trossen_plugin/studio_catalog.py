# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D102, FBT003, PLC0415, PLR6301, RUF029

"""Studio catalog plugin for Trossen WidowX AI robot arms."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from physicalai_studio_plugin import (
    CatalogRobot,
    CatalogRobotFactory,
    PortScanner,
    RobotAdapterOptions,
    RobotAsset,
    RobotCatalogDefinition,
    robot_payload_ui,
)
from pydantic import BaseModel, ConfigDict, Field

from physicalai_trossen_plugin import BimanualWidowXAI, WidowXAI, get_urdf_path

if TYPE_CHECKING:
    from physicalai_studio_plugin import SerialPortInfo


class _RobotCatalogRegistry(Protocol):
    def register_robot(self, definition: RobotCatalogDefinition) -> None: ...


class TrossenSingleArmPayload(BaseModel):
    """Connection configuration for a Trossen single-arm robot."""

    connection_string: str = Field(..., description="IP address of the robot")

    model_config = ConfigDict(
        json_schema_extra={  # pyrefly: ignore [bad-argument-type]
            "example": {"connection_string": "192.168.1.100"},
            **robot_payload_ui([
                {
                    "kind": "ip_address",
                    "name": "connection_string",
                    "label": "Robot IP address",
                    "identify": True,
                },
            ]),
        },
    )


class TrossenBimanualPayload(BaseModel):
    """Connection configuration for a Trossen bimanual robot."""

    connection_string_left: str = Field(..., description="IP address of the left arm")
    connection_string_right: str = Field(..., description="IP address of the right arm")

    model_config = ConfigDict(
        json_schema_extra={  # pyrefly: ignore [bad-argument-type]
            "example": {
                "connection_string_left": "192.168.1.100",
                "connection_string_right": "192.168.1.101",
            },
            **robot_payload_ui([
                {
                    "kind": "ip_address",
                    "name": "connection_string_left",
                    "label": "Left arm IP address",
                    "identify": True,
                    "identify_robot_type": "Trossen_WidowXAI_Follower",
                },
                {
                    "kind": "ip_address",
                    "name": "connection_string_right",
                    "label": "Right arm IP address",
                    "identify": True,
                    "identify_robot_type": "Trossen_WidowXAI_Follower",
                },
            ]),
        },
    )


_TROSSEN_TO_URDF = {
    "shoulder_pan.pos": ["joint_0"],
    "shoulder_lift.pos": ["joint_1"],
    "elbow_flex.pos": ["joint_2"],
    "wrist_flex.pos": ["joint_3"],
    "wrist_yaw.pos": ["joint_4"],
    "wrist_roll.pos": ["joint_5"],
    "gripper.pos": ["left_carriage_joint", "right_carriage_joint"],
}

_BIMANUAL_TROSSEN_TO_URDF = {
    "left_shoulder_pan.pos": ["follower_left_joint_0"],
    "left_shoulder_lift.pos": ["follower_left_joint_1"],
    "left_elbow_flex.pos": ["follower_left_joint_2"],
    "left_wrist_flex.pos": ["follower_left_joint_3"],
    "left_wrist_yaw.pos": ["follower_left_joint_4"],
    "left_wrist_roll.pos": ["follower_left_joint_5"],
    "left_gripper.pos": ["follower_left_left_carriage_joint", "follower_left_right_carriage_joint"],
    "right_shoulder_pan.pos": ["follower_right_joint_0"],
    "right_shoulder_lift.pos": ["follower_right_joint_1"],
    "right_elbow_flex.pos": ["follower_right_joint_2"],
    "right_wrist_flex.pos": ["follower_right_joint_3"],
    "right_wrist_yaw.pos": ["follower_right_joint_4"],
    "right_wrist_roll.pos": ["follower_right_joint_5"],
    "right_gripper.pos": ["follower_right_left_carriage_joint", "follower_right_right_carriage_joint"],
}

_TROSSEN_SINGLE_ARM_FOLLOWER_ASSET = RobotAsset(
    urdf_relative_path=Path("urdf/generated/wxai/wxai_follower.urdf"),
    packages={"trossen_arm_description": Path()},
    joint_map=_TROSSEN_TO_URDF,
    root_resolver=get_urdf_path,
)

_TROSSEN_SINGLE_ARM_LEADER_ASSET = RobotAsset(
    urdf_relative_path=Path("urdf/generated/wxai/wxai_leader_left.urdf"),
    packages={"trossen_arm_description": Path()},
    joint_map=_TROSSEN_TO_URDF,
    root_resolver=get_urdf_path,
)

_TROSSEN_BIMANUAL_FOLLOWER_ASSET = RobotAsset(
    urdf_relative_path=Path("urdf/generated/stationary_ai.urdf"),
    packages={"trossen_arm_description": Path()},
    joint_map=_BIMANUAL_TROSSEN_TO_URDF,
    root_resolver=get_urdf_path,
)

_BIMANUAL_TROSSEN_LEADER_TO_URDF = {
    **{
        action_name: [joint_name.replace("follower_left_", "leader_left_") for joint_name in joint_names]
        for action_name, joint_names in _BIMANUAL_TROSSEN_TO_URDF.items()
        if action_name.startswith("left_")
    },
    **{
        action_name: [joint_name.replace("follower_right_", "leader_right_") for joint_name in joint_names]
        for action_name, joint_names in _BIMANUAL_TROSSEN_TO_URDF.items()
        if action_name.startswith("right_")
    },
}

_TROSSEN_BIMANUAL_LEADER_ASSET = RobotAsset(
    urdf_relative_path=Path("urdf/generated/stationary_ai_leader.urdf"),
    packages={"trossen_arm_description": Path()},
    joint_map=_BIMANUAL_TROSSEN_LEADER_TO_URDF,
    root_resolver=get_urdf_path,
)


async def _build_trossen_single_arm_driver(
    robot: CatalogRobot[TrossenSingleArmPayload], _factory: CatalogRobotFactory
) -> WidowXAI:
    payload = robot.payload
    if not isinstance(payload, TrossenSingleArmPayload):
        msg = f"Expected TrossenSingleArmPayload, got {type(payload).__name__}"
        raise TypeError(msg)
    role = "follower" if robot.type == "Trossen_WidowXAI_Follower" else "leader"
    return WidowXAI(ip=payload.connection_string, role=role)


async def _build_trossen_bimanual_driver(
    robot: CatalogRobot[TrossenBimanualPayload], _factory: CatalogRobotFactory
) -> BimanualWidowXAI:
    payload = robot.payload
    if not isinstance(payload, TrossenBimanualPayload):
        msg = f"Expected TrossenBimanualPayload, got {type(payload).__name__}"
        raise TypeError(msg)
    role = "follower" if robot.type == "Trossen_Bimanual_WidowXAI_Follower" else "leader"
    return BimanualWidowXAI(
        left=WidowXAI(ip=payload.connection_string_left, role=role),
        right=WidowXAI(ip=payload.connection_string_right, role=role),
    )


async def _ping(ip: str, ping_timeout: float = 1.0) -> bool:
    param = "-n" if sys.platform.lower().startswith("win") else "-c"
    command = ["ping", param, "1", ip]
    proc = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        return (await asyncio.wait_for(proc.wait(), timeout=ping_timeout + 0.5)) == 0
    except TimeoutError:
        proc.kill()
        await proc.wait()
        return False


async def identify_trossen_robot_visually(payload: TrossenSingleArmPayload) -> None:
    """Identify a robot by opening and closing its gripper."""
    import trossen_arm
    from loguru import logger

    driver = trossen_arm.TrossenArmDriver()
    logger.info("Configuring the driver for visual identification.")
    driver.configure(
        trossen_arm.Model.wxai_v0,
        trossen_arm.StandardEndEffector.wxai_v0_leader,
        payload.connection_string,
        True,
        timeout=5,
    )
    driver.set_gripper_mode(trossen_arm.Mode.position)
    driver.set_gripper_position(0.02, 0.5, True)
    driver.set_gripper_position(0.0, 0.5, True)


class TrossenSingleArmProbe:
    """Probe for Trossen single-arm robots."""

    async def discover(self, manager: PortScanner) -> list[SerialPortInfo]:
        _ = manager
        return []

    async def identify(
        self,
        payload: TrossenSingleArmPayload,
        manager: PortScanner | None = None,
        joint: str | None = None,
    ) -> None:
        _ = manager, joint
        await identify_trossen_robot_visually(payload)

    async def is_online(self, payload: TrossenSingleArmPayload, manager: PortScanner | None = None) -> bool:
        _ = manager
        return bool(payload.connection_string) and await _ping(payload.connection_string)


class TrossenBimanualProbe:
    """Probe for Trossen bimanual robots."""

    async def discover(self, manager: PortScanner) -> list[SerialPortInfo]:
        _ = manager
        return []

    async def identify(
        self,
        payload: TrossenBimanualPayload,
        manager: PortScanner | None = None,
        joint: str | None = None,
    ) -> None:
        _ = manager, joint
        for ip in (payload.connection_string_left, payload.connection_string_right):
            await identify_trossen_robot_visually(TrossenSingleArmPayload(connection_string=ip))

    async def is_online(self, payload: TrossenBimanualPayload, manager: PortScanner | None = None) -> bool:
        _ = manager
        if not payload.connection_string_left or not payload.connection_string_right:
            return False
        left_ok, right_ok = await asyncio.gather(
            _ping(payload.connection_string_left), _ping(payload.connection_string_right)
        )
        return left_ok and right_ok


_SINGLE_ARM_PROBE = TrossenSingleArmProbe()
_BIMANUAL_PROBE = TrossenBimanualProbe()


def _definitions() -> list[RobotCatalogDefinition]:
    return [
        RobotCatalogDefinition(
            type="Trossen_WidowXAI_Follower",
            display_name="Trossen WidowX AI Follower",
            role="follower",
            robot_builder=_build_trossen_single_arm_driver,
            robot_payload=TrossenSingleArmPayload,
            asset=_TROSSEN_SINGLE_ARM_FOLLOWER_ASSET,
            adapter_options=RobotAdapterOptions(include_velocities=True, goal_time_scale=1.0, external_effort_gain=0.1),
            probe=_SINGLE_ARM_PROBE,
        ),
        RobotCatalogDefinition(
            type="Trossen_WidowXAI_Leader",
            display_name="Trossen WidowX AI Leader",
            role="leader",
            robot_builder=_build_trossen_single_arm_driver,
            robot_payload=TrossenSingleArmPayload,
            asset=_TROSSEN_SINGLE_ARM_LEADER_ASSET,
            adapter_options=RobotAdapterOptions(include_velocities=True, goal_time_scale=1.0, external_effort_gain=0.1),
            probe=_SINGLE_ARM_PROBE,
        ),
        RobotCatalogDefinition(
            type="Trossen_Bimanual_WidowXAI_Follower",
            display_name="Trossen Bimanual WidowX AI Follower",
            role="follower",
            robot_builder=_build_trossen_bimanual_driver,
            robot_payload=TrossenBimanualPayload,
            asset=_TROSSEN_BIMANUAL_FOLLOWER_ASSET,
            adapter_options=RobotAdapterOptions(include_velocities=True, goal_time_scale=1.0, external_effort_gain=0.1),
            probe=_BIMANUAL_PROBE,
        ),
        RobotCatalogDefinition(
            type="Trossen_Bimanual_WidowXAI_Leader",
            display_name="Trossen Bimanual WidowX AI Leader",
            role="leader",
            robot_builder=_build_trossen_bimanual_driver,
            robot_payload=TrossenBimanualPayload,
            asset=_TROSSEN_BIMANUAL_LEADER_ASSET,
            adapter_options=RobotAdapterOptions(include_velocities=True, goal_time_scale=1.0, external_effort_gain=0.1),
            probe=_BIMANUAL_PROBE,
        ),
    ]


def register_physicalai_studio_plugin(registry: _RobotCatalogRegistry) -> None:
    """Register Trossen catalog entries with Physical AI Studio."""
    for definition in _definitions():
        registry.register_robot(definition)
