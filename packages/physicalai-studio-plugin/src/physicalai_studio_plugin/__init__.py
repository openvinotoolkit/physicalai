# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Public API for the Physical AI Studio plugin."""

from .assets import RobotAsset
from .autopilot import (
    AUTOPILOT_PATH,
    AUTOPILOT_PROTOCOL,
    AutopilotClient,
    AutopilotError,
    AutopilotInfo,
    AutopilotSpec,
    Episode,
    EpisodeOptions,
    EpisodeRequest,
    LeaderFactory,
    Outcome,
    Phase,
)
from .catalog import (
    BuildRobotCallable,
    CatalogRobot,
    PayloadContainer,
    RobotAdapterOptions,
    RobotCatalogDefinition,
    RobotCatalogRegistry,
)
from .factory import CatalogRobotFactory
from .probe import PortScanner, RobotProbe
from .schemas import SerialPortInfo
from .simulation import SimulationArgvBuilder, SimulationLaunch, SimulationPayloadArgvBuilder, SimulationScene
from .transport import shared_robot_name
from .ui_schema import (
    RobotFieldUiOptions,
    RobotPayloadUiOptions,
    RobotUiCalibrationItem,
    RobotUiConnectionBinding,
    RobotUiConnectionItem,
    RobotUiFieldItem,
    RobotUiInfoItem,
    RobotUiIpAddressItem,
    RobotUiItem,
    RobotUiSectionOptions,
    robot_field_ui,
    robot_payload_ui,
    validate_robot_payload_ui,
)
from .zero_calibration import RobotZeroCalibration

__all__ = [
    "AUTOPILOT_PATH",
    "AUTOPILOT_PROTOCOL",
    "AutopilotClient",
    "AutopilotError",
    "AutopilotInfo",
    "AutopilotSpec",
    "BuildRobotCallable",
    "CatalogRobot",
    "CatalogRobotFactory",
    "Episode",
    "EpisodeOptions",
    "EpisodeRequest",
    "LeaderFactory",
    "Outcome",
    "PayloadContainer",
    "Phase",
    "PortScanner",
    "RobotAdapterOptions",
    "RobotAsset",
    "RobotCatalogDefinition",
    "RobotCatalogRegistry",
    "RobotFieldUiOptions",
    "RobotPayloadUiOptions",
    "RobotProbe",
    "RobotUiCalibrationItem",
    "RobotUiConnectionBinding",
    "RobotUiConnectionItem",
    "RobotUiFieldItem",
    "RobotUiInfoItem",
    "RobotUiIpAddressItem",
    "RobotUiItem",
    "RobotUiSectionOptions",
    "RobotZeroCalibration",
    "SerialPortInfo",
    "SimulationArgvBuilder",
    "SimulationLaunch",
    "SimulationPayloadArgvBuilder",
    "SimulationScene",
    "robot_field_ui",
    "robot_payload_ui",
    "shared_robot_name",
    "validate_robot_payload_ui",
]
