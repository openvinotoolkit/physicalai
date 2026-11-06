# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""SO-101 hardware constants (Feetech STS3215 servos)."""

from __future__ import annotations

from enum import IntEnum
from typing import Final

import numpy as np

# ---------------------------------------------------------------------------
# Encoder / conversion
# ---------------------------------------------------------------------------

TICKS_PER_REVOLUTION: Final = 4096
"""STS3215 encoder resolution: 4096 ticks per full 360° revolution."""

RADIANS_PER_TICK: Final = 2.0 * np.pi / TICKS_PER_REVOLUTION

# ---------------------------------------------------------------------------
# Speed limits
# ---------------------------------------------------------------------------

MAX_SPEED_RAD_S: Final = 4.712389
"""Maximum angular velocity in rad/s (STS3215 @ 12V: 60 deg / 0.222 s ~ 270 deg/s)."""

MAX_SPEED_DEG_S: Final = float(np.degrees(MAX_SPEED_RAD_S))
"""Maximum angular velocity in deg/s."""

# ---------------------------------------------------------------------------
# Joint ordering
# ---------------------------------------------------------------------------

SO101_JOINT_ORDER: Final = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

VALID_ROLES: Final = frozenset({"leader", "follower"})

# ---------------------------------------------------------------------------
# Feetech STS3215 control table
# ---------------------------------------------------------------------------

PROTOCOL_VERSION: Final = 0

POSITION_MODE: Final = 0
"""Operating mode value for position control."""

STS3215_MODEL_NUMBER: Final = 777
STS3215_BAUDRATES: Final = {
    1_000_000: 0,
    500_000: 1,
    250_000: 2,
    128_000: 3,
    115_200: 4,
    57_600: 5,
    38_400: 6,
    19_200: 7,
}


class STS3215Addr(IntEnum):
    """Feetech STS3215 control table addresses.

    Reference: STS/SMS series control table
    http://doc.feetech.cn/#/prodinfodownload?srcType=FT-SMS-STS-emanual
    """

    # EPROM registers
    MODEL_NUMBER = 3
    ID = 5
    BAUD_RATE = 6
    RETURN_DELAY_TIME = 7
    MIN_POSITION_LIMIT = 9
    MAX_POSITION_LIMIT = 11
    MAX_TORQUE_LIMIT = 16
    PHASE = 18
    P_COEFFICIENT = 21
    D_COEFFICIENT = 22
    I_COEFFICIENT = 23
    PROTECTION_CURRENT = 28
    HOMING_OFFSET = 31
    OPERATING_MODE = 33
    OVERLOAD_TORQUE = 36

    # SRAM registers
    TORQUE_ENABLE = 40
    ACCELERATION = 41
    GOAL_POSITION = 42
    EEPROM_LOCK = 55
    PRESENT_POSITION = 56
    PRESENT_VOLTAGE = 62

    # Factory registers
    MAXIMUM_ACCELERATION = 85


class STS3215Len(IntEnum):
    """Byte widths for STS3215 sync read/write fields."""

    MODEL_NUMBER = 2
    ID = 1
    BAUD_RATE = 1
    RETURN_DELAY_TIME = 1
    MIN_POSITION_LIMIT = 2
    MAX_POSITION_LIMIT = 2
    MAX_TORQUE_LIMIT = 2
    PHASE = 1
    P_COEFFICIENT = 1
    D_COEFFICIENT = 1
    I_COEFFICIENT = 1
    PROTECTION_CURRENT = 2
    HOMING_OFFSET = 2
    OPERATING_MODE = 1
    OVERLOAD_TORQUE = 1
    TORQUE_ENABLE = 1
    ACCELERATION = 1
    GOAL_POSITION = 2
    EEPROM_LOCK = 1
    PRESENT_POSITION = 2
    PRESENT_VOLTAGE = 1
    MAXIMUM_ACCELERATION = 1
