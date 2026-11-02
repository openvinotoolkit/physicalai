# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Shared already-homed tick conversion; hardware offsets are not applied twice."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from physicalai_so101_plugin.constants import SO101_JOINT_ORDER, TICKS_PER_REVOLUTION

if TYPE_CHECKING:
    from physicalai_so101_plugin.calibration import SO101Calibration


def ticks_to_normalized(ticks: np.ndarray, calibration: SO101Calibration) -> np.ndarray:
    """Convert positions with the normal driver's ordering, clipping and dtype.

    Returns:
        Canonically ordered float32 normalized positions.
    """
    result = np.empty(len(SO101_JOINT_ORDER), dtype=np.float32)
    for i, name in enumerate(SO101_JOINT_ORDER):
        cal = calibration.joints[name]
        span = cal.range_max - cal.range_min
        if span <= 0:
            result[i] = 0.0
            continue
        tick = int(np.clip(ticks[i], cal.range_min, cal.range_max))
        if name == "gripper":
            norm = ((tick - cal.range_min) / span) * 100.0
            result[i] = float(np.clip(norm, 0.0, 100.0))
        else:
            norm = ((tick - cal.range_min) / span) * 200.0 - 100.0
            result[i] = float(np.clip(norm, -100.0, 100.0))
    return result


def ticks_to_degrees(ticks: np.ndarray, calibration: SO101Calibration) -> np.ndarray:
    """Use LeRobot's body degree convention, keeping gripper percentage units.

    Body zero is the recorded range midpoint, not the EEPROM homing offset.
    The encoder denominator is 4095 and body reads are NOT range-clipped.

    Returns:
        Float32 body degrees and gripper percentage in canonical order.

    Raises:
        ValueError: If a recorded range has no positive span.
    """
    result = ticks_to_normalized(ticks, calibration)
    for i, name in enumerate(SO101_JOINT_ORDER):
        cal = calibration.joints[name]
        if cal.range_max <= cal.range_min:
            msg = f"Motor {name} requires an ordered, nonzero calibration range for degrees"
            raise ValueError(msg)
        if name != "gripper":
            midpoint = (cal.range_min + cal.range_max) / 2
            result[i] = (int(ticks[i]) - midpoint) * 360 / (TICKS_PER_REVOLUTION - 1)
    return result


def degrees_to_ticks(values: np.ndarray, calibration: SO101Calibration) -> np.ndarray:
    """Convert body degrees and gripper percentage with LeRobot's truncation.

    Do not clamp body commands to recorded ranges. The driver's existing final
    encoder safety clamp still applies before hardware writes.

    Returns:
        Int32 servo ticks in canonical order.

    Raises:
        ValueError: If a recorded range has no positive span or inputs are not finite.
    """
    if not np.all(np.isfinite(values)):
        msg = "Degree-mode actions must contain only finite values"
        raise ValueError(msg)
    result = np.empty(len(SO101_JOINT_ORDER), dtype=np.int32)
    for i, name in enumerate(SO101_JOINT_ORDER):
        cal = calibration.joints[name]
        span = cal.range_max - cal.range_min
        if span <= 0:
            msg = f"Motor {name} requires an ordered, nonzero calibration range for degrees"
            raise ValueError(msg)
        if name == "gripper":
            fraction = float(np.clip(values[i], 0.0, 100.0)) / 100
            tick = int(fraction * span + cal.range_min)
        else:
            midpoint = (cal.range_min + cal.range_max) / 2
            raw_tick = float(values[i]) * (TICKS_PER_REVOLUTION - 1) / 360 + midpoint
            tick = int(min(np.iinfo(np.int32).max, max(np.iinfo(np.int32).min, raw_tick)))
        # Guard int32 conversion rather than letting extreme inputs wrap before
        # the driver's final hardware safety clamp.
        result[i] = tick
    return result
