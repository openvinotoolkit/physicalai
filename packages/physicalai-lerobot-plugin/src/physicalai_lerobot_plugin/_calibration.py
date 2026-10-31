# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Non-interactive LeRobot setup and calibration diagnostics."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar, cast

from loguru import logger

from physicalai.robot.errors import RobotCalibrationError

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

_Device = TypeVar("_Device")
_Config = TypeVar("_Config")
_ARM_ATTRIBUTES = ("left_arm", "right_arm", "left", "right")


def build_device(factory: Callable[[_Config], _Device], config: _Config) -> _Device:
    """Translate malformed saved LeRobot calibration during construction.

    LeRobot loads calibration before the adapter has a device and its resolved
    calibration path to inspect. Other construction errors remain unchanged.

    Returns:
        The disconnected LeRobot device.

    Raises:
        RobotCalibrationError: If calibration JSON cannot be decoded.
    """
    from draccus.utils import DecodingError  # noqa: PLC0415

    try:
        return factory(config)
    except (json.JSONDecodeError, DecodingError) as exc:
        msg = "Invalid LeRobot calibration file. Verify the saved calibration before reconnecting."
        raise RobotCalibrationError(msg, reason="invalid") from exc


def _arms(device: object) -> Iterator[Any]:
    for name in _ARM_ATTRIBUTES:
        arm = getattr(device, "__dict__", {}).get(name)
        if arm is not None and callable(getattr(arm, "connect", None)):
            yield arm


def _calibration_devices(device: object) -> Iterator[Any]:
    arms = list(_arms(device))
    if arms:
        for arm in arms:
            yield from _calibration_devices(arm)
    else:
        yield device


def connect_device(device: object) -> None:
    """Connect without calibration prompts and validate before any observation.

    Calibration-free devices retain their argument-free connection API.
    No saved values are written to motors to resolve a mismatch.

    Raises:
        RobotCalibrationError: If calibration is missing, empty, or mismatched.
    """
    devices = list(_calibration_devices(device))
    for part in devices:
        path = getattr(part, "calibration_fpath", None)
        calibration = getattr(part, "calibration", None)
        bus = getattr(part, "bus", None)
        # Base devices can have an empty calibration even when they require
        # none. Only preflight motor buses that consume calibration.
        if (
            isinstance(path, Path)
            and isinstance(calibration, dict)
            and isinstance(getattr(bus, "calibration", None), dict)
            and not calibration
        ):
            reason = "invalid" if path.is_file() else "missing"
            msg = (
                f"LeRobot calibration is {reason}: {path}. "
                "Set calibration_dir to the directory containing this robot's calibration file, "
                "or calibrate in a terminal before reconnecting."
            )
            raise RobotCalibrationError(msg, reason=reason, calibration_path=str(path))

    connect = cast("Any", device).connect
    parameters = inspect.signature(connect).parameters.values()
    supports_calibrate = any(
        parameter.name == "calibrate" or parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters
    )
    if supports_calibrate:
        connect(calibrate=False)
    else:
        connect()

    for part in devices:
        if not getattr(part, "is_calibrated", True):
            path = getattr(part, "calibration_fpath", None)
            msg = "Saved LeRobot calibration does not match the motors. Verify or reapply calibration in a terminal."
            raise RobotCalibrationError(
                msg,
                reason="mismatch",
                calibration_path=str(path) if isinstance(path, Path) else None,
            )


def disconnect_device(device: object) -> None:
    """Release resources, including a partially connected composite.

    Follower is_connected includes cameras, so a bus can remain open even
    when the device reports False after a failed connection.
    """
    if getattr(device, "is_connected", False):
        try:
            cast("Any", device).disconnect()
        except Exception:  # noqa: BLE001
            logger.exception("Error disconnecting LeRobot device")
        else:
            return

    for arm in _arms(device):
        disconnect_device(arm)
    resources = [getattr(device, "bus", None)]
    cameras = getattr(device, "cameras", {})
    if isinstance(cameras, dict):
        resources.extend(cameras.values())
    for resource in resources:
        if resource is not None and getattr(resource, "is_connected", False):
            try:
                resource.disconnect()
            except Exception:  # noqa: BLE001
                logger.exception("Error disconnecting partially connected LeRobot resource")
