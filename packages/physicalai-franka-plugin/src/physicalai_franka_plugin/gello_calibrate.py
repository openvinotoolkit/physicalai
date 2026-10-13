# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# ruff: file-ignore[any-type, docstring-missing-exception, docstring-missing-returns, print]

"""Non-actuating calibration command for Franka GELLO leaders."""

from __future__ import annotations

import argparse
import importlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from physicalai_franka_plugin.gello import GelloCalibration

_RAD_PER_COUNT = 2.0 * math.pi / 4095.0
_PRESENT_POSITION_ADDRESS = 132
_TORQUE_ENABLE_ADDRESS = 64
_COMM_SUCCESS = 0
_JOINT_SIGNS = (1, -1, 1, -1, 1, 1, 1)
_REFERENCE_POSES = {
    "single": (0.0, 0.0, 0.0, -1.57, 0.0, 1.57, 0.0),
    "left": (-1.57, -0.80, 1.80, -3.00, 1.40, 1.50, -2.10),
    "right": (1.57, -0.80, -1.80, -3.00, -1.40, 1.50, 2.10),
}


def calculate_calibration(
    arm_raw_rad: np.ndarray,
    *,
    reference_pose_rad: tuple[float, float, float, float, float, float, float],
    gripper_closed_rad: float,
    gripper_open_rad: float,
    baudrate: int,
) -> GelloCalibration:
    """Calculate FR3 GELLO assembly offsets from a reference pose readout."""
    if arm_raw_rad.shape != (7,):
        msg = f"Expected seven raw arm positions, got {arm_raw_rad.shape}"
        raise ValueError(msg)
    signs = np.asarray(_JOINT_SIGNS)
    offsets = np.mod(
        np.round((arm_raw_rad * signs - np.asarray(reference_pose_rad)) / (math.pi / 2)) * (math.pi / 2),
        2 * math.pi,
    )
    return GelloCalibration(
        joint_signs=_JOINT_SIGNS,
        assembly_offsets_rad=tuple(float(value) for value in offsets),  # type: ignore[arg-type]
        gripper_range_rad=(gripper_closed_rad, gripper_open_rad),
        baudrate=baudrate,
    )


def _read_rad(packet_handler: Any, port_handler: Any, motor_id: int) -> float:
    value, comm_result, error = packet_handler.read4ByteTxRx(port_handler, motor_id, _PRESENT_POSITION_ADDRESS)
    if comm_result != _COMM_SUCCESS or error:
        msg = f"Failed to read GELLO motor ID {motor_id}"
        raise ConnectionError(msg)
    return float(value) * _RAD_PER_COUNT


def _write_calibration(path: Path, calibration: GelloCalibration, *, force: bool) -> None:
    if path.exists() and not force:
        msg = f"Refusing to overwrite {path}. Pass --force after reviewing the existing calibration."
        raise FileExistsError(msg)
    payload = {
        "joint_signs": list(calibration.joint_signs),
        "assembly_offsets_rad": list(calibration.assembly_offsets_rad),
        "gripper_range_rad": list(calibration.gripper_range_rad),
        "baudrate": calibration.baudrate,
    }
    path.write_text(f"{json.dumps(payload, indent=2)}\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Run interactive, non-actuating GELLO calibration and write Studio-uploadable JSON."""
    parser = argparse.ArgumentParser(description="Calibrate a passive Franka GELLO leader without enabling torque.")
    parser.add_argument("--port", required=True, help="Stable serial path, preferably /dev/serial/by-id/…")
    parser.add_argument("--side", choices=sorted(_REFERENCE_POSES), default="single")
    parser.add_argument("--output", required=True, type=Path, help="Calibration JSON to upload in Studio")
    parser.add_argument("--baudrate", type=int, default=57_600)
    parser.add_argument("--force", action="store_true", help="Allow overwriting --output")
    args = parser.parse_args(argv)
    sdk = importlib.import_module("dynamixel_sdk")
    port_handler = sdk.PortHandler(args.port)
    packet_handler = sdk.PacketHandler(2.0)
    if not port_handler.openPort() or not port_handler.setBaudRate(args.baudrate):
        port_handler.closePort()
        msg = f"Could not open {args.port!r} at {args.baudrate} baud"
        raise ConnectionError(msg)
    try:
        for motor_id in range(1, 9):
            _model, comm_result, error = packet_handler.ping(port_handler, motor_id)
            if comm_result != _COMM_SUCCESS or error:
                msg = f"GELLO motor ID {motor_id} did not respond"
                raise ConnectionError(msg)
            comm_result, error = packet_handler.write1ByteTxRx(port_handler, motor_id, _TORQUE_ENABLE_ADDRESS, 0)
            if comm_result != _COMM_SUCCESS or error:
                msg = f"Could not disable torque for GELLO motor ID {motor_id}"
                raise ConnectionError(msg)
        input(f"Move the {args.side} GELLO to its documented reference pose, then press Enter. ")
        arm_raw = np.asarray([_read_rad(packet_handler, port_handler, motor_id) for motor_id in range(1, 8)])
        input("Release the trigger fully (open), then press Enter. ")
        gripper_open = _read_rad(packet_handler, port_handler, 8)
        input("Press the trigger fully (closed), then press Enter. ")
        gripper_closed = _read_rad(packet_handler, port_handler, 8)
        calibration = calculate_calibration(
            arm_raw,
            reference_pose_rad=_REFERENCE_POSES[args.side],
            gripper_closed_rad=gripper_closed,
            gripper_open_rad=gripper_open,
            baudrate=args.baudrate,
        )
        print(
            json.dumps(
                {
                    "joint_signs": calibration.joint_signs,
                    "assembly_offsets_rad": calibration.assembly_offsets_rad,
                    "gripper_range_rad": calibration.gripper_range_rad,
                    "baudrate": calibration.baudrate,
                },
                indent=2,
            )
        )
        if input(f"Write this calibration to {args.output}? [y/N] ").strip().lower() != "y":
            print("Calibration was not written.")
            return 1
        _write_calibration(args.output, calibration, force=args.force)
    finally:
        port_handler.closePort()
    print(f"Wrote calibration to {args.output}")
    return 0
