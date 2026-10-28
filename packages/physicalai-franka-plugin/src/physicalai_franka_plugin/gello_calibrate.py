# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: ANN401, DOC501, DOC201, T201

"""Non-actuating calibration command for Franka GELLO leaders.

Three ways to find the assembly offsets, none of which moves anything:

* reference pose (default): the GELLO rests in the documented calibration pose;
  offsets are rounded to the 90 deg motor-flange increments (as upstream
  ``get_offsets.py``);
* ``--robot HOST``: match a robot's current pose (for example with every joint
  on its marks), one joint at a time; exact offsets, so small assembly errors
  are removed too. The robot is only read, never commanded;
* ``--mirror-port``/``--mirror-calibration``: both Duo GELLOs rest on their
  pins; this GELLO is set to the mirror image of the other, already calibrated
  one (J1, J3, J5 and J7 change sign).
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import select
import sys
from pathlib import Path
from typing import Any

import numpy as np

from physicalai_franka_plugin.gello import GelloCalibration, GelloLeader, present_position_counts

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
    if abs(gripper_open_rad - gripper_closed_rad) < 0.1:  # ruff: ignore[PLR2004] - stock travel is ~1.2 rad
        msg = "Trigger open and pressed readings are the same; fully release, then fully press the trigger"
        raise ValueError(msg)
    signs = np.asarray(_JOINT_SIGNS)
    # Invert the driver's mapping q = (raw - offset) * sign - pi (upstream convention) at q = reference,
    # rounding to the motor-flange increment of pi/2: offset = sign * (raw * sign - pi - reference).
    difference = arm_raw_rad * signs - math.pi - np.asarray(reference_pose_rad)
    offsets = np.mod(signs * np.round(difference / (math.pi / 2)) * (math.pi / 2), 2 * math.pi)
    return GelloCalibration(
        joint_signs=_JOINT_SIGNS,
        assembly_offsets_rad=tuple(float(value) for value in offsets),  # type: ignore[arg-type]
        gripper_range_rad=(gripper_closed_rad, gripper_open_rad),
        baudrate=baudrate,
    )


# Joints whose angle changes sign between the left and the right arm of a mirrored Duo.
MIRROR_SIGNS = (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0)


def joint_midpoints_rad() -> np.ndarray:
    """Midpoints of the joint ranges the driver wraps GELLO readings around."""
    return GelloLeader._LIMITS_RAD.mean(axis=1)  # ruff: ignore[SLF001] - same window as the driver


def normalize_raw(arm_raw_rad: np.ndarray, offsets_rad: np.ndarray, signs: np.ndarray) -> np.ndarray:
    """Map raw motor angles to FR3 joint angles exactly as :class:`GelloLeader` does (before clipping)."""
    mid = joint_midpoints_rad()
    return np.mod((np.asarray(arm_raw_rad) - offsets_rad) * signs - mid, 2 * math.pi) - math.pi + mid


def matching_offsets(arm_raw_rad: np.ndarray, target_rad: np.ndarray, signs: np.ndarray) -> np.ndarray:
    """Exact offsets (not rounded) that make the GELLO read ``target_rad`` at ``arm_raw_rad``.

    Inverts ``q = (raw - offset) * sign - pi`` (mod 2 pi), the driver's mapping.
    """
    return np.mod(np.asarray(arm_raw_rad) - signs * (np.asarray(target_rad) + math.pi), 2 * math.pi)


def calculate_matched_calibration(
    arm_raw_rad: np.ndarray,
    *,
    target_rad: np.ndarray,
    gripper_closed_rad: float,
    gripper_open_rad: float,
    baudrate: int,
    joint_signs: tuple[int, int, int, int, int, int, int] = _JOINT_SIGNS,
) -> GelloCalibration:
    """Calibration whose offsets make the GELLO read exactly ``target_rad`` at ``arm_raw_rad``."""
    if np.shape(arm_raw_rad) != (7,) or np.shape(target_rad) != (7,):
        msg = "Expected seven raw positions and seven target angles"
        raise ValueError(msg)
    if abs(gripper_open_rad - gripper_closed_rad) < 0.1:  # ruff: ignore[PLR2004] - stock travel is ~1.2 rad
        msg = "Trigger open and pressed readings are the same; fully release, then fully press the trigger"
        raise ValueError(msg)
    offsets = matching_offsets(arm_raw_rad, target_rad, np.asarray(joint_signs, dtype=float))
    return GelloCalibration(
        joint_signs=joint_signs,
        assembly_offsets_rad=tuple(float(value) for value in offsets),  # type: ignore[arg-type]
        gripper_range_rad=(gripper_closed_rad, gripper_open_rad),
        baudrate=baudrate,
    )


def mirror_pose(joints_rad: np.ndarray) -> np.ndarray:
    """Mirror a Duo arm's joint angles to the other side (J1, J3, J5, J7 change sign)."""
    return np.asarray(joints_rad) * np.asarray(MIRROR_SIGNS)


def _read_rad(packet_handler: Any, port_handler: Any, motor_id: int) -> float:
    value, comm_result, error = packet_handler.read4ByteTxRx(port_handler, motor_id, _PRESENT_POSITION_ADDRESS)
    if comm_result != _COMM_SUCCESS or error:
        msg = f"Failed to read GELLO motor ID {motor_id}"
        raise ConnectionError(msg)
    return float(present_position_counts(value)) * _RAD_PER_COUNT


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


def _open_bus(port: str, baudrate: int) -> tuple[Any, Any]:
    """Open a GELLO bus, check IDs 1-8 answer and switch their torque off."""
    sdk = importlib.import_module("dynamixel_sdk")
    port_handler = sdk.PortHandler(port)
    packet_handler = sdk.PacketHandler(2.0)
    if not port_handler.openPort() or not port_handler.setBaudRate(baudrate):
        port_handler.closePort()
        msg = f"Could not open {port!r} at {baudrate} baud"
        raise ConnectionError(msg)
    try:
        for motor_id in range(1, 9):
            _check_motor(packet_handler, port_handler, motor_id, port)
    except Exception:
        port_handler.closePort()
        raise
    return port_handler, packet_handler


def _check_motor(packet_handler: Any, port_handler: Any, motor_id: int, port: str) -> None:
    """Ping one motor and switch its torque off."""
    _model, comm_result, error = packet_handler.ping(port_handler, motor_id)
    if comm_result != _COMM_SUCCESS or error:
        msg = f"GELLO motor ID {motor_id} did not respond on {port!r} (another program holding the port?)"
        raise ConnectionError(msg)
    comm_result, error = packet_handler.write1ByteTxRx(port_handler, motor_id, _TORQUE_ENABLE_ADDRESS, 0)
    if comm_result != _COMM_SUCCESS or error:
        msg = f"Could not disable torque for GELLO motor ID {motor_id}"
        raise ConnectionError(msg)


def _read_arm_raw(packet_handler: Any, port_handler: Any, samples: int = 10) -> np.ndarray:
    """Average ``samples`` reads of the seven arm motors (rad)."""
    reads = [[_read_rad(packet_handler, port_handler, motor_id) for motor_id in range(1, 8)] for _ in range(samples)]
    return np.mean(np.asarray(reads), axis=0)


def _read_robot_rad(host: str) -> np.ndarray:
    """Read an FR3's joint angles once over the FCI; nothing is commanded."""
    pylibfranka = importlib.import_module("pylibfranka")
    robot = pylibfranka.Robot(host, pylibfranka.RealtimeConfig.kIgnore)
    return np.asarray(robot.read_once().q, dtype=np.float64)


def _match_joints_interactively(
    packet_handler: Any, port_handler: Any, target_rad: np.ndarray, signs: np.ndarray, start_offsets: np.ndarray
) -> np.ndarray:
    """Capture one joint at a time while showing its live deviation from the target."""
    arm_raw = _read_arm_raw(packet_handler, port_handler, samples=1)
    for joint in range(7):
        print(f"\nJ{joint + 1}: line it up (e.g. on its marks), hold it, press Enter")
        while True:
            raw = _read_rad(packet_handler, port_handler, joint + 1)
            now = normalize_raw(np.full(7, raw), start_offsets, signs)[joint]
            delta = math.degrees((now - target_rad[joint] + math.pi) % (2 * math.pi) - math.pi)
            sys.stdout.write(
                f"\r  GELLO {math.degrees(now):8.2f}  target {math.degrees(target_rad[joint]):8.2f}"
                f"  delta {delta:+7.2f} deg   "
            )
            sys.stdout.flush()
            if select.select([sys.stdin], [], [], 0.1)[0]:
                sys.stdin.readline()
                break
        arm_raw[joint] = np.mean([_read_rad(packet_handler, port_handler, joint + 1) for _ in range(10)])
    print()
    return arm_raw


def main(argv: list[str] | None = None) -> int:  # ruff: ignore[PLR0914, PLR0915]
    """Run interactive, non-actuating GELLO calibration and write Studio-uploadable JSON."""
    parser = argparse.ArgumentParser(
        description="Calibrate a passive Franka GELLO leader without enabling torque or moving a robot."
    )
    parser.add_argument("--port", required=True, help="Stable serial path, preferably /dev/serial/by-id/…")
    parser.add_argument("--side", choices=sorted(_REFERENCE_POSES), default="single")
    parser.add_argument("--output", required=True, type=Path, help="Calibration JSON to upload in Studio")
    parser.add_argument("--baudrate", type=int, default=57_600)
    parser.add_argument("--force", action="store_true", help="Allow overwriting --output")
    method = parser.add_mutually_exclusive_group()
    method.add_argument("--robot", help="Match this FR3's current pose joint by joint (read only), e.g. on its marks")
    method.add_argument("--mirror-port", help="Serial path of the other, already calibrated GELLO of the Duo")
    parser.add_argument("--mirror-calibration", type=Path, help="Calibration JSON of the --mirror-port GELLO")
    args = parser.parse_args(argv)
    if args.mirror_port and not args.mirror_calibration:
        parser.error("--mirror-port needs --mirror-calibration")
    signs = np.asarray(_JOINT_SIGNS, dtype=float)
    port_handler, packet_handler = _open_bus(args.port, args.baudrate)
    try:
        if args.robot:
            target = _read_robot_rad(args.robot)
            print(f"Robot {args.robot} pose (deg, read only): {np.round(np.degrees(target), 1).tolist()}")
            start = np.asarray(GelloCalibration.default().assembly_offsets_rad)
            arm_raw = _match_joints_interactively(packet_handler, port_handler, target, signs, start)
        elif args.mirror_port:
            input("Rest both GELLOs on their pins (white dots on J2 up), then press Enter. ")
            other = GelloCalibration.from_json(args.mirror_calibration.read_text(encoding="utf-8"))
            other_ports = _open_bus(args.mirror_port, other.baudrate)
            try:
                other_raw = _read_arm_raw(other_ports[1], other_ports[0])
            finally:
                other_ports[0].closePort()
            other_pose = normalize_raw(
                other_raw, np.asarray(other.assembly_offsets_rad), np.asarray(other.joint_signs, dtype=float)
            )
            target = mirror_pose(other_pose)
            print(f"Other GELLO rests at (deg): {np.round(np.degrees(other_pose), 1).tolist()}")
            arm_raw = _read_arm_raw(packet_handler, port_handler)
        else:
            input(f"Move the {args.side} GELLO to its documented reference pose, then press Enter. ")
            arm_raw = _read_arm_raw(packet_handler, port_handler)
            target = None
        input("Release the trigger fully (open), then press Enter. ")
        gripper_open = _read_rad(packet_handler, port_handler, 8)
        input("Press the trigger fully (closed), then press Enter. ")
        gripper_closed = _read_rad(packet_handler, port_handler, 8)
        if target is None:
            calibration = calculate_calibration(
                arm_raw,
                reference_pose_rad=_REFERENCE_POSES[args.side],
                gripper_closed_rad=gripper_closed,
                gripper_open_rad=gripper_open,
                baudrate=args.baudrate,
            )
        else:
            calibration = calculate_matched_calibration(
                arm_raw,
                target_rad=target,
                gripper_closed_rad=gripper_closed,
                gripper_open_rad=gripper_open,
                baudrate=args.baudrate,
            )
            default = np.asarray(GelloCalibration.default().assembly_offsets_rad)
            change = (np.asarray(calibration.assembly_offsets_rad) - default + math.pi) % (2 * math.pi) - math.pi
            print(f"Offset change from the factory values (deg): {np.round(np.degrees(change), 1).tolist()}")
            if np.any(np.abs(np.degrees(change)) > 45):  # ruff: ignore[PLR2004] - beyond half a flange step
                print("WARNING: a joint moved by more than 45 deg: wrong pose, swapped GELLOs or a flipped joint?")
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
