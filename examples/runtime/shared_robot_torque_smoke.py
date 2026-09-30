#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Check SO101 leader torque control through SharedRobot on hardware."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from physicalai.robot import SO101, SharedRobot
from physicalai.robot.interface import RobotObservation


def wait_for_fresh_observation(robot: SharedRobot, after: float, timeout: float) -> RobotObservation:
    """Wait for an owner-published observation newer than ``after``."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        observation = robot.get_observation()
        if observation.timestamp > after:
            return observation
        time.sleep(0.02)
    raise TimeoutError("No fresh SO101 observation arrived before timeout")


def wait_for_joint_target(
    robot: SharedRobot,
    target: np.ndarray,
    joint_index: int,
    after: float,
    tolerance: float,
    timeout: float,
) -> float:
    """Wait until a fresh observation reaches one commanded joint target."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        observation = robot.get_observation()
        if observation.timestamp > after:
            after = observation.timestamp
            actual = float(observation.joint_positions[joint_index])
            if abs(actual - float(target[joint_index])) <= tolerance:
                return actual
        time.sleep(0.02)
    raise TimeoutError("The leader did not reach the test target before timeout")


def main() -> None:
    """Exercise acknowledged torque control and, optionally, one tiny move."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True, help="SO101 serial port")
    parser.add_argument("--calibration", required=True, type=Path, help="Calibrated SO101 leader JSON")
    parser.add_argument("--name", required=True, help="Unique SharedRobot owner name")
    parser.add_argument(
        "--joint",
        choices=("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"),
        help="Optional joint for a small physical movement test",
    )
    parser.add_argument("--delta", type=float, default=1.0, help="Movement in normalized units; maximum 1")
    parser.add_argument("--timeout", type=float, default=8.0)
    parser.add_argument("--tolerance", type=float, default=0.25)
    args = parser.parse_args()

    calibration = args.calibration.expanduser()
    if not calibration.is_file():
        parser.error(f"Calibration file not found: {calibration}")
    if abs(args.delta) > 1.0:
        parser.error("--delta is limited to 1 normalized unit")
    if args.timeout <= 0 or args.tolerance <= 0:
        parser.error("--timeout and --tolerance must be positive")

    driver = SO101(
        port=args.port,
        role="leader",
        unit="normalized",
        calibration=calibration,
    )
    robot = SharedRobot.from_config(driver, name=args.name, idle_timeout=2.0)
    torque_requested = False
    cleanup_error: Exception | None = None

    try:
        robot.connect()
        before = robot.get_observation()
        print("Joints:", robot.joint_names)
        print("Starting pose:", np.asarray(before.joint_positions).round(2).tolist())
        if input("Clear the arm, then type ENABLE to turn leader torque on: ") != "ENABLE":
            raise SystemExit("Cancelled before enabling torque")

        # Set before the request because a lost reply leaves torque state uncertain.
        torque_requested = True
        robot.set_torque(enabled=True)
        print("PASS: torque enable acknowledged")
        current = wait_for_fresh_observation(robot, before.timestamp, args.timeout)
        target = np.asarray(current.joint_positions, dtype=np.float32).copy()

        if args.joint:
            joint_index = robot.joint_names.index(args.joint)
            lower, upper = (0.0, 100.0) if args.joint == "gripper" else (-100.0, 100.0)
            target[joint_index] = np.clip(target[joint_index] + args.delta, lower, upper)
            if np.isclose(target[joint_index], current.joint_positions[joint_index]):
                raise ValueError("The requested delta is already at this joint's limit")
            print(f"Target {args.joint}: {current.joint_positions[joint_index]:.2f} -> {target[joint_index]:.2f}")
            if input("Type MOVE to send this small test movement: ") != "MOVE":
                raise SystemExit("Cancelled before sending the movement")
        else:
            print("No movement requested; sending the current pose as a position command.")

        robot.send_action(target)
        if args.joint:
            actual = wait_for_joint_target(
                robot,
                target,
                joint_index,
                current.timestamp,
                args.tolerance,
                args.timeout,
            )
            print(f"PASS: {args.joint} reached {actual:.2f}")
        else:
            wait_for_fresh_observation(robot, current.timestamp, args.timeout)
            print("PASS: owner state continued after the command (send_action is fire-and-forget)")
    finally:
        if torque_requested:
            try:
                robot.set_torque(enabled=False)
                print("PASS: torque disable acknowledged")
            except Exception as exc:
                cleanup_error = exc
                print(f"STOP: torque state is uncertain: {exc}", file=sys.stderr)
        try:
            robot.disconnect()
        except Exception as exc:
            cleanup_error = cleanup_error or exc
            print(f"Owner disconnect failed: {exc}", file=sys.stderr)

    if cleanup_error:
        raise SystemExit("Inspect the arm and owner before continuing")


if __name__ == "__main__":
    main()
