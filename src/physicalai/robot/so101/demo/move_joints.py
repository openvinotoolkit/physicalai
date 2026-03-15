# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Actuation smoke test for the SO-101 robot arm.

Connects as a follower, reads the current pose, then moves each joint
individually by a small offset and back. Verifies that ``send_action()``
works and that joint ordering matches the physical wiring.

Requires a calibration file (LeRobot format) so that positions are in radians.

Usage::

    python -m physicalai.robot.so101.demo.move_joints --port /dev/ttyUSB0 --calibration cal.json
    python -m physicalai.robot.so101.demo.move_joints --port /dev/ttyUSB0 --calibration cal.json --offset 0.15
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from physicalai.robot.so101.so101 import SO101

_DEFAULT_OFFSET = 0.08
"""Default movement offset in radians (~4.6 degrees)."""

_STEP_DELAY = 0.5
"""Seconds to wait after each movement so you can visually confirm it."""


def _test_joint(
    robot: SO101,
    joint_idx: int,
    name: str,
    start_pose: np.ndarray,
    offset: float,
    delay: float,
) -> bool:
    """Move a single joint by +offset and -offset, return True if either works.

    Tries both directions because one side may be blocked at a joint limit.

    Returns:
        True if the joint responded within tolerance in at least one direction.
    """
    tolerance = 0.035  # ~2 degrees in radians
    any_ok = False

    for sign, label in [(+1, "+"), (-1, "-")]:
        target = start_pose.copy()
        target[joint_idx] += sign * offset
        robot.send_action(target)
        time.sleep(delay)

        obs = robot.get_observation()
        actual = obs["state"][joint_idx]
        expected = start_pose[joint_idx] + sign * offset
        delta = abs(actual - expected)
        print(  # noqa: T201
            f"  {label}{offset:.3f} rad -> read: {actual:.4f} "
            f"(expected: {expected:.4f}, delta: {delta:.4f})"
        )

        # Return to start
        robot.send_action(start_pose)
        time.sleep(delay)

        if delta < tolerance:
            any_ok = True

    if any_ok:
        print(f"  OK {name}")  # noqa: T201
    else:
        print(f"  FAIL {name} (delta too large in both directions - check wiring or servo ID)")  # noqa: T201
    return any_ok


def main(argv: list[str] | None = None) -> None:
    """Run the SO-101 joint movement smoke test.

    Args:
        argv: Command-line arguments.  Defaults to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(
        description="SO-101 actuation smoke test: move each joint one at a time to verify wiring.",
    )
    parser.add_argument("--port", required=True, help="Serial port, e.g. /dev/ttyUSB0")
    parser.add_argument("--baudrate", type=int, default=1_000_000, help="Serial baudrate (default: 1000000)")
    parser.add_argument("--calibration", required=True, help="Path to LeRobot calibration JSON file")
    parser.add_argument(
        "--offset",
        type=float,
        default=_DEFAULT_OFFSET,
        help=f"Movement offset in radians (default: {_DEFAULT_OFFSET}, ~4.6 degrees)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=_STEP_DELAY,
        help=f"Seconds to pause after each movement (default: {_STEP_DELAY})",
    )
    args = parser.parse_args(argv)

    robot = SO101(
        port=args.port,
        baudrate=args.baudrate,
        role="follower",
        calibration_path=args.calibration,
    )

    print(f"Connecting to SO-101 on {args.port} (role=follower)...")  # noqa: T201
    robot.connect()

    # Read starting pose
    obs = robot.get_observation()
    start_pose = obs["state"].copy()
    print(  # noqa: T201
        f"Connected. Starting pose (rad): "
        f"{', '.join(f'{v:.4f}' for v in start_pose)}\n"
    )

    passed = 0
    failed = 0

    try:
        for i, name in enumerate(SO101.JOINT_ORDER):
            print(f"Testing joint {i} ({name})...")  # noqa: T201
            if _test_joint(robot, i, name, start_pose, args.offset, args.delay):
                passed += 1
            else:
                failed += 1
            print()  # noqa: T201

    except KeyboardInterrupt:
        print("\nInterrupted by user.")  # noqa: T201
    finally:
        # Return to starting pose, then release torque
        print("Returning to starting pose...")  # noqa: T201
        robot.send_action(start_pose)
        time.sleep(args.delay)
        print("Releasing torque (motors off)...")  # noqa: T201
        robot._set_torque(enabled=False)  # noqa: SLF001
        # Switch to leader so disconnect() won't re-enable torque via _hold_position()
        robot.role = "leader"
        robot.disconnect()
        print("Disconnected.")  # noqa: T201

    print(f"\nResults: {passed} passed, {failed} failed out of {SO101.NUM_JOINTS} joints.")  # noqa: T201
    if failed > 0:
        print("Check servo IDs and wiring for failed joints.")  # noqa: T201


if __name__ == "__main__":
    main()
