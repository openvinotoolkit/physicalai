# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for runtime examples — robot and camera construction."""

from __future__ import annotations

import argparse
import sys

from physicalai.capture.camera import Camera
from physicalai.robot.interface import Robot


def build_robot(args: argparse.Namespace) -> Robot:
    """Construct a robot from CLI args (--robot, --port, --ip, etc.)."""
    if args.robot == "so101":
        from physicalai.robot import SO101

        if not args.port:
            sys.exit("error: --port is required for so101")
        if not args.calibration:
            sys.exit("error: --calibration is required for so101")
        return SO101(port=args.port, calibration=args.calibration, role="follower")

    if args.robot == "widowxai":
        from physicalai.robot import WidowXAI

        if not args.ip:
            sys.exit("error: --ip is required for widowxai")
        return WidowXAI(ip=args.ip, role="follower")

    if args.robot == "bimanual_widowxai":
        from physicalai.robot import BimanualWidowXAI, WidowXAI

        if not args.ip_left or not args.ip_right:
            sys.exit("error: --ip-left and --ip-right are required for bimanual")
        left = WidowXAI(ip=args.ip_left, role="follower")
        right = WidowXAI(ip=args.ip_right, role="follower")
        return BimanualWidowXAI(left, right)

    sys.exit(f"error: unknown robot type: {args.robot}")


def parse_camera_specs(
    specs: list[str],
    width: int,
    height: int,
    fps: int,
) -> dict[str, Camera]:
    """Parse CLI camera specs into a SharedCamera dict.

    Each spec is "name:driver:device_id", e.g.:
        --camera overhead:uvc:/dev/video0
        --camera arm:realsense:353322271391
    """
    from physicalai.capture import create_camera

    cameras: dict[str, Camera] = {}
    for spec in specs:
        parts = spec.split(":", 2)
        if len(parts) != 3:
            sys.exit(f"error: invalid camera spec '{spec}'. Expected name:driver:device_id")
        name, driver, device_id = parts
        kwargs: dict = {"width": width, "height": height, "fps": fps}
        if driver == "realsense":
            kwargs["serial_number"] = device_id
        else:
            kwargs["device"] = device_id
        cameras[name] = create_camera(driver, shared=True, **kwargs)
    return cameras
