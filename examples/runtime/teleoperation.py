#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Teleoperate a follower robot from a leader robot with runtime callbacks.

Prerequisites::

    uv sync --extra capture --extra robots --extra observer-rerun

Examples:

uv run --extra observer-rerun --extra robots --extra capture examples/runtime/teleoperation.py \
  --robot so101 \
  --leader-port /dev/ttyACM0 \
  --follower-port /dev/ttyACM1 \
  --leader-calibration /home/mark/.cache/physicalai/robots/leader/calibrations/leader.json \
  --follower-calibration /home/mark/.cache/physicalai/robots/follower/calibrations/follower.json \
  --camera overhead:uvc:/dev/video34 \
  --camera arm:uvc:/dev/video32 \
  --fps 30 \
  --rerun spawn \
  --rerun-image-decimation 15 \
  --rerun-jpeg-quality 75 \
  --rerun-image-max-dim 480 \
  --duration-s 30

uv run --extra observer-rerun --extra robots --extra capture examples/runtime/teleoperation.py \
  --robot widowxai \
  --leader-ip 192.168.1.2 \
  --follower-ip 192.168.1.3 \
  --camera front:uvc:/dev/video0 \
  --fps 30 \
  --rerun spawn \
  --duration-s 90

uv run --extra observer-rerun --extra robots --extra capture examples/runtime/teleoperation.py \
  --robot bimanual_widowxai \
  --leader-ip-left 192.168.1.2 \
  --leader-ip-right 192.168.1.3 \
  --follower-ip-left 192.168.1.4 \
  --follower-ip-right 192.168.1.5 \
  --fps 30 \
  --duration-s 90
"""

from __future__ import annotations

import argparse
import signal

from physicalai.capture import select_cameras_interactive
from physicalai.robot import Robot, connect
from physicalai.runtime import (
    ActionQueue,
    PolicyRuntime,
    RerunCallback,
    SyncExecution,
    TeleoperatorPolicy,
)

from utils import parse_camera_specs


def build_teleop_robots(args: argparse.Namespace) -> tuple[Robot, Robot]:
    """Construct leader and follower robots from CLI args."""
    if args.robot == "so101":
        from physicalai.robot import SO101

        if not args.leader_port:
            raise SystemExit("error: --leader-port is required for so101")
        if not args.follower_port:
            raise SystemExit("error: --follower-port is required for so101")
        if not args.leader_calibration:
            raise SystemExit("error: --leader-calibration is required for so101")
        if not args.follower_calibration:
            raise SystemExit("error: --follower-calibration is required for so101")
        leader = SO101(port=args.leader_port, calibration=args.leader_calibration, role="leader")
        follower = SO101(port=args.follower_port, calibration=args.follower_calibration, role="follower")
        return leader, follower

    if args.robot == "widowxai":
        from physicalai.robot import WidowXAI

        if not args.leader_ip:
            raise SystemExit("error: --leader-ip is required for widowxai")
        if not args.follower_ip:
            raise SystemExit("error: --follower-ip is required for widowxai")
        return WidowXAI(ip=args.leader_ip, role="leader"), WidowXAI(ip=args.follower_ip, role="follower")

    if args.robot == "bimanual_widowxai":
        from physicalai.robot import BimanualWidowXAI, WidowXAI

        if not args.leader_ip_left or not args.leader_ip_right:
            raise SystemExit("error: --leader-ip-left and --leader-ip-right are required for bimanual_widowxai")
        if not args.follower_ip_left or not args.follower_ip_right:
            raise SystemExit("error: --follower-ip-left and --follower-ip-right are required for bimanual_widowxai")
        leader = BimanualWidowXAI(
            WidowXAI(ip=args.leader_ip_left, role="leader"),
            WidowXAI(ip=args.leader_ip_right, role="leader"),
        )
        follower = BimanualWidowXAI(
            WidowXAI(ip=args.follower_ip_left, role="follower"),
            WidowXAI(ip=args.follower_ip_right, role="follower"),
        )
        return leader, follower

    raise SystemExit(f"error: unknown robot type: {args.robot}")


def main() -> None:
    """Run the teleoperation example."""

    def _handle_sigint(sig: int, frame: object) -> None:
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        print("\nInterrupting... press Ctrl+C again to force kill.")
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _handle_sigint)

    parser = argparse.ArgumentParser(
        description="Teleoperate a follower robot from a leader robot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    robot_group = parser.add_argument_group("robot")
    robot_group.add_argument("--robot", required=True, choices=("so101", "widowxai", "bimanual_widowxai"))
    robot_group.add_argument("--leader-port", help="Leader serial port (so101)")
    robot_group.add_argument("--follower-port", help="Follower serial port (so101)")
    robot_group.add_argument("--leader-calibration", help="Leader calibration JSON path (so101)")
    robot_group.add_argument("--follower-calibration", help="Follower calibration JSON path (so101)")
    robot_group.add_argument("--leader-ip", help="Leader robot IP (widowxai)")
    robot_group.add_argument("--follower-ip", help="Follower robot IP (widowxai)")
    robot_group.add_argument("--leader-ip-left", help="Left leader IP (bimanual_widowxai)")
    robot_group.add_argument("--leader-ip-right", help="Right leader IP (bimanual_widowxai)")
    robot_group.add_argument("--follower-ip-left", help="Left follower IP (bimanual_widowxai)")
    robot_group.add_argument("--follower-ip-right", help="Right follower IP (bimanual_widowxai)")

    cam_group = parser.add_argument_group("cameras")
    cam_group.add_argument(
        "--camera", action="append", dest="cameras", metavar="NAME:DRIVER:DEVICE",
        help="Camera as name:driver:device_id (repeatable). Omit for interactive selection.",
    )
    cam_group.add_argument("--cam-width", type=int, default=640, help="Camera width (default: 640)")
    cam_group.add_argument("--cam-height", type=int, default=480, help="Camera height (default: 480)")
    cam_group.add_argument("--cam-fps", type=int, default=30, help="Camera FPS (default: 30)")

    rt_group = parser.add_argument_group("runtime")
    rt_group.add_argument("--fps", type=float, default=30.0, help="Control loop FPS (default: 30)")
    rt_group.add_argument("--duration-s", type=float, default=60.0, help="Duration in seconds")
    rt_group.add_argument(
        "--shared-camera",
        action="store_true",
        help="Use shared memory cameras (iceoryx2) — faster but incompatible with debugger",
    )

    rr_group = parser.add_argument_group("rerun")
    rr_group.add_argument("--rerun", choices=("off", "spawn", "connect", "save"), default="off")
    rr_group.add_argument("--rerun-addr", default="127.0.0.1:9876")
    rr_group.add_argument("--rerun-save-path", default="teleop.rrd")
    rr_group.add_argument("--rerun-no-images", action="store_true", help="Scalars only")
    rr_group.add_argument("--rerun-image-decimation", type=int, default=1, help="Only send 1/N frames to Rerun")
    rr_group.add_argument(
        "--rerun-jpeg-quality",
        type=int,
        default=None,
        help="JPEG quality for Rerun images (0-100, default: no re-encoding)",
    )
    rr_group.add_argument(
        "--rerun-image-max-dim",
        type=int,
        default=None,
        help="Max width/height for Rerun images (default: no resizing)",
    )

    args = parser.parse_args()

    leader, follower = build_teleop_robots(args)
    if args.cameras:
        cameras = parse_camera_specs(args.cameras, args.cam_width, args.cam_height, args.cam_fps, shared=args.shared_camera)
    else:
        cameras = select_cameras_interactive(args.cam_width, args.cam_height, args.cam_fps)

    callbacks: list = []
    if args.rerun != "off":
        callbacks.append(
            RerunCallback(
                cameras=cameras,
                image_decimation=args.rerun_image_decimation,
                log_images=not args.rerun_no_images,
                image_jpeg_quality=args.rerun_jpeg_quality,
                image_max_dim=args.rerun_image_max_dim,
                mode=args.rerun,
                connect_addr=args.rerun_addr,
                save_path=args.rerun_save_path if args.rerun == "save" else None,
            )
        )

    runtime = PolicyRuntime(
        robot=follower,
        model=TeleoperatorPolicy(leader),
        execution=SyncExecution(fps=int(args.fps), request_threshold=1.0),
        action_queue=ActionQueue(),
        cameras=cameras,
        fps=args.fps,
        callbacks=callbacks,
    )

    with connect(leader), runtime:
        for name, cam in cameras.items():
            w = getattr(cam, "actual_width", None)
            h = getattr(cam, "actual_height", None)
            f = getattr(cam, "actual_fps", None)
            print(f"  {name}: {w}x{h} @ {f}fps" if w and h else f"  {name}: connected")
        print(f"Teleoperating {args.robot} at {args.fps} fps for {args.duration_s}s...")
        stats = runtime.run(duration_s=args.duration_s)

    print(f"\nDone — {stats.steps} steps, {stats.inference_count} leader reads, {stats.total_holds} holds")


if __name__ == "__main__":
    main()
