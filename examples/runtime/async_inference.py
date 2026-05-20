#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Run a trained policy on hardware with real-time Rerun visualization.

Examples:

    # SO101 with 3 cameras, Rerun viewer auto-launched
    python examples/runtime/async_inference.py \\
      --robot so101 --port /dev/ttyACM0 --calibration ./cal.json \\
      --model ./exports/my_model \\
      --camera overhead:uvc:/dev/video0 \\
      --camera arm:realsense:353322271391 \\
      --camera front:uvc:/dev/video2 \\
      --rerun spawn

    # Trossen WidowXAI
    python examples/runtime/async_inference.py \\
      --robot widowxai --ip 192.168.1.2 \\
      --model ./exports/my_model \\
      --camera front:uvc:/dev/video0

    # Bimanual Trossen WidowXAI
    python examples/runtime/async_inference.py \\
      --robot bimanual_widowxai --ip-left 192.168.1.2 --ip-right 192.168.1.3 \\
      --model ./exports/my_model

    # No --camera args → interactive selection
    python examples/runtime/async_inference.py \\
      --robot so101 --port /dev/ttyACM0 --calibration ./cal.json \\
      --model ./exports/my_model --rerun spawn
"""

from __future__ import annotations

import argparse

from physicalai.capture import select_cameras_interactive
from physicalai.inference import InferenceModel
from physicalai.runtime import (
    ActionQueue,
    AsyncExecution,
    LerpSmoother,
    PolicyRuntime,
    RerunCallback,
)

from utils import build_robot, parse_camera_specs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a trained policy on hardware",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Robot
    robot_group = parser.add_argument_group("robot")
    robot_group.add_argument("--robot", required=True, choices=("so101", "widowxai", "bimanual_widowxai"))
    robot_group.add_argument("--port", help="Serial port (so101)")
    robot_group.add_argument("--calibration", help="Calibration JSON path (so101)")
    robot_group.add_argument("--ip", help="Robot IP (widowxai)")
    robot_group.add_argument("--ip-left", help="Left arm IP (bimanual_widowxai)")
    robot_group.add_argument("--ip-right", help="Right arm IP (bimanual_widowxai)")

    # Model
    model_group = parser.add_argument_group("model")
    model_group.add_argument("--model", required=True, help="Exported model directory")
    model_group.add_argument("--device", default="GPU.0", help="OpenVINO device (default: GPU.0)")

    # Cameras
    cam_group = parser.add_argument_group("cameras")
    cam_group.add_argument(
        "--camera", action="append", dest="cameras", metavar="NAME:DRIVER:DEVICE",
        help="Camera as name:driver:device_id (repeatable). Omit for interactive selection.",
    )
    cam_group.add_argument("--cam-width", type=int, default=640, help="Camera width (default: 640)")
    cam_group.add_argument("--cam-height", type=int, default=480, help="Camera height (default: 480)")
    cam_group.add_argument("--cam-fps", type=int, default=30, help="Camera FPS (default: 30)")

    # Runtime
    rt_group = parser.add_argument_group("runtime")
    rt_group.add_argument("--fps", type=float, default=30.0, help="Control loop FPS (default: 30)")
    rt_group.add_argument("--duration-s", type=float, default=60.0, help="Duration in seconds")

    # Rerun
    rr_group = parser.add_argument_group("rerun")
    rr_group.add_argument("--rerun", choices=("off", "spawn", "connect", "save"), default="off")
    rr_group.add_argument("--rerun-addr", default="127.0.0.1:9876")
    rr_group.add_argument("--rerun-save-path", default="run.rrd")
    rr_group.add_argument("--rerun-no-images", action="store_true", help="Scalars only")
    rr_group.add_argument("--rerun-image-decimation", type=int, default=3)
    rr_group.add_argument("--rerun-jpeg-quality", type=int, default=None)
    rr_group.add_argument("--rerun-image-max-dim", type=int, default=None)

    args = parser.parse_args()

    # ── Load model ──
    import openvino_tokenizers  # noqa: F401 — registers OV tokenizer ops

    model = InferenceModel.load(args.model, device=args.device)

    # ── Build robot & cameras ──
    robot = build_robot(args)
    if args.cameras:
        cameras = parse_camera_specs(args.cameras, args.cam_width, args.cam_height, args.cam_fps)
    else:
        cameras = select_cameras_interactive(args.cam_width, args.cam_height, args.cam_fps)

    # ── Callbacks ──
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

    # ── Run ──
    runtime = PolicyRuntime(
        robot=robot,
        model=model,
        execution=AsyncExecution(threshold=0.3, fps=int(args.fps)),
        action_queue=ActionQueue(smoother=LerpSmoother(duration_frames=5)),
        cameras=cameras,
        fps=args.fps,
        callbacks=callbacks,
    )

    with runtime:
        for name, cam in cameras.items():
            w = getattr(cam, "actual_width", None)
            h = getattr(cam, "actual_height", None)
            f = getattr(cam, "actual_fps", None)
            print(f"  {name}: {w}x{h} @ {f}fps" if w and h else f"  {name}: connected")
        print(f"Running at {args.fps} fps for {args.duration_s}s...")
        stats = runtime.run(duration_s=args.duration_s)

    print(f"\nDone — {stats.steps} steps, {stats.inference_count} inferences, {stats.total_holds} holds")


if __name__ == "__main__":
    main()
