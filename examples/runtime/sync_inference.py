#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Run a trained policy synchronously — simplest possible control loop.

Inference blocks the control thread. The loop pauses during each inference
call, so real-time guarantees do NOT hold. Use this to verify model behaviour
before moving to async execution with lerping.

Examples:

    # Bimanual Trossen, no shared camera (debugger-safe)
    python examples/runtime/sync_inference.py \
      --robot bimanual_widowxai --ip-left 192.168.1.2 --ip-right 192.168.1.3 \
      --model ./exports/pi05_cans_openvino \
      --camera front:uvc:/dev/video0 \
      --task "pick up the can" \
      --fps 30 --duration-s 30

    # SO101
    python examples/runtime/sync_inference.py \
      --robot so101 --port /dev/ttyACM0 --calibration ./cal.json \
      --model ./exports/my_model \
      --camera overhead:uvc:/dev/video0 \
      --task "pick up the can"
"""

from __future__ import annotations

import argparse
import signal
from pathlib import Path

from physicalai.capture import select_cameras_interactive
from physicalai.config import save_yaml, to_config
from physicalai.inference import InferenceModel
from physicalai.runtime import (
    ChunkedActionQueue,
    PolicySource,
    RerunCallback,
    RobotRuntime,
    SyncExecution,
)

from utils import build_robot, parse_camera_specs, prompt_torque_disable


def main() -> None:
    def _handle_sigint(sig: int, frame: object) -> None:
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        print("\nInterrupting... press Ctrl+C again to force kill.")
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _handle_sigint)

    parser = argparse.ArgumentParser(
        description="Run a trained policy synchronously (blocking inference)",
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
    robot_group.add_argument(
        "--shared-robot",
        action="store_true",
        help="Wrap the driver in SharedRobot (Zenoh transport)",
    )
    robot_group.add_argument(
        "--robot-name",
        help="Logical SharedRobot name (required with --shared-robot)",
    )

    # Model
    model_group = parser.add_argument_group("model")
    model_group.add_argument("--model", required=True, help="Exported model directory")
    model_group.add_argument("--device", default="GPU", help="OpenVINO device (default: GPU)")

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
    rt_group.add_argument("--task", type=str, default=None, help="Task string for the model (e.g. 'pick up the can')")
    rt_group.add_argument(
        "--shared-camera",
        action="store_true",
        help="Use SharedCamera (iceoryx2 transport) — faster but incompatible with debugger",
    )
    rt_group.add_argument("--request-threshold", type=float, default=0.5, help="Request new inference when queue drops below this fraction of chunk_size (default: 0.75 = trigger when 75%% of actions remain)")
    rt_group.add_argument(
        "--export-config",
        type=Path,
        help="Save the constructed runtime as YAML and exit before connecting hardware.",
    )

    # Rerun
    rr_group = parser.add_argument_group("rerun")
    rr_group.add_argument("--rerun", choices=("off", "spawn", "connect", "save"), default="off")
    rr_group.add_argument("--rerun-addr", default="127.0.0.1:9876")
    rr_group.add_argument("--rerun-save-path", default="run.rrd")

    args = parser.parse_args()

    # ── Load model ──
    model = InferenceModel(args.model, device=args.device)

    # ── Build robot & cameras ──
    robot = build_robot(args)
    if args.cameras:
        cameras = parse_camera_specs(args.cameras, args.cam_width, args.cam_height, args.cam_fps, shared=args.shared_camera)
    else:
        cameras = select_cameras_interactive(
            args.cam_width, args.cam_height, args.cam_fps, shared=args.shared_camera,
        )

    # ── Callbacks ──
    callbacks: list = []
    if args.rerun != "off":
        callbacks.append(
            RerunCallback(
                log_images=True,
                mode=args.rerun,
                connect_addr=args.rerun_addr,
                save_path=args.rerun_save_path if args.rerun == "save" else None,
            )
        )

    # ── Run (synchronous — inference blocks the loop) ──
    execution = SyncExecution(request_threshold=args.request_threshold)
    policy_source = PolicySource(
        model=model,
        execution=execution,
        action_queue=ChunkedActionQueue(),  # no smoother — raw chunk playback
        task=args.task,
    )
    runtime = RobotRuntime(
        robot=robot,
        action_source=policy_source,
        cameras=cameras,
        fps=args.fps,
        callbacks=callbacks,
    )

    if args.export_config:
        save_yaml(to_config(runtime), args.export_config)
        print(f"Saved runtime config to {args.export_config}")
        return

    with runtime:
        print(f"Running SYNC at {args.fps} fps for {args.duration_s}s...")
        if args.task:
            print(f"  task: {args.task!r}")
        print("  (inference blocks the loop — expect pauses)")
        steps = runtime.run(duration_s=args.duration_s)
        print(f"\nDone — {steps} steps, {execution.inference_count} inferences, "
              f"{policy_source.action_queue.total_holds} holds")
        prompt_torque_disable(robot)


if __name__ == "__main__":
    main()
