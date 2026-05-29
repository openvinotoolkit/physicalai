#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Real-Time Chunking (RTC) inference with PolicyRuntime.

Demonstrates how to run a Pi0.5 model with RTC denoising baked into
the graph. The model produces 50-action chunks; the RTCExecution
strategy handles async inference, dual-track queue management, and
latency-aware delay compensation.

Examples:

    # SO101 with 2 cameras
    python examples/runtime/rtc_inference.py \
      --robot so101 --port /dev/ttyACM0 --calibration ./cal.json \
      --model ./exports/pi05_rtc_openvino --device GPU.0 \
      --camera overhead:uvc:/dev/v4l/by-id/usb-... \
      --camera arm:uvc:/dev/v4l/by-id/usb-... \
      --fps 30 --duration-s 60

    # Trossen WidowXAI
    python examples/runtime/rtc_inference.py \
      --robot widowxai --ip 192.168.1.2 \
      --model ./exports/pi05_rtc_openvino \
      --camera front:uvc:/dev/video0

    # No --camera args → interactive selection
    python examples/runtime/rtc_inference.py \
      --robot so101 --port /dev/ttyACM0 --calibration ./cal.json \
      --model ./exports/pi05_rtc_openvino --rerun spawn
"""

from __future__ import annotations

import argparse
import signal

from physicalai.capture import select_cameras_interactive
from physicalai.inference import InferenceModel
from physicalai.inference.callbacks import RTCLatencyTracker
from physicalai.runtime import (
    LowPassFilterCallback,
    PolicyRuntime,
    RTCActionQueue,
    RTCExecution,
    RerunCallback,
)

from utils import build_robot, parse_camera_specs, prompt_torque_disable


def main() -> None:
    def _handle_sigint(sig: int, frame: object) -> None:
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        print("\nInterrupting... press Ctrl+C again to force kill.")
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _handle_sigint)

    parser = argparse.ArgumentParser(
        description="Run Pi0.5 RTC policy with PolicyRuntime",
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
    model_group.add_argument("--model", required=True, help="Exported RTC model directory")
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
    rt_group.add_argument("--duration-s", type=float, default=None, help="Run duration in seconds (default: run indefinitely)")
    rt_group.add_argument("--task", type=str, default=None, help="Task string for the model (e.g. 'pick up the can')")
    rt_group.add_argument("--shared-camera", action="store_true", help="Use shared memory cameras (iceoryx2) — faster but incompatible with debugger")

    # RTC parameters
    rtc_group = parser.add_argument_group("rtc")
    rtc_group.add_argument("--chunk-size", type=int, default=50)
    rtc_group.add_argument("--execution-horizon", type=int, default=10)
    rtc_group.add_argument("--max-action-dim", type=int, default=32)
    rtc_group.add_argument("--max-guidance-weight", type=float, default=10.0)
    rtc_group.add_argument("--queue-threshold", type=int, default=30)
    rtc_group.add_argument("--low-pass-alpha", type=float, default=None, help="Alpha parameter for stateful LowPassFilterCallback. E.g. 0.5. Defaults to None (disabled).")

    # Rerun
    rr_group = parser.add_argument_group("rerun")
    rr_group.add_argument("--rerun", choices=("off", "spawn", "connect", "save"), default="off")
    rr_group.add_argument("--rerun-addr", default="127.0.0.1:9876")
    rr_group.add_argument("--rerun-save-path", default="run.rrd")
    rr_group.add_argument("--rerun-no-images", action="store_true", help="Scalars only")
    rr_group.add_argument("--rerun-image-decimation", type=int, default=1, help="Only send 1/N frames to Rerun")
    rr_group.add_argument("--rerun-jpeg-quality", type=int, default=None, help="JPEG quality for Rerun images (0-100, default: no re-encoding)")
    rr_group.add_argument("--rerun-image-max-dim", type=int, default=None, help="Max width/height for Rerun images (default: no resizing)")

    args = parser.parse_args()

    # ── Load model ──
    import openvino_tokenizers  # noqa: F401 — registers OV tokenizer ops

    latency_tracker = RTCLatencyTracker(window_size=100)

    print(f"Loading model from {args.model} on {args.device} (this may take a minute)...", flush=True)
    model = InferenceModel.load(
        args.model,
        device=args.device,
        callbacks=[latency_tracker],
    )
    print("Model loaded.")

    # ── Build robot & cameras ──
    robot = build_robot(args)
    if args.cameras:
        cameras = parse_camera_specs(args.cameras, args.cam_width, args.cam_height, args.cam_fps, shared=args.shared_camera)
    else:
        cameras = select_cameras_interactive(args.cam_width, args.cam_height, args.cam_fps)

    # ── RTC queue and execution ──
    rtc_queue = RTCActionQueue()
    execution = RTCExecution(
        execution_horizon=args.execution_horizon,
        fps=args.fps,
        max_guidance_weight=args.max_guidance_weight,
        latency_tracker=latency_tracker,
    )

    # ── Callbacks ──
    callbacks: list = []
    if args.low_pass_alpha is not None:
        print(f"Applying LowPassFilterCallback with alpha={args.low_pass_alpha}")
        callbacks.append(LowPassFilterCallback(alpha=args.low_pass_alpha))
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
        execution=execution,
        action_queue=rtc_queue,
        cameras=cameras,
        fps=args.fps,
        callbacks=callbacks,
        task=args.task,
    )

    with runtime:
        for name, cam in cameras.items():
            w = getattr(cam, "actual_width", None)
            h = getattr(cam, "actual_height", None)
            f = getattr(cam, "actual_fps", None)
            print(f"  {name}: {w}x{h} @ {f}fps" if w and h else f"  {name}: connected")
        print(
            f"Running RTC — chunk={args.chunk_size}, "
            f"horizon={args.execution_horizon}, fps={args.fps}"
        )
        if args.task:
            print(f"  task: {args.task!r}")
        stats = runtime.run(duration_s=args.duration_s)
        print(
            f"\nDone — {stats.steps} steps, {stats.inference_count} inferences, "
            f"{stats.total_holds} holds"
        )
        print(
            f"Latency — max={latency_tracker.max_latency_s:.3f}s, "
            f"p95={latency_tracker.percentile_s(95):.3f}s"
        )
        prompt_torque_disable(robot)


if __name__ == "__main__":
    main()
