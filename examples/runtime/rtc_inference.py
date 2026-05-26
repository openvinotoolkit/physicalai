#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Real-Time Chunking (RTC) inference with PolicyRuntime.

Demonstrates how to run a Pi0.5 model with RTC denoising baked into
the graph. The model produces 50-action chunks; the RTCExecution
strategy handles async inference, dual-track queue management, and
latency-aware delay compensation.

Usage:
    python examples/runtime/rtc_inference.py \
      --model ./exports/pi05_rtc_openvino \
      --device GPU.0 \
      --port /dev/ttyACM0 \
      --calibration /path/to/calibration.json \
      --overhead-camera /dev/v4l/by-id/usb-... \
      --arm-camera 353322271391 \
      --fps 30 \
      --duration-s 60
"""

from __future__ import annotations

import argparse

from physicalai.capture.transport import SharedCamera
from physicalai.inference import InferenceModel
from physicalai.inference.callbacks import RTCLatencyTracker
from physicalai.robot import SO101
from physicalai.runtime import (
    PolicyRuntime,
    RTCActionQueue,
    RTCExecution,
)
from physicalai.inference.component_factory import instantiate_component, resolve_artifact
from physicalai.inference.manifest import Manifest
import openvino_tokenizers  # noqa: F401 — registers OV tokenizer ops


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Pi0.5 RTC policy with PolicyRuntime")
    parser.add_argument("--model", required=True, help="Exported RTC model directory")
    parser.add_argument("--device", default="GPU.0", help="OpenVINO device")
    parser.add_argument("--port", default="/dev/ttyACM0", help="Robot serial port")
    parser.add_argument("--calibration", required=True, help="Robot calibration file")
    parser.add_argument("--overhead-camera", required=True, help="Overhead camera device path (e.g. /dev/v4l/by-id/usb-...)")
    parser.add_argument("--arm-camera", required=True, help="Arm camera device path (e.g. /dev/v4l/by-id/usb-...)")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--duration-s", type=float, default=60.0)
    # RTC parameters
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--execution-horizon", type=int, default=10)
    parser.add_argument("--max-action-dim", type=int, default=32)
    parser.add_argument("--max-guidance-weight", type=float, default=10.0)
    parser.add_argument("--queue-threshold", type=int, default=30)
    args = parser.parse_args()

    # --- Latency tracker callback (lives on the model) ---
    latency_tracker = RTCLatencyTracker(window_size=100)

    # --- Load model WITHOUT postprocessors (RTCExecution owns denorm) ---
    model = InferenceModel.load(
        args.model,
        device=args.device,
        postprocessors=[],  # skip manifest denorm — execution handles it
        callbacks=[latency_tracker],
    )

    # --- Load denormalization postprocessors from manifest ---
    # RTCExecution applies these in its background thread to produce
    # the processed (denormalized) track for the robot.

    manifest = Manifest.load(f"{args.model}/manifest.json")
    rtc_postprocessors: list = [
        instantiate_component(resolve_artifact(spec, args.model))
        for spec in manifest.model.postprocessors
    ]

    # --- RTC queue (shared between execution and runtime) ---
    rtc_queue = RTCActionQueue()

    # --- RTC execution strategy ---
    execution = RTCExecution(
        chunk_size=args.chunk_size,
        execution_horizon=args.execution_horizon,
        fps=args.fps,
        max_action_dim=args.max_action_dim,
        max_guidance_weight=args.max_guidance_weight,
        queue_threshold=args.queue_threshold,
        latency_tracker=latency_tracker,
        postprocessors=rtc_postprocessors,
    )

    # --- Hardware ---
    robot = SO101(port=args.port, calibration=args.calibration, role="follower")
    cameras = {
        "overhead": SharedCamera(
            "uvc", device=args.overhead_camera,
            width=args.width, height=args.height, fps=int(args.fps),
        ),
        "arm": SharedCamera(
            "uvc", device=args.arm_camera,
            width=args.width, height=args.height, fps=int(args.fps),
        ),
    }

    # --- PolicyRuntime ---
    runtime = PolicyRuntime(
        robot=robot,
        model=model,
        execution=execution,
        action_queue=rtc_queue,
        cameras=cameras,
        fps=args.fps,
    )

    try:
        runtime.connect()
    except Exception as e:
        print(f"Failed to connect: {e}")
        from physicalai.capture import discover_all
        print("Available cameras:")
        for driver, devices in discover_all().items():
            for dev in devices:
                print(f"  Driver: {driver}, Device: {dev.device_id}, Info: {dev.name}")
        return

    for name, cam in cameras.items():
        print(f"Camera '{name}': {cam.actual_width}x{cam.actual_height} @ {cam.actual_fps}fps")

    print(
        f"Starting RTC runtime — chunk={args.chunk_size}, "
        f"horizon={args.execution_horizon}, fps={args.fps}"
    )
    try:
        stats = runtime.run(duration_s=args.duration_s)
        print(
            f"\nDone — {stats.steps} steps, {stats.inference_count} inferences, "
            f"{stats.total_holds} holds"
        )
        print(
            f"Latency — max={latency_tracker.max_latency_s:.3f}s, "
            f"p95={latency_tracker.percentile_s(95):.3f}s"
        )
    finally:
        runtime.disconnect()
        print("Disconnected")


if __name__ == "__main__":
    main()
