#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Async inference with PolicyRuntime.

python examples/runtime/async_inference.py \
  --model ./exports/pi05_cans_openvino \
  --device GPU.0 \
  --port /dev/ttyACM0 \
  --calibration /home/max/.cache/physicalai/robots/a8d8d997-a59e-4423-9006-5d991d223887/calibrations/0b2f185a-8ab2-4956-91c2-3a2ac2dbd8c1.json \
  --overhead-camera /dev/v4l/by-id/usb-UGREEN_Camera_2K_UGREEN_Camera_2K_SN0001-video-index0 \
  --arm-camera 353322271391 \
  --front-camera /dev/v4l/by-id/usb-Innomaker_Innomaker-U20CAM-1080p-S1_SN0001-video-index0 \
  --width 640 \
  --height 480 \
  --fps 30 \
  --duration-s 60
"""

from __future__ import annotations

import argparse

import openvino as ov
import numpy as np

from physicalai.capture import discover_all
from physicalai.capture.transport import SharedCamera
from physicalai.inference import InferenceModel
from physicalai.robot import SO101
from physicalai.runtime import (
    AsyncExecution,
    ChunkedActionQueue,
    LerpSmoother,
    PolicyRuntime,
)


def main():
    parser = argparse.ArgumentParser(description="Run policy with PolicyRuntime")
    parser.add_argument("--model", required=True, help="Exported model directory")
    parser.add_argument("--device", default="GPU.0", help="OpenVINO device")
    parser.add_argument("--port", default="/dev/ttyACM0", help="Robot serial port")
    parser.add_argument("--calibration", required=True, help="Robot calibration file")
    parser.add_argument("--overhead-camera", required=True, help="Overhead camera device path")
    parser.add_argument("--arm-camera", required=True, help="Arm camera serial number")
    parser.add_argument("--front-camera", required=True, help="Front camera device path")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--duration-s", type=float, default=60.0)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()

    import openvino_tokenizers  # noqa: F401 — registers OV tokenizer ops

    print(f"Available devices:")
    core = ov.Core()
    devices = core.available_devices
    for dev in devices:
        print(f"  {dev}: {core.get_property(dev, 'FULL_DEVICE_NAME')}")
    print(f"Selected device: {args.device}")

    model = InferenceModel.load(args.model, device=args.device)
    robot = SO101(port=args.port, calibration=args.calibration, role="follower")
    cameras = {
        "overhead": SharedCamera("uvc", device=args.overhead_camera, width=args.width, height=args.height, fps=int(args.fps)),
        "front": SharedCamera("uvc", device=args.front_camera, width=args.width, height=args.height, fps=int(args.fps)),
        "arm": SharedCamera("realsense", serial_number=args.arm_camera, width=args.width, height=args.height, fps=int(args.fps)),
    }

    runtime = PolicyRuntime(
        robot=robot,
        model=model,
        execution=AsyncExecution(threshold=0.3, fps=int(args.fps)),
        action_queue=ChunkedActionQueue(smoother=LerpSmoother(duration_frames=5)),
        cameras=cameras,
        fps=args.fps,
    )

    try:
        runtime.connect()
    except Exception as e:
        print(f"Failed to connect: {e}")
        print("Available cameras:")
        for driver, devices in discover_all().items():
            for dev in devices:
                print(f"  Driver: {driver}, Device: {dev.device_id}, Info: {dev.name}")
        return

    for name, cam in cameras.items():
        print(f"Camera '{name}' connected: {cam.actual_width}x{cam.actual_height} @ {cam.actual_fps}fps")

    print("Starting policy runtime...")
    try:
        stats = runtime.run(duration_s=args.duration_s)
        print(f"\nDone — {stats.steps} steps, {stats.inference_count} inferences, {stats.total_holds} holds")
    finally:
        runtime.disconnect()
        print("Disconnected")


if __name__ == "__main__":
    main()
