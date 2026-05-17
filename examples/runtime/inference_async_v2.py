# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env python3
"""Async inference with PolicyRuntime.

Replaces the original 764-line inference_async.py. The runtime handles
async scheduling, action smoothing, and queue management internally.
"""

import argparse

from physicalai.capture.transport import SharedCamera
from physicalai.inference import InferenceModel
from physicalai.robot.so101 import SO101Follower
from physicalai.runtime import (
    ActionQueue,
    AsyncExecution,
    LerpSmoother,
    PolicyRuntime,
)


def main():
    parser = argparse.ArgumentParser(description="Run policy with PolicyRuntime")
    parser.add_argument("--model", required=True, help="Exported model directory")
    parser.add_argument("--port", default="/dev/ttyACM0", help="Robot serial port")
    parser.add_argument("--duration-s", type=float, default=60.0)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()

    model = InferenceModel(args.model)
    robot = SO101Follower(port=args.port)
    cameras = {"overhead": SharedCamera("overhead"), "arm": SharedCamera("arm")}

    runtime = PolicyRuntime(
        robot=robot,
        model=model,
        execution=AsyncExecution(threshold=0.5, fps=int(args.fps)),
        action_queue=ActionQueue(smoother=LerpSmoother(duration_frames=5)),
        cameras=cameras,
        fps=args.fps,
    )

    robot.connect()
    for cam in cameras.values():
        cam.connect()
    try:
        stats = runtime.run(duration_s=args.duration_s)
        print(f"Done: {stats}")
    finally:
        for cam in cameras.values():
            cam.disconnect()
        robot.disconnect()


if __name__ == "__main__":
    main()
