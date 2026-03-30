# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Discover a UVC camera and print a few live frame summaries."""

# ruff: noqa: D103, INP001

import sys

from physicalai.capture.cameras.uvc import UVCCamera, discover_uvc


def main() -> None:
    devices = discover_uvc()
    for i, device in enumerate(devices):
        sys.stdout.write(f"[{i}] {device.name} ({device.device_id})\n")
    if not devices:
        sys.stdout.write("No cameras found.\n")
        return

    device_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    with UVCCamera(device=device_index) as cam:
        for _ in range(10):
            frame = cam.read_latest()
            sys.stdout.write(
                f"shape={frame.data.shape} timestamp={frame.timestamp} sequence={frame.sequence}\n",
            )


if __name__ == "__main__":
    main()
