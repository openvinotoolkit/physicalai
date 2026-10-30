# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Create two-camera ROS 2 recordings without ROS or robot hardware."""

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from rosbags.rosbag2 import Writer
from rosbags.rosbag2.writer import StoragePlugin
from rosbags.typesys import Stores, get_typestore


def demo_mapping() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "fps": 10,
        "task": "Move the synthetic arm",
        "robot_type": "so101",
        "max_age_ms": 100,
        "state": [
            {
                "name": "shoulder_pan.pos",
                "topic": "/state",
                "message_type": "sensor_msgs/msg/JointState",
                "field": "position",
                "joint": "shoulder_pan",
                "conversion": "rad_to_deg",
            },
            {
                "name": "gripper.pos",
                "topic": "/state",
                "message_type": "sensor_msgs/msg/JointState",
                "field": "position",
                "joint": "gripper",
            },
        ],
        "action": [
            {
                "name": "shoulder_pan.pos",
                "topic": "/command",
                "message_type": "std_msgs/msg/Float64",
                "field": "data",
                "conversion": "rad_to_deg",
            },
            {
                "name": "gripper.pos",
                "topic": "/gripper_command",
                "message_type": "std_msgs/msg/Float64",
                "field": "data",
            },
        ],
        "cameras": [
            {"name": name, "topic": f"/{name}", "message_type": "sensor_msgs/msg/Image"} for name in ("front", "wrist")
        ],
    }


def write_demo(
    path: Path,
    storage: str = "sqlite3",
    frames: int = 8,
    times: list[int] | None = None,
    duplicate: str | None = None,
    missing: str | None = None,
    compressed: bool = False,
) -> None:
    store = get_typestore(Stores.ROS2_HUMBLE)
    types = store.types
    time_type = types["builtin_interfaces/msg/Time"]
    header_type = types["std_msgs/msg/Header"]
    state_type = types["sensor_msgs/msg/JointState"]
    command_type = types["std_msgs/msg/Float64"]
    image_type = types["sensor_msgs/msg/Image"]
    image_name = "sensor_msgs/msg/CompressedImage" if compressed else "sensor_msgs/msg/Image"
    with Writer(path, version=9, storage_plugin=StoragePlugin[storage.upper()]) as writer:
        connections = {
            topic: writer.add_connection(topic, typename, typestore=store)
            for topic, typename in (
                ("/state", "sensor_msgs/msg/JointState"),
                ("/command", "std_msgs/msg/Float64"),
                ("/gripper_command", "std_msgs/msg/Float64"),
                ("/front", image_name),
                ("/wrist", image_name),
            )
            if topic != missing
        }
        for i, timestamp in enumerate(
            times if times is not None else [1_700_000_000_000_000_000 + j * 100_000_000 for j in range(frames)]
        ):
            header = header_type(time_type(timestamp // 1_000_000_000, timestamp % 1_000_000_000), "camera")
            pixels = np.zeros((64, 64, 3), dtype=np.uint8)
            pixels[:, :, 0] = 30 + i * 10
            pixels[:, :, 1] = np.arange(64, dtype=np.uint8)
            # Reordered joints exercise name-based selection instead of positional assumptions.
            names = ["gripper", "shoulder_pan"] if i % 2 else ["shoulder_pan", "gripper"]
            positions = np.array([0.25, i * 0.01] if i % 2 else [i * 0.01, 0.25], dtype=np.float64)
            messages = {
                "/state": state_type(
                    header, names, positions, np.array([], dtype=np.float64), np.array([], dtype=np.float64)
                ),
                "/command": command_type(i * 0.02),
                "/gripper_command": command_type(0.25 + i * 0.005),
                "/front": image_type(header, 64, 64, "rgb8", 0, 64 * 3, pixels.flatten()),
                "/wrist": image_type(header, 64, 64, "rgb8", 0, 64 * 3, pixels[:, ::-1].flatten()),
            }
            if compressed:
                import cv2

                for camera, extension in (("front", ".jpg"), ("wrist", ".png")):
                    ok, encoded = cv2.imencode(extension, pixels[:, :, ::-1])
                    assert ok
                    messages[f"/{camera}"] = types[image_name](
                        header, "jpeg" if extension == ".jpg" else "png", encoded
                    )
            for topic, connection in connections.items():
                raw = store.serialize_cdr(messages[topic], connection.msgtype)
                writer.write(connection, timestamp, raw)
                if i == 0 and topic == "/command" and duplicate:
                    if duplicate == "conflict":
                        raw = store.serialize_cdr(command_type(999.0), connection.msgtype)
                    writer.write(connection, timestamp, raw)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    bags = args.output / "bags"
    bags.mkdir()
    write_demo(bags / "episode-01", "sqlite3")
    write_demo(bags / "episode-02", "mcap")
    (args.output / "mapping.yaml").write_text(yaml.safe_dump(demo_mapping(), sort_keys=False))
