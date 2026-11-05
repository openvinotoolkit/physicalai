# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import numpy as np
import pytest

from physicalai_rosbag_plugin.config import MappingConfig, ResourceLimits
from physicalai_rosbag_plugin.images import decode_image
from physicalai_rosbag_plugin.reader import frame_times


def mapping_data():
    return {
        "schema_version": 1,
        "fps": 30,
        "task": "Move arm",
        "robot_type": "so101",
        "max_age_ms": 100,
        "state": [
            {
                "name": "shoulder_pan.pos",
                "topic": "/state",
                "message_type": "sensor_msgs/msg/JointState",
                "field": "position",
                "joint": "shoulder_pan",
            }
        ],
        "action": [
            {"name": "shoulder_pan.pos", "topic": "/command", "message_type": "std_msgs/msg/Float64", "field": "data"}
        ],
        "cameras": [{"name": "front", "topic": "/image", "message_type": "sensor_msgs/msg/Image"}],
    }


def test_mapping_preserves_names():
    config = MappingConfig.model_validate(mapping_data())
    assert config.state[0].name == "shoulder_pan.pos"


@pytest.mark.parametrize(
    "key,value", [("fps", 0), ("fps", 1001), ("fps", True), ("schema_version", True), ("max_age_ms", -1)]
)
def test_invalid_config(key, value):
    data = mapping_data()
    data[key] = value
    with pytest.raises(ValueError):
        MappingConfig.model_validate(data)


def test_duplicate_components_and_ambiguous_conversion():
    data = mapping_data()
    data["state"] *= 2
    with pytest.raises(ValueError, match="Duplicate"):
        MappingConfig.model_validate(data)
    data = mapping_data()
    data["state"][0].update(conversion="rad_to_deg", scale=2)
    with pytest.raises(ValueError, match="conversion"):
        MappingConfig.model_validate(data)


def test_integer_frame_grid():
    start = 1_700_000_000_000_000_001
    assert list(frame_times(start, start + 100_000_000, 30)) == [
        start,
        start + 33_333_333,
        start + 66_666_666,
        start + 100_000_000,
    ]
    assert list(frame_times(start, start + 33_333_333, 30)) == [start, start + 33_333_333]


def test_padded_raw_image():
    msg = SimpleNamespace(
        height=2,
        width=2,
        step=8,
        encoding="bgr8",
        is_bigendian=0,
        data=np.array([0, 0, 255, 0, 255, 0, 99, 99, 255, 0, 0, 255, 255, 255, 88, 88], dtype=np.uint8),
    )
    image = decode_image(msg, "sensor_msgs/msg/Image", ResourceLimits())
    np.testing.assert_array_equal(image, [[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 255]]])


def test_image_limits_before_decode():
    msg = SimpleNamespace(height=10000, width=10000, step=30000, encoding="rgb8", data=b"")
    with pytest.raises(ValueError, match="pixels"):
        decode_image(msg, "sensor_msgs/msg/Image", ResourceLimits())


@pytest.mark.parametrize("encoding,channels", [("rgb8", 3), ("bgr8", 3), ("rgba8", 4), ("bgra8", 4), ("mono8", 1)])
def test_raw_encodings(encoding, channels):
    msg = SimpleNamespace(
        height=2, width=2, step=2 * channels, encoding=encoding, data=np.full(4 * channels, 17, dtype=np.uint8)
    )
    image = decode_image(msg, "sensor_msgs/msg/Image", ResourceLimits())
    assert image.shape == (2, 2, 3) and image.dtype == np.uint8
    assert np.all(image == 17)


@pytest.mark.parametrize("encoding", ["16UC1", "32FC1", "yuv422", "bayer_rggb8"])
def test_unsupported_raw_images(encoding):
    msg = SimpleNamespace(height=2, width=2, step=4, encoding=encoding, data=b"\x00" * 8)
    with pytest.raises(ValueError, match="encoding"):
        decode_image(msg, "sensor_msgs/msg/Image", ResourceLimits())


@pytest.mark.parametrize("extension", [".png", ".jpg"])
@pytest.mark.parametrize("channels", [1, 3])
def test_compressed_images(extension, channels):
    import cv2

    pixels = np.full((8, 8, channels), 80, dtype=np.uint8)
    ok, encoded = cv2.imencode(extension, pixels)
    assert ok
    msg = SimpleNamespace(format="png" if extension == ".png" else "jpeg", data=encoded)
    image = decode_image(msg, "sensor_msgs/msg/CompressedImage", ResourceLimits())
    assert image.shape == (8, 8, 3)
    np.testing.assert_allclose(image, 80, atol=1)
    with pytest.raises(ValueError, match="pixels"):
        decode_image(msg, "sensor_msgs/msg/CompressedImage", ResourceLimits(image_pixels=4))


def test_invalid_payload_and_stride():
    from PIL import UnidentifiedImageError

    with pytest.raises(UnidentifiedImageError):
        decode_image(
            SimpleNamespace(format="jpeg", data=b"broken"), "sensor_msgs/msg/CompressedImage", ResourceLimits()
        )
    with pytest.raises(ValueError, match="step"):
        decode_image(
            SimpleNamespace(height=2, width=2, step=4, encoding="rgb8", data=b"\0" * 8),
            "sensor_msgs/msg/Image",
            ResourceLimits(),
        )


def test_numeric_selection_and_units():
    from physicalai_rosbag_plugin.config import Component
    from physicalai_rosbag_plugin.reader import _component

    spec = Component(
        name="gripper.pos",
        topic="/cmd",
        message_type="std_msgs/msg/Float64MultiArray",
        field="data",
        index=1,
        scale=2,
        offset=1,
    )
    assert _component(SimpleNamespace(data=np.array([9.0, 3.0])), spec) == 7
    with pytest.raises(ValueError, match="outside"):
        _component(SimpleNamespace(data=np.array([9.0])), spec)
    with pytest.raises(ValueError, match="nonfinite"):
        _component(SimpleNamespace(data=np.array([9.0, float("nan")])), spec)
    joint = Component(
        name="elbow.pos",
        topic="/state",
        message_type="sensor_msgs/msg/JointState",
        field="position",
        joint="elbow",
        conversion="deg_to_rad",
    )
    assert _component(SimpleNamespace(name=["elbow"], position=[180.0]), joint) == pytest.approx(np.pi)
    with pytest.raises(ValueError, match="missing joint"):
        _component(SimpleNamespace(name=["wrong"], position=[1.0]), joint)
