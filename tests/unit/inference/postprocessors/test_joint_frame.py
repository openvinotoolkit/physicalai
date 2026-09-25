# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import re

import numpy as np
import pytest

from physicalai.inference.component_factory import instantiate_component
from physicalai.inference.manifest import ComponentSpec
from physicalai.inference.postprocessors import JointFramePostprocessor
from physicalai.inference.postprocessors.joint_frame import JointFrameTransform


def test_joint_transform_round_trip_uses_supplied_frame() -> None:
    transform = JointFrameTransform(signs=(1.0, -1.0), offsets=(10.0, 20.0))
    robot_values = np.array([[2.0, 3.0, 4.0]], dtype=np.float32)

    checkpoint_values = transform.forward(robot_values)

    np.testing.assert_array_equal(checkpoint_values, [[12.0, 17.0, 4.0]])
    np.testing.assert_array_equal(transform.inverse(checkpoint_values), robot_values)


def test_joint_transform_rejects_invalid_frame() -> None:
    with pytest.raises(ValueError, match=re.escape("signs (1), offsets (2) and scales (1) must match")):
        JointFrameTransform(signs=(1.0,), offsets=(0.0, 1.0))
    with pytest.raises(ValueError, match="either -1 or 1"):
        JointFrameTransform(signs=(2.0,), offsets=(0.0,))


def test_joint_transform_round_trip_applies_scales() -> None:
    transform = JointFrameTransform(signs=(1.0, -1.0), offsets=(10.0, 20.0), scales=(2.0, 0.5))
    robot_values = np.array([[2.0, 4.0, 5.0]], dtype=np.float32)

    checkpoint_values = transform.forward(robot_values)

    np.testing.assert_allclose(checkpoint_values, [[14.0, 18.0, 5.0]])
    np.testing.assert_allclose(transform.inverse(checkpoint_values), robot_values)


def test_joint_transform_rejects_invalid_scales() -> None:
    with pytest.raises(ValueError, match=re.escape("signs (2), offsets (2) and scales (1) must match")):
        JointFrameTransform(signs=(1.0, 1.0), offsets=(0.0, 0.0), scales=(1.0,))


@pytest.mark.parametrize("scale", [0.0, -1.0, float("nan"), float("inf")])
def test_joint_transform_rejects_non_finite_or_non_positive_scales(scale: float) -> None:
    with pytest.raises(ValueError, match="scales must be finite and positive"):
        JointFrameTransform(signs=(1.0,), offsets=(0.0,), scales=(scale,))


def test_postprocessor_transforms_configured_feature() -> None:
    processor = instantiate_component(
        ComponentSpec(
            type="joint_frame_postprocess",
            feature="action",
            signs=[1.0, -1.0],
            offsets=[10.0, 20.0],
        )
    )
    outputs = {
        "action": np.array([[12.0, 17.0, 4.0]], dtype=np.float32),
        "other": np.array([5.0]),
    }

    assert isinstance(processor, JointFramePostprocessor)
    result = processor(outputs)
    np.testing.assert_array_equal(result["action"], [[2.0, 3.0, 4.0]])
    np.testing.assert_array_equal(result["other"], outputs["other"])
    np.testing.assert_array_equal(outputs["action"], [[12.0, 17.0, 4.0]])


def test_postprocessor_applies_configured_scales() -> None:
    processor = instantiate_component(
        ComponentSpec(
            type="joint_frame_postprocess",
            feature="action",
            signs=[1.0, -1.0],
            offsets=[10.0, 20.0],
            scales=[2.0, 0.5],
        )
    )

    assert isinstance(processor, JointFramePostprocessor)
    result = processor({"action": np.array([[14.0, 18.0, 5.0]], dtype=np.float32)})

    np.testing.assert_allclose(result["action"], [[2.0, 4.0, 5.0]])


def test_postprocessor_rejects_missing_feature() -> None:
    processor = JointFramePostprocessor(feature="action", signs=[1.0], offsets=[0.0])

    with pytest.raises(ValueError, match="expected feature 'action'"):
        processor({})


@pytest.mark.parametrize("scale", [0.0, -1.0, float("nan"), float("inf")])
def test_postprocessor_rejects_non_finite_or_non_positive_scales(scale: float) -> None:
    with pytest.raises(ValueError, match="scales must be finite and positive"):
        JointFramePostprocessor(feature="action", signs=[1.0], offsets=[0.0], scales=[scale])
