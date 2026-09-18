# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest

from physicalai.inference.component_factory import instantiate_component
from physicalai.inference.manifest import ComponentSpec
from physicalai.inference.preprocessors import Preprocessor


@pytest.mark.parametrize(
    ("short_name", "class_name"),
    [
        ("resize", "ResizePreprocessor"),
        ("smolvla_resize", "ResizeSmolVLA"),
    ],
)
@pytest.mark.parametrize("config_style", ["type", "class_path"])
@pytest.mark.parametrize("layout", ["BCHW", "BHWC"])
def test_resize_layout_from_config(
    short_name: str,
    class_name: str,
    config_style: str,
    layout: str,
) -> None:
    bchw = np.arange(27, dtype=np.float32).reshape(1, 3, 3, 3) / 26
    img = bchw if layout == "BCHW" else bchw.transpose(0, 2, 3, 1)

    args = {
        "image_resolution": [3, 3],
        "image_layout": layout,
    }
    if config_style == "type":
        config = {"type": short_name, **args}
    else:
        config = {
            "class_path": f"physicalai.inference.preprocessors.{class_name}",
            "init_args": args,
        }

    spec = ComponentSpec.model_validate(config)
    prep = instantiate_component(Preprocessor, spec)
    assert isinstance(prep, Preprocessor)
    result = prep({"images": img})

    expected = bchw
    if short_name == "smolvla_resize":
        expected = (bchw * 2 - 1)[None]
        np.testing.assert_array_equal(result["image_masks"], [[True]])

    np.testing.assert_array_equal(result["images"], expected)


@pytest.mark.parametrize(
    ("short_name", "class_name"),
    [
        ("resize", "ResizePreprocessor"),
        ("smolvla_resize", "ResizeSmolVLA"),
    ],
)
@pytest.mark.parametrize("config_style", ["type", "class_path"])
@pytest.mark.parametrize(
    ("layout_args", "error_match"),
    [
        pytest.param({}, "image_layout", id="missing"),
        pytest.param({"image_layout": "HWC"}, "ImageLayout", id="unbatched"),
        pytest.param({"image_layout": "AUTO"}, "ImageLayout", id="automatic"),
        pytest.param({"image_layout": ""}, "ImageLayout", id="empty"),
    ],
)
def test_resize_config_rejects_missing_or_invalid_layout(
    short_name: str,
    class_name: str,
    config_style: str,
    layout_args: dict[str, str],
    error_match: str,
) -> None:
    args = {"image_resolution": [3, 3], **layout_args}
    if config_style == "type":
        config = {"type": short_name, **args}
    else:
        config = {
            "class_path": f"physicalai.inference.preprocessors.{class_name}",
            "init_args": args,
        }

    spec = ComponentSpec.model_validate(config)
    # The component factory wraps parsing and constructor ValueErrors as TypeError.
    with pytest.raises(TypeError, match=error_match):
        instantiate_component(Preprocessor, spec)
