# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest

from physicalai.inference.component_factory import instantiate_component
from physicalai.inference.constants import ACTION
from physicalai.inference.manifest import ComponentSpec
from physicalai.inference.postprocessors import JointFramePostprocessor, MolmoAct2Postprocessor


class TestMolmoAct2Postprocessor:
    def test_clamps_and_masked_denormalizes_before_joint_transform(self) -> None:
        processor = MolmoAct2Postprocessor(
            action_key="actions",
            action_stats={
                "q01": [0.0, 0.0, 0.0],
                "q99": [2.0, 2.0, 2.0],
                "mask": [True, False, True],
            },
        )
        joint_transform = JointFramePostprocessor(
            feature=ACTION,
            signs=[1.0, -1.0],
            offsets=[0.0, 2.0],
        )

        result = joint_transform(processor({"actions": np.array([[[2.0, 0.5, -2.0]]], dtype=np.float32)}))

        np.testing.assert_allclose(result[ACTION], [[[2.0, 1.5, 0.0]]])
        assert "actions" not in result

    def test_identity_without_stats(self) -> None:
        processor = MolmoAct2Postprocessor(action_key=ACTION)
        result = processor({ACTION: np.array([[-0.5, 0.5]], dtype=np.float32)})
        np.testing.assert_array_equal(result[ACTION], [[-0.5, 0.5]])

    def test_missing_action_raises(self) -> None:
        with pytest.raises(ValueError, match="expected action key 'model_actions'"):
            MolmoAct2Postprocessor(action_key="model_actions")({"other": np.zeros(1)})

    def test_registry_alias_uses_manifest_action_key(self) -> None:
        processor = instantiate_component(
            ComponentSpec(type="molmoact2_postprocess", action_key="model_actions"),
        )

        assert isinstance(processor, MolmoAct2Postprocessor)
        result = processor(
            {
                "model_actions": np.array([[-0.5, 0.5]], dtype=np.float32),
                "other": np.ones(1),
            },
        )
        np.testing.assert_array_equal(result[ACTION], [[-0.5, 0.5]])
        np.testing.assert_array_equal(result["other"], np.ones(1))
        assert "model_actions" not in result
