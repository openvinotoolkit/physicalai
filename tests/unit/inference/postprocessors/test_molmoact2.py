# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest

from physicalai.inference.component_factory import instantiate_component
from physicalai.inference.constants import ACTION
from physicalai.inference.manifest import ComponentSpec
from physicalai.inference.postprocessors import MolmoAct2Postprocessor


class TestMolmoAct2Postprocessor:
    def test_clamps_masked_denormalizes_and_transforms(self) -> None:
        processor = MolmoAct2Postprocessor(
            action_stats={
                "q01": [0.0, 0.0, 0.0],
                "q99": [2.0, 2.0, 2.0],
                "mask": [True, False, True],
            },
            adapt_to_so101=True,
            joint_signs=[1.0, -1.0],
            joint_offsets=[0.0, 2.0],
        )

        result = processor({"actions": np.array([[[2.0, 0.5, -2.0]]], dtype=np.float32)})

        np.testing.assert_allclose(result[ACTION], [[[2.0, 1.5, 0.0]]])
        assert "actions" not in result

    def test_identity_without_stats(self) -> None:
        processor = MolmoAct2Postprocessor()
        result = processor({ACTION: np.array([[-0.5, 0.5]], dtype=np.float32)})
        np.testing.assert_array_equal(result[ACTION], [[-0.5, 0.5]])

    def test_uses_fixed_so101_transform_by_default(self) -> None:
        checkpoint_values = [1.0, 88.0, 93.0]
        processor = MolmoAct2Postprocessor(
            action_stats={"q01": checkpoint_values, "q99": checkpoint_values},
            adapt_to_so101=True,
        )

        result = processor({ACTION: np.zeros((1, 3), dtype=np.float32)})

        np.testing.assert_array_equal(result[ACTION], [[1.0, 2.0, 3.0]])

    def test_missing_action_raises(self) -> None:
        with pytest.raises(ValueError, match="action tensor"):
            MolmoAct2Postprocessor()({"other": np.zeros(1)})

    def test_registry_alias_instantiates(self) -> None:
        processor = instantiate_component(ComponentSpec(type="molmoact2_postprocess"))
        assert isinstance(processor, MolmoAct2Postprocessor)
