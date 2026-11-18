# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest

from physicalai.inference.component_factory import component_registry
from physicalai.inference.constants import TASK, TASK_INDEX
from physicalai.inference.preprocessors import TaskIndexPreprocessor

TASKS = ["pick", "place"]


def test_registered_type() -> None:
    assert component_registry.resolve("task_index") == "physicalai.inference.preprocessors.TaskIndexPreprocessor"


@pytest.mark.parametrize("task", ["place", ["place"], np.array(["place"])])
def test_maps_a_task_to_its_index(task: object) -> None:
    result = TaskIndexPreprocessor(TASKS)({TASK: task})
    np.testing.assert_array_equal(result[TASK_INDEX], [[1]])
    assert result[TASK_INDEX].dtype == np.int64


def test_batch() -> None:
    result = TaskIndexPreprocessor(TASKS)({TASK: ["place", "pick", "place"]})
    np.testing.assert_array_equal(result[TASK_INDEX], [[1], [0], [1]])


def test_reused_inputs_are_not_changed() -> None:
    """Control loops reuse one observation dict; a later task must not get this index."""
    inputs = {TASK: ["place"]}
    TaskIndexPreprocessor(TASKS)(inputs)
    assert TASK_INDEX not in inputs


def test_caller_index_is_kept() -> None:
    given = np.array([[0]], dtype=np.int64)
    assert TaskIndexPreprocessor(TASKS)({TASK_INDEX: given})[TASK_INDEX] is given


def test_unknown_task_raises() -> None:
    with pytest.raises(ValueError, match=r"Unknown task\(s\) \['dance'\].*'pick', 'place'"):
        TaskIndexPreprocessor(TASKS)({TASK: ["dance"]})


def test_missing_task_raises() -> None:
    with pytest.raises(ValueError, match="takes one of the tasks"):
        TaskIndexPreprocessor(TASKS)({})


def test_instantiates_from_a_manifest_spec() -> None:
    from physicalai.inference.component_factory import instantiate_component
    from physicalai.inference.manifest import ComponentSpec
    from physicalai.inference.preprocessors import Preprocessor

    prep = instantiate_component(Preprocessor, ComponentSpec(type="task_index", tasks=TASKS))
    assert isinstance(prep, TaskIndexPreprocessor)
    np.testing.assert_array_equal(prep({TASK: ["pick"]})[TASK_INDEX], [[0]])
