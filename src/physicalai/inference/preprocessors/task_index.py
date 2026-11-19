# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Preprocessor that maps task prompts to the integer index a multi-task export takes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from physicalai.inference.constants import TASK, TASK_INDEX

from .base import Preprocessor

if TYPE_CHECKING:
    from collections.abc import Sequence


class TaskIndexPreprocessor(Preprocessor):
    """Add each ``task`` prompt's position in ``tasks`` as a ``(batch, 1)`` int64 ``task_index``.

    For policies trained on a fixed set of tasks without a language model, such as
    a multi-task ACT: the export takes the task as an index, not as text, so only
    the prompts it was trained on are valid. Prompts take precedence over dataset
    indices; a supplied ``task_index`` is kept only when no prompt is provided.

    Args:
        tasks: The prompts the policy was trained on; a prompt's position is its index.
    """

    def __init__(self, tasks: Sequence[str]) -> None:
        """Initialize the task index preprocessor."""
        super().__init__()
        self._index = {task: index for index, task in enumerate(tasks)}

    def __call__(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Add ``task_index`` for the inputs' ``task`` prompts.

        Args:
            inputs: Observation dict with ``task`` as a string or a list of strings.

        Returns:
            The inputs, or a shallow copy of them with a ``(batch, 1)`` int64 ``task_index`` array.

        Raises:
            ValueError: If ``task`` is missing or is not one of the known tasks.
        """
        task = inputs.get(TASK)
        if task is None:
            if inputs.get(TASK_INDEX) is not None:
                return inputs
            msg = f"This policy takes one of the tasks {list(self._index)} as `task`."
            raise ValueError(msg)
        prompts = [task] if isinstance(task, str) else [str(prompt) for prompt in np.atleast_1d(task)]
        unknown = [prompt for prompt in prompts if prompt not in self._index]
        if unknown:
            msg = f"Unknown task(s) {unknown}; this policy only knows {list(self._index)}."
            raise ValueError(msg)
        # Copy so a reused observation dict doesn't keep this index for the next task.
        return {**inputs, TASK_INDEX: np.array([[self._index[prompt]] for prompt in prompts], dtype=np.int64)}

    def __repr__(self) -> str:
        """Return string representation of the preprocessor."""
        return f"{self.__class__.__name__}(tasks={list(self._index)!r})"
