# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Two-stage inference runner.

Chains a second model onto the primary one: the primary adapter runs, and its
outputs are fed by name into a second artifact loaded on a backend of its own.

This exists for policies whose graph cannot be exported in one piece. A
vision-language-action model, for example, may have a backbone that no exporter
can capture and an action head that exports cleanly - and it is the head that is
evaluated repeatedly per action chunk, so running it on an accelerated backend is
where the time goes. Splitting the two lets each part run on the backend that can
actually take it, without pretending the whole policy is one graph.

The runner is deliberately generic: it knows nothing about which policy it is
serving. The contract is only that the primary adapter's output names match the
second model's input names.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger
from typing_extensions import override

from physicalai.inference.adapters import get_adapter
from physicalai.inference.runners.base import InferenceRunner

if TYPE_CHECKING:
    import numpy as np

    from physicalai.inference.adapters.base import RuntimeAdapter


class TwoStage(InferenceRunner):
    """Run the primary adapter, then feed its outputs to a second model.

    The second model is loaded lazily on the first call, so constructing the
    runner - which happens while the manifest is being read - never touches the
    filesystem or compiles a graph.

    Example:
        A manifest declaring a Torch backbone with an OpenVINO action head::

            "runner": {
                "type": "two_stage",
                "backend": "openvino",
                "artifact": "xr1_action_expert.xml"
            }

        ``artifact`` is resolved against the export directory by
        :func:`~physicalai.inference.component_factory.resolve_artifact`, the same
        way preprocessor artifacts are.
    """

    def __init__(
        self,
        backend: str,
        artifact: str,
        device: str = "auto",
        **adapter_kwargs: Any,  # noqa: ANN401 - forwarded to the second adapter's constructor
    ) -> None:
        """Configure the second stage.

        Args:
            backend: Backend name for the second model, e.g. ``"openvino"``.
            artifact: Path to the second model, absolute after the manifest has
                been resolved against the export directory.
            device: Device for the second model, or ``"auto"`` to use the
                backend's own default.
            **adapter_kwargs: Forwarded to the second adapter's constructor.
        """
        self.backend = backend
        self.artifact = artifact
        self.device = device
        self._adapter_kwargs = adapter_kwargs
        self._second: RuntimeAdapter | None = None

    def _second_adapter(self) -> RuntimeAdapter:
        """Return the second stage's adapter, loading it on first use.

        Returns:
            The loaded adapter.
        """
        if self._second is None:
            kwargs = dict(self._adapter_kwargs)
            if self.device != "auto":
                kwargs["device"] = self.device
            adapter = get_adapter(self.backend, **kwargs)
            logger.info("Loading second stage: {} from {}", self.backend, self.artifact)
            adapter.load(Path(self.artifact))
            self._second = adapter
        return self._second

    @override
    def run(
        self,
        adapter: RuntimeAdapter,
        inputs: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Run both stages and return the second stage's output.

        Args:
            adapter: The primary adapter, already loaded by ``InferenceModel``.
            inputs: Pre-processed model inputs.

        Returns:
            The second stage's output dict.

        Raises:
            KeyError: If the primary stage does not produce every input the
                second stage declares. Reported up front, because the alternative
                is a shape error from inside the second backend.
        """
        intermediate = dict(adapter.predict(inputs))
        second = self._second_adapter()

        expected = second.input_names
        if not expected:
            return dict(second.predict(intermediate))

        missing = [name for name in expected if name not in intermediate]
        if missing:
            msg = (
                f"The first stage did not produce {missing}, which the second stage needs. "
                f"It produced {sorted(intermediate)}."
            )
            raise KeyError(msg)

        return dict(second.predict({name: intermediate[name] for name in expected}))

    @override
    def reset(self) -> None:
        """Reset for a new episode.

        Both stages are stateless, so there is nothing to clear; the loaded
        second-stage adapter is kept, since reloading it would recompile the graph.
        """

    def __repr__(self) -> str:
        """Return string representation of the runner.

        Returns:
            A representation naming the second stage.
        """
        return f"{self.__class__.__name__}(backend={self.backend!r}, artifact={self.artifact!r})"
