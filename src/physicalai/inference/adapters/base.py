# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Base adapter interface for inference backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np


class RuntimeAdapter(ABC):
    """Abstract base class for backend-specific inference adapters.

    Adapters provide a unified interface for running inference across
    different backends (OpenVINO, ONNX, TensorRT, etc.).

    Each adapter is responsible for:
    - Loading model files from disk
    - Running inference with backend-specific APIs
    - Converting inputs/outputs to/from numpy arrays
    """

    def __init__(self, device: str = "cpu", **kwargs: Any) -> None:  # noqa: ANN401
        """Initialize the adapter.

        Args:
            device: Device to run inference on ('cpu', 'cuda', 'gpu', etc.)
            **kwargs: Backend-specific configuration options
        """
        self.device = device
        self.config = kwargs
        self.model = None

    @abstractmethod
    def load(self, model_path: Path) -> None:
        """Load model from disk.

        Args:
            model_path: Path to the model file
        """

    @abstractmethod
    def predict(self, inputs: dict[str, Any]) -> dict[str, np.ndarray]:
        """Run inference on the model.

        Args:
            inputs: Dictionary mapping input names to actual inputs

        Returns:
            Dictionary mapping output names to numpy arrays
        """

    @property
    @abstractmethod
    def input_names(self) -> list[str]:
        """Get model input names.

        Returns:
            List of input tensor names
        """

    @property
    @abstractmethod
    def output_names(self) -> list[str]:
        """Get model output names.

        Returns:
            List of output tensor names
        """

    @property
    def input_dtypes(self) -> dict[str, np.dtype]:
        """Get the element type each named model input declares, where the backend reports one.

        :class:`~physicalai.inference.model.InferenceModel` casts each input to
        it before calling :meth:`predict`. The default declares none.

        Returns:
            Mapping from input name to NumPy dtype.
        """
        return {}

    @property
    def input_shapes(self) -> dict[str, tuple[int | None, ...]]:
        """Get the shape each named model input declares, where the backend reports one.

        ``None`` marks a dimension the graph leaves open. The default declares none.

        Returns:
            Mapping from input name to shape.
        """
        return {}

    def close(self) -> None:  # noqa: B027  # Optional hook with a no-op default.
        """Release resources such as worker processes or device memory.

        :class:`~physicalai.inference.model.InferenceModel` calls this once when
        it is closed. The default holds no resources.
        """

    def default_device(self) -> str:  # noqa: PLR6301
        """Get default device for this runtime.

        Returns:
            Default device name ('cpu', 'GPU', etc.)
        """
        return "cpu"

    def __repr__(self) -> str:
        """Return string representation of the adapter."""
        return f"{self.__class__.__name__}(device={self.device})"
