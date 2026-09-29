# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""OpenVINO adapter for inference."""

from __future__ import annotations

import platform
from typing import TYPE_CHECKING, Any

from loguru import logger

from physicalai.inference.adapters.base import RuntimeAdapter
from physicalai.inference.adapters.registry import adapter_registry

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    import openvino

_ARM_MACHINES = frozenset({"arm64", "aarch64"})
_PRECISION_HINT = "INFERENCE_PRECISION_HINT"
_logged_arm_f32 = False


def _compile_config(device: str, config: dict[str, Any]) -> dict[str, Any]:
    """Return the compile config with platform-specific safe defaults applied.

    OpenVINO's f16 inference precision on CPU makes results depend on
    previous calls to the same compiled model (outputs drift from the second
    call on). f16 is the default on ARM CPUs, so default to f32 there. x86
    already defaults to f32 (bf16 with AMX). Only the exact ``"CPU"`` device
    is affected; an explicit ``INFERENCE_PRECISION_HINT`` from the caller
    always wins.

    TODO: remove once the OpenVINO f16 drift is fixed upstream.

    Args:
        device: OpenVINO device name.
        config: Caller-provided compile options.

    Returns:
        A new config dict; ``config`` is not modified.
    """
    global _logged_arm_f32  # noqa: PLW0603
    resolved = dict(config)
    if device.upper() == "CPU" and platform.machine().lower() in _ARM_MACHINES and _PRECISION_HINT not in resolved:
        resolved[_PRECISION_HINT] = "f32"
        if not _logged_arm_f32:
            logger.info("Using f32 inference precision on ARM CPU (set {} to override)", _PRECISION_HINT)
            _logged_arm_f32 = True
    return resolved


@adapter_registry.register("openvino", extensions=(".xml",))
class OpenVINOAdapter(RuntimeAdapter):
    """OpenVINO inference adapter.

    Provides inference through Intel OpenVINO Runtime, optimized
    for Intel hardware (CPU, GPU, NPU).

    Examples:
        >>> adapter = OpenVINOAdapter(device="CPU")
        >>> adapter.load(Path("model.xml"))
        >>> outputs = adapter.predict({"input": input_array})
    """

    def __init__(self, device: str = "CPU", **kwargs: Any) -> None:  # noqa: ANN401
        """Initialize OpenVINO adapter.

        Args:
            device: OpenVINO device ('CPU', 'GPU', 'NPU', 'AUTO')
            **kwargs: Additional OpenVINO compile options
        """
        super().__init__(device, **kwargs)
        self.compiled_model: openvino.CompiledModel | None = None
        self._input_names: list[str] = []
        self._output_names: list[str] = []

    def load(self, model_path: Path) -> None:
        """Load OpenVINO model from XML file.

        Args:
            model_path: Path to .xml model file (.bin file should be in same directory)

        Raises:
            ImportError: If OpenVINO is not installed
            FileNotFoundError: If model files don't exist
            RuntimeError: If model compilation fails (e.g. missing GPU drivers when ``device='GPU'``)
        """
        try:
            import openvino as ov  # noqa: PLC0415
        except ImportError as e:
            msg = "OpenVINO is not installed. Install with: uv pip install openvino"
            raise ImportError(msg) from e

        if not model_path.exists():
            msg = f"Model file not found: {model_path}"
            raise FileNotFoundError(msg)

        # Load and compile model
        core = ov.Core()
        model = core.read_model(model=str(model_path))
        try:
            self.compiled_model = core.compile_model(
                model=model,
                device_name=self.device,
                config=_compile_config(self.device, self.config),
            )
        except RuntimeError as e:
            err = str(e)
            is_gpu = "GPU" in self.device.upper()
            opencl_missing = "libOpenCL.so" in err
            no_gpu_devices = is_gpu and ("m_device_map.empty()" in err or "no supported devices found" in err)
            if is_gpu and (opencl_missing or no_gpu_devices):
                if opencl_missing:
                    cause = "the OpenCL loader (libOpenCL.so.1) is missing"
                else:
                    cause = "no Intel GPU devices were detected (compute runtime missing or device not accessible)"
                msg = (
                    f"OpenVINO GPU plugin failed to load: {e}\n"
                    f"Cause: {cause}.\n"
                    "On Debian/Ubuntu install the OpenCL loader and Intel compute runtime:\n"
                    "  sudo apt install ocl-icd-libopencl1 intel-opencl-icd\n"
                    "Ensure your user can access the GPU (add to the 'render' group, then re-login):\n"
                    "  sudo usermod -aG render $USER\n"
                    "Verify with: clinfo -l\n"
                    "Or set device='CPU' (or 'AUTO') to run inference on the CPU. "
                    "See https://github.com/openvinotoolkit/physicalai/tree/main/docs/getting-started/installation.md "
                    "for details."
                )
                raise RuntimeError(msg) from e
            raise

        # Cache input/output names
        # It's important to use the original model here, since compilation step adds extra auto-generated names
        self._input_names = [input_node.any_name for input_node in model.inputs]
        self._output_names = [output_node.any_name for output_node in self.compiled_model.outputs]

    def predict(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Run inference with OpenVINO.

        Args:
            inputs: Dictionary mapping input names to numpy arrays

        Returns:
            Dictionary mapping output names to numpy arrays

        Raises:
            RuntimeError: If model is not loaded
        """
        import numpy as np  # noqa: PLC0415

        if self.compiled_model is None:
            msg = "Model not loaded. Call load() first."
            raise RuntimeError(msg)

        # Run inference
        results = self.compiled_model(inputs)

        # Convert to dictionary with output names
        return {name: np.array(results[i]) for i, name in enumerate(self._output_names)}

    def default_device(self) -> str:  # noqa: PLR6301
        """Get default OpenVINO device.

        Returns:
            'CPU' (most compatible OpenVINO device)
        """
        return "CPU"

    @property
    def input_names(self) -> list[str]:
        """Get input tensor names.

        Returns:
            List of input names
        """
        return self._input_names

    @property
    def output_names(self) -> list[str]:
        """Get output tensor names.

        Returns:
            List of output names
        """
        return self._output_names
