# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Production-ready inference model with unified API."""

from __future__ import annotations

import re
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

import numpy as np

from physicalai.inference.adapters import adapter_registry, get_adapter
from physicalai.inference.component_factory import instantiate_component, resolve_artifact
from physicalai.inference.constants import ACTION
from physicalai.inference.data.features import InferenceFeature
from physicalai.inference.manifest import ComponentSpec, Manifest
from physicalai.inference.runners import get_runner

if TYPE_CHECKING:
    from physicalai.inference.adapters.base import RuntimeAdapter
    from physicalai.inference.callbacks.base import Callback
    from physicalai.inference.postprocessors.base import Postprocessor
    from physicalai.inference.preprocessors.base import Preprocessor
    from physicalai.inference.runners.base import InferenceRunner


# Policy names from the manifest are used to construct filesystem paths.
# Restrict to safe characters to prevent "../" traversal attacks.
_SAFE_POLICY_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.\-]*$", re.ASCII)


def _is_safe_policy_name(name: str) -> bool:
    """Return True if *name* matches ``[a-zA-Z0-9][a-zA-Z0-9-_.]*`` (ASCII only)."""
    return _SAFE_POLICY_NAME_RE.fullmatch(name) is not None


class InferenceModel:
    """Unified inference interface for exported policies.

    Automatically detects backend and provides consistent API across
    all export formats (OpenVINO, ONNX, Torch Export IR).

    The interface matches PyTorch policy API:
    - ``select_action(obs)`` — Get action from observation
    - ``reset()`` — Reset policy state for new episode
    - ``__call__(inputs)`` — Primary inference API (delegates to runner)

    Examples:
        >>> # Auto-detect everything
        >>> policy = InferenceModel.load("./exports/act_policy")
        >>> policy.reset()
        >>> action = policy.select_action(obs)
        >>> action = policy.predict_action_chunk(obs)

        >>> # Explicit backend and device
        >>> policy = InferenceModel(
        ...     export_dir="./exports",
        ...     policy_name="act",
        ...     backend="openvino",
        ...     device="CPU"
        ... )
    """

    def __init__(
        self,
        export_dir: str | Path,
        policy_name: str | None = None,
        backend: str = "auto",
        device: str = "auto",
        runner: InferenceRunner | None = None,
        preprocessors: list[Preprocessor] | None = None,
        postprocessors: list[Postprocessor] | None = None,
        callbacks: list[Callback] | None = None,
        **adapter_kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Initialize InferenceModel with optional auto-detection.

        Args:
            export_dir: Directory containing exported policy files
            policy_name: Policy name (auto-detected if None)
            backend: Backend to use, or 'auto' to detect from manifest/files
            device: Device for inference ('auto', 'cpu', 'cuda', 'CPU', 'GPU', etc.)
            runner: Execution runner override. If None, auto-selected from manifest.
            preprocessors: Pipeline stages applied to observations before the
                runner.  If ``None``, loaded from manifest (empty if not
                declared).
            postprocessors: Pipeline stages applied to runner output.  If
                ``None``, loaded from manifest (empty if not declared).
            callbacks: Lifecycle callbacks for instrumentation (timing,
                logging, safety checks, etc.).  Defaults to no callbacks.
            **adapter_kwargs: Backend-specific configuration options

        Raises:
            FileNotFoundError: If export directory or required files don't exist.
            ValueError: If ``policy_name`` contains invalid characters.
        """
        self.export_dir = Path(export_dir)
        if not self.export_dir.exists():
            msg = f"Export directory not found: {export_dir}"
            raise FileNotFoundError(msg)

        self.manifest = self._load_manifest()

        if policy_name is None:
            policy_name = self._detect_policy_name()
        elif not _is_safe_policy_name(policy_name):
            msg = (
                f"policy_name {policy_name!r} contains invalid characters; "
                "only alphanumeric characters, hyphens, underscores, and dots are allowed"
            )
            raise ValueError(msg)
        self.policy_name = policy_name

        if backend == "auto":
            backend = self._detect_backend_from_manifest() or self._detect_backend()
        self.backend: str = str(backend)

        if device == "auto":
            device = self._detect_device()
        self.device = device

        self.adapter: RuntimeAdapter = get_adapter(self.backend, device=device, **adapter_kwargs)
        model_path = self._get_model_path()
        self.adapter.load(model_path)

        self.runner: InferenceRunner = runner if runner is not None else get_runner(self.manifest)

        self.preprocessors: list[Preprocessor] = (
            preprocessors if preprocessors is not None else self._load_processors(self.manifest.model.preprocessors)
        )
        self.postprocessors: list[Postprocessor] = (
            postprocessors if postprocessors is not None else self._load_processors(self.manifest.model.postprocessors)
        )

        self.input_features: list[InferenceFeature] = self._load_features(self.manifest.model.input_features)
        self.output_features: list[InferenceFeature] = self._load_features(self.manifest.model.output_features)

        self.callbacks: list[Callback] = callbacks if callbacks is not None else []

        for callback in self.callbacks:
            callback.on_load(self)

        self._action_buffer: deque[np.ndarray] = deque()

    @property
    def chunk_size(self) -> int:
        """Action chunk size from manifest (backward compat)."""
        runner_spec = self.manifest.model.runner
        if runner_spec is not None:
            chunk = runner_spec.init_args.get("chunk_size")
            if chunk is not None:
                return int(chunk)
            flat_chunk = runner_spec.flat_params.get("chunk_size")
            if flat_chunk is not None:
                return int(flat_chunk)
        return 1

    @classmethod
    def load(
        cls,
        export_dir: str | Path,
        **kwargs: Any,  # noqa: ANN401
    ) -> InferenceModel:
        """Load inference model with auto-detection.

        Args:
            export_dir: Directory containing exported policy files
            **kwargs: Additional arguments passed to __init__

        Returns:
            Initialized InferenceModel instance

        Examples:
            >>> policy = InferenceModel.load("./exports/act_policy")
            >>> policy = InferenceModel.load("./exports", backend="onnx")
        """
        return cls(export_dir=export_dir, **kwargs)

    def __call__(self, inputs: dict[str, np.ndarray | list[str]]) -> dict[str, np.ndarray]:
        """Run the full inference pipeline and return model outputs.

        Pipeline: callbacks(start) → preprocessors → _prepare_inputs →
        runner → postprocessors → callbacks(end).

        This is the generic inference API — it returns the full output
        dict without assuming any domain-specific keys.

        Args:
            inputs: Input payload as a dict mapping names to numpy arrays or lists of strings.

        Returns:
            Model outputs after runner execution and postprocessing.
        """
        for callback in self.callbacks:
            modified = callback.on_predict_start(inputs)
            if modified is not None:
                inputs = modified

        for preprocessor in self.preprocessors:
            inputs = preprocessor(inputs)

        prepared = self._prepare_inputs(inputs)
        outputs = self.runner.run(self.adapter, prepared)

        for postprocessor in self.postprocessors:
            outputs = postprocessor(outputs)

        for callback in self.callbacks:
            modified = callback.on_predict_end(outputs)
            if modified is not None:
                outputs = modified

        return outputs

    def select_action(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        """Select action for given observation.

        Domain-specific convenience method for robotics policies.
        Delegates to ``__call__`` and extracts the ``"action"`` key.

        Args:
            observation: Observation dict mapping names to numpy arrays.

        Returns:
            1-D action vector with shape ``(action_dim,)``.

        Examples:
            >>> obs = env.reset()
            >>> action = policy.select_action(obs)
            >>> next_obs, reward, done = env.step(action)
        """
        if not self._action_buffer:
            self._action_buffer.extend(self.predict_action_chunk(observation))
        return self._action_buffer.popleft()

    def predict_action_chunk(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        """Predict a chunk of actions for the given observation.

        Delegates to ``__call__`` and extracts the ``"action"`` key.

        Args:
            observation: Observation dict mapping names to numpy arrays.

        Returns:
            2-D action chunk with shape ``(chunk_size, action_dim)``.

        Raises:
            ValueError: If the output has a batch dimension greater than 1.
        """
        outputs = self(observation)
        actions = outputs[ACTION]
        # Strip the batch dimension; reject actual batches (batch > 1).
        if actions.ndim == 3:  # noqa: PLR2004
            if actions.shape[0] != 1:
                msg = (
                    f"Batched inference is not supported by predict_action_chunk: "
                    f"expected batch dimension of 1, got shape {actions.shape}"
                )
                raise ValueError(msg)
            actions = actions[0]
        return np.atleast_2d(actions)

    def reset(self) -> None:
        """Reset policy state for new episode.

        Clears runner internal state (e.g. action queues) and
        notifies all callbacks.
        Call this at the start of each episode.

        Examples:
            >>> for episode in range(num_episodes):
            ...     policy.reset()
            ...     obs = env.reset()
            ...     done = False
            ...     while not done:
            ...         action = policy.select_action(obs)
            ...         obs, reward, done = env.step(action)
        """
        self.runner.reset()
        self._action_buffer.clear()
        for callback in self.callbacks:
            callback.on_reset()

    def __enter__(self) -> Self:
        """Enter the context manager.

        Returns:
            The model instance.
        """
        return self

    def __exit__(self, *args: object) -> None:
        """Exit the context manager."""

    def _prepare_inputs(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Flatten and filter input dict for the adapter.

        Flattens nested dicts using dot notation (e.g., ``{"obs": {"image": x}}``
        becomes ``{"obs.image": x}``), then filters to only the keys the adapter
        expects.

        Args:
            inputs: Input dict mapping names to arrays. Values
                may be nested dicts, which are flattened with dot-separated keys.

        Returns:
            Flat dict containing only the adapter's expected inputs. If the
            adapter has no declared input names, returns ``inputs`` unchanged.

        Raises:
            KeyError: If an expected adapter input is not found in the
                (flattened) inputs.
        """
        expected = self.adapter.input_names

        if expected:
            flat_inputs: dict[str, np.ndarray] = {}
            for key, value in inputs.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        flat_inputs[f"{key}.{sub_key}"] = sub_value
                else:
                    flat_inputs[key] = value

            filtered: dict[str, np.ndarray] = {}
            for k in expected:
                if k in flat_inputs:
                    filtered[k] = flat_inputs[k]
                else:
                    msg = f"Expected input '{k}' not found in inputs.\nAvailable keys: {list(flat_inputs.keys())}"
                    raise KeyError(msg)

            return filtered
        return inputs

    def _load_manifest(self) -> Manifest:
        """Load export manifest from ``manifest.json``.

        Returns:
            Parsed Manifest instance, or an empty Manifest if no
            ``manifest.json`` exists in the export directory.
        """
        manifest_path = self.export_dir / "manifest.json"
        if manifest_path.exists():
            return Manifest.load(manifest_path)
        return Manifest()

    def _load_processors(self, specs: list[ComponentSpec]) -> list[Any]:
        """Instantiate preprocessors or postprocessors from component specs.

        Resolves relative ``artifact`` paths to absolute paths using
        the export directory before instantiation.

        Args:
            specs: List of component specifications to instantiate.

        Returns:
            List of instantiated processor objects.
        """
        return [instantiate_component(resolve_artifact(spec, self.export_dir)) for spec in specs]

    def _load_features(self, specs: list[ComponentSpec]) -> list[InferenceFeature]:
        """Instantiate :class:`InferenceFeature` objects from manifest specs.

        Args:
            specs: Component specifications declared in the manifest.

        Returns:
            List of materialised :class:`InferenceFeature` instances,
            preserving the declared order.

        Raises:
            TypeError: If any instantiated component is not an
                :class:`InferenceFeature` instance.
        """
        features: list[InferenceFeature] = []
        for spec in specs:
            component = instantiate_component(resolve_artifact(spec, self.export_dir))
            if not isinstance(component, InferenceFeature):
                msg = f"Expected an InferenceFeature instance from spec, got {type(component).__name__}"
                raise TypeError(msg)
            features.append(component)
        return features

    def _detect_policy_name(self) -> str:
        """Auto-detect policy name from manifest or file heuristics.

        Checks manifest ``policy.name`` first, then falls back to
        ``policy.source.class_path`` extraction, then file-name heuristics.

        Returns:
            Policy name (e.g., 'act', 'diffusion')

        Raises:
            ValueError: If policy name cannot be determined
        """
        if self.manifest.policy.name:
            name = self.manifest.policy.name
            if not _is_safe_policy_name(name):
                msg = (
                    f"manifest policy.name {name!r} contains invalid characters; "
                    "only alphanumeric characters, hyphens, underscores, and dots are allowed"
                )
                raise ValueError(msg)
            return name

        class_path = self.manifest.policy.source.class_path
        if class_path:
            parts = class_path.lower().split(".")
            min_parts_for_module_extraction = 3
            if len(parts) >= min_parts_for_module_extraction:
                return parts[-2]

        model_files = list(self.export_dir.glob("*.*"))
        if model_files:
            name = model_files[0].stem
            for suffix in ["_policy", "_model"]:
                name = name.removesuffix(suffix)
            return name

        msg = f"Cannot determine policy name from {self.export_dir}"
        raise ValueError(msg)

    def _detect_backend_from_manifest(self) -> str | None:
        """Extract backend from manifest artifacts.

        Returns:
            Backend string, or ``None`` if not found.
        """
        artifacts = self.manifest.model.artifacts
        if artifacts:
            return next(iter(artifacts))

        return None

    def _detect_backend(self) -> str:
        """Auto-detect backend from model files.

        Iterates registered backends and returns the first whose extension
        is present in :attr:`export_dir`.  Registration order in
        :data:`~physicalai.inference.adapters.backend_registry` defines
        priority when extensions overlap.

        Returns:
            Backend name.

        Raises:
            ValueError: If no registered extension matches a file in the
                export directory.
        """
        for backend in adapter_registry.names():
            for ext in adapter_registry.extensions_of(backend):
                if any(self.export_dir.glob(f"*{ext}")):
                    return backend

        msg = f"Cannot detect backend from files in {self.export_dir}"
        raise ValueError(msg)

    def _detect_device(self) -> str:
        """Auto-detect best available device using adapter-native detection.

        Returns:
            Device string for the best available device.
        """
        adapter = get_adapter(self.backend, device="cpu")
        return adapter.default_device()

    def _get_model_path(self) -> Path:
        """Get path to model file based on backend.

        Uses extensions registered in
        :data:`~physicalai.inference.adapters.backend_registry` to locate
        the artifact, in registration order.

        Returns:
            Path to model file.

        Raises:
            FileNotFoundError: If no matching model file is found.
        """
        extensions = adapter_registry.extensions_of(self.backend)

        if self.policy_name:
            for ext in extensions:
                model_path = self.export_dir / f"{self.policy_name}{ext}"
                if model_path.exists():
                    return model_path

        for ext in extensions:
            files = list(self.export_dir.glob(f"*{ext}"))
            if files:
                return files[0]

        ext_str = " or ".join(extensions)
        msg = f"No {ext_str} model file found in {self.export_dir}"
        raise FileNotFoundError(msg)

    def __repr__(self) -> str:
        """Return string representation of the model."""
        return (
            f"{self.__class__.__name__}("
            f"policy={self.policy_name}, "
            f"backend={self.backend}, "
            f"device={self.device}, "
            f"runner={self.runner!r})"
        )
