# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Production-ready inference model with unified API."""

from __future__ import annotations

import math
import re
import time
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self, cast

import numpy as np
from loguru import logger

from physicalai.config import export_config
from physicalai.inference.adapters import adapter_registry, get_adapter
from physicalai.inference.callbacks.base import Callback
from physicalai.inference.component_factory import instantiate_component, resolve_artifact
from physicalai.inference.constants import (
    ACTION,
    IMAGES,
    PREV_CHUNK_LEFT_OVER,
    RTC_EXECUTION_HORIZON,
    RTC_INFERENCE_DELAY,
    RTC_MAX_GUIDANCE_WEIGHT,
)
from physicalai.inference.data.features import InferenceFeature, InferenceFeatureType
from physicalai.inference.manifest import ComponentSpec, Manifest
from physicalai.inference.postprocessors.base import Postprocessor
from physicalai.inference.preprocessors.base import Preprocessor
from physicalai.inference.runners import get_runner
from physicalai.inference.utils._hub import download_from_hub  # noqa: PLC2701

if TYPE_CHECKING:
    from physicalai.inference.adapters.base import RuntimeAdapter
    from physicalai.inference.runners.base import InferenceRunner


# Policy names from the manifest are used to construct filesystem paths.
# Restrict to safe characters to prevent "../" traversal attacks.
_SAFE_POLICY_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.\-]*$", re.ASCII)

# Inputs an RTC (real-time chunking) export takes besides the observation.
_RTC_INPUTS = (PREV_CHUNK_LEFT_OVER, RTC_INFERENCE_DELAY, RTC_MAX_GUIDANCE_WEIGHT, RTC_EXECUTION_HORIZON)


def _is_safe_policy_name(name: str) -> bool:
    """Return True if *name* matches ``[a-zA-Z0-9][a-zA-Z0-9-_.]*`` (ASCII only)."""
    return _SAFE_POLICY_NAME_RE.fullmatch(name) is not None


def _training_camera_order(features: list[InferenceFeature]) -> list[str]:
    """Return the camera names in the order the manifest declares them.

    Exports list their visual features in training order, as ``images.<name>``
    when there is more than one camera. A single camera has no order to keep.
    """
    prefix = f"{IMAGES}."
    names = [
        feature.name.removeprefix(prefix)
        for feature in features
        if feature.ftype == InferenceFeatureType.VISUAL and feature.name.startswith(prefix)
    ]
    return names if len(names) > 1 else []


_FLOATS = frozenset({np.dtype(np.float16), np.dtype(np.float32), np.dtype(np.float64)})
_INTS = frozenset({np.dtype(np.int32), np.dtype(np.int64)})
# Largest RTC first-chunk tail filled in (elements); a pi05 tail is 50 x 32.
_MAX_FIRST_CHUNK_ELEMENTS = 1 << 20


def _as_graph_input(name: str, value: Any, dtype: np.dtype | None, *, declares_dtypes: bool) -> Any:  # noqa: ANN401
    """Return ``value`` as the graph takes it.

    Only for adapters that declare input types (the built-in OpenVINO and ONNX
    Runtime adapters); others get ``value`` unchanged, as before. ONNX Runtime
    rejects NumPy scalars such as ``np.int64(0)``, so scalars become rank-0
    arrays. A float16, float32 or float64 array becomes the one of these the
    graph declares (manifest stats turn float32 inputs into float64), and int64
    and int32 convert into each other. Anything else is left to the backend.

    Raises:
        ValueError: If a value does not fit the declared type (overflow).
    """
    if not declares_dtypes:
        return value
    if isinstance(value, np.generic):
        value = np.asarray(value)
    if dtype is None or not isinstance(value, np.ndarray) or value.dtype == dtype:
        return value
    if value.dtype in _FLOATS and dtype in _FLOATS:
        with np.errstate(over="ignore"):  # overflow is checked just below
            cast = value.astype(dtype)
        if not np.array_equal(np.isfinite(cast), np.isfinite(value)):
            msg = f"Input '{name}' has {value.dtype} values outside the range of the graph's {dtype}"
            raise ValueError(msg)
        return cast
    if value.dtype in _INTS and dtype in _INTS:
        info = np.iinfo(dtype)
        if value.size and (value.min() < info.min or value.max() > info.max):
            msg = f"Input '{name}' has {value.dtype} values outside the range of the graph's {dtype}"
            raise ValueError(msg)
        return value.astype(dtype)
    return value


def _with_keys_in_order(mapping: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    """Return a copy of *mapping* whose *keys* appear in the given order.

    The keys keep the positions they had among the other entries; only which
    key sits in each of those positions changes.
    """
    wanted = set(keys)
    in_order = iter(keys)
    slots = [next(in_order) if key in wanted else key for key in mapping]
    return {key: mapping[key] for key in slots}


@export_config(class_path="physicalai.inference.InferenceModel", scalar_var_kwargs=True)
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
        >>> policy = InferenceModel("./exports/act_policy")
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
        *,
        fill_rtc_inputs: bool = True,
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
                logging, safety checks, etc.).  If ``None``, callbacks declared
                under ``model.callbacks`` are loaded from the manifest.  An
                explicitly supplied list, including an empty list, takes
                precedence over manifest callbacks.
            fill_rtc_inputs: Give an export that takes all four RTC inputs the
                values ``RTCExecution`` sends for a first chunk when a call has
                none of them, so it runs under sync and async execution. Set it
                to ``False`` when a callback or preprocessor supplies the RTC
                inputs itself; calls then reach the hooks exactly as they are.
            **adapter_kwargs: Backend-specific configuration options

        Raises:
            FileNotFoundError: If export directory or required files don't exist.
            ValueError: If ``policy_name`` contains invalid characters.
        """
        load_started = time.monotonic()
        logger.info("Loading policy from {}", export_dir)

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

        logger.info(
            "Loaded manifest: policy={}, backend={}, device={}",
            self.policy_name,
            self.backend,
            self.device,
        )

        self.adapter: RuntimeAdapter = get_adapter(self.backend, device=device, **adapter_kwargs)
        model_path = self._get_model_path()
        compile_started = time.monotonic()
        logger.info("Compiling {} model on {}...", self.backend, self.device)
        self.adapter.load(model_path)
        logger.info("Model compiled in {:.1f}s", time.monotonic() - compile_started)

        self.runner: InferenceRunner = runner if runner is not None else get_runner(self.manifest)

        self.preprocessors: list[Preprocessor] = (
            preprocessors
            if preprocessors is not None
            else self._load_processors(self.manifest.model.preprocessors, Preprocessor)
        )
        self.postprocessors: list[Postprocessor] = (
            postprocessors
            if postprocessors is not None
            else self._load_processors(self.manifest.model.postprocessors, Postprocessor)
        )
        self.input_features: list[InferenceFeature] = self._load_features(self.manifest.model.input_features)
        self.output_features: list[InferenceFeature] = self._load_features(self.manifest.model.output_features)
        self._camera_order = _training_camera_order(self.input_features)
        self._warned_camera_mismatch = False
        self._rtc_first_chunk = self._plan_first_chunk() if fill_rtc_inputs else None
        self._logged_rtc_first_chunk = False

        self.callbacks: list[Callback] = (
            callbacks if callbacks is not None else self._load_callbacks(self.manifest.model.callbacks)
        )

        logger.info(
            "Loaded {} preprocessors, {} postprocessors, {} callbacks",
            len(self.preprocessors),
            len(self.postprocessors),
            len(self.callbacks),
        )

        for callback in self.callbacks:
            callback.on_load(self)

        self._action_buffer: deque[np.ndarray] = deque()
        self._closed = False

        logger.info(
            "InferenceModel ready (chunk_size={}, total {:.1f}s)",
            self.chunk_size,
            time.monotonic() - load_started,
        )

    @property
    def rtc_inputs(self) -> tuple[str, ...] | None:
        """The RTC inputs the graph takes, or ``None`` when that is unknown.

        Unknown when the adapter does not name its inputs or there is no local
        adapter (a remote model). ``RTCExecution`` refuses a graph that takes
        none of them, since it would drop them and run every chunk unguided.
        """
        graph_inputs = getattr(getattr(self, "adapter", None), "input_names", None)
        if not graph_inputs:
            return None
        return tuple(name for name in _RTC_INPUTS if name in graph_inputs)

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
    def from_pretrained(
        cls,
        repo_id: str,
        *,
        revision: str | None = None,
        cache_dir: str | Path | None = None,
        allow_patterns: list[str] | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> InferenceModel:
        """Load an inference model from a Hugging Face Hub repository or local path.

        If ``repo_id`` points to an existing local directory, it is loaded
        directly as an export. Otherwise the repository snapshot is downloaded
        to a local cache directory and then loaded like a local export.

        .. warning::
            The consumer shall ensure the ``repo_id`` points to a trusted directory.

        Args:
            repo_id: Hub repository identifier (e.g. ``"OpenVINO/act-fp16-ov"``)
                or a path to a local export directory.
            revision: Hub git revision (branch, tag, or commit SHA). Pin to a
                commit SHA for reproducible, tamper-evident loads. Ignored when
                ``repo_id`` is a local directory.
            cache_dir: Cache directory for the download. Ignored when ``repo_id``
                is a local directory.
            allow_patterns: Optional glob patterns limiting which files are
                downloaded. When ``None``, the full snapshot is fetched. Ignored
                when ``repo_id`` is a local directory.
            **kwargs: Additional arguments passed to ``__init__``.

        Returns:
            Initialized InferenceModel instance.

        Raises:
            TypeError: If ``export_dir`` is passed in ``kwargs``.

        Examples:
            >>> policy = InferenceModel.from_pretrained("OpenVINO/act-fp16-ov")
            >>> policy = InferenceModel.from_pretrained(
            ...     "OpenVINO/act-fp16-ov", revision="main"
            ... )
            >>> policy = InferenceModel.from_pretrained("./exports/act_policy")
        """
        if "export_dir" in kwargs:
            msg = "from_pretrained() does not accept export_dir; it is set from the downloaded snapshot"
            raise TypeError(msg)

        if Path(repo_id).is_dir():
            return cls(export_dir=repo_id, **kwargs)

        local_dir = download_from_hub(
            repo_id,
            revision=revision,
            cache_dir=cache_dir,
            allow_patterns=allow_patterns,
        )
        return cls(export_dir=local_dir, **kwargs)

    def __call__(self, inputs: dict[str, np.ndarray | list[str]]) -> dict[str, np.ndarray]:
        """Run the full inference pipeline and return model outputs.

        Pipeline: RTC first-chunk inputs → callbacks(start) → preprocessors →
        _prepare_inputs → runner → postprocessors → callbacks(end).

        An RTC capable export called without RTC inputs, as sync and async
        execution call it, gets the inputs ``RTCExecution`` sends for a first
        chunk: no previous chunk, guidance weight 1e-8, horizon 1, delay 0.
        They go through the whole pipeline, as RTC's own do. This happens only
        when none of the four RTC inputs is in ``inputs`` and the manifest
        declares the shape of ``prev_chunk_left_over``. A pipeline that supplies
        its own RTC values must therefore put them in ``inputs``: a callback or
        preprocessor that adds them only when they are missing sees the
        first-chunk values already there.

        This is the generic inference API — it returns the full output
        dict without assuming any domain-specific keys.

        Args:
            inputs: Input payload as a dict mapping names to numpy arrays or lists of strings.

        Returns:
            Model outputs after runner execution and postprocessing.
        """
        inputs = self._add_rtc_first_chunk_inputs(inputs)
        for callback in self.callbacks:
            modified = callback.on_predict_start(inputs)
            if modified is not None:
                inputs = modified

        inputs = self._order_cameras(inputs)
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

    def close(self) -> None:
        """Release the adapter's resources.

        Notifies every callback, then closes the adapter, even if a cleanup hook
        fails. Any cleanup failures are raised together as an ``ExceptionGroup``.
        Later calls have no effect. Adapters that run a worker process or hold
        device memory release it here.

        Raises:
            ExceptionGroup: If one or more callbacks or the adapter fail to close.

        Examples:
            >>> with InferenceModel("./exports/act_policy") as policy:
            ...     action = policy.select_action(obs)
        """
        if self._closed:
            return
        self._closed = True
        errors: list[Exception] = []
        for callback in self.callbacks:
            try:
                callback.on_close()
            except Exception as exc:  # noqa: BLE001  # Continue closing resources after a hook failure.
                errors.append(exc)
        try:
            self.adapter.close()
        except Exception as exc:  # noqa: BLE001  # Continue cleanup before reporting failures.
            errors.append(exc)
        if errors:
            msg = "Errors occurred while closing the inference model"
            raise ExceptionGroup(msg, errors)

    def __enter__(self) -> Self:
        """Enter the context manager.

        Returns:
            The model instance.
        """
        return self

    def __exit__(self, *args: object) -> None:
        """Close the model when the context manager exits."""
        self.close()

    def _plan_first_chunk(self) -> tuple[tuple[int, ...], np.dtype, np.dtype] | None:
        """Return the RTC first-chunk tail shape and the tail and weight dtypes, or ``None`` to fill nothing.

        Only an export that takes all four RTC inputs gets one, and only when its
        manifest declares a positive ``(chunk_size, action_dim)`` shape for
        ``prev_chunk_left_over`` that fits what the graph declares and is not
        unreasonably large. Floats follow the graph's declared float type.
        """
        if self.rtc_inputs != _RTC_INPUTS:
            return None
        feature = next((f for f in self.input_features if f.name == PREV_CHUNK_LEFT_OVER), None)
        shape = (1, *feature.shape) if feature is not None else ()
        graph_shapes = getattr(self.adapter, "input_shapes", None)
        graph_shapes = graph_shapes if isinstance(graph_shapes, dict) else {}
        reason = None
        if len(shape) != 3 or not all(isinstance(size, int) and size > 0 for size in shape):  # noqa: PLR2004
            reason = f"the manifest declares no fixed (chunk_size, action_dim) shape for it (got {list(shape[1:])})"
        elif math.prod(shape) > _MAX_FIRST_CHUNK_ELEMENTS:  # Python ints: no wraparound
            reason = f"its declared shape {list(shape)} is larger than {_MAX_FIRST_CHUNK_ELEMENTS} elements"
        else:
            expected = {PREV_CHUNK_LEFT_OVER: shape, **dict.fromkeys(_RTC_INPUTS[1:], ())}
            for name, generated in expected.items():
                declared = graph_shapes.get(name)
                if declared is not None and (
                    len(declared) != len(generated)
                    or any(want is not None and want != got for want, got in zip(declared, generated, strict=True))
                ):
                    reason = f"the graph declares {name} as {list(declared)}, not {list(generated)}"
                    break
        if reason is not None:
            logger.warning(
                "This export takes RTC inputs but they cannot be filled in for sync or async execution: {}. It runs"
                " with RTCExecution or with RTC inputs in the observation.",
                reason,
            )
            return None
        dtypes = getattr(self.adapter, "input_dtypes", None)
        dtypes = dtypes if isinstance(dtypes, dict) else {}

        def float_type(name: str) -> np.dtype:
            declared = dtypes.get(name)
            return declared if isinstance(declared, np.dtype) and declared in _FLOATS else np.dtype(np.float32)

        return shape, float_type(PREV_CHUNK_LEFT_OVER), float_type(RTC_MAX_GUIDANCE_WEIGHT)

    def _add_rtc_first_chunk_inputs(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Add what ``RTCExecution`` sends for a first chunk when an RTC export gets no RTC inputs.

        Never raises: inputs that still do not fit fail later, where the graph
        inputs are checked.

        Returns:
            ``inputs`` itself, or a shallow copy with the four RTC inputs added.
        """
        if self._rtc_first_chunk is None or any(name in inputs for name in _RTC_INPUTS):
            return inputs
        if not self._logged_rtc_first_chunk:
            logger.info(
                "This export takes RTC inputs and the observation has none (sync or async execution); using what"
                " RTCExecution sends for a first chunk: no previous chunk, guidance weight 1e-8."
            )
            self._logged_rtc_first_chunk = True
        shape, tail_dtype, weight_dtype = self._rtc_first_chunk
        return {
            **inputs,
            PREV_CHUNK_LEFT_OVER: np.zeros(shape, dtype=tail_dtype),
            RTC_INFERENCE_DELAY: np.int64(0),
            RTC_MAX_GUIDANCE_WEIGHT: weight_dtype.type(1e-8),
            RTC_EXECUTION_HORIZON: np.int64(1),
        }

    def _order_cameras(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Put the observation's cameras in the order the policy was trained on.

        Preprocessors such as Pi0.5's stack cameras in observation order, so an
        observation that lists the same cameras in another order would silently
        feed each image to the wrong camera slot. The manifest's input features
        name the cameras in training order.

        Args:
            inputs: Observation with cameras under ``images.<name>`` keys or a
                nested ``images`` dict.

        Returns:
            The observation with its cameras in training order, or ``inputs``
            itself when they already are or do not match the trained cameras.
        """
        if not self._camera_order:
            return inputs
        nested = inputs.get(IMAGES)
        prefix = "" if isinstance(nested, dict) else f"{IMAGES}."
        images = nested if isinstance(nested, dict) else inputs
        names = [key.removeprefix(prefix) for key in images if key.startswith(prefix) and "is_pad" not in key]
        if names == self._camera_order:
            return inputs
        if set(names) != set(self._camera_order):
            if not self._warned_camera_mismatch:
                logger.warning(
                    "Observation cameras {} do not match the trained cameras {}; using the observation order.",
                    sorted(names),
                    self._camera_order,
                )
                self._warned_camera_mismatch = True
            return inputs
        ordered = _with_keys_in_order(images, [f"{prefix}{name}" for name in self._camera_order])
        return {**inputs, IMAGES: ordered} if isinstance(nested, dict) else ordered

    def _prepare_inputs(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Flatten and filter input dict for the adapter.

        Flattens nested dicts using dot notation (e.g., ``{"obs": {"image": x}}``
        becomes ``{"obs.image": x}``), then filters to only the keys the adapter
        expects. Only the four RTC inputs are converted, and only for adapters
        that declare input types: NumPy scalars become rank-0 arrays, a float
        array becomes the float16, float32 or float64 the graph declares (stats in
        float64 would otherwise turn the float32 tail into float64, which ONNX
        Runtime rejects), and int64 and int32 convert into each other when the
        values fit. Every other input reaches the adapter unchanged.

        Args:
            inputs: Input dict mapping names to arrays. Values
                may be nested dicts, which are flattened with dot-separated keys.

        Returns:
            Flat dict containing only the adapter's expected inputs. If the
            adapter has no declared input names, returns ``inputs`` unchanged.

        Raises:
            KeyError: If an expected adapter input is not found in the
                (flattened) inputs.
            ValueError: If two inputs expand to the same flattened key.
        """
        expected = self.adapter.input_names

        if expected:
            declared = getattr(self.adapter, "input_dtypes", None)
            dtypes: dict[str, np.dtype] = declared if isinstance(declared, dict) else {}
            flat_inputs: dict[str, np.ndarray] = {}
            for key, value in inputs.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        flat_key = f"{key}.{sub_key}"
                        if flat_key in flat_inputs:
                            msg = (
                                f"Key collision in inputs: '{flat_key}' produced by"
                                f" both a flat key and nested dict '{key}'"
                            )
                            raise ValueError(msg)
                        flat_inputs[flat_key] = sub_value
                else:
                    if key in flat_inputs:
                        msg = f"Key collision in inputs: '{key}' produced by both a nested expansion and a flat key"
                        raise ValueError(msg)
                    flat_inputs[key] = value

            filtered: dict[str, np.ndarray] = {}
            for k in expected:
                if k in flat_inputs:
                    value = flat_inputs[k]
                    if k in _RTC_INPUTS:
                        value = _as_graph_input(k, value, dtypes.get(k), declares_dtypes=bool(dtypes))
                    filtered[k] = value
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

    def _load_processors(self, specs: list[ComponentSpec], base: type) -> list[Any]:
        """Instantiate preprocessors or postprocessors from component specs.

        Resolves relative ``artifact`` paths to absolute paths using
        the export directory before instantiation.

        Args:
            specs: List of component specifications to instantiate.
            base: Expected processor base class.

        Returns:
            List of instantiated processor objects.
        """
        return [instantiate_component(base, resolve_artifact(spec, self.export_dir)) for spec in specs]

    def _load_callbacks(self, specs: list[ComponentSpec]) -> list[Callback]:
        """Instantiate callbacks declared in a manifest in declaration order.

        Args:
            specs: Callback component specifications.

        Returns:
            Instantiated callbacks in declaration order.

        Raises:
            TypeError: If a callback specification cannot be resolved or
                instantiated.
        """
        callbacks: list[Callback] = []
        for index, spec in enumerate(specs):
            try:
                callback = cast("Callback", instantiate_component(Callback, resolve_artifact(spec, self.export_dir)))
            except (ImportError, TypeError, ValueError) as exc:
                msg = f"Invalid manifest callback at index {index}: {exc}"
                raise TypeError(msg) from exc
            callbacks.append(callback)
        return callbacks

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
            component = instantiate_component(InferenceFeature, resolve_artifact(spec, self.export_dir))
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

        model_files = sorted(self.export_dir.glob("*.*"))
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

        Prefers the filename the manifest declares for this backend in
        ``model.artifacts`` so the loaded model is exactly the reviewed
        artifact. Only when the manifest declares none does it fall back to
        ``<policy_name><ext>`` and then to a search of the export directory,
        using extensions registered in
        :data:`~physicalai.inference.adapters.backend_registry`.

        Returns:
            Path to model file.

        Raises:
            FileNotFoundError: If the declared artifact is missing, or no
                matching model file is found.
        """
        declared = self.manifest.model.artifacts.get(self.backend)
        if declared:
            model_path = self._resolve_export_path(declared)
            if not model_path.exists():
                msg = f"Model artifact {declared!r} declared in the manifest is missing from {self.export_dir}"
                raise FileNotFoundError(msg)
            return model_path

        extensions = adapter_registry.extensions_of(self.backend)

        if self.policy_name:
            for ext in extensions:
                model_path = self.export_dir / f"{self.policy_name}{ext}"
                if model_path.exists():
                    return model_path

        for ext in extensions:
            files = sorted(self.export_dir.glob(f"*{ext}"))
            if files:
                return files[0]

        ext_str = " or ".join(extensions)
        msg = f"No {ext_str} model file found in {self.export_dir}"
        raise FileNotFoundError(msg)

    def _resolve_export_path(self, name: str) -> Path:
        r"""Resolve a manifest-declared filename against the export directory.

        The check is lexical and uses the parts of *name* as the current
        platform parses them: a rooted or drive-qualified name (any non-empty
        :attr:`~pathlib.PurePath.anchor`) or a ``".."`` component is rejected,
        so the returned path always lies inside the export directory by name.
        On Windows this rejects names such as ``\Windows\model.xml``; on POSIX
        a backslash is an ordinary filename character, so such a name is one
        literal file inside the export directory.

        The result is deliberately not resolved: in a Hugging Face Hub snapshot
        the artifact is a symlink into a sibling ``blobs/`` store, and following
        it would reject every downloaded model. This matches
        :func:`~physicalai.inference.component_factory.resolve_artifact`.

        Returns:
            The absolute path to the artifact inside the export directory.

        Raises:
            ValueError: If *name* escapes the export directory or names the
                export directory itself (for example ``"."``).
        """
        relative = Path(name)
        if relative.anchor or ".." in relative.parts:
            msg = f"Model artifact {name!r} escapes the export directory"
            raise ValueError(msg)
        if not relative.parts:
            msg = f"Model artifact {name!r} does not name a file in the export directory"
            raise ValueError(msg)
        return self.export_dir.resolve() / relative

    def __repr__(self) -> str:
        """Return string representation of the model."""
        return (
            f"{self.__class__.__name__}("
            f"policy={self.policy_name}, "
            f"backend={self.backend}, "
            f"device={self.device}, "
            f"runner={self.runner!r})"
        )
