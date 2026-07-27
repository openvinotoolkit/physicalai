# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""OpenVINO golden-action tests — golden_action marker.

Numerically compares the action chunks produced by an OpenVINO exported policy
against those produced by the corresponding native PyTorch policy on identical
real-world observations.  A regression in numerical accuracy caused by an
OpenVINO upgrade will surface as an L2 distance above the tolerance threshold.

Models tested:
    OpenVINO/pi05-libero-fp32-ov     vs  lerobot/pi05_libero_finetuned_v044
    OpenVINO/smolvla-libero-fp16-ov  vs  HuggingFaceVLA/smolvla_libero

Observations:
    Real samples from lerobot/libero_10_image (5 samples by default).

Design notes:
  - The native model is loaded with use_random_input_noise=False so the
    flow-matching denoising step is deterministic.  The same RNG seed is
    applied to both paths before each forward pass.
  - L2 threshold: 0.02 (agreed between physicalai and OV teams for FP16
    exports on CPU; matches THRESHOLD_L2 in the dl-benchmark pai_accuracy.py).
  - The test is self-contained: it downloads all artifacts at run time via
    download_from_hub / snapshot_download and requires no pre-staged files.

Run:
    pytest -m golden_action -s                         # all models
    pytest -m golden_action -k pi05                    # pi05 only
    pytest -m golden_action -s --log-cli-level=INFO    # verbose, shows L2 per sample

Environment:
    OV_GOLDEN_CACHE_DIR   optional path for caching downloaded artifacts
                          (models + dataset). Avoids re-downloading on
                          repeated runs.  Separate from OV_SMOKE_CACHE_DIR
                          but the same directory may be reused.
    OV_GOLDEN_NUM_SAMPLES number of real observations to compare (default 5).
    OV_GOLDEN_L2_THRESHOLD L2 tolerance override (default 0.02).
    OV_GOLDEN_DEVICE      OpenVINO inference device (default "CPU").

Required extras:
    physicalai-train  — native PyTorch model loading (Pi05 / SmolVLA).
    lerobot           — LeRobot dataset access (LeRobotDataModule, FormatConverter).
    openvino          — OV runtime.

Co-install notes (physicalai + physicalai-train in the same environment):
    Two dependency conflicts arise when both packages are installed together.
    Use ``uv pip install --override <overrides.txt>`` to resolve:

    1. openvino pin: physicalai pins ``openvino==2026.1`` (exact), which blocks
       any newer OV release.  Override to ``openvino>=2026.1.0``.

    2. transformers pin: physicalai requires ``transformers>=5.5.0,<5.6.0``
       (bumped for a security fix on 2026-07-14) while physicalai-train requires
       ``transformers>=5.3.0,<5.4.0`` (5.4+ breaks SmolVLA torch.jit.trace
       during export).  Override to ``transformers>=5.3.0,<5.4.0``.

    Example overrides.txt::

        openvino>=2026.1.0
        openvino-tokenizers>=2026.1.0
        transformers>=5.3.0,<5.4.0
"""

from __future__ import annotations

import importlib.metadata
import inspect
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from physicalai.inference.model import InferenceModel
from physicalai.inference.utils._hub import download_from_hub

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)

pytestmark = [
    pytest.mark.golden_action,
    pytest.mark.integration,
    pytest.mark.requires_download,
]

# Skip the entire module when either runtime is absent.
pytest.importorskip("openvino", reason="openvino not installed — skipping golden_action suite")
pytest.importorskip("torch", reason="torch not installed — golden_action requires physicalai-train")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# L2-norm ceiling on the per-sample action-vector diff.
# Matches THRESHOLD_L2 in the dl-benchmark pai_accuracy.py.
_THRESHOLD_L2: float = 0.02

# Dataset keys that are label/bookkeeping metadata and must never be forwarded
# to the policy (same list as _DATASET_METADATA_KEYS in pai_accuracy.py).
_DATASET_METADATA_KEYS: frozenset[str] = frozenset({
    "action", "next_reward", "next_success",
    "episode_index", "frame_index", "index", "task_index", "timestamp",
    "info", "extra",
})

# ---------------------------------------------------------------------------
# Model specifications
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _GoldenSpec:
    """Paired OV export and native PyTorch checkpoint for golden-action comparison."""

    ov_repo_id: str
    native_repo_id: str
    family: str  # "pi05" or "smolvla"
    l2_threshold: float | None = None  # per-model override; None → use global default

    @property
    def short_id(self) -> str:
        return self.ov_repo_id.split("/")[-1]


_ALL_GOLDEN: list[_GoldenSpec] = [
    _GoldenSpec(
        # pi05-libero-fp32-ov was re-exported with adaRMSNorm conditioning support.
        # The native checkpoint (lerobot/pi05_libero_finetuned_v044) runs in bfloat16;
        # the OV model runs in float32, producing an unavoidable ~0.05 L2 gap.
        # Threshold is set to 0.10 to give 2× headroom above the observed ~0.047 floor.
        ov_repo_id="OpenVINO/pi05-libero-fp32-ov",
        native_repo_id="lerobot/pi05_libero_finetuned_v044",
        family="pi05",
        l2_threshold=0.10,
    ),
    _GoldenSpec(
        ov_repo_id="OpenVINO/smolvla-libero-fp16-ov",
        # The OV export was converted from HuggingFaceVLA/smolvla_libero
        # (8-dim state, image + image2 cameras), NOT lerobot/smolvla_libero
        # (6-dim state, camera1/2/3).  See README of smolvla-libero-fp16-ov.
        native_repo_id="HuggingFaceVLA/smolvla_libero",
        family="smolvla",
    ),
]

_DATASET_ID = "lerobot/libero_10_image"

# ---------------------------------------------------------------------------
# Helpers — native model loading (mirrors pai_accuracy.py patterns)
# ---------------------------------------------------------------------------


def _load_policy_class(module_name: str, class_name: str) -> Any:
    """Dynamically load a policy class; returns None when the import fails."""
    try:
        module = __import__(module_name, fromlist=[class_name])
        return getattr(module, class_name)
    except Exception:  # noqa: BLE001
        return None


def _parse_policy_features(cfg: dict[str, Any]) -> tuple[Any, Any]:
    """Parse input/output PolicyFeature dicts from config.json; returns (None, None) on failure."""
    try:
        from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: PLC0415
    except ImportError:
        return None, None

    def _parse(section: dict[str, Any] | None) -> dict[str, Any] | None:
        if not isinstance(section, dict):
            return None
        result = {}
        for name, spec in section.items():
            result[name] = PolicyFeature(type=FeatureType(spec["type"]), shape=tuple(spec["shape"]))
        return result or None

    return _parse(cfg.get("input_features")), _parse(cfg.get("output_features"))


def _load_native_model(checkpoint_dir: Path, family: str) -> Any:
    """Load a native Pi05 / SmolVLA policy from a HF-style checkpoint directory.

    Mirrors the load_native_model + _create_native_policy pattern from
    pai_accuracy.py.  The policy is always loaded with
    ``use_random_input_noise=False`` so every forward pass is deterministic.

    Args:
        checkpoint_dir: Local path containing config.json + model.safetensors.
        family: "pi05" or "smolvla".

    Returns:
        Policy instance in eval() mode.

    Raises:
        RuntimeError: When no compatible policy class can be loaded.
    """
    import torch  # noqa: PLC0415

    with (checkpoint_dir / "config.json").open(encoding="utf-8") as f:
        cfg = json.load(f)

    chunk_size = int(cfg.get("chunk_size", 50))
    n_action_steps = min(int(cfg.get("n_action_steps", chunk_size)), chunk_size)
    input_features, output_features = _parse_policy_features(cfg)

    candidates: dict[str, list[tuple[str, str]]] = {
        "pi05": [
            ("physicalai.policies.pi05", "Pi05"),
            ("physicalai.policies", "Pi05"),
            ("physicalai.policies.lerobot", "PI05"),
        ],
        "smolvla": [
            ("physicalai.policies", "SmolVLA"),
            ("physicalai.policies.lerobot", "SmolVLA"),
        ],
    }

    torch.manual_seed(42)
    errors: list[str] = []
    for module_name, class_name in candidates.get(family, []):
        policy_cls = _load_policy_class(module_name, class_name)
        if policy_cls is None:
            continue
        params = inspect.signature(policy_cls).parameters
        kwargs: dict[str, Any] = {}
        if "pretrained_name_or_path" in params:
            kwargs["pretrained_name_or_path"] = checkpoint_dir
        if "compile_model" in params:
            kwargs["compile_model"] = False
        if "use_random_input_noise" in params:
            kwargs["use_random_input_noise"] = False
        if "n_action_steps" in params:
            kwargs["n_action_steps"] = n_action_steps
        if "chunk_size" in params:
            kwargs["chunk_size"] = chunk_size
        if "input_features" in params and input_features is not None:
            kwargs["input_features"] = input_features
        if "output_features" in params and output_features is not None:
            kwargs["output_features"] = output_features
        try:
            log.info("golden_action | trying %s.%s", module_name, class_name)
            policy = policy_cls(**kwargs)
            policy.eval()
            log.info("golden_action | loaded native %s from %s", type(policy).__name__, checkpoint_dir)
            return policy
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{module_name}.{class_name}: {exc}")

    details = "\n  ".join(errors) if errors else "no compatible class found"
    msg = f"Cannot load native {family} policy from {checkpoint_dir}:\n  {details}"
    raise RuntimeError(msg)


# ---------------------------------------------------------------------------
# Helpers — observation preparation
# ---------------------------------------------------------------------------


def _unwrap_feature_dict(key: str, value: Any) -> Any:
    """Unwrap nested state/action dicts produced by FormatConverter.to_observation().

    FormatConverter nests merged arrays under "observation.<key>" alongside
    per-joint sub-components.  Only the merged entry matches the model dim.
    """
    if not isinstance(value, dict):
        return value
    for candidate in (f"observation.{key}", key):
        if candidate in value:
            return value[candidate]
    return next(iter(value.values()))


def _load_dataset_observations(
    dataset_id: str,
    num_samples: int,
    cache_dir: Path | None,
) -> list[dict[str, Any]]:
    """Load real observations from a LeRobot dataset.

    Args:
        dataset_id: HuggingFace dataset repo ID.
        num_samples: Maximum number of samples to load.
        cache_dir: Optional local cache directory for the dataset.

    Returns:
        List of flat observation dicts (numpy arrays, no batch dim).

    Raises:
        ImportError: When physicalai-train / lerobot is not installed.
    """
    if cache_dir is not None:
        os.environ.setdefault("HF_LEROBOT_HOME", str(cache_dir))
        os.environ.setdefault("HF_DATASETS_CACHE", str(cache_dir))

    try:
        import torch  # noqa: PLC0415
        from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: PLC0415
        from physicalai.data import LeRobotDataModule  # noqa: PLC0415
        from physicalai.data.lerobot import FormatConverter  # noqa: PLC0415
    except ImportError as exc:
        msg = (
            "physicalai-train + lerobot are required for dataset observations. "
            f"Original error: {exc}"
        )
        raise ImportError(msg) from exc

    torch.manual_seed(42)
    # episodes=list(range(5)) limits to the first 5 episodes — enough for
    # num_samples=5 while avoiding a full dataset scan on large repos.
    lerobot_dataset = LeRobotDataset(repo_id=dataset_id, episodes=list(range(5)))
    datamodule = LeRobotDataModule(
        dataset=lerobot_dataset,
        train_batch_size=1,
        val_batch_size=1,
        num_workers=0,
    )
    datamodule.setup("fit")
    dataloader = datamodule.train_dataloader()

    samples: list[dict[str, Any]] = []
    for i, batch in enumerate(dataloader):
        if i >= num_samples:
            break
        obs_obj = FormatConverter.to_observation(batch)
        raw = obs_obj.to_numpy().to_dict(flatten=False)
        # Strip the leading batch dim added by train_batch_size=1.
        sample = {k: (v[0] if isinstance(v, np.ndarray) and v.shape[:1] == (1,) else v)
                  for k, v in raw.items()}
        # Also strip batch dim from per-camera arrays nested inside the images dict.
        if isinstance(sample.get("images"), dict):
            sample["images"] = {
                cam: (arr[0] if isinstance(arr, np.ndarray) and arr.shape[:1] == (1,) else arr)
                for cam, arr in sample["images"].items()
            }
        # Normalize LIBERO camera names to the convention used by all models:
        # lerobot/libero_10_image uses "wrist_image"; smolvla / pi05 exports
        # and HuggingFaceVLA/smolvla_libero expect "image2".
        if isinstance(sample.get("images"), dict) and "wrist_image" in sample["images"]:
            sample["images"]["image2"] = sample["images"].pop("wrist_image")
        samples.append(sample)

    if not samples:
        msg = f"No samples loaded from '{dataset_id}'."
        raise RuntimeError(msg)

    log.info("golden_action | loaded %d observations from %s", len(samples), dataset_id)
    return samples


# ---------------------------------------------------------------------------
# Helpers — prediction
# ---------------------------------------------------------------------------


def _predict_native(policy: Any, observation: dict[str, Any], seed: int) -> np.ndarray:
    """Run one forward pass through the native PyTorch policy.

    Remaps flat observation keys (LeRobot dotted / dataset-nested format) to
    the Observation dataclass fields (state, images, task).  uint8 images are
    scaled to float32 [0, 1] to match the scaling applied by OV preprocessors.

    Args:
        policy: Native Pi05 / SmolVLA instance.
        observation: Flat numpy dict (no batch dim).
        seed: RNG seed applied before the forward pass for determinism.

    Returns:
        Flattened action chunk as float32 ndarray.
    """
    import torch  # noqa: PLC0415
    from physicalai.data import Observation  # noqa: PLC0415

    torch.manual_seed(seed)
    np.random.seed(seed)
    policy.reset()

    def _to_tensor(arr: np.ndarray) -> "torch.Tensor":  # type: ignore[name-defined]
        a = np.asarray(arr)
        if a.dtype == np.uint8:
            a = a.astype(np.float32) / 255.0
        return torch.from_numpy(a).unsqueeze(0)

    images: dict[str, Any] = {}
    batched: dict[str, Any] = {}

    for k, v in observation.items():
        if k in _DATASET_METADATA_KEYS:
            continue
        if k == "images" and isinstance(v, dict):
            for cam, arr in v.items():
                images[cam] = _to_tensor(arr) if isinstance(arr, np.ndarray) else arr
            continue
        v = _unwrap_feature_dict(k, v)
        t = _to_tensor(v) if isinstance(v, np.ndarray) else v
        if k.startswith("observation.images."):
            images[k[len("observation.images."):]] = t
        elif k in ("observation.state", "state"):
            batched["state"] = t
        elif k in ("task", "language", "prompt", "observation.task"):
            batched["task"] = [v] if isinstance(v, str) else v
        else:
            batched[k] = t

    if images:
        batched["images"] = images
    batched.setdefault("task", ["do the task"])

    obs = Observation.from_dict(batched)
    with torch.no_grad():
        action_tensor = policy.predict_action_chunk(obs)
    return action_tensor.cpu().numpy().reshape(-1)


def _predict_exported(model: InferenceModel, observation: dict[str, Any], seed: int) -> np.ndarray:
    """Run one forward pass through the exported InferenceModel.

    Remaps flat observation keys to the canonical keys consumed by the
    physicalai preprocessors (state, images, task) and adds a batch dim.

    Args:
        model: Loaded InferenceModel.
        observation: Flat numpy dict (no batch dim).
        seed: RNG seed applied before the forward pass for determinism.

    Returns:
        Flattened action chunk as float32 ndarray.
    """
    np.random.seed(seed)
    model.reset()

    images: dict[str, np.ndarray] = {}
    inputs: dict[str, Any] = {}
    task_value: Any = None

    for k, v in observation.items():
        if k in _DATASET_METADATA_KEYS:
            continue
        if k != "images":
            v = _unwrap_feature_dict(k, v)
        if k.startswith("observation.images."):
            images[k[len("observation.images."):]] = np.asarray(v)[None, ...]
        elif k.startswith("images."):
            images[k[len("images."):]] = np.asarray(v)[None, ...]
        elif k in ("observation.state", "state"):
            inputs["state"] = np.asarray(v)[None, ...]
        elif k in ("task", "language", "prompt", "observation.task"):
            task_value = v
        elif k == "images" and isinstance(v, dict):
            for cam, arr in v.items():
                images[cam] = np.asarray(arr)[None, ...]
        else:
            arr = np.asarray(v)
            inputs[k] = arr[None, ...] if arr.dtype.kind in "fiub" else v

    if images:
        inputs["images"] = images
    if task_value is None:
        inputs["task"] = ["do the task"]
    elif isinstance(task_value, str):
        inputs["task"] = [task_value]
    else:
        inputs["task"] = list(task_value)

    action = model.predict_action_chunk(inputs)
    if hasattr(action, "numpy"):
        action = action.numpy()
    return np.asarray(action).reshape(-1)


# ---------------------------------------------------------------------------
# Session-scoped fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def golden_cache_dir() -> Path | None:
    """Optional artifact + dataset cache from OV_GOLDEN_CACHE_DIR env var."""
    env = os.environ.get("OV_GOLDEN_CACHE_DIR")
    return Path(env) if env else None


@pytest.fixture(scope="session")
def golden_device() -> str:
    """OpenVINO inference device from OV_GOLDEN_DEVICE env var (default 'CPU')."""
    return os.environ.get("OV_GOLDEN_DEVICE", "CPU")


@pytest.fixture(scope="session")
def golden_num_samples() -> int:
    """Number of real observations to compare from OV_GOLDEN_NUM_SAMPLES (default 5)."""
    return int(os.environ.get("OV_GOLDEN_NUM_SAMPLES", "5"))


@pytest.fixture(scope="session")
def golden_l2_threshold() -> float:
    """L2 tolerance override from OV_GOLDEN_L2_THRESHOLD (default 0.02)."""
    return float(os.environ.get("OV_GOLDEN_L2_THRESHOLD", str(_THRESHOLD_L2)))


@pytest.fixture(scope="session", autouse=True)
def _report_golden_versions() -> None:
    """Log OV / openvino-tokenizers versions once per session."""
    def _ver(pkg: str) -> str:
        try:
            return importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            return "not installed"

    log.info(
        "golden_action | openvino==%s | openvino-tokenizers==%s",
        _ver("openvino"),
        _ver("openvino-tokenizers"),
    )


@pytest.fixture(scope="session")
def dataset_observations(
    golden_num_samples: int,
    golden_cache_dir: Path | None,
) -> list[dict[str, Any]]:
    """Load real observations from lerobot/libero_10_image (session-scoped, downloaded once)."""
    try:
        return _load_dataset_observations(
            dataset_id=_DATASET_ID,
            num_samples=golden_num_samples,
            cache_dir=golden_cache_dir,
        )
    except ImportError as exc:
        pytest.skip(f"physicalai-train / lerobot not installed: {exc}")


@pytest.fixture(
    scope="session",
    params=_ALL_GOLDEN,
    ids=lambda s: s.short_id,
)
def golden_pair(
    request: pytest.FixtureRequest,
    golden_cache_dir: Path | None,
    golden_device: str,
) -> tuple[Any, InferenceModel, _GoldenSpec]:
    """Download and load both the native and OV exported policy for each spec.

    Returns:
        (native_policy, ov_model, spec)
    """
    spec: _GoldenSpec = request.param

    # --- OV export -----------------------------------------------------------
    ov_dir = download_from_hub(spec.ov_repo_id, cache_dir=golden_cache_dir)
    log.info("golden_action | loading OV export: %s  device=%s", spec.ov_repo_id, golden_device)
    ov_model = InferenceModel(export_dir=ov_dir, device=golden_device)

    # --- Native checkpoint ---------------------------------------------------
    log.info("golden_action | downloading native checkpoint: %s", spec.native_repo_id)
    try:
        from huggingface_hub import snapshot_download  # noqa: PLC0415
    except ImportError as exc:
        pytest.skip(f"huggingface_hub not installed: {exc}")

    hf_kwargs: dict[str, Any] = {}
    if golden_cache_dir is not None:
        hf_kwargs["cache_dir"] = str(golden_cache_dir)

    native_dir = Path(str(snapshot_download(repo_id=spec.native_repo_id, **hf_kwargs)))

    try:
        native_policy = _load_native_model(native_dir, family=spec.family)
    except (RuntimeError, ImportError) as exc:
        pytest.skip(f"[{spec.short_id}] native model load failed: {exc}")

    return native_policy, ov_model, spec


# ---------------------------------------------------------------------------
# Golden-action test
# ---------------------------------------------------------------------------


def test_golden_action(
    golden_pair: tuple[Any, InferenceModel, _GoldenSpec],
    dataset_observations: list[dict[str, Any]],
    golden_l2_threshold: float,
) -> None:
    # Use per-model threshold when set; fall back to session-level default.
    """Golden-action: OV action chunks are numerically close to native PyTorch outputs.

    For each sample in dataset_observations:
      1. Run the native PyTorch policy (deterministic — use_random_input_noise=False).
      2. Run the OV exported model with the same RNG seed.
      3. Assert L2(native, ov) <= golden_l2_threshold.

    A failure means the OV export produces action chunks that deviate from the
    PyTorch reference beyond the agreed tolerance, indicating a numerical
    regression caused by an OV version change or export bug.
    """
    native_policy, ov_model, spec = golden_pair
    golden_l2_threshold = spec.l2_threshold if spec.l2_threshold is not None else golden_l2_threshold
    ov_ver = importlib.metadata.version("openvino") if importlib.util.find_spec("openvino") else "?"

    log.info(
        "golden_action | model=%s | openvino==%s | samples=%d | l2_threshold=%.4f",
        spec.short_id,
        ov_ver,
        len(dataset_observations),
        golden_l2_threshold,
    )

    l2_diffs: list[float] = []

    for i, obs in enumerate(dataset_observations):
        seed = 42 + i

        native_action = _predict_native(native_policy, obs, seed=seed)
        ov_action = _predict_exported(ov_model, obs, seed=seed)

        # Align lengths — OV may return a different chunk_size slice.
        min_len = min(len(native_action), len(ov_action))
        native_action = native_action[:min_len]
        ov_action = ov_action[:min_len]

        l2 = float(np.linalg.norm(native_action - ov_action))
        l2_diffs.append(l2)
        log.info(
            "golden_action | model=%s | sample=%d | L2=%.6f | threshold=%.4f | %s",
            spec.short_id,
            i,
            l2,
            golden_l2_threshold,
            "PASS" if l2 <= golden_l2_threshold else "FAIL",
        )

    max_l2 = max(l2_diffs)
    mean_l2 = float(np.mean(l2_diffs))
    log.info(
        "golden_action | model=%s | max_L2=%.6f | mean_L2=%.6f | threshold=%.4f",
        spec.short_id,
        max_l2,
        mean_l2,
        golden_l2_threshold,
    )

    failures = [i for i, l2 in enumerate(l2_diffs) if l2 > golden_l2_threshold]
    assert not failures, (
        f"[{spec.short_id}] {len(failures)}/{len(l2_diffs)} sample(s) exceeded "
        f"L2 threshold {golden_l2_threshold:.4f} (openvino=={ov_ver}).\n"
        f"  Failing samples: {failures}\n"
        f"  L2 values:       {[f'{l2:.6f}' for l2 in l2_diffs]}\n"
        f"  Max L2:          {max_l2:.6f}\n"
        "This indicates a numerical regression in the OV export or a version mismatch."
    )
