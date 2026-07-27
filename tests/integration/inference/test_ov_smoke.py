# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""OpenVINO smoke tests — ov_smoke marker.

Validates the three-stage contract "OpenVINO works with physicalai" against the
three published model exports in the OpenVINO/physical-ai HuggingFace collection:

    Stage 1 — load:      OpenVINOAdapter.load() — read_model + compile_model succeed
    Stage 2 — tokenizer: OVTokenizer — openvino_tokenizers extension loads and is
                         compatible with the installed core OpenVINO version
    Stage 3 — predict:   InferenceModel.predict_action_chunk() — outputs are finite

Models tested:
    OpenVINO/act-fp16-ov            → stages 1, 3            (no tokenizer)
    OpenVINO/pi05-libero-fp16-ov    → stages 1, 2, 3         (tokenizer-bearing, VLA)
    OpenVINO/smolvla-libero-fp16-ov → stages 1, 2, 3         (tokenizer-bearing, VLA)

The tokenizer stage is critical: openvino and openvino_tokenizers are versioned
independently. A mismatch silently passes for ACT (no tokenizer) but breaks at the
tokenizer stage for pi05 / smolvla — exactly the failure mode that caused the
openvino pin in pyproject.toml.

Run:
    pytest -m ov_smoke -s                       # all models, all stages
    pytest -m ov_smoke -k pi05                  # pi05 only
    pytest -m ov_smoke -k "load or tokenizer"   # stages 1 and 2 only

Environment:
    OV_SMOKE_CACHE_DIR   optional path for the HuggingFace model cache
                         (avoids re-downloading on repeated runs)
"""

from __future__ import annotations

import importlib.metadata
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from physicalai.inference.constants import TASK, TOKENIZED_PROMPT, TOKENIZED_PROMPT_MASK
from physicalai.inference.manifest import Manifest
from physicalai.inference.model import InferenceModel
from physicalai.inference.preprocessors import OVTokenizer
from physicalai.inference.utils._hub import download_from_hub

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)

# All tests in this module carry both markers.
pytestmark = [pytest.mark.ov_smoke, pytest.mark.integration]

# Skip the entire module when OpenVINO is not installed — these tests require
# the real runtime, not a mock.
pytest.importorskip("openvino", reason="openvino not installed — skipping ov_smoke suite")

# ---------------------------------------------------------------------------
# Model specifications
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(42)

_LIBERO_IMAGES = {
    "images.image": _RNG.random((1, 3, 256, 256)).astype(np.float32),
    "images.image2": _RNG.random((1, 3, 256, 256)).astype(np.float32),
}
_LIBERO_STATE = {"state": np.zeros((1, 8), dtype=np.float32)}
_LIBERO_TASK = {TASK: ["pick up the red block and place it in the bowl"]}


@dataclass(frozen=True)
class _ModelSpec:
    """Specification for one smoke-tested model export."""

    repo_id: str
    has_tokenizer: bool
    observation: dict[str, Any]

    @property
    def short_id(self) -> str:
        """Return the repo slug (last path component) for test IDs and logging."""
        return self.repo_id.split("/")[-1]


_ALL_MODELS: list[_ModelSpec] = [
    _ModelSpec(
        repo_id="OpenVINO/act-fp16-ov",
        has_tokenizer=False,
        observation={**_LIBERO_IMAGES, **_LIBERO_STATE},
    ),
    _ModelSpec(
        repo_id="OpenVINO/pi05-libero-fp16-ov",
        has_tokenizer=True,
        observation={**_LIBERO_IMAGES, **_LIBERO_STATE, **_LIBERO_TASK},
    ),
    _ModelSpec(
        repo_id="OpenVINO/smolvla-libero-fp16-ov",
        has_tokenizer=True,
        observation={**_LIBERO_IMAGES, **_LIBERO_STATE, **_LIBERO_TASK},
    ),
]

_TOKENIZER_MODELS: list[_ModelSpec] = [m for m in _ALL_MODELS if m.has_tokenizer]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pkg_version(pkg: str) -> str:
    """Return the installed version of *pkg*, or ``'not installed'``."""
    try:
        return importlib.metadata.version(pkg)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def _find_tokenizer_artifact(export_dir: Path) -> Path | None:
    """Return the path to the OVTokenizer artifact in *export_dir*.

    Uses os.path.normpath (not Path.resolve) deliberately — same reasoning as
    component_factory._resolve_artifact_path: HuggingFace Hub snapshot dirs
    contain symlinks that point into a sibling blobs/ store.  Resolving the
    symlink would give the raw blob path which has no `.bin` sibling, causing
    OpenVINO's read_model to fail with "Empty weights data".  normpath keeps
    the path inside the snapshot dir where the `.xml` and `.bin` symlinks
    coexist.
    """
    manifest = Manifest.load(export_dir)
    for spec in manifest.model.preprocessors:
        is_ov_tokenizer = spec.type == "ov_tokenizer" or "OVTokenizer" in spec.class_path
        if not is_ov_tokenizer:
            continue
        artifact = spec.flat_params.get("artifact") or spec.init_args.get("artifact")
        if artifact:
            return Path(os.path.normpath(export_dir / artifact))
    return None


# ---------------------------------------------------------------------------
# Session fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def smoke_cache_dir() -> Path | None:
    """Optional HuggingFace cache directory from the environment."""
    env = os.environ.get("OV_SMOKE_CACHE_DIR")
    return Path(env) if env else None


@pytest.fixture(scope="session", autouse=True)
def _report_ov_versions() -> None:
    """Log the installed openvino / openvino-tokenizers version pair once per session."""
    ov_ver = _pkg_version("openvino")
    tok_ver = _pkg_version("openvino-tokenizers")
    log.info("ov_smoke | openvino==%s | openvino-tokenizers==%s", ov_ver, tok_ver)


@pytest.fixture(
    scope="session",
    params=_ALL_MODELS,
    ids=lambda m: m.short_id,
)
def all_exports(
    request: pytest.FixtureRequest,
    smoke_cache_dir: Path | None,
) -> tuple[Path, _ModelSpec]:
    """Download (or reuse cached) export dir for each model in *_ALL_MODELS*."""
    spec: _ModelSpec = request.param
    export_dir = download_from_hub(spec.repo_id, cache_dir=smoke_cache_dir)
    return export_dir, spec


@pytest.fixture(
    scope="session",
    params=_TOKENIZER_MODELS,
    ids=lambda m: m.short_id,
)
def tokenizer_exports(
    request: pytest.FixtureRequest,
    smoke_cache_dir: Path | None,
) -> tuple[Path, _ModelSpec]:
    """Download (or reuse cached) export dir for tokenizer-bearing models."""
    spec: _ModelSpec = request.param
    export_dir = download_from_hub(spec.repo_id, cache_dir=smoke_cache_dir)
    return export_dir, spec


# ---------------------------------------------------------------------------
# Stage 1 — load
# ---------------------------------------------------------------------------


def test_load(all_exports: tuple[Path, _ModelSpec]) -> None:
    """Stage 1: OpenVINOAdapter.load — read_model + compile_model succeed.

    Loads the policy XML via the OpenVINO runtime. Preprocessors and
    postprocessors are suppressed so this stage only exercises the adapter.
    A failure here means the core OpenVINO runtime cannot read or compile the
    exported model IR.
    """
    export_dir, spec = all_exports
    log.info("ov_smoke | stage=load | model=%s | openvino==%s", spec.short_id, _pkg_version("openvino"))

    # Bypass preprocessors/postprocessors — isolates adapter.load() from the tokenizer stage.
    model = InferenceModel(export_dir=export_dir, device="CPU", preprocessors=[], postprocessors=[])

    assert model.adapter.compiled_model is not None, (
        f"[{spec.short_id}] compiled_model is None after load — "
        "read_model or compile_model failed silently"
    )
    assert model.adapter.input_names, f"[{spec.short_id}] adapter reports no input names after load"
    assert model.adapter.output_names, f"[{spec.short_id}] adapter reports no output names after load"


# ---------------------------------------------------------------------------
# Stage 2 — tokenizer
# ---------------------------------------------------------------------------


def test_tokenizer(tokenizer_exports: tuple[Path, _ModelSpec]) -> None:
    """Stage 2: OVTokenizer — openvino_tokenizers extension loads and matches core OpenVINO.

    Instantiates OVTokenizer with the artifact declared in the manifest and
    runs a single tokenisation pass. An incompatible openvino / openvino_tokenizers
    version pair raises an exception during OVTokenizer.__init__ (the import
    registers the runtime extension); this stage catches exactly the failure mode
    that caused the openvino pin.
    """
    export_dir, spec = tokenizer_exports
    ov_ver = _pkg_version("openvino")
    tok_ver = _pkg_version("openvino-tokenizers")
    log.info(
        "ov_smoke | stage=tokenizer | model=%s | openvino==%s | openvino-tokenizers==%s",
        spec.short_id,
        ov_ver,
        tok_ver,
    )

    tokenizer_path = _find_tokenizer_artifact(export_dir)
    assert tokenizer_path is not None, (
        f"[{spec.short_id}] No ov_tokenizer artifact found in manifest — "
        "model is declared as tokenizer-bearing but manifest.json has no "
        "'ov_tokenizer' preprocessor entry"
    )
    assert tokenizer_path.exists(), (
        f"[{spec.short_id}] Tokenizer artifact declared in manifest but not on disk: {tokenizer_path}"
    )

    # Constructing OVTokenizer imports openvino_tokenizers, registering the OV runtime
    # extension. A core/tokenizer version mismatch raises here.
    tokenizer = OVTokenizer(tokenizer_path)

    # Run a single tokenisation to confirm the loaded model produces valid outputs.
    outputs = tokenizer({TASK: ["pick up the red block and place it in the bowl"]})

    assert TOKENIZED_PROMPT in outputs, (
        f"[{spec.short_id}] OVTokenizer output is missing the '{TOKENIZED_PROMPT}' key"
    )
    assert TOKENIZED_PROMPT_MASK in outputs, (
        f"[{spec.short_id}] OVTokenizer output is missing the '{TOKENIZED_PROMPT_MASK}' key"
    )

    ids = outputs[TOKENIZED_PROMPT]
    assert ids.ndim == 2, (  # noqa: PLR2004
        f"[{spec.short_id}] Expected 2-D token ids (batch, seq_len), got shape {ids.shape}"
    )
    assert ids.shape[0] == 1, (
        f"[{spec.short_id}] Expected batch size 1, got {ids.shape[0]}"
    )
    assert ids.shape[1] > 0, f"[{spec.short_id}] Token sequence length must be > 0"


# ---------------------------------------------------------------------------
# Stage 3 — predict
# ---------------------------------------------------------------------------


def test_predict(all_exports: tuple[Path, _ModelSpec]) -> None:
    """Stage 3: OpenVINOAdapter.predict — inference runs and outputs are finite.

    Loads the full pipeline (including tokenizer for VLA models) and calls
    predict_action_chunk. Validates that the returned action chunk is a finite
    numpy array with a non-zero action dimension. A failure here means inference
    either crashes or produces NaN / Inf — both indicate a broken OV version.
    """
    export_dir, spec = all_exports
    log.info(
        "ov_smoke | stage=predict | model=%s | openvino==%s",
        spec.short_id,
        _pkg_version("openvino"),
    )

    model = InferenceModel(export_dir=export_dir, device="CPU")
    model.reset()
    chunk = model.predict_action_chunk(spec.observation)

    assert isinstance(chunk, np.ndarray), (
        f"[{spec.short_id}] predict_action_chunk returned {type(chunk).__name__}, expected np.ndarray"
    )
    assert chunk.ndim >= 1, f"[{spec.short_id}] Action chunk must have at least one dimension"
    assert chunk.shape[-1] > 0, f"[{spec.short_id}] Action dimension must be > 0"
    assert np.isfinite(chunk).all(), (
        f"[{spec.short_id}] predict_action_chunk returned non-finite values (NaN or Inf) — "
        f"non-finite count: {(~np.isfinite(chunk)).sum()}"
    )
