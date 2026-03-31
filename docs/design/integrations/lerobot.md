# PhysicalAI: LeRobot Integration Design

**Status**: Proposal
**Author**: Samet Akcay
**Date**: 2026-03-31
**Relates to**: [Inference Core Design](../components/inferencekit.md)

---

## Executive Summary

This document describes how **PhysicalAI** integrates with **LeRobot** exported models using a **single converged manifest format**. Both frameworks produce `manifest.json` files with the same schema, eliminating the need for format adapters or translation layers.

**Key principles:**

1. **One schema, two expressiveness levels** --- The manifest schema supports two component formats: `type` + flat params (interoperable, used by LeRobot) and `class_path` + `init_args` (full-power, used by PhysicalAI). PhysicalAI reads both; LeRobot reads `type` only.
2. **LeRobot is standalone** --- LeRobot's export system works perfectly without PhysicalAI installed. No PhysicalAI imports, no PhysicalAI class paths in manifests.
3. **PhysicalAI loads LeRobot exports natively** --- `InferenceModel.load("./lerobot_export")` works out of the box. No adapter class, no special-casing.
4. **Dependency is strictly one-way** --- LeRobot does not depend on PhysicalAI. PhysicalAI reads LeRobot's output (pure JSON) without importing LeRobot.

```text
LeRobot (standalone)                    PhysicalAI
--------------------                    ----------
policy.export("./out") --produces-->    InferenceModel.load("./out")
                                            |
  Same manifest.json schema                 +-- reads manifest.json
  Writes: type + flat params                +-- resolves via type OR class_path
  Own runners (numpy-only)                  +-- builds preprocessors/postprocessors from io
  Zero physicalai deps                      +-- runs inference through pipeline
```

### Dual Component Resolution

The manifest supports two ways to specify components (runners, preprocessors, postprocessors):

| Format                         | Who writes                         | Who reads            | Example                                                               |
| ------------------------------ | ---------------------------------- | -------------------- | --------------------------------------------------------------------- |
| **`type` + flat params**       | LeRobot, simple PhysicalAI exports | Both (interoperable) | `{"type": "action_chunking", "chunk_size": 100}`                      |
| **`class_path` + `init_args`** | PhysicalAI (full-power)            | PhysicalAI only      | `{"class_path": "physicalai.inference.runners.ActionChunkingRunner", "init_args": {"chunk_size": 100}}` |

PhysicalAI resolves both through the same `ComponentRegistry` + `instantiate_component()` pipeline. LeRobot only reads `type` and maps to its own implementations. See [Runner Resolution](#runner-resolution) for the resolution algorithm.

---

## Table of Contents

- [PhysicalAI: LeRobot Integration Design](#physicalai-lerobot-integration-design)
  - [Executive Summary](#executive-summary)
    - [Dual Component Resolution](#dual-component-resolution)
  - [Table of Contents](#table-of-contents)
  - [1. Architecture Overview](#1-architecture-overview)
  - [2. Converged Manifest Format](#2-converged-manifest-format)
    - [Schema Overview](#schema-overview)
    - [Full Example: ACT Policy](#full-example-act-policy)
    - [Runner Variants](#runner-variants)
    - [Field Reference](#field-reference)
      - [Top-Level Envelope](#top-level-envelope)
      - [`policy` --- Identity](#policy-----identity)
      - [`model` --- How to Run](#model-----how-to-run)
      - [`hardware` --- Deployment](#hardware-----deployment)
      - [`metadata` --- Provenance](#metadata-----provenance)
      - [Preprocessor / Postprocessor Entry](#preprocessor--postprocessor-entry)
    - [Design Decisions](#design-decisions)
  - [3. How PhysicalAI Loads the Manifest](#3-how-physicalai-loads-the-manifest)
    - [Loading Flow](#loading-flow)
    - [Runner Resolution](#runner-resolution)
    - [Preprocessor and Postprocessor Construction](#preprocessor-and-postprocessor-construction)
  - [4. How LeRobot Uses the Manifest](#4-how-lerobot-uses-the-manifest)
  - [5. Runner Mapping](#5-runner-mapping)
    - [`model.runner.type` to Runner](#modelrunnertype-to-runner)
    - [Runner Parameters (All in `model.runner`)](#runner-parameters-all-in-modelrunner)
  - [6. Normalization Handling](#6-normalization-handling)
    - [Problem](#problem)
    - [Solution: Preprocessor and Postprocessor Entries](#solution-preprocessor-and-postprocessor-entries)
    - [PhysicalAI Implementation](#physicalai-implementation)
    - [Normalization Modes](#normalization-modes)
    - [Stats File Format](#stats-file-format)
  - [7. Usage Examples](#7-usage-examples)
    - [Basic Usage](#basic-usage)
    - [With Callbacks](#with-callbacks)
    - [Override Runner Parameters](#override-runner-parameters)
    - [Real-Time Control](#real-time-control)
  - [8. Supported Policies](#8-supported-policies)
  - [9. Testing Strategy](#9-testing-strategy)
    - [Conformance Tests](#conformance-tests)
    - [Parity Tests](#parity-tests)
    - [Backward Compatibility Tests](#backward-compatibility-tests)
  - [10. Migration from Legacy Formats](#10-migration-from-legacy-formats)
    - [Migration Path](#migration-path)
    - [Schema Enforcement](#schema-enforcement)
  - [Appendix A: Design Rationale](#appendix-a-design-rationale)
    - [Why One Format Instead of Two?](#why-one-format-instead-of-two)
    - [Why `model` as a Container?](#why-model-as-a-container)
    - [Why Not `model: null` for Single-Pass?](#why-not-model-null-for-single-pass)
    - [Why Preprocessors Inside `io`?](#why-preprocessors-inside-io)
    - [Why `policy.source.class_path`?](#why-policysourceclass_path)
  - [Appendix B: Comparison with Previous Design](#appendix-b-comparison-with-previous-design)
  - [Related Documents](#related-documents)

---

## 1. Architecture Overview

The integration is seamless because both frameworks share the same manifest schema. PhysicalAI's `InferenceModel` reads the manifest, resolves components (runner, preprocessors, postprocessors, adapter), and runs inference --- regardless of which framework produced the export.

```text
+-----------------------------------------------------------------------+
|                           PhysicalAI                                  |
|                                                                       |
|  +----------------+  +-----------------+  +------------------------+  |
|  |    Adapters    |  |    Built-in     |  |      Callbacks         |  |
|  |   (backends)   |  |    Runners      |  |   (instrumentation)    |  |
|  |                |  |                 |  |                        |  |
|  | ONNX, OpenVINO |  | SinglePass      |  | TimingCallback         |  |
|  | TensorRT       |  | ActionChunking  |  | LoggingCallback        |  |
|  | TorchExportIR  |  | Iterative       |  | ActionSafetyCallback   |  |
|  |                |  | TwoPhase        |  |                        |  |
|  +----------------+  +-----------------+  +------------------------+  |
|                                                                       |
|  +---------------------------------------------------------------+    |
|  |                   Manifest Loader                              |   |
|  |                                                                |   |
|  |  manifest.json  -->  parse  -->  resolve components  -->  run  |   |
|  |  (same schema for all sources: PhysicalAI, LeRobot, custom)   |   |
|  +---------------------------------------------------------------+    |
+-----------------------------------------------------------------------+
                              |
                              | reads (pure JSON file I/O)
                              v
               +----------------------------+
               |     Exported Package       |
               |     (any source)           |
               |                            |
               |   manifest.json            |
               |   model.onnx              |
               |   stats.safetensors       |
               +----------------------------+
```

**What PhysicalAI adds over LeRobot's standalone runtime:**

| Feature                             | LeRobot Standalone | PhysicalAI             |
| ----------------------------------- | ------------------ | ---------------------- |
| Load exported policy                | Yes                | Yes                    |
| Single-pass inference               | Yes                | Yes                    |
| Iterative inference                 | Yes                | Yes                    |
| Two-phase inference                 | Yes                | Yes                    |
| Action chunking                     | Yes                | Yes                    |
| Callbacks (timing, logging, safety) | No                 | Yes                    |
| Multi-backend with fallback         | ONNX + OpenVINO    | ONNX + OpenVINO + TRT  |
| Preprocessor/postprocessor chains   | Fixed pipeline     | Extensible chain       |
| HuggingFace Hub loading             | No                 | Yes (`hf://user/repo`) |
| `select_action()` / `reset()` API   | No                 | Yes                    |

---

## 2. Converged Manifest Format

### Schema Overview

The manifest mirrors PhysicalAI's `InferenceModel` class hierarchy, following the same philosophy as training configs (which split into `model`, `data`, `trainer` sections):

```text
manifest.json
+-- format + version        (envelope --- what is this file?)
+-- policy                  (identity --- what policy is this?)
|   +-- name                (human-readable name)
|   +-- source              (provenance: repo_id, class_path)
+-- model                   (exported model --- how to run it?)
|   +-- n_obs_steps         (observation window size)
|   +-- runner              (execution pattern + parameters)
|   +-- artifacts           (model files by named role)
|   +-- io                  (I/O contract: shapes, preprocessors, postprocessors)
+-- hardware                (deployment --- what hardware?)
|   +-- robots              (robot configurations)
|   +-- cameras             (camera configurations)
+-- metadata                (provenance --- when/who created this?)
```

### Full Example: ACT Policy

```json
{
  "format": "policy_package",
  "version": "1.0",
  "policy": {
    "name": "act",
    "source": {
      "repo_id": "lerobot/act_aloha_sim_transfer_cube_human",
      "class_path": "physicalai.policies.act.policy.ACT"
    }
  },
  "model": {
    "n_obs_steps": 1,
    "runner": {
      "type": "action_chunking",
      "chunk_size": 100,
      "n_action_steps": 100
    },
    "artifacts": {
      "model": "model.onnx"
    },
    "io": {
      "inputs": [
        {"name": "observation.image", "dtype": "float32", "shape": ["B", 3, 96, 96]},
        {"name": "observation.state", "dtype": "float32", "shape": ["B", 14]}
      ],
      "outputs": [
        {"name": "action", "dtype": "float32", "shape": ["B", 100, 14]}
      ],
      "preprocessors": [
        {
          "type": "normalize",
          "mode": "mean_std",
          "artifact": "stats.safetensors",
          "features": ["observation.state"]
        }
      ],
      "postprocessors": [
        {
          "type": "denormalize",
          "mode": "mean_std",
          "artifact": "stats.safetensors",
          "features": ["action"]
        }
      ]
    }
  },
  "hardware": {
    "robots": [],
    "cameras": []
  },
  "metadata": {
    "created_at": "2026-03-27T12:00:00Z",
    "created_by": "lerobot.export"
  }
}
```

> **Note on image inputs:** Image normalization (uint8 to float32, divide by 255) is baked into the ONNX graph during export. Only non-image features that use dataset-level statistics (e.g., `observation.state`) need explicit preprocessor entries.

### Runner Variants

The `model.runner` section is open-ended --- policy-specific parameters go directly in the runner object alongside `type`. This avoids the need for a rigid union schema. Each runner implementation declares its expected parameters and logs a warning for any unrecognized keys (see [Runner Parameter Validation](#runner-parameter-validation)).

**ACT / VQBeT** (single-pass with action chunking):

```json
"runner": {
  "type": "action_chunking",
  "chunk_size": 100,
  "n_action_steps": 100
}
```

**Diffusion Policy** (iterative denoising):

```json
"runner": {
  "type": "iterative",
  "horizon": 16,
  "n_action_steps": 8,
  "num_inference_steps": 100,
  "scheduler": "ddpm"
}
```

**PI0** (two-phase: encode once + denoise iteratively):

For two-phase policies, all model artifacts are listed in `model.artifacts` with named roles (`encoder`, `denoise`) rather than backend names. The runner references these roles:

```json
"artifacts": {
  "encoder": "encoder.onnx",
  "denoise": "denoise.onnx"
},
"runner": {
  "type": "two_phase",
  "chunk_size": 50,
  "n_action_steps": 50,
  "num_inference_steps": 10,
  "scheduler": "euler"
}
```

**SmolVLA** (two-phase, no explicit scheduler):

```json
"artifacts": {
  "encoder": "encoder.onnx",
  "denoise": "denoise.onnx"
},
"runner": {
  "type": "two_phase",
  "chunk_size": 50,
  "n_action_steps": 50,
  "num_inference_steps": 10
}
```

**TDMPC** (iterative with model-predictive control):

```json
"runner": {
  "type": "iterative",
  "horizon": 5,
  "n_action_steps": 1,
  "use_mpc": true,
  "cem_iterations": 6
}
```

### Field Reference

#### Top-Level Envelope

| Field     | Type   | Required | Description                                      |
| --------- | ------ | -------- | ------------------------------------------------ |
| `format`  | string | Yes      | Always `"policy_package"`. Schema identification |
| `version` | string | Yes      | Schema version (semver). Currently `"1.0"`       |

#### `policy` --- Identity

| Field                      | Type   | Required | Description                                         |
| -------------------------- | ------ | -------- | --------------------------------------------------- |
| `policy.name`              | string | Yes      | Human-readable policy name (e.g., `"act"`, `"pi0"`) |
| `policy.source`            | object | No       | Provenance information                              |
| `policy.source.repo_id`    | string | No       | HuggingFace repo ID                                 |
| `policy.source.class_path` | string | No       | Original Python class path                          |

#### `model` --- How to Run

| Field                         | Type   | Required | Description                                                                                                                                    |
| ----------------------------- | ------ | -------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| `model.n_obs_steps`       | int    | Yes      | Number of observation timesteps needed by the model (see [Design Decisions](#design-decisions))                               |
| `model.runner`            | object | Yes      | Runner configuration (see variants)                                                                                                            |
| `model.runner.type`       | string | Yes      | Runner type: `action_chunking`, `iterative`, `two_phase`                                                                                       |
| `model.artifacts`         | object | Yes      | Map of artifact role to filename. Single-model: `{"model": "model.onnx"}`. Two-phase: `{"encoder": "encoder.onnx", "denoise": "denoise.onnx"}` |
| `model.io`                | object | Yes      | I/O specification                                                                                                                              |
| `model.io.inputs`         | array  | Yes      | Input tensor specifications                                                                                                                    |
| `model.io.outputs`        | array  | Yes      | Output tensor specifications                                                                                                                   |
| `model.io.preprocessors`  | array  | No       | Input transforms (normalize, etc.)                                                                                                             |
| `model.io.postprocessors` | array  | No       | Output transforms (denormalize, etc.)                                                                                                          |

#### `hardware` --- Deployment

| Field              | Type  | Required | Description                      |
| ------------------ | ----- | -------- | -------------------------------- |
| `hardware.robots`  | array | No       | Robot configurations (optional)  |
| `hardware.cameras` | array | No       | Camera configurations (optional) |

#### `metadata` --- Provenance

| Field                 | Type   | Required | Description        |
| --------------------- | ------ | -------- | ------------------ |
| `metadata.created_at` | string | No       | ISO 8601 timestamp |
| `metadata.created_by` | string | No       | Creator identifier |

#### Preprocessor / Postprocessor Entry

| Field        | Type   | Required | Description                                                                                                                                          |
| ------------ | ------ | -------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| `type`       | string | Yes      | Processor type: `"normalize"`, `"denormalize"`, or custom                                                                                            |
| `class_path` | string | No       | Python class path for custom processor types. Built-in types (`normalize`, `denormalize`) resolve by convention; unknown types require `class_path`. |
| `mode`       | string | No       | Normalization mode: `"mean_std"`, `"min_max"`, `"identity"` (required for `normalize`/`denormalize`)                                                 |
| `artifact`   | string | No       | Path to stats file (e.g., `"stats.safetensors"`) (required for `normalize`/`denormalize`)                                                            |
| `features`   | array  | No       | Feature names to process (e.g., `["observation.state"]`) (required for `normalize`/`denormalize`)                                                    |

Built-in types resolve by convention: `"normalize"` maps to `StatsNormalizer`, `"denormalize"` maps to `StatsDenormalizer`. For custom processor types, provide a `class_path`:

```json
{
  "type": "clamp",
  "class_path": "physicalai.inference.postprocessors.ActionClamp",
  "min": -1.0,
  "max": 1.0
}
```

### Design Decisions

| Decision                                   | Rationale                                                                                                                                                                                                                                                                                                                            |
| ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **No `policy.type` field**                 | `model.runner.type` is the sole source of truth for runner construction. A separate `policy.type` would overlap without matching (e.g., ACT is `single_pass` but uses `action_chunking` runner), causing confusion. Eliminated to avoid ambiguity.                                                                               |
| **`hardware` is top-level**                | Deployment needs hardware information even if LeRobot doesn't use it yet. Cheap future-proofing.                                                                                                                                                                                                                                     |
| **Preprocessors inside `model.io`**    | They are I/O transforms, logically part of the I/O boundary. Not separate top-level sections.                                                                                                                                                                                                                                        |
| **`format` + `version` kept**              | Cheap future-proofing for schema evolution. `format` enables detection, `version` enables migration.                                                                                                                                                                                                                                 |
| **`model` is a container**             | Mirrors `InferenceModel` class hierarchy. Matches training config philosophy (`model`/`data`/`trainer`).                                                                                                                                                                                                                             |
| **No separate `action` section**           | `chunk_size` and `n_action_steps` are runner behavioral params. `action_dim` is redundant with output shape.                                                                                                                                                                                                                         |
| **`mode` per processor**                   | Different features may use different normalization modes (mean_std vs min_max).                                                                                                                                                                                                                                                      |
| **Runner params are open with validation** | Policy-specific fields go directly in runner. Each runner declares expected params and warns on unknown keys (see [Runner Parameter Validation](#runner-parameter-validation)).                                                                                                                                                      |
| **Named artifact roles**                   | `model.artifacts` uses role names (`model`, `encoder`, `denoise`) not backend names. This eliminates the split between `model.artifacts` and runner-level artifact refs for two-phase policies, giving a single authoritative location for all model files.                                                                  |
| **Extensible processor types**             | Built-in types (`normalize`, `denormalize`) resolve by convention. Unknown types fall back to `class_path`, allowing custom processors without code changes to the loader.                                                                                                                                                           |
| **`n_obs_steps` in `model`**           | Although `n_obs_steps` describes what the model expects (a contract), it is consumed during inference preparation --- the caller uses it to decide how many observation frames to collect before calling the model. It sits alongside other inference-time parameters rather than in `policy` (which is purely identity/provenance). |
| **Shared JSON Schema for CI**              | Both projects validate exported manifests against a shared `manifest.schema.json` to prevent schema drift (see [Schema Enforcement](#schema-enforcement)).                                                                                                                                                                           |
| **Dual component resolution**              | Components support both `type` + flat params (interoperable, LeRobot-compatible) and `class_path` + `init_args` (PhysicalAI full-power). Both resolve through the same `ComponentRegistry` + `instantiate_component()` pipeline. This avoids per-type if-chains while preserving the existing component system. See [Dual Component Resolution](#dual-component-resolution). |
| **No `_normalize_metadata()`**             | The nested manifest structure is parsed directly into Pydantic models. No flattening shim needed --- both the schema and the loader are designed together. Legacy `metadata.yaml` files (pre-manifest era) are handled by `from_legacy_metadata()` only. |

---

## 3. How PhysicalAI Loads the Manifest

### Loading Flow

The manifest is parsed directly into nested Pydantic models --- no intermediate flattening or normalization step. The nested JSON maps 1:1 to the Pydantic model hierarchy:

```python
# In InferenceModel.load():
raw = json.loads((path / "manifest.json").read_text())

# Validate format
if raw.get("format") != "policy_package":
    msg = f"Unknown manifest format: {raw.get('format')}"
    raise ValueError(msg)

# Parse directly into nested Pydantic models
manifest = Manifest.model_validate(raw)

# Resolve components from typed manifest fields
runner = resolve_runner(manifest.model.runner)
adapter = create_adapter(manifest.model.artifacts, path)
preprocessors = resolve_processors(manifest.model.io.preprocessors, path)
postprocessors = resolve_processors(manifest.model.io.postprocessors, path)
```

> **Legacy `metadata.yaml` files** (pre-manifest era, before `manifest.json` existed) are handled separately by `from_legacy_metadata()` in `manifest.py`. This is unrelated to the manifest format --- it handles the old YAML-based metadata from early PhysicalAI exports.

### Runner Resolution

The runner factory uses **dual-path resolution** --- a single if-check, not an if-chain per type. If `class_path` is present, it goes straight to `ComponentSpec` instantiation. Otherwise, `type` is resolved through the same `ComponentRegistry` pipeline:

```python
def resolve_runner(runner_config: dict) -> InferenceRunner:
    """Resolve runner from manifest config using dual-path resolution.

    Path 1: class_path + init_args → ComponentSpec → instantiate_component()
    Path 2: type + flat params → registry lookup → ComponentSpec → instantiate_component()

    Both paths end at the same instantiate_component() call.
    """
    if "class_path" in runner_config:
        # PhysicalAI-native: full ComponentSpec path
        spec = ComponentSpec.model_validate(runner_config)
        return instantiate_component(spec)

    # Framework-agnostic: type → registry resolves short name to class_path
    runner_type = runner_config["type"]
    init_args = {k: v for k, v in runner_config.items() if k != "type"}
    spec = ComponentSpec(class_path=runner_type, init_args=init_args)
    return instantiate_component(spec)
```

**How `instantiate_component()` handles the two paths:**

- **`class_path`** (full Python path, e.g., `"physicalai.inference.runners.ActionChunkingRunner"`) → direct import
- **`type`** (short name, e.g., `"action_chunking"`) → `ComponentRegistry.resolve()` maps to full path → import

`class_path` always uses the full Python class path for explicit, unambiguous resolution. `type` uses the registry as the single source of truth for mapping short names to classes.

**Example: How the same runner loads from both formats:**

```json
// LeRobot writes (type + flat params):
{"type": "action_chunking", "chunk_size": 100, "n_action_steps": 100}

// PhysicalAI writes (class_path + init_args):
{"class_path": "physicalai.inference.runners.ActionChunkingRunner", "init_args": {"chunk_size": 100, "n_action_steps": 100}}

// Both resolve to the same ActionChunkingRunner:
// type path:       "action_chunking" → ComponentRegistry.resolve() → ActionChunkingRunner(...)
// class_path path: "physicalai.inference.runners.ActionChunkingRunner" → direct import → ActionChunkingRunner(...)
```

### Runner Parameter Validation

Each runner declares the parameters it consumes. Unknown keys trigger a warning, catching typos without breaking forward compatibility:

```python
class IterativeRunner(InferenceRunner):
    EXPECTED_PARAMS = {"type", "num_inference_steps", "scheduler", "horizon", "n_action_steps"}

    @classmethod
    def from_config(cls, config: dict) -> "IterativeRunner":
        unknown = set(config.keys()) - cls.EXPECTED_PARAMS
        if unknown:
            logger.warning("IterativeRunner: ignoring unknown params: %s", unknown)
        return cls(
            num_steps=config.get("num_inference_steps", 10),
            scheduler=config.get("scheduler", "euler"),
        )
```

### Preprocessor and Postprocessor Construction

Processors use the same dual-path resolution as runners. The `resolve_processor()` function handles both `class_path` + `init_args` (PhysicalAI-native) and `type` + flat params (interoperable):

```python
def resolve_processors(specs: list[dict], path: Path) -> list:
    """Build processor chain from manifest specs using dual-path resolution.

    Each spec is resolved identically to runners:
    - class_path present → ComponentSpec → instantiate_component()
    - type present → registry lookup → ComponentSpec → instantiate_component()
    """
    processors = []
    for spec in specs:
        if "class_path" in spec:
            # PhysicalAI-native: full ComponentSpec path
            component_spec = ComponentSpec.model_validate(spec)
            processors.append(instantiate_component(component_spec))
        else:
            # Framework-agnostic: type → registry → ComponentSpec
            processor_type = spec["type"]
            init_args = {k: v for k, v in spec.items() if k != "type"}
            # Resolve relative artifact paths to absolute
            if "artifact" in init_args:
                init_args["stats_path"] = path / init_args.pop("artifact")
            component_spec = ComponentSpec(class_path=processor_type, init_args=init_args)
            processors.append(instantiate_component(component_spec))
    return processors
```

**Example: Normalize processor from both formats:**

```json
// LeRobot writes (type + flat params):
{"type": "normalize", "mode": "mean_std", "artifact": "stats.safetensors", "features": ["observation.state"]}

// PhysicalAI writes (class_path + init_args):
{"class_path": "physicalai.inference.preprocessors.StatsNormalizer", "init_args": {"mode": "mean_std", "stats_path": "stats.safetensors", "features": ["observation.state"]}}

// Both resolve to: StatsNormalizer(mode="mean_std", stats_path=..., features=["observation.state"])
```

> **Note:** The `artifact` → `stats_path` key rename happens during resolution for `type`-format specs. In `class_path` format, the key is already `stats_path` (matching the constructor parameter name).

---

## 4. How LeRobot Uses the Manifest

LeRobot reads the same `manifest.json` with its own tooling. It does NOT use pydantic --- it uses `draccus` dataclasses or plain `json.load()`.

```python
# LeRobot's own loading (no physicalai dependency)
import json
from pathlib import Path

def load_exported_policy(path: str | Path) -> ExportedPolicy:
    """Load an exported policy package."""
    path = Path(path)
    raw = json.loads((path / "manifest.json").read_text())

    # Read runner config
    runner_config = raw["model"]["runner"]
    runner_type = runner_config["type"]

    # Build LeRobot's own runner (standalone, numpy-only)
    if runner_type == "action_chunking":
        runner = ActionChunkingWrapper(
            SinglePassRunner(),
            chunk_size=runner_config["chunk_size"],
            n_action_steps=runner_config["n_action_steps"],
        )
    elif runner_type == "iterative":
        runner = IterativeRunner(
            num_steps=runner_config["num_inference_steps"],
            scheduler=runner_config.get("scheduler", "euler"),
        )
    elif runner_type == "two_phase":
        runner = TwoPhaseRunner(...)
    ...

    # Load normalizer from io specs
    preprocessors = raw["model"]["io"].get("preprocessors", [])
    postprocessors = raw["model"]["io"].get("postprocessors", [])
    normalizer = Normalizer.from_specs(preprocessors + postprocessors, path)

    # Load backend adapter (from named artifact role)
    artifacts = raw["model"]["artifacts"]
    adapter = ONNXRuntimeAdapter(path / artifacts["model"])

    return ExportedPolicy(runner=runner, adapter=adapter, normalizer=normalizer)
```

**Key point:** LeRobot's runners, normalizer, and adapters are its own implementations. They have zero overlap with PhysicalAI's implementations. The only shared artifact is the `manifest.json` file on disk.

---

## 5. Runner Mapping

### `model.runner.type` to Runner

| `runner.type`     | PhysicalAI Runner                    | LeRobot Runner                            | Policies         |
| ----------------- | ------------------------------------ | ----------------------------------------- | ---------------- |
| `action_chunking` | `ActionChunkingRunner(SinglePass())` | `ActionChunkingWrapper(SinglePassRunner)` | ACT, VQBeT       |
| `iterative`       | `IterativeRunner(SinglePass())`      | `IterativeRunner`                         | Diffusion, TDMPC |
| `two_phase`       | `TwoPhaseRunner(encoder, Iterative)` | `TwoPhaseRunner`                          | PI0, SmolVLA     |

### Runner Parameters (All in `model.runner`)

| Parameter             | Used By                               | Description                             |
| --------------------- | ------------------------------------- | --------------------------------------- |
| `chunk_size`          | action_chunking                       | Size of predicted action chunk          |
| `n_action_steps`      | action_chunking, two_phase, iterative | Actions to execute per chunk            |
| `num_inference_steps` | iterative, two_phase                  | Number of denoising steps               |
| `scheduler`           | iterative, two_phase                  | Scheduler algorithm (euler, ddpm, ddim) |
| `horizon`             | iterative                             | Planning horizon (Diffusion, TDMPC)     |
| `use_mpc`             | iterative                             | Enable model-predictive control (TDMPC) |
| `cem_iterations`      | iterative                             | CEM optimization iterations (TDMPC)     |

> **Note:** Two-phase artifact paths (`encoder`, `denoise`) live in `model.artifacts`, not in the runner config. See [Runner Variants](#runner-variants) for examples.

---

## 6. Normalization Handling

### Problem

LeRobot policies operate on **normalized** inputs and produce **normalized** outputs. Normalization statistics are saved alongside the model in `stats.safetensors`. At inference time:

1. **Observations must be normalized** before feeding to the model
2. **Actions must be denormalized** after the model produces them

### Solution: Preprocessor and Postprocessor Entries

The manifest declares normalization as I/O transforms in `model.io.preprocessors` and `model.io.postprocessors`:

```json
"io": {
  "inputs": [...],
  "outputs": [...],
  "preprocessors": [
    {
      "type": "normalize",
      "mode": "mean_std",
      "artifact": "stats.safetensors",
      "features": ["observation.state"]
    }
  ],
  "postprocessors": [
    {
      "type": "denormalize",
      "mode": "mean_std",
      "artifact": "stats.safetensors",
      "features": ["action"]
    }
  ]
}
```

### PhysicalAI Implementation

Two pipeline components handle normalization:

**`StatsNormalizer`** (preprocessor):

```python
class StatsNormalizer(Preprocessor):
    """Normalize input features using saved statistics."""

    def __init__(self, stats_path: Path, features: list[str], mode: str = "mean_std"):
        self.stats = load_stats(stats_path)
        self.features = features
        self.mode = mode

    def __call__(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        for feature in self.features:
            if feature in inputs:
                inputs[feature] = self._normalize(inputs[feature], feature)
        return inputs
```

**`StatsDenormalizer`** (postprocessor):

```python
class StatsDenormalizer(Postprocessor):
    """Denormalize output features using saved statistics."""

    def __init__(self, stats_path: Path, features: list[str], mode: str = "mean_std"):
        self.stats = load_stats(stats_path)
        self.features = features
        self.mode = mode

    def __call__(self, outputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        for feature in self.features:
            if feature in outputs:
                outputs[feature] = self._denormalize(outputs[feature], feature)
        return outputs
```

### Normalization Modes

| Mode       | Normalize                         | Denormalize                       |
| ---------- | --------------------------------- | --------------------------------- |
| `mean_std` | `(x - mean) / std`                | `x * std + mean`                  |
| `min_max`  | `(x - min) / (max - min) * 2 - 1` | `(x + 1) / 2 * (max - min) + min` |
| `identity` | passthrough                       | passthrough                       |

### Stats File Format

Normalization statistics are stored in `safetensors` format. Each feature has `{feature}/mean`, `{feature}/std`, `{feature}/min`, `{feature}/max` tensors as needed by the normalization mode.

---

## 7. Usage Examples

### Basic Usage

```python
from physicalai import InferenceModel

# Load LeRobot-exported policy (detected automatically via manifest.json)
model = InferenceModel("./act_exported")

# Run inference
observation = {
    "observation.image": image_array,      # float32, shape (1, 3, 96, 96)
    "observation.state": state_array,      # float32, shape (1, 14)
}
outputs = model(observation)
action = outputs["action"]  # float32, shape (1, 14)
```

### With Callbacks

```python
from physicalai import InferenceModel
from physicalai.inference.callbacks import TimingCallback

model = InferenceModel(
    "./pi0_exported",
    callbacks=[TimingCallback()],
)

outputs = model(observation)
# TimingCallback logs: "Inference: 12.3ms"
```

### Override Runner Parameters

```python
# Override denoising steps at load time (no re-export needed)
model = InferenceModel(
    "./diffusion_exported",
    num_steps=20,         # Override manifest default of 100
    scheduler="ddim",     # Override manifest default of "ddpm"
)
```

### Real-Time Control

```python
from physicalai import InferenceModel

policy = InferenceModel("./act_exported")
policy.reset()

while not done:
    action = policy.select_action(observation)
    observation, reward, done, info = env.step(action)

# Reset between episodes
policy.reset()
```

---

## 8. Supported Policies

All LeRobot policy types are supported through the converged runner system:

| Policy    | `runner.type`   | Runner Stack                                 | Artifact Roles       |
| --------- | --------------- | -------------------------------------------- | -------------------- |
| ACT       | action_chunking | ActionChunking(SinglePass)                   | `model`              |
| VQBeT     | action_chunking | ActionChunking(SinglePass)                   | `model`              |
| Diffusion | iterative       | ActionChunking(Iterative(SinglePass))        | `model`              |
| TDMPC     | iterative       | Iterative(SinglePass) with MPC               | `model`              |
| PI0       | two_phase       | ActionChunking(TwoPhase(encoder, Iterative)) | `encoder`, `denoise` |
| SmolVLA   | two_phase       | ActionChunking(TwoPhase(encoder, Iterative)) | `encoder`, `denoise` |

---

## 9. Testing Strategy

### Conformance Tests

Verify that PhysicalAI correctly loads manifests produced by LeRobot:

```python
class TestConvergedManifestLoading:
    """Verify PhysicalAI loads converged manifest format."""

    def test_detect_policy_package(self, package_path):
        """Detect exported package via format field."""
        manifest = json.loads((package_path / "manifest.json").read_text())
        assert manifest["format"] == "policy_package"

    def test_load_action_chunking(self, act_package):
        """Load ACT policy with action chunking runner."""
        model = InferenceModel(act_package)
        assert isinstance(model.runner, ActionChunkingRunner)

    def test_load_iterative(self, diffusion_package):
        """Load Diffusion policy with iterative runner."""
        model = InferenceModel(diffusion_package)
        assert isinstance(model.runner, IterativeRunner)

    def test_load_two_phase(self, pi0_package):
        """Load PI0 policy with two-phase runner."""
        model = InferenceModel(pi0_package)
        assert isinstance(model.runner, TwoPhaseRunner)

    def test_preprocessors_created(self, act_package):
        """Preprocessors auto-created from io.preprocessors."""
        model = InferenceModel(act_package)
        assert len(model.preprocessors) > 0
        assert isinstance(model.preprocessors[0], StatsNormalizer)

    def test_postprocessors_created(self, act_package):
        """Postprocessors auto-created from io.postprocessors."""
        model = InferenceModel(act_package)
        assert len(model.postprocessors) > 0
        assert isinstance(model.postprocessors[0], StatsDenormalizer)
```

### Parity Tests

Verify PhysicalAI output matches LeRobot's standalone runtime:

```python
def test_parity_with_lerobot_runtime(pi0_package):
    """Output matches LeRobot's own runtime (bit-for-bit with same seed)."""
    # Load with PhysicalAI
    pai_model = InferenceModel(pi0_package)

    # Load with LeRobot standalone
    from lerobot.export import load_exported_policy
    lr_model = load_exported_policy(pi0_package)

    # Compare outputs with same random seed
    obs = generate_test_observation()
    np.random.seed(42)
    pai_output = pai_model(obs)
    np.random.seed(42)
    lr_output = lr_model.predict(obs)

    np.testing.assert_allclose(pai_output["action"], lr_output["action"], rtol=1e-5)
```

### Backward Compatibility Tests

Verify v1.0 (flat) manifests still load:

```python
def test_legacy_flat_manifest(legacy_package):
    """v1.0 flat manifest loads without error."""
    model = InferenceModel(legacy_package)
    assert model.runner is not None
    assert model.adapter is not None
```

---

## 10. Migration from Legacy Formats

This is a **clean cut** to the nested manifest format. There is no `_normalize_metadata()` shim --- the nested Pydantic models are the only manifest representation. Two legacy scenarios are handled:

### Legacy `metadata.yaml` (Pre-Manifest Era)

Early PhysicalAI exports used a flat `metadata.yaml` file instead of `manifest.json`. The existing `from_legacy_metadata()` classmethod on `Manifest` handles this case:

```python
class Manifest(BaseModel):
    @classmethod
    def from_legacy_metadata(cls, metadata: dict) -> "Manifest":
        """Convert old metadata.yaml fields to the nested Manifest structure.

        This handles truly old exports that predate manifest.json entirely.
        """
        ...
```

This is the **only** backward compatibility code needed. It converts old YAML metadata to the new nested `Manifest` model once, at load time.

### Migration Path

| Step | Action | Breaking? | Target Version |
| ---- | ------ | --------- | -------------- |
| 1    | Implement nested `Manifest` Pydantic models (`manifest.py`) | No --- new code | v1.x (current) |
| 2    | Update `mixin_policy.py` to write nested `manifest.json` | No --- new exports use new format | v1.x |
| 3    | Update `model.py` and `factory.py` to use `Manifest` directly | No --- `from_legacy_metadata()` handles old YAML | v1.x |
| 4    | Update LeRobot PR to write same nested format | No --- same schema | v1.x |
| 5    | Add `manifest.schema.json` for CI validation in both repos | No --- additive | v1.x |

> **Key point:** There is no flat-to-nested migration shim. All new manifests are nested from day one. Only pre-manifest `metadata.yaml` files need the legacy path, and that already exists.

### Schema Enforcement

To prevent schema drift between PhysicalAI and LeRobot, a shared `manifest.schema.json` (JSON Schema) is maintained and validated against in CI for both projects:

```text
manifest.schema.json          (shared, vendored into both repos)
    |
    +-- physicalai CI: validate exported manifests against schema
    +-- lerobot CI: validate exported manifests against schema
```

This catches divergence at PR time rather than at runtime. The schema file is the single source of truth for manifest structure.

---

## Appendix A: Design Rationale

### Why One Format Instead of Two?

The previous design (v1 plan) proposed two manifest formats: `lerobot_exported_policy` for LeRobot and `policy_package` for PhysicalAI, bridged by a `LeRobotManifestAdapter` class. This was rejected because:

1. **Unnecessary complexity** --- An adapter class to translate between nearly-identical JSON schemas is pure overhead.
2. **Divergence risk** --- Two formats inevitably drift apart over time, making the adapter increasingly complex.
3. **Testing burden** --- Every feature needs testing against both formats.
4. **User confusion** --- Which format should I use? Does it matter?

The converged format eliminates all of these problems. One schema, two producers, zero translation.

### Why `model` as a Container?

The `model` section mirrors the `InferenceModel` class hierarchy:

- `InferenceModel` composes runner, adapter, preprocessors, postprocessors
- `model` contains runner, artifacts (for adapter), io (for pre/postprocessors)

This follows the same pattern as training configs where the top-level sections (`model`, `data`, `trainer`) mirror the class hierarchy. It makes the manifest self-documenting: the JSON structure tells you the code structure.

### Why Not `model: null` for Single-Pass?

All policies need `n_obs_steps`, `artifacts`, and `io` regardless of runner type. Making `model` nullable would force these universal fields elsewhere (top-level or in `policy`), breaking the logical grouping. Instead, `model` is always present --- only the runner params differ between policy types.

### Why Preprocessors Inside `io`?

Preprocessors and postprocessors are I/O transforms --- they sit at the boundary between raw observations and model inputs. Placing them inside `io` (alongside `inputs` and `outputs`) makes this relationship explicit. The alternative (top-level `preprocessors`/`postprocessors`) separates logically related concepts.

### Why `policy.source.class_path`?

The `class_path` field enables PhysicalAI to instantiate the original policy class when the full PhysicalAI training framework is available. LeRobot ignores this field entirely. It is optional --- packages exported by LeRobot standalone may omit it or use a LeRobot-specific class path.

---

## Appendix B: Comparison with Previous Design

| Aspect             | Previous Design (v1)                               | Current Design (converged)                     |
| ------------------ | -------------------------------------------------- | ---------------------------------------------- |
| Manifest formats   | Two (`lerobot_exported_policy` + `policy_package`) | One (`policy_package`)                         |
| Format adapter     | `LeRobotManifestAdapter` class (~100 lines)        | None --- direct Pydantic parsing               |
| Format detection   | `if format == "lerobot_exported_policy"` branching | Not needed --- single format                   |
| Schema maintenance | Two schemas to keep in sync                        | One shared `manifest.schema.json`              |
| Test matrix        | 2x (each feature tested against both formats)      | 1x                                             |
| Normalization      | Adapter auto-generates ComponentSpecs              | Manifest declares pre/postprocessors directly  |
| Runner resolution  | `policy.kind` + separate `inference` block         | Dual-path: `class_path` OR `type` → registry → `instantiate_component()` |
| Action params      | Separate `action` section                          | Params in `model.runner`                   |
| Backward compat    | Format detection + adapter routing                 | `from_legacy_metadata()` for pre-manifest YAML only |
| Component formats  | `class_path` + `init_args` only                    | Both `type` + flat params (interop) and `class_path` + `init_args` (full-power) |

---

## Related Documents

- **[Inference Core Design](../components/inferencekit.md)** --- Domain-agnostic inference layer
- **[Strategy](../architecture/strategy.md)** --- Big-picture architecture and layering decisions
- **[Architecture](../architecture/architecture.md)** --- PhysicalAI runtime CLI and packaging

---

_Document version: 5.1_
_Last updated: 2026-03-31_
