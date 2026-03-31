# PhysicalAI: LeRobot Integration Design

**Status**: Proposal
**Author**: Samet Akcay
**Date**: 2026-03-31
**Relates to**: [Inference Core Design](../components/inferencekit.md)

---

## Executive Summary

This document describes how **PhysicalAI** integrates with **LeRobot** exported models using a **single converged manifest format**. Both frameworks produce `manifest.json` files with the same schema, eliminating the need for format adapters or translation layers.

**Key principles:**

1. **One schema, two expressiveness levels** --- The manifest supports two component formats: `type` + flat params (interoperable, used by LeRobot) and `class_path` + `init_args` (full-power, used by PhysicalAI). PhysicalAI reads both; LeRobot reads `type` only.
2. **LeRobot is standalone** --- LeRobot's export system works without PhysicalAI installed. No PhysicalAI imports, no PhysicalAI class paths in manifests.
3. **PhysicalAI loads LeRobot exports natively** --- `InferenceModel.load("./lerobot_export")` works out of the box. No adapter class, no special-casing.
4. **Dependency is strictly one-way** --- LeRobot does not depend on PhysicalAI. PhysicalAI reads LeRobot's output (pure JSON) without importing LeRobot.

```text
LeRobot (standalone)                    PhysicalAI
--------------------                    ----------
policy.export("./out") --produces-->    InferenceModel.load("./out")
                                            |
  Same manifest.json schema                 +-- reads manifest.json
  Writes: type + flat params                +-- resolves via type OR class_path
  Own runners (numpy-only)                  +-- builds preprocessors/postprocessors
  Zero physicalai deps                      +-- runs inference through pipeline
```

---

## Table of Contents

- [Executive Summary](#executive-summary)
- [1. Architecture Overview](#1-architecture-overview)
- [2. Converged Manifest Format](#2-converged-manifest-format)
  - [Schema Overview](#schema-overview)
  - [Full Example: ACT Policy](#full-example-act-policy)
  - [Runner Variants](#runner-variants)
  - [Field Reference](#field-reference)
  - [Dual Component Resolution](#dual-component-resolution)
- [3. How PhysicalAI Loads the Manifest](#3-how-physicalai-loads-the-manifest)
- [4. How LeRobot Uses the Manifest](#4-how-lerobot-uses-the-manifest)
- [5. Runner Mapping](#5-runner-mapping)
- [6. Normalization Handling](#6-normalization-handling)
- [7. Usage Examples](#7-usage-examples)
- [8. Supported Policies](#8-supported-policies)
- [Related Documents](#related-documents)

---

## 1. Architecture Overview

Both frameworks share the same manifest schema. PhysicalAI's `InferenceModel` reads the manifest, resolves components (runner, preprocessors, postprocessors, adapter), and runs inference --- regardless of which framework produced the export.

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
| Single-pass / iterative / two-phase | Yes                | Yes                    |
| Action chunking                     | Yes                | Yes                    |
| Callbacks (timing, logging, safety) | No                 | Yes                    |
| Multi-backend with fallback         | ONNX + OpenVINO    | ONNX + OpenVINO + TRT  |
| Preprocessor/postprocessor chains   | Fixed pipeline     | Extensible chain       |
| HuggingFace Hub loading             | No                 | Yes (`hf://user/repo`) |
| `select_action()` / `reset()` API   | No                 | Yes                    |

---

## 2. Converged Manifest Format

### Schema Overview

The manifest mirrors PhysicalAI's `InferenceModel` class hierarchy:

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
|   +-- preprocessors       (input transforms: normalize, etc.)
|   +-- postprocessors      (output transforms: denormalize, etc.)
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
  },
  "hardware": {
    "robots": [
      {
        "name": "main",
        "type": "SO-100",
        "state": {
          "shape": [6],
          "dtype": "float32",
          "order": ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
        },
        "action": {
          "shape": [6],
          "dtype": "float32",
          "order": ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
        }
      }
    ],
    "cameras": [
      {"name": "top", "shape": [3, 480, 640], "dtype": "uint8"},
      {"name": "wrist", "shape": [3, 480, 640], "dtype": "uint8"}
    ]
  },
  "metadata": {
    "created_at": "2026-03-27T12:00:00Z",
    "created_by": "lerobot.export"
  }
}
```

> **Note on image inputs:** Image normalization (uint8 to float32, divide by 255) is baked into the ONNX graph during export. Only non-image features that use dataset-level statistics (e.g., `observation.state`) need explicit preprocessor entries.

### Runner Variants

The `model.runner` section is open-ended --- policy-specific parameters go directly in the runner object alongside `type`.

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

| Field                      | Type   | Required | Description                                                                                                                                    |
| -------------------------- | ------ | -------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| `model.n_obs_steps`        | int    | Yes      | Number of observation timesteps needed by the model                                                                                            |
| `model.runner`             | object | Yes      | Runner configuration (see [Runner Variants](#runner-variants))                                                                                 |
| `model.runner.type`        | string | Yes      | Runner type: `action_chunking`, `iterative`, `two_phase`                                                                                       |
| `model.artifacts`          | object | Yes      | Map of artifact role to filename. Single-model: `{"model": "model.onnx"}`. Two-phase: `{"encoder": "encoder.onnx", "denoise": "denoise.onnx"}` |
| `model.preprocessors`      | array  | No       | Input transforms (normalize, etc.)                                                                                                             |
| `model.postprocessors`     | array  | No       | Output transforms (denormalize, etc.)                                                                                                          |

#### `hardware` --- Deployment

| Field                           | Type   | Required | Description                                                        |
| ------------------------------- | ------ | -------- | ------------------------------------------------------------------ |
| `hardware.robots`               | array  | No       | Robot configurations                                               |
| `hardware.robots[].name`        | string | Yes      | Logical name (e.g., `"main"`, `"left_arm"`)                        |
| `hardware.robots[].type`        | string | No       | Robot model string (informational, e.g., `"SO-100"`)               |
| `hardware.robots[].state`       | object | No       | Expected state tensor: `shape`, `dtype`, `order` (joint ordering)  |
| `hardware.robots[].action`      | object | No       | Expected action tensor: `shape`, `dtype`, `order` (joint ordering) |
| `hardware.cameras`              | array  | No       | Camera configurations                                              |
| `hardware.cameras[].name`       | string | Yes      | Logical name matching training data keys (e.g., `"top"`, `"wrist"`) |
| `hardware.cameras[].shape`      | array  | No       | `[C, H, W]` tensor shape (e.g., `[3, 480, 640]`)                  |
| `hardware.cameras[].dtype`      | string | No       | Numpy dtype string (default: `"uint8"`)                            |

The `order` field in robot specs declares joint ordering. This is critical for multi-arm setups where `[left, right]` vs `[right, left]` concatenation produces valid shapes with wrong semantics. When present, the runtime can compare declared order against the robot's actual joint order and catch mismatches at startup. Camera and robot `name` fields are **logical names** matching the keys used during training — at deployment, the user maps these to physical devices.

#### `metadata` --- Provenance

| Field                 | Type   | Required | Description        |
| --------------------- | ------ | -------- | ------------------ |
| `metadata.created_at` | string | No       | ISO 8601 timestamp |
| `metadata.created_by` | string | No       | Creator identifier |

#### Preprocessor / Postprocessor Entry

| Field        | Type   | Required | Description                                                                                                                                          |
| ------------ | ------ | -------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| `type`       | string | Yes      | Processor type: `"normalize"`, `"denormalize"`, or custom                                                                                            |
| `class_path` | string | No       | Full Python class path (required for custom types; built-in types resolve by convention)                                                              |
| `mode`       | string | No       | Normalization mode: `"mean_std"`, `"min_max"`, `"identity"`                                                                                          |
| `artifact`   | string | No       | Path to stats file (e.g., `"stats.safetensors"`)                                                                                                     |
| `features`   | array  | No       | Feature names to process (e.g., `["observation.state"]`)                                                                                             |

### Dual Component Resolution

The manifest supports two ways to specify components (runners, preprocessors, postprocessors):

| Format                         | Who writes                         | Who reads            | Example                                                               |
| ------------------------------ | ---------------------------------- | -------------------- | --------------------------------------------------------------------- |
| **`type` + flat params**       | LeRobot, simple PhysicalAI exports | Both (interoperable) | `{"type": "action_chunking", "chunk_size": 100}`                      |
| **`class_path` + `init_args`** | PhysicalAI (full-power)            | PhysicalAI only      | `{"class_path": "physicalai.inference.runners.ActionChunkingRunner", "init_args": {"chunk_size": 100}}` |

Both formats resolve through the same `ComponentRegistry` + `instantiate_component()` pipeline:

- **`class_path`** (full Python path) → direct import → instantiate
- **`type`** (short name) → registry lookup → resolve to full path → instantiate

```json
// LeRobot writes (type + flat params):
{"type": "action_chunking", "chunk_size": 100, "n_action_steps": 100}

// PhysicalAI writes (class_path + init_args):
{"class_path": "physicalai.inference.runners.ActionChunkingRunner", "init_args": {"chunk_size": 100, "n_action_steps": 100}}

// Both resolve to the same ActionChunkingRunner(chunk_size=100, n_action_steps=100)
```

---

## 3. How PhysicalAI Loads the Manifest

The manifest is parsed directly into nested Pydantic models --- no intermediate flattening step:

```python
# In InferenceModel.load():
raw = json.loads((path / "manifest.json").read_text())
manifest = Manifest.model_validate(raw)

# Resolve components from typed manifest fields
runner = resolve_runner(manifest.model.runner)
adapter = create_adapter(manifest.model.artifacts, path)
preprocessors = resolve_processors(manifest.model.preprocessors, path)
postprocessors = resolve_processors(manifest.model.postprocessors, path)
```

Runner and processor resolution both use **dual-path resolution** --- a single if-check, not an if-chain per type:

```python
def resolve_runner(runner_config: dict) -> InferenceRunner:
    if "class_path" in runner_config:
        # PhysicalAI-native: class_path + init_args → ComponentSpec → instantiate
        spec = ComponentSpec.model_validate(runner_config)
        return instantiate_component(spec)

    # Framework-agnostic: type → registry lookup → instantiate
    runner_type = runner_config["type"]
    init_args = {k: v for k, v in runner_config.items() if k != "type"}
    spec = ComponentSpec(class_path=runner_type, init_args=init_args)
    return instantiate_component(spec)
```

Processors follow the same pattern, with one addition: the `artifact` key in `type`-format specs is resolved to an absolute `stats_path` at load time.

> **Legacy `metadata.yaml` files** (pre-manifest era) are handled separately by `from_legacy_metadata()` in `manifest.py`.

---

## 4. How LeRobot Uses the Manifest

LeRobot reads the same `manifest.json` with its own tooling (no PhysicalAI dependency):

```python
import json
from pathlib import Path

def load_exported_policy(path: str | Path) -> ExportedPolicy:
    path = Path(path)
    raw = json.loads((path / "manifest.json").read_text())

    # Build LeRobot's own runner (standalone, numpy-only)
    runner_config = raw["model"]["runner"]
    runner = build_runner(runner_config)

    # Load normalizer from manifest specs
    preprocessors = raw["model"].get("preprocessors", [])
    postprocessors = raw["model"].get("postprocessors", [])
    normalizer = Normalizer.from_specs(preprocessors + postprocessors, path)

    # Load backend adapter
    artifacts = raw["model"]["artifacts"]
    adapter = ONNXRuntimeAdapter(path / artifacts["model"])

    return ExportedPolicy(runner=runner, adapter=adapter, normalizer=normalizer)
```

LeRobot's runners, normalizer, and adapters are its own implementations with zero overlap with PhysicalAI's. The only shared artifact is `manifest.json` on disk.

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

---

## 6. Normalization Handling

LeRobot policies operate on **normalized** inputs and produce **normalized** outputs. The manifest declares normalization as transforms in `model.preprocessors` and `model.postprocessors`:

```json
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
```

PhysicalAI resolves these to `StatsNormalizer` (preprocessor) and `StatsDenormalizer` (postprocessor), which load stats from `stats.safetensors` and apply per-feature transforms.

### Normalization Modes

| Mode       | Normalize                         | Denormalize                       |
| ---------- | --------------------------------- | --------------------------------- |
| `mean_std` | `(x - mean) / std`                | `x * std + mean`                  |
| `min_max`  | `(x - min) / (max - min) * 2 - 1` | `(x + 1) / 2 * (max - min) + min` |
| `identity` | passthrough                       | passthrough                       |

Statistics are stored in `safetensors` format with `{feature}/mean`, `{feature}/std`, `{feature}/min`, `{feature}/max` tensors.

---

## 7. Usage Examples

### Basic Usage

```python
from physicalai import InferenceModel

# Load LeRobot-exported policy (detected automatically via manifest.json)
model = InferenceModel("./act_exported")

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

model = InferenceModel("./pi0_exported", callbacks=[TimingCallback()])
outputs = model(observation)
```

### Override Runner Parameters

```python
model = InferenceModel(
    "./diffusion_exported",
    num_steps=20,         # Override manifest default of 100
    scheduler="ddim",     # Override manifest default of "ddpm"
)
```

### Real-Time Control

```python
policy = InferenceModel("./act_exported")
policy.reset()

while not done:
    action = policy.select_action(observation)
    observation, reward, done, info = env.step(action)

policy.reset()
```

---

## 8. Supported Policies

| Policy    | `runner.type`   | Runner Stack                                 | Artifact Roles       |
| --------- | --------------- | -------------------------------------------- | -------------------- |
| ACT       | action_chunking | ActionChunking(SinglePass)                   | `model`              |
| VQBeT     | action_chunking | ActionChunking(SinglePass)                   | `model`              |
| Diffusion | iterative       | ActionChunking(Iterative(SinglePass))        | `model`              |
| TDMPC     | iterative       | Iterative(SinglePass) with MPC               | `model`              |
| PI0       | two_phase       | ActionChunking(TwoPhase(encoder, Iterative)) | `encoder`, `denoise` |
| SmolVLA   | two_phase       | ActionChunking(TwoPhase(encoder, Iterative)) | `encoder`, `denoise` |

---

## Related Documents

- **[Inference Core Design](../components/inferencekit.md)** --- Domain-agnostic inference layer
- **[Strategy](../architecture/strategy.md)** --- Big-picture architecture and layering decisions
- **[Architecture](../architecture/architecture.md)** --- PhysicalAI runtime CLI and packaging

---

_Document version: 6.0_
_Last updated: 2026-03-31_
