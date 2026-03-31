# Inference Core Design Document

> **Scope note:** This document describes the design of the **domain‑agnostic inference core** — the layer that provides backend execution, metadata IO, and the base `InferenceModel`. In our proposed architecture, this layer lives **inside physicalai** as `physicalai.inference`, not as a separate package. References to "inferencekit" in this document describe the module's design; the module path is `physicalai.inference.*`. This layer can be silently extracted as a standalone package later if other domains (e.g., vision via model_api) need it independently.

**Base inference framework providing unified model loading, prediction, and extensibility across backends and domains.**

---

## Table of Contents

- [Inference Core Design Document](#inference-core-design-document)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
    - [Purpose](#purpose)
    - [Architecture Position](#architecture-position)
    - [Design Goals](#design-goals)
    - [Non-Goals](#non-goals)
  - [Architecture](#architecture)
    - [Package Structure](#package-structure)
    - [Design Principles](#design-principles)
  - [Core Components](#core-components)
    - [InferenceModel](#inferencemodel)
    - [RuntimeAdapter](#runtimeadapter)
    - [InferenceRunner](#inferencerunner)
    - [Callback System](#callback-system)
    - [Preprocessors and Postprocessors](#preprocessors-and-postprocessors)
    - [Manifest Format](#manifest-format)
  - [Extension \& Plugin System](#extension--plugin-system)
  - [Runners (Domain-Provided)](#runners-domain-provided)
  - [Supported Backends](#supported-backends)
  - [Domain Layer Example](#domain-layer-example)
  - [Usage Examples](#usage-examples)
    - [Basic usage](#basic-usage)
    - [With explicit backend](#with-explicit-backend)
    - [With callbacks](#with-callbacks)
    - [Context manager for resource cleanup](#context-manager-for-resource-cleanup)
  - [Related Documents](#related-documents)

---

## Overview

### Purpose

**inferencekit** is the base execution engine for the Geti ecosystem. It standardizes backend execution and metadata IO. It provides:

- Backend abstraction (OpenVINO, ONNX, TensorRT, Torch)
- Manifest loading (`manifest.json`)
- Minimal `InferenceModel(path)` + `model(inputs)` API

**inferencekit knows nothing about vision, robotics, or any specific domain.** Domain plugins live above it (physical‑ai‑framework, model_api, custom layers).

### Architecture Position

inferencekit is the **foundation layer** in a layered architecture. Domain-specific systems build on top of it, each adding their own preprocessing, postprocessing, runners, and model types:

```text
┌───────────────────────────────────────────────────────────────────────────────┐
│                       Domain Layers                                           │
│                                                                               │
│  ┌──────────────────┐  ┌──────────────────────────────┐  ┌─────────────────┐  │
│  │    model_api     │  │  physical‑ai‑framework       │  │  custom-xyz     │  │
│  │  (vision)        │  │  (physical‑AI)               │  │  (your domain)  │  │
│  │                  │  │                              │  │                 │  │
│  │  YOLO, SAM,      │  │  Policy plugins:             │  │  Your models,   │  │
│  │  Anomaly, OTX,   │  │  physicalai-train, LeRobot,  │  │  your runners,  │  │
│  │  Ultralytics,    │  │  custom frameworks           │  │  publishable    │  │
│  │  Roboflow        │  │                              │  │  on HuggingFace │  │
│  └────────┬─────────┘  └────────┬─────────────────────┘  └───────┬─────────┘  │
│           │                     │                     │                       │
│           └─────────────────────┼─────────────────────┘                       │
│                                 │                                             │
│                          depends on                                           │
│                                 │                                             │
│                                 ▼                                             │
│  ┌──────────────────────────────────────────────────────────────┐             │
│  │                       inferencekit                           │             │
│  │                    (base framework)                          │             │
│  │                                                              │             │
│  │  InferenceModel  │  RuntimeAdapter  │  InferenceRunner       │             │
│  │  Callbacks       │  Pre/Post ABCs   │  Plugin Registry       │             │
│  │  OpenVINO, ONNX, TensorRT, Torch backends                    │             │
│  └──────────────────────────────────────────────────────────────┘             │
└───────────────────────────────────────────────────────────────────────────────┘
```

**Key principle:** Domain layers depend on inferencekit. inferencekit depends on nothing domain-specific. Dependencies flow upward only.

| Layer                     | Owns                                                                                                                                                      | Does NOT own                          |
| ------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------- |
| **inferencekit**          | Backend adapters, manifest IO, base InferenceModel                                                                                                        | Vision models, robotics, domain logic |
| **model_api**             | Vision preprocessing, task wrappers (YOLO, SAM), result types                                                                                             | Backend execution, robotics           |
| **physical‑ai‑framework** | Policy plugins, unified APIs, orchestration, observation pipeline, safety runtime, episode orchestration, device management, camera/robot interfaces, CLI | Backend execution, training           |
| **custom layers**         | Domain-specific models, runners, pre/postprocessors                                                                                                       | Backend execution, core infra         |

### Design Goals

| Goal                         | Description                                                  |
| ---------------------------- | ------------------------------------------------------------ |
| **G1: Execution Engine**     | Provide backend execution and manifest IO                    |
| **G2: Minimal API**          | `InferenceModel(path)` + `model(inputs)` across backends     |
| **G3: Backend Agnostic**     | Support OpenVINO, ONNX, TensorRT, Torch without code changes |
| **G4: Minimal Dependencies** | Core has few requirements; optional extras per backend       |
| **G5: Domain Agnostic**      | No vision, robotics, or domain-specific code                 |

### Non-Goals

| Non-Goal                                        | Rationale                        |
| ----------------------------------------------- | -------------------------------- |
| Vision preprocessing/postprocessing             | Belongs in model_api             |
| Physical‑AI orchestration                       | Belongs in physical‑ai‑framework |
| Training infrastructure                         | Separate concern                 |
| Result types (DetectionResult, etc.)            | Domain-layer concern             |
| Framework-specific wrappers (Ultralytics, etc.) | Domain-layer concern             |

---

## Architecture

### Package Structure

```text
inferencekit/
├── __init__.py                              # Public API: InferenceModel
├── model.py                                 # InferenceModel - main entry point
├── runners/
│   ├── __init__.py
│   ├── base.py                              # InferenceRunner ABC
│   ├── single_pass.py                       # SinglePassRunner (default)
│   ├── batch.py                             # BatchRunner
│   └── streaming.py                         # StreamingRunner
├── adapters/
│   ├── __init__.py                          # get_adapter() factory
│   ├── base.py                              # RuntimeAdapter ABC
│   ├── openvino.py                          # OpenVINO backend
│   ├── onnx.py                              # ONNX Runtime backend
│   ├── tensorrt.py                          # TensorRT backend
│   └── torch_export.py                      # Torch Export IR / ExecuTorch
├── callbacks/
│   ├── __init__.py
│   ├── base.py                              # Callback ABC
│   ├── timing.py                            # TimingCallback
│   └── logging.py                           # LoggingCallback
├── preprocessors/
│   ├── __init__.py
│   └── base.py                              # Preprocessor ABC
├── postprocessors/
│   ├── __init__.py
│   └── base.py                              # Postprocessor ABC
├── io/
│   ├── __init__.py
│   ├── manifest.py                          # Manifest loading (JSON)
│   └── instantiate.py                       # class_path + init_args
├── plugins/
│   ├── __init__.py                          # Plugin registry + entry points
│   ├── base.py                              # Plugin ABC
│   └── registry.py                          # BackendRegistry, RunnerRegistry
└── contrib/
    ├── __init__.py
    ├── iterative.py                         # IterativeRunner (flow-matching)
    └── tiled.py                             # TiledRunner (large inputs)
```

### Design Principles

| Principle                        | Description                                                                          |
| -------------------------------- | ------------------------------------------------------------------------------------ |
| **Foundation, Not Application**  | inferencekit provides ABCs and infrastructure; domain layers provide implementations |
| **Composition over Inheritance** | Runners, callbacks, adapters are composable building blocks                          |
| **Progressive Disclosure**       | Simple API for 90% of users, full control for power users                            |
| **Plugin-First Extensibility**   | New backends, runners, formats via registry + entry points                           |
| **Minimal Dependencies**         | Core has few requirements; backends and contrib are optional extras                  |

---

## Core Components

### InferenceModel

The main entry point for inference. Orchestrates runners, adapters, and callbacks.

**Design Philosophy:**

90% of users should only need:

```python
from inferencekit import InferenceModel

model = InferenceModel("./exports/my_model")
outputs = model(inputs)
```

Progressive customization for advanced users:

```python
# Tier 2: Override parameters
model = InferenceModel(
    "./exports/my_model",
    backend="onnx",
    device="cuda",
)

# Tier 3: Explicit components
from inferencekit.callbacks import TimingCallback

model = InferenceModel(
    "./exports/my_model",
    callbacks=[TimingCallback()],
)

# Tier 4: Full control (domain layers use this)
from inferencekit.adapters import ONNXAdapter
from inferencekit.runners import SinglePassRunner

adapter = ONNXAdapter(device="cuda")
adapter.load(Path("./model.onnx"))
runner = SinglePassRunner()
model = InferenceModel(adapter=adapter, runner=runner)
```

**API:**

```python
class InferenceModel:
    """Unified inference interface for exported models.

    Automatically detects backend, device, and configuration from
    export directory metadata. Domain layers can subclass or compose
    this to add domain-specific behavior.

    Callable: use model(inputs) to run inference.
    """

    def __init__(
        self,
        path: str | Path | None = None,
        backend: str | None = None,
        device: str = "auto",
        callbacks: list[Callback] | None = None,
        *,
        adapter: RuntimeAdapter | None = None,
        runner: InferenceRunner | None = None,
        **kwargs,
    ):
        """Initialize and load model.

        Args:
            path: Directory containing exported model and metadata,
                  or HuggingFace URI (hf://user/repo)
            backend: Backend to use (auto-detected from metadata if None)
            device: Device for inference ("auto", "cpu", "cuda", "CPU", "GPU")
            callbacks: Optional callbacks for instrumentation
            adapter: Explicit RuntimeAdapter (advanced; skips auto-detection)
            runner: Explicit InferenceRunner (advanced; skips metadata lookup)
            **kwargs: Additional arguments passed to runner/adapter
        """
        ...

    def __call__(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Run inference on inputs.

        Args:
            inputs: Dictionary mapping input names to arrays/tensors

        Returns:
            Dictionary mapping output names to arrays/tensors
        """
        ...

    def reset(self) -> None:
        """Reset model state (for stateful runners)."""
        ...

    def __enter__(self) -> "InferenceModel":
        """Context manager entry."""
        ...

    def __exit__(self, *args) -> None:
        """Context manager exit - cleanup resources."""
        ...
```

### RuntimeAdapter

Adapters execute **one forward pass** on a specific backend. Intentionally stateless.

```python
class RuntimeAdapter(ABC):
    """Abstract base class for backend-specific inference.

    Each backend (OpenVINO, ONNX, TensorRT, Torch) implements this
    interface. Domain layers should NOT need to subclass this — they
    compose adapters via runners and callbacks instead.
    """

    def __init__(self, device: str = "cpu", **kwargs):
        self.device = device
        self.config = kwargs

    @abstractmethod
    def load(self, model_path: Path) -> None:
        """Load model from disk."""
        ...

    @abstractmethod
    def predict(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Run single forward pass."""
        ...

    @property
    @abstractmethod
    def input_names(self) -> list[str]:
        """Model input names."""
        ...

    @property
    @abstractmethod
    def output_names(self) -> list[str]:
        """Model output names."""
        ...
```

### InferenceRunner

Runners define **how inference runs** — the algorithm, not what happens to outputs.

Runners are implemented in domain layers (physical‑ai plugins, model_api, custom). inferencekit provides only the interface.

```python
class InferenceRunner(ABC):
    """Abstract base class for inference execution patterns.

    Runners control the inference algorithm: single pass, iterative
    denoising, tiled execution, streaming, etc. Domain layers should
    subclass InferenceRunner to implement domain-specific patterns.
    """

    @abstractmethod
    def run(self, adapter: RuntimeAdapter, inputs: dict) -> dict:
        """Execute the inference pattern.

        Args:
            adapter: Backend adapter for forward passes
            inputs: Model inputs

        Returns:
            Model outputs
        """
        ...

    def reset(self) -> None:
        """Reset runner state between episodes/sequences."""
        pass
```

### Callback System

Lightning-compatible hooks for cross-cutting concerns:

```python
class Callback:
    """Base callback class. Override hooks as needed.

    Callbacks are the preferred way to add instrumentation,
    safety checks, logging, and other cross-cutting concerns
    without modifying model or runner code.
    """

    def on_predict_start(self, inputs: dict) -> dict | None:
        """Called before prediction. Can modify inputs."""
        pass

    def on_predict_end(self, outputs: dict) -> dict | None:
        """Called after prediction. Can modify outputs."""
        pass

    def on_reset(self) -> None:
        """Called when model state is reset."""
        pass

    def on_load(self, model: "InferenceModel") -> None:
        """Called after model is loaded."""
        pass
```

Callbacks are domain‑provided. inferencekit defines the interface; domain layers supply implementations.

### Preprocessors and Postprocessors

Transform inputs before inference and outputs after:

```python
class Preprocessor(ABC):
    """Transform inputs before inference.

    Domain layers implement concrete preprocessors:
    - model_api: ImageResize, ImageNormalize, LayoutTransform
    - physicalai-train: ObservationNormalizer, ActionUnnormalizer
    """

    @abstractmethod
    def __call__(self, inputs: dict) -> dict:
        ...

class Postprocessor(ABC):
    """Transform outputs after inference.

    Domain layers implement concrete postprocessors:
    - model_api: NMS, BoxDecoder, MaskDecoder
    - physicalai-train: ActionChunker, ActionClamp
    """

    @abstractmethod
    def __call__(self, outputs: dict) -> dict:
        ...
```

### Manifest Format

All exported models use a unified `manifest.json` format. The manifest uses a nested structure that mirrors the `InferenceModel` class hierarchy, with logical sections for policy identity, model configuration, hardware, and metadata:

```text
manifest.json
├── format + version        (envelope)
├── policy                  (identity — what policy is this?)
│   ├── name                (human-readable name)
│   └── source              (provenance: repo_id, class_path)
├── model                   (exported model — how to run it?)
│   ├── n_obs_steps         (observation window)
│   ├── runner              (execution pattern + params)
│   ├── artifacts           (model files by named role)
│   ├── preprocessors       (input transforms: normalize, etc.)
│   └── postprocessors      (output transforms: denormalize, etc.)
├── hardware                (deployment — what hardware?)
│   ├── robots              (robot configurations)
│   └── cameras             (camera configurations)
└── metadata                (provenance — when/who created this?)
```

```json
{
  "format": "policy_package",
  "version": "1.0",
  "policy": {
    "name": "my_model",
    "source": {
      "repo_id": "user/my_model",
      "class_path": "mypackage.policies.MyPolicy"
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
    "robots": [],
    "cameras": []
  },
  "metadata": {
    "created_at": "2026-03-27T12:00:00Z",
    "created_by": "mypackage.export"
  }
}
```

> **Note:** For the full manifest schema reference (all runner variants, field descriptions, and design rationale), see [LeRobot Integration Design](../integrations/lerobot.md#2-converged-manifest-format). The format is shared by both PhysicalAI and LeRobot exports.

**PhysicalAI-native format (`class_path` + `init_args`):**

PhysicalAI can also write manifests using the full `class_path` + `init_args` format for components. This gives full power over component instantiation (custom classes, nested configs) while remaining loadable by PhysicalAI's `ComponentRegistry`:

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
      "class_path": "physicalai.inference.runners.ActionChunkingRunner",
      "init_args": {
        "chunk_size": 100,
        "n_action_steps": 100
      }
    },
    "artifacts": {
      "model": "model.onnx"
    },
    "preprocessors": [
      {
        "class_path": "physicalai.inference.preprocessors.StatsNormalizer",
        "init_args": {
          "mode": "mean_std",
          "stats_path": "stats.safetensors",
          "features": ["observation.state"]
        }
      }
    ],
    "postprocessors": [
      {
        "class_path": "physicalai.inference.postprocessors.StatsDenormalizer",
        "init_args": {
          "mode": "mean_std",
          "stats_path": "stats.safetensors",
          "features": ["action"]
        }
      }
    ]
  },
  "hardware": {
    "robots": [],
    "cameras": []
  },
  "metadata": {
    "created_at": "2026-03-27T12:00:00Z",
    "created_by": "physicalai.export"
  }
}
```

> **Both formats resolve identically.** The `type`-based example above (used by LeRobot) and this `class_path`-based example both resolve to the same runner and processor instances through the `ComponentRegistry`. See [Dual Component Resolution](../integrations/lerobot.md#dual-component-resolution) for the full resolution algorithm.

**How models are loaded:**

The framework reads `manifest.json` and resolves the model configuration using **dual-path component resolution**:

1. **Manifest parsing**: `manifest.json` is parsed directly into nested Pydantic models --- no flattening or normalization step.
2. **Runner resolution**: Components support two formats that both resolve through the same `ComponentRegistry` + `instantiate_component()` pipeline:
   - **`type` + flat params** (interoperable, written by LeRobot): `{"type": "action_chunking", "chunk_size": 100}` → registry lookup → `ComponentSpec` → `instantiate_component()`
   - **`class_path` + `init_args`** (full-power, written by PhysicalAI): `{"class_path": "physicalai.inference.runners.ActionChunkingRunner", "init_args": {"chunk_size": 100}}` → `ComponentSpec` → `instantiate_component()`
3. **Backend selection**: `model.artifacts` maps named roles (e.g., `"model"`, `"encoder"`) to filenames. The first available backend is auto-selected, or the user can override at load time.
4. **I/O pipeline**: `model.preprocessors` and `model.postprocessors` declare input/output transforms (normalization, denormalization) resolved via the same dual-path mechanism.
5. **Hardware validation**: `hardware.robots` and `hardware.cameras` sections declare expected shapes. The runtime can validate observations against these.
6. **Custom components**: Domain layers can extend the manifest with custom processor types or runner parameters without modifying inferencekit. Any component with a `class_path` is instantiated directly; any component with a `type` goes through the registry.

> **See also**: [LeRobot Integration Design — Runner Resolution](../integrations/lerobot.md#runner-resolution) for the full resolution algorithm and examples.

---

## Extension & Plugin System

inferencekit supports **backend adapters** as extensions via a registry. Domain-specific plugins (runners, processors, models) live in their respective domain layers, not in inferencekit.

**Backend registry:** New backends implement `RuntimeAdapter` and register via Python entry points (`inferencekit.backends`). Domain layers register runners and processors via their own entry points (`inferencekit.runners`, `inferencekit.callbacks`).

**Building a custom domain layer:** Subclass `InferenceModel`, implement domain-specific runners and pre/postprocessors, and register via entry points:

```python
# my_domain/model.py — subclass InferenceModel
class MyDomainModel(InferenceModel):
    def __init__(self, path, **kwargs):
        super().__init__(path, **kwargs)
        self.preprocessors = [MyPreprocessor()]

    def domain_predict(self, domain_inputs):
        inputs = self._preprocess(domain_inputs)
        return self(inputs)
```

```toml
# pyproject.toml — register custom runners
[project.entry-points."inferencekit.runners"]
my_runner = "my_domain.runners:MyDomainRunner"
```

**HuggingFace publishing:** Domain layers can publish model packages to HuggingFace containing exported artifacts + `manifest.json`. Loading is automatic:

```python
model = InferenceModel("hf://username/my-model")
outputs = model(inputs)
```

---

## Runners (Domain-Provided)

inferencekit defines the `InferenceRunner` interface. Domain layers implement concrete runners:

| Runner | Description | Stateful |
| --- | --- | --- |
| **SinglePassRunner** | Default. One forward pass per call. Covers 90% of use cases. | No |
| **BatchRunner** | Splits inputs into batches for throughput optimization. | No |
| **StreamingRunner** | Buffers inputs for real-time streaming applications. | Yes |

```python
class SinglePassRunner(InferenceRunner):
    """Default runner. Covers 90% of use cases."""

    def run(self, adapter: RuntimeAdapter, inputs: dict) -> dict:
        return adapter.predict(inputs)
```

**Contrib runners** (`inferencekit.contrib`): Reference implementations for common patterns, shipped as optional extras:

| Runner | Description | Use Case |
| --- | --- | --- |
| **IterativeRunner** | Multi-step denoising with configurable scheduler | Diffusion, flow-matching policies |
| **TiledRunner** | Tile-based inference with overlap and merging | High-resolution images, satellite imagery |

Domain layers can contribute runners back to `inferencekit.contrib` via pull request, or ship them in their own packages.

---

## Supported Backends

| Backend             | Hardware             | Status      | Installation                         |
| ------------------- | -------------------- | ----------- | ------------------------------------ |
| **OpenVINO**        | Intel CPU/GPU/NPU    | Implemented | `pip install inferencekit[openvino]` |
| **ONNX Runtime**    | Cross-platform, CUDA | Implemented | `pip install inferencekit[onnx]`     |
| **TensorRT**        | NVIDIA GPU           | Planned     | `pip install inferencekit[tensorrt]` |
| **Torch Export IR** | CPU/CUDA, ExecuTorch | Implemented | Built-in                             |

Third-party backends can be added via the backend registry without modifying inferencekit.

---

## Domain Layer Example

This example shows how physicalai builds on inferencekit's interfaces. Policy-specific behavior (`select_action`, episode reset) is implemented in physical‑ai‑framework's `InferenceModel` wrapper:

```python
from inferencekit import InferenceModel
from inferencekit.runners import InferenceRunner


class ActionChunkingRunner(InferenceRunner):
    """Runner that manages action chunk queues.

    Policies output action chunks (multiple future actions).
    This runner queues them and dispenses one action per call.
    """

    def __init__(self, chunk_size: int = 16, n_action_steps: int = 1):
        self.chunk_size = chunk_size
        self.n_action_steps = n_action_steps
        self._action_queue = []

    def run(self, adapter, inputs):
        if not self._action_queue:
            outputs = adapter.predict(inputs)
            chunk = outputs["action"]  # shape: (chunk_size, action_dim)
            self._action_queue = list(chunk[:self.n_action_steps])

        action = self._action_queue.pop(0)
        return {"action": action}

    def reset(self):
        self._action_queue = []
```

Other domain layers (model_api for vision, custom audio/NLP packages) follow the same pattern: subclass `InferenceModel`, implement domain runners and pre/postprocessors, register via entry points.

---

## Usage Examples

### Basic usage

```python
from inferencekit import InferenceModel

# Load model (auto-detects backend)
model = InferenceModel("./exports/my_model")

# Run inference
inputs = {"input": data_array}
outputs = model(inputs)
```

### With explicit backend

```python
model = InferenceModel(
    "./exports/my_model",
    backend="openvino",
    device="CPU",
)
```

### With callbacks

```python
from inferencekit.callbacks import TimingCallback, LoggingCallback

model = InferenceModel(
    "./exports/my_model",
    callbacks=[TimingCallback(), LoggingCallback()],
)

# Callbacks fire automatically
outputs = model(inputs)
```

### Context manager for resource cleanup

```python
with InferenceModel("./exports/my_model") as model:
    outputs = model(inputs)
# Resources automatically cleaned up
```

---

## Related Documents

- **[Strategy](../architecture/strategy.md)** — Big-picture architecture and layering decisions
- **[Architecture](../architecture/architecture.md)** — physicalai runtime CLI and packaging
- **[LeRobot Integration](../integrations/lerobot.md)** — LeRobot integration for physicalai (built‑in, reads manifest.json)

---

_Document Version: 6.0_
_Last Updated: 2026-03-31_
