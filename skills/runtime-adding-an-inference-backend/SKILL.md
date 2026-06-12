---
name: runtime-adding-an-inference-backend
description: Add or modify a Physical AI Runtime inference backend adapter. Use when implementing RuntimeAdapter classes, backend_registry registration, ONNX, OpenVINO, Torch, ExecuTorch companion adapters, file extension detection, InferenceModel.load(...), exported artifact metadata, or the Runtime side of the Studio export/load contract.
license: Apache-2.0
---

# Adding a Runtime Inference Backend

Use this skill for changes under `src/physicalai/inference/adapters`, `InferenceModel.load(...)`, backend discovery, or exported artifact compatibility.

## Workflow

1. Identify the exported artifact format, file extensions, optional dependencies, and target hardware/runtime.
2. Inspect the existing ONNX and OpenVINO adapters before adding a new adapter.
3. Implement the `RuntimeAdapter` contract with explicit load, predict, and cleanup behavior.
4. Register the backend with `backend_registry.register(...)` or `register_lazy_module(...)` so auto-detection can find it without importing heavy dependencies too early.
5. Validate the adapter against the shared export/load contract in `references/export-contract.md`.
6. Add tests for registration, extension detection, missing dependency errors, and at least one load/predict path using mocks when real runtime dependencies are heavy.

## Runtime Packaging Rules

- Runtime core ships ONNX and OpenVINO adapters.
- Additional backends should keep optional dependencies isolated, ideally through extras or companion distributions.
- Do not add user-facing support claims unless installation and loading are documented.

## Required Checks

- Backend names and extensions are stable and unique.
- Missing optional dependencies raise helpful installation guidance.
- The adapter consumes Studio export metadata without changing field semantics unilaterally.
- Auto-detection remains deterministic when multiple model files are present.

## References

- See `references/backend-adapters.md` for adapter implementation patterns.
- See `references/export-contract.md` for the shared Studio/Runtime artifact contract.
