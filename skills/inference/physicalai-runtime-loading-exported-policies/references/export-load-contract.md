# Export/Load Contract (Runtime)

Studio produces deployment artifacts; Runtime loads them with `InferenceModel` / `InferenceModel.from_pretrained(...)`.

## Required Artifact Properties

- A backend-identifying model file exists in the export directory (e.g. `.onnx`, `.xml` + `.bin`).
- `manifest.json` describes the policy, backend artifacts, preprocessors/postprocessors, runner, and action chunk semantics.
- Runtime auto-detects the backend from registered file extensions or loads when `backend=` is set explicitly.
- Optional backend dependencies fail with clear installation guidance.

## Backend Ownership

- Studio owns export implementation and export metadata generation.
- Runtime owns adapter discovery, backend loading, preprocessing, inference execution, and action selection from exported artifacts.
- Studio and Runtime must keep this contract synchronized conceptually. Update this file when Studio export metadata changes.

## Compatibility Rules

- Do not change expected manifest field semantics without coordinating Studio export changes.
- Do not add a backend to user-facing instructions unless an adapter is registered under `src/physicalai/inference/adapters/`.
- Treat numerical parity (Studio export validation) and deployment latency as separate checks.
