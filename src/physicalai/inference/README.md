# Inference

Loads exported policy packages and runs inference through backend-specific adapters.

## Public API

- `InferenceModel`: main entrypoint for loading an export and selecting actions
- `Manifest`: exported package metadata and component specs

## Main Modules

- `model.py`: `InferenceModel` implementation
- `manifest.py`: manifest loading and config primitives
- `adapters/`: backend-specific runtimes such as OpenVINO and ONNX
- `preprocessors/` and `postprocessors/`: inference pipeline components
- `runners/`: policy execution strategies

## Related Docs

- `docs/explanation/inference.md`
- `docs/explanation/manifests.md`
- `docs/reference/inference-api.md`
- `docs/reference/manifest-schema.md`
