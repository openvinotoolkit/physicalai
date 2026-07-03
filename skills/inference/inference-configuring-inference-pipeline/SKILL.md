---
name: inference-configuring-inference-pipeline
description: Configures preprocessors, postprocessors, and runners around InferenceModel via manifest specs and ComponentRegistry. Use when editing physicalai.inference.preprocessors or postprocessors, manifest preprocessor/postprocessor lists, instantiate_component, registered type names, or class_path init_args for inference pipeline components.
license: Apache-2.0
---

# Configuring the Inference Pipeline

Pipeline order: observation → preprocessors → runner → postprocessors → action output. See `docs/how-to/inference/configure-pre-post-processing.md`.

Core code:

- `src/physicalai/inference/component_factory.py` — `ComponentRegistry`, `instantiate_component`, `_MAX_COMPONENT_DEPTH`.
- `src/physicalai/inference/model.py` — builds processor chains from manifest specs.
- Built-ins under `preprocessors/` and `postprocessors/`; runners under `runners/`.

## Workflow

1. **Read the manifest slice** for `preprocessors`, `postprocessors`, and `model.runner`.
   - Done when: you know whether specs use `type` (registry short name) or `class_path`.
2. **Prefer `type` for built-ins** registered in `component_factory` (e.g. normalize/denormalize patterns in docs).
3. **Use `class_path` + `init_args`** for explicit classes:

   ```yaml
   preprocessors:
     - class_path: physicalai.inference.preprocessors.StatsNormalizer
       init_args:
         artifact: stats.safetensors
   ```

   - Done when: `init_args` paths resolve relative to the export directory via `resolve_artifact`.

4. **Add a new built-in processor**:
   - Implement subclass of `Preprocessor` / `Postprocessor` in the appropriate package.
   - Register a short `type` name in `component_factory` if manifest-friendly aliases are needed.
   - Add unit tests under `tests/unit/inference/preprocessors/` or `postprocessors/`.
   - Done when: manifest using `type` or `class_path` instantiates in a minimal `InferenceModel` test.
5. **Nested components** in `init_args` must stay within `_MAX_COMPONENT_DEPTH`; avoid cyclic specs.

## Validation loop

```bash
uv run pytest tests/unit/inference/preprocessors tests/unit/inference/postprocessors tests/unit/inference/test_manifest.py -q
```

## Required checks

- Processor order matches training/export semantics (normalization before runner, denormalization after).
- Artifact file names in manifests do not traverse paths (`..`, absolute paths).
- New public processors appear in `docs/reference/inference-api.md` or how-to docs when user-visible.
- Runner choice (`SinglePass`, chunking runners) stays consistent with `predict_action_chunk` vs `select_action` docs.

## References

- `docs/reference/manifest-schema.md`
- `docs/how-to/inference/use-manifest.md`
