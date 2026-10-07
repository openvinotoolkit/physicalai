---
name: pai-configure-inference
description: Configures an exported policy's inference pipeline with Runtime manifest preprocessors, runners, and postprocessors. Use when writing preprocessor/postprocessor lists, choosing a registered type or class_path/init_args, resolving export artifacts, or fixing processor order for InferenceModel. DO NOT USE FOR implementing a built-in processor or registry entry in Runtime source.
license: Apache-2.0
---

# Configuring the Inference Pipeline

Pipeline order: observation → preprocessors → runner → postprocessors → action output. See the [pre/post-processing guide](https://github.com/openvinotoolkit/physicalai/blob/main/docs/how-to/inference/configure-pre-post-processing.md) and [manifest schema](https://github.com/openvinotoolkit/physicalai/blob/main/docs/reference/manifest-schema.md).

## Workflow

1. **Read the manifest slice** for `preprocessors`, `postprocessors`, and `model.runner`.
   - Done when: you know whether specs use `type` (registry short name) or `class_path`.
2. **Prefer `type` for built-ins** registered by your installed Runtime package (e.g. normalize/denormalize patterns in the docs).
3. **Use `class_path` + `init_args`** for explicit classes:

   ```yaml
   preprocessors:
     - class_path: physicalai.inference.preprocessors.StatsNormalizer
       init_args:
         artifact: stats.safetensors
   ```

   - Done when: `init_args` paths resolve relative to the export directory via `resolve_artifact`.

4. **Validate the manifest with a representative export** by constructing `InferenceModel` and running a fake observation; verify preprocessing precedes the runner and postprocessing follows it.
   - Done when: the model returns an action with the expected shape.
5. **Nested components** in `init_args` must stay within Runtime's configured depth limit; avoid cyclic specs.

## Validation loop

Run a one-observation smoke inference in the user's project after each manifest change; do not connect robot hardware to test a config edit.

## Required checks

- Processor order matches training/export semantics (normalization before runner, denormalization after).
- Artifact file names in manifests do not traverse paths (`..`, absolute paths).
- A named `type` exists in the installed package; use `class_path` only for trusted local components.
- Runner choice (`SinglePass`, chunking runners) stays consistent with `predict_action_chunk` vs `select_action` docs.

## References

- [Manifest schema](https://github.com/openvinotoolkit/physicalai/blob/main/docs/reference/manifest-schema.md)
- [Using a manifest](https://github.com/openvinotoolkit/physicalai/blob/main/docs/how-to/inference/use-manifest.md)
