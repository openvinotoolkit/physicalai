---
name: pai-load-policy
description: Loads and validates policies exported from Physical AI Studio for Runtime deployment. Use when calling InferenceModel or from_pretrained on a local export or Hub snapshot, checking manifest.json, selecting an ONNX/OpenVINO backend or device, or smoke-testing an action. DO NOT USE FOR implementing an inference adapter in Runtime source.
license: Apache-2.0
---

# Loading Exported Policies

Runtime loads Studio export directories (or Hub snapshots that mirror them) through `InferenceModel`. It reads `manifest.json`, auto-detects registered ONNX/OpenVINO adapters from model files, and supports Hub revisions. See the [inference API](https://github.com/openvinotoolkit/physicalai/blob/main/docs/reference/inference-api.md) and [manifest schema](https://github.com/openvinotoolkit/physicalai/blob/main/docs/reference/manifest-schema.md).

## Workflow

1. **Identify the artifact**: local export directory or Hub `repo_id`, expected backend, and whether the user needs `select_action` vs `predict_action_chunk`.
   - Done when: load path and API entry point are chosen before editing code.
2. **Load with auto-detection first** (local):

   ```python
   from physicalai.inference import InferenceModel
   model = InferenceModel("./exports/act_policy")
   ```

   - Done when: `model` constructs without explicit `backend=` when artifacts match a registered extension.

3. **Hub load** when the package is published:

   ```python
   model = InferenceModel.from_pretrained("OpenVINO/act-fp16-ov", revision="<commit-sha>")
   ```

   - Done when: revision is pinned for reproducibility when security or CI matters.

4. **Explicit backend** only when auto-detection is ambiguous:

   ```python
   model = InferenceModel("./exports/act_policy", backend="openvino", device="CPU")
   ```

5. **Validate structure** against the [manifest schema](https://github.com/openvinotoolkit/physicalai/blob/main/docs/reference/manifest-schema.md) and backend notes (`references/onnx.md`, `references/openvino.md`).
   - Done when: manifest, model file, and processor artifacts resolve under the export directory.
6. **Smoke inference** without owning robot timing:

   ```python
   model.reset()
   action = model.select_action(observation)
   ```

   - Done when: one forward pass succeeds on representative observation keys/shapes. For hardware loops, hand off to `pai-run-policy`.

## Validation loop

Construct `InferenceModel` from the export and run one representative fake observation without hardware. If a backend cannot be loaded, check its installed optional dependency and the [documented adapter support](https://github.com/openvinotoolkit/physicalai/blob/main/docs/reference/inference-api.md).

## Required checks

- Export directory contains backend model file(s) and `manifest.json` consistent with `docs/reference/manifest-schema.md`.
- Metadata input/output/feature names align with preprocessors in the manifest.
- Optional backends fail with clear install guidance (`onnxruntime`, OpenVINO).
- Do not claim support for an adapter absent from the installed Runtime package.
- Hub loads must not log tokens; prefer pinned `revision=` for production docs.

## References

- [Manifest schema](https://github.com/openvinotoolkit/physicalai/blob/main/docs/reference/manifest-schema.md) — canonical export/load contract.
- `references/onnx.md`, `references/openvino.md` — adapter constraints.
