---
name: physicalai-runtime-loading-exported-policies
description: Loads and validates policies exported from Physical AI Studio for Runtime deployment. Use when working on InferenceModel, InferenceModel.from_pretrained, manifest.json, adapter auto-detection (onnx, openvino), backend/device kwargs, Hugging Face Hub policy packages, or the Runtime side of the export/load contract that Studio produces with physicalai export.
license: Apache-2.0
---

# Loading Exported Policies

Runtime loads Studio export directories (or Hub snapshots that mirror them) through `InferenceModel` in `src/physicalai/inference/model.py`. Manifest parsing lives in `src/physicalai/inference/manifest.py`; backends register in `src/physicalai/inference/adapters/registry.py` (`onnx` → `.onnx`, `openvino` → `.xml`). Hub downloads use `src/physicalai/inference/utils/_hub.py`.

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

5. **Validate structure** against `references/export-load-contract.md` and backend notes (`references/onnx.md`, `references/openvino.md`).
   - Done when: manifest, model file, and processor artifacts resolve under the export directory.
6. **Smoke inference** without owning robot timing:

   ```python
   model.reset()
   action = model.select_action(observation)
   ```

   - Done when: one forward pass succeeds on representative observation keys/shapes. For hardware loops, hand off to `physicalai-runtime-running-policy-on-robot`.

## Validation loop

```bash
uv run pytest tests/unit/inference/test_model.py tests/unit/inference/test_manifest.py -q
```

For adapter changes, add or run targeted tests under `tests/unit/inference/`.

## Required checks

- Export directory contains backend model file(s) and `manifest.json` consistent with `docs/reference/manifest-schema.md`.
- Metadata input/output/feature names align with preprocessors in the manifest.
- Optional backends fail with clear install guidance (`onnxruntime`, OpenVINO).
- Do not document a backend unless an adapter is registered in `src/physicalai/inference/adapters/`.
- Hub loads must not log tokens; prefer pinned `revision=` for production docs.

## References

- `references/export-load-contract.md` — consumer-side contract (coordinate with Studio export skill).
- `references/onnx.md`, `references/openvino.md` — adapter constraints.
