---
name: pai-add-inference-component
description: Adds a built-in preprocessor, postprocessor, or runner to Physical AI Runtime. Use when editing src/physicalai/inference component classes, ComponentRegistry, or their unit tests so a manifest can instantiate a new type. DO NOT USE FOR configuring an existing manifest in a customer project (use pai-configure-inference).
license: Apache-2.0
---

# Add an Inference Component

Runtime owns built-in processors and runners under `src/physicalai/inference/`; the manifest contract is described in `docs/reference/manifest-schema.md`. Keep this source-level workflow separate from configuring an exported policy with `pai-configure-inference`.

## Workflow

1. **Choose the component boundary.** Inspect `src/physicalai/inference/component_factory.py` and an existing preprocessor, postprocessor, or runner. Confirm whether the requested behavior belongs before or after the runner.
   - Done when: the input/output contract and manifest location are identified.
2. **Implement the component.** Subclass the matching Runtime base in `src/physicalai/inference/preprocessors/`, `postprocessors/`, or `runners/`. Resolve artifacts through the existing export-directory resolver; reject absolute and parent-traversal paths.
   - Done when: it works on representative tensors without depending on a robot or network service.
3. **Expose it to trusted manifests.** Register a short `type` with `ComponentRegistry` only when a stable built-in alias is useful. Otherwise require the existing `class_path` / `init_args` path. Keep nested specs under `_MAX_COMPONENT_DEPTH`.
   - Done when: `InferenceModel` can construct the component from a minimal local manifest.
4. **Test order and compatibility.** Extend `tests/unit/inference/preprocessors/` or `postprocessors/` and `tests/unit/inference/test_manifest.py` as appropriate. Check preprocessor → runner → postprocessor order and preserve old manifest behavior.
   - Done when: focused tests pass and one fake-observation inference returns the expected action shape.
5. **Document public additions** in `docs/reference/inference-api.md` or `docs/how-to/inference/configure-pre-post-processing.md`.
   - Done when: users can select the new type without reading source.

## Verify

```bash
uv run pytest tests/unit/inference/preprocessors tests/unit/inference/postprocessors tests/unit/inference/test_manifest.py -q
python3 .github/scripts/skills/agent_skills.py validate
```
