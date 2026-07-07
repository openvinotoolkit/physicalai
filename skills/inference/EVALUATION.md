# Skill Evaluation Scenarios

Use these prompts to test whether an agent correctly invokes and follows each inference skill. Run the agent from the repo root with no extra hints beyond the prompt.

Expected rubric per scenario:

- **Activates the right skill** — loaded `SKILL.md` matches the topic.
- **Uses real paths and commands** — references `src/physicalai/inference/...`, `uv run pytest tests/unit/inference/...` as documented.
- **Follows the workflow checklist** — does not skip Required checks / validation loop steps.
- **Produces a checkable artifact** — a command run, a file written, or a test result.

## `physicalai-runtime-loading-exported-policies`

### Scenario 1: Load a local OpenVINO export

> "Load the policy in `./exports/act_policy` with InferenceModel and run one `select_action` on a fake observation dict. Use auto-detection if possible."

Expected behavior:

- Uses `InferenceModel("./exports/act_policy")` or explicit `backend="openvino"` only if needed.
- Calls `reset()` then `select_action`.
- Does not implement a custom robot loop.
- Points to manifest/adapter paths under `src/physicalai/inference/`.

### Scenario 2: Hub load with pinned revision

> "Load `OpenVINO/act-fp16-ov` from the Hub with a pinned commit SHA and explain what files are required in the snapshot."

Expected behavior:

- Uses `InferenceModel.from_pretrained(..., revision="<sha>")`.
- Mentions `manifest.json` and model artifacts per export-load contract.
- Marks full Hub test as `requires_download` if it cannot run offline.

### Scenario 3: Debug missing backend dependency

> "InferenceModel fails when loading a `.onnx` file on a machine without ONNX Runtime. What should the error path and docs say?"

Expected behavior:

- References `src/physicalai/inference/adapters/onnx.py` and `onnxruntime` dependency.
- Does not invent unsupported backends.

## `physicalai-runtime-configuring-inference-pipeline`

### Scenario 4: Add StatsNormalizer to a manifest

> "Extend a manifest YAML to normalize observations using `stats.safetensors` in the export directory. Use the registered pattern from the how-to docs."

Expected behavior:

- Shows preprocessor spec with `type` or `class_path: physicalai.inference.preprocessors.StatsNormalizer`.
- Mentions `resolve_artifact` / artifact relative paths.
- Suggests `uv run pytest tests/unit/inference/preprocessors -q`.

### Scenario 5: Register a short type name

> "We added a new preprocessor class. Register a manifest-friendly short name in component_factory and add a unit test."

Expected behavior:

- Edits `src/physicalai/inference/component_factory.py` registry without breaking `_MAX_COMPONENT_DEPTH`.
- Adds test under `tests/unit/inference/preprocessors/`.

### Scenario 6: Fix processor order bug

> "Actions are denormalized before the runner runs. Fix the manifest pipeline order and verify with tests."

Expected behavior:

- Correct order: preprocessors → runner → postprocessors.
- Runs manifest or model unit tests.
