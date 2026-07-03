# Inference agent skills

Skills for `src/physicalai/inference/`: loading exported policies, manifests, adapters, and the pre/post inference pipeline.

Run commands from the repo root unless noted otherwise (`uv sync`, `uv run pytest tests/unit/inference/...`).

## Skills

| Skill                                      | Covers                                                                                                        |
| ------------------------------------------ | ------------------------------------------------------------------------------------------------------------- |
| `inference-loading-exported-policies`      | `InferenceModel`, `Manifest`, Hub loads, ONNX/OpenVINO adapters, export/load contract (consumer side).        |
| `inference-configuring-inference-pipeline` | Preprocessors, postprocessors, `ComponentRegistry`, `instantiate_component`, manifest `type` vs `class_path`. |

New inference skills must pass at least three scenarios in [`EVALUATION.md`](EVALUATION.md).

## Add an inference skill

```bash
NAME=inference-my-workflow
mkdir -p "skills/inference/$NAME"
$EDITOR "skills/inference/$NAME/SKILL.md"
python3 .github/scripts/skills/agent_skills.py sync
```

Global authoring rules: [`../README.md`](../README.md).
