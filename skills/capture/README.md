# Capture agent skills

Skills for `src/physicalai/capture/`: camera implementations, discovery, frames, and shared transport.

Run commands from the repo root (`uv run pytest tests/unit/capture/...`).

## Skills

| Skill                             | Covers                                                                                          |
| --------------------------------- | ----------------------------------------------------------------------------------------------- |
| `capture-adding-a-camera-backend` | New camera types, `create_camera`, `Camera` base class, optional extras, unit tests with fakes. |

New capture skills must pass at least three scenarios in [`EVALUATION.md`](EVALUATION.md).

## Add a capture skill

```bash
NAME=capture-my-workflow
mkdir -p "skills/capture/$NAME"
$EDITOR "skills/capture/$NAME/SKILL.md"
python3 .github/scripts/skills/agent_skills.py sync
```

Global authoring rules: [`../README.md`](../README.md).
