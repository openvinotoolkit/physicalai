# Runtime agent skills

Skills for deployment orchestration: `src/physicalai/runtime/`, `src/physicalai/robot/`, and `src/physicalai/cli/` (`physicalai run`).

Run commands from the repo root (`uv run pytest tests/unit/runtime/...`, `physicalai run --help`).

## Skills

| Skill                                           | Covers                                                                          |
| ----------------------------------------------- | ------------------------------------------------------------------------------- |
| `physicalai-runtime-running-policy-on-robot`    | `PolicyRuntime`, execution modes, YAML config, `physicalai run`, callbacks.     |
| `physicalai-runtime-adding-a-robot-integration` | `Robot` protocol, concrete drivers (SO-101, WidowX), `verify`, optional extras. |

New runtime skills must pass at least three scenarios in [`EVALUATION.md`](EVALUATION.md).

## Add a runtime skill

```bash
NAME=runtime-my-workflow
mkdir -p "skills/runtime/$NAME"
$EDITOR "skills/runtime/$NAME/SKILL.md"
python3 .github/scripts/skills/agent_skills.py sync
```

Global authoring rules: [`../README.md`](../README.md).
