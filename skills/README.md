# Physical AI Runtime agent skills

Runtime owns these canonical skills. **`skills/using/` is the set intended for the OEP catalog; `skills/contributing/` stays in this repository.** A using skill must work with the installed Runtime package and a user's own project, without a Runtime source checkout. A contributing skill changes the Runtime implementation. A third-party robot plugin is external; a built-in driver under `src/physicalai/robot` is a contribution.

| Audience     | Skill                                                                      | Workflow                                         |
| ------------ | -------------------------------------------------------------------------- | ------------------------------------------------ |
| Using        | [`pai-load-policy`](using/pai-load-policy/SKILL.md)                        | Load Studio exports and Hub artifacts            |
| Using        | [`pai-configure-inference`](using/pai-configure-inference/SKILL.md)        | Configure processors and runners in a manifest   |
| Using        | [`pai-configure-runtime`](using/pai-configure-runtime/SKILL.md)            | Author trusted local Runtime recipes             |
| Using        | [`pai-run-policy`](using/pai-run-policy/SKILL.md)                          | Run an exported policy on a robot                |
| Using        | [`pai-create-robot-plugin`](using/pai-create-robot-plugin/SKILL.md)        | Develop and integrate a third-party robot plugin |
| Contributing | [`add-robot-integration`](contributing/add-robot-integration/SKILL.md)     | Add a built-in robot driver                      |
| Contributing | [`add-camera-backend`](contributing/add-camera-backend/SKILL.md)           | Add a built-in camera backend                    |
| Contributing | [`add-inference-component`](contributing/add-inference-component/SKILL.md) | Extend the inference component registry          |
| Contributing | [`maintain-config`](contributing/maintain-config/SKILL.md)                 | Change the shared config implementation          |

## Discovery and authoring

Canonical content is in `skills/<audience>/<name>/`. Both `.agents/skills/<name>` and `.claude/skills/<name>` are committed adapter symlinks, exposing **both audiences locally**; only `using/` is intended for the OEP catalog. Do not publish adapters as extra copies. Windows requires symlink support (Developer Mode or `core.symlinks true`); the sync script falls back to local junctions where available. After a rename, remove any stale junction reported by validation before rerunning sync.

- Name skills with short, distinct actions; directory and frontmatter must match (`^[a-z0-9]+(-[a-z0-9]+)*$`, at most 64 characters). **Prefix `pai-` if and only if the skill is published:** `using/` names are installed into a flat, shared namespace next to other products' skills, so they carry the prefix and keep the same name in source and catalog. `contributing/` names never leave this repository and stay unprefixed (for example `add-camera-backend`). `agent_skills.py validate` enforces this; coordinate any `using/` rename with the OEP catalog.
- Frontmatter uses portable `name`, `description`, and optional `license`; put future agent-specific information in `metadata:`. Describe what and when, and use a negative trigger to disambiguate neighboring skills.
- Keep source-level edits out of using workflows. Link existing docs at their canonical URL; only bundle references that provide unique skill knowledge. Write numbered steps with checkable outcomes and validate offline with fakes before touching hardware. Obtain approval before commands that install, mutate, or command a real device.
- Run at least three prompts from [`using/EVALUATION.md`](using/EVALUATION.md) or [`contributing/EVALUATION.md`](contributing/EVALUATION.md). Catalog RFC job families and quality gates are separate from this audience split.

```bash
# from the Runtime repo root after adding, moving, or renaming a skill
python3 .github/scripts/skills/agent_skills.py sync
python3 .github/scripts/skills/agent_skills.py validate
```

CI checks committed adapters; it does not regenerate them. Register new using skills from `skills/using` in `open-edge-platform/skills` after the source branch lands. Existing installed names must be removed and reinstalled under their new `pai-` names.
