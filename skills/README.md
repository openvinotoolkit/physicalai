# Physical AI Runtime agent skills

Runtime owns these canonical skills. **`skills/using/` is published to the OEP catalog; `skills/contributing/` stays in this repository.** A using skill must work with the installed Runtime package and a user's own project, without a Runtime source checkout. A contributing skill changes the Runtime implementation. A third-party robot plugin is external; a built-in driver under `src/physicalai/robot` is a contribution.

| Audience     | Skill                                                                              | Workflow                                       |
| ------------ | ---------------------------------------------------------------------------------- | ---------------------------------------------- |
| Using        | [`pai-load-policy`](using/pai-load-policy/SKILL.md)                                | Load Studio exports and Hub artifacts          |
| Using        | [`pai-configure-inference`](using/pai-configure-inference/SKILL.md)                | Configure processors and runners in a manifest |
| Using        | [`pai-configure-runtime`](using/pai-configure-runtime/SKILL.md)                    | Author trusted local Runtime recipes           |
| Using        | [`pai-run-policy`](using/pai-run-policy/SKILL.md)                                  | Run an exported policy on a robot              |
| Contributing | [`pai-add-robot-integration`](contributing/pai-add-robot-integration/SKILL.md)     | Add a built-in robot driver                    |
| Contributing | [`pai-add-camera-backend`](contributing/pai-add-camera-backend/SKILL.md)           | Add a built-in camera backend                  |
| Contributing | [`pai-add-inference-component`](contributing/pai-add-inference-component/SKILL.md) | Extend the inference component registry        |
| Contributing | [`pai-maintain-config`](contributing/pai-maintain-config/SKILL.md)                 | Change the shared config implementation        |

## Discovery and authoring

Canonical content is in `skills/<audience>/<name>/`. Both `.agents/skills/<name>` and `.claude/skills/<name>` are committed adapter symlinks, exposing **both audiences locally**; only `using/` is registered in the OEP catalog. Do not publish adapters as extra copies. Windows requires symlink support (Developer Mode or `core.symlinks true`); the sync script falls back to local junctions where available. After a rename, remove any stale junction reported by validation before rerunning sync.

- Use short, distinct action names with the shared `pai-` prefix. Directory and frontmatter must match (`^[a-z0-9]+(-[a-z0-9]+)*$`, at most 64 characters); source and catalog names are identical.
- Frontmatter uses portable `name`, `description`, and optional `license`; put future agent-specific information in `metadata:`. Describe what and when, and use a negative trigger to disambiguate neighboring skills.
- Keep source-level edits out of using workflows. Link existing docs at their canonical URL; only bundle references that provide unique skill knowledge. Write numbered steps with checkable outcomes and validate offline with fakes before touching hardware. Obtain approval before commands that install, mutate, or command a real device.
- Run at least three prompts from [`using/EVALUATION.md`](using/EVALUATION.md) or [`contributing/EVALUATION.md`](contributing/EVALUATION.md). Catalog RFC job families and quality gates are separate from this audience split.

```bash
# from the Runtime repo root after adding, moving, or renaming a skill
python3 .github/scripts/skills/agent_skills.py sync
python3 .github/scripts/skills/agent_skills.py validate
```

CI checks committed adapters; it does not regenerate them. Register new using skills from `skills/using` in `open-edge-platform/skills` after the source branch lands. Existing installed names must be removed and reinstalled under their new `pai-` names.
