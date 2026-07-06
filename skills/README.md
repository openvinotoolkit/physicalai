# Physical AI Runtime Agent Skills

Canonical, repo-specific agent skills for Physical AI Runtime. Skills are grouped by modular subpackage so agents load the right paths and commands.

## Buckets

| Bucket        | Path                       | Scope                                                                                                             |
| ------------- | -------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| **Inference** | [`inference/`](inference/) | `src/physicalai/inference/`: `InferenceModel`, manifests, adapters, preprocessors/postprocessors, runners         |
| **Capture**   | [`capture/`](capture/)     | `src/physicalai/capture/`: cameras, discovery, transport                                                          |
| **Runtime**   | [`runtime/`](runtime/)     | `src/physicalai/runtime/`, `src/physicalai/robot/`, `src/physicalai/cli/`: control loop, robots, `physicalai run` |

Each bucket has its own skill list and [`EVALUATION.md`](inference/EVALUATION.md) scenarios.

## Layout

```text
skills/
├── README.md                 # this file — global authoring rules
├── inference/
│   ├── README.md
│   ├── EVALUATION.md
│   └── <skill-name>/
│       ├── SKILL.md
│       └── references/
├── capture/
│   ├── README.md
│   ├── EVALUATION.md
│   └── <skill-name>/
└── runtime/
    ├── README.md
    ├── EVALUATION.md
    └── <skill-name>/
```

Client adapters are **committed symlinks** so a fresh clone works for agents (no setup step):

- `.claude/skills/<name>` → `../../skills/<bucket>/<name>`
- `.agents/skills/<name>` → `../../skills/<bucket>/<name>`

When you add or rename a skill, run `python3 .github/scripts/skills/agent_skills.py sync` and commit the updated symlinks. Pre-commit runs `sync` then `validate` when `skills/` changes. **CI only runs `validate`** on what is in the PR (it does not regenerate symlinks). GitHub may show `\ No newline at end of file` on symlink diffs; that is normal and harmless.

**Windows:** enable [Developer Mode](https://learn.microsoft.com/en-us/windows/apps/get-started/enable-your-device-for-development) or clone with `git config core.symlinks true` so Git checks out symlinks. If symlink creation fails, `agent_skills.py sync` falls back to a directory junction for local use.

## Authoring standard

These skills follow the open [Agent Skills](https://agentskills.io) format (originated by Anthropic, adopted by Claude Code, opencode, Codex, Gemini CLI, Cursor, and others). Write to the **portable core** so a skill works across every agent, not just one.

### Portability rules (non-negotiable)

- **Frontmatter = the portable subset only:** `name`, `description`, and optionally `license`. These three are understood everywhere.
- **Do not rely on vendor-only fields for behavior.** `disable-model-invocation` (Claude Code's "user-invoked" switch) is ignored by opencode/Codex/Gemini — never make a skill's correctness depend on it. Default to **model-invoked** skills.
- Put any client-specific hints in a `metadata:` map (opencode reads it; others ignore it), not in top-level fields.
- Use forward-slash paths; no Windows backslashes.

### `name`

- Lowercase, hyphenated, matches the directory name, regex `^[a-z0-9]+(-[a-z0-9]+)*$`.
- Prefix with the bucket id: `inference-`, `capture-`, or `runtime-`.
- Prefer gerund/verb phrasing (`physicalai-runtime-loading-exported-policies`, `physicalai-runtime-adding-a-camera-backend`).
- Must not contain the reserved words `anthropic` or `claude`.

### `description` (highest-leverage field)

This is what an agent matches against to decide whether to load the skill. Get it right.

- **Third person**, always: "Loads and validates…", never "I can…" / "You can…".
- State **what it does and when to use it**, with concrete triggers (CLI names, class names, file paths, backends).
- **One trigger per distinct branch** — collapse synonyms. Keep it under 1024 characters.

### Body

- **Be concise; assume the model is smart.** Only add what it can't infer. Every loaded token competes with real context. Keep `SKILL.md` well under 500 lines.
- **Ground every claim in real paths** for that bucket (`src/physicalai/...`) and real commands. No invented flags.
- **Numbered workflow steps, each ending in a checkable completion criterion** ("Done when: …") so the agent can tell done from not-done and doesn't stop early.
- **Match freedom to fragility:** high freedom (prose) for open tasks; low freedom ("run exactly this, don't add flags") for fragile ops like hardware or manifest loading.
- **Feedback loops** for quality-critical work: run → validate → fix → repeat, with the exact command.
- Consistent terminology; no time-sensitive text (use a collapsed "old patterns" section if needed).
- **Progressive disclosure:** push long detail into `references/*.md`, linked **one level deep** from `SKILL.md`. Add a table of contents to any reference file over ~100 lines.

### Before you ship

- Description has both _what_ and _when_, third person, distinct triggers.
- Every path/command verified against the current tree.
- Steps have checkable completion criteria.
- References are one level deep.
- Runs offline by default; tests needing downloads are marked `requires_download`.
- Pass at least three scenarios in the bucket's `EVALUATION.md`.

## Add a new skill

```bash
BUCKET=inference   # or capture, runtime
NAME=inference-my-workflow
mkdir -p "skills/$BUCKET/$NAME"
$EDITOR "skills/$BUCKET/$NAME/SKILL.md"
python3 .github/scripts/skills/agent_skills.py sync
```

Then dry-run the workflow in the skill end-to-end and fix any step where an agent could stall or guess.

## Org skills catalog (`open-edge-platform/skills`)

Runtime is the **source of truth** under `skills/`. The org [`open-edge-platform/skills`](https://github.com/open-edge-platform/skills) repo pulls from product repos on its own schedule. **Do not** open PRs from this repo into that catalog directly; register or update this product in the skills-repo automation instead.
