---
name: pai-maintain-config
description: Maintains the Physical AI Runtime configuration package shared with Studio. Use when changing src/physicalai/config, export_config, Config serialization and instantiation, jsonargparse integration, or tests/unit/config. DO NOT USE FOR writing a trusted local runtime YAML recipe (use pai-configure-runtime).
license: Apache-2.0
---

# Maintain `physicalai.config`

Runtime owns `src/physicalai/config/`. Studio consumes `Config`, `FromConfig`, and `instantiate_obj` from Runtime rather than carrying a second implementation. Public recipe use belongs to `pai-configure-runtime`.

## Workflow

1. **Identify the API surface.** Use `Config` / `@export_config` for portable recipes; use jsonargparse for known, typed CLI construction. Read `docs/explanation/configuration.md` and a matching unit test before changing behavior.
   - Done when: the change is located in one owner and preserves existing recipes.
2. **Implement with the trust boundary intact.** Only instantiate `class_path` from trusted local config. Keep the depth limit, artifact-path validation, and `ConfigError` messages precise. Do not introduce a generic loader that executes untrusted YAML.
   - Done when: accepted and rejected inputs have explicit outcomes.
3. **Check downstream construction.** Round-trip an `@export_config` instance through `Config.from_instance(...).save()` and `instantiate()`; verify an existing Runtime YAML and Studio's use of `physicalai.config` still work.
   - Done when: round-trip and dependent construction tests pass without copying a config module into Studio.
4. **Update canonical documentation** under `docs/how-to/config/`, `docs/explanation/configuration.md`, or `docs/reference/config-schema.md` when the public contract changes.
   - Done when: an installed-package user can understand the new behavior without reading source.

## Verify

```bash
uv run pytest tests/unit/config/ -q
prek run ruff-check --all-files
python3 .github/scripts/skills/agent_skills.py validate
```
