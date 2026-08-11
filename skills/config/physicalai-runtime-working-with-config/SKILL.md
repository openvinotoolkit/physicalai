---
name: physicalai-runtime-working-with-config
description: Works with physicalai.config (Config recipes, export_config, jsonargparse, YAML). Use when editing src/physicalai/config, class_path YAML, runtime or policy construction, or docs under docs/how-to/config and docs/explanation/configuration.md. Runtime owns this module; Studio imports it from physicalai.
license: Apache-2.0
---

# Working with `physicalai.config`

Runtime owns `src/physicalai/config/`. Physical AI Studio should import this
package from `physicalai`, not copy it under `library/`.

## Workflow

1. **Pick the API**
   - Portable YAML recipes (robots, cameras, exported components): `Config` and
     `@export_config`.
   - Known Python types (trainers, dataclass configs, CLI models): jsonargparse
     (`ArgumentParser`, `add_class_arguments`, `parse_object`, `instantiate`).
2. **Author a recipe** — `class_path` + `init_args`; nest recipes only for
   trusted local config. See `docs/how-to/config/instantiate-components.md`.
   - Done when: dict/YAML passes validation without `ConfigError`.
3. **Export live objects** — `@export_config`, then `Config.from_instance(obj)`
   and `Config.save()`. Only trusted local sources.
   - Done when: saved YAML reloads with `instantiate()` in tests.
4. **Typed construction** — use jsonargparse in the owning package (runtime CLI,
   inference, and so on). Avoid new generic loaders under `physicalai.config`.
5. **Document and test** — update `docs/explanation/configuration.md` or
   `docs/how-to/config/`; extend `tests/unit/config/`.

## Validation loop

```bash
uv run pytest tests/unit/config/ -q
prek run ruff-check --all-files
```

## Required checks

- Deeply nested config hits `_MAX_CONFIG_DEPTH` and raises `ConfigError`.
- `class_path` comes only from trusted local config (`docs/development/security.md`).
- Studio imports `Config` from runtime, not a duplicate tree in the library.

## References

- `docs/explanation/configuration.md`
- `docs/how-to/config/instantiate-components.md`
- `tests/unit/config/`
