---
name: physicalai-runtime-working-with-config
description: Works with the shared physicalai.config package (Config recipes, export_config, jsonargparse, YAML). Use when changing src/physicalai/config, wiring class_path configs, policy or runtime construction from YAML, or docs under docs/how-to/config and docs/explanation/configuration.md. Runtime owns this module; Studio consumes it.
license: Apache-2.0
---

# Working with `physicalai.config`

Runtime owns `src/physicalai/config/`. Do not add a parallel config tree in
Physical AI Studio.

## Workflow

1. **Pick the API** — portable transport/export uses `Config` and
   `@export_config`; typed classes and workflows use jsonargparse directly.
   - Done when: the change touches the module that matches the call site.
2. **Author or edit a recipe** — use `class_path` + `init_args` for dynamic
   dispatch; nest recipes inside `init_args` only for trusted local configs.
   See `docs/how-to/config/instantiate-components.md`.
   - Done when: YAML/dict round-trips through `validate_config` without
     `ConfigError`.
3. **Export live components** — decorate constructors with `@export_config`,
   capture with `Config.from_instance(obj)` and persist with `Config.save()`.
   Never feed network or untrusted payloads into construction.
   - Done when: exported YAML reloads via `instantiate()` on a test double.
4. **Use jsonargparse for typed construction** — package-owned parsers define
   known classes, workflows, and CLI/file behavior. Do not add generic loader
   utilities under `physicalai.config`.
5. **Document and test** — update `docs/explanation/configuration.md` or the
   relevant how-to under `docs/how-to/config/`; extend `tests/unit/config/`.

## Validation loop

```bash
uv run pytest tests/unit/config/ -q
prek run ruff-check --all-files
```

## Required checks

- Recursive walkers respect `_MAX_CONFIG_DEPTH` and raise `ConfigError` on overflow.
- `class_path` strings resolve only from trusted local config (see
  `docs/development/security.md`).
- Studio must import `Config` from the runtime package, not ship
  `library/src/physicalai/config/`.

## References

- `docs/explanation/configuration.md`
- `docs/how-to/config/instantiate-components.md`
- `tests/unit/config/`
