---
name: physicalai-runtime-working-with-config
description: Works with the shared physicalai.config package (Config recipes, FromConfig, instantiate_obj, export_config, YAML). Use when changing src/physicalai/config, wiring class_path configs, policy or runtime construction from YAML, or docs under docs/how-to/config and docs/explanation/configuration.md. Runtime owns this module; Studio consumes it.
license: Apache-2.0
---

# Working with `physicalai.config`

Runtime owns `src/physicalai/config/`. Do not add a parallel config tree in
Physical AI Studio.

## Workflow

1. **Pick the API** — strict transport/export uses `Config`, `@export_config`,
   and `instantiate()` from `physicalai.config`; generic CLI/YAML loaders use
   `instantiate_obj` and `FromConfig` from `physicalai.config` /
   `physicalai.config.loading` /
   `physicalai.config.mixin`; typed policy hyperparameters subclass
   `physicalai.config.base.Config`.
   - Done when: the change touches the module that matches the call site.
2. **Author or edit a recipe** — use `class_path` + `init_args` for dynamic
   dispatch; nest recipes inside `init_args` only for trusted local configs.
   See `docs/how-to/config/instantiate-components.md`.
   - Done when: YAML/dict round-trips through `validate_config` without
     `ConfigError`.
3. **Export live components** — decorate constructors with `@export_config`,
   capture with `Config.from_instance(obj)` or `to_config(obj)`, persist with
   `save_yaml`. Never feed network or untrusted payloads into `instantiate`.
   - Done when: exported YAML reloads via `instantiate()` on a test double.
4. **Use FromConfig in user code** — add `from_yaml` / `from_dict` via
   `FromConfig` or `@from_config` when a class should construct itself from
   config; prefer `instantiate_obj` in infrastructure that picks the target
   class from YAML alone (`docs/how-to/config/use-from-config.md`).
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
- `docs/how-to/config/instantiate-objects.md`
- `docs/how-to/config/use-from-config.md`
- `tests/unit/config/`
