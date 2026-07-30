---
name: physicalai-runtime-working-with-config
description: Works with the shared physicalai.config package (Config recipes, FromConfig, instantiate_obj, export_config, YAML). Use when changing src/physicalai/config, wiring class_path configs, policy or runtime construction from YAML, or docs under docs/how-to/config and docs/explanation/configuration.md. Runtime owns this module; Studio consumes it.
license: Apache-2.0
---

# Working with `physicalai.config`

Runtime owns `src/physicalai/config/`. Do not add a parallel config tree in
Physical AI Studio.

## Entry points

| API | Module | Use |
| --- | ------ | --- |
| `Config`, `export_config`, strict `instantiate` | `physicalai.config` | Captured construction recipes, transport export |
| `instantiate_obj`, `import_class` | `physicalai.config.instantiate` | Generic Lightning/CLI loaders |
| `FromConfig`, `from_config` | `physicalai.config.mixin` | `from_yaml`, `from_dict`, `from_config` on classes |
| Typed dataclass `Config` | `physicalai.config.base` | Policy hyperparameters, save/load |

## Docs (canonical)

- [`docs/explanation/configuration.md`](../../docs/explanation/configuration.md)
- [`docs/how-to/config/instantiate-components.md`](../../docs/how-to/config/instantiate-components.md)
- [`docs/how-to/config/instantiate-objects.md`](../../docs/how-to/config/instantiate-objects.md)
- [`docs/how-to/config/use-from-config.md`](../../docs/how-to/config/use-from-config.md)

## Tests

- `tests/unit/config/` — unit tests for the package

## Trust

Only trusted local YAML/dict configs reach `class_path` resolution. See
`docs/development/security.md` and the trust section in
`docs/explanation/configuration.md`.
