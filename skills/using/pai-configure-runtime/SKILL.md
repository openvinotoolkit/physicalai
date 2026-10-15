---
name: pai-configure-runtime
description: Configures Physical AI Runtime objects with physicalai.config and YAML recipes. Use when writing class_path/init_args for robots, cameras, runtime or inference components, exporting a local object with @export_config, or choosing jsonargparse for typed construction. DO NOT USE FOR editing src/physicalai/config or untrusted remote recipes.
license: Apache-2.0
---

# Working with `physicalai.config`

Runtime owns `physicalai.config`; Studio consumes it as a package. Use trusted local configuration files only—`class_path` can import and instantiate Python objects.

## Workflow

1. **Pick the API**
   - Portable YAML recipes (robots, cameras, exported components): `Config` and
     `@export_config`.
   - Known Python types (trainers, dataclass configs, CLI models): jsonargparse
     (`ArgumentParser`, `add_class_arguments`, `parse_object`, `instantiate`).
2. **Author a recipe** — `class_path` + `init_args`; nest recipes only for
   trusted local config. See [instantiating components](https://github.com/openvinotoolkit/physicalai/blob/main/docs/how-to/config/instantiate-components.md).
   - Done when: dict/YAML passes validation without `ConfigError`.
3. **Export live objects** — `@export_config`, then `Config.from_instance(obj)`
   and `Config.save()`. Only trusted local sources.
   - Done when: the saved YAML reloads with `instantiate()` in a local smoke test.
4. **Typed construction** — use jsonargparse in your own CLI or app for typed arguments; avoid a generic untrusted YAML loader.
5. **Verify** — load the recipe in a short project-local smoke test with fake hardware, then use it with `physicalai run` only after reviewing every `class_path`.

## Validation loop

Validate the local YAML against the [config schema](https://github.com/openvinotoolkit/physicalai/blob/main/docs/reference/config-schema.md) and round-trip a sample object in the user's Python environment.

## Required checks

- Deeply nested config is rejected with `ConfigError` rather than silently truncated.
- `class_path` comes only from trusted local config; never execute a downloaded recipe without review.
- Studio and external plugins import `Config` from Runtime rather than copying the module.

## References

- [Configuration explained](https://github.com/openvinotoolkit/physicalai/blob/main/docs/explanation/configuration.md)
- [Instantiate components](https://github.com/openvinotoolkit/physicalai/blob/main/docs/how-to/config/instantiate-components.md)
- [Config schema](https://github.com/openvinotoolkit/physicalai/blob/main/docs/reference/config-schema.md)
