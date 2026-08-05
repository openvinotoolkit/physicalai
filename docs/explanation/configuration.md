# Configuration

PhysicalAI uses constructor-shaped data so Python, YAML, the CLI, and Studio
can describe the same runtime graph.

## Construction Recipes

A `Config` names one class and the arguments needed to create it.

```yaml
class_path: physicalai.capture.UVCCamera
init_args:
  device: /dev/v4l/by-id/usb-Example_Camera-video-index0
  width: 640
  height: 480
```

Classes opt in to exporting this recipe with `@export_config`.
`Config.from_instance()`
captures construction intent rather than mutable runtime state: supplied
arguments are preserved, omitted defaults remain omitted, and connections or
open handles are never exported.

```text
live opted-in component -> Config.from_instance() -> JSON/YAML -> config.instantiate() -> new component
```

`Config.instantiate()` calls constructors only. The application remains responsible
for lifecycle methods such as `connect()` and `run()`.

## Workflow Documents

A workflow document adds command-level structure around components. For
example, `physicalai run` expects runtime constructor arguments under
`runtime:` and optional run-method arguments under `run:`. A bare exported
`RobotRuntime` Config is also accepted and reshaped by the loader.

Nested components retain the same `class_path` + `init_args` form, so a robot,
action source, model, camera map, and callbacks form one recursive recipe.

## Manifests

An inference manifest describes an exported policy package: artifacts,
features, processors, runners, and compatibility metadata. Its
`ComponentSpec` supports registry aliases and artifact handling and remains
separate from strict captured `Config` data.

Use workflow configuration for deployment intent. Use manifests for package
metadata produced during export.

## Generic instantiation (Training and plugins)

Runtime also ships helpers used heavily by Physical AI Studio and
`jsonargparse`-driven CLIs:

| Entry point                                                                 | Use when                                            |
| --------------------------------------------------------------------------- | --------------------------------------------------- |
| [`instantiate_obj`](../how-to/config/instantiate-objects.md)                | Generic loaders; class chosen by config             |
| [`FromConfig` / `@from_config`](../how-to/config/use-from-config.md)        | Class-level `from_yaml`, `from_dict`, `from_config` |
| [`Config` dataclass subclasses](../how-to/config/instantiate-components.md) | Typed hyperparameter objects with save/load         |

Strict captured recipes use the package-level :func:`instantiate`; generic
loaders use the package-level :func:`instantiate_obj`. Both are intentionally
available from one public import surface. Generic loader helpers are
implemented in `physicalai.config.loading`.

## Trust

Resolving `class_path` imports and executes local Python code. Only trusted
application or user-authored configuration belongs at the instantiation
boundary; peer metadata and transport control messages do not.
