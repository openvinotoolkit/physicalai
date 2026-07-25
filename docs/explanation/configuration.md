# Configuration

PhysicalAI uses constructor-shaped data so Python, YAML, the CLI, and Studio
can describe the same runtime graph.

## Construction Recipes

A `ComponentConfig` names one class and the arguments needed to create it.

```yaml
class_path: physicalai.capture.UVCCamera
init_args:
  device: /dev/v4l/by-id/usb-Example_Camera-video-index0
  width: 640
  height: 480
```

Classes opt in to exporting this recipe with `@export_config`. `to_config()`
captures construction intent rather than mutable runtime state: supplied
arguments are preserved, omitted defaults remain omitted, and connections or
open handles are never exported.

```text
live opted-in component -> to_config() -> JSON/YAML -> instantiate() -> new component
```

`instantiate()` calls constructors only. The application remains responsible
for lifecycle methods such as `connect()` and `run()`.

## Workflow Documents

A workflow document adds command-level structure around components. For
example, `physicalai run` expects runtime constructor arguments under
`runtime:` and optional run-method arguments under `run:`. A bare exported
`RobotRuntime` ComponentConfig is also accepted and reshaped by the loader.

Nested components retain the same `class_path` + `init_args` form, so a robot,
action source, model, camera map, and callbacks form one recursive recipe.

## Manifests

An inference manifest describes an exported policy package: artifacts,
features, processors, runners, and compatibility metadata. Its
`ComponentSpec` supports registry aliases and artifact handling and remains
separate from strict captured `ComponentConfig` data.

Use workflow configuration for deployment intent. Use manifests for package
metadata produced during export.

## Trust

Resolving `class_path` imports and executes local Python code. Only trusted
application or user-authored configuration belongs at the instantiation
boundary; peer metadata and transport control messages do not.
