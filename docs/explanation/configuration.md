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

## Typed configuration

Typed classes, workflows, and CLIs use jsonargparse directly:

```python
parser = ArgumentParser(exit_on_error=False)
parser.add_class_arguments(Trainer, "trainer")
parsed = parser.parse_object(document, defaults=False)
trainer = parser.instantiate(parsed).trainer
```

`physicalai.config.Config` is the portable `class_path` recipe and live-object
capture boundary. It is not a generic loader framework.

## Trust

Resolving `class_path` imports and executes local Python code. Only trusted
application or user-authored configuration belongs at the instantiation
boundary; peer metadata and transport control messages do not.
