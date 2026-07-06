# Config Schema Reference

> **Preview:** The config system is a planned API. The schemas below document the target config design.

Config files use `class_path` and `init_args` to describe explicit component construction.

## ComponentSpec

Direct class mode:

```yaml
class_path: package.module.ClassName
init_args:
  key: value
```

Registry mode:

```yaml
type: registered_name
key: value
```

The `ComponentSpec` fields are listed below.

| Field        | Type   | Description                             |
| ------------ | ------ | --------------------------------------- |
| `class_path` | string | Fully qualified import path             |
| `init_args`  | object | Constructor keyword arguments           |
| `type`       | string | Registered short name                   |
| extra fields | any    | Flat constructor args for registry mode |

The core rules are straightforward.

- A component spec must include either `class_path` or `type`.
- If both fields are present, `class_path` takes precedence.
- Nested component specs are instantiated recursively.

## RuntimeConfig

```yaml
runtime:
  robot:
    class_path: physicalai.robot.so101.SO101
    init_args:
      port: /dev/ttyACM0
  action_source:
    class_path: physicalai.runtime.PolicySource
    init_args:
      model:
        class_path: physicalai.inference.InferenceModel
        init_args:
          export_dir: ./exports/act_policy
      execution:
        class_path: physicalai.runtime.SyncExecution
  fps: 30
```

The most common runtime fields are listed below.

| Field                   | Type            | Description                                                          |
| ----------------------- | --------------- | -------------------------------------------------------------------- |
| `runtime.robot`         | `ComponentSpec` | Robot implementation                                                 |
| `runtime.action_source` | `ComponentSpec` | Action source — always explicit, e.g. `PolicySource`, `TeleopSource` |
| `runtime.fps`           | number          | Control loop frequency                                               |
| `runtime.cameras`       | mapping         | Optional camera components                                           |
| `runtime.callbacks`     | list            | Optional runtime callbacks                                           |

`action_source.init_args` depends on the chosen class — `PolicySource` takes `model` and `execution` (and optionally `task`, `action_queue`); `TeleopSource` takes `leader`.

## InferenceConfig

```yaml
model:
  class_path: physicalai.inference.InferenceModel
  init_args:
    export_dir: ./exports/act_policy
    backend: openvino
    device: CPU
```

The most common inference fields are listed below.

| Field                        | Type            | Description                |
| ---------------------------- | --------------- | -------------------------- |
| `model`                      | `ComponentSpec` | Inference model component  |
| `model.init_args.export_dir` | string          | Exported package directory |
| `model.init_args.backend`    | string          | Backend name or `auto`     |
| `model.init_args.device`     | string          | Backend device or `auto`   |

## Config vs Manifest

| Schema          | Use                                                      |
| --------------- | -------------------------------------------------------- |
| Workflow config | A workflow config describes a workflow before execution. |
| Manifest        | A manifest describes an exported package after export.   |
