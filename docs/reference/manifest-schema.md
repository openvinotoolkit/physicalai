# Manifest Schema Reference

The on-disk manifest is `manifest.json`. YAML is shown here for readability.

## Top Level

```yaml
format: policy_package
version: "1.1"
policy: {}
model: {}
hardware: {}
```

Fields:

| Field      | Type   | Description                               |
| ---------- | ------ | ----------------------------------------- |
| `format`   | string | Manifest format, usually `policy_package` |
| `version`  | string | Manifest schema version                   |
| `policy`   | object | Policy identity and source metadata       |
| `model`    | object | Artifacts and inference pipeline          |
| `hardware` | object | Expected robot and camera metadata        |

## Policy

```yaml
policy:
  name: pi05
  source:
    repo_id: physical-ai/example
    class_path: physicalai.policies.pi05.Pi05
```

## Model

```yaml
model:
  n_obs_steps: 1
  artifacts:
    openvino: model.xml
  runner:
    type: action_chunking
    chunk_size: 50
  preprocessors:
    - type: normalize
      artifact: stats.safetensors
  postprocessors:
    - type: denormalize
      artifact: stats.safetensors
  callbacks:
    - type: rldx1_vtc
      video_length: 4
      video_stride: 2
```

Fields:

| Field                 | Type                             | Description                                                           |
| --------------------- | -------------------------------- | --------------------------------------------------------------------- |
| `n_obs_steps`         | integer                          | Expected observation count (metadata only; does not activate history) |
| `observation_history` | `ObservationHistorySpec` or null | Image/state history config to pass explicitly to `PolicySource`       |
| `artifacts`           | mapping                          | Logical artifact name to file path                                    |
| `runner`              | `ComponentSpec`                  | Inference runner                                                      |
| `preprocessors`       | list[`ComponentSpec`]            | Preprocessing pipeline                                                |
| `postprocessors`      | list[`ComponentSpec`]            | Postprocessing pipeline                                               |
| `callbacks`           | list[`ComponentSpec`]            | Inference lifecycle callbacks                                         |

### Observation History

```yaml
version: "1.1"
model:
  observation_history:
    n_obs_steps: 3
    stride: 2
  preprocessors:
    - type: stack_observation_history
    # Policy preprocessing follows.
```

`ObservationHistorySpec` has only two fields: a required positive integer
`n_obs_steps` and a positive integer `stride` defaulting to `1`. It has no
`preprocessor` field. `PolicySource` does not discover this config from the
manifest; pass it explicitly as
`observation_history=model.manifest.model.observation_history`. Omitting that
argument or passing `None` leaves history disabled. Omitting the manifest field
or setting it to `null` provides no history config.
`SinglePass` is unchanged. The separate `model.n_obs_steps` remains metadata only.
Images and state share the temporal window; task inputs remain non-temporal.
History requires version `1.1` or greater. New manifests default to `1.1`;
existing `1.0` manifests without history retain their previous behavior.

`InferenceModel` does not automatically insert a history formatter. User/Studio
exports must put `type: stack_observation_history` explicitly in the normal
`model.preprocessors` list before policy processing, or choose a custom formatter
there. It accepts a registered `type` or `class_path` with `init_args`. An explicitly
supplied `preprocessors` list must likewise format the full selected history
received as raw fixed tuples. Without history, the preprocessing pipeline is unchanged.

`stack_observation_history` stacks fixed tuples of owned, read-only image/state frames into
`B,T,...` arrays along axis `1`; complete arrays and task inputs pass through
unchanged. `on_predict_start` callbacks run first and must be tuple-compatible;
do not add another history-stacking callback. Remote serialization does not
currently support these tuple frames; changing the formatter does not fix this.
See [usage and compatibility limits](../how-to/inference/use-manifest.md#enable-observation-history).

## Hardware

```yaml
hardware:
  robots:
    - name: main
      type: SO101
      state:
        shape: [6]
        dtype: float32
        order: [shoulder, elbow, wrist, gripper, lift, rotate]
      action:
        shape: [6]
        dtype: float32
  cameras:
    - name: wrist
      shape: [3, 224, 224]
      dtype: uint8
```

Robot and camera specs are metadata only. The runtime config still selects the concrete hardware classes.
