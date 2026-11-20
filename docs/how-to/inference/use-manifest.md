# Use a Manifest

An exported policy package contains a `manifest.json` file.

```text
exports/act_policy/
├── manifest.json
├── model.xml
└── stats.safetensors
```

In most cases, you load the package through `InferenceModel`.

```python
model = InferenceModel("./exports/act_policy")
```

If you need to inspect metadata directly, load the manifest itself.

```python
from physicalai.inference.manifest import Manifest

manifest = Manifest.load("./exports/act_policy/manifest.json")
print(manifest.model.runner)
print(manifest.model.artifacts)
```

## Enable observation history

Opt in to image and state windows collected by `PolicySource` in `manifest.json`
(shown as YAML):

```yaml
version: "1.1"
model:
  observation_history:
    n_obs_steps: 3
    stride: 2
    preprocessor: { type: stack_observation_history } # Optional; this is the default.
```

`PolicySource` automatically consumes this typed config. Without it, behavior is
unchanged; the existing `model.n_obs_steps` metadata does not activate history.
`SinglePass` is unchanged. Standalone `InferenceModel.select_action` and
`predict_action_chunk` do not collect this history: callers must supply complete
windows themselves.

History requires manifest version `1.1` or greater. New manifests default to
`1.1`; existing `1.0` manifests without history continue to load unchanged.
This check applies in the updated Runtime; it cannot make older Runtime versions
reject fields they do not recognize.

- History records observations delivered to `PolicySource.update` at policy-cycle
  cadence, not interpolated robot substeps. `stride` counts received updates, not
  model calls; there is no independent camera capture. Any FPS-derived timing is
  nominal: history does not resample by timestamp.
- Windows run oldest to newest, repeating the first frame for missing startup
  history. `PolicySource.reset()` clears history even with `reset_model=False`.
- By default, images and state are stacked at inference time before policy preprocessing.
  Image axis order is preserved:
  `(B, H, W, C)` becomes `(B, T, H, W, C)`, and `(B, C, H, W)` becomes
  `(B, T, C, H, W)`. State changes from `(B, D)` to `(B, T, D)` using the
  same observation indices, stride, and startup padding. Task remains current
  and non-temporal.
- The exported model and preprocessors must accept these temporal inputs.
- `PolicySource.to_model_input(...)` previews a window without recording history.
  Its result can be passed to `PolicySource.warmup(...)`; only `update` records
  observations.

### Stacking and compatibility

`observation_history.preprocessor` is an optional `ComponentSpec` or `null`,
defaulting to `{type: stack_observation_history}`. `InferenceModel` prepends the
declared formatter before the regular preprocessor list, including an explicitly
supplied list. A formatter can use a registered `type` or `class_path` with
`init_args`. Do not also declare the same formatter in the regular list.

To leave history formatting to the remaining pipeline:

```yaml
model:
  observation_history:
    n_obs_steps: 3
    preprocessor: null
```

`null` disables only the automatic formatter, not history collection or temporal
sampling. The remaining pipeline receives the full selected history as raw fixed
tuples and is responsible for formatting it. Omitting `observation_history` (or
setting that whole field to `null`) disables history and leaves preprocessing
unchanged.

Deferred inputs contain fixed tuples of owned, read-only image/state frames, not
the live history buffer. Executors pass these tuples through unchanged; the
default formatter stacks them along axis `1` into the same `B,T,...` arrays described
above. Task inputs are unchanged. Incoming frames still need copying on every
update because camera buffers may be reused. No new threads are added: normal
async/RTC inference stacks in the existing workers, while sync stacks in the
control thread only when inference runs; warmup follows its execution strategy.

`on_predict_start` callbacks run **before** preprocessors and therefore see tuples
with history enabled, even when the formatter is `null`. Direct inference callers can still
supply complete array windows, which `StackObservationHistory` passes through
unchanged. Remote serialization does not currently support these tuple frames;
setting the formatter to `null` does not fix that limitation.

## Declare inference callbacks

Callbacks use the same `ComponentSpec` syntax and declaration order as
preprocessors and postprocessors. Built-in callbacks can use their registered
short name with flat constructor arguments, or a fully qualified class path
with `init_args`:

```yaml
model:
  callbacks:
    - type: rldx1_vtc
      video_length: 4
      video_stride: 2
    - class_path: physicalai.inference.callbacks.LatencyMonitor
      init_args:
        window_size: 100
```

The RLDX-1 callback above creates temporal image windows before preprocessing
without Python-side callback wiring. Manifest callbacks receive `on_load`,
`on_predict_start`, `on_predict_end`, and `on_reset` lifecycle events.

When constructing `InferenceModel`, an explicitly supplied `callbacks` list
takes precedence over the manifest, including an explicitly supplied empty
list. Invalid, unknown, or incompatible callback specifications fail while the
model is loading.

Use manifests to describe exported artifacts. Use workflow config to author training, inference, or runtime workflows before execution.
