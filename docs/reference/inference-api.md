# Inference API Reference

## `InferenceModel`

```python
InferenceModel(
    export_dir: str | Path,
    policy_name: str | None = None,
    backend: str = "auto",
    device: str = "auto",
    runner: InferenceRunner | None = None,
    preprocessors: list[Preprocessor] | None = None,
    postprocessors: list[Postprocessor] | None = None,
    callbacks: list[Callback] | None = None,
    **adapter_kwargs,
)
```

The model can be constructed directly from an export directory or nested in a
trusted workflow `Config`.

## Constructors

```python
model = InferenceModel("./exports/act_policy")
```

### `from_pretrained`

```python
model = InferenceModel.from_pretrained(
    "OpenVINO/act-fp16-ov",
    revision="main",      # branch, tag, or commit SHA (optional)
    cache_dir=None,        # download cache directory (optional)
)
```

This downloads the policy package snapshot from the Hugging Face Hub and loads
it like a local export. Additional keyword arguments are forwarded to the
constructor (e.g. `backend`, `device`).

## Methods

### `select_action`

```python
action = model.select_action(observation)
```

This method returns one action.

### `predict_action_chunk`

```python
chunk = model.predict_action_chunk(observation)
```

This method returns a chunk of actions for runtime queueing.

### `reset`

```python
model.reset()
```

This method clears the runner state and the action cursor.

### `close`

```python
model.close()
```

This method releases backend resources.

## Observation

Observations are dictionaries of NumPy arrays.

```python
observation = {
    "state": joint_positions,
    "image.wrist": wrist_image,
}
```

The expected keys and shapes come from the exported package and its preprocessing pipeline.

## Remote Inference

```python
from physicalai.inference import InferenceModel, RemoteInferenceModel
from physicalai.inference.remote import InferenceServer

model = InferenceModel("./exports/act_policy")
server = InferenceServer(model, name="cell1-act")
server.serve_forever()  # Blocks until interrupted, then stops the server.
```

On the client, connect to the same deployment name:

```python
from physicalai.inference import RemoteInferenceModel

client = RemoteInferenceModel(name="cell1-act")
```

Both constructors accept keyword-only `host` (default `127.0.0.1`) and `port`
(default `None`). Omitting `port` derives the same deterministic TCP port from
`name` on both sides. Set an explicit port on both constructors to override it.
Read-only `host` and `port` properties expose the configured host and resolved
port. Zenoh endpoint strings are internal, not constructor arguments or public
properties.

`InferenceServer` requires `allow_all_interfaces=True` only for wildcard hosts
(`0.0.0.0` or `::`), not specific LAN or Tailscale addresses. Every non-loopback
listener logs an unauthenticated-exposure warning. The flag does not change the
configured host; without `host`, the server still binds `127.0.0.1`. See
[Run Inference Remotely](../how-to/runtime/run-inference-remotely.md) for server
lifecycle, CLI, SSH tunneling, and deployment configuration.

`expected_policy` and `expected_manifest_sha256` are client-side consistency
checks against metadata reported by the server. The manifest hash is not signed
and does not establish server authenticity or artifact integrity.
