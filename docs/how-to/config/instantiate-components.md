# Instantiate Components

A `ComponentConfig` is a construction recipe with one importable class and
its supplied constructor arguments.

```yaml
class_path: physicalai.capture.UVCCamera
init_args:
  device: /dev/video0
  width: 640
  height: 480
```

Use `instantiate()` to construct trusted local configuration. It calls the
constructor but does not call lifecycle methods such as `connect()`, `run()`,
or `start()`.

```python
from physicalai.config import instantiate

camera = instantiate({
    "class_path": "physicalai.capture.UVCCamera",
    "init_args": {"device": "/dev/video0", "width": 640, "height": 480},
})
camera.connect()
```

## Export a live component

Classes opt in with `@export_config`. The decorator remembers only arguments
the caller supplied, so omitted constructor defaults remain omitted.

```python
from physicalai.capture import UVCCamera
from physicalai.config import to_config

camera = UVCCamera(device="/dev/video0", width=640, height=480)
config = to_config(camera)
```

Nested opted-in components use the same shape recursively. Configs contain
JSON values only; paths become strings, tuples become lists during export,
and non-finite floats are rejected.

```python
from physicalai.config import save_yaml

save_yaml(camera, "camera.yaml")
```

## Trust boundary

`class_path` selects Python code to import and execute. Pass only trusted
application or user-authored configuration to `instantiate()`. Do not
instantiate robot metadata, camera metadata, shared-memory messages, or other
peer-controlled payloads.

Inference manifest `ComponentSpec` is a separate compatibility schema with
registry aliases and artifact handling. Use `physicalai.config` for captured
construction recipes; use the manifest APIs when loading exported policy
metadata.
