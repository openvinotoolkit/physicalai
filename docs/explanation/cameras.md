# Cameras

Cameras expose a small capture interface for connecting to a device and retrieving frames.

```python
camera.connect()
frame = camera.read_latest()
camera.disconnect()
```

## Read Modes

| Method          | Behavior                      | Use                                 |
| --------------- | ----------------------------- | ----------------------------------- |
| `read()`        | next frame, blocking          | recording or complete frame streams |
| `read_latest()` | newest frame, non-blocking    | real-time control                   |
| `async_read()`  | async wrapper around `read()` | async applications                  |

## Runtime Use

Control loops usually care more about freshness than completeness.

```python
observation["image.wrist"] = wrist_camera.read_latest()
```

Camera instances are not thread-safe. Use one thread per camera instance or add external synchronization.

## SharedCamera

`SharedCamera` lets one publisher subprocess own a camera's exclusive hardware
connection while any number of subscribers read frames over iceoryx2. It
satisfies the same `Camera` protocol as a direct driver.

Construction is ComponentConfig-only (`from_config` / `from_camera`), matching
`SharedRobot`. Prefer `from_config` (or YAML) when sharing. `from_camera` is
export-only sugar after disconnect — it does not hand off an open device into
the child; the publisher always opens fresh. Never keep a direct camera
connected to the same device while sharing: another connected holder causes
open failure.

```python
from physicalai.capture import SharedCamera
from physicalai.config import to_config

# Prefer config-only / disconnected export — no live direct camera held open
shared = SharedCamera.from_config(
    {
        "class_path": "physicalai.capture.UVCCamera",
        "init_args": {"device": "/dev/video0", "backend": "v4l2"},
    },
)
# Equivalent after to_config(disconnected_driver):
# shared = SharedCamera.from_config(to_config(driver))
shared.connect()

# Safe to read from multiple threads/processes
frame = shared.read_latest()
```

`create_camera(..., shared=True)` remains a convenience for shareable built-ins
(`uvc`, `realsense`, `basler`) that packs a type into `SharedCamera(camera=...)`.

`SharedCamera` is the recommended approach for production deployments where multiple consumers need camera frames. It avoids the need for manual synchronization and handles frame distribution efficiently.
