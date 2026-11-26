# Camera API Reference

Camera implementations are expected to inherit from `Camera`.

```python
class Camera:
    def connect(self, timeout: float = 5.0) -> None: ...
    def disconnect(self) -> None: ...
    def read(self, timeout: float = 2.0) -> Frame: ...
    def read_latest(self) -> Frame: ...
    async def async_read(self, timeout: float = 2.0) -> Frame: ...

    @property
    def is_connected(self) -> bool: ...

    @property
    def device_id(self) -> str: ...
```

## Color Mode

```python
class ColorMode(StrEnum):
    RGB = "rgb"
    BGR = "bgr"
    GRAY = "gray"
```

## Camera Type

```python
class CameraType(StrEnum):
    UVC = "uvc"
    IP = "ip"
    REALSENSE = "realsense"
    GENICAM = "genicam"
    BASLER = "basler"
```

## Read Semantics

| Method          | Semantics                                                      |
| --------------- | -------------------------------------------------------------- |
| `read()`        | Blocks for a new frame; backend-specific skipping applies.     |
| `read_latest()` | Returns the newest frame, does not block, and may skip frames. |
| `async_read()`  | Provides an async wrapper around `read()`.                     |

`IPCamera`, default `UVCCamera`, and `SharedCamera` provide latest-oriented reads
that can skip intermediate frames; they do not guarantee lossless recording.
`IPCamera.read()` waits for a frame newer than the last returned by either read method.
Cached frames retain their sequence and timestamp; `read(timeout=...)` raises
`CaptureTimeoutError` if no new frame arrives. `read_latest()` never waits for a new
frame and raises `CaptureError` if no frame is available.

IP timestamps mark local decode/conversion completion; default UVC timestamps mark
Python's first observation of a decoded frame. Neither measures source capture age.
Linux UVC with `backend="v4l2"` keeps sequential `read()` semantics and preserves kernel
sequences and monotonic driver timestamps, falling back to local dequeue time for
other clocks. Shared transport preserves the backend's metadata.

IP consumer read timeouts bound frame waits independently of native network I/O.
After a permanent IP stream failure, explicitly disconnect/reconnect; there is no
automatic reconnection. Shared-publisher reconfiguration of IP cameras is unsupported
and leaves the existing stream running.

Camera instances are not thread-safe.

## RealSense Depth Capture

`RealSenseCamera` streams color only by default. Set `enable_depth=True` when
constructing the camera to enable depth streaming and alignment to color:

```python
from physicalai.capture import RealSenseCamera

with RealSenseCamera(serial_number="<serial>", enable_depth=True) as camera:
    color, depth = camera.read_rgbd()
```

The same option works with `create_camera("realsense", ...)` and camera YAML
recipes (`init_args.enable_depth: true`). Existing callers of `read_depth()` or
`read_rgbd()` must opt in; these methods raise `CaptureError` when depth is disabled.
`read()` and `read_latest()` continue to return color frames in both modes.
`depth_width` and `depth_height` only configure an enabled depth stream; supplying
dimensions alone does not enable depth. Color-only capture does not create an
alignment processor or transfer unused depth frames over USB.

## Multi-Camera Synchronisation

`read_cameras()` and `async_read_cameras()` read from multiple cameras concurrently:

```python
from physicalai.capture import CaptureTimeoutError, read_cameras

try:
    synced = read_cameras(cameras, timeout=1.0)
    print(f"Skew: {synced.max_skew_ms:.2f}ms")
except CaptureTimeoutError as exc:
    print(f"Read timed out: {exc}")
```

- When given an empty mapping (`cameras={}`), both functions return `SyncedFrames(frames={}, max_skew_ms=0.0)`.
- If one or more cameras do not return a frame within `timeout` seconds, `CaptureTimeoutError` is raised with the set of unresponsive cameras.
- `timeout` must be positive; non-positive values raise `ValueError`.
