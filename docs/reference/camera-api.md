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
