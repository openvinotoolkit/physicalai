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

## Stable Linux UVC Selectors

`UVCCamera(device="/dev/v4l/by-path/...")` accepts Linux port/interface aliases
with either the default OmniCamera backend or explicit `backend="v4l2"`. The alias
is retained in exported configuration and resolved when connecting, so a changed
`/dev/videoN` does not require rewriting the configuration. A missing alias fails;
it does not fall back to another camera.

Use `pai devices inspect /dev/videoN` to find the alias for the intended capture
stream. Metadata and image nodes can share a physical camera, and `by-id` names
can collide even when a real camera serial is present. A port alias identifies a
location, not an authenticated physical unit. See
[hardware diagnostics](../how-to/cli/doctor.md) for identity and topology limits.

## Read Semantics

| Method          | Semantics                                                      |
| --------------- | -------------------------------------------------------------- |
| `read()`        | Returns the next frame, blocks, and preserves sequence.        |
| `read_latest()` | Returns the newest frame, does not block, and may skip frames. |
| `async_read()`  | Provides an async wrapper around `read()`.                     |

Camera instances are not thread-safe.

## Optional Payload Accounting

`camera.payload_bytes_received` returns the number of encoded frame bytes dequeued
during the current connection when a backend supports it, otherwise `None`.
V4L2 resets the counter on connect and includes frames discarded by `read_latest()`;
the UVC facade forwards the selected backend's value. Sampling code should compare
counter values around the measurement window, excluding warmup.

This is not decoded `Frame.data.nbytes` and not complete USB traffic accounting:
protocol headers, retries, and frames lost before dequeue are excluded. Existing
backends need not implement the optional counter; unsupported does not mean zero.
