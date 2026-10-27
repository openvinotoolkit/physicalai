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
| `read()`        | Returns the next frame, blocks, and preserves sequence.        |
| `read_latest()` | Returns the newest frame, does not block, and may skip frames. |
| `async_read()`  | Provides an async wrapper around `read()`.                     |

`IPCamera` continuously drains OpenCV/FFmpeg into a single latest-frame slot.
Its `read()` waits for a frame newer than the last returned by either read method,
skipping overwritten frames. `read_latest()` only reads that slot: it never waits
for network I/O. `SharedCamera` also skips intermediate frames, draining its bounded
transport queue to the newest sample. Neither backend promises lossless recording.
A stalled stream's cached frame keeps its original sequence and timestamp;
`read(timeout=...)` times out instead of reporting the cache as a new frame.

IP frame timestamps measure **local decode/conversion completion**, not source
capture or simulator-pose publication. Sequences count locally decoded frames.
They cannot expose upstream latency, repeated source images, or source drops.
Shared transport preserves this metadata; it does not make source time available.
A source faster than the host can decode can still accumulate upstream latency.

The shared publisher waits for acquisition, then asks the backend for its latest
frame before publishing. This discards hardware backlogs without changing direct
sequential `read()` calls. Read and publish work count toward its FPS pacing budget.
Unpaced cameras retain a publication rate limit; IP acquisition drains independently
even when publication or consumers pause.

FFmpeg's native open/read timeouts are set at open time. Consumer read timeouts
bound the slot wait independently of native I/O. Disconnect wakes waiting consumers,
stops and joins the reader, then releases capture in that reader (never concurrently
with a native read). Connection-time failure cleanup can additionally take up to one
native read timeout (at most one second, plus scheduling/cleanup). A backend that
ignores its native timeout raises a shutdown error rather than being reopened over
an active reader. After a permanent stream failure, disconnect/reconnect explicitly;
this backend does not automatically reconnect.

### USB frame freshness

`UVCCamera` defaults to OmniCamera/pynokhwa on every platform, including Linux.
The native SDK already captures continuously into a latest-frame slot. Returned
sequence numbers identify native decoded frames, not necessarily sensor exposures.
Timestamps mark when Python first observes each sequence; repeated or cached reads
preserve that timestamp. Native/driver latency before that observation is not measured.

Linux callers can explicitly select `backend="v4l2"`. Its direct `read()` remains
sequential, while `read_latest()` drains at most one allocated buffer set and returns
the newest drained frame. If none are ready it returns its cached frame, or raises
`CaptureError` immediately when no cache exists. It never waits for a new frame.
V4L2 frames retain kernel sequence numbers, including gaps. Monotonic driver
timestamps are preserved; unknown/non-monotonic clocks use local dequeue time with
a warning because source capture age is unavailable. Neither cached reads nor
shared transport restamp those frames.

Camera instances are not thread-safe.
