---
name: physicalai-runtime-adding-a-camera-backend
description: Adds or modifies a camera backend under physicalai.capture. Use when implementing a new Camera type, extending create_camera in src/physicalai/capture/factory.py, discovery helpers, optional pip extras for vendor SDKs, SharedCamera transport, or tests under tests/unit/capture with fake devices.
license: Apache-2.0
---

# Adding a Camera Backend

Cameras implement the `Camera` interface in `src/physicalai/capture/camera.py`. Factory entry: `create_camera()` in `src/physicalai/capture/factory.py` maps lowercase type names (`uvc`, `realsense`, `basler`, …) to implementations. Reference layouts: `src/physicalai/capture/cameras/uvc/`, `src/physicalai/capture/cameras/realsense/`, `src/physicalai/capture/cameras/basler/`.

## Workflow

1. **Pick a reference backend** closest to the new hardware (UVC for USB video, RealSense for RGB-D, Basler for GenICam industrial).
   - Done when: you can list which modules to mirror (`_camera.py`, `_discover.py`, `__init__.py` exports).
2. **Implement `Camera`**: `connect()`, `disconnect()`, `read()` / `read_latest()`, context manager support, monotonic timestamps on `Frame` (`src/physicalai/capture/frame.py`).
   - Done when: fake or mocked device tests can exercise connect/read without hardware.
3. **Wire discovery** if the device is enumerable — add helpers under `src/physicalai/capture/discovery.py` or backend-specific `_discover.py`.
4. **Register the type** in `src/physicalai/capture/factory.py` and export public class from `src/physicalai/capture/__init__.py` when user-facing.
5. **Optional extra** in `pyproject.toml` for vendor SDKs; lazy-import inside the camera module so `pip install physicalai` stays light.
   - Done when: `import physicalai.capture` works without the extra; importing the camera class fails with a clear message if the extra is missing.
6. **Tests** in `tests/unit/capture/` using existing fakes (`tests/unit/capture/fake.py`, `conftest.py` patterns).
   - Done when: `uv run pytest tests/unit/capture -k <backend>` passes.

## Shared transport

For multi-process access, `create_camera(..., shared=True)` wraps with `SharedCamera` (`physicalai[capture]` / `transport` extra, iceoryx2). Only document shared mode when the transport extra is installed.

## Required checks

- `CameraType` or factory string is documented in `docs/reference/camera-api.md` when user-visible.
- Frame shapes and dtypes match README examples (RGB `(H, W, 3)`).
- No blocking discovery at import time.

## Verify

```bash
uv run pytest tests/unit/capture -q
prek run --all-files
```
