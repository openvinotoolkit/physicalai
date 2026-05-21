# Capture

Provides camera interfaces, discovery, frame types, and shared transport utilities.

## Public API

- `Camera`: base camera interface
- camera implementations exposed from `physicalai.capture`
- `read_cameras(...)`: multi-camera read helper
- `SharedCamera`: transport-backed camera wrapper for multi-process access

## Main Modules

- `camera.py`: base interface and common behavior
- `cameras/`: backend-specific camera implementations
- `discovery.py`: device discovery helpers
- `multi.py`: multi-camera coordination helpers
- `transport/`: publisher/subscriber transport for shared camera access

## Related Docs

- `docs/explanation/cameras.md`
- `docs/reference/camera-api.md`
