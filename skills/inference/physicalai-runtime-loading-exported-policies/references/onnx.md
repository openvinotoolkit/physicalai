# ONNX adapter

- Registry key: `onnx`, extension `.onnx`.
- Implementation: `src/physicalai/inference/adapters/onnx.py` (`ONNXAdapter`).
- Dependency: `onnxruntime` (core install includes it).
- `device` kwargs follow ONNX Runtime providers (`cpu`, `cuda`, etc.) — verify against the adapter before documenting new values.
