# OpenVINO adapter

- Registry key: `openvino`, extension `.xml` (`.bin` alongside).
- Implementation: `src/physicalai/inference/adapters/openvino.py` (`OpenVINOAdapter`).
- Dependency: `openvino` (core install pins a version in `pyproject.toml`).
- Default device string is OpenVINO-style (`CPU`, `GPU`, `NPU`, `AUTO`).
