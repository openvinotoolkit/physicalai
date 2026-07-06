# Skill Evaluation Scenarios

Use these prompts to test whether an agent correctly invokes and follows the capture skill. Run from the repo root.

Expected rubric per scenario:

- **Activates the right skill** — `physicalai-runtime-adding-a-camera-backend`.
- **Uses real paths** — `src/physicalai/capture/...`, `tests/unit/capture/`.
- **Follows workflow steps** with Done when criteria.
- **Produces a checkable artifact** — test run or new module skeleton.

## `physicalai-runtime-adding-a-camera-backend`

### Scenario 1: Scaffold a new USB camera variant

> "Add a minimal UVC-like camera under `src/physicalai/capture/cameras/` for a new V4L2 device path API. Mirror the existing UVCCamera layout and add a unit test with fakes."

Expected behavior:

- References `src/physicalai/capture/cameras/uvc/` structure.
- Uses `tests/unit/capture/fake.py` or conftest patterns.
- Runs `uv run pytest tests/unit/capture -k <name>`.

### Scenario 2: Wire factory and optional extra

> "Register camera type `myvendor` in `create_camera` behind optional extra `physicalai[myvendor]` with lazy import."

Expected behavior:

- Updates `src/physicalai/capture/factory.py` and `pyproject.toml` optional dependency.
- Clear ImportError/message without the extra.

### Scenario 3: SharedCamera documentation

> "User wants two processes to read the same RealSense. What capture API and extra should they use?"

Expected behavior:

- Mentions `create_camera(..., shared=True)`, `SharedCamera`, `transport` / `capture` extra, iceoryx2 pin in pyproject.
- Does not claim shared mode works without the extra.
