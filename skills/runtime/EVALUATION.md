# Skill Evaluation Scenarios

Use these prompts to test runtime bucket skills from the repo root.

Expected rubric per scenario:

- **Activates the right skill** — matches `physicalai-runtime-running-policy-on-robot` or `physicalai-runtime-adding-a-robot-integration`.
- **Uses real paths and commands** — `src/physicalai/runtime/`, `physicalai run`, `uv run pytest tests/unit/runtime/`.
- **Follows workflow checklists**.
- **Produces a checkable artifact**.

## `physicalai-runtime-running-policy-on-robot`

### Scenario 1: YAML-driven run

> "Write a `runtime.yaml` that runs an exported policy on SO-101 with one wrist UVC camera for 30 seconds, then give the exact `physicalai run` command."

Expected behavior:

- Nested `class_path` / `init_args` for `SO101`, `PolicySource`, `InferenceModel`, `UVCCamera`, `SyncExecution`, under a top-level `runtime:` key; `model` and `execution` sit under `action_source.init_args`.
- Command includes `--config` and duration override pattern from `src/physicalai/cli/run.py` docs.

### Scenario 2: Execution mode choice

> "User needs lower inference latency with chunked actions. Which execution class should they use and where is it documented?"

Expected behavior:

- Points to `docs/how-to/runtime/use-execution-modes.md` and `src/physicalai/runtime/execution/` (`sync.py`, `async_execution.py`, `rtc.py`).
- Distinguishes `AsyncExecution(request_threshold=...)` from `RTCExecution(fps=...)` + `RTCActionQueue`; does not put `fps` on `AsyncExecution`.
- Does not reimplement timing inside `InferenceModel.select_action`.

### Scenario 3: Runtime test with fakes

> "Add a unit test that RobotRuntime steps one tick with a fake robot and a `PolicySource` over a fake model."

Expected behavior:

- Uses patterns from `tests/unit/runtime/test_runtime.py`.
- Runs `uv run pytest tests/unit/runtime -k <test>`.

## `physicalai-runtime-adding-a-robot-integration`

### Scenario 4: New protocol-compliant robot stub

> "Create a minimal fake robot class that satisfies the Robot protocol for use in runtime tests."

Expected behavior:

- Implements methods from `src/physicalai/robot/interface.py`; cites `references/robot-protocol.md`.
- `isinstance(..., Robot)` check in test.

### Scenario 5: SO-101 port validation

> "User passes `port='../../etc/passwd'` to SO101. What should the integration do?"

Expected behavior:

- Path validation / rejection per security rules; no shelling out with raw path.

### Scenario 6: Export new robot from package

> "Add a new arm driver package under `src/physicalai/robot/myarm/` with optional extra and unit tests."

Expected behavior:

- Lazy SDK import, `pyproject.toml` extra, tests under `tests/unit/robot/`.
- Documents joint_names ordering.
