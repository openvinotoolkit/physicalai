---
name: physicalai-runtime-running-policy-on-robot
description: Runs exported policies on hardware with RobotRuntime, action sources, execution modes, and physicalai run. Use when wiring RobotRuntime, PolicySource or TeleopSource, SyncExecution, AsyncExecution or RTCExecution, runtime YAML configs, action queues, runtime callbacks, or docs/how-to/runtime run-policy-on-robot and execution modes.
license: Apache-2.0
---

# Running a Policy on a Robot

`RobotRuntime` (`src/physicalai/runtime/core.py`) owns the control loop, hardware I/O, callbacks, and timing. It takes a required `action_source`: `PolicySource` wraps model + execution + action queue, `TeleopSource` forwards a leader arm (`src/physicalai/runtime/action_sources/`). `InferenceModel` owns policy math. Execution strategies live under `src/physicalai/runtime/execution/` (`sync.py`, `async_execution.py`, `rtc.py`, queues in `queue.py` and `rtc_queue.py`). CLI: `physicalai run` in `src/physicalai/cli/run.py` instantiates from YAML via jsonargparse.

## Workflow

1. **Choose API vs config**: Python for notebooks/tests; YAML + `physicalai run` for reproducible deployment.
   - Done when: entry point matches the user's task.
2. **Python minimal loop** (see `docs/how-to/runtime/run-policy-on-robot.md`):

   ```python
   from physicalai.runtime import RobotRuntime, PolicySource, SyncExecution
   from physicalai.inference import InferenceModel
   from physicalai.robot import SO101
   from physicalai.capture import UVCCamera

   runtime = RobotRuntime(
       fps=30,
       robot=SO101(port="/dev/ttyACM0"),
       action_source=PolicySource(
           model=InferenceModel("./exports/act_policy"),
           execution=SyncExecution(),
       ),
       cameras={"wrist": UVCCamera(device="/dev/video0", width=640, height=480)},
   )
   with runtime:
       runtime.run(duration_s=60)
   ```

   - Done when: components connect and the loop runs in a test or dry-run with fakes.

3. **YAML config** — nest `class_path` / `init_args` under `runtime:` for `robot`, `action_source`, `cameras`, `fps`. `model`, `execution`, and `action_queue` go under `action_source.init_args` (they belong to `PolicySource`, not the runtime); run:

   ```bash
   physicalai run --config runtime.yaml --run.duration_s=60
   ```

4. **Execution mode** — pick `SyncExecution`, `AsyncExecution(request_threshold=...)`, or `RTCExecution(fps=...)` + `RTCActionQueue` per `docs/how-to/runtime/use-execution-modes.md`. `AsyncExecution` does not take `fps`; the control-loop rate lives on `RobotRuntime`. Do not build ad-hoc timing around `InferenceModel.select_action` when `PolicySource` should own the queue.
5. **Callbacks** — register via runtime callback APIs (`docs/how-to/runtime/add-runtime-callbacks.md`) for telemetry/latency, not inside inference adapters.

## Validation loop

```bash
uv run pytest tests/unit/runtime/ -q
```

Use fake robots/cameras from runtime tests when hardware is unavailable.

## Required checks

- `fps` and camera read rates are consistent.
- Action dimensions match robot `send_action` expectations.
- Config `class_path` targets are importable without training packages.
- Document breaking changes to runtime config schema in `docs/reference/config-schema.md`.

## References

- `docs/how-to/runtime/run-policy-on-robot.md`
- `docs/how-to/runtime/use-execution-modes.md`
- `docs/how-to/config/write-runtime-config.md`
- `docs/reference/runtime-api.md`
