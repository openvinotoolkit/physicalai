---
name: physicalai-runtime-running-policy-on-robot
description: Runs exported policies on hardware with PolicyRuntime, execution modes, and physicalai run. Use when wiring PolicyRuntime, SyncExecution or RTC execution, runtime YAML configs, action queues, runtime callbacks, or docs/how-to/runtime run-policy-on-robot and execution modes.
license: Apache-2.0
---

# Running a Policy on a Robot

`PolicyRuntime` (`src/physicalai/runtime/runtime.py`) owns the control loop; `InferenceModel` owns policy math. Execution strategy lives under `src/physicalai/runtime/execution.py` and `src/physicalai/runtime/rtc_execution.py`. CLI: `physicalai run` in `src/physicalai/cli/run.py` instantiates from YAML via jsonargparse.

## Workflow

1. **Choose API vs config**: Python for notebooks/tests; YAML + `physicalai run` for reproducible deployment.
   - Done when: entry point matches the user's task.
2. **Python minimal loop** (see `docs/how-to/runtime/run-policy-on-robot.md`):

   ```python
   from physicalai.runtime import PolicyRuntime, SyncExecution
   from physicalai.inference import InferenceModel
   from physicalai.robot import SO101
   from physicalai.capture import UVCCamera

   runtime = PolicyRuntime(
       fps=30,
       robot=SO101(port="/dev/ttyACM0"),
       model=InferenceModel("./exports/act_policy"),
       cameras={"wrist": UVCCamera(device="/dev/video0", width=640, height=480)},
       execution=SyncExecution(),
   )
   with runtime:
       runtime.run(duration_s=60)
   ```

   - Done when: components connect and the loop runs in a test or dry-run with fakes.

3. **YAML config** — nest `class_path` / `init_args` for robot, model, cameras, execution; run:

   ```bash
   physicalai run --config runtime.yaml --run.duration_s=60
   ```

4. **Execution mode** — pick sync vs RTC per `docs/how-to/runtime/use-execution-modes.md`; do not build ad-hoc timing around `InferenceModel.select_action` when `PolicyRuntime` should own the queue.
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
- `docs/how-to/config/write-runtime-config.md`
