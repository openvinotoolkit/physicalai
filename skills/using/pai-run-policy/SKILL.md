---
name: pai-run-policy
description: Runs an exported Physical AI policy on a robot with RobotRuntime and physicalai run. Use when wiring a robot, cameras, PolicySource or TeleopSource, choosing SyncExecution/AsyncExecution/RTCExecution, configuring a runtime YAML recipe, or diagnosing action queues and callbacks. DO NOT USE FOR implementing a new robot driver inside the Runtime source tree.
license: Apache-2.0
---

# Running a Policy on a Robot

`RobotRuntime` owns the control loop, hardware I/O, callbacks, and timing. It takes a required `action_source`: `PolicySource` wraps model + execution + action queue; `TeleopSource` forwards a leader arm. `InferenceModel` owns policy math. The `physicalai run` CLI instantiates the same objects from YAML. See the [Runtime how-to](https://github.com/openvinotoolkit/physicalai/blob/main/docs/how-to/runtime/run-policy-on-robot.md).

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

   - Done when: components run in a test with fake devices. Obtain explicit approval before connecting to hardware or sending actions.

3. **YAML config** — nest `class_path` / `init_args` under `runtime:` for `robot`, `action_source`, `cameras`, `fps`. `model`, `execution`, and `action_queue` go under `action_source.init_args` (they belong to `PolicySource`, not the runtime); run:

   ```bash
   physicalai run --config runtime.yaml --run.duration_s=60
   ```

4. **Execution mode** — pick `SyncExecution`, `AsyncExecution(request_threshold=...)`, or `RTCExecution(fps=...)` + `RTCActionQueue` per the [execution modes guide](https://github.com/openvinotoolkit/physicalai/blob/main/docs/how-to/runtime/use-execution-modes.md). `AsyncExecution` does not take `fps`; the control-loop rate lives on `RobotRuntime`. Do not build ad-hoc timing around `InferenceModel.select_action` when `PolicySource` should own the queue.
5. **Callbacks** — register via the [runtime callback API](https://github.com/openvinotoolkit/physicalai/blob/main/docs/how-to/runtime/add-runtime-callbacks.md) for telemetry/latency, not inside inference adapters.

## Validation loop

Use fake robots/cameras for a short dry run in the user's project. Before a real run, check the robot's joint order and action dimensions, ensure the work area is safe, and obtain approval for hardware I/O.

## Required checks

- `fps` and camera read rates are consistent.
- Action dimensions match robot `send_action` expectations.
- Config `class_path` targets are importable without training packages.
- Check config fields against the [Runtime config schema](https://github.com/openvinotoolkit/physicalai/blob/main/docs/reference/config-schema.md).

## References

- [Run a policy on a robot](https://github.com/openvinotoolkit/physicalai/blob/main/docs/how-to/runtime/run-policy-on-robot.md)
- [Execution modes](https://github.com/openvinotoolkit/physicalai/blob/main/docs/how-to/runtime/use-execution-modes.md)
- [Runtime config](https://github.com/openvinotoolkit/physicalai/blob/main/docs/how-to/config/write-runtime-config.md)
- [Runtime API](https://github.com/openvinotoolkit/physicalai/blob/main/docs/reference/runtime-api.md)
