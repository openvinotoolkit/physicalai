# Runtime API Reference

## `RobotRuntime`

`RobotRuntime` is the main orchestrator for running a policy (or teleop, or any custom control logic) on hardware. It always takes an explicit, pluggable `action_source`.

```python
RobotRuntime(
    robot: Robot,
    action_source: ActionSource,
    fps: float,
    cameras: Mapping[str, Camera] | None = None,
    callbacks: Sequence[RuntimeCallback] = (),
)
```

The most important methods are shown below.

```python
runtime.connect() -> None
runtime.disconnect() -> None
runtime.run(*, duration_s: float | None = None) -> int
```

`RobotRuntime` also supports context-manager usage so connections are cleaned up automatically. A "step" is one iteration of the control loop at `fps`: read an observation, get one action from `action_source`, and send it to the robot. `run()` returns the number of steps completed this run — there is no aggregate stats object. Other stats are read directly off the objects the caller already holds, e.g. `runtime.action_source.action_queue.total_pops` or `execution.inference_count`.

```python
with RobotRuntime(...) as runtime:
    steps = runtime.run(duration_s=60)
```

## `ActionSource`

`ActionSource` is the protocol a developer implements to plug custom decision logic into `RobotRuntime`. Three required methods, nothing optional — no capability protocols, no `isinstance` checks anywhere in the runtime.

```python
class ActionSource(Protocol):
    def connect(self, *, bus: _CallbackBus, session_id: str) -> None: ...
    def update(self, robot_state: RobotObservation, camera_frames: Mapping[str, Frame], step: int) -> np.ndarray: ...
    def disconnect(self) -> None: ...
```

`bus` is an internal callback bus injected fresh by `RobotRuntime` on every `run()`; action sources typically only forward it into an `Execution` strategy (see `PolicySource` below) rather than using it directly.

The action-source implementations shipped today are listed below.

| Class          | Purpose                                                                |
| -------------- | ---------------------------------------------------------------------- |
| `PolicySource` | runs a trained model through an `Execution` strategy + `ActionQueue`   |
| `TeleopSource` | reads a leader arm and forwards its observation as the follower action |

```python
PolicySource(
    model: InferenceModel,
    execution: Execution,
    action_queue: ActionQueue | None = None,
    *,
    task: str | None = None,
)

TeleopSource(
    leader: Robot,
    *,
    to_action: Callable[[RobotObservation], np.ndarray] | None = None,
)
```

## `Execution`

```python
class Execution:
    def start(self, model: InferenceModel, action_queue: ActionQueue) -> None: ...
    def maybe_request(self, observation: dict[str, np.ndarray]) -> None: ...
    def warmup(self, sample_observation: dict[str, np.ndarray]) -> None: ...
    def stop(self) -> None: ...
    @property
    def chunk_size(self) -> int: ...
```

The execution implementations shipped today are listed below.

| Class            | Purpose                               |
| ---------------- | ------------------------------------- |
| `SyncExecution`  | runs inference in the runtime thread  |
| `AsyncExecution` | runs inference in a background thread |

> **Preview:** `RemoteExecution` is a planned API and is not part of the current package release.

## `ActionQueue`

```python
queue.push_chunk(chunk)
action = queue.pop()
queue.clear()
```

The action queue owns runtime buffering, merging, smoothing, and the policy for handling late results.
