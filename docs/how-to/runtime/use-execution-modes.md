# Use Execution Modes

The `Execution` component decides where inference runs and how requests are scheduled.

## Synchronous

This mode runs inference in the runtime thread.

```python
from physicalai.runtime import SyncExecution

execution = SyncExecution()
```

This mode is appropriate for simple deployments and debugging.

## Thread Worker

This mode runs inference in a background thread.

```python
from physicalai.runtime import AsyncExecution

execution = AsyncExecution(request_threshold=0.5)
```

Use this mode when model latency should not block robot timing. `request_threshold` is the fraction of the action chunk still queued when the next inference starts. The control-loop rate stays on `RobotRuntime` as `fps`. Since inference backends typically release the GIL, thread-based execution works well for most use cases.

`AsyncExecution` warns when the robot holds its last action because the next chunk arrived late. The warning suggests a `request_threshold` that avoids this, or says that inference is too slow for any threshold to help. It appears the first time this happens in a run and then every 50th time, and only when the threshold is above 0.

## Real-Time Chunking

This mode runs real-time chunking on a background thread. Pair it with an `RTCActionQueue`.

```python
from physicalai.runtime import RTCActionQueue, RTCExecution

execution = RTCExecution(fps=30)
action_queue = RTCActionQueue()
```

`fps` here is the robot control rate `RTCExecution` uses to turn measured inference latency into an action delay. `RTCExecution` times every inference itself. Until the first steady measurement it uses the duration of the warmup call, or `initial_latency_s` if you pass the latency you expect. In YAML, both objects sit under `action_source.init_args`:

```yaml
action_source:
  class_path: physicalai.runtime.PolicySource
  init_args:
    execution:
      class_path: physicalai.runtime.RTCExecution
      init_args:
        fps: 30
    action_queue:
      class_path: physicalai.runtime.RTCActionQueue
```

Use this mode for exports that expect real-time chunking inputs.

## Remote Inference

Remote inference is supported through `RemoteInferenceModel`; choose `SyncExecution`, `AsyncExecution`, or `RTCExecution` to control how requests are scheduled. There is no separate `RemoteExecution` class.

```yaml
action_source:
  class_path: physicalai.runtime.PolicySource
  init_args:
    model:
      class_path: physicalai.inference.RemoteInferenceModel
      init_args:
        name: cell1-act
    execution:
      class_path: physicalai.runtime.AsyncExecution
      init_args:
        request_threshold: 0.5
    task: "pick up the red cube and place it in the blue bowl"
```

See [Run Inference Remotely](run-inference-remotely.md) for server setup, networking, and configuration details.
