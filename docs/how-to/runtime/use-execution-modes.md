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

## Real-Time Chunking

This mode runs real-time chunking on a background thread. Pair it with an `RTCActionQueue`.

```python
from physicalai.runtime import RTCActionQueue, RTCExecution

execution = RTCExecution(fps=30)
action_queue = RTCActionQueue()
```

`fps` here is the robot control rate `RTCExecution` uses to turn measured inference latency into an action delay. In YAML, both objects sit under `action_source.init_args`:

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

## Remote

> **Preview:** `RemoteExecution` is not yet implemented.

This mode would send inference requests to a policy server. The class is not in the current package. When it lands, it belongs on `PolicySource`, under `action_source.init_args`, like the other execution modes:

```yaml
action_source:
  class_path: physicalai.runtime.PolicySource
  init_args:
    execution:
      class_path: physicalai.runtime.RemoteExecution
      init_args:
        endpoint: http://robot-server:8080
```

Use this mode when the robot host should not hold policy weights or accelerator dependencies.
