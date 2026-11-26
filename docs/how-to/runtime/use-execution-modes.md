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

Use this mode for exports that expect real-time chunking inputs (`prev_chunk_left_over`, `inference_delay`, `max_guidance_weight` and `execution_horizon`).

An RTC export also runs under `SyncExecution` and `AsyncExecution` if its graph takes all four inputs and its manifest declares a fixed `(chunk_size, action_dim)` shape for `prev_chunk_left_over` among its input features that matches the graph, as Studio exports do. Otherwise a warning at load says why nothing is filled in. `InferenceModel` then adds the inputs `RTCExecution` sends for a first chunk to every observation, before the callbacks and preprocessors: no previous chunk, a guidance weight of 1e-8, horizon 1 and delay 0. It logs this once.

The first-chunk inputs are added only when the observation has none of the four RTC inputs. A callback or preprocessor that adds RTC values only when they are missing finds the first-chunk values already there; create the model with `InferenceModel(..., fill_rtc_inputs=False)` to keep such a hook in charge.

`RTCExecution.start()` refuses a local model whose graph takes none of the RTC inputs, because it would drop them and run every chunk unguided. Re-export the policy with RTC enabled, or pick another mode. When the graph takes only some of them, or when it cannot tell (for example with `RemoteInferenceModel`), it logs a warning and starts.

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
