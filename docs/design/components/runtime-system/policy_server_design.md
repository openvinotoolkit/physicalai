# PolicyServer and RemoteExecution

This is the remote inference design for the runtime.

> **Status: speculative, not implemented.** `PolicyServer` and
> `RemoteExecution` do not exist yet — this document proposes how remote
> inference would fit into the shipped `RobotRuntime`/`ActionSource` design
> (see [runtime_design.md](./runtime_design.md)) once built. Terminology here
> follows that design (`RobotRuntime` + `PolicySource`, not the retired
> `PolicyRuntime`).

## Goal

Run the robot loop on one host and the policy model on another host.

```text
Robot host                              Server host
----------                              -----------
RobotRuntime                            PolicyServer
  Robot                                   InferenceModel
  cameras                                 runner / backend
  PolicySource                            predict_action_chunk()
    ActionQueue
    RemoteExecution  <------ gRPC ------>   warmup / health
```

The robot host owns timing and action dispatch. The server host owns model inference.

## Robot Host

```python
from physicalai.runtime import RobotRuntime, PolicySource, RemoteExecution
from physicalai.inference.remote import RemoteInferenceModel

runtime = RobotRuntime(
    robot=robot,
    action_source=PolicySource(
        model=RemoteInferenceModel(endpoint="grpc://gpu-host:50051"),
        execution=RemoteExecution(endpoint="grpc://gpu-host:50051"),
    ),
    fps=30,
)

runtime.run()
```

`RemoteExecution` would implement the same interface as sync and async execution:

```python
class RemoteExecution(Execution):
    def start(self, action_queue, model): ...
    def maybe_request(self, observation): ...
    def warmup(self, sample_observation, n=2): ...
    def stop(self): ...
```

It sends observation snapshots to the server and pushes returned chunks into the local `ActionQueue` owned by `PolicySource`.

## Server Host

```bash
physicalai serve --config policy_server.yaml
```

Server config:

```yaml
server:
  host: 0.0.0.0
  port: 50051

model:
  class_path: physicalai.inference.InferenceModel
  init_args:
    path: ./exports/pi0_policy
```

`PolicyServer` loads the real `InferenceModel` and exposes:

- handshake
- warmup
- health
- `predict_action_chunk`
- optionally `select_action`

## Data Flow

```text
RobotRuntime tick
  robot_state, camera_frames = read once
  action = action_source.update(robot_state, camera_frames, step)   # PolicySource

PolicySource.update()
  execution.maybe_request(model_input)   # RemoteExecution

RemoteExecution
  serialize model_input
  send PredictRequest

PolicyServer
  chunk = model.predict_action_chunk(model_input)
  send PredictReply

RemoteExecution
  action_queue.push_chunk(chunk)

PolicySource.update() (same or later tick)
  action = action_queue.pop()   # falls back to last action if empty
  return action

RobotRuntime tick
  robot.send_action(action)
```

## Transport

Recommended default: gRPC bidirectional streaming.

Sketch:

```proto
service PolicyServer {
  rpc Handshake(HandshakeRequest) returns (HandshakeReply);
  rpc Warmup(WarmupRequest) returns (WarmupReply);
  rpc Predict(stream PredictRequest) returns (stream PredictReply);
  rpc Health(google.protobuf.Empty) returns (HealthReply);
}

message PredictRequest {
  string request_id = 1;
  double t0 = 2;
  map<string, Tensor> observation = 3;
  optional int32 inference_delay = 4;
  optional ActionChunk prev_chunk_left_over = 5;
}

message PredictReply {
  string request_id = 1;
  double t0 = 2;
  Tensor actions = 3;
  optional double policy_dt = 4;
  map<string, Tensor> extra = 5;
}
```

## RTC

RTC works the same way as local async execution.

```text
RemoteExecution computes delay
RemoteExecution sends inference_delay + prev_chunk_left_over
PolicyServer calls model.predict_action_chunk(...)
server-side runner uses FlowMatching(guidance=RTC())
```

The client-side `PolicySource` still owns `ActionQueue` and `RTCQueueMerger`. The server returns chunks; it does not smooth or dispatch actions.

## Failure Policy

| Failure                                   | Behavior                                                                                  |
| ----------------------------------------- | ----------------------------------------------------------------------------------------- |
| Cannot connect at startup                 | `PolicySource.connect()` raises; runtime's connect/disconnect rollback tears down cleanly |
| Connection lost with no request in flight | reconnect until `reconnect_budget_s` is exhausted                                         |
| Connection lost with request in flight    | drop that request; next observation creates a new request                                 |
| Server error                              | surface error on next runtime call                                                        |
| Deadline exceeded                         | drop request; continue with next tick/request                                             |
| Schema mismatch                           | fail during handshake                                                                     |

Do not retry stale observations. In control loops, a late retry is usually worse than dropping the request.

## CLI

`physicalai serve` is registered by the runtime distribution:

```toml
[project.entry-points."physicalai.cli.subcommands"]
serve = "physicalai.cli.serve:register"
```

It uses the same runtime-side CLI as `physicalai run`, without Torch or Lightning imports.

## Out of Scope

- multi-robot single-server
- server-side action smoothing
- server-side robot state
- model hot-swap semantics beyond rejecting in-flight requests
- low-level transport optimization beyond gRPC defaults

## Config

Follows the runtime's single `action_source:` schema (see
[runtime_design.md](./runtime_design.md#config-one-schema-no-shorthand)) — no
flat/legacy shorthand:

```yaml
runtime:
  robot:
    { class_path: physicalai.robot.SO101, init_args: { port: /dev/ttyACM0 } }
  action_source:
    class_path: physicalai.runtime.PolicySource
    init_args:
      model:
        class_path: physicalai.inference.remote.RemoteInferenceModel
        init_args: { endpoint: grpc://gpu-host:50051 }
      execution:
        class_path: physicalai.runtime.RemoteExecution
        init_args: { endpoint: grpc://gpu-host:50051 }
  fps: 30.0
```

## Build Target

Build after local `RobotRuntime`, `PolicySource`, `AsyncExecution`, and
`ActionQueue` are stable (all shipped today; `RemoteExecution`/`PolicyServer`
are the only pieces this document proposes).

Acceptance test:

```text
RemoteExecution + PolicyServer produces actions equivalent to
AsyncExecution(transport="process") for the same model and observations.
```
