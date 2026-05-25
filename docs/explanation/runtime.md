# Runtime

> **Preview:** `PolicyRuntime` is a planned API. The content below documents the target design.

`PolicyRuntime` runs a policy on robot hardware. It owns the control loop, the callback lifecycle, and the interaction between observations, inference requests, and actions.

```python
runtime = PolicyRuntime.from_config("runtime.yaml")
runtime.run(duration_s=60)
```

## Responsibilities

| Component            | Owns                                                       | Does not own      |
| -------------------- | ---------------------------------------------------------- | ----------------- |
| `InferenceModel`     | model load, preprocess, inference, postprocess             | robot loop timing |
| `TeleoperatorPolicy` | reading a leader robot and producing action chunks         | follower robot IO |
| `Execution`          | where policy evaluation runs                               | robot IO          |
| `ActionQueue`        | action chunks and buffering                                | model inference   |
| `PolicyRuntime`      | observe, request inference, send action, callbacks, timing | policy math       |
| `Robot`              | hardware connection, observations, actions                 | policy inference  |

## Loop

The runtime loop follows this general pattern:

```text
while running:
    observation = get_robot_state() + get_camera_frames()
    maybe_request_inference(observation)
    action = get_next_action_or_hold()
    send_action_to_robot(action)
    sleep_until_next_tick()
```

The exact observation structure and merging strategy may change as the API stabilizes.

`PolicyRuntime` accepts any policy-like object with `predict_action_chunk()`. A trained `InferenceModel` is one implementation. `TeleoperatorPolicy` is another implementation that reads a leader robot and returns the leader joint positions as a one-step action chunk for the follower.

## Execution Modes

| Mode                                  | Where inference runs | Use                                      |
| ------------------------------------- | -------------------- | ---------------------------------------- |
| `SyncExecution(mode="single_action")` | runtime thread       | simple policies                          |
| `SyncExecution(mode="chunk")`         | runtime thread       | chunk policies without background worker |
| `AsyncExecution(transport="thread")`  | worker thread        | avoid blocking control loop              |
| `RemoteExecution`                     | remote server        | robot host without policy weights        |

## Product Workflows

HIL, recording, highlight, and DAgger should be composed through callbacks until they justify reusable runtime primitives.

```python
class HILCallback:
    def before_send_action(self, action, step):
        if teleop.enabled:
            return teleop.read_action()
        return action
```
