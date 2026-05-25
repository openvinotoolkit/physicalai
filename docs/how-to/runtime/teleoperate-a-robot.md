# Teleoperate a Robot

Use `TeleoperatorPolicy` when you have a leader robot instead of a trained model. The leader robot is read every control tick and its joint positions are sent as actions to the follower robot through the normal `PolicyRuntime` loop.

## When To Use

Use this workflow for direct leader-follower teleoperation, data collection, hardware bring-up, and debugging robot action conventions before running a learned policy.

Do not use this workflow when the leader and follower have different joint order, different units, or different action semantics unless you add an explicit mapping layer.

## Run The Example

For SO101, pass separate calibration files for the leader and follower arms.

```bash
uv run --extra observer-rerun --extra robots --extra capture examples/runtime/teleoperation.py \
  --robot so101 \
  --leader-port /dev/ttyACM0 \
  --follower-port /dev/ttyACM1 \
  --leader-calibration /path/to/leader-calibration.json \
  --follower-calibration /path/to/follower-calibration.json \
  --camera overhead:uvc:/dev/video0 \
  --camera arm:uvc:/dev/video2 \
  --fps 30 \
  --rerun spawn \
  --rerun-image-decimation 15 \
  --rerun-jpeg-quality 75 \
  --rerun-image-max-dim 480 \
  --duration-s 90
```

For WidowXAI:

```bash
uv run --extra observer-rerun --extra robots --extra capture examples/runtime/teleoperation.py \
  --robot widowxai \
  --leader-ip 192.168.1.2 \
  --follower-ip 192.168.1.3 \
  --camera front:uvc:/dev/video0 \
  --fps 30 \
  --rerun spawn \
  --duration-s 90
```

For bimanual WidowXAI:

```bash
uv run --extra observer-rerun --extra robots --extra capture examples/runtime/teleoperation.py \
  --robot bimanual_widowxai \
  --leader-ip-left 192.168.1.2 \
  --leader-ip-right 192.168.1.3 \
  --follower-ip-left 192.168.1.4 \
  --follower-ip-right 192.168.1.5 \
  --fps 30 \
  --duration-s 90
```

## Python API

```python
from physicalai.robot import SO101, connect
from physicalai.runtime import ActionQueue, PolicyRuntime, SyncExecution, TeleoperatorPolicy

leader = SO101(
    port="/dev/ttyACM0",
    calibration="/path/to/leader-calibration.json",
    role="leader",
)
follower = SO101(
    port="/dev/ttyACM1",
    calibration="/path/to/follower-calibration.json",
    role="follower",
)

runtime = PolicyRuntime(
    robot=follower,
    model=TeleoperatorPolicy(leader),
    execution=SyncExecution(fps=30, request_threshold=1.0),
    action_queue=ActionQueue(),
    fps=30,
)

with connect(leader), runtime:
    runtime.run(duration_s=60)
```

## How It Works

`PolicyRuntime` accepts a policy-like object with `predict_action_chunk()`. `InferenceModel` satisfies that contract, and so does `TeleoperatorPolicy`.

```text
PolicyRuntime
  -> SyncExecution.maybe_request(follower_observation)
  -> TeleoperatorPolicy.predict_action_chunk(...)
  -> leader.get_observation().joint_positions[None, :]
  -> ActionQueue.push_chunk(...)
  -> follower.send_action(action)
```

`TeleoperatorPolicy` returns a chunk of size one, so `SyncExecution` should be used. The example uses `request_threshold=1.0` so each consumed action causes the next control tick to read the leader again.

## Safety Checks

Before running teleoperation, verify these conditions:

- Leader and follower are the same robot type.
- `leader.joint_names` and `follower.joint_names` have the same order.
- Leader and follower observations/actions use compatible units.
- The follower workspace is clear and safe.
- The leader starts from a pose that the follower can safely reach.

## Limitations

`TeleoperatorPolicy` currently mirrors absolute joint positions. It does not do retargeting, scaling, joint remapping, gripper adaptation, velocity control, filtering, or workspace constraints.

For human-in-the-loop policy control, wrap a learned policy and switch between `model_policy.predict_action_chunk()` and `TeleoperatorPolicy.predict_action_chunk()` based on a takeover signal.
