# PhysicalAI SO-101 Plugin

Physical AI Studio catalog definitions, hardware drivers, and URDF assets for
single-arm and bimanual SO-101 robots, built for
[PhysicalAI](https://github.com/openvinotoolkit/physicalai).

## Installation

```bash
uv add physicalai-so101-plugin
```

Installing the package registers `SO101_Follower`, `SO101_Leader`,
`BimanualSO101_Follower`, and `BimanualSO101_Leader` with the Physical AI
Studio robot catalog. It provides `SO101`, `BimanualSO101`, `SO101Calibration`,
and `SO101JointCalibration`.

```python
from physicalai_so101_plugin import BimanualSO101, SO101

left = SO101(port="/dev/ttyUSB0", calibration="left-calibration.json")
right = SO101(port="/dev/ttyUSB1", calibration="right-calibration.json")
bimanual = BimanualSO101(left=left, right=right)
```

## Leader motion and torque check

Calibrated normalized drivers declare `HandoverCapabilities` with coordinate
identifier `so101.normalized.v1` and per-joint tolerance and catch-up limits of
`0.5` normalized units. Uncalibrated raw-tick drivers do not declare compatible
coordinates. Bimanual drivers compose both arms' declarations and require both
leaders to acknowledge torque enable before either accepts a position command.

Leaders start with torque off. Enabling torque first latches the measured current
pose, so an old goal cannot pull a free leader before alignment begins. After
`set_torque(enabled=True)` succeeds, `send_action(target)` accepts positions in
the arm's own calibrated coordinates.
The driver writes only goal-position bytes; it does not change servo speed limits.
`goal_time` does not set hardware timing. Callers must pace intermediate targets
and check fresh measured positions rather than infer arrival from elapsed time.

Keep the movement path clear. A handover caller must hold the follower, check
coordinate compatibility and fresh feedback, bound catch-up motion, disable leader
torque before operator control, and attempt torque-off on failure or cancellation.
A failed acknowledgment leaves physical torque state uncertain.

Stop Studio and other controllers before the hardware check. Support the arm,
keep its emergency stop reachable, and provide its own calibration. The script
uses leader mode even on follower hardware, so connecting can let the arm fall.
From the repository root:

```bash
uv sync --extra transport --extra so101 --extra tests
uv run --no-sync pytest packages/physicalai-so101-plugin/tests tests/unit/robot/transport -q
uv run --no-sync python packages/physicalai-so101-plugin/examples/shared_robot_torque_smoke.py \
  --port /dev/serial/by-id/<leader> \
  --calibration ~/.cache/physicalai/robots/<leader-calibration>.json \
  --name so101-torque-test
```

Use a fresh owner name and start without an intentional movement. The script waits
for `CONNECT` while the arm is supported, then `ENABLE` before powering it. It stays
torque-on at the `DISABLE` prompt for inspection. Do not force the joints. Support
the arm before disabling torque, then inspect it before pressing Enter to disconnect.

For an optional small movement, pass one joint and a delta of at most one normalized
unit, such as `--joint shoulder_pan --delta 1`. The script also requires `MOVE` and
checks measured arrival before waiting at `DISABLE`. Without `--joint`, it sends
the current pose. That confirms action publication, not owner execution:
`SharedRobot.send_action()` is fire-and-forget.

Ctrl+C or closed terminal input attempts torque-off cleanup. Force-killing cannot
guarantee cleanup. If torque-off is not acknowledged, inspect and secure the arm
before continuing; software state does not prove hardware state. These checks do
not establish hardware safety certification.

## URDF assets

This package vendors the `Simulation/SO101` directory from
[`TheRobotStudio/SO-ARM100`](https://github.com/TheRobotStudio/SO-ARM100) at
revision `fda892cba81032c46c40976a48c9ceadbf40a9ca`.

```python
from physicalai_so101_plugin import get_urdf_path

follower_urdf = get_urdf_path() / "so101-follower/so101_follower.urdf"
leader_urdf = get_urdf_path() / "so101-leader/urdf/so101_leader.urdf"
bimanual_follower_urdf = get_urdf_path() / "so101-follower/so101_bimanual_follower.urdf"
bimanual_leader_urdf = get_urdf_path() / "so101-leader/urdf/so101_bimanual_leader.urdf"
```

The bimanual models mount the left and right arms 0.4 m apart. Each model uses
the matching role-specific single-arm mesh tree. Leader meshes live beside the
URDFs under `so101-leader/urdf/meshes/`; leader descriptions include
the leader handle and trigger meshes.

## Bimanual examples

The repository provides calibration examples and a teleoperation recipe under
[`examples/`](examples/). Run the recipe from the repository root after editing
the serial ports and calibration paths:

```bash
uv run physicalai run --config packages/physicalai-so101-plugin/examples/runtime/bimanual_teleop.yaml
```

## Development

```bash
uv sync
uv run pytest packages/physicalai-so101-plugin/tests
```
