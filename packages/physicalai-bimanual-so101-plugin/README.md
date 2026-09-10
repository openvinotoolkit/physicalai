# PhysicalAI Bimanual SO-101 Plugin

Third-party bimanual SO-101 robot arm plugin for [PhysicalAI](https://github.com/openvinotoolkit/physicalai), the Python library and runtime for robot control, transport, and CLI workflows. It registers with [Physical AI Studio](https://github.com/open-edge-platform/physical-ai-studio), the application that discovers catalog plugins and provides robot setup, teleoperation, and workflow experiences. Part of the [physicalai-plugins](https://github.com/MarkRedeman/physicalai-plugins) monorepo.

[![PyPI version](https://img.shields.io/pypi/v/physicalai-bimanual-so101-plugin.svg)](https://pypi.org/project/physicalai-bimanual-so101-plugin/)
[![Python versions](https://img.shields.io/pypi/pyversions/physicalai-bimanual-so101-plugin.svg)](https://pypi.org/project/physicalai-bimanual-so101-plugin/)

## Features

- Composes two SO-101 arms (left + right) behind a single `Robot` protocol
- Follower and leader (read-only) roles for teleoperation
- Bundled dual-arm URDF for kinematics and visualization

## Hardware

| Class           | Robot                   | Motors          | Protocol |
| --------------- | ----------------------- | --------------- | -------- |
| `BimanualSO101` | Dual 6-DOF STS3215 arms | Feetech STS3215 | POSITION |

## Screenshots

_Placeholder images — replace them with real screenshots._

![Bimanual SO-101 in the PhysicalAI Studio robot catalog](https://raw.githubusercontent.com/MarkRedeman/physicalai-plugins/main/screenshots/studio-catalog.png)

![Connecting to a Bimanual SO-101 in PhysicalAI Studio](https://raw.githubusercontent.com/MarkRedeman/physicalai-plugins/main/packages/physicalai-bimanual-so101-plugin/screenshots/studio.png)

## Installation

```bash
uv add physicalai-bimanual-so101-plugin
```

`feetech-servo-sdk` is included as a core dependency.

## Calibration examples

Bundled SO-101 calibration files (LeRobot format) live under
[`examples/calibration/`](examples/calibration/):

| File                                 | Arm      | Use                        |
| ------------------------------------ | -------- | -------------------------- |
| `examples/calibration/follower.json` | Follower | Left + right follower arms |
| `examples/calibration/leader.json`   | Leader   | Left + right leader arms   |

Pass them as the `calibration` path when constructing a `SO101`:

```python
from physicalai.robot.so101 import SO101, SO101Calibration

cal = SO101Calibration.from_path("examples/calibration/follower.json")
arm = SO101(port="/dev/ttyACM0", calibration=cal, role="follower")
```

The bundled runtime config (`examples/runtime/teleop.yaml`) references these
files already.

## Quick start

### Calibrated follower

```python
import numpy as np
from physicalai.robot import connect
from physicalai_bimanual_so101_plugin import BimanualSO101

# Build left and right arms manually
from physicalai.robot.so101 import SO101, SO101Calibration

left_cal = SO101Calibration.from_path("left_calibration.json")
right_cal = SO101Calibration.from_path("right_calibration.json")

left = SO101(port="/dev/ttyACM0", calibration=left_cal, role="follower")
right = SO101(port="/dev/ttyACM1", calibration=right_cal, role="follower")

robot = BimanualSO101(left=left, right=right)

with connect(robot) as arm:
    obs = arm.get_observation()     # shape (12,)
    action = obs.joint_positions.copy()
    arm.send_action(action)
```

### Uncalibrated bringup mode

```python
from physicalai.robot import connect
from physicalai_bimanual_so101_plugin import BimanualSO101
from physicalai.robot.so101 import SO101

left = SO101.uncalibrated(port="/dev/ttyACM0", role="follower")
right = SO101.uncalibrated(port="/dev/ttyACM1", role="follower")

robot = BimanualSO101(left=left, right=right)
```

### Leader (read-only teleoperation)

```python
from physicalai.robot import connect
from physicalai.robot.so101 import SO101
from physicalai_bimanual_so101_plugin import BimanualSO101

left = SO101.uncalibrated(port="/dev/ttyACM0", role="leader")
right = SO101.uncalibrated(port="/dev/ttyACM1", role="leader")

robot = BimanualSO101(left=left, right=right)

with connect(robot) as arm:
    while True:
        obs = arm.get_observation()
        print(obs.joint_positions)
```

### Teleoperation

Because `BimanualSO101` satisfies the `Robot` protocol, it can be driven by
the built-in `physicalai.runtime.TeleopSource` with a leader `BimanualSO101`
(both arms in `role="leader"`), e.g. via `physicalai run --config` or directly
in Python:

```python
from physicalai.robot.so101 import SO101
from physicalai.runtime import RobotRuntime, TeleopSource
from physicalai_bimanual_so101_plugin import BimanualSO101

follower = BimanualSO101(
    left=SO101.uncalibrated(port="/dev/ttyACM0", role="follower"),
    right=SO101.uncalibrated(port="/dev/ttyACM1", role="follower"),
)
leader = BimanualSO101(
    left=SO101.uncalibrated(port="/dev/ttyUSB0", role="leader"),
    right=SO101.uncalibrated(port="/dev/ttyUSB1", role="leader"),
)

runtime = RobotRuntime(fps=30, robot=follower, action_source=TeleopSource(leader=leader))
with runtime:
    runtime.run()
```

The same teleoperation can be run from a config with the
[PhysicalAI CLI](https://github.com/openvinotoolkit/physicalai):

```bash
uv run physicalai run --config packages/physicalai-bimanual-so101-plugin/examples/runtime/teleop.yaml
```

Press `Ctrl+C` to stop.

## URDF Model

```python
from physicalai_bimanual_so101_plugin import get_urdf_path

urdf_path = get_urdf_path()
```

| URDF                         | Model           | Use                        |
| ---------------------------- | --------------- | -------------------------- |
| `so101_dual/so101_dual.urdf` | Bimanual SO-101 | Kinematics & visualization |

Two arms mounted 0.4 m apart: left at +0.2 m (Y), right at -0.2 m (Y).

## Joint Order

Left arm (indices 0-5): `left_shoulder_pan`, `left_shoulder_lift`, `left_elbow_flex`, `left_wrist_flex`, `left_wrist_roll`, `left_gripper`

Right arm (indices 6-11): `right_shoulder_pan`, `right_shoulder_lift`, `right_elbow_flex`, `right_wrist_flex`, `right_wrist_roll`, `right_gripper`

## Development

```bash
uv sync
uv run pytest packages/physicalai-bimanual-so101-plugin/tests/
```

## Acknowledgments

URDF model generated from Onshape via onshape-to-robot.
SO-101 arm by [LeRobot](https://github.com/huggingface/lerobot).
