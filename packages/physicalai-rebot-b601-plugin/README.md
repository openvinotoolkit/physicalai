# PhysicalAI reBot B601 Plugin

Third-party Seeed reBot B601 robot arm plugin for [PhysicalAI](https://github.com/openvinotoolkit/physicalai), the Python library and runtime for robot control, transport, and CLI workflows. It registers with [Physical AI Studio](https://github.com/open-edge-platform/physical-ai-studio), the application that discovers catalog plugins and provides robot setup, teleoperation, and workflow experiences. Part of the [physicalai-plugins](https://github.com/MarkRedeman/physicalai-plugins) monorepo.

[![PyPI version](https://img.shields.io/pypi/v/physicalai-rebot-b601-plugin.svg)](https://pypi.org/project/physicalai-rebot-b601-plugin/)
[![Python versions](https://img.shields.io/pypi/pyversions/physicalai-rebot-b601-plugin.svg)](https://pypi.org/project/physicalai-rebot-b601-plugin/)

## Features

- Concrete implementations of the `Robot` protocol — no inheritance or registration required
- B601-DM (Damiao) and B601-RS (RobStride) follower drivers
- Optional Star Arm leader → follower teleoperation
- Bundled URDFs for gravity compensation and kinematics

## Hardware

| Class         | Arm              | Motors                        | Protocol                     |
| ------------- | ---------------- | ----------------------------- | ---------------------------- |
| `ReBotB601DM` | B601-DM follower | Damiao (via `motorbridge`)    | POS_VEL / FORCE_POS          |
| `ReBotB601RS` | B601-RS follower | RobStride (via `motorbridge`) | MIT mode + gripper impedance |

## Screenshots

_Placeholder images — replace them with real screenshots._

![reBot B601 in the PhysicalAI Studio robot catalog](https://raw.githubusercontent.com/MarkRedeman/physicalai-plugins/main/screenshots/studio-catalog.png)

![Connecting to a reBot B601 in PhysicalAI Studio](https://raw.githubusercontent.com/MarkRedeman/physicalai-plugins/main/packages/physicalai-rebot-b601-plugin/screenshots/studio.png)

![reBot B601 leader-to-follower teleop in the PhysicalAI CLI](https://raw.githubusercontent.com/MarkRedeman/physicalai-plugins/main/packages/physicalai-rebot-b601-plugin/screenshots/cli-teleop.png)

## Installation

```bash
uv add physicalai-rebot-b601-plugin
```

`motorbridge` is included as a core dependency.

## Quick start

```python
import numpy as np
from physicalai.robot import Robot, connect
from physicalai_rebot_b601_plugin import ReBotB601DM

robot = ReBotB601DM(port="/dev/ttyACM0", can_adapter="damiao")

with connect(robot) as arm:
    obs = arm.get_observation()
    action = obs.joint_positions.copy()
    arm.send_action(action)
```

All classes satisfy `isinstance(robot, Robot)` — no inheritance or registration
required. Use with `physicalai.robot.connect` and `physicalai.robot.verify_robot`.

## Run with the PhysicalAI CLI

The [PhysicalAI CLI](https://github.com/openvinotoolkit/physicalai) `run`
subcommand executes a `RobotRuntime` from a YAML config. The bundled teleop configs
relay a `StarArm102HDLeader` to a B601 follower. They install
`physicalai-stararm-plugin` for the command only:

```bash
uv run --with physicalai-stararm-plugin physicalai run --config packages/physicalai-rebot-b601-plugin/examples/runtime/teleop-dm.yaml
uv run --with physicalai-stararm-plugin physicalai run --config packages/physicalai-rebot-b601-plugin/examples/runtime/teleop-rs.yaml
```

Press `Ctrl+C` to stop.

### Sinusoidal motion and joint reading

```bash
uv run physicalai run --config packages/physicalai-rebot-b601-plugin/examples/runtime/move-joints-dm.yaml
uv run physicalai run --config packages/physicalai-rebot-b601-plugin/examples/runtime/move-joints-rs.yaml
uv run physicalai run --config packages/physicalai-rebot-b601-plugin/examples/runtime/read-joints-dm.yaml
uv run physicalai run --config packages/physicalai-rebot-b601-plugin/examples/runtime/read-joints-rs.yaml
```

## URDF Models

Bundled URDF descriptions for gravity compensation and kinematics:

```python
from physicalai_rebot_b601_plugin import get_urdf_path

urdf_dir = get_urdf_path()

# B601-DM / fixend arm (for gravity compensation)
dm_urdf = urdf_dir / "rebot-b601-dm" / "urdf" / "reBot-DevArm_fixend.urdf"

# B601-RS arm
rs_urdf = urdf_dir / "rebot-b601-rs" / "urdf" / "00-arm-rs_asm-v3.urdf"

```

| URDF            | Model            | Use                                    |
| --------------- | ---------------- | -------------------------------------- |
| `rebot-b601-dm` | B601-DM (fixend) | Gravity compensation for `ReBotB601DM` |
| `rebot-b601-rs` | B601-RS v3       | Kinematics for `ReBotB601RS`           |

## Development

```bash
uv sync
uv run pytest packages/physicalai-rebot-b601-plugin/tests/
```

## Acknowledgments

URDF models for the reBot Arm B601 are from the
[reBotArm_control_py](https://github.com/vectorBH6/reBotArm_control_py) project,
released under the MIT License by vectorBH6.
