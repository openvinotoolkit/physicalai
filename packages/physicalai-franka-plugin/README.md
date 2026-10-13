# PhysicalAI Franka Plugin

[Franka Research 3](https://franka.de/research) robot arm plugin for [PhysicalAI](https://github.com/openvinotoolkit/physicalai). It registers single-arm and bimanual FR3 followers with [Physical AI Studio](https://github.com/open-edge-platform/physical-ai-studio).

[![PyPI version](https://img.shields.io/pypi/v/physicalai-franka-plugin.svg)](https://pypi.org/project/physicalai-franka-plugin/)
[![Python versions](https://img.shields.io/pypi/pyversions/physicalai-franka-plugin.svg)](https://pypi.org/project/physicalai-franka-plugin/)

## Features

- Concrete implementation of the `Robot` protocol — no inheritance or registration required
- 7-DOF FR3 follower over the Franka Control Interface (FCI) via the official `pylibfranka` bindings
- Bimanual left/right FR3 follower with deterministic left-then-right action layout
- Optional Franka Hand gripper commanded by width as an extra action element
- 1 kHz joint position control loop with `goal_time`-based interpolation
- Bundled FR3 URDF for the PhysicalAI Studio 3D viewer

## Hardware

| Class       | Arm                                         | Protocol              | Units                                             |
| ----------- | ------------------------------------------- | --------------------- | ------------------------------------------------- |
| `FrankaFR3` | FR3 follower (7-DOF + optional Franka Hand) | FCI via `pylibfranka` | Arm positions in degrees; gripper width in meters |
| `BimanualFrankaFR3` | Left/right FR3 followers | Two FCI connections | Left arm values followed by right arm values |

## Installation

```bash
uv add physicalai-franka-plugin[franka]
```

`pylibfranka` is an optional extra because its wheels currently target Python
3.9–3.12 only. Install the plugin without the extra to use the catalog and
driver classes on any supported Python; add the extra (on Python ≤ 3.12) to
control a real robot.

## Quick start

```python
import numpy as np
from physicalai.robot import Robot, connect
from physicalai_franka_plugin import FrankaFR3

robot = FrankaFR3(host="172.16.0.2", gripper_enabled=False)

with connect(robot) as arm:
    obs = arm.get_observation()
    action = obs.joint_positions.copy()
    arm.send_action(action)
```

All classes satisfy `isinstance(robot, Robot)` — no inheritance or registration
required. Use with `physicalai.robot.connect` and `physicalai.robot.verify_robot`.

## Run with the PhysicalAI CLI

The [PhysicalAI CLI](https://github.com/openvinotoolkit/physicalai) `run`
subcommand executes a `RobotRuntime` from a YAML config:

```bash
uv run physicalai run --config packages/physicalai-franka-plugin/examples/runtime/move-joints.yaml
uv run physicalai run --config packages/physicalai-franka-plugin/examples/runtime/read-joints.yaml
uv run physicalai run --config packages/physicalai-franka-plugin/examples/runtime/bimanual-read-joints.yaml
```

Press `Ctrl+C` to stop. The arm WILL move with `move-joints.yaml` — keep the
workspace clear.

## Gello DUO teleoperation

`GelloLeader` and `BimanualGelloLeader` are passive USB/serial drivers for the
Franka seven-axis GELLO and GELLO Duo. Each arm uses IDs 1–7 for joints and ID
8 for the trigger. The default calibration matches the preassembled Franka
GELLO; custom values can be uploaded in Studio’s advanced **Calibration**
field.

### Calibrating a GELLO

Use the non-actuating calibration command after self-assembly, motor/flange
changes, or when the default mapping does not match the physical leader:

```bash
physicalai-gello-calibrate \
  --port /dev/serial/by-id/usb-ROBOTIS_OpenRB-150_... \
  --side left \
  --output left-gello-calibration.json
```

For a Duo, run it once per USB adapter using `--side left` and `--side right`.
The command pings IDs 1–8, disables torque, reads the official reference pose,
captures released and pressed trigger positions, displays the generated JSON,
and writes only after confirmation. Upload the result through Studio’s
calibration control; it is not read from a runtime filesystem path.

It never enables torque or sends a position command. Use `/dev/serial/by-id/`
paths rather than transient `/dev/ttyACM*` names.

The bundled GELLO visual assets are not collision-certified; verify the physical
mounting and link geometry before using them for collision-sensitive setup.

## URDF Model

Bundled FR3 description for the PhysicalAI Studio 3D viewer:

```python
from physicalai_franka_plugin import get_urdf_path

fr3_urdf = get_urdf_path() / "franka_description" / "urdf" / "fr3.urdf"
```

| URDF                 | Model             | Use                          |
| -------------------- | ----------------- | ---------------------------- |
| `franka_description` | Franka Research 3 | 3D preview and joint mapping |

## Development

```bash
uv sync
uv run pytest packages/physicalai-franka-plugin/tests/
```

## Acknowledgments

The FR3 URDF and meshes are from the
[franka_description](https://github.com/frankarobotics/franka_description)
repository, Copyright Franka Robotics GmbH, released under the Apache License 2.0
(see the bundled `urdf/franka_description/LICENSE`).
