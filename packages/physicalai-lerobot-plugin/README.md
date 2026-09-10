# PhysicalAI LeRobot Plugin

Bridges [LeRobot](https://github.com/huggingface/lerobot) robot and teleoperator configs into [PhysicalAI](https://github.com/openvinotoolkit/physicalai), the Python library and runtime for robot control, transport, and CLI workflows. It registers with [Physical AI Studio](https://github.com/open-edge-platform/physical-ai-studio), the application that discovers catalog plugins and provides robot setup, teleoperation, and workflow experiences. Part of the [physicalai-plugins](https://github.com/MarkRedeman/physicalai-plugins) monorepo.

[![PyPI version](https://img.shields.io/pypi/v/physicalai-lerobot-plugin.svg)](https://pypi.org/project/physicalai-lerobot-plugin/)
[![Python versions](https://img.shields.io/pypi/pyversions/physicalai-lerobot-plugin.svg)](https://pypi.org/project/physicalai-lerobot-plugin/)

## Overview

This plugin lets Studio users select supported LeRobot hardware from a
schema-driven UI, auto-resolve serial devices, and run them through a
PhysicalAI-compatible adapter without writing robot-specific glue code.

Each installed LeRobot follower robot and teleoperator is registered as a
Studio catalog entry:

- follower type: `LeRobot_<follower_type>_Follower`
- leader type: `LeRobot_<teleoperator_type>_Leader`

## Screenshots

_Placeholder images — replace them with real screenshots._

![LeRobot entries in the PhysicalAI Studio robot catalog](https://raw.githubusercontent.com/MarkRedeman/physicalai-plugins/main/screenshots/studio-catalog.png)

![A LeRobot robot in the PhysicalAI Studio catalog](https://raw.githubusercontent.com/MarkRedeman/physicalai-plugins/main/packages/physicalai-lerobot-plugin/screenshots/studio.png)

## Installation

```bash
uv add physicalai-lerobot-plugin
```

## Run with the PhysicalAI CLI

The [PhysicalAI CLI](https://github.com/openvinotoolkit/physicalai) `run`
subcommand executes a `RobotRuntime` from a YAML config. The bundled config
teleoperates a LeRobot follower with its matching leader teleoperator:

```bash
uv run physicalai run --config packages/physicalai-lerobot-plugin/examples/runtime/teleop.yaml
```

The example uses the `so101_follower` / `so101_leader` pair; set `config_type`
and `config_kwargs` to any of the bundled follower/leader types below. Press
`Ctrl+C` to stop.

## Third-party LeRobot extensions

Install this package first, then install a LeRobot extension into the same
Python environment. On Studio startup, the catalog invokes LeRobot's native
third-party discovery and imports installed extension packages with names that
start with `lerobot_robot_` or `lerobot_teleoperator_`.

For example, after installing a compatible LeSlider package, its registered
follower and leader types appear automatically in the Studio catalog. An
extension must import its config registration code from its package root and
use LeRobot's `register_subclass(...)` mechanism. Hardware SDK dependencies
remain the responsibility of the extension package.

## Bundled LeRobot followers

- `bi_so_follower`
- `bi_rebot_b601_follower`
- `bi_openarm_follower`
- `so100_follower`
- `so101_follower`
- `koch_follower`
- `omx_follower`
- `hope_jr_hand`
- `hope_jr_arm`
- `openarm_follower`
- `rebot_b601_follower`
- `reachy2`
- `earthrover_mini_plus`
- `unitree_g1`
- `lekiwi`
- `lekiwi_client`

## Bundled LeRobot leader teleoperators

- `so100_leader` (for `so100_follower`)
- `so101_leader` (for `so101_follower`)
- `koch_leader` (for `koch_follower`)
- `omx_leader` (for `omx_follower`)
- `openarm_leader` (for `openarm_follower`)
- `rebot_102_leader` (for `rebot_b601_follower`)
- `reachy2_teleoperator` (for `reachy2`)

## Payload models

### How payload fields are built

Payload classes are generated dynamically from each LeRobot config dataclass.

1. Fields are derived directly from the selected LeRobot config dataclass.
2. Complex fields are supported recursively:
   - nested dataclasses, `dict[...]`, `list[...]`, tuples, optional/union types
3. Required/default behavior:
   - fields without dataclass defaults are required
   - fields with defaults/factories are optional in payload
4. Runtime endpoint resolution:
   - `port` fields are resolved at build time through the Studio factory when
     possible (including nested bimanual arm configs)
   - for configs with a serial `port` field, the schema is annotated with a
     Studio "Select robot" connection section bound to that `port`; the payload
     model itself adds no extra Studio fields

### Common payload fields

There is no fixed cross-robot payload contract anymore. Fields come from the
selected LeRobot config type.

### Payload examples

#### 1) `LeRobot_so100_follower_Follower`

```json
{
  "port": "/dev/ttyACM0",
  "disable_torque_on_disconnect": true,
  "use_degrees": true,
  "id": "so100-main"
}
```

#### 2) `LeRobot_hope_jr_hand_Follower`

```json
{
  "port": "/dev/ttyACM1",
  "side": "left",
  "disable_torque_on_disconnect": true,
  "id": "hope-left-hand"
}
```

#### 3) `LeRobot_bi_so_follower_Follower` (nested bimanual config)

```json
{
  "left_arm_config": {
    "port": "/dev/ttyACM0",
    "disable_torque_on_disconnect": true
  },
  "right_arm_config": {
    "port": "/dev/ttyACM1",
    "disable_torque_on_disconnect": true
  },
  "id": "bi-so-main"
}
```

## Notes

- Test-only robots are intentionally not registered by this plugin.
- Payload models are generated from LeRobot config dataclasses, so fields can
  differ by robot type.

## Development

```bash
uv sync
uv run pytest
```

See [`docs/creating-a-studio-plugin.md`](../../docs/creating-a-studio-plugin.md) for the full plugin development guide.
