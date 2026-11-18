# PhysicalAI Trossen Plugin

Trossen WidowX AI robot drivers and Physical AI Studio catalog definitions for
[PhysicalAI](https://github.com/openvinotoolkit/physicalai).

## Installation

```bash
uv add physicalai-trossen-plugin
```

The plugin depends on `trossen-arm`. That SDK does not currently publish
wheels for Python 3.14+, so use Python 3.12 or 3.13 when controlling a robot.

## Usage

```python
from physicalai_trossen_plugin import WidowXAI

robot = WidowXAI(ip="192.168.1.100", role="follower")
```

For two arms, construct `BimanualWidowXAI` from left and right `WidowXAI`
instances. Installing this distribution also registers the Trossen entries in
the Physical AI Studio robot catalog.

## Motion timing

The driver multiplies each action's `goal_time` by `goal_time_scale`, which
defaults to **3.0**. At Studio's 100 Hz robot control rate, a 10 ms command
therefore gets a 30 ms motion goal. Overlapping motion goals reduce stop-start
motion caused by scheduling jitter. This applies to follower position commands,
not homing, visual identification, or leader force feedback.

In Studio, enable **Advanced configuration** in the robot form and set
**Goal time multiplier**. For a bimanual robot, the setting applies to both arms.
Existing payloads without this field use the new default. Restart Studio after
updating the plugin to reload its payload schemas.

For standalone use, pass `goal_time_scale` to `WidowXAI`:

```python
robot = WidowXAI(ip="192.168.1.100", goal_time_scale=3.0)
```

Larger values increase motion overlap and smoothing but add tracking lag; `1.0`
restores unscaled action timing.

## URDF assets

This package vendors the generated URDFs and mesh assets required by its
catalog entries from Trossen `trossen_arm_description` revision
`21d8b360c211c2ad8a065d8f462cbec0207626e7`. The source license is preserved
at [`urdf/LICENSE`](urdf/LICENSE).

```python
from physicalai_trossen_plugin import get_urdf_path

urdf_root = get_urdf_path()
single_arm_follower = urdf_root / "urdf/generated/wxai/wxai_follower.urdf"
single_arm_leader = urdf_root / "urdf/generated/wxai/wxai_leader_left.urdf"
bimanual_follower = urdf_root / "urdf/generated/stationary_ai.urdf"
bimanual_leader = urdf_root / "urdf/generated/stationary_ai_leader.urdf"
```

The single-arm leader catalog entry uses the left-handed leader model. The
bimanual leader model includes left- and right-handed leader arms.

## Development

```bash
uv sync
uv run pytest packages/physicalai-trossen-plugin/tests
```
