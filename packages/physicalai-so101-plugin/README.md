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

## Degree-mode deployment and teleoperation

Opt into body degree units through Python or a Runtime recipe:

```python
from physicalai_so101_plugin import SO101

leader = SO101("/dev/ttyACM0", role="leader", calibration="leader.json", unit="degrees")
follower = SO101("/dev/ttyACM1", calibration="follower.json", unit="degrees")
```

```yaml
class_path: physicalai_so101_plugin.SO101
init_args:
  port: /dev/ttyACM0
  role: leader
  calibration: leader.json
  unit: degrees
```

`unit` is the single units setting; there is no separate `use_degrees` flag.
Set both arms to matching units when teleoperating. Degree mode requires
calibration; explicit uncalibrated bringup still uses raw ticks. The default
remains normalized. Studio's single-arm SO101 catalog exposes the same
**Joint units** selector with help text explaining both choices. Existing payloads
without `unit` retain normalized behavior. The setup wizard continues streaming
normalized preview values regardless of the selected deployment units.

The convention matches LeRobot's degree conversion for non-inverted calibration:

- Body position: `(already_homed_tick - (range_min + range_max) / 2) * 360 / 4095`.
  Body reads are not clipped to the recorded range, and hardware homing is not
  subtracted again. Zero degrees is the recorded range midpoint, not an assumed
  fixed encoder tick or an absolute URDF joint angle.
- Body command: `int(degrees * 4095 / 360 + midpoint)`, truncating rather than
  rounding. Body conversion is not clipped to the recorded range; the existing
  driver's final 0–4095 encoder safety clamp still applies to hardware writes.
- The gripper **remains [0, 100]**, not degrees. Degree-mode gripper commands also
  use integer truncation. Observations remain canonically ordered float32 arrays.

These formulas are checked against numeric fixtures derived from
[LeRobot v0.6.0](https://github.com/huggingface/lerobot/blob/v0.6.0/src/lerobot/motors/motors_bus.py)
and the current SO leader `use_degrees` convention, without importing LeRobot in
the plugin or tests. Degree mode does not alter Runtime follower torque behavior,
setup torque-off behavior, or the existing normalized/inversion semantics.
This establishes units parity, not blanket driver equivalence: software/hardware
inversion, normal-driver error handling and disconnect behavior require separate
compatibility review. Policies and consumers must expect the selected mixed
body-degree/gripper-percentage units; normalized policies/previews are not
automatically converted.

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
