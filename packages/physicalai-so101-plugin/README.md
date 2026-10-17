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
