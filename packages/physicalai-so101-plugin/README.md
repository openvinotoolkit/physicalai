# PhysicalAI SO-101 Plugin

Physical AI Studio catalog definitions, hardware driver, and URDF assets for
the single-arm SO-101 robot, built for
[PhysicalAI](https://github.com/openvinotoolkit/physicalai).

## Installation

```bash
uv add physicalai-so101-plugin
```

Installing the package registers `SO101_Follower` and `SO101_Leader` with the
Physical AI Studio robot catalog and provides `SO101`, `SO101Calibration`, and
`SO101JointCalibration`.

```python
from physicalai_so101_plugin import SO101

robot = SO101(port="/dev/ttyUSB0", calibration="calibration.json")
```

## URDF assets

This package vendors the `Simulation/SO101` directory from
[`TheRobotStudio/SO-ARM100`](https://github.com/TheRobotStudio/SO-ARM100) at
revision `fda892cba81032c46c40976a48c9ceadbf40a9ca`.

```python
from physicalai_so101_plugin import get_urdf_path

follower_urdf = get_urdf_path() / "so101-follower/so101_follower.urdf"
leader_urdf = get_urdf_path() / "so101-leader/urdf/so101_leader.urdf"
```

## Development

```bash
uv sync
uv run pytest packages/physicalai-so101-plugin/tests
```
