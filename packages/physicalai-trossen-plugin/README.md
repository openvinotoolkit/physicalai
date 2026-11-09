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
