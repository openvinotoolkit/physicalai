# Physical AI Studio Plugin

Types, protocols, and utilities for building robot catalog plugins for
**Physical AI Studio**.

For installing curated or unofficial plugins in Studio, and for registering a
plugin in Studio's curated manifest, see
[`application/docs/robot-plugins.md`](https://github.com/open-edge-platform/physical-ai-studio/blob/main/application/docs/robot-plugins.md).

External robot types register themselves with Studio through an
[entry-point](#entry-point-registration) mechanism. This lets Studio discover,
configure, and drive them without modifying Studio's internal code.

---

## Installation

```bash
uv add physicalai-studio-plugin
```

Requires Python 3.12+. Dependencies are `pydantic>=2.12` and `physicalai`.

---

## Quick Start

A minimal plugin has this structure:

```text
physicalai-my-robot-plugin/
├── pyproject.toml
├── README.md
└── src/
    └── physicalai_my_robot_plugin/
        ├── __init__.py
        └── studio_catalog.py
```

### `pyproject.toml`

```toml
[project]
name = "physicalai-my-robot-plugin"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "physicalai",
    "physicalai-studio-plugin",
]

[project.entry-points."physicalai.studio.catalog_plugins"]
my-robot = "physicalai_my_robot_plugin.studio_catalog:register_physicalai_studio_plugin"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

### `studio_catalog.py`

```python
from __future__ import annotations

from pathlib import Path
from typing import Any

from physicalai.robot.interface import Robot as PhysicalAIRobot
from physicalai_studio_plugin import (
    CatalogRobot,
    CatalogRobotFactory,
    PortScanner,
    RobotAdapterOptions,
    RobotAsset,
    RobotCatalogDefinition,
    RobotProbe,
    SerialPortInfo,
)
from pydantic import BaseModel, Field


class MyRobotPayload(BaseModel):
    connection_string: str = ""
    serial_number: str = Field(...)


async def _build_my_robot(
    robot: CatalogRobot[MyRobotPayload],
    factory: CatalogRobotFactory,
) -> PhysicalAIRobot:
    port = await factory.find_port(
        SerialPortInfo(
            connection_string=robot.payload.connection_string or None,
            serial_number=robot.payload.serial_number or None,
        )
    )
    if port is None:
        msg = f"Robot not found: {robot.payload.serial_number}"
        raise RuntimeError(msg)
    # Return a plain Physical AI driver. Studio owns the hardware process.
    return MyRobotDriver(port=port)


class MyRobotProbe:
    """Structurally implements RobotProbe[MyRobotPayload]."""

    async def discover(self, manager: PortScanner) -> list[SerialPortInfo]:
        await manager.find_robots()
        return manager.robots

    async def identify(
        self, payload: MyRobotPayload, manager: PortScanner | None, joint: str | None = None
    ) -> None:
        pass

    async def is_online(
        self, payload: MyRobotPayload, manager: PortScanner | None = None
    ) -> bool:
        return True


def _definitions() -> list[RobotCatalogDefinition[MyRobotPayload]]:
    return [
        RobotCatalogDefinition[MyRobotPayload](
            type="MyRobot_Follower",
            display_name="My Robot Follower",
            role="follower",
            robot_builder=_build_my_robot,
            robot_payload=MyRobotPayload,
            asset=RobotAsset(
                urdf_relative_path=Path("my_robot/model.urdf"),
                packages={"my_robot": Path("my_robot")},
                joint_map={"gripper.pos": ["gripper"]},
                root_resolver=lambda: Path("/path/to/urdf"),
            ),
            adapter_options=RobotAdapterOptions(include_velocities=True),
            probe=MyRobotProbe(),
        ),
    ]


def register_physicalai_studio_plugin(registry: Any) -> None:
    for definition in _definitions():
        registry.register_robot(definition)
```

---

## API Reference

### Robot Form UI Schema

`robot_payload_ui(...)` defines optional ordered form presentation metadata for
a Pydantic payload model. Before publishing a plugin, call
`validate_robot_payload_ui(MyPayload)` in a test to verify that item shapes,
field references, connection bindings, and ownership are valid. Studio also
validates payload UI metadata during catalog registration and rejects invalid
robot definitions with an actionable error.

### `RobotCatalogDefinition`

The primary data class that describes a robot type to Studio. It is generic over
the payload model: use `RobotCatalogDefinition[MyRobotPayload]` to link the
payload, probe, and robot builder types together.

```python
@dataclass
class RobotCatalogDefinition(Generic[_PayloadT]):
    type: str                        # Unique identifier, e.g. "MyRobot_Follower"
    display_name: str                # Human-readable name
    role: Literal["follower", "leader"]
    robot_builder: BuildRobotCallable | None = None
    robot_payload: type[_PayloadT] | None = None
    asset: RobotAsset | None = None
    adapter_options: RobotAdapterOptions = field(default_factory=RobotAdapterOptions)
    probe: RobotProbe[_PayloadT] | None = None
    zero_calibration: RobotZeroCalibration | None = None
    simulation: SimulationLaunch | None = None
```

| Field              | Description                                                                                         |
| ------------------ | --------------------------------------------------------------------------------------------------- |
| `type`             | Stable identifier used in DB storage and API paths. Must be unique across all plugins.              |
| `display_name`     | Human-readable name shown in the Studio UI.                                                         |
| `role`             | Either `"follower"` (executes actions) or `"leader"` (provides demonstrations).                     |
| `robot_builder`    | Async callable that receives a robot payload and factory, then returns a `PhysicalAIRobot`.         |
| `robot_payload`    | Pydantic model defining this robot type's configuration fields.                                     |
| `asset`            | Optional URDF and package maps for 3D visualization.                                                |
| `adapter_options`  | Controls velocity, timing, and effort-forwarding behavior.                                          |
| `probe`            | Optional [`RobotProbe[_PayloadT]`](#robotprobe) for device interaction.                             |
| `zero_calibration` | Optional [`RobotZeroCalibration`](#robotzerocalibration) for Studio's guided zero-pose calibration. |
| `simulation`       | Optional [`SimulationLaunch`](#simulationlaunch) that Studio uses to start a simulated robot.       |

### `RobotAdapterOptions`

```python
@dataclass(frozen=True)
class RobotAdapterOptions:
    include_velocities: bool = False
    goal_time_scale: float = 1.0
    external_effort_gain: float | None = 0.1
```

### `RobotAsset`

```python
@dataclass(frozen=True)
class RobotAsset:
    urdf_relative_path: Path
    packages: dict[str, Path]
    joint_map: dict[str, list[str]]
    root_resolver: Callable[[], Path] | None = None
```

| Field                | Description                                                            |
| -------------------- | ---------------------------------------------------------------------- |
| `urdf_relative_path` | Path to the URDF file, relative to the packages root.                  |
| `packages`           | Maps ROS package names to their filesystem paths.                      |
| `joint_map`          | Maps Studio observation keys, such as `"gripper.pos"`, to URDF joints. |
| `root_resolver`      | Callable returning the root directory used to resolve URDF paths.      |

### `RobotZeroCalibration`

```python
@dataclass(frozen=True)
class RobotZeroCalibration(Generic[_RobotT]):
    instructions: str
    set_zero: Callable[[_RobotT], Awaitable[None]]
    release: Callable[[_RobotT], Awaitable[None]] | None = None
    zero_tolerance_deg: float = 5.0
```

For arms whose calibration is a zero pose stored on the motors. Parameterize it with the driver class `robot_builder` returns so the steps can call that driver's methods. When a robot type sets `zero_calibration`, Studio offers a guided step while the robot is added: it opens its own exclusive connection to the driver from `robot_builder`, calls `release` so the arm can be moved by hand, and shows `instructions`, next to a live 3D view when the type has a `RobotAsset`. When the user confirms the pose it calls `set_zero`, then checks that every joint reads within `zero_tolerance_deg` of zero.

```python
async def _release(robot: MyRobot) -> None:
    await asyncio.to_thread(robot.disable_torque)


async def _set_zero(robot: MyRobot) -> None:
    await asyncio.to_thread(robot.set_zero_position)


zero_calibration = RobotZeroCalibration[MyRobot](
    instructions="Move the arm to its rest pose and close the gripper.",
    release=_release,
    set_zero=_set_zero,
)
```

### `SimulationLaunch`

```python
@dataclass(frozen=True)
class SimulationScene:
    id: str
    display_name: str
    description: str = ""


class SimulationArgvBuilder(Protocol):
    def __call__(self, *, scene: str, owner_name: str, seed: int | None) -> Sequence[str]: ...


@dataclass(frozen=True)
class SimulationLaunch(Generic[_PayloadT]):
    scenes: tuple[SimulationScene, ...]
    default_scene: str
    payload_owner_name: Callable[[_PayloadT], str]
    build_argv: SimulationArgvBuilder
    max_seed: int | None = None
    labels: tuple[str, ...] = ()

    def owner_name(self, payload: _PayloadT) -> str: ...
    def argv(self, scene: str, payload: _PayloadT, seed: int | None = None) -> list[str]: ...
```

For simulated robot types whose `robot_builder` attaches to a running simulation. Parameterize it with the payload model, like the definition. When a robot type sets `simulation`, Studio can start that simulation itself:

- The start dialog shows `labels` (short facts such as the arm count) as chips and `scenes` in their order, with `default_scene` preselected.
- Studio runs `argv(scene, payload, seed)` for the robot's payload as a child process. `payload_owner_name` maps the payload to the name the robot attaches to; `build_argv` gets that name, and `owner_name(payload)` returns it so Studio can check `ready.name`.
- `seed` is `None` for random scene layouts, or a fixed seed from `0` to `max_seed`. With `max_seed=None` the simulation takes no seed and Studio offers none.

`SimulationLaunch` rejects an empty `scenes`, duplicate scene ids, a `default_scene` that is not one of them, a negative `max_seed` and blank labels; `argv` rejects an unknown scene, a blank owner name and an out-of-range seed with `ValueError`.

The command line must follow this process contract:

- **stdin** is a pipe that Studio keeps open and never writes to. The simulation stops when stdin reaches end of file, which also happens when Studio exits or crashes.
- **stdout** carries one JSON object per line, flushed after each line; logs go to stderr. Studio reads these events:
  - `{"event": "phase", "phase": "<name>"}` while the simulation starts, shown as progress; a phase may add fields such as download `bytes` and `total`.
  - `{"event": "ready", ...}` once the robot can be attached to, with `name` (the owner name), `http_url` (the simulation's HTTP server, or `null`), `viewer_url` (its browser viewer, or `null`) and `cameras` (the camera names it streams). Other fields are allowed.
  - `{"event": "error", "message": "<text>"}` when starting fails; Studio shows `message`.
- Every run ends its events with exactly one `ready` or one `error`, including a run whose parent went away before it was ready.
- The process **exits with a non-zero code** when it fails.
- **Studio uses the addresses from `ready`, not the payload's.** A simulation may pick free ports (the MuJoCo plugin's argv asks for them), so after `ready` Studio copies `http_url` and `viewer_url` onto the robot's payload (or otherwise uses them for cameras and the embedded viewer). The payload's defaults, such as the MuJoCo payload's `http_url` of `http://127.0.0.1:8080`, describe a simulation started by hand.

The MuJoCo plugin is the first implementation; its `start --status-json` events are described in [Start from another program](../physicalai-mujoco-plugin/README.md#start-from-another-program).

```python
class MySimPayload(BaseModel):
    sim_id: str = "1"


def _start_argv(*, scene: str, owner_name: str, seed: int | None) -> list[str]:
    argv = [sys.executable, "-m", "my_sim", "start", "--scene", scene, f"--name={owner_name}"]
    return argv if seed is None else [*argv, "--seed", str(seed)]


simulation = SimulationLaunch[MySimPayload](
    scenes=(SimulationScene(id="pick_place", display_name="Pick & Place", description="One block and a target"),),
    default_scene="pick_place",
    payload_owner_name=lambda payload: f"my-sim-{payload.sim_id}",
    build_argv=_start_argv,
    max_seed=2**32 - 1,
    labels=("1 arm",),
)
```

### `RobotProbe`

```python
@runtime_checkable
class RobotProbe(Protocol[_PayloadT]):
    async def discover(self, manager: PortScanner) -> list[SerialPortInfo]: ...
    async def identify(
        self, payload: _PayloadT, manager: PortScanner | None, joint: str | None = None
    ) -> None: ...
    async def is_online(
        self, payload: _PayloadT, manager: PortScanner | None = None
    ) -> bool: ...
```

Generic protocol over your robot's payload model. Implement it structurally — your class receives the typed payload directly instead of a raw dict. The `_PayloadT` type parameter is automatically inferred from `identify` / `is_online` signatures; you do not need to explicitly inherit from `RobotProbe`.

### `PortScanner`

```python
class PortScanner(Protocol):
    async def find_robots(self) -> None: ...
    @property
    def robots(self) -> list[SerialPortInfo]: ...
```

Duck-type protocol for serial/network port scanners. Call `find_robots()` to refresh the device list, then read `robots` for the current results.

### `CatalogRobotFactory`

```python
class CatalogRobotFactory(Protocol):
    async def find_port(self, port_info: SerialPortInfo) -> str | None: ...
```

Factory protocol passed to your `robot_builder` callable. Use `find_port(SerialPortInfo(...))` to resolve a connection by serial number and/or configured connection string. Calibration data is now embedded in the robot payload model and does not require a factory method.

### `SerialPortInfo`

```python
class SerialPortInfo(BaseModel):
    connection_string: str | None
    serial_number: str | None
```

Describes a discovered serial or network connection.

### `PayloadContainer` / `CatalogRobot`

```python
class PayloadContainer(Protocol[_PayloadT]):
    payload: _PayloadT

class CatalogRobot(PayloadContainer[_PayloadT], Protocol[_PayloadT]):
    type: str
```

Protocols for the robot descriptor passed to `robot_builder`. The `payload` is an instance of your `robot_payload` model.

### `BuildRobotCallable`

```python
BuildRobotCallable = Callable[[_RobotT, _FactoryT], Awaitable[PhysicalAIRobot]]
```

Type alias for the `robot_builder` callable signature.

---

## Entry Point Registration

Studio discovers plugins via Python [entry points](https://packaging.python.org/en/latest/specifications/entry-points/) in the group `physicalai.studio.catalog_plugins`.

In your `pyproject.toml`:

```toml
[project.entry-points."physicalai.studio.catalog_plugins"]
my-robot = "physicalai_my_robot_plugin.studio_catalog:register_physicalai_studio_plugin"
```

The callable must accept a single argument — the registry — and call `registry.register_robot(definition)` for each robot type:

```python
def register_physicalai_studio_plugin(registry: Any) -> None:
    for definition in _definitions():
        registry.register_robot(definition)
```

Studio calls all discovered entry points at startup. Duplicate `type` values raise a `ValueError`.

---

## Robot Builder Pattern

The `robot_builder` is an async function that receives a robot descriptor and a factory, performs connection setup, and returns a `PhysicalAIRobot`:

```python
from physicalai_studio_plugin import CatalogRobot

async def _build_my_robot(
    robot: CatalogRobot[MyRobotPayload],
    factory: CatalogRobotFactory,
) -> PhysicalAIRobot:
    # `robot.payload` is already a validated MyRobotPayload instance.
    # 1. Resolve the connection
    port = await factory.find_port(
        SerialPortInfo(
            connection_string=robot.payload.connection_string or None,
            serial_number=robot.payload.serial_number or None,
        )
    )
    if port is None:
        msg = f"Robot not found: {robot.payload.serial_number}"
        raise RuntimeError(msg)

    # 2. Return the driver
    return MyRobotDriver(port=port, ...)
```

---

## Testing

Create a minimal test file alongside your plugin:

```python
from __future__ import annotations

from physicalai_studio_plugin import RobotCatalogDefinition, SerialPortInfo


def _fake_registry():
    class _FakeRegistry:
        def __init__(self):
            self.definitions: list[RobotCatalogDefinition] = []

        def register_robot(self, definition: RobotCatalogDefinition) -> None:
            self.definitions.append(definition)

    return _FakeRegistry()


def test_plugin_registration():
    from physicalai_my_robot_plugin.studio_catalog import (
        register_physicalai_studio_plugin,
    )

    registry = _fake_registry()
    register_physicalai_studio_plugin(registry)
    assert len(registry.definitions) == 1
    assert registry.definitions[0].type == "MyRobot_Follower"
```
