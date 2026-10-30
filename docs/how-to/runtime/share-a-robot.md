# Share a Robot Across Processes

`SharedRobot` lets one process own a robot's exclusive hardware connection
while any number of other processes read its state and send actions over
[Zenoh](https://zenoh.io/). It satisfies the same `Robot` protocol as a
direct driver, so it is a drop-in replacement anywhere a robot is expected
(including `RobotRuntime`).

## Install

```bash
pip install "physicalai[transport]"
```

## Create or attach

Every `SharedRobot` has a required, caller-chosen logical `name` — it keys
the Zenoh topics directly. The first `SharedRobot` constructed for a given
`name` that finds no existing owner spawns one (in a detached subprocess);
later instances (same or different process, same `name`) attach to it.

Construction uses `robot=` or `from_config()`. Prefer `from_config()` when you
already have a recipe. A disconnected `@export_config` driver can be passed
directly to the constructor or exported explicitly:

```python
import numpy as np
from physicalai.config import Config
from physicalai.robot import SharedRobot
from physicalai_so101_plugin import SO101

driver = SO101(
    port="/dev/ttyUSB0",
    calibration="~/.cache/calibration/so101.json",  # path stays relative/as given
)
robot = SharedRobot.from_config(driver, name="left-arm")
# equivalent: SharedRobot.from_config(Config.from_instance(driver), name="left-arm")
# or: SharedRobot("left-arm", robot={"class_path": "physicalai.robot.SO101", "init_args": {...}})
robot.connect()

obs = robot.get_observation()          # pull latest state, non-blocking
robot.send_action(np.asarray(obs.joint_positions), goal_time=0.1)

robot.disconnect()                     # detaches; the owner keeps running
```

Any importable `@export_config` robot class works (including third-party
plugins) — pass its public `class_path` + `init_args`; there is no flat
`robot_class` / `robot_kwargs` API.

## Serve a robot in the foreground

Use the operator command when a shell, systemd, Docker, or Kubernetes should own the
robot lifecycle:

```bash
physicalai robot serve --config examples/so101/serve.yaml
```

The command constructs and connects the driver in its own foreground process. Normal
output reports readiness, state-subscriber presence changes, a health summary every
30 seconds, and clean shutdown. Add `--verbose` for startup and cleanup details. The
command does not daemonize; use your service manager for background supervision.

List reachable owners without importing their advertised driver class:

```bash
physicalai robot discover
physicalai robot discover --json
physicalai robot discover --allow_remote
```

Discovery is local-only unless `--allow_remote` is explicit. Human output is a sorted
ASCII table. JSON mode writes one sorted array to stdout, including `[]` when no robot
answers.

Notes:

- `get_observation()` returns the newest owner-published state; if no new
  sample arrived since the last call, the cached last-known observation is
  returned. Staleness is visible via `obs.timestamp`.
- `obs.state` is the **owner-computed** state vector (e.g. WidowXAI ships
  positions + velocities, 14 values), so model inputs are correct without
  robot-specific logic on the subscriber side.
- `obs.images` is always `None` — camera frames go through `SharedCamera`
  (the capture transport), not the robot transport.
- Actions are **absolute joint targets**, delivered latest-wins and
  fire-and-forget. When no action is pending, the owner holds the last
  commanded position.
- `get_fresh_observation()` requests a new capture on the owner thread. It does
  not use the subscriber's cached stream or a read already in flight. Use it for
  causal checks after torque changes; counting subsequent stream packets is not
  enough. Buffered drivers can provide the same method, and composites recurse
  into their components. Errors and unsupported older owners raise rather than
  falling back to cached state. The read-only `/state` query accepts no payload
  and has a bounded pending queue.
- `set_torque(enabled=...)` is a synchronous owner request and reports
  whether the driver applied it; it raises when the driver has no torque
  control. Driver-specific torque gating belongs to the robot plugin.

## Declare handover capabilities

Any robot plugin can expose an optional `handover_capabilities` property returning
`physicalai.robot.HandoverCapabilities`. For example, a two-joint position driver
can declare:

```python
from physicalai.robot import HandoverCapabilities

@property
def handover_capabilities(self) -> HandoverCapabilities:
    return HandoverCapabilities(
        position_space="my-arm.radians.v1",
        position_tolerance=(0.01, 0.01),
        max_position_step=(0.005, 0.005),
        supports_alignment=True,
    )
```

Use the same `position_space` only for matching units, directions, zero convention,
and gripper mapping. Callers must also compare ordered joint names. Do not put
ports, calibration paths, or per-device calibration data in this identifier.
The two limit vectors contain one finite positive value per joint, in its own
coordinates. Mixed angular and linear joints can use different limits.
`max_position_step` bounds extra follower catch-up per control tick, not new
leader motion or absolute teleoperation speed.

`supports_alignment=True` promises position commands and an acknowledged
`set_torque(enabled=...)` method. Enable must report partial failures; disable
must attempt every actuator. Unsupported or passive leaders can declare compatible
coordinates with `supports_alignment=False` for an operator-controlled handover.
Return `None`, or omit the property, when coordinate compatibility is unknown.

The owner validates these declarations and publishes them in `/metadata`.
`SharedRobot.handover_capabilities` exposes the validated result after connection;
it returns `None` for older owners without declarations. `CompositeRobot` and
`BimanualRobot` combine component coordinate identifiers and limits in joint order.
They advertise powered alignment only when every component supports it, roll back
partial torque enable, and attempt every component during torque-off. Their
observation timestamp advances only when every component advances, so one live
arm cannot hide stale feedback from another. Cached reads retain the complete
accepted snapshot, not older positions paired with a newer capture timestamp.

Callers still own follower hold, fresh-feedback checks, bounded motion, explicit
operator release, and torque-off cleanup. Declarations do not prove calibration
quality or physical torque state. See the [SO-101 plugin hardware check](../../../packages/physicalai-so101-plugin/README.md#leader-motion-and-torque-check)
for a model-specific example; the transport does not import that driver.

## Attach to a known owner

For manually-launched or remote owners, attach by name only — no
construction recipe needed:

```python
from physicalai.robot import SharedRobot

robot = SharedRobot.attach("left-arm")
robot.connect()
```

Enumerate reachable robots with:

```python
from physicalai.robot.transport import discover_robots

for metadata in discover_robots():
    print(metadata["name"], metadata["robot_class"], metadata["joint_names"])
```

## Reuse a remote client

For a long-running process that repeatedly discovers or attaches to remote
owners, use `SharedRobotClient(allow_remote=True)`. It keeps one Zenoh
scouting session open, so the first operation establishes remote routes and
later operations reuse them:

```python
from physicalai.robot.transport import SharedRobotClient

with SharedRobotClient(allow_remote=True) as client:
  robots = client.discover()
  robot = client.attach(robots[0]["name"])
  robot.connect()

  # Later discovery reuses the established remote session.
  robots = client.discover()
```

`SharedRobotClient` is attach-only: it never starts an owner process. It
disconnects every robot created through `attach()` before closing its shared
session on context exit. Its first `discover()` call uses a one-second budget
for Zenoh scouting; later calls use 0.1 seconds with the warmed session.
Pass `timeout=` to override either budget. The timeout remains a wildcard-query
collection window, so use a larger explicit value when a more complete inventory
is needed after a network or owner change. `SharedRobotClient()` also works for
same-host owners; omit `allow_remote=True` to keep that client local-only.

## Network scope: local-only by default

`allow_remote=False` (the default) keeps the owner's Zenoh session
unreachable off-host — multicast/gossip scouting is disabled and the owner
listens on `127.0.0.1` only. Same-host spawn-or-attach still works without
depending on multicast: owner and subscriber derive the same deterministic
loopback port from `name`.

Opt into cross-host reachability explicitly when you need it:

```python
robot = SharedRobot.from_config(
    {
        "class_path": "physicalai.robot.SO101",
        "init_args": {"port": "/dev/ttyUSB0", "calibration": "calibration.json"},
    },
    name="left-arm",
    allow_remote=True,
)
```

Each caller has its own Zenoh session. For an attacher, `allow_remote`
controls only that session's ability to find remote owners. For the caller
that spawns an owner, it also fixes the owner's reachability for its
entire lifetime. Later attachers cannot change an existing owner's scope.
See [Security](#security-trusted-network-required) below.

Remote owners omit physical `device_ids` from their `/metadata`
responses. Other discovery information, including the logical name, driver
class, host, joint layout, and state dimensions, remains visible to reachable
peers.

## Physical device identity vs. logical name

`name` is what you choose and what keys the Zenoh topics — it never
requires constructing a driver to resolve. Physical device identity
(`Robot.device_ids`, e.g. `("serial:ttyUSB0",)`) is a separate concern the
_owner_ uses to enforce host-local exclusivity: two different `name`s
cannot claim the same physical device at the same time
(`RobotDeviceAlreadyOwned`), and a race for the _same_ `name` with
_different_ devices is rejected (`RobotNameConflict`) rather than silently
picked by whichever process happened to start first.

An existing owner's advertised `robot_class` is compared against yours only
as a diagnostic (logged on mismatch, never fatal, and never imported from
the network) — subclasses, wrappers, and re-exports can all preserve the
wire contract.

## Owner lifecycle

- One owner process per `name`, and one owner per physical device — both
  enforced by host-local, crash-safe `flock` locks
  (under `$XDG_RUNTIME_DIR/physicalai/robot-locks/` on Linux, or a private
  per-user temporary directory elsewhere on Unix). If two processes race to
  spawn the same `name`, the loser attaches (same devices) or raises
  `RobotNameConflict` (different devices).
- The owner runs a single write-first control loop at a fixed rate
  (`rate_hz` spawn parameter, default 100 Hz; override per instance when
  hardware measurements justify a different value).
- When the last subscriber disconnects (cleanly or by crashing), the owner
  waits `idle_timeout` seconds, then calls the driver's `disconnect()` —
  honoring the safe-state contract (hold/home) — and exits.
- An explicit `physicalai robot serve` owner has no idle timeout. It remains in the
  foreground until interrupted or until the owner loop fails.
- A subscriber's `disconnect()` never stops the robot's motors; the owner
  owns safe-state.
- Subscribers reject an owner advertising an unsupported transport
  protocol version before ever declaring the action publisher.

## Security: trusted network required

In `allow_remote=True` mode, this transport accepts actions on its Zenoh
`/action` key and torque-control requests on `/torque` **without
authentication or encryption** — any peer that can reach the owner's Zenoh
session can move the physical robot or change its torque state. It is designed
for a **trusted robot-cell network** (isolated LAN/VLAN) in that mode.
Isolating the network — via VLAN/firewall segmentation or Zenoh's own
ACL/TLS features — is the deployer's responsibility. `allow_remote=False`
(the default) avoids this exposure entirely by keeping the owner
unreachable off-host.
