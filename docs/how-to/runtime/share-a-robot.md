# Share a Robot Across Processes

`SharedRobot` lets one process own a robot's exclusive hardware connection
while any number of other processes read its state and send actions over
[Zenoh](https://zenoh.io/). It satisfies the same `Robot` protocol as a
direct driver, so it is a drop-in replacement anywhere a robot is expected
(including `PolicyRuntime`).

## Install

```bash
pip install "physicalai[transport]"
```

## Create or attach

Every `SharedRobot` has a required, caller-chosen logical `name` — it keys
the Zenoh topics directly. The first `SharedRobot` constructed for a given
`name` that finds no existing owner spawns one (in a detached subprocess);
later instances (same or different process, same `name`) attach to it:

```python
import numpy as np
from physicalai.robot import SharedRobot
from physicalai.robot.so101 import SO101

robot = SharedRobot(
    "left-arm",
    robot_class=SO101,
    robot_kwargs={
        "port": "/dev/ttyUSB0",
        "calibration": "~/.cache/calibration/so101.json",  # a path — kwargs must be serializable
    },
)
robot.connect()

obs = robot.get_observation()          # pull latest state, non-blocking
robot.send_action(np.asarray(obs.joint_positions), goal_time=0.1)

robot.disconnect()                     # detaches; the owner keeps running
```

`robot_class` can be the class object (normalized to its dotted import path)
or the path itself, e.g. `robot_class="physicalai.robot.so101.SO101"` — any
importable class works, including third-party plugin robots, with no
registry to update.

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

## Network scope: local-only by default

`allow_remote=False` (the default) keeps the owner's Zenoh session
unreachable off-host — multicast/gossip scouting is disabled and the owner
listens on `127.0.0.1` only. Same-host spawn-or-attach still works without
depending on multicast: owner and subscriber derive the same deterministic
loopback port from `name`.

Opt into cross-host reachability explicitly when you need it:

```python
robot = SharedRobot(
    "left-arm",
    robot_class=SO101,
    robot_kwargs={"port": "/dev/ttyUSB0", "calibration": "calibration.json"},
    allow_remote=True,
)
```

The caller that spawns the owner fixes this scope for the owner's entire
lifetime — a later attacher's `allow_remote` only configures its own
session, it never widens or narrows an already-running owner's
reachability. See [Security](#security-trusted-network-required) below.

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
  (`~/.cache/physicalai/robot-locks/`). If two processes race to spawn the
  same `name`, the loser attaches (same devices) or raises `RobotNameConflict`
  (different devices).
- The owner runs a single write-first control loop at a fixed rate
  (`rate_hz` spawn parameter, default 100 Hz; override per instance when
  hardware measurements justify a different value).
- When the last subscriber disconnects (cleanly or by crashing), the owner
  waits `idle_timeout` seconds, then calls the driver's `disconnect()` —
  honoring the safe-state contract (hold/home) — and exits.
- A subscriber's `disconnect()` never stops the robot's motors; the owner
  owns safe-state.
- Subscribers reject an owner advertising an unsupported transport
  protocol version before ever declaring the action publisher.

## Security: trusted network required

In `allow_remote=True` mode, this transport applies any action received on
its Zenoh `/action` key **without authentication** — any peer that can
reach the owner's Zenoh session can move the physical robot. It is designed
for a **trusted robot-cell network** (isolated LAN/VLAN) in that mode.
Isolating the network — via VLAN/firewall segmentation or Zenoh's own
ACL/TLS features — is the deployer's responsibility. `allow_remote=False`
(the default) avoids this exposure entirely by keeping the owner
unreachable off-host.

## Design background

See `docs/development/robot-zenoh-transport-design.md` and
`docs/development/robot-zenoh-transport-identity-followup.md` for the full
design (key layout, wire format, QoS choices, identity/locking model, and
decision log).
