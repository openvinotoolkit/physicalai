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

## Spawn or attach

The first `SharedRobot` that finds no existing owner spawns one in a
detached subprocess; later instances (same or different process) attach to
it. The owner is keyed by a `robot_id` derived from the connection
parameters, so two processes constructing the same robot automatically
share one hardware connection:

```python
import numpy as np
from physicalai.robot import SharedRobot

robot = SharedRobot(
    "so101",
    port="/dev/ttyUSB0",
    calibration="~/.cache/calibration/so101.json",  # a path — specs must be serializable
)
robot.connect()

obs = robot.get_observation()          # pull latest state, non-blocking
robot.send_action(np.asarray(obs.joint_positions), goal_time=0.1)

robot.disconnect()                     # detaches; the owner keeps running
```

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

For manually-launched or remote owners, attach by explicit id:

```python
robot = SharedRobot.from_owner("physicalai/robot/so101/mybot-host/ttyUSB0")
robot.connect()
```

Enumerate reachable robots with:

```python
from physicalai.robot.transport import discover_robots

for meta in discover_robots():
    print(meta["robot_id"], meta["robot_type"], meta["joint_names"])
```

## Owner lifecycle

- One owner process per robot, enforced by a user-scoped lock file
  (`~/.cache/physicalai/robot-locks/`). If two processes race to spawn,
  the loser attaches to the winner.
- The owner runs a single write-first control loop at a fixed, per-robot
  rate (`rate_hz` spawn parameter; defaults: SO-101 100 Hz, WidowXAI
  200 Hz).
- When the last subscriber disconnects (cleanly or by crashing), the owner
  waits `idle_timeout` seconds, then calls the driver's `disconnect()` —
  honoring the safe-state contract (hold/home) — and exits.
- A subscriber's `disconnect()` never stops the robot's motors; the owner
  owns safe-state.

## Security: trusted network required

This transport is network-capable and applies any action received on its
Zenoh `/action` key **without authentication** — any peer that can reach
the owner's Zenoh session can move the physical robot. It is designed for
a **trusted robot-cell network** (isolated LAN/VLAN). Isolating the
network — via VLAN/firewall segmentation or Zenoh's own ACL/TLS features —
is the deployer's responsibility.

## Design background

See `docs/development/robot-zenoh-transport-design.md` for the full design
(key layout, wire format, QoS choices, and decision log).
