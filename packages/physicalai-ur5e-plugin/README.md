# Physical AI Universal Robots plugin

`physicalai-ur5e-plugin` supplies a standalone Runtime driver and Studio
catalog entries for **UR3e, UR5e, UR7e, UR10e, UR12e, and UR16e**. It uses
[`ur_rtde`](https://sdurobotics.gitlab.io/ur_rtde/) over Ethernet, without ROS.

The initial scope is one six-joint follower arm. There is no gripper, leader,
freedrive, Cartesian, torque-control, or CB3 implementation. Model profiles
are not hardware detection: selecting the correct model/controller is the
operator's responsibility. The driver does not currently verify the model
reported by the controller.

## Validation status and installation

Automated tests use fake hardware. Real-robot and URSim validation must be
performed separately before deployment; model availability does not imply
hardware certification. PolyScope X External Control support is a configured
SDK path, not a tested compatibility guarantee.

The initial target is **Linux x86_64, Python 3.12, PolyScope 5**. `ur_rtde`
1.6.5 has Python 3.12 Linux x86 and Windows wheels; newer Python versions,
macOS, and ARM64 require a vendor SDK source build. Do not assume wheels
exist on every platform supported by Physical AI.

```bash
uv add physicalai-ur5e-plugin
```

From this repository, install the workspace package with:

```bash
uv sync --package physicalai-ur5e-plugin --extra tests
```

Studio loads the package's catalog entry point at backend startup. Install
it into the backend's own environment and restart the backend:

```bash
uv add --editable /path/to/physicalai/packages/physicalai-ur5e-plugin
uv sync
```

No Studio source changes or curated Plugins-page listing are required for
the robot picker. Stable types are `UniversalRobots_UR5e_Follower`, etc.

## Controller setup and safety

1. Mount the correct arm, configure its safety system, and set payload/TCP
   correctly on the teach pendant. Keep the workcell clear and the stop
   controls accessible.
2. Use a trusted, isolated robot-cell network. RTDE/script interfaces are
   not authenticated or encrypted. This plugin connects only to an explicit
   literal IPv4 address; it does not scan networks or expose a server.
3. Enable RTDE and the controller interfaces required by `ur_rtde` in the
   controller's service/security settings.
4. For `control_mode="remote"`, enable PolyScope 5 remote control. The SDK
   uploads and starts its script. Stop any other controller using the arm.
5. Alternatively, install the External Control URCap, configure its host/port
   and firewall, choose `control_mode="external_control"`, and manually start
   the pendant program. Current `ur_rtde` documentation requires this route
   for PolyScope X. The robot must already be powered on with brakes released.

**Connecting never homes, powers on, releases brakes, or resets safety stops.**
Connection still takes control of a script interface and must be treated as
a hardware operation. The driver checks running mode and normal/reduced
safety mode, and refuses other states.

Software limits and watchdogs are **not safety-rated** and do not implement
collision avoidance. Never bypass the robot's safety system. Network loss
can prevent a software stop from reaching the robot; configure and validate
the controller watchdog and physical safety measures for your workcell.

## Driver and units

```python
from physicalai.config import Config
from physicalai_ur5e_plugin import UniversalRobot

# Construction/config export has no hardware I/O and does not import ur_rtde.
robot = UniversalRobot(ip="192.168.1.10", model="ur5e")
recipe = Config.from_instance(robot)
```

The driver implements `connect()`, `disconnect()`, `is_connected()`,
`get_observation()`, `send_action(action, goal_time=...)`, `joint_names`, and
`device_ids`. All public joint positions/actions are **degrees**; internal
RTDE values and URDF limits are **radians**. Observations contain position-only
state, a monotonic acquisition timestamp, and `sensor_data=None`, `images=None`.

Joint order:

1. `shoulder_pan_joint`
2. `shoulder_lift_joint`
3. `elbow_joint`
4. `wrist_1_joint`
5. `wrist_2_joint`
6. `wrist_3_joint`

Actions must be exactly `(6,)` and finite. They are absolute, unwrapped joint
positions; the driver does not choose a shortest-angle path or silently clip
invalid actions. The default maximum offset from measured position is 10°.
Move to any required policy starting pose separately under supervised control.

A single worker owns **all** SDK calls, including shutdown. `send_action`
queues the latest target without blocking inference. The worker interpolates
linearly from the previous command over at least `goal_time`, lengthening
the interval to respect the configured setpoint speed (default 30°/s, capped
by each model's joint velocity limits). New targets replace unfinished ones.
This bounds setpoint velocity, **not acceleration or physical tracking speed**.
Controller-side target checks run asynchronously; a rejection stops the worker
and appears on the next observation/action call.

The default servo rate is 500 Hz; Python and the operating system do not
guarantee hard-real-time scheduling. Use the vendor real-time setup guidance
and measure timing on the deployment host. Watchdog kicks and controller
safety queries can consume cycles. Missed deadlines, stale receive timestamps,
excessive tracking error, connection loss, and protective stops latch a fault.

**Actions must keep arriving within `command_timeout` (default 0.5 s), even
after reaching a target.** A long `goal_time` does not extend this timeout.
After any fault, inspect the robot and explicitly disconnect/reconnect; there
is no automatic retry, safety reset, or motion resumption. Disconnect attempts
servo stop, script stop, and both socket closes independently, reporting failures.

## URDF source, licensing, and regeneration

Assets come from the official
[UR ROS2 Description repository](https://github.com/UniversalRobots/Universal_Robots_ROS2_Description)
at `81f7c9fe003c2f0e02eec5a7253f1b262ad38b61`. Generated plain URDFs,
meshes, and limits ship inside the Python package for offline Studio previews.
No ROS, Xacro, or download is needed at runtime. See `NOTICE` and the bundled
BSD-3-Clause license. Newer models with non-open-source graphical terms are
intentionally excluded.

Regenerate from a clean checkout of that exact commit:

```bash
uv run --no-project --with xacro==2.1.1 python \
  packages/physicalai-ur5e-plugin/scripts/generate_assets.py /path/to/ur_description
npx --yes prettier@3.5.3 --write \
  packages/physicalai-ur5e-plugin/src/physicalai_ur5e_plugin/assets/limits.json
```

The generator only copies referenced meshes from an explicit BSD-model allowlist.
Nominal kinematics are for preview, not robot-specific calibrated Cartesian
planning. For accurate transforms, extract the individual robot's factory
calibration using UR's tooling and build an appropriately reviewed description.
UR3e's continuous final wrist has no artificial position bound.

## Tests

```bash
uv run --package physicalai-ur5e-plugin --extra tests pytest \
  packages/physicalai-ur5e-plugin/tests
```

The opt-in integration test is **URSim only**, not a physical robot test.
Start the official `universalrobots/ursim_e-series` container yourself, select
the matching model, power on/release simulated brakes, and enable remote mode.
Use a private/local-only simulator network (do not publish simulator ports to
all host interfaces). Then explicitly opt in:

```bash
PHYSICALAI_URSIM_IP=192.168.56.101 PHYSICALAI_URSIM_ONLY=1 \
  uv run --package physicalai-ur5e-plugin --extra tests pytest \
  packages/physicalai-ur5e-plugin/tests/integration -m ursim
```

Optional `PHYSICALAI_URSIM_MODEL` defaults to `ur5e`. Never point this test at
physical hardware: it takes control and streams a small joint movement.
