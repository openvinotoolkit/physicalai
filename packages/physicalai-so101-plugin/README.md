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

## Noninteractive setup

`SO101SetupSession` owns diagnostics, single-motor ID/baudrate assignment, and
EEPROM calibration without LeRobot motor imports. It is separate from the
deployment driver: **setup never enables torque or commands a position**, even
for followers. Support the arm throughout setup and disconnect; disabling torque
can let it fall under gravity.

```python
from physicalai_so101_plugin import SO101SetupSession

session = SO101SetupSession("/dev/ttyACM0", role="follower")
try:
    session.connect()  # Opens the port only; missing motors are allowed.
    for name, servo_id in session.servo_ids.items():
        print(name, servo_id, session.ping(name))  # Actual model or None.
    session.disable_torque()  # Required before asking a user to move the arm.
    # An application now asks the user to support and center the arm.
    offsets = session.set_half_turn_homings()
    positions = session.read_positions()  # Already-homed signed ticks.
finally:
    session.disconnect()  # Best-effort torque off, then close; never hold position.
```

The API does not prompt, record ranges, persist application data, or implement
websocket events. Keep one session for diagnostics, calibration, and preview.
Each hardware operation is serialized with an internal reentrant lock, including
multi-register writes and disconnect. Async applications must also shield and
drain `asyncio.to_thread()` work before accepting another logical transaction or
closing the session: cancelling an await does not stop its hardware thread.

- **Single-motor setup:** physically connect ONLY the named STS3215, then call
  `setup_motor(name)`. Discovery checks all eight supported baudrates and rejects
  multiple or ambiguous responders before mutation. Duplicate physical IDs
  cannot always be detected. The expected model is 777 and canonical IDs are 1–6.
- **Destructive homing:** `set_half_turn_homings()` clears previous homing offsets
  and limits, selects angle-feedback position mode, and centers around tick 2047.
  Previous EEPROM calibration can be lost even if a later step fails. Offsets
  outside ±2047 produce a centering error rather than wrapping. No rollback is
  promised after EEPROM or ID/baudrate writes.
- **Range recording:** record returned positions directly; do not subtract the
  hardware homing offset again. Reject negative or out-of-encoder-range samples.
  Every joint needs an ordered, nonzero range within 0–4095.
- **Calibration:** `read_calibration()` inspects hardware without changing cache;
  `load_calibration(calibration)` validates/caches without EEPROM writes;
  `write_calibration(calibration)` writes and verifies before publishing cache.
  `clear_calibration()` invalidates only software state. Setup accepts exactly
  the canonical IDs and `drive_mode=0`; inverted imports are not supported.
- **Configuration and preview:** `configure_motors()` applies role-specific
  settings with torque off. `read_positions(normalized=True)` requires validated
  cached calibration and returns body joints in [-100, 100], gripper in [0, 100].
  Verification remains manually movable, including on followers.
- **Errors:** invalid arguments raise `ValueError`; permission failures retain
  `PermissionError`; device/communication failures raise `ConnectionError`;
  servo-status and read-back failures raise `RuntimeError` with operation context.
  Voltage reads return raw 0.1 V units and raise on failure; zero is real data,
  not a placeholder for a missing motor.

Construction and `connect()` do not change existing torque state. An application
must explicitly release a complete arm after diagnostics and before manual
instructions. Setup operations release the motors they mutate. EEPROM unlocks
are explicit and relocked on completion, with best-effort relocking on failures.
The private receiver uses monotonic deadlines, bounded buffers and serial write
timeouts rather than globally patching SDK receive loops.

### Release and hardware validation

This API must be published (or pinned to a reviewed immutable artifact) before
Studio depends on it. Current source requires Runtime >=0.4; Studio deployments
still using Runtime 0.3 need either a tested setup-only backport or coordinated
Runtime/library dependency alignment. Do not lower that minimum just to resolve
the lock: current bimanual/catalog code relies on newer Runtime types. Release
versions and changelogs are managed by hatch-vcs and release-please.

Automated tests use fake devices. Supervised, explicit opt-in hardware acceptance
is still required: absent motors; isolated nondefault IDs/baudrates; chain reprobe;
leader/follower voltage; torque-off homing and recording; calibration persistence
after power cycle; skipped-calibration preview; repeated nonzero-offset homing;
and disconnect during streaming/setup. Broadcast discovery latency and firmware
ID/baudrate acknowledgement behavior must be checked on supported hardware.

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
