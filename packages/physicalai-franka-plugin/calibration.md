# Calibrating Franka GELLO Leaders

This guide covers the passive Franka GELLO and GELLO Duo leaders supported by
`physicalai-franka-plugin`.

The leaders are read-only teleoperation devices. The calibration procedure
reads encoder positions but never enables torque or sends a position command.

## Hardware layout

Each Franka GELLO arm is one Dynamixel bus with the following IDs:

| IDs | Component                               |
| --- | --------------------------------------- |
| 1–7 | Seven arm joints, ordered base to wrist |
| 8   | Trigger / gripper input                 |

The Duo has two independent buses: one for the left arm and one for the right
arm. Use stable `/dev/serial/by-id/...` paths rather than `/dev/ttyACM*` paths,
which may change after reconnecting USB hardware.

```bash
ls -l /dev/serial/by-id/
```

## When calibration is needed

Preassembled Franka GELLO and GELLO Duo hardware uses the plugin's built-in
default calibration. You only need a custom calibration when an arm was
self-assembled, a motor or flange was remounted, a joint direction is wrong,
or the trigger range differs from the default.

For a Duo, calibrate the two arms separately. Their final values may be the
same, but each physical assembly can have different motor-flange offsets.

## Calibration values

A GELLO calibration contains:

| Field                  | Meaning                                                   |
| ---------------------- | --------------------------------------------------------- |
| `joint_signs`          | Per-joint direction multipliers for the seven arm motors. |
| `assembly_offsets_rad` | Seven physical motor-flange offsets in radians.           |
| `gripper_range_rad`    | Trigger positions as `[closed, open]` in radians.         |
| `baudrate`             | Dynamixel bus speed. Stock Franka GELLO uses `57600`.     |

The built-in calibration for preassembled Franka GELLO hardware is:

```json
{
  "joint_signs": [1, -1, 1, -1, 1, 1, 1],
  "assembly_offsets_rad": [0.0, 0.0, 3.142, 3.142, 3.142, 4.712, 0.0],
  "gripper_range_rad": [2.0, 3.22],
  "baudrate": 57600
}
```

## Run the calibration command

The calibration command verifies all eight IDs, disables torque, reads the
reference pose, and writes Studio-uploadable JSON after confirmation.

### Single arm

```bash
physicalai-gello-calibrate \
  --port /dev/serial/by-id/usb-ROBOTIS_OpenRB-150_... \
  --side single \
  --output single-gello-calibration.json
```

### GELLO Duo

Run the command once for each USB adapter:

```bash
physicalai-gello-calibrate \
  --port /dev/serial/by-id/usb-ROBOTIS_OpenRB-150_...left... \
  --side left \
  --output left-gello-calibration.json

physicalai-gello-calibrate \
  --port /dev/serial/by-id/usb-ROBOTIS_OpenRB-150_...right... \
  --side right \
  --output right-gello-calibration.json
```

The command prompts you to perform these actions:

1. Move the arm to the requested reference pose and confirm.
2. Release the trigger fully and confirm its open position.
3. Press the trigger fully and confirm its closed position.
4. Review the generated JSON and approve writing it.

It will not overwrite an existing file unless you pass `--force`.

```bash
physicalai-gello-calibrate --help
```

## Exact offsets: match a robot or mirror the other arm

The reference-pose method rounds every offset to a 90° motor-flange step, so a
GELLO whose horns sit a few degrees off still reads a few degrees off. On the
GELLO Duo used to develop this plugin the error reached 11.6° on J1 and 5–7° on
J2, J5, J6 and J7. Two methods find exact offsets instead. Neither moves
anything: the robot is only read.

### Match the robot's pose (recommended)

Put the FR3 at a known pose with every joint on its zero mark (Franka's home
pose `[0, 0, 0, -90°, 0, 90°, 0]`), activate FCI, then line up the marks on the
GELLO one joint at a time:

```bash
physicalai-gello-calibrate \
  --port /dev/serial/by-id/usb-ROBOTIS_OpenRB-150_...right... \
  --robot 172.16.1.41 \
  --output right-gello-calibration.json
```

For each joint the command shows the GELLO reading, the robot reading and the
difference, and records the joint when you press Enter. It then asks for the
trigger positions, shows the change from the factory offsets and warns about any
joint that changed by more than 45° (wrong pose, swapped GELLOs or a flipped
joint).

### Mirror the other arm of a Duo

Once one arm is calibrated, rest both GELLOs on their pins (white dot on J2 up)
and derive the other arm from it. The Duo's rest poses are mirror images: J1,
J3, J5 and J7 change sign.

```bash
physicalai-gello-calibrate \
  --port /dev/serial/by-id/usb-ROBOTIS_OpenRB-150_...left... \
  --mirror-port /dev/serial/by-id/usb-ROBOTIS_OpenRB-150_...right... \
  --mirror-calibration right-gello-calibration.json \
  --output left-gello-calibration.json
```

This assumes the stand holds both arms symmetrically. Matching the robot is the
direct check.

### Check a calibration

With the GELLO's marks lined up, its readings should equal the robot's at its
marks within a degree or two. `examples/gello_teleop.py --dry-run` prints the
per-joint GELLO-minus-robot offset without moving anything.

## Reference poses

The command uses the published Franka GELLO reference poses below, in radians.

| Configuration | Joint reference pose                             |
| ------------- | ------------------------------------------------ |
| Single        | `[0.0, 0.0, 0.0, -1.57, 0.0, 1.57, 0.0]`         |
| Duo left      | `[-1.57, -0.80, 1.80, -3.00, 1.40, 1.50, -2.10]` |
| Duo right     | `[1.57, -0.80, -1.80, -3.00, -1.40, 1.50, 2.10]` |

The tool applies the official joint signs and rounds assembly offsets to the
nearest motor-flange increment of π/2.

These are nominal. Measured with exact offsets, the Duo's rest pose has J1 at
about ±101° rather than ±90°, which the rounding absorbs. The rest pose is a
GELLO-only pose: the FR3 itself refuses it (self-collision of the forearm with
the base), so never command a follower into it. Lift the GELLO off its pins
before teleoperating.

## Upload a custom calibration in Studio

1. Add a **GELLO Leader** or **Bimanual GELLO Leader**.
2. Choose the USB connection for each arm.
3. Open the advanced configuration section.
4. Open the advanced configuration section and paste the generated JSON:
   - single leader: paste it into `calibration`;
   - Duo: paste each JSON into its corresponding `left_calibration` or
     `right_calibration` field.
5. Connect and confirm that the observed joint motion matches the physical
   leader before teleoperating a follower arm.

Custom calibration is stored in the robot payload, not read from a path on the
runtime machine.

## Safety

- Keep the follower robot stopped while calibrating the leader.
- The calibration command disables torque and never enables it.
- Do not enable torque on an OpenRB-150 unless its Dynamixel power supply and
  jumper are configured correctly; USB power alone can damage a host USB port.
- If an ID fails to respond, stop and verify the bus wiring, power, baud rate,
  and Dynamixel IDs before retrying.
- After any mechanical change, inspect the generated calibration before using
  it with a robot.

## Sources

This procedure and its reference poses follow the Franka GELLO documentation:

- <https://github.com/wuphilipp/gello_mechanical/tree/main/franka_fr3>
- <https://github.com/wuphilipp/gello_software/tree/main/ros2>
