# Troubleshooting

Common issues when using `physicalai-rebot-b601-plugin` and suggested fixes.

---

## Connection Issues

### Permission denied on serial port

```text
PermissionError: [Errno 13] Permission denied: '/dev/ttyACM0'
```

Your user is not in the `dialout` (or `uucp`) group that owns the serial device.

```bash
sudo usermod -a -G dialout "$USER"
# Log out and back in, or run:  newgrp dialout
```

If the issue persists, check the port's group:

```bash
ls -l /dev/ttyACM0
```

For persistent udev rules (recommended if you use many serial devices), create
`/etc/udev/rules.d/99-rebot.rules`:

```ini
SUBSYSTEM=="tty", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="7523", MODE="0666"
SUBSYSTEM=="tty", ATTRS{idVendor}=="10c4", ATTRS{idProduct}=="ea60", MODE="0666"
```

Then reload udev:

```bash
sudo udevadm control --reload-rules && sudo udevadm trigger
```

---

### CAN interface not found

```text
OSError: [Errno 19] No such device
```

The SocketCAN interface has not been brought up.

```bash
sudo ip link set can0 up type can bitrate 1000000
```

To make it persistent, add to `/etc/network/interfaces` or use a systemd
service:

```ini
# /etc/systemd/network/80-can.link
[Match]
Type=can

[Link]
Name=can0
```

```ini
# /etc/systemd/network/80-can.network
[Match]
Name=can0

[CAN]
BitRate=1M
```

---

### Damiao serial adapter not detected (DM only)

The Damiao USB-CAN adapter appears as a serial port (`/dev/ttyACM0`). If it
does not show up:

1. Check `dmesg | grep tty` after plugging it in.
2. Ensure the adapter is in the correct mode (some adapters need a jumper or
   button press to enter CAN mode).
3. Try a different USB cable — some cables are power-only.

---

### can_adapter="robstride" raises NotImplementedError

The `motorbridge` Python SDK does not currently expose the RobStride CAN adapter
protocol. Always use `can_adapter="socketcan"` for RS arms.

---

### RobStride adapter not visible on CAN bus (RS only)

If the RobStride USB-CAN adapter is not detected:

```bash
# Check if the device appears
lsusb | grep -i robstride

# Check kernel messages
dmesg | grep -i robstride
```

The adapter typically uses a GD32 microcontroller and should appear as a
USB serial device. If missing, try a different USB port or cable.

---

## Motor Communication

### Motor / servo not responding

```text
ConnectionError: No feedback received for motor 'shoulder_pan'
ConnectionError: Servo 'shoulder_pan' (ID 1) did not respond on /dev/ttyUSB0
```

Possible causes:

- **Wrong port or baud rate** — verify the Damiao serial baud (DM, defaults to 921600).
- **Power issue** — motors need 24 V (DM/RS). Check the
  power supply LED.
- **Damaged cable** — try reseating the CAN/RS-485/ UART cable.
- **Wrong motor ID** — `REBOT_B601_DM_MOTOR_IDS` / `REBOT_B601_RS_MOTOR_IDS`
  must match your hardware configuration.
- **CAN bus termination** — ensure the CAN bus has proper 120 Ω termination
  at both ends for RS arms.

---

### Motor state is None after poll

```text
ConnectionError: No feedback received for motor 'wrist_flex'
```

The motor did not return feedback within the poll cycle. Try:

1. Power-cycling the arm.
2. Checking CAN bus wiring (RS) or RS-485 wiring (DM Damiao).
3. Reducing the number of motors polled simultaneously.

---

## Joint Behavior

### Joint moves in the wrong direction

The `REBOT_B601_DM_JOINT_DIRECTIONS` and `REBOT_B601_RS_JOINT_DIRECTIONS`
constants map the PhysicalAI action space (degrees) to the motor's native
coordinate system. If a joint moves opposite to the commanded direction:

- Check your hardware's wiring phase.
- Verify the direction sign in `src/physicalai_rebot_b601_plugin/constants.py`.
- Some motors have a `reverse` parameter in their firmware — consult the
  motor manual.

---

### Joint hits a limit unexpectedly

The driver clips commanded positions to `REBOT_B601_DM_JOINT_LIMITS_DEG` /
`REBOT_B601_RS_JOINT_LIMITS_DEG`. If the arm stops short:

- Verify the limits match your hardware revision. The values in `constants.py`
  may differ from your specific arm.
- The gripper limit for DM is `(-270, 0)` degrees — that is a large range;
  verify your gripper's actual mechanical stop.

---

### Gripper does not close / open (DM)

The DM gripper runs in `FORCE_POS` mode with a configurable torque ratio
(`force_pos_torque_ratio`, default 0.1). If the gripper feels weak:

```python
robot = ReBotB601DM(port="...", force_pos_torque_ratio=0.3)
```

Values above ~0.5 may cause overheating — increase gradually.

---

### Gripper oscillates (RS)

The RS gripper uses impedance control with velocity-limited torque. If the
gripper oscillates or chatters:

- Increase `gripper_mit_kd` (damping).
- Decrease `gripper_mit_kp` (stiffness).
- Ensure the gripper is not mechanically obstructed.

---

## Installation

### ImportError: No module named 'motorbridge'

```python
ImportError: motorbridge is required for ReBotB601DM. Install with: uv add physicalai-rebot-b601-plugin.
```

Install the optional hardware SDKs:

```bash
uv add physicalai-rebot-b601-plugin
```

If you are developing the plugin locally (from the repo root):

```bash
uv add -e packages/physicalai-rebot-b601-plugin
```

---

### Version conflicts with motorbridge

This plugin requires `motorbridge>=0.4.4`.
Check installed versions:

```bash
uv pip show motorbridge
```

Upgrade if needed:

```bash
uv add motorbridge@latest
```

---

## Runtime Errors

### "Robot is not connected. Call connect() first."

You called `get_observation()` or `send_action()` before `connect()`.
Always use the context manager:

```python
from physicalai.robot import connect

with connect(robot) as arm:
    obs = arm.get_observation()
```

Or call `connect()` / `disconnect()` manually (not recommended):

```python
robot.connect()
try:
    obs = robot.get_observation()
finally:
    robot.disconnect()
```

---

### verify_robot() fails

If `physicalai.robot.verify_robot(robot)` fails, check:

1. All required protocol methods are present (`connect`, `disconnect`,
   `get_observation`, `send_action`, `is_connected`, `joint_names`).
2. The observation dataclass has `joint_positions`, `timestamp`, `state`.
3. `send_action` accepts `(np.ndarray, *, goal_time=0.1)`.

For the plugin drivers, both satisfy the `Robot` protocol — but if you
have subclassed or wrapped them, re-check with `isinstance(robot, Robot)`.

---

## Hardware-Specific

### Arm does not hold position when idle

The motors default to torque-off on disconnect
(`disable_torque_on_disconnect=True`). If you want the arm to hold its
position after your script exits:

```python
robot = ReBotB601DM(port="...", disable_torque_on_disconnect=False)
```

Be careful — the arm will remain powered and can be dangerous if unattended.

---

### Motor overheating

- Reduce the command rate (send actions less frequently).
- Lower `force_pos_torque_ratio` (DM) or `gripper_mit_kp` (RS).
- Ensure proper cooling and that the arm is not mechanically stalled.
- Check that the load does not exceed the motor's rated torque.

---

### CAN bus communication errors (RS)

If you see frequent CRC errors or dropped frames:

```bash
# Check CAN bus stats
ip -details -statistics link show can0
```

Look for increasing `bus-error` counters. Common fixes:

- Add or verify 120 Ω termination resistors at both ends.
- Use a shorter, shielded CAN cable.
- Lower the bit rate (e.g., 500 kbit/s instead of 1 Mbit/s).
- Check for loose connectors.

---

## Getting Help

If the above does not resolve your issue:

- Open a [GitHub issue](https://github.com/MarkRedeman/physicalai-plugins/issues).
- Include the full error traceback, your command line, and the output of:

  ```bash
  uv pip list | grep -i -E "motorbridge|physicalai|rebot"
  ```
