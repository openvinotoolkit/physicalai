# SO-101 Demo Scripts

Quick hardware verification scripts. No extra dependencies beyond `physicalai[so101]`.

## read_joints

Read and display live joint positions. Torque is off — move the arm by hand to verify readings.

```bash
python -m physicalai.robot.so101.demo.read_joints --port /dev/ttyUSB0
```

## move_joints

Move each joint ±50 ticks one at a time to verify actuation and wiring. Torque is released when done.

```bash
python -m physicalai.robot.so101.demo.move_joints --port /dev/ttyUSB0
```

Use `--offset 100` for larger movements or `--delay 1.0` for more time to observe.

## Finding your port

```bash
ls /dev/tty.usb*        # macOS
ls /dev/ttyUSB*          # Linux
```
