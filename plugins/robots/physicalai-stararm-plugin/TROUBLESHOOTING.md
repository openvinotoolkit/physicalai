# Troubleshooting

Common issues when using `physicalai-stararm-plugin`.

## Serial port permission denied

```text
PermissionError: [Errno 13] Permission denied: '/dev/ttyUSB0'
```

Add your user to `dialout` and re-login:

```bash
sudo usermod -a -G dialout "$USER"
```

## Servo did not respond

```text
ConnectionError: Servo 'shoulder_pan' (ID 0) did not respond on /dev/ttyUSB0.
```

- Verify port and baudrate (`1_000_000` by default).
- Verify 12V power supply.
- Check USB/UART cable quality.

## Follower motion is too abrupt

- Increase `goal_time` in runtime sources.
- Increase `command_interval_ms` on `StarArm102FLFollower`.

## Import errors

```bash
uv add physicalai-stararm-plugin
```
