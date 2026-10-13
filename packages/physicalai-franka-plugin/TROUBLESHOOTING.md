# Troubleshooting

## pylibfranka is not installed

Controlling a real robot requires the optional `franka` extra:

```bash
pip install "physicalai-franka-plugin[franka]"
```

`pylibfranka` ships wheels for Python 3.9–3.12 only (glibc ≥ 2.34). Use Python
≤ 3.12 when controlling the robot; the plugin itself still imports and runs on
Python 3.13+ for catalog and driver configuration.

## The control loop terminates unexpectedly

The FCI raises a `ControlException` if the real-time command stream is
interrupted (for example, the host cannot sustain the 1 kHz loop or a reflex
was triggered). Reconnect with `robot.connect()` and consider:

- running on a machine with a low-latency/preempt-RT kernel,
- using `realtime_config: kIgnore` for development (less strict scheduling),
- releasing the brakes and activating the FCI in the Franka Desk before
  connecting.

## "Robot is not connected" from get_observation

`get_observation()` returns the latest state received by the 1 kHz control
loop. Immediately after `connect()` a small delay may elapse before the first
state arrives. If the message persists, the control loop exited — see above.

## The gripper is ignored

`FrankaFR3(..., gripper_enabled=True)` requires a reachable Franka Hand. If the
gripper connection fails during `connect()`, a warning is logged and the driver
continues arm-only; gripper action elements are then silently dropped. Check
the Franka Hand is attached and powered, or disable `gripper_enabled`.

## Joint targets appear clipped

Actions are clipped to the FR3 joint limits before being sent. Joints 4 and 6
have asymmetric ranges that do not span zero, so sine-wave demos around zero
clip them to their nearest limit. This is expected and intentional.
