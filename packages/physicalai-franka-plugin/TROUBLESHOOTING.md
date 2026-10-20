# Troubleshooting

## pylibfranka is not installed

Controlling a real robot requires the optional `franka` extra:

```bash
pip install "physicalai-franka-plugin[franka]"
```

`pylibfranka` ships wheels for Python 3.9–3.12 only (glibc ≥ 2.34). Use Python
≤ 3.12 when controlling the robot; the plugin itself still imports and runs on
Python 3.13+ for catalog and driver configuration.

## Prepare the real-time PC

The 1 kHz loop has to answer every robot packet within a millisecond. Before
connecting, check on the PC that talks to the robot:

| Check                               | Expected                                            |
| ----------------------------------- | --------------------------------------------------- |
| `uname -a \| grep -o PREEMPT_RT`    | `PREEMPT_RT`                                        |
| `ulimit -r`                         | `99` (user in the `realtime` group)                 |
| `ulimit -l`                         | `unlimited` (otherwise `mlockall failed` is logged) |
| `sysctl kernel.sched_rt_runtime_us` | `-1`                                                |
| CPU governor                        | `performance`                                       |
| `ethtool -c <robot NIC>`            | interrupt coalescing off or a few µs                |

Connect the PC to the LAN port of the **Control** box. The port on the arm's
base is a different network and does not provide FCI.

## The robot stops with a `joint_motion_generator_*_discontinuity` reflex

These reflexes belong to the joint position interface: when the PC answers a
robot packet late, the robot extrapolates the missing cycles, and the next
command can look like a jump in velocity or acceleration. Python loops are late
now and then, so this happens sooner or later in `control_mode="position"`.

`FrankaFR3` uses `control_mode="impedance"` by default: it sends torques with
the joint impedance law of Franka's own GELLO controller
(`tau = K (q_target - q) - D dq`), which has no such checks; a late cycle just
repeats the last torque. If you switched to `"position"`, switch back.

## The control loop terminates unexpectedly

The control process reports the reason with the next call, for example
`ConnectionError: FR3 172.16.1.41 control loop is not running: ControlException:
... reflex ... [after N ticks; longest gap X ms at tick T]`. The last 500 cycles
(robot time step, command, measured position, PC time step, torque) are saved to
`franka-<host>-abort.npz`, and after a clean stop to `franka-<host>-last.npz`, in
the system temp directory (`/tmp` on Linux). A "longest gap" of several milliseconds means the
PC stalled; see the real-time checklist above.

`robot.recover()` restarts the loop with automatic error recovery. Also check in
Desk that the brakes are open, the robot is in Execution mode and FCI is active.

## `self_collision_avoidance_violation`

The FR3 refused a pose because its links would come too close. The GELLO's
folded rest pose is one of them: the GELLO's thin links fit, the FR3's do not.
Lift the GELLO off its pins and hold it near the robot's pose before
teleoperating.

## The robot is not reachable after moving or re-cabling

If the PC shares its robot ports with NetworkManager ("shared to other
computers"), the PC's DHCP server gives each Control its address: the host part
follows the Control's MAC address, the subnet follows the PC port. Swapping two
cables at the PC therefore swaps the subnets, for example `172.16.0.41` becomes
`172.16.1.41`. Identify each robot by its serial number:

```bash
curl -sk https://<ip>/api/system   # "controlSerialNumber": "..."
```

A link at 100 Mbit/s or 2.5 Gbit/s instead of 1 Gbit/s usually means the cable
does not end at a Control's LAN port.

## The GELLO stops answering (`motor ID n did not respond`)

Only one program can use a GELLO's serial port. A joint inspector, a calibration
command or an earlier teleoperation run (including orphaned child processes)
holding the port makes every other reader fail. Stop them, then retry. If no
other program is running, check the Dynamixel power and wiring.

## "Robot is not connected" from get_observation

`get_observation()` returns the latest state published by the control process.
If the message persists after `connect()`, the loop exited: the error carries
the reason (see above).

## The gripper is ignored

`FrankaFR3(..., gripper_enabled=True)` runs the Franka Hand in its own process.
If the hand cannot be reached, a warning is logged and the arm keeps working;
gripper action elements are then dropped. Check the Franka Hand is attached and
powered, or disable `gripper_enabled`.

## Joint targets appear clipped

Actions are clipped to the FR3 hardware joint limits minus `limit_margin_deg`
(5° by default); `joint_limits="suggested"` or `"urdf"` select narrower sets.
Joints 4 and 6 have ranges that do not span zero, so sine-wave demos around
zero clip them to their nearest limit. This is expected and intentional.
