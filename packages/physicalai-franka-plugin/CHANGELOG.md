# Changelog

All notable changes to this package are documented in this file.

## Unreleased

- `FrankaFR3` runs its 1 kHz FCI loop in a separate process and the Franka Hand in
  another: `pylibfranka` holds the GIL while waiting for robot packets, so a loop in
  a thread starved, and was starved by, the rest of the interpreter.
- `FrankaFR3` defaults to `control_mode="impedance"`: torques with the joint
  impedance gains of Franka's GELLO controller, which avoids the
  `joint_motion_generator_*_discontinuity` reflexes of joint position control.
  `control_mode="position"` keeps the joint position interface.
- Targets follow a jerk-limited tracker (`dynamics_factor`, default 0.3 of the FR3
  limits) instead of `goal_time` interpolation; `goal_time` is still accepted.
- New `FrankaFR3` options: `dynamics_factor`, `gripper_force_n`, `joint_limits`,
  `limit_margin_deg`, `cpu`, `rt_priority`; `prepare()`, `last_error` and
  `loop_overruns`. Closing the hand grasps with `gripper_force_n`.
- Joint targets and GELLO readings are clipped to the FR3 hardware limits minus 5°
  instead of the URDF limits, which clipped real poses such as the GELLO rest pose.
- `BimanualFrankaFR3` prepares both arms before either control loop starts.
- `physicalai-gello-calibrate --robot HOST` matches a robot's current pose and
  `--mirror-port`/`--mirror-calibration` mirrors the other Duo arm; both give exact
  offsets instead of 90° steps.
- The last 500 control cycles are saved to `/tmp/franka-<host>-abort.npz` (or
  `-last.npz`) for post-mortems.
- Examples: `gello_teleop.py` (teleoperation workflow on the plugin drivers) and
  `joint_inspector.py` (read-only robot/GELLO values in a browser).

## 0.1.0

- Add single-arm and bimanual Franka Research 3 follower drivers.
- Add Physical AI Studio catalog entries and bundled FR3 visualization assets.
