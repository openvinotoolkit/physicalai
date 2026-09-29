# SO-101 model from MuJoCo Menagerie

This directory holds `robotstudio_so101` from
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) at commit
[`c96a32d28fb5da84da38c1da4d749e7a13212855`](https://github.com/google-deepmind/mujoco_menagerie/tree/c96a32d28fb5da84da38c1da4d749e7a13212855/robotstudio_so101),
released under the Apache License 2.0 (`LICENSE`). Upstream's history is in `CHANGELOG.md`.

`so101.xml`, `assets/*.stl`, `LICENSE`, and `CHANGELOG.md` are copied unchanged.
Upstream's `README.md`, `scene.xml`, `scene_box.xml`, and `so101.png` are left out.

## Local changes

The files are not edited. When a scene loads, `robot_profile.py` (`SO101_PROFILE`) attaches
`so101.xml` at the scene's `robot_mount` frames and adjusts the loaded spec so that the arm
matches the plugin's earlier SO-101 model:

- Joint ranges are the calibrated ranges of `urdf/so101/so101_new_calib.urdf`. Menagerie
  caps `wrist_roll` at 2.74385 rad; the plugin keeps 2.84121 rad, which the actuator's
  `ctrlrange` already allows. Normalized joint units span these ranges.
- Actuator `forcerange` is ±3.35 N m instead of the class default of ±2.94.
- The wrist camera is renamed from `wrist_cam` to `wrist` and keeps the earlier pose (on the
  gripper's -y side) and its 75° field of view. `camera_box1` and `camera_box2` move to
  that side as well.
- The gripper's collision meshes, the extra collision box on the fixed jaw, and the camera
  mount's visual geoms (mount, PCB, lens) are removed, together with their meshes.
- The `gripperframe` site keeps its earlier orientation (`quat="0 0 1 0"`).
- The scene's physics options apply (for example, a 2 ms timestep instead of 5 ms).
