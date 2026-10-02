# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""MuJoCo SO-101 robot constants."""

from __future__ import annotations

from typing import Final

SO101_JOINT_ORDER: Final = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

BIMANUAL_SO101_JOINT_ORDER: Final = tuple(f"{side}_{name}" for side in ("left", "right") for name in SO101_JOINT_ORDER)
"""Joints of the bimanual scene: each arm's joints with its ``left_``/``right_`` mount prefix, left first."""

# Zenoh owner names used by ``physicalai-mujoco-so101 start`` (``--name``
# default) and the Studio MuJoCo payload ``name`` field default. The two arm
# counts get distinct names so a single-arm and a bimanual simulation can run
# side by side, and so Studio's online probe reports them independently.
DEFAULT_MUJOCO_OWNER_NAME: Final = "mujoco-so101-follow"
DEFAULT_BIMANUAL_MUJOCO_OWNER_NAME: Final = "mujoco-so101-bimanual-follow"
