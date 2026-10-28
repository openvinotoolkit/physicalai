# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Model identifiers and limits generated from the official UR descriptions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

Model = Literal["ur3e", "ur5e", "ur7e", "ur10e", "ur12e", "ur16e"]
SUPPORTED_MODELS: tuple[Model, ...] = ("ur3e", "ur5e", "ur7e", "ur10e", "ur12e", "ur16e")
JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)


def get_asset_root() -> Path:
    """Return the bundled description root (no ROS installation or download)."""
    return Path(__file__).parent / "assets"


def joint_limits(model: Model) -> dict[str, list[float | None]]:
    """Return limits in radians; ``None`` denotes an unbounded continuous joint.

    Raises:
        ValueError: The model is not one of the bundled e-Series profiles.
    """
    if model not in SUPPORTED_MODELS:
        msg = f"Unsupported model: {model!r}"
        raise ValueError(msg)
    return json.loads((get_asset_root() / "limits.json").read_text(encoding="utf-8"))[model]
