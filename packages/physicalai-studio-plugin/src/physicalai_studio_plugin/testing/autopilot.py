# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Example payloads of the autopilot protocol (:mod:`physicalai_studio_plugin.autopilot`), as JSON decodes them.

For a simulation and a client to test against the same messages. Each example names the model it is one of.
"""

from __future__ import annotations

import copy
from typing import Any

_EXAMPLES: dict[str, tuple[str, dict[str, Any]]] = {
    # GET {path}: a scene with an autopilot, no episode yet.
    "info_supported": (
        "AutopilotInfo",
        {
            "protocol": 1,
            "scene": "bucket_sort",
            "supported": True,
            "reason": None,
            "options": ["max_attempts", "recovery_fraction"],
            "joint_names": ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "gripper"],
            "episode": None,
        },
    ),
    # GET {path}: a scene without a demonstrator.
    "info_unsupported": (
        "AutopilotInfo",
        {
            "protocol": 1,
            "scene": "tabletop_cube",
            "supported": False,
            "reason": "Scene 'tabletop_cube' has no autopilot; automatic recording needs conveyor or bucket_sort.",
            "options": [],
            "joint_names": [],
            "episode": None,
        },
    ),
    # POST {path}/episodes
    "episode_request": ("EpisodeRequest", {"options": {"max_attempts": 3, "recovery_fraction": 0.2}}),
    "episode_preparing": ("Episode", {"id": "e-0007", "phase": "preparing"}),
    "episode_ready": ("Episode", {"id": "e-0007", "phase": "ready", "task": "Put the red cube in the blue bucket."}),
    "episode_running": (
        "Episode",
        {"id": "e-0007", "phase": "running", "task": "Put the red cube in the blue bucket."},
    ),
    "episode_success_regrasp": (
        "Episode",
        {
            "id": "e-0007",
            "phase": "ended",
            "task": "Put the red cube in the blue bucket.",
            "outcome": "success",
            "reason": "Success after a regrasp (the cube slipped).",
            "clean": False,
        },
    ),
    "episode_aborted_by_liveness": (
        "Episode",
        {
            "id": "e-0008",
            "phase": "ended",
            "task": "Sort the items into the bin of their color.",
            "outcome": "aborted",
            "reason": "Physical AI Studio stopped reading the leader.",
        },
    ),
    "error_wrong_phase": ("AutopilotError", {"code": "wrong_phase", "message": "Episode 'e-0007' is not ready."}),
    "error_bad_option": (
        "AutopilotError",
        {"code": "bad_option", "message": "Scene 'conveyor_sort' does not take recovery_fraction."},
    ),
}


def example_names() -> tuple[str, ...]:
    """Return the names of every example."""
    return tuple(_EXAMPLES)


def example(name: str) -> dict[str, Any]:
    """Return a fresh copy of the example *name*, as JSON decodes it (``KeyError`` for another name)."""
    return copy.deepcopy(_EXAMPLES[name][1])


def example_model(name: str) -> str:
    """Return the name of the model in :mod:`physicalai_studio_plugin.autopilot` that example *name* is one of."""
    return _EXAMPLES[name][0]


__all__ = ["example", "example_model", "example_names"]
