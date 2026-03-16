# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Robot conformance testing utilities.

Provides ``check_robot_conformance()`` for verifying that a robot implementation
satisfies the :class:`~physicalai.robot.protocol.Robot` protocol contract.
"""

from __future__ import annotations

import time

import numpy as np

from physicalai.robot.utils import connect

def check_robot_conformance(robot: object, _num_steps: int = 10) -> None:
    """Verify a robot implementation satisfies the Protocol contract.

    Runs a sequence of checks against a *real* (or sufficiently realistic mock)
    robot instance.  The robot must **not** be connected when this function is
    called - the function manages the full lifecycle itself.

    Checks:
        1. ``connect()`` / ``disconnect()`` lifecycle.
        2. ``get_observation()`` returns a dict with ``"state"`` (np.ndarray)
           and ``"timestamp"`` (numeric).
        3. If ``"images"`` is present, it must be a dict of 3-D np.ndarrays.
        4. ``send_action()`` accepts a numpy array shaped like the state.
        5. After ``disconnect()`` -> ``connect()``, the robot should be
           stationary (state unchanged within tolerance over 0.1 s).

    Args:
        robot: An object that is expected to satisfy the Robot protocol.
        _num_steps: Number of observation/action round-trips to execute
            (currently reserved for future use).
    """
    with connect(robot):
        obs = robot.get_observation()  # type: ignore[attr-defined]
        assert isinstance(obs, dict), "get_observation() must return a dict"  # noqa: S101

        assert "state" in obs, "observation must contain 'state'"  # noqa: S101
        assert isinstance(obs["state"], np.ndarray), "state must be np.ndarray"  # noqa: S101

        assert "timestamp" in obs, "observation must contain 'timestamp'"  # noqa: S101
        assert isinstance(obs["timestamp"], (int, float)), "timestamp must be numeric"  # noqa: S101

        if "images" in obs:
            assert isinstance(obs["images"], dict), "images must be a dict"  # noqa: S101
            for name, img in obs["images"].items():
                assert isinstance(img, np.ndarray), f"image '{name}' must be np.ndarray"  # noqa: S101
                assert img.ndim == 3, f"image '{name}' must be 3D (C, H, W)"  # noqa: S101, PLR2004

        # Echo the state back as an action to verify send_action() accepts it without error
        action = obs["state"].copy()
        robot.send_action(action)  # type: ignore[attr-defined]

        robot.disconnect()  # type: ignore[attr-defined]
        robot.connect()  # type: ignore[attr-defined]

        obs1 = robot.get_observation()  # type: ignore[attr-defined]
        time.sleep(0.1)
        obs2 = robot.get_observation()  # type: ignore[attr-defined]

        assert np.allclose(obs1["state"], obs2["state"], atol=0.01), (  # noqa: S101
            "Robot must be stationary after disconnect(). "
            f"State changed from {obs1['state']} to {obs2['state']}"
        )
