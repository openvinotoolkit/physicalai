# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Robot verification utilities.

Provides ``verify_robot()`` for verifying that a robot implementation
satisfies the :class:`~physicalai.robot.interface.Robot` protocol contract.
"""

from __future__ import annotations

import time

import numpy as np
from loguru import logger

from physicalai.robot.connect import connect


class RobotVerificationError(ValueError):
    """Raised when a robot implementation violates the protocol contract."""


def verify_robot(robot: object, _num_steps: int = 10) -> None:
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

    Raises:
        RobotVerificationError: If any protocol contract check fails.
    """
    logger.info("Verifying robot: {}", repr(robot))

    def _fail(msg: str) -> None:
        logger.error("Verification failed: {}", msg)
        raise RobotVerificationError(msg)

    with connect(robot):
        logger.success("Check 1 passed: connect() lifecycle")

        obs = robot.get_observation()
        if not isinstance(obs, dict):
            _fail("get_observation() must return a dict")
        logger.debug("get_observation() returned dict")

        if "state" not in obs:
            _fail("observation must contain 'state'")
        if not isinstance(obs["state"], np.ndarray):
            _fail("state must be np.ndarray")
        logger.debug("state: shape={}, dtype={}", obs["state"].shape, obs["state"].dtype)

        if "timestamp" not in obs:
            _fail("observation must contain 'timestamp'")
        if not isinstance(obs["timestamp"], (int, float)):
            _fail("timestamp must be numeric")
        logger.debug("timestamp: {}", obs["timestamp"])
        logger.success("Check 2 passed: observation contains valid 'state' and 'timestamp'")

        if "images" in obs:
            if not isinstance(obs["images"], dict):
                _fail("images must be a dict")
            for name, img in obs["images"].items():
                if not isinstance(img, np.ndarray):
                    _fail(f"image '{name}' must be np.ndarray")
                if img.ndim != 3:
                    _fail(f"image '{name}' must be 3D (C, H, W), got ndim={img.ndim}")
                logger.debug("image '{}': shape={}, dtype={}", name, img.shape, img.dtype)
            logger.success("Check 3 passed: images dict contains valid 3D arrays")

        # Echo the state back as an action to verify send_action() accepts it without error
        action = obs["state"].copy()
        robot.send_action(action)
        logger.success("Check 4 passed: send_action() accepted state-shaped action")

        logger.debug("Testing disconnect() -> connect() lifecycle")
        robot.disconnect()
        robot.connect()

        obs1 = robot.get_observation()
        time.sleep(0.1)
        obs2 = robot.get_observation()

        if not np.allclose(obs1["state"], obs2["state"], atol=0.01):
            _fail(
                f"Robot must be stationary after disconnect(). "
                f"State changed from {obs1['state']} to {obs2['state']}"
            )
        logger.success("Check 5 passed: robot is stationary after disconnect() -> connect()")

    logger.success("All checks passed for robot: {}", repr(robot))
