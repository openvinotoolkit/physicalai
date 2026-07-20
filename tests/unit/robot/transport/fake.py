# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""FakeRobot implementation for transport testing.

A concrete driver satisfying the :class:`~physicalai.robot.Robot` protocol
without touching any hardware. Ships a state vector that differs from
``joint_positions`` (positions + velocities) so tests can verify the
owner-computed state is shipped as-is, not re-derived subscriber-side.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

NUM_JOINTS = 6


@dataclass
class FakeObservation:
    joint_positions: np.ndarray
    timestamp: float
    sensor_data: dict[str, np.ndarray] | None = None
    images: dict | None = None

    @property
    def state(self) -> np.ndarray:
        """Positions + velocities, mimicking WidowXAI's concat contract."""
        if self.sensor_data and "velocities" in self.sensor_data:
            return np.concatenate([self.joint_positions, self.sensor_data["velocities"]])
        return self.joint_positions


class FakeRobot:
    """In-memory robot producing synthetic observations.

    Args:
        port: Identity source for :attr:`device_ids`; unused otherwise.
        device_ids: Explicit override for :attr:`device_ids` — lets tests
            exercise single-device, multi-device, and virtual (empty
            tuple) cases without relying on any constructor-kwarg
            convention.
        fail_connect: If True, ``connect()`` raises (owner ERROR-path tests).
    """

    def __init__(
        self,
        port: str = "/dev/fake0",
        *,
        device_ids: tuple[str, ...] | None = None,
        fail_connect: bool = False,
        fail_observation: bool = False,
        fail_observation_after: int | None = None,
        fail_disconnect: bool = False,
        disconnect_marker: str | None = None,
        **_ignored: object,
    ) -> None:
        self._port = port
        self._device_ids = device_ids if device_ids is not None else (f"fake:{port}",)
        self._fail_connect = fail_connect
        self._fail_observation = fail_observation
        self._fail_observation_after = fail_observation_after
        self._fail_disconnect = fail_disconnect
        self._disconnect_marker = disconnect_marker
        self._observation_calls = 0
        self._connected = False
        self._last_action = np.zeros(NUM_JOINTS, dtype=np.float32)
        self.disconnect_called = False

    @property
    def joint_names(self) -> list[str]:
        return [f"joint_{i}" for i in range(NUM_JOINTS)]

    @property
    def device_ids(self) -> tuple[str, ...]:
        return self._device_ids

    def connect(self) -> None:
        if self._fail_connect:
            msg = f"fake hardware failure on {self._port}"
            raise ConnectionError(msg)
        self._connected = True

    def disconnect(self) -> None:
        self.disconnect_called = True
        self._connected = False
        if self._disconnect_marker is not None:
            Path(self._disconnect_marker).touch()
        if self._fail_disconnect:
            msg = f"fake disconnect failure on {self._port}"
            raise RuntimeError(msg)

    def get_observation(self) -> FakeObservation:
        if not self._connected:
            msg = "Robot is not connected. Call connect() first."
            raise ConnectionError(msg)
        self._observation_calls += 1
        if self._fail_observation or (
            self._fail_observation_after is not None and self._observation_calls > self._fail_observation_after
        ):
            msg = f"fake observation failure on {self._port}"
            raise RuntimeError(msg)
        # Echo the last commanded action as measured position so tests can
        # observe the action round-trip through the owner loop.
        return FakeObservation(
            joint_positions=self._last_action.copy(),
            timestamp=time.monotonic(),
            sensor_data={"velocities": np.ones(NUM_JOINTS, dtype=np.float32)},
        )

    def send_action(self, action: np.ndarray, *, goal_time: float = 0.1) -> None:  # noqa: ARG002
        if not self._connected:
            msg = "Robot is not connected. Call connect() first."
            raise ConnectionError(msg)
        self._last_action = np.asarray(action, dtype=np.float32).copy()

    def is_connected(self) -> bool:
        return self._connected
