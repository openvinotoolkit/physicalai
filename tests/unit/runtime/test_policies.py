# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import numpy as np

from physicalai.runtime import ActionQueue, PolicyRuntime, SyncExecution, TeleoperatorPolicy


@dataclass
class FakeRobotObservation:
    joint_positions: np.ndarray
    timestamp: float
    sensor_data: dict[str, np.ndarray] | None = None
    images: dict | None = None

    @property
    def state(self) -> np.ndarray:
        return self.joint_positions


def _make_robot(joint_positions: np.ndarray) -> MagicMock:
    robot = MagicMock()
    robot.joint_names = [f"joint_{i}" for i in range(joint_positions.shape[0])]
    robot.get_observation.return_value = FakeRobotObservation(
        joint_positions=joint_positions,
        timestamp=time.monotonic(),
    )
    return robot


def test_teleoperator_policy_returns_leader_state_as_single_action_chunk() -> None:
    leader_state = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    leader = _make_robot(leader_state)
    policy = TeleoperatorPolicy(leader)

    chunk = policy.predict_action_chunk({"state": np.zeros((1, 3), dtype=np.float32)})

    assert chunk.shape == (1, 3)
    assert chunk.dtype == np.float32
    np.testing.assert_array_equal(chunk[0], leader_state.astype(np.float32))


def test_policy_runtime_can_drive_follower_from_teleoperator_policy() -> None:
    leader_state = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    leader = _make_robot(leader_state)
    follower = _make_robot(np.zeros(3, dtype=np.float32))

    runtime = PolicyRuntime(
        robot=follower,
        model=TeleoperatorPolicy(leader),
        execution=SyncExecution(request_threshold=1.0),
        action_queue=ActionQueue(),
        fps=10.0,
    )
    runtime._connected = True  # noqa: SLF001

    with patch("physicalai.runtime.runtime.time") as mock_time:
        mock_time.perf_counter.return_value = 0.0
        mock_time.sleep = MagicMock()
        mock_time.time.return_value = 0.0
        stats = runtime.run(duration_s=0.3)

    assert stats.steps == 3
    assert follower.send_action.call_count == 3
    for call in follower.send_action.call_args_list:
        np.testing.assert_array_equal(call.args[0], leader_state)
