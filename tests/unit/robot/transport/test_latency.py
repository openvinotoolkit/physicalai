# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Action-latency jitter measurement under the D20 QoS settings.

Batching regressions (Zenoh's throughput-tuned defaults delaying small
messages) are invisible to functional tests — this measures the full
action round trip (send -> owner applies -> state reflects it) at the
target loop rate and asserts the p99 stays within a loop-period budget.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
import pytest

from physicalai.robot.transport import SharedRobot

from .conftest import FAKE_ROBOT_CLASS, requires_zenoh

if TYPE_CHECKING:
    from collections.abc import Generator

_RATE_HZ = 100.0
_PERIOD_S = 1.0 / _RATE_HZ
_NUM_SAMPLES = 50
# Round trip spans up to ~2 owner periods (action pickup + state publish)
# plus transport overhead; 5 periods of headroom keeps CI stable while
# still catching the ~x00 ms stalls batching regressions introduce.
_P99_BUDGET_S = 5 * _PERIOD_S


@requires_zenoh
@pytest.mark.slow
class TestActionLatency:
    @pytest.fixture
    def robot(self, unique_id: str) -> Generator[SharedRobot, None, None]:
        robot = SharedRobot(
            unique_id.replace("/", "-"),
            robot={
                "class_path": FAKE_ROBOT_CLASS,
                "init_args": {"device_ids": [f"fake:{unique_id}"]},
            },
            rate_hz=_RATE_HZ,
            idle_timeout=3.0,
        )
        robot.connect()
        yield robot
        owner = robot._owner
        robot.disconnect()
        if owner is not None:
            owner.stop()

    def test_p99_action_latency_within_budget(self, robot: SharedRobot) -> None:
        latencies: list[float] = []
        for i in range(1, _NUM_SAMPLES + 1):
            target = np.full(6, float(i), dtype=np.float32)
            t0 = time.monotonic()
            robot.send_action(target, goal_time=0.01)

            deadline = t0 + 2.0
            while time.monotonic() < deadline:
                obs = robot.get_observation()
                if np.array_equal(obs.joint_positions, target):
                    latencies.append(time.monotonic() - t0)
                    break
                time.sleep(0.0005)
            else:
                pytest.fail(f"action {i} never reflected in state within 2s")

            time.sleep(_PERIOD_S)  # pace at the loop rate

        p99 = float(np.percentile(latencies, 99))
        p50 = float(np.percentile(latencies, 50))
        assert p99 < _P99_BUDGET_S, f"p99={p99 * 1e3:.1f}ms (p50={p50 * 1e3:.1f}ms) exceeds {_P99_BUDGET_S * 1e3:.0f}ms"
