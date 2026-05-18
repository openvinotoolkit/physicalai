# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Event dataclasses for the runtime callback bus."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np


@dataclass(frozen=True, slots=True)
class TickEvent:
    """Emitted once per control-loop tick.

    All timestamps are wall-clock UTC seconds (``time.time()``).
    """

    session_id: str
    step: int
    timestamp: float
    joint_positions: np.ndarray | None
    action_sent: np.ndarray | None
    queue_remaining: int
    loop_duration_s: float
    sleep_time_s: float
    stale_obs: bool


@dataclass(frozen=True, slots=True)
class InferenceEvent:
    """Emitted when an inference call completes (sync or async).

    All timestamps are wall-clock UTC seconds (``time.time()``).
    """

    session_id: str
    timestamp: float
    latency_s: float
    offset: int
    chunk: np.ndarray


@dataclass(frozen=True, slots=True)
class LifecycleEvent:
    """Emitted on session boundaries and error conditions.

    All timestamps are wall-clock UTC seconds (``time.time()``).
    """

    session_id: str
    timestamp: float
    event: str
    metadata: dict[str, Any]
