# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Event dataclasses for the runtime callback bus."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

    import numpy as np

    from physicalai.capture.frame import Frame
    from physicalai.robot.interface import RobotObservation


@dataclass(frozen=True, slots=True)
class TickEvent:
    """Emitted once per control-loop tick.

    ``robot_state`` and ``camera_frames`` are the same eager, plain values
    read once this tick and passed to the action source's ``update()`` — no
    lazy handle, no wrapper.

    Timestamps:
        - ``timestamp``: wall-clock UTC seconds (``time.time()``) when the event was emitted.
    """

    session_id: str
    step: int
    timestamp: float
    robot_state: RobotObservation
    camera_frames: Mapping[str, Frame]
    action_sent: np.ndarray | None
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


@dataclass(frozen=True, slots=True)
class MetricsEvent:
    """Optional, fire-and-forget per-tick live values owned by an action source.

    Unlike :class:`TickEvent` (runtime-owned) and :class:`InferenceEvent`
    (execution-owned, fires on inference completion), ``MetricsEvent`` carries
    source-owned values with no other home — e.g. ``queue_remaining`` for a
    policy source. An action source with nothing to report simply never emits
    one; there is no capability protocol or ``isinstance`` check involved.
    """

    session_id: str
    step: int
    timestamp: float
    values: Mapping[str, float]
