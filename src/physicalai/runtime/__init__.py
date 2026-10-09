# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Runtime system for running trained policies on robot hardware.

Public API::

    from physicalai.runtime import ActionSource, PolicySource, TeleopSource
    from physicalai.runtime import RobotRuntime, RuntimeCallback
    from physicalai.runtime import RunReason, StopSignal
    from physicalai.runtime import SyncExecution, AsyncExecution, Execution, WorkerDiedError
    from physicalai.runtime import ActionQueue, ChunkedActionQueue
    from physicalai.runtime import ChunkSmoother, LerpSmoother, ReplaceSmoother
    from physicalai.runtime import ActionInterpolator, LinearInterpolator
    from physicalai.runtime import TickEvent, InferenceEvent, LifecycleEvent, MetricsEvent
    from physicalai.runtime import ConsoleCallback, JsonlCallback, AsyncCallback, RerunCallback
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from physicalai.runtime.action_sources import ActionSource, PolicySource, TeleopSource
from physicalai.runtime.callbacks import (
    AsyncCallback,
    ConsoleCallback,
    JsonlCallback,
    LowPassFilterCallback,
    RerunCallback,
)
from physicalai.runtime.core import RobotRuntime, RunReason, RuntimeCallback, StopSignal
from physicalai.runtime.events import InferenceEvent, LifecycleEvent, MetricsEvent, TickEvent
from physicalai.runtime.execution import (
    ActionQueue,
    AsyncExecution,
    ChunkedActionQueue,
    Execution,
    RTCActionQueue,
    RTCExecution,
    SyncExecution,
    WorkerDiedError,
)
from physicalai.runtime.interpolation import ActionInterpolator, LinearInterpolator
from physicalai.runtime.smoothers import ChunkSmoother, LerpSmoother, ReplaceSmoother

if TYPE_CHECKING:
    from physicalai.inference.remote import InferenceServer as InferenceServer
    from physicalai.inference.remote import RemoteInferenceModel as RemoteInferenceModel


def __getattr__(name: str) -> object:
    if name in {"InferenceServer", "RemoteInferenceModel"}:
        from physicalai.inference.remote import InferenceServer, RemoteInferenceModel  # noqa: PLC0415

        return {"InferenceServer": InferenceServer, "RemoteInferenceModel": RemoteInferenceModel}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ActionInterpolator",
    "ActionQueue",
    "ActionSource",
    "AsyncCallback",
    "AsyncExecution",
    "ChunkSmoother",
    "ChunkedActionQueue",
    "ConsoleCallback",
    "Execution",
    "InferenceEvent",
    "JsonlCallback",
    "LerpSmoother",
    "LifecycleEvent",
    "LinearInterpolator",
    "LowPassFilterCallback",
    "MetricsEvent",
    "PolicySource",
    "RTCActionQueue",
    "RTCExecution",
    "ReplaceSmoother",
    "RerunCallback",
    "RobotRuntime",
    "RunReason",
    "RuntimeCallback",
    "StopSignal",
    "SyncExecution",
    "TeleopSource",
    "TickEvent",
    "WorkerDiedError",
]
