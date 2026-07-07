# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Append-only JSONL recording callback."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

    from physicalai.runtime.events import InferenceEvent, LifecycleEvent, TickEvent


class JsonlCallback:
    """Append-only JSONL recording. Numpy arrays converted to lists."""

    def __init__(self, path: str | Path, *, record_chunks: bool = False) -> None:  # noqa: D107
        self._path = Path(path)
        self._file = self._path.open("a")
        self._record_chunks = record_chunks

    def on_tick(self, event: TickEvent) -> None:  # noqa: D102
        self._write(
            "tick",
            {
                "session_id": event.session_id,
                "step": event.step,
                "timestamp": event.timestamp,
                "joint_positions": _np_to_list(event.robot_state.joint_positions),
                "action_sent": _np_to_list(event.action_sent),
                "loop_duration_s": event.loop_duration_s,
                "sleep_time_s": event.sleep_time_s,
                "stale_obs": event.stale_obs,
            },
        )

    def on_inference(self, event: InferenceEvent) -> None:  # noqa: D102
        payload: dict[str, Any] = {
            "session_id": event.session_id,
            "timestamp": event.timestamp,
            "latency_s": event.latency_s,
            "offset": event.offset,
            "chunk_shape": list(event.chunk.shape),
        }
        if self._record_chunks:
            payload["chunk"] = event.chunk.tolist()
        self._write("inference", payload)

    def on_lifecycle(self, event: LifecycleEvent) -> None:  # noqa: D102
        self._write(
            "lifecycle",
            {
                "session_id": event.session_id,
                "timestamp": event.timestamp,
                "event": event.event,
                "metadata": event.metadata,
            },
        )

    def close(self) -> None:  # noqa: D102
        self._file.close()

    def _write(self, kind: str, payload: dict[str, Any]) -> None:
        record = {"type": kind, **payload}
        self._file.write(json.dumps(record, default=_json_default) + "\n")
        self._file.flush()


def _np_to_list(arr: np.ndarray | None) -> list[float] | None:
    if arr is None:
        return None
    return arr.tolist()


def _json_default(obj: object) -> Any:  # noqa: ANN401
    import numpy as np  # noqa: PLC0415

    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    msg = f"Object of type {type(obj)} is not JSON serializable"
    raise TypeError(msg)
