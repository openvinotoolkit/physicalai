# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np

from physicalai.runtime.observer._console import ConsoleHandler
from physicalai.runtime.observer._recorder import RecorderHandler


class TestConsoleHandler:
    def test_formats_tick(self, capsys: Any) -> None:
        handler = ConsoleHandler(target_fps=30.0)
        handler(
            "sess1",
            "tick",
            {
                "step": 10,
                "physicalai.runtime.loop_duration_s": 0.033,
                "queue_remaining": 5,
                "stale_obs": False,
            },
        )
        captured = capsys.readouterr()
        assert "step=10" in captured.out
        assert "queue=5" in captured.out

    def test_formats_lifecycle(self, capsys: Any) -> None:
        handler = ConsoleHandler()
        handler("sess1", "lifecycle", {"event": "start", "fps": 30})
        captured = capsys.readouterr()
        assert "start" in captured.out


class TestRecorderHandler:
    def test_writes_jsonl(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
            path = Path(f.name)

        recorder = RecorderHandler(path)
        recorder("sess1", "tick", {"step": 0, "value": 1.0})
        recorder("sess1", "lifecycle", {"event": "start"})
        recorder.close()

        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2
        record = json.loads(lines[0])
        assert record["session_id"] == "sess1"
        assert record["topic"] == "tick"
        assert record["step"] == 0

        path.unlink()

    def test_handles_numpy_arrays(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
            path = Path(f.name)

        recorder = RecorderHandler(path)
        recorder("sess1", "tick", {"action": np.array([1.0, 2.0, 3.0])})
        recorder.close()

        line = path.read_text().strip()
        record = json.loads(line)
        assert record["action"] == [1.0, 2.0, 3.0]

        path.unlink()
