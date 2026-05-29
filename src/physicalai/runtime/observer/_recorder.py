# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class _NumpyEncoder(json.JSONEncoder):
    def default(self, o: object) -> object:
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        return super().default(o)


class RecorderHandler:
    def __init__(self, output_path: Path) -> None:
        self._path = Path(output_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = Path(self._path).open("a", encoding="utf-8")  # noqa: SIM115

    def __call__(self, session_id: str, topic: str, payload: dict[str, Any]) -> None:
        record = {"session_id": session_id, "topic": topic, **payload}
        self._file.write(json.dumps(record, cls=_NumpyEncoder) + "\n")
        self._file.flush()

    def close(self) -> None:
        if self._file and not self._file.closed:
            self._file.close()
