# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Fuzz deterministic ChunkedActionQueue push/pop operation sequences."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import atheris
import numpy as np

with atheris.instrument_imports():
    from physicalai.runtime.execution.queue import ChunkedActionQueue
    from physicalai.runtime.smoothers import LerpSmoother, ReplaceSmoother


def test_one_input(data: bytes) -> None:
    if len(data) < 8:
        return

    fdp = atheris.FuzzedDataProvider(data)
    use_lerp = fdp.ConsumeBool()
    smoother = LerpSmoother(duration_frames=fdp.ConsumeIntInRange(0, 16)) if use_lerp else ReplaceSmoother()
    queue = ChunkedActionQueue(smoother=smoother)

    action_dim = fdp.ConsumeIntInRange(1, 16)
    n_ops = fdp.ConsumeIntInRange(1, 12)
    expected_remaining = 0
    expected_consecutive_holds = 0
    expected_total_holds = 0
    expected_total_pops = 0

    for _ in range(n_ops):
        if fdp.ConsumeBool():
            rows = fdp.ConsumeIntInRange(0, 16)
            offset = fdp.ConsumeIntInRange(0, rows + 3)
            n_bytes = rows * action_dim * 4
            raw = fdp.ConsumeBytes(n_bytes)
            if len(raw) < n_bytes:
                raw += b"\x00" * (n_bytes - len(raw))
            chunk = np.frombuffer(raw[:n_bytes], dtype=np.float32).copy().reshape((rows, action_dim))
            queue.push_chunk(chunk, offset=offset)
            expected_remaining = max(rows - offset, 0)
        else:
            result = queue.pop()
            if expected_remaining:
                assert result is not None
                assert result.ndim == 1
                expected_remaining -= 1
                expected_consecutive_holds = 0
                expected_total_pops += 1
            else:
                assert result is None
                expected_consecutive_holds += 1
                expected_total_holds += 1

        assert queue.remaining == expected_remaining
        assert queue.consecutive_holds == expected_consecutive_holds
        assert queue.total_holds == expected_total_holds
        assert queue.total_pops == expected_total_pops


def main() -> None:
    atheris.Setup(sys.argv, test_one_input)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
