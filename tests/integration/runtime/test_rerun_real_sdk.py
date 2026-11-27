# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""RerunCallback in connect mode against the real rerun-sdk when no viewer is running.

Runs in a subprocess with a hard timeout, so a call that never returns fails the test
instead of hanging the suite.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

pytest.importorskip("rerun")

_SCRIPT = r"""
import json, os, socket, sys, time
import numpy as np
from physicalai.capture.frame import Frame
from physicalai.runtime import RerunCallback
from physicalai.runtime.events import LifecycleEvent, MetricsEvent, TickEvent

# Bound for the whole run but never listening: connects are refused and no other
# process can take the port meanwhile.
held = socket.socket()
held.bind(("127.0.0.1", 0))
port = held.getsockname()[1]

class State:
    joint_positions = np.zeros(6)

# Three 1080p cameras: about 18 MiB per tick. On main, rr.log blocks once the SDK's
# 50 MiB input budget is full, within a few ticks.
rng = np.random.default_rng(0)
frames = {f"cam{i}": Frame(rng.integers(0, 255, (1080, 1920, 3), dtype=np.uint8), 0.0, 0) for i in range(3)}
cb = RerunCallback(mode="connect", connect_addr=f"127.0.0.1:{port}", image_decimation=1)
t = time.perf_counter()
cb.on_lifecycle(LifecycleEvent("s", time.time(), "start", {"fps": 30, "joint_names": [f"j{i}" for i in range(6)]}))
start_s = time.perf_counter() - t
costs = []
for step in range(int(sys.argv[1])):
    t = time.perf_counter()
    cb.on_tick(TickEvent(
        session_id="s", step=step, timestamp=time.time(), robot_state=State(), camera_frames=frames,
        action_sent=np.zeros(6), loop_duration_s=0.0, sleep_time_s=0.0, stale_obs=False, source_updated=True,
    ))
    cb.on_metrics(MetricsEvent("s", step, time.time(), {"queue_remaining": 1.0}))
    costs.append(time.perf_counter() - t)
    print(json.dumps({"ticks": len(costs)}), flush=True)  # progress, readable even if a later tick hangs
    time.sleep(max(0.0, 1 / 30 - costs[-1]))
print(json.dumps({"ticks": len(costs), "start_s": start_s, "max_tick_s": max(costs)}), flush=True)
os._exit(0)
"""


@pytest.mark.integration
@pytest.mark.slow
def test_no_viewer_running_does_not_stall_the_hooks() -> None:
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in sys.path if p)}
    try:
        proc = subprocess.run(  # noqa: S603
            [sys.executable, "-c", _SCRIPT, "300"], env=env, capture_output=True, text=True, timeout=20, check=False
        )
    except subprocess.TimeoutExpired as exc:
        progress = (exc.stdout.decode() if isinstance(exc.stdout, bytes) else exc.stdout or "").strip().splitlines()
        pytest.fail(f"still running after 20 s; last progress: {progress[-1:] or 'none'}")
        raise  # unreachable: pytest.fail raises
    assert proc.returncode == 0, proc.stderr[-2000:]
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert "start_s" in result, proc.stderr[-2000:]  # the final summary, not a progress line

    assert result["ticks"] == 300
    assert result["start_s"] < 1.0
    assert result["max_tick_s"] < 0.2


@pytest.mark.integration
def test_no_viewer_leaves_an_unrelated_recording_alone() -> None:
    import socket

    import numpy as np
    import rerun as rr

    from physicalai.runtime import RerunCallback
    from physicalai.runtime.events import LifecycleEvent, MetricsEvent, TickEvent

    other = rr.RecordingStream("unrelated-app")
    memory = rr.memory_recording(recording=other)
    previous = rr.get_global_data_recording()
    rr.set_global_data_recording(other)
    try:
        other.flush()
        before = memory.num_msgs()
        with socket.socket() as held:  # bound, never listening
            held.bind(("127.0.0.1", 0))
            cb = RerunCallback(mode="connect", connect_addr=f"127.0.0.1:{held.getsockname()[1]}")
            cb.on_lifecycle(LifecycleEvent("s", 0.0, "start", {"fps": 30}))

        class State:
            joint_positions = np.zeros(6)

        cb.on_tick(
            TickEvent(
                session_id="s", step=0, timestamp=0.0, robot_state=State(), camera_frames={},  # type: ignore[arg-type]
                action_sent=np.zeros(6), loop_duration_s=0.0, sleep_time_s=0.0, stale_obs=False, source_updated=True,
            )
        )
        cb.on_metrics(MetricsEvent("s", 0, 0.0, {"queue_remaining": 1.0}))
        cb.on_lifecycle(LifecycleEvent("s", 1.0, "shutdown", {}))
        cb.close()
        other.flush()
        assert memory.num_msgs() == before
    finally:
        rr.set_global_data_recording(previous)  # type: ignore[arg-type]
