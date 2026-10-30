# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Deterministic watch tests with injected time and no real waiting."""

import json

from physicalai.diagnostics.checks import assess
from physicalai.diagnostics.models import Alias, Device, Endpoint, Inventory
from physicalai.diagnostics.watch import changes, watch


def snapshot(path: str, speed: float = 5000) -> Inventory:
    return Inventory(platform="linux", devices=[Device(
        id="usb:3-1", name="camera", serial="SN", speed_mbps=speed,
        endpoints=[Endpoint(id="capture:port1", path=path, kind="video", role="capture", aliases=[
            Alias(path="/dev/v4l/by-path/port1", binding="port", target=path),
        ])],
    )])


def test_renumber_and_speed_change_are_not_new_camera_identity() -> None:
    events = changes(assess(snapshot("/dev/video40")), assess(snapshot("/dev/video44", 480)))
    assert [event.kind for event in events].count("device_changed") == 1
    assert not any(event.kind in {"device_added", "device_removed"} for event in events)
    change = next(event for event in events if event.kind == "device_changed")
    assert change.data["identity_verified"] is False
    assert any(event.kind == "finding_opened" for event in events)


def test_bounded_watch_ndjson_with_fake_clock() -> None:
    class Clock:
        now = 0.0

        def clock(self):
            return self.now

        def sleep(self, seconds):
            self.now += seconds

    class Collector:
        def collect(self):
            return snapshot("/dev/video40")

    clock = Clock()
    events = list(watch(Collector(), duration=2, interval=1, clock=clock.clock, sleep=clock.sleep))
    assert [event.kind for event in events] == ["snapshot", "final"]
    assert [json.loads(event.model_dump_json())["sequence"] for event in events] == [1, 2]
    assert clock.now == 2
