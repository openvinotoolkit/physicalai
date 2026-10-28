# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import re
from typing import Any, cast

from physicalai.diagnostics.styling import color_enabled, emphasize


class Output:
    def __init__(self, tty: bool):
        self.tty = tty
    def isatty(self):
        return self.tty


def test_auto_color_requires_tty_and_respects_no_color_and_dumb_term() -> None:
    tty, pipe = cast(Any, Output(True)), cast(Any, Output(False))
    assert color_enabled("auto", stream=tty, environ={})
    assert not color_enabled("auto", stream=pipe, environ={})
    assert not color_enabled("auto", stream=tty, environ={"NO_COLOR": "1"})
    assert not color_enabled("auto", stream=tty, environ={"TERM": "dumb"})
    assert color_enabled("always", stream=pipe, environ={"NO_COLOR": "1"})
    assert not color_enabled("never", stream=tty, environ={})


def test_emphasis_preserves_plain_layout_and_highlights_generation_and_rate() -> None:
    plain = "USB topology\n└── Camera — USB 2.0 / 480 Mb/s link\nWARN: not measured\nUSB 3.x / 5 Gb/s link"
    styled = emphasize(plain, enabled=True)
    assert "\x1b[1;33mUSB 2.0\x1b[0m" in styled
    assert "\x1b[1;36mUSB 3.x\x1b[0m" in styled
    assert "\x1b[1m480 Mb/s\x1b[0m" in styled
    assert re.sub(r"\x1b\[[0-9;]*m", "", styled) == plain
    assert emphasize(plain, enabled=False) == plain
