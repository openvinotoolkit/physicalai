# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Optional ANSI emphasis applied after plain-text layout and escaping."""

from __future__ import annotations

import os
import re
import sys
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import TextIO

_TOKENS = re.compile(
    r"(?P<header>^(?:Camera throughput|Serial throughput|USB topology|Shared upstream links|"
    r"Needs attention|Identity and endpoint details)[^\n]*)"
    r"|(?P<doctor>^Doctor:)"
    r"|(?P<usb2>USB 2\.0)"
    r"|(?P<usb3>USB 3\.x)"
    r"|(?P<rate>\b\d+(?:\.\d+)? (?:MB/s|Mb/s|Gb/s|FPS)\b)"
    r"|(?P<status>\b(?:FAIL|WARN|PASS|SKIP)\b)"
    r"|(?P<unknown>\b(?:not measured|unknown)\b)"
    r"|(?P<selected><selected>)",
    re.MULTILINE,
)
_STYLES = {
    "header": "1",
    "doctor": "1",
    "usb2": "1;33",
    "usb3": "1;36",
    "rate": "1",
    "unknown": "2",
    "selected": "1;36",
}
_STATUS = {"FAIL": "1;31", "WARN": "1;33", "PASS": "32", "SKIP": "2"}


def color_enabled(
    mode: Literal["auto", "always", "never"],
    *,
    stream: TextIO | None = None,
    environ: Mapping[str, str] | None = None,
) -> bool:
    """Return whether human output should use ANSI, honoring explicit overrides.

    Auto mode requires a TTY, a non-dumb terminal, and no ``NO_COLOR`` variable.
    Explicit ``always`` supports forced-color pagers; JSON bypasses styling.
    """
    if mode != "auto":
        return mode == "always"
    environment = os.environ if environ is None else environ
    output = sys.stdout if stream is None else stream
    return output.isatty() and "NO_COLOR" not in environment and environment.get("TERM") != "dumb"


def emphasize(text: str, *, enabled: bool) -> str:
    """Return emphasized output without affecting table widths or plain output.

    Call only on terminal-escaped renderer output, not raw device metadata.
    """
    if not enabled:
        return text

    def replace(match: re.Match[str]) -> str:
        style = _STATUS[match.group()] if match.lastgroup == "status" else _STYLES[match.lastgroup or "rate"]
        return f"\x1b[{style}m{match.group()}\x1b[0m"

    return _TOKENS.sub(replace, text)
