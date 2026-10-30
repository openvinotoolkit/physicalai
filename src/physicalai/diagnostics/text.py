# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Small terminal-safe presentation primitives, without terminal dependencies."""


def safe_text(value: object) -> str:
    """Return printable text that cannot inject terminal control sequences."""
    return "".join(char if char.isprintable() else repr(char)[1:-1] for char in str(value))


def table(headers: list[str], rows: list[list[str]]) -> str:
    """Return a compact plain-text table with escaped cell contents."""
    escaped = [[safe_text(cell) for cell in row] for row in [headers, *rows]]
    widths = [max(len(row[column]) for row in escaped) for column in range(len(headers))]
    lines = ["  ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True)).rstrip() for row in escaped]
    lines.insert(1, "  ".join("-" * width for width in widths))
    return "\n".join(lines)
