# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Shared argument-path formatting for component-config errors."""

from __future__ import annotations


def format_path(path: str) -> str:
    """Return *path* for error messages, or ``<root>`` when empty."""
    return path or "<root>"
