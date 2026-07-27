# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Errors for component configuration export and instantiation."""

from __future__ import annotations


class ComponentConfigError(Exception):
    """Raised when a component config cannot be exported, validated, or built.

    Callers typically recover by correcting configuration. Subclasses mark
    distinct failure phases when callers need to distinguish them.
    """


class ComponentImportError(ComponentConfigError):
    """Raised when a ``class_path`` cannot be resolved to an importable class."""
