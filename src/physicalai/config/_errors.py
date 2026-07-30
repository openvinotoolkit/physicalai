# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Errors for configuration export and instantiation."""

from __future__ import annotations


class ConfigError(Exception):
    """Raised when a config cannot be exported, validated, or built.

    Callers typically recover by correcting configuration. Subclasses mark
    distinct failure phases when callers need to distinguish them.
    """


class ConfigImportError(ConfigError):
    """Raised when a ``class_path`` cannot be resolved to an importable class."""
