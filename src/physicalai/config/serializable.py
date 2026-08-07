# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: PLR0911

"""Serialization utilities for typed dataclass configs."""

from __future__ import annotations

import dataclasses
from enum import Enum
from pathlib import PurePath

__all__ = ["dataclass_to_dict"]


def dataclass_to_dict(obj: object, *, recursive: bool = True) -> object:
    """Convert a dataclass or nested structure to plain Python data.

    Returns:
        Plain dicts, lists, and scalars suitable for ``torch.save(weights_only=True)``.
    """
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        if not recursive:
            return {field.name: getattr(obj, field.name) for field in dataclasses.fields(obj)}
        return {field.name: dataclass_to_dict(getattr(obj, field.name)) for field in dataclasses.fields(obj)}
    if not recursive:
        return obj
    if isinstance(obj, dict):
        return {(key.value if isinstance(key, Enum) else key): dataclass_to_dict(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [dataclass_to_dict(item) for item in obj]
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, PurePath):
        return str(obj)
    if hasattr(obj, "tolist") and hasattr(obj, "ndim"):
        return obj.tolist()  # type: ignore[union-attr]
    return obj
