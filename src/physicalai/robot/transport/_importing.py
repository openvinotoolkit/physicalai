# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Dotted-path imports for trusted robot owner configuration."""

from __future__ import annotations

import importlib


def import_dotted_path(path: str) -> object:
    """Resolve a fully-qualified path, including nested attributes.

    Args:
        path: Dotted path such as ``"pkg.mod.Outer.Inner"``.

    Returns:
        The resolved object.

    Raises:
        ValueError: If no module prefix can be imported.
    """
    if "." not in path:
        msg = f"dotted path must contain at least one '.': {path!r}"
        raise ValueError(msg)

    segments = path.split(".")
    for split_index in range(len(segments), 0, -1):
        try:
            obj: object = importlib.import_module(".".join(segments[:split_index]))
        except ImportError:
            continue
        for attr in segments[split_index:]:
            obj = getattr(obj, attr)
        return obj

    msg = f"could not import any module prefix of {path!r}"
    raise ValueError(msg)
