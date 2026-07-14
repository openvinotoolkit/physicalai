# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Shared dotted-path import primitive.

Resolves a dotted path like ``"pkg.mod.ClassName"`` (or a nested qualified
name like ``"pkg.mod.Outer.Inner"``) to the object it names. Several
call sites in this codebase need this exact primitive — the inference
component factory, the inference adapter registry, and the robot transport's
arbitrary-robot-class resolution — so it lives here once instead of being
reimplemented per call site.

This module makes no trust decisions and does not check the result's type;
callers are responsible for deciding whether a given path is trusted input
and whether the resolved object must be a class, a callable, etc.
"""

from __future__ import annotations

import importlib


def import_dotted_path(path: str) -> object:
    """Import an object from a fully-qualified dotted path.

    Tries the *longest* importable module prefix first, then walks the
    remaining dotted segments as attribute access. A naive
    ``path.rsplit(".", 1)`` assumes the last segment is always the
    attribute and everything before it is the module — which breaks for
    nested qualified names (e.g. a class defined inside another class,
    ``pkg.mod.Outer.Inner``). Trying progressively shorter module prefixes
    correctly finds the real module/attribute boundary in both cases.

    Args:
        path: Fully-qualified dotted path, e.g. ``"physicalai.robot.SO101"``
            or ``"pkg.mod.Outer.Inner"``.

    Returns:
        The resolved object (a module attribute, possibly nested).

    Raises:
        ValueError: If *path* has no ``.`` separator, or no prefix of it
            imports successfully as a module.

    Note:
        A module prefix that imports but whose remaining attribute chain
        cannot be resolved lets the natural ``AttributeError`` from
        ``getattr`` propagate — not wrapped, so tracebacks point at the
        exact missing attribute.
    """
    if "." not in path:
        msg = f"dotted path must contain at least one '.': {path!r}"
        raise ValueError(msg)

    segments = path.split(".")
    module = None
    split_index = 0
    for i in range(len(segments), 0, -1):
        candidate = ".".join(segments[:i])
        try:
            module = importlib.import_module(candidate)
        except ImportError:
            continue
        split_index = i
        break

    if module is None:
        msg = f"could not import any module prefix of {path!r}"
        raise ValueError(msg)

    obj: object = module
    for attr in segments[split_index:]:
        obj = getattr(obj, attr)
    return obj
