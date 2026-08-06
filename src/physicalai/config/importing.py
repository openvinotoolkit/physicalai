# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Dotted-path imports for trusted configuration."""

from __future__ import annotations

import importlib

__all__ = ["import_dotted_path"]


def import_dotted_path(path: str) -> object:
    """Resolve a fully-qualified path, including nested attributes.

    Args:
        path: Dotted path such as ``"pkg.mod.Outer.Inner"``.

    Returns:
        The resolved object.

    Raises:
        ValueError: If no module prefix can be imported, *path* has no ``'.'``,
            *path* contains an empty segment (leading/trailing/consecutive ``'.'``),
            or the resolved module has no attribute matching the remaining segments.
        ModuleNotFoundError: If an existing module prefix fails because a
            dependency is missing (not merely because a longer prefix is not a
            module).
    """
    if "." not in path:
        msg = f"dotted path must contain at least one '.': {path!r}"
        raise ValueError(msg)

    segments = path.split(".")
    # Reject leading/trailing/consecutive dots up front — an empty segment
    # here would produce a relative-import module name (e.g. ".foo") that
    # makes importlib raise TypeError instead of ModuleNotFoundError.
    if any(not segment for segment in segments):
        msg = f"dotted path must not contain empty segments: {path!r}"
        raise ValueError(msg)

    for split_index in range(len(segments), 0, -1):
        module_name = ".".join(segments[:split_index])
        try:
            obj: object = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            # Continue only when this prefix itself is missing — not when a real
            # module fails because an inner dependency is missing.
            missing = exc.name
            if missing is not None and (module_name == missing or module_name.startswith(f"{missing}.")):
                continue
            raise
        for attr in segments[split_index:]:
            try:
                obj = getattr(obj, attr)
            except AttributeError as exc:
                msg = f"could not resolve attribute {attr!r} in {path!r}: {exc}"
                raise ValueError(msg) from exc
        return obj

    msg = f"could not import any module prefix of {path!r}"
    raise ValueError(msg)
