# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Dotted-path imports for trusted component configuration."""

from __future__ import annotations

import importlib


def import_dotted_path(path: str) -> object:
    """Resolve a fully-qualified path, including nested attributes.

    Args:
        path: Dotted path such as ``"pkg.mod.Outer.Inner"``.

    Returns:
        The resolved object.

    Raises:
        ValueError: If no module prefix can be imported, or *path* has no ``.``.
        ModuleNotFoundError: If an existing module prefix fails because a
            dependency is missing (not merely because a longer prefix is not a
            module).
    """
    if "." not in path:
        msg = f"dotted path must contain at least one '.': {path!r}"
        raise ValueError(msg)

    segments = path.split(".")
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
            obj = getattr(obj, attr)
        return obj

    msg = f"could not import any module prefix of {path!r}"
    raise ValueError(msg)
