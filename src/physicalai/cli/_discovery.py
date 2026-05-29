# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Entry-point discovery for ``physicalai.cli.subcommands``.

Discovery is lazy: this module returns :class:`EntryPoint` objects without
calling :meth:`EntryPoint.load`, so the top-level ``physicalai --help`` stays
torch-free.
"""

from __future__ import annotations

import logging
from importlib.metadata import entry_points
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from importlib.metadata import EntryPoint

ENTRY_POINT_GROUP = "physicalai.cli.subcommands"

logger = logging.getLogger(__name__)


def discover_subcommands(
    builtin_names: frozenset[str] = frozenset(),
) -> dict[str, EntryPoint]:
    """Return ``{name: EntryPoint}`` for the ``physicalai.cli.subcommands`` group.

    Built-in subcommand names always win over third-party entry points sharing
    the same name; a ``WARNING`` is logged on collision. Among third-party
    entries, the first one returned by :func:`importlib.metadata.entry_points`
    wins deterministically.

    Args:
        builtin_names: Subcommand names registered in-process by the runtime
            host. These are excluded from the returned mapping and used to
            detect collisions with third-party entry points.

    Returns:
        Mapping of subcommand name to the entry point that provides it.
        ``EntryPoint.load()`` is **not** called by this function.
    """
    discovered: dict[str, EntryPoint] = {}
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        if ep.name in builtin_names:
            logger.warning(
                "physicalai.cli: subcommand '%s' from %s collides with built-in; using built-in.",
                ep.name,
                ep.dist.name if ep.dist else "<unknown>",
            )
            continue
        if ep.name in discovered:
            existing = discovered[ep.name]
            logger.warning(
                "physicalai.cli: subcommand '%s' registered by both %s and %s; using %s.",
                ep.name,
                existing.dist.name if existing.dist else "<unknown>",
                ep.dist.name if ep.dist else "<unknown>",
                existing.dist.name if existing.dist else "<unknown>",
            )
            continue
        discovered[ep.name] = ep
    return discovered
