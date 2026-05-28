# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Subcommand registration contract for the ``physicalai`` CLI host."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from jsonargparse import ArgumentParser, Namespace


class Dispatch(Protocol):
    """Callable that executes a subcommand given its parser and parsed config."""

    def __call__(self, parser: ArgumentParser, cfg: Namespace) -> int:
        """Run the subcommand.

        Args:
            parser: The subcommand's own parser, used to call ``parser.instantiate``.
            cfg: Parsed configuration namespace for this subcommand.

        Returns:
            Process exit code (``0`` on success).
        """


@dataclass(frozen=True)
class SubcommandSpec:
    """Registration record returned by a subcommand's ``register()`` callable.

    Attributes:
        name: Subcommand name surfaced as ``physicalai <name>``.
        parser: Fully built :class:`jsonargparse.ArgumentParser` for this subcommand.
        dispatch: Callable invoked as ``dispatch(parser, cfg) -> int`` after parsing.
        help: Short description shown in the top-level ``--help`` listing.
    """

    name: str
    parser: ArgumentParser
    dispatch: Dispatch
    help: str = ""
