# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Top-level ``physicalai`` CLI host.

The host uses a two-pass design so ``physicalai --help`` lists subcommands
without importing their (potentially heavy) parser builders:

1. Pass 1 builds a parser with ``--help``-only entries for every discovered
   subcommand. Help text comes from each subcommand module's ``HELP`` constant.
   Reading it imports the (light) subcommand *module* but never builds its
   parser, so the listing stays free of torch / Lightning.
2. Pass 2 calls the selected subcommand's ``register()`` to obtain its real
   parser, re-parses ``argv`` against it, then dispatches.

Built-in and third-party subcommands share a single ``register() ->
SubcommandSpec`` contract; the only difference is where each is discovered.
Subcommand modules SHOULD expose a module-level ``HELP: str`` so their
description appears in the top-level listing without building a parser; the
distribution ``Summary`` is used as a fallback.
"""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING, cast

from jsonargparse import ArgumentParser

from physicalai.cli import run as run_cmd
from physicalai.cli._discovery import discover_subcommands  # noqa: PLC2701
from physicalai.cli._spec import SubcommandSpec  # noqa: PLC2701

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from importlib.metadata import EntryPoint

logger = logging.getLogger(__name__)

# Built-in subcommands shipped by the runtime distribution. Each maps a name to
# its ``register()`` callable; help text is kept alongside so the top-level
# ``--help`` listing never has to build (or import) a parser.
_BUILTINS: dict[str, Callable[[], SubcommandSpec]] = {
    "run": run_cmd.register,
}
_BUILTIN_HELP: dict[str, str] = {
    "run": run_cmd.HELP,
}


def _resolve_register(
    name: str,
    entry_points: dict[str, EntryPoint],
) -> Callable[[], SubcommandSpec]:
    """Return the ``register()`` callable for ``name`` (built-in or entry point).

    Returns:
        The zero-argument callable that produces the subcommand's spec.
    """
    if name in _BUILTINS:
        return _BUILTINS[name]
    return entry_points[name].load()


def _load_spec(name: str, entry_points: dict[str, EntryPoint]) -> SubcommandSpec:
    """Resolve ``name`` to a fully built :class:`SubcommandSpec` (eager import).

    Both built-in and third-party subcommands go through the same validation,
    so a misbehaving registration fails identically regardless of origin.

    Returns:
        The spec produced by the subcommand's ``register()`` callable.

    Raises:
        TypeError: If ``register()`` returns a non-spec value.
        ValueError: If the returned spec's ``name`` does not match ``name``.
    """
    spec = _resolve_register(name, entry_points)()
    if not isinstance(spec, SubcommandSpec):
        msg = f"Subcommand '{name}' register() returned {type(spec).__name__}, expected SubcommandSpec."
        raise TypeError(msg)
    if spec.name != name:
        msg = f"Subcommand '{name}' returned SubcommandSpec(name={spec.name!r}); name mismatch."
        raise ValueError(msg)
    return spec


def _ep_help(ep: EntryPoint) -> str:
    """Per-subcommand help for an entry point, falling back to distribution metadata.

    Imports the subcommand *module* (cheap by contract — heavy deps are imported
    lazily inside the parser builder) to read its module-level ``HELP`` constant.
    If the module is missing ``HELP`` or fails to import, falls back to the
    providing distribution's ``Summary``.

    Returns:
        A short description suitable for the top-level ``--help`` listing.
    """
    try:
        register = ep.load()
        module = sys.modules.get(getattr(register, "__module__", ""))
        help_text = getattr(module, "HELP", "")
    except Exception:
        logger.debug("Failed to load entry point '%s' for help text", ep.name, exc_info=True)
        help_text = ""
    return help_text or _dist_help(ep)


def _dist_help(ep: EntryPoint) -> str:
    """Fallback help text from an entry point's distribution metadata.

    Returns:
        ``(from <dist>) <summary>`` when available, else ``(from <dist>)``.
    """
    if ep.dist is None:
        return ""
    summary = ep.dist.metadata.get("Summary")
    return f"(from {ep.dist.name}) {summary}" if summary else f"(from {ep.dist.name})"


def _subcommand_help(entry_points: dict[str, EntryPoint]) -> dict[str, str]:
    """Map every available subcommand name to its ``--help`` description.

    Returns:
        ``{name: help}`` covering built-ins (static help) and third-party
        entry points (module ``HELP`` constant, or distribution metadata).
        Subcommand parsers are never built.
    """
    helps = dict(_BUILTIN_HELP)
    for name, ep in entry_points.items():
        helps[name] = _ep_help(ep)
    return helps


def _build_host_parser(helps: dict[str, str]) -> ArgumentParser:
    """Build a top-level parser exposing only subcommand *names* (no builders).

    Args:
        helps: ``{name: help}`` for every available subcommand.

    Returns:
        Parser whose ``--help`` lists every available subcommand without
        importing their parser builders.
    """
    parser = ArgumentParser(prog="physicalai", description="PhysicalAI runtime CLI.")
    subcommands = parser.add_subcommands(required=True)
    for name, help_text in helps.items():
        subcommands.add_subcommand(
            name,
            ArgumentParser(prog=f"physicalai {name}", description=help_text),
            help=help_text,
        )
    return parser


def _select_subcommand(argv: Sequence[str], known: set[str]) -> str | None:
    """Peek at ``argv`` to find the selected subcommand without parsing args.

    Returns:
        The subcommand name if the first positional token is known, else ``None``.
    """
    for token in argv:
        if token.startswith("-"):
            continue
        return token if token in known else None
    return None


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and dispatch to the selected subcommand.

    Args:
        argv: Command-line arguments. Defaults to :data:`sys.argv` when ``None``.

    Returns:
        Process exit code returned by the dispatched subcommand.

    Raises:
        AssertionError: If host parsing returns without selecting a subcommand.
    """
    argv_list = list(sys.argv[1:] if argv is None else argv)
    entry_points = discover_subcommands(frozenset(_BUILTINS))
    known = set(_BUILTINS) | set(entry_points)
    selected = _select_subcommand(argv_list, known)

    if selected is None:
        host = _build_host_parser(_subcommand_help(entry_points))
        host.parse_args(argv_list)
        msg = "Host parser returned without selecting a subcommand."
        raise AssertionError(msg)
    selected_name = cast("str", selected)

    spec = _load_spec(selected_name, entry_points)
    sub_argv = argv_list[argv_list.index(selected_name) + 1 :]
    cfg = spec.parser.parse_args(sub_argv)
    return spec.dispatch(spec.parser, cfg)


if __name__ == "__main__":
    sys.exit(main())
