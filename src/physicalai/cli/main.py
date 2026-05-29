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
import pathlib
import sys
from typing import TYPE_CHECKING

from jsonargparse import ArgumentParser

from physicalai.cli import run as run_cmd
from physicalai.cli._discovery import discover_subcommands  # noqa: PLC2701
from physicalai.cli._spec import SubcommandSpec  # noqa: PLC2701

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from importlib.metadata import EntryPoint
    from types import ModuleType

logger = logging.getLogger(__name__)

# Built-in subcommands shipped by the runtime distribution. Each maps a name to
# its ``register()`` callable; help text is kept alongside so the top-level
# ``--help`` listing never has to build (or import) a parser.
_BUILTINS: dict[str, Callable[[], SubcommandSpec]] = {
    "run": run_cmd.register,
}
_BUILTIN_HELP: dict[str, str] = {
    "completion": "Print a shell completion script.",
    "run": run_cmd.HELP,
}
_COMPLETION_SHELLS = frozenset({"bash", "zsh", "fish"})
_HELP_FLAGS = frozenset({"-h", "--help"})


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


def _load_subcommand_module(name: str, entry_points: dict[str, EntryPoint]) -> ModuleType | None:
    """Load the selected subcommand module without building its parser.

    Returns:
        Imported subcommand module when available, otherwise ``None``.
    """
    if name == "run":
        return run_cmd

    try:
        register = entry_points[name].load()
    except Exception:
        logger.debug("Failed to load entry point '%s' for fast help", name, exc_info=True)
        return None
    return sys.modules.get(getattr(register, "__module__", ""))


def _print_fast_help(name: str, entry_points: dict[str, EntryPoint], prog: str) -> bool:
    """Print lightweight subcommand help when the module provides it.

    Returns:
        ``True`` if help was printed, otherwise ``False`` so the caller can fall
        back to the full parser.
    """
    module = _load_subcommand_module(name, entry_points)
    print_help = getattr(module, "print_help", None)
    if not callable(print_help):
        return False
    print_help(f"{prog} {name}")
    return True


def _is_help_request(argv: Sequence[str]) -> bool:
    """Return whether ``argv`` asks for subcommand help."""
    return any(token in _HELP_FLAGS for token in argv)


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


def _build_host_parser(helps: dict[str, str], prog: str) -> ArgumentParser:
    """Build a top-level parser exposing only subcommand *names* (no builders).

    Args:
        helps: ``{name: help}`` for every available subcommand.
        prog: Program name to display in help and completion output.

    Returns:
        Parser whose ``--help`` lists every available subcommand without
        importing their parser builders.
    """
    parser = ArgumentParser(prog=prog, description="PhysicalAI runtime CLI.")
    subcommands = parser.add_subcommands(required=True)
    for name, help_text in helps.items():
        subcommands.add_subcommand(
            name,
            ArgumentParser(prog=f"{prog} {name}", description=help_text),
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


def _print_completion(argv: Sequence[str], entry_points: dict[str, EntryPoint], prog: str) -> int:
    """Print a shell completion script for the top-level CLI.

    Args:
        argv: Arguments after the ``completion`` subcommand.
        entry_points: Discovered plugin subcommands to include in the script.
        prog: Program name to register in the generated completion script.

    Returns:
        Process exit code.
    """
    parser = ArgumentParser(prog=f"{prog} completion", description=_BUILTIN_HELP["completion"])
    parser.add_argument("shell", choices=sorted(_COMPLETION_SHELLS), help="Shell to generate completion for.")
    if not argv:
        parser.print_help()
        return 0
    cfg = parser.parse_args(argv)
    host = _build_host_parser(_subcommand_help(entry_points), prog)
    sys.stdout.write(_completion_script(host, cfg.shell, prog))
    return 0


def _completion_script(parser: ArgumentParser, shell: str, prog: str) -> str:
    """Return a shell completion script suitable for interactive sourcing.

    Args:
        parser: Host parser to generate completion for.
        shell: Shell name (`bash`, `zsh`, or `fish`).
        prog: Program name being registered.

    Returns:
        Completion script text.
    """
    script = parser.get_completion_script(f"shtab-{shell}")
    if shell != "zsh":
        return script

    register_block = f"\n\ntypeset -A opt_args\n\ncompdef _shtab_{prog} -N {prog}\n"
    marker = "\n\ntypeset -A opt_args\n\nif [[ $zsh_eval_context[-1] == eval ]]; then\n"
    if marker not in script:
        return script
    prefix = script.split(marker, maxsplit=1)[0]
    return prefix + register_block


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
    prog = pathlib.Path(sys.argv[0]).name or "physicalai"
    entry_points = discover_subcommands(frozenset(_BUILTIN_HELP))
    known = set(_BUILTIN_HELP) | set(entry_points)
    selected = _select_subcommand(argv_list, known)

    if selected is None:
        host = _build_host_parser(_subcommand_help(entry_points), prog)
        host.parse_args(argv_list)
        msg = "Host parser returned without selecting a subcommand."
        raise AssertionError(msg)
    selected_name = selected
    sub_argv = argv_list[argv_list.index(selected_name) + 1 :]

    if selected_name == "completion":
        return _print_completion(sub_argv, entry_points, prog)

    if _is_help_request(sub_argv) and _print_fast_help(selected_name, entry_points, prog):
        return 0

    spec = _load_spec(selected_name, entry_points)
    cfg = spec.parser.parse_args(sub_argv)
    return spec.dispatch(spec.parser, cfg)


if __name__ == "__main__":
    sys.exit(main())
