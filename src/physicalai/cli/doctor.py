# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Thin terminal entry point for the reusable USB diagnostics API."""

from __future__ import annotations

from typing import TYPE_CHECKING

from jsonargparse import ArgumentParser

from physicalai.cli._spec import SubcommandSpec  # noqa: PLC2701

if TYPE_CHECKING:
    from jsonargparse import Namespace

HELP = "Inspect USB camera/serial connections and shared hubs without starting hardware."


def build_parser() -> ArgumentParser:
    """Build the doctor parser without collecting hardware evidence.

    Returns:
        A parser with overview, selected-device, tree, and JSON views.
    """
    parser = ArgumentParser(prog="pai doctor", description=HELP)
    parser.add_argument("device", nargs="?", help="USB device ID, endpoint path, or unique alias to inspect.")
    parser.add_argument("--tree", action="store_true", help="Show camera/serial USB branches and upstream links.")
    parser.add_argument("--json", action="store_true", help="Print the structured API report as JSON.")
    parser.add_argument("--no-ansi", action="store_true", help="Disable terminal colors and bold text.")
    return parser


def run(parser: ArgumentParser, cfg: Namespace) -> int:
    """Call the public API and render its report.

    Returns:
        Zero for a collected report without errors (warnings are allowed), one
        for observed access/collection errors, or two for invalid arguments/OS.
    """
    import json  # noqa: PLC0415 - command-specific imports stay off the help path
    import os  # noqa: PLC0415
    import sys  # noqa: PLC0415

    from physicalai.diagnostics import diagnose  # noqa: PLC0415

    try:
        report = diagnose(cfg.device)
    except (ValueError, NotImplementedError) as exc:
        parser.error(str(exc))
        return 2
    if cfg.json:
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=True))  # noqa: T201 - CLI output
    else:
        from physicalai.diagnostics.render import render_report  # noqa: PLC0415

        ansi = (
            not cfg.no_ansi
            and sys.stdout.isatty()
            and not os.environ.get("NO_COLOR")
            and os.environ.get("TERM") != "dumb"
        )
        print(render_report(report, tree=cfg.tree, ansi=ansi))  # noqa: T201 - CLI output
    return int(report.has_errors)


def register() -> SubcommandSpec:
    """Register the built-in doctor command.

    Returns:
        The CLI host's standard subcommand specification.
    """
    return SubcommandSpec(name="doctor", parser=build_parser(), dispatch=run, help=HELP)
