# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Interactive hardware selection and passive Runtime configuration export."""

from __future__ import annotations

import asyncio
import shlex
import sys
from pathlib import Path

from jsonargparse import ArgumentParser, Namespace

from physicalai.cli._spec import SubcommandSpec  # noqa: PLC2701

HELP = "Interactively select robots/cameras and write a Runtime YAML/JSON configuration."


def _run_hint(path: Path) -> str:
    if any(not char.isprintable() for char in str(path)):
        return (
            "After safety review, pass the exact saved path to pai run. "
            "No copyable command is shown for paths containing control characters."
        )
    if sys.platform == "win32":
        quoted = "'" + str(path).replace("'", "''") + "'"
        return f"After safety review, run in PowerShell: pai run --config {quoted}"
    return "After safety review, run: " + shlex.join(["pai", "run", "--config", str(path)])


def build_parser() -> ArgumentParser:
    """Build discovery options without loading plugins or starting hardware.

    Returns:
        The interactive setup parser.
    """
    parser = ArgumentParser(prog="pai discover", description=HELP)
    parser.add_argument("--output", type=Path, help="User-selected YAML/JSON destination; prompted when omitted.")
    parser.add_argument("--format", choices=["yaml", "json"], default=None, help="Infer from extension when omitted.")
    parser.add_argument(
        "--components-only", action="store_true", help="Export named recipes, not a complete pai run config."
    )
    return parser


def run(parser: ArgumentParser, cfg: Namespace) -> int:
    """Run the wizard; never call Runtime, a driver builder, or visual identify.

    Returns:
        Zero on export/declined write, or the cancellation/error status.
    """
    from physicalai.diagnostics import collect_inventory  # noqa: PLC0415
    from physicalai.diagnostics.discovery import DiscoverySession, Prompts  # noqa: PLC0415
    from physicalai.diagnostics.export import encode_document, output_format, write_document  # noqa: PLC0415
    from physicalai.diagnostics.plugins import discover_candidates, load_catalog  # noqa: PLC0415
    from physicalai.diagnostics.render import render_inventory, safe_text  # noqa: PLC0415

    if not sys.stdin.isatty():
        parser.error("Discovery requires an interactive terminal. Use 'pai devices --json' for scripted inventory.")
    prompts = Prompts()
    try:
        path = (cfg.output or Path(prompts.ask("Output configuration path"))).expanduser().absolute()
        file_format = output_format(path, cfg.format)
        inventory = collect_inventory()
        catalog = load_catalog()
        candidates = asyncio.run(discover_candidates(catalog, inventory))
        prompts.write(render_inventory(inventory))
        for error in catalog.errors:
            prompts.write(f"Plugin limitation: {safe_text(error.source)}: {safe_text(error.message)}")
        if candidates:
            prompts.write("Passive probe candidates (not verified robot models): " + safe_text(candidates))
        prompts.write("Setup exports data only. It will not connect robots, preview cameras, or run a policy.")
        session = DiscoverySession(inventory, catalog, prompts, refresh=collect_inventory)
        document = session.build(components_only=cfg.components_only)
        text = encode_document(document, format=file_format)
        prompts.write("\nConfiguration for review:\n" + text)
        prompts.write("Structure only: hardware operation, policy compatibility, and calibration have not been tested.")
        if not prompts.confirm("Write this configuration?"):
            prompts.write("Cancelled; no file written.")
            return 0
        if not path.parent.exists():
            if not prompts.confirm(f"Create destination directory {path.parent}?"):
                return 0
            path.parent.mkdir(parents=True, exist_ok=True)
        overwrite = path.exists() or path.is_symlink()
        if overwrite and not prompts.confirm(f"Replace existing file {path}?"):
            return 0
        refreshed = collect_inventory()
        for selection in session.selections:
            selection.validate(refreshed)
        write_document(path, text, overwrite=overwrite)
        prompts.write(f"Saved {safe_text(path)}")
        prompts.write(
            "Component bundle: load named recipes with physicalai.config.Config; not a pai run file."
            if cfg.components_only
            else safe_text(_run_hint(path))
        )
    except (KeyboardInterrupt, EOFError):
        prompts.write("\nCancelled. Check the destination if cancellation occurred during the final save.")
        return 130
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    else:
        return 0
    return 2


def register() -> SubcommandSpec:
    """Register interactive discovery.

    Returns:
        The host CLI registration.
    """
    return SubcommandSpec(name="discover", parser=build_parser(), dispatch=run, help=HELP)
