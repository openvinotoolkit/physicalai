# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Hardware health checks and replug monitoring."""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

from jsonargparse import ArgumentParser, Namespace

from physicalai.cli._spec import SubcommandSpec  # noqa: PLC2701

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from physicalai.diagnostics.models import Inventory, Report, WatchEvent
    from physicalai.diagnostics.platforms.base import Collector

HELP = "Diagnose camera/robot connections; watch reconnects without moving hardware."


def build_parser() -> ArgumentParser:
    """Build diagnostics options without importing collectors or SDKs.

    Returns:
        A parser with explicit passive/active operation choices.
    """
    parser = ArgumentParser(prog="pai doctor", description=HELP)
    parser.add_argument(
        "--json", action="store_true", help="JSON report; with --watch, one JSON event per line (NDJSON)."
    )
    parser.add_argument("--strict", action="store_true", help="Return exit code 1 for warnings as well as failures.")
    parser.add_argument(
        "--redact", action="store_true", help="Pseudonymize known serials, hostname, and home path for support reports."
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Monitor reconnects and fresh logs; native notifications with polling fallback.",
    )
    parser.add_argument("--duration", type=float, default=None, help="Stop watching/testing after this many seconds.")
    parser.add_argument("--interval", type=float, default=1.0, help="Watch polling interval in seconds (default: 1).")
    parser.add_argument(
        "--config", type=Path, help="Inspect YAML/JSON selectors without constructing configured objects."
    )
    parser.add_argument(
        "--stream", action="store_true", help="Explicitly start selected camera streams (never robots)."
    )
    parser.add_argument(
        "--camera", action="append", default=[], help="Selected capture endpoint for --stream; repeat for concurrency."
    )
    return parser


def _validate_options(parser: ArgumentParser, cfg: Namespace) -> None:
    if not math.isfinite(cfg.interval) or cfg.interval <= 0:
        parser.error("--interval must be finite and positive.")
    if cfg.duration is not None and (not math.isfinite(cfg.duration) or cfg.duration <= 0):
        parser.error("--duration must be finite and positive.")
    if cfg.watch and cfg.stream:
        parser.error("Choose --watch or --stream, not both.")
    if cfg.camera and not cfg.stream:
        parser.error("--camera requires --stream; passive discovery never starts cameras.")
    if cfg.duration is not None and not (cfg.watch or cfg.stream):
        parser.error("--duration requires --watch or --stream.")
    if cfg.stream and (not cfg.camera or cfg.duration is None):
        parser.error("--stream requires at least one --camera and an explicit --duration.")


def _watch(collector: Collector, cfg: Namespace, assessment: Callable[[Inventory], Report]) -> int:
    from physicalai.diagnostics.platforms.events import change_source  # noqa: PLC0415
    from physicalai.diagnostics.watch import watch  # noqa: PLC0415

    with change_source() as source:
        if not cfg.json:
            print(f"Monitoring: {source.detail}", flush=True)  # noqa: T201
        events = watch(
            collector,
            interval=cfg.interval,
            duration=cfg.duration,
            strict=cfg.strict,
            assessment=assessment,
            source=source,
        )
        return _print_watch(events, cfg)


def _print_watch(events: Iterable[WatchEvent], cfg: Namespace) -> int:
    from physicalai.diagnostics.models import Report  # noqa: PLC0415
    from physicalai.diagnostics.redact import Redactor  # noqa: PLC0415
    from physicalai.diagnostics.render import render_event, render_report  # noqa: PLC0415

    redactor = Redactor() if cfg.redact else None
    exit_code = 0
    for raw_event in events:
        event = redactor.model(raw_event) if redactor is not None else raw_event
        if cfg.json:
            print(event.model_dump_json(), flush=True)  # noqa: T201
        elif event.kind == "snapshot":
            print(render_report(Report.model_validate(event.data)), flush=True)  # noqa: T201
            print(render_event(event), flush=True)  # noqa: T201
        else:
            print(render_event(event), flush=True)  # noqa: T201
        if event.kind == "final":
            value = event.data["exit_code"]
            exit_code = value if isinstance(value, int) else 2
    return exit_code


def run(parser: ArgumentParser, cfg: Namespace) -> int:
    """Run passive assessment, watch, or explicitly selected streaming tests.

    Returns:
        Health status, partial visibility, cancellation, or usage error code.
    """
    from physicalai.diagnostics.checks import assess  # noqa: PLC0415
    from physicalai.diagnostics.config_check import inspect_config  # noqa: PLC0415
    from physicalai.diagnostics.platforms import create_collector  # noqa: PLC0415
    from physicalai.diagnostics.render import render_report  # noqa: PLC0415

    _validate_options(parser, cfg)

    def assessment(inventory: Inventory) -> Report:
        report = assess(inventory)
        if cfg.config:
            report.findings.extend(inspect_config(cfg.config, inventory))
        return report

    try:
        collector = create_collector()
        if cfg.watch:
            return _watch(collector, cfg, assessment)
        report = assessment(collector.collect())
        from physicalai.diagnostics.redact import Redactor  # noqa: PLC0415

        redactor = Redactor() if cfg.redact else None
        if redactor:
            redactor.register(report)
        if cfg.stream:
            from physicalai.diagnostics.streaming import test_streams  # noqa: PLC0415

            stream_findings = test_streams(report.inventory, cfg.camera, duration=cfg.duration)
            report = assessment(collector.collect())
            report.mode = "camera_streams"
            report.findings.extend(stream_findings)
        if redactor:
            report = redactor.model(report)
        print(report.model_dump_json(indent=2) if cfg.json else render_report(report))  # noqa: T201
        return report.exit_code(strict=cfg.strict)
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    return 2


def register() -> SubcommandSpec:
    """Register the doctor command.

    Returns:
        The host CLI registration.
    """
    return SubcommandSpec(name="doctor", parser=build_parser(), dispatch=run, help=HELP)
