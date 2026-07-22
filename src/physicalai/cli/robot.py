# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Serve and discover shared robots over Zenoh."""

from __future__ import annotations

import json
import math
import signal
import sys
import threading
import time
from typing import TYPE_CHECKING

from jsonargparse import ActionConfigFile, ArgumentParser
from loguru import logger

from physicalai.cli._spec import SubcommandSpec  # noqa: PLC2701
from physicalai.robot.errors import RobotTransportError
from physicalai.robot.transport import (
    DEFAULT_RATE_HZ,
    OwnerEvent,
    RobotOwnerConfig,
    discover_robots,
    run_owner,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from jsonargparse import Namespace

HELP = "Serve or discover shared robots over Zenoh."
_SERVE_HELP = "Serve one robot in the foreground as a persistent shared-robot owner."
_DISCOVER_HELP = "Enumerate reachable shared robots."
_ALLOW_REMOTE_WARNING = (
    "WARNING: The action endpoint is unauthenticated and reachable from all "
    "network interfaces on the derived port. Use only on an isolated robot-cell "
    "network or with Zenoh ACL/TLS.\n"
)
_HELP_TEMPLATE = """usage: {prog} {{serve,discover}} ...

{description}

subcommands:
  serve      {serve_help}
  discover   {discover_help}

Run '{prog} serve --help' or '{prog} discover --help' for subcommand options.
"""


def _build_serve_parser() -> ArgumentParser:
    parser = ArgumentParser(description=_SERVE_HELP)
    parser.add_argument("--config", action=ActionConfigFile, help="YAML/JSON config file.")
    parser.add_argument("--name", type=str, required=True, help="Robot logical name.")
    parser.add_argument("--robot_class", type=str, required=True, help="Trusted dotted path to the driver class.")
    parser.add_argument(
        "--robot_kwargs",
        type=dict,
        default=None,
        help="JSON-serializable driver constructor arguments.",
    )
    parser.add_argument(
        "--allow_remote",
        action="store_true",
        default=False,
        help=(
            "Expose the unauthenticated action endpoint beyond localhost. "
            "Use only on an isolated robot-cell network or with Zenoh ACL/TLS."
        ),
    )
    parser.add_argument("--rate_hz", type=float, default=DEFAULT_RATE_HZ, help="Owner loop rate in Hz.")
    parser.add_argument("--verbose", action="store_true", default=False, help="Show startup and cleanup details.")
    return parser


def _build_discover_parser() -> ArgumentParser:
    parser = ArgumentParser(description=_DISCOVER_HELP)
    parser.add_argument(
        "--allow_remote",
        action="store_true",
        default=False,
        help="Discover owners beyond localhost.",
    )
    parser.add_argument("--timeout", type=float, default=2.0, help="Discovery timeout in seconds.")
    parser.add_argument("--json", action="store_true", default=False, help="Write one JSON array to stdout.")
    return parser


def build_parser() -> ArgumentParser:
    """Build the nested robot command parser.

    Returns:
        Parser for ``physicalai robot``.
    """
    parser = ArgumentParser(prog="physicalai robot", description=HELP)
    subcommands = parser.add_subcommands(required=True)
    subcommands.add_subcommand("serve", _build_serve_parser(), help=_SERVE_HELP)
    subcommands.add_subcommand("discover", _build_discover_parser(), help=_DISCOVER_HELP)
    return parser


def print_help(prog: str) -> None:
    """Print group help without building nested parsers."""
    print(  # noqa: T201
        _HELP_TEMPLATE.format(prog=prog, description=HELP, serve_help=_SERVE_HELP, discover_help=_DISCOVER_HELP),
    )


def _configure_serve_logging(*, verbose: bool) -> None:
    """Configure concise process-wide Loguru output for foreground serving."""
    logger.remove()
    logger.add(
        sys.stderr,
        level="TRACE" if verbose else "INFO",
        format="<green>{time:HH:mm:ss}</green>  <level>{level: <8}</level> {message}",
        colorize=sys.stderr.isatty(),
    )


def _prepare_serve_logging(*, allow_remote: bool, verbose: bool) -> None:
    """Emit the remote-exposure banner (if needed), then configure Loguru."""
    if allow_remote:
        sys.stderr.write(_ALLOW_REMOTE_WARNING)
    _configure_serve_logging(verbose=verbose)


def _log_serve_start(config: RobotOwnerConfig) -> None:
    """Log the audited serve start line, including local vs remote mode."""
    mode_tag = "remote" if config.allow_remote else "local-only"
    logger.info(f"Starting robot {config.name!r} using {config.robot_class} [{mode_tag}]")


def _format_duration(seconds: float) -> str:
    """Format elapsed seconds as ``HH:MM:SS``.

    Returns:
        The formatted duration.
    """
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _format_table(headers: tuple[str, ...], rows: Sequence[tuple[str, ...]], *, right_align: set[int]) -> str:
    """Render a compact dependency-free ASCII table.

    Returns:
        The rendered table.
    """
    widths = [max(len(header), *(len(row[index]) for row in rows)) for index, header in enumerate(headers)]

    def _row(values: tuple[str, ...]) -> str:
        cells = [
            value.rjust(widths[index]) if index in right_align else value.ljust(widths[index])
            for index, value in enumerate(values)
        ]
        return "  ".join(cells).rstrip()

    separator = "  ".join("-" * width for width in widths)
    return "\n".join([_row(headers), separator, *(_row(row) for row in rows)])


def serve(cfg: Namespace) -> int:
    """Serve one robot in the current foreground process.

    Returns:
        The owner runtime exit code, or 1 for an expected startup failure.
    """
    _prepare_serve_logging(allow_remote=cfg.allow_remote, verbose=cfg.verbose)
    try:
        config = RobotOwnerConfig(
            name=cfg.name,
            robot_class=cfg.robot_class,
            robot_kwargs=dict(cfg.robot_kwargs or {}),
            allow_remote=cfg.allow_remote,
            rate_hz=cfg.rate_hz,
            idle_timeout=None,
        )
    except (TypeError, ValueError) as exc:
        logger.error(f"Invalid robot configuration: {exc}")
        return 1

    shutdown = threading.Event()
    ready_at: float | None = None
    subscribers_present = False
    received_signum: int | None = None

    def _handle_signal(signum: int, _frame: object) -> None:
        nonlocal received_signum
        received_signum = signum
        shutdown.set()

    def _ready() -> None:
        nonlocal ready_at
        ready_at = time.monotonic()
        logger.success(f"Serving robot {config.name!r} · {config.rate_hz:g} Hz")

    def _on_event(event: OwnerEvent) -> None:
        nonlocal subscribers_present
        if event is OwnerEvent.SUBSCRIBERS_PRESENT:
            subscribers_present = True
            logger.info("Subscriber(s) connected")
        elif event is OwnerEvent.NO_SUBSCRIBERS:
            subscribers_present = False
            logger.info("No subscribers remain")
        elif event is OwnerEvent.HEARTBEAT:
            uptime_s = 0.0 if ready_at is None else time.monotonic() - ready_at
            status = "subscriber(s) connected" if subscribers_present else "no subscribers"
            logger.info(f"Healthy · uptime {_format_duration(uptime_s)} · {status}")

    _log_serve_start(config)
    previous_sigterm = signal.signal(signal.SIGTERM, _handle_signal)
    previous_sigint = signal.signal(signal.SIGINT, _handle_signal)
    try:
        try:
            result = run_owner(config, shutdown, ready=_ready, on_event=_on_event)
        except RobotTransportError as exc:
            phase = f" during {exc.phase}" if exc.phase else ""
            logger.error(f"Failed to start robot owner{phase}: {exc}")
            return 1
        else:
            if received_signum is not None:
                logger.info(f"Shutdown requested by {signal.Signals(received_signum).name}")
            if result.exit_code == 0:
                logger.success(f"Robot {config.name!r} disconnected safely")
            else:
                logger.error(f"Robot {config.name!r} stopped with reason {result.reason.value}")
            return result.exit_code
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
        signal.signal(signal.SIGINT, previous_sigint)


def discover(cfg: Namespace) -> int:
    """Discover robots without importing advertised driver classes.

    Returns:
        Zero on successful discovery, including an empty result; otherwise 1.
    """
    if isinstance(cfg.timeout, bool) or not math.isfinite(cfg.timeout) or cfg.timeout <= 0:
        sys.stderr.write(f"timeout must be finite and greater than zero, got {cfg.timeout!r}\n")
        return 1

    started_at = time.monotonic()
    robots = sorted(
        discover_robots(timeout=cfg.timeout, allow_remote=cfg.allow_remote),
        key=lambda robot: (str(robot.get("name", "")), str(robot.get("host", ""))),
    )
    if cfg.json:
        sys.stdout.write(json.dumps(robots) + "\n")
        return 0
    if not robots:
        print(f"No robots discovered in {time.monotonic() - started_at:.1f}s.")  # noqa: T201
        return 0
    headers = ("NAME", "ROBOT CLASS", "HOST", "JOINTS")
    rows = [
        (
            str(robot.get("name", "?")),
            str(robot.get("robot_class", "?")),
            str(robot.get("host", "?")),
            str(robot.get("num_joints", "?")),
        )
        for robot in robots
    ]
    print(_format_table(headers, rows, right_align={3}))  # noqa: T201
    noun = "robot" if len(robots) == 1 else "robots"
    print(f"\n{len(robots)} {noun} found in {time.monotonic() - started_at:.1f}s")  # noqa: T201
    return 0


def _dispatch(parser: ArgumentParser, cfg: Namespace) -> int:  # noqa: ARG001
    if cfg.subcommand == "serve":
        return serve(cfg.serve)
    if cfg.subcommand == "discover":
        return discover(cfg.discover)
    msg = f"unknown robot subcommand: {cfg.subcommand!r}"
    raise AssertionError(msg)


def register() -> SubcommandSpec:
    """Register the robot command group with the CLI host.

    Returns:
        The robot subcommand specification.
    """
    return SubcommandSpec(name="robot", parser=build_parser(), dispatch=_dispatch, help=HELP)
