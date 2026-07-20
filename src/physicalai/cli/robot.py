# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Serve and discover shared robots over Zenoh."""

from __future__ import annotations

import json
import math
import signal
import sys
import threading
from typing import TYPE_CHECKING

from jsonargparse import ActionConfigFile, ArgumentParser

from physicalai.cli._spec import SubcommandSpec  # noqa: PLC2701
from physicalai.robot.errors import RobotTransportError
from physicalai.robot.transport import discover_robots
from physicalai.robot.transport._owner_config import DEFAULT_RATE_HZ, RobotOwnerConfig  # noqa: PLC2701
from physicalai.robot.transport._owner_worker import run_owner  # noqa: PLC2701

if TYPE_CHECKING:
    from jsonargparse import Namespace

HELP = "Serve or discover shared robots over Zenoh."
_SERVE_HELP = "Serve one robot in the foreground as a persistent shared-robot owner."
_DISCOVER_HELP = "Enumerate reachable shared robots."
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
        default={},
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


def serve(cfg: Namespace) -> int:
    """Serve one robot in the current foreground process.

    Returns:
        The owner runtime exit code, or 1 for an expected startup failure.
    """
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
        sys.stderr.write(f"Invalid robot configuration: {exc}\n")
        return 1

    shutdown = threading.Event()

    def _handle_signal(_signum: int, _frame: object) -> None:
        shutdown.set()

    def _ready() -> None:
        scope = "remote" if config.allow_remote else "local-only"
        sys.stderr.write(f"Serving robot {config.name!r} ({scope}).\n")
        sys.stderr.flush()

    previous_sigterm = signal.signal(signal.SIGTERM, _handle_signal)
    previous_sigint = signal.signal(signal.SIGINT, _handle_signal)
    try:
        try:
            return run_owner(config, shutdown, ready=_ready).exit_code
        except RobotTransportError as exc:
            phase = f" during {exc.phase}" if exc.phase else ""
            sys.stderr.write(f"Failed to start robot owner{phase}: {exc}\n")
            return 1
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

    robots = sorted(
        discover_robots(timeout=cfg.timeout, allow_remote=cfg.allow_remote),
        key=lambda robot: (str(robot.get("name", "")), str(robot.get("host", ""))),
    )
    if cfg.json:
        sys.stdout.write(json.dumps(robots) + "\n")
        return 0
    if not robots:
        print("No robots discovered.")  # noqa: T201
        return 0
    for robot in robots:
        print(  # noqa: T201
            f"{robot.get('name', '?')}\t{robot.get('robot_class', '?')}\t"
            f"host={robot.get('host', '?')}\tjoints={robot.get('num_joints', '?')}",
        )
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
