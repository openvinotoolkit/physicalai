# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""``physicalai run`` — execute a RobotRuntime (any pluggable action source)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import yaml
from jsonargparse import ActionConfigFile, ArgumentParser
from loguru import logger

from physicalai.cli._spec import SubcommandSpec  # noqa: PLC2701

if TYPE_CHECKING:
    from jsonargparse import Namespace

    from physicalai.runtime import RobotRuntime

HELP = "Run a trained policy (or any action source) on robot hardware."
_HELP_TEMPLATE = """usage: {prog} --config CONFIG [--run.duration_s SECONDS]

{description}

options:
  -h, --help                    Show this help message and exit.
  --config CONFIG               YAML/JSON runtime config file.
  --run.duration_s SECONDS      Stop after the given duration in seconds.
  --verbose                     Enable debug logging during startup and runtime.

Runtime constructor arguments are available under --runtime.* when executing
the command. Use --print_config with a complete command to inspect the full
jsonargparse schema.

One config schema — ``action_source:`` is always explicit:
  runtime:
    robot: {{...}}
    action_source:
      class_path: physicalai.runtime.PolicySource
      init_args: {{model: {{...}}, execution: {{...}}}}
    fps: 30.0
"""


def _reshape_config_value(parser: ArgumentParser, value: object) -> object:
    """Return CLI-shaped config content for a bare exported runtime document.

    A ``--config`` value that points to a document with a top-level
    ``class_path`` (as produced by ``to_config(runtime)`` / ``save_yaml``) is
    loaded and unwrapped to the ``runtime:`` mapping the parser expects,
    returned as JSON content (valid YAML that ``ActionConfigFile`` accepts
    inline). Any other value — a non-path, an unreadable file, or a
    ``runtime:`` CLI document — is returned unchanged.

    Returns:
        The original value, or unwrapped config content for a bare document.
    """
    if not isinstance(value, str):
        return value
    try:
        if not Path(value).is_file():
            return value
        document = yaml.safe_load(Path(value).read_text(encoding="utf-8"))
    except (OSError, ValueError, yaml.YAMLError):
        return value
    if not isinstance(document, dict) or "class_path" not in document:
        return value

    from physicalai.config import ComponentConfigError  # noqa: PLC0415
    from physicalai.runtime import RobotRuntime  # noqa: PLC0415
    from physicalai.runtime.core import _unwrap_runtime_document  # noqa: PLC0415, PLC2701

    try:
        return json.dumps(_unwrap_runtime_document(document, target=RobotRuntime))
    except ComponentConfigError as exc:
        parser.error(str(exc))


class _RuntimeConfigFile(ActionConfigFile):
    """``--config`` action accepting both the CLI document and a bare export.

    Reshapes a bare exported ``RobotRuntime`` ComponentConfig (a document with a
    top-level ``class_path``) into the ``runtime:`` document jsonargparse
    expects before applying it; a ``runtime:`` / ``run:`` CLI document is passed
    through unchanged. This keeps ``to_config(runtime)`` → ``save_yaml`` →
    ``physicalai run --config`` a single-shape round-trip.
    """

    def __call__(
        self,
        parser: ArgumentParser,
        cfg: Namespace,
        values: object,
        option_string: str | None = None,
    ) -> None:
        """Reshape a bare exported runtime document, then delegate to the base action."""
        super().__call__(parser, cfg, _reshape_config_value(parser, values), option_string)


def build_parser() -> ArgumentParser:
    """Build the ``run`` subcommand parser.

    One path: ``RobotRuntime`` constructor arguments (``action_source:``
    always explicit) plus ``run()`` method arguments. ``--config`` accepts
    both the CLI document (``runtime:`` / ``run:``) and a bare exported
    ComponentConfig produced by ``to_config(runtime)``.

    Returns:
        Parser for the ``physicalai run`` subcommand.
    """
    from physicalai.runtime import RobotRuntime  # noqa: PLC0415

    parser = ArgumentParser(prog="physicalai run", description=HELP)
    parser.add_argument("--config", action=_RuntimeConfigFile, help="YAML/JSON config file.")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging during startup and runtime.",
    )
    parser.add_class_arguments(RobotRuntime, "runtime")
    parser.add_method_arguments(RobotRuntime, "run", "run")
    return parser


def print_help(prog: str) -> None:
    """Print lightweight help without building the full runtime parser."""
    print(_HELP_TEMPLATE.format(prog=prog, description=HELP))  # noqa: T201


def _configure_run_logging(*, verbose: bool) -> None:
    """Configure concise process-wide Loguru output for ``physicalai run``."""
    logger.remove()
    logger.add(
        sys.stderr,
        level="TRACE" if verbose else "INFO",
        format="<green>{time:HH:mm:ss}</green>  <level>{level: <8}</level> {message}",
        colorize=sys.stderr.isatty(),
    )


def run(parser: ArgumentParser, cfg: Namespace) -> int:
    """Instantiate the runtime from ``cfg`` and invoke ``run()``.

    Args:
        parser: The ``run`` subcommand parser used to instantiate classes from ``cfg``.
        cfg: Parsed configuration namespace produced by ``parser.parse_args``.

    Returns:
        Process exit code (``0`` on success).
    """
    _configure_run_logging(verbose=bool(getattr(cfg, "verbose", False)))
    init = parser.instantiate(cfg)
    runtime: RobotRuntime = init.runtime
    run_kwargs: dict = {}
    if hasattr(cfg, "run"):
        raw = cfg.run
        run_kwargs = raw.as_dict() if hasattr(raw, "as_dict") else {"duration_s": raw.duration_s}

    with runtime:
        steps = runtime.run(**run_kwargs)

    logger.info("Run complete: {} steps", steps)
    return 0


def register() -> SubcommandSpec:
    """Return the :class:`SubcommandSpec` for ``physicalai run``.

    Returns:
        Spec wiring :func:`build_parser` and :func:`run` for the host parser.
    """
    return SubcommandSpec(name="run", parser=build_parser(), dispatch=run, help=HELP)
