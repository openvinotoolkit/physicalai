# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""``physicalai run`` — execute a :class:`~physicalai.runtime.PolicyRuntime`."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from jsonargparse import ActionConfigFile, ArgumentParser

from physicalai.cli._spec import SubcommandSpec  # noqa: PLC2701

if TYPE_CHECKING:
    from jsonargparse import Namespace

    from physicalai.runtime import PolicyRuntime

logger = logging.getLogger(__name__)

HELP = "Run a trained policy on robot hardware via PolicyRuntime."


def build_parser() -> ArgumentParser:
    """Build the ``run`` subcommand parser.

    :class:`PolicyRuntime` is imported lazily so merely importing this module
    (e.g. for the top-level ``physicalai --help`` listing) stays cheap.

    Returns:
        Parser exposing :class:`PolicyRuntime` constructor and ``run`` method args.
    """
    from physicalai.runtime import PolicyRuntime  # noqa: PLC0415

    parser = ArgumentParser(prog="physicalai run", description=HELP)
    parser.add_argument("--config", action=ActionConfigFile, help="YAML/JSON config file.")
    parser.add_class_arguments(PolicyRuntime, "runtime")
    parser.add_method_arguments(PolicyRuntime, "run", "run")
    parser.add_argument(
        "--duration_s",
        type=float,
        default=None,
        help="Compatibility alias for --run.duration_s.",
    )
    return parser


def run(parser: ArgumentParser, cfg: Namespace) -> int:
    """Instantiate :class:`PolicyRuntime` from ``cfg`` and invoke ``run()``.

    Args:
        parser: The ``run`` subcommand parser used to instantiate classes from ``cfg``.
        cfg: Parsed configuration namespace produced by ``parser.parse_args``.

    Returns:
        Process exit code (``0`` on success).
    """
    init = parser.instantiate(cfg)
    runtime: PolicyRuntime = init.runtime
    run_kwargs = cfg.run.as_dict() if hasattr(cfg, "run") else {}
    if getattr(cfg, "duration_s", None) is not None and run_kwargs.get("duration_s") is None:
        run_kwargs["duration_s"] = cfg.duration_s

    with runtime:
        stats = runtime.run(**run_kwargs)

    logger.info(
        "Run complete: %d steps, %d pops, %d holds, %d inferences",
        stats.steps,
        stats.total_pops,
        stats.total_holds,
        stats.inference_count,
    )
    return 0


def register() -> SubcommandSpec:
    """Return the :class:`SubcommandSpec` for ``physicalai run``.

    Returns:
        Spec wiring :func:`build_parser` and :func:`run` for the host parser.
    """
    return SubcommandSpec(name="run", parser=build_parser(), dispatch=run, help=HELP)
