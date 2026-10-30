# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Lightweight host CLI registration."""

from __future__ import annotations

import signal
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from jsonargparse import ArgumentParser

from physicalai.cli._spec import SubcommandSpec  # noqa: PLC2701

if TYPE_CHECKING:
    from types import FrameType

    from jsonargparse import Namespace

HELP = "Convert ROS 2 recordings to LeRobot v3 datasets."


def _interrupt(signum: int, _frame: FrameType | None) -> None:
    msg = f"Interrupted by signal {signum}"
    raise KeyboardInterrupt(msg)


def _dispatch(parser: ArgumentParser, cfg: Namespace) -> int:  # noqa: ARG001 - host dispatch protocol
    from physicalai_rosbag_plugin.config import MappingConfig  # noqa: PLC0415
    from physicalai_rosbag_plugin.converter import convert_rosbags  # noqa: PLC0415

    previous = {sig: signal.signal(sig, _interrupt) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        args = cfg.convert
        result = convert_rosbags(args.input, MappingConfig.from_yaml(args.mapping), args.output, args.repo_id)
        print(f"Published {result.episodes} episodes / {result.frames} frames: {result.output}")  # noqa: T201
    except KeyboardInterrupt:
        print("Conversion interrupted; no partial dataset published.", file=sys.stderr)  # noqa: T201
        return 130
    except Exception as error:  # noqa: BLE001 - CLI boundary reports dependency and conversion failures.
        print(f"ROS bag conversion failed: {error}", file=sys.stderr)  # noqa: T201
        return 1
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return 0


def register() -> SubcommandSpec:
    """Register the converter with the existing physicalai/pai host.

    Returns:
        The host subcommand specification.
    """
    parser = ArgumentParser(prog="physicalai rosbag", description=HELP)
    convert = ArgumentParser(description=HELP)
    for name in ("input", "mapping", "output"):
        convert.add_argument(f"--{name}", type=Path, required=True)
    convert.add_argument("--repo-id", required=True)
    parser.add_subcommands(required=True).add_subcommand("convert", convert, help=HELP)
    return SubcommandSpec(name="rosbag", parser=parser, dispatch=_dispatch, help=HELP)
