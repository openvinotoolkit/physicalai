# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import contextlib
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from physicalai.cli._config import RuntimeConfig


def _load_config(path: str | Path) -> RuntimeConfig:
    from physicalai.cli._config import load_config  # noqa: PLC0415, PLC2701

    return load_config(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="physicalai")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run a policy from YAML config")
    run_parser.add_argument("--config", required=True, help="Path to YAML config file")
    run_parser.add_argument("--duration-s", type=float, help="Override YAML duration_s")
    run_parser.add_argument("--fps", type=float, help="Override YAML fps")
    run_parser.add_argument("--dry-run", action="store_true", help="Load config, print summary, exit")
    run_parser.set_defaults(func=handle_run)

    return parser


def handle_run(args: argparse.Namespace) -> None:
    try:
        config = _load_config(args.config)
    except (FileNotFoundError, TypeError) as exc:
        sys.stderr.write(f"Error: {exc}\n")
        raise SystemExit(2) from None
    except ValueError as exc:
        sys.stderr.write(f"Error: {exc}\n")
        raise SystemExit(2) from None

    if args.duration_s is not None:
        config.duration_s = args.duration_s
    if args.fps is not None:
        config.fps = args.fps

    if args.dry_run:
        sys.stdout.write(f"Config loaded: fps={config.fps} duration_s={config.duration_s}\n")
        return

    from physicalai.runtime.runtime import PolicyRuntime  # noqa: PLC0415

    runtime = PolicyRuntime.from_config(config)

    try:
        runtime.robot.connect()
        for cam in runtime.cameras.values():
            cam.connect()
        runtime.run(duration_s=config.duration_s)
    except ConnectionError:
        raise SystemExit(1) from None
    finally:
        for cam in runtime.cameras.values():
            with contextlib.suppress(Exception):
                cam.disconnect()
        with contextlib.suppress(Exception):
            runtime.robot.disconnect()


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not hasattr(args, "func"):
        parser.print_help()
        raise SystemExit(2)

    try:
        args.func(args)
    except KeyboardInterrupt:
        raise SystemExit(0) from None


if __name__ == "__main__":
    main()
