#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Load a PolicyRuntime from a YAML config and run it.

Showcases :meth:`PolicyRuntime.from_config` — loads the same ``runtime:``
schema as ``physicalai run --config``, but the caller owns ``duration_s``
(passed directly to :meth:`PolicyRuntime.run`). Any ``run:`` block in the
YAML is parsed and ignored by ``from_config``.

Examples:

    # Run for the default 60s
    python examples/runtime/run_from_config.py examples/runtime/runtime.yaml

    # Override duration
    python examples/runtime/run_from_config.py examples/runtime/runtime.yaml --duration-s 30
"""

from __future__ import annotations

import argparse
import signal
import sys
from pathlib import Path

from physicalai.runtime import PolicyRuntime

_DEFAULT_DURATION_S = 60.0


def main() -> int:
    def _handle_sigint(sig: int, frame: object) -> None:  # noqa: ARG001
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        print("\nInterrupting... press Ctrl+C again to force kill.")
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _handle_sigint)

    parser = argparse.ArgumentParser(
        description="Load a PolicyRuntime from a YAML config and run it.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("config", type=Path, help="Path to runtime YAML config.")
    parser.add_argument(
        "--duration-s",
        type=float,
        default=_DEFAULT_DURATION_S,
        help=f"Run duration in seconds (default: {_DEFAULT_DURATION_S:g}).",
    )
    args = parser.parse_args()

    if not args.config.is_file():
        print(f"Config not found: {args.config}", file=sys.stderr)
        return 2

    runtime = PolicyRuntime.from_config(args.config)

    with runtime:
        print(f"Running (duration_s={args.duration_s})")
        stats = runtime.run(duration_s=args.duration_s)

    print(
        f"\nDone — {stats.steps} steps, {stats.inference_count} inferences, "
        f"{stats.total_holds} holds, {stats.transient_errors} transient errors",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
