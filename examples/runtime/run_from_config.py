#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Load a PolicyRuntime from a YAML config and run it.

Equivalent to ``physicalai run --config <path>`` but from a Python script,
useful for notebooks, debugging, and apps that want to drive the runtime
programmatically while still using the CLI's config schema.

Examples:

    # Run for the duration specified inside the YAML (or indefinitely if absent)
    python examples/runtime/run_from_config.py examples/runtime/rtc_runtime.yaml

    # Override duration from the command line
    python examples/runtime/run_from_config.py examples/runtime/rtc_runtime.yaml --duration-s 30
"""

from __future__ import annotations

import argparse
import signal
import sys
from pathlib import Path

from physicalai.runtime import PolicyRuntime


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
        default=None,
        help="Run duration in seconds. Defaults to running indefinitely.",
    )
    args = parser.parse_args()

    if not args.config.is_file():
        print(f"Config not found: {args.config}", file=sys.stderr)
        return 2

    print(f"Loading runtime from {args.config}...")
    runtime = PolicyRuntime.from_config(args.config)
    print("Runtime loaded.")

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
