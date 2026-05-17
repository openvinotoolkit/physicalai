# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""CLI entry point for ``python -m physicalai.runtime.observer``."""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> None:
    """Run the telemetry observer with optional recording.

    Raises:
        SystemExit: If zenoh/msgpack dependencies are missing.
    """
    parser = argparse.ArgumentParser(
        prog="python -m physicalai.runtime.observer",
        description="Observe runtime telemetry from a running PolicyRuntime session",
    )
    parser.add_argument("--session-id", default=None, help="Filter to a specific session ID")
    parser.add_argument("--record", default=None, metavar="PATH", help="Record events to JSONL file")
    parser.add_argument("--no-console", action="store_true", help="Disable live console output")
    args = parser.parse_args(argv)

    try:
        from physicalai.runtime.observer._subscriber import TelemetrySubscriber  # noqa: PLC0415, PLC2701
    except ImportError:
        raise SystemExit(1) from None

    subscriber = TelemetrySubscriber(session_id=args.session_id)

    if not args.no_console:
        from physicalai.runtime.observer._console import ConsoleHandler  # noqa: PLC0415, PLC2701

        subscriber.add_handler(ConsoleHandler())

    recorder = None
    if args.record:
        from pathlib import Path  # noqa: PLC0415

        from physicalai.runtime.observer._recorder import RecorderHandler  # noqa: PLC0415, PLC2701

        recorder = RecorderHandler(Path(args.record))
        subscriber.add_handler(recorder)

    subscriber.start()

    try:
        import signal  # noqa: PLC0415

        signal.pause()
    except AttributeError:
        import time  # noqa: PLC0415

        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        subscriber.stop()
        if recorder:
            recorder.close()


if __name__ == "__main__":
    main()
