# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Launch a MuJoCo simulation as a zenoh robot owner, from Python.

Usage:
    uv run python examples/run_mujoco_owner.py [--profile so101] [--scene <id>] [--name <name>]

The simulation publishes state and accepts actions via zenoh. Use the
PhysicalAI Studio plugin to connect to it.
"""

from __future__ import annotations

import argparse
import signal
import time

from loguru import logger
from physicalai.config import Config
from physicalai.robot.transport import SharedRobot

from physicalai_mujoco_plugin.constants import DEFAULT_MUJOCO_OWNER_NAME
from physicalai_mujoco_plugin.robot import MuJoCoRobot


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a MuJoCo simulation as a zenoh robot owner")
    parser.add_argument("--profile", type=str, default="so101", help="Robot profile or Menagerie model name")
    parser.add_argument("--scene", type=str, default=None, help="Scene id (default: the profile's)")
    parser.add_argument(
        "--name",
        type=str,
        default=DEFAULT_MUJOCO_OWNER_NAME,
        help="Zenoh robot name",
    )
    parser.add_argument("--rate-hz", type=float, default=100.0, help="Control loop rate")
    parser.add_argument("--substeps", type=int, default=None, help="Sim steps per control cycle (default: real time)")
    parser.add_argument("--allow-remote", action="store_true", help="Allow remote zenoh connections")
    args = parser.parse_args()

    robot = SharedRobot.from_config(
        Config.from_instance(
            MuJoCoRobot(args.profile, scene=args.scene, substeps=args.substeps, rate_hz=args.rate_hz, cameras=[])
        ),
        name=args.name,
        allow_remote=args.allow_remote,
        rate_hz=args.rate_hz,
    )

    logger.info("Connecting MuJoCo {} zenoh owner '{}' ...", args.profile, args.name)
    robot.connect()
    logger.info("Running. Press Ctrl+C to stop.")

    shutdown = False

    def _signal_handler(signum: int, frame: object) -> None:
        _ = signum, frame
        nonlocal shutdown
        shutdown = True

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    try:
        while not shutdown:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        robot.disconnect()
        logger.info("Stopped.")


if __name__ == "__main__":
    main()
