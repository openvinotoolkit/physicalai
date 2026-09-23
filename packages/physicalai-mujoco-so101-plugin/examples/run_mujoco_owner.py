"""Launch a MuJoCo SO-101 simulation as a zenoh robot owner.

Usage:
    uv run python examples/run_mujoco_owner.py [--model <path>] [--name <name>]

The simulation publishes state and accepts actions via zenoh. Use the
PhysicalAI Studio plugin to connect to it.
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

from loguru import logger
from physicalai.config import Config
from physicalai.robot.transport import SharedRobot

from physicalai_mujoco_so101_plugin.constants import DEFAULT_MUJOCO_OWNER_NAME
from physicalai_mujoco_so101_plugin.mujoco_robot import MuJoCoSO101


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MuJoCo SO-101 as a zenoh robot owner")
    parser.add_argument("--model", type=str, default=None, help="MuJoCo model XML/URDF path")
    parser.add_argument(
        "--name",
        type=str,
        default=DEFAULT_MUJOCO_OWNER_NAME,
        help="Zenoh robot name",
    )
    parser.add_argument("--rate-hz", type=float, default=100.0, help="Control loop rate")
    parser.add_argument("--substeps", type=int, default=1, help="Sim steps per control cycle")
    parser.add_argument("--allow-remote", action="store_true", help="Allow remote zenoh connections")
    args = parser.parse_args()

    model_path = args.model
    if model_path is None:
        from physicalai_mujoco_so101_plugin._urdf import get_urdf_path

        urdf_root = get_urdf_path()
        model_path = str(urdf_root / "so101" / "so101.xml")
        if not Path(model_path).exists():
            logger.error("Bundled model not found at {}", model_path)
            sys.exit(1)

    robot_kwargs = {"model_path": model_path, "substeps": args.substeps}
    robot = SharedRobot.from_config(
        Config.from_instance(MuJoCoSO101(**robot_kwargs)),
        name=args.name,
        allow_remote=args.allow_remote,
        rate_hz=args.rate_hz,
    )

    logger.info("Connecting MuJoCo SO-101 zenoh owner '{}' ...", args.name)
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
