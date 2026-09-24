# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""CLI entrypoint for the MuJoCo SO-101 simulation.

Usage:

    physicalai-mujoco-so101 start --model <path> [options]

Start a MuJoCo SO-101 simulation as a zenoh robot owner, making it
discoverable and controllable from PhysicalAI Studio.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import shlex
import signal
import sys
import threading
from dataclasses import asdict
from pathlib import Path

from loguru import logger

from physicalai.config import Config
from physicalai.robot.transport import SharedRobot
from physicalai_mujoco_so101_plugin.constants import (
    DEFAULT_BIMANUAL_MUJOCO_OWNER_NAME,
    DEFAULT_MUJOCO_OWNER_NAME,
)
from physicalai_mujoco_so101_plugin.mujoco_robot import BiMuJoCoSO101, MuJoCoSO101

_CLI_NAME = "physicalai-mujoco-so101"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="physicalai-mujoco-so101",
        description="MuJoCo SO-101 simulation for PhysicalAI Studio",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    start = sub.add_parser("start", help="Start the MuJoCo simulation as a zenoh robot owner")
    start.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to MuJoCo model XML or URDF (default: bundled SO101 model)",
    )
    start.add_argument(
        "--name",
        type=str,
        default=None,
        help=(
            f"Zenoh robot name (default: {DEFAULT_MUJOCO_OWNER_NAME}, "
            f"or {DEFAULT_BIMANUAL_MUJOCO_OWNER_NAME} with --bimanual)"
        ),
    )
    start.add_argument(
        "--rate-hz",
        type=float,
        default=50.0,
        help="Owner control loop rate in Hz (default: 50)",
    )
    start.add_argument(
        "--substeps",
        type=int,
        default=10,
        help="MuJoCo simulation steps per control cycle (default: 10, real-time at 50 Hz with dt=0.002)",
    )
    start.add_argument(
        "--bimanual",
        action="store_true",
        default=False,
        help="Run the dual-arm SO101 (bimanual) model (default scene: garment_fold)",
    )
    start.add_argument(
        "--scene",
        type=str,
        default=None,
        help="Scene name (default: garment_fold for bimanual, single_pick_place otherwise)",
    )
    start.add_argument(
        "--allow-remote",
        action="store_true",
        default=False,
        help="Allow remote zenoh connections (default: loopback only)",
    )
    start.add_argument(
        "--idle-timeout",
        type=float,
        default=None,
        help=(
            "Seconds with zero subscribers before self-exit "
            "(default: 10 without HTTP, disabled with HTTP so stream viewers keep the sim alive)"
        ),
    )
    start.add_argument(
        "--no-gui",
        action="store_true",
        default=False,
        help="Disable all viewers: the browser-based one (viser/mjviser) and the native MuJoCo fallback",
    )
    start.add_argument(
        "--viser-port",
        type=int,
        default=9090,
        help="Port for the browser-based 3D viewer (default: 9090)",
    )
    start.add_argument(
        "--viser-host",
        type=str,
        default="127.0.0.1",
        help="Host for the browser-based 3D viewer (default: 127.0.0.1; use 0.0.0.0 to expose it remotely)",
    )
    start.add_argument(
        "--no-cameras",
        action="store_true",
        default=False,
        help="Disable camera rendering entirely (HTTP streams and v4l2loopback)",
    )
    start.add_argument(
        "--http-host",
        type=str,
        default="127.0.0.1",
        help="Host for the camera/control HTTP server (default: 127.0.0.1)",
    )
    start.add_argument(
        "--http-port",
        type=int,
        default=8080,
        help="Port for the camera/control HTTP server (default: 8080)",
    )
    start.add_argument(
        "--no-http",
        action="store_true",
        default=False,
        help="Disable the camera/control HTTP server",
    )
    start.add_argument(
        "--v4l2",
        action="store_true",
        default=False,
        help="Also publish cameras to v4l2loopback devices (requires modprobe v4l2loopback)",
    )
    start.add_argument(
        "--wrist-video-id",
        type=int,
        default=60,
        help="v4l2loopback video ID for the wrist camera (only with --v4l2, default: 60)",
    )
    start.add_argument(
        "--right-wrist-video-id",
        type=int,
        default=61,
        help="v4l2loopback video ID for the right wrist camera (bimanual, only with --v4l2, default: 61)",
    )
    start.add_argument(
        "--overview-video-id",
        type=int,
        default=62,
        help="v4l2loopback video ID for the overview camera (only with --v4l2, default: 62)",
    )

    stop = sub.add_parser("stop", help="Stop a running MuJoCo simulation owner")
    stop.add_argument(
        "--name",
        type=str,
        default=DEFAULT_MUJOCO_OWNER_NAME,
        help=(
            "Zenoh robot name to stop when the HTTP endpoint is unreachable "
            f"(default: {DEFAULT_MUJOCO_OWNER_NAME}; pass {DEFAULT_BIMANUAL_MUJOCO_OWNER_NAME} for --bimanual runs)"
        ),
    )
    stop.add_argument(
        "--http-host",
        type=str,
        default="127.0.0.1",
        help="Host for the camera/control HTTP server (default: 127.0.0.1)",
    )
    stop.add_argument(
        "--http-port",
        type=int,
        default=8080,
        help="Port for the camera/control HTTP server (default: 8080)",
    )

    return parser


def _resolve_model_and_scene(
    model_arg: str | None,
    scene_arg: str | None,
    *,
    bimanual: bool,
) -> tuple[str, object | None]:
    if model_arg is not None:
        path = Path(model_arg).resolve()
        if not path.exists():
            logger.error("Model file not found: {}", path)
            sys.exit(1)
        return str(path), None

    from physicalai_mujoco_so101_plugin.scene_registry import get_scene  # noqa: PLC0415

    scene_id = scene_arg or ("garment_fold" if bimanual else "single_pick_place")
    scene = get_scene(scene_id)
    xml_path = scene.scene_xml_path
    if not xml_path.exists():
        logger.error("Scene XML not found: {}", xml_path)
        sys.exit(1)
    return str(xml_path), scene


def _resolve_owner_name(args: argparse.Namespace) -> str:
    """Return the explicit ``--name``, or the default for this arm count.

    Single-arm and bimanual simulations get distinct defaults so that running
    both at once does not have them fight over one zenoh name.

    Returns:
        The zenoh owner name to publish under.
    """
    if args.name is not None:
        return str(args.name)
    return DEFAULT_BIMANUAL_MUJOCO_OWNER_NAME if args.bimanual else DEFAULT_MUJOCO_OWNER_NAME


def _start(args: argparse.Namespace) -> None:  # noqa: C901, PLR0912, PLR0914, PLR0915
    model_path, scene_config = _resolve_model_and_scene(args.model, args.scene, bimanual=args.bimanual)
    owner_name = _resolve_owner_name(args)

    http_enabled = not args.no_http and args.http_port > 0

    cameras: list[dict[str, object]] = []
    if not args.no_cameras:
        if args.v4l2:
            video_ids = [args.wrist_video_id, args.overview_video_id]
            if args.bimanual:
                video_ids.append(args.right_wrist_video_id)
            if any(v < 0 for v in video_ids):
                msg = "Video IDs must be non-negative integers"
                raise ValueError(msg)
            if len(set(video_ids)) != len(video_ids):
                msg = "Wrist and overview video IDs must be different"
                raise ValueError(msg)

        def _device(video_id: int) -> str | None:
            return f"/dev/video{video_id}" if args.v4l2 else None

        left_wrist_name = "left_wrist" if args.bimanual else "wrist"
        cameras = [
            {
                "name": left_wrist_name,
                "device": _device(args.wrist_video_id),
                "width": 640,
                "height": 480,
                "fps": 30,
                "mirror_horizontal": True,
            },
            {
                "name": "overview",
                "device": _device(args.overview_video_id),
                "width": 640,
                "height": 480,
                "fps": 30,
                "mirror_horizontal": True,
            },
        ]
        if args.bimanual:
            cameras.append(
                {
                    "name": "right_wrist",
                    "device": _device(args.right_wrist_video_id),
                    "width": 640,
                    "height": 480,
                    "fps": 30,
                    "mirror_horizontal": True,
                },
            )

    robot_kwargs = {
        "model_path": model_path,
        "substeps": args.substeps,
        "enable_viewer": not args.no_gui,
        "cameras": cameras,
        "owner_name": owner_name,
        "http_host": args.http_host,
        "http_port": args.http_port if http_enabled else 0,
        "viser_host": args.viser_host,
        "viser_port": args.viser_port if not args.no_gui else 0,
    }
    if scene_config is not None:
        robot_kwargs["scene_config"] = asdict(scene_config)  # type: ignore[arg-type]

    idle_timeout = args.idle_timeout
    if idle_timeout is None and not http_enabled:
        idle_timeout = 10.0

    robot = SharedRobot.from_config(
        Config.from_instance((BiMuJoCoSO101 if args.bimanual else MuJoCoSO101)(**robot_kwargs)),  # type: ignore[arg-type]
        name=owner_name,
        allow_remote=args.allow_remote,
        rate_hz=args.rate_hz,
        idle_timeout=idle_timeout,
    )

    logger.info("Connecting MuJoCo SO-101 as zenoh owner '{}' ...", owner_name)
    robot.connect()
    owner_pid = _owner_pid(owner_name)
    logger.info(
        "MuJoCo SO-101 running (model={}, rate={} Hz, substeps={}, bimanual={})",
        model_path,
        args.rate_hz,
        args.substeps,
        args.bimanual,
    )
    if http_enabled:
        base_url = f"http://{args.http_host}:{args.http_port}"
        logger.info("Camera/control HTTP server: {}", base_url)
        for cam in cameras:
            # `cameras` is a list[dict[str, object]] built from CLI args.
            name = cam.get("name")
            if not isinstance(name, str):
                continue
            logger.info("MJPEG: {} -> {}/cameras/{}/mjpeg", name, base_url, name)
            logger.info("Snapshot: {} -> {}/cameras/{}/frame.jpg", name, base_url, name)
    if not args.no_gui and args.viser_port > 0:
        logger.info("3D viewer: http://{}:{}", args.viser_host, args.viser_port)

    shutdown = threading.Event()

    def _signal_handler(signum: int, frame: object) -> None:
        _ = frame, signum
        logger.info("Shutdown requested")
        shutdown.set()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    try:
        _wait_for_owner_shutdown(shutdown, owner_name, owner_pid)
    except KeyboardInterrupt:
        shutdown.set()
    finally:
        try:
            # An external stop (HTTP, viewer, or CLI) needs only subscriber
            # cleanup. Never forward shutdown to a replacement owner.
            if shutdown.is_set() and owner_pid is not None:
                stopped = http_enabled and _stop_owner_over_http(args.http_host, args.http_port, owner_name, owner_pid)
                if not stopped:
                    _stop_owner_by_signal(owner_name, owner_pid)
        finally:
            robot.disconnect()
        logger.info("MuJoCo SO-101 stopped")


def _wait_for_owner_shutdown(shutdown: threading.Event, name: str, pid: int | None) -> None:
    """Wait for an operator signal or the connected local owner to release its lock.

    Compare the PID as well as the name so a replacement owner cannot keep an
    old launcher alive. A missing initial PID means the local owner has already
    exited or this invocation attached remotely; detach without stopping it.
    """
    if pid is None:
        logger.info("No local owner registered for '{}'; detaching launcher", name)
        return
    while not shutdown.wait(timeout=0.25):
        if _owner_pid(name) != pid:
            logger.info("Owner '{}' exited; closing launcher", name)
            return


def _request_http_shutdown(host: str, port: int) -> bool:
    connection: http.client.HTTPConnection | None = None
    try:
        connection = http.client.HTTPConnection(host, port, timeout=5)
        connection.request("POST", "/shutdown")
        connection.getresponse().read()
    except (OSError, http.client.HTTPException):
        return False
    finally:
        if connection is not None:
            connection.close()
    return True


def _http_owner_name(host: str, port: int) -> str | None:
    """Return the owner name reported by the sim listening on *host*:*port*.

    Returns:
        The ``service`` field from its root endpoint, or ``None`` when
        unreachable or the response is not from this CLI's HTTP server.
    """
    connection: http.client.HTTPConnection | None = None
    try:
        connection = http.client.HTTPConnection(host, port, timeout=5)
        connection.request("GET", "/")
        payload = json.loads(connection.getresponse().read())
    except (OSError, http.client.HTTPException, ValueError):
        return None
    finally:
        if connection is not None:
            connection.close()
    service = payload.get("service") if isinstance(payload, dict) else None
    return service if isinstance(service, str) else None


def _stop_owner_over_http(host: str, port: int, name: str, pid: int) -> bool:
    """Request shutdown only while the original local owner serves this endpoint.

    Returns:
        Whether the HTTP shutdown request was delivered to the verified owner.
    """
    if _http_owner_name(host, port) != name or _owner_pid(name) != pid:
        logger.debug("HTTP shutdown skipped: endpoint or owner changed for '{}'", name)
        return False
    if _request_http_shutdown(host, port):
        logger.info("Owner shutdown requested via HTTP")
        return True
    logger.debug("HTTP shutdown endpoint unavailable for '{}'; falling back to SIGTERM", name)
    return False


def _stop_owner_by_signal(name: str, pid: int) -> bool:
    """Stop the original local owner when its HTTP shutdown endpoint is unavailable.

    Returns:
        Whether SIGTERM was delivered to the still-registered owner process.
    """
    if _owner_pid(name) != pid:
        logger.debug("Signal shutdown skipped: owner '{}' changed or exited", name)
        return False
    return _terminate(pid, f"owner '{name}'")


def _owner_pid(name: str) -> int | None:
    """Return the live PID recorded by the owner registered under *name*.

    Owner workers are spawned as a bare ``python -m ...._owner_worker`` with
    their config on stdin, so their command line says nothing about which robot
    they drive; the name lock is the only thing that identifies one.

    Returns:
        The owner's PID, or ``None`` when no live owner holds that name.
    """
    try:
        # No public API exposes the owner registry; the name lock is what the
        # transport itself uses to find a live owner by name.
        from physicalai.robot.transport._lock import (  # noqa: PLC0415, PLC2701
            NAME_KIND,
            _read_live_name_diagnostics,
            lock_path,
        )
    except ImportError:
        return None

    try:
        diagnostics = _read_live_name_diagnostics(lock_path(NAME_KIND, name))
    except (OSError, ValueError, RuntimeError):
        return None
    if diagnostics is None or diagnostics.get("identity") != name:
        return None
    pid = diagnostics.get("pid")
    if not isinstance(pid, int) or pid <= 0:
        return None
    try:
        # Signal 0 only checks that the PID exists and is signalable by us.
        os.kill(pid, 0)
    except OSError:
        return None
    return pid


def _terminate(pid: int, description: str) -> bool:
    """Send SIGTERM to *pid*, reporting whether the signal was delivered.

    Returns:
        ``True`` when the process was signalled.
    """
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError as exc:
        logger.warning("Could not stop {} (pid {}): {}", description, pid, exc)
        return False
    logger.info("Stopped {} (pid {})", description, pid)
    return True


def _pid_command_line(pid: int) -> str | None:
    """Return the full command line for *pid* via ``ps``, or ``None`` if unavailable.

    Returns:
        The process's command line, or ``None`` when it can't be read.
    """
    # The subprocess is limited to local process inspection; it never executes
    # a user-provided command or shell script.
    import subprocess  # noqa: PLC0415, S404  # nosec B404

    try:
        # The command is fixed and only reads this local process's command
        # line; it never executes a user-provided command or shell script.
        result = subprocess.run(  # nosec B603, B607  # noqa: S603
            ["ps", "-o", "args=", "-p", str(pid)],  # noqa: S607
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    output = result.stdout.strip()
    return output or None


def _pid_owner_name(pid: int) -> str | None:
    """Return the zenoh owner name a ``start`` process at *pid* resolves to.

    Parses ``--name`` and ``--bimanual`` out of the process's own command
    line, applying the same defaulting rules as :func:`_resolve_owner_name`,
    so the pgrep fallback in :func:`_stop` can filter matches down to the
    requested owner instead of killing every ``start`` process on the
    machine.

    Returns:
        The resolved owner name, or ``None`` when the command line for *pid*
        can't be read.
    """
    command_line = _pid_command_line(pid)
    if command_line is None:
        return None
    try:
        tokens = shlex.split(command_line)
    except ValueError:
        return None
    name: str | None = None
    bimanual = False
    for i, arg in enumerate(tokens):
        if arg == "--name" and i + 1 < len(tokens):
            name = tokens[i + 1]
        elif arg.startswith("--name="):
            name = arg.split("=", 1)[1]
        elif arg == "--bimanual":
            bimanual = True
    if name is not None:
        return name
    return DEFAULT_BIMANUAL_MUJOCO_OWNER_NAME if bimanual else DEFAULT_MUJOCO_OWNER_NAME


def _matching_pids(pattern: str) -> list[int]:
    """Return PIDs whose full command line matches *pattern*, excluding our own.

    Returns:
        Matching PIDs, with this process and its parent removed.
    """
    # The subprocess is limited to local process inspection; it never executes
    # a user-provided command or shell script.
    import subprocess  # noqa: PLC0415, S404  # nosec B404

    try:
        # The command is fixed and only lists local processes; the pattern is
        # an internal CLI constant, not a shell expression or executable.
        result = subprocess.run(  # nosec B603, B607  # noqa: S603
            ["pgrep", "-f", pattern],  # noqa: S607
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        # `pgrep` is not available on Windows; there's no portable fallback.
        return []
    # `stop` runs from the same console script, so it can match its own
    # ancestry; the CLI wrapper is `uv run ... <cli> stop` at minimum.
    excluded = {os.getpid(), os.getppid()}
    pids = []
    for token in result.stdout.split():
        try:
            pid = int(token)
        except ValueError:
            continue
        if pid not in excluded:
            pids.append(pid)
    return pids


def _stop(args: argparse.Namespace) -> None:
    # `--name` identifies which owner to stop, so resolving it locally is the
    # only precise path: `--http-host`/`--http-port` are unrelated to `--name`
    # and just happen to hit whatever process is bound to that host:port, so
    # trying HTTP shutdown first (or at all, when a named owner is found)
    # would stop the wrong simulation whenever two owners run concurrently.
    owner_pid = _owner_pid(args.name)
    if owner_pid is not None:
        # `_terminate` already logs the specific reason on failure (e.g. the
        # process was unsignalable), so there is nothing useful to add here.
        _terminate(owner_pid, f"owner '{args.name}'")
        return

    logger.warning(
        "No local owner registered under '{}'; trying HTTP shutdown at {}:{}",
        args.name,
        args.http_host,
        args.http_port,
    )
    # The port might belong to a *different* named owner (e.g. the bimanual
    # sim's default port), so confirm identity via its `/` endpoint before
    # shutting it down; this is best-effort (TOCTOU between the check and the
    # POST), not a hard guarantee.
    if _http_owner_name(args.http_host, args.http_port) == args.name and _request_http_shutdown(
        args.http_host,
        args.http_port,
    ):
        logger.info("Shutdown requested at http://{}:{}/shutdown", args.http_host, args.http_port)
        return

    stopped = False
    # Last resort: kill our own `start` process(es) by command line, but only
    # the ones whose own `--name`/`--bimanual` args resolve to this name.
    # Never this `stop` command or another plugin's owner worker, which
    # shares the same module path on the command line. This only runs once
    # both name-based and HTTP-based lookups have failed.
    for pid in _matching_pids(f"{_CLI_NAME} start"):
        if _pid_owner_name(pid) != args.name:
            continue
        stopped = _terminate(pid, f"{_CLI_NAME} start") or stopped

    if not stopped:
        logger.warning("No running MuJoCo SO-101 simulation found for name '{}'", args.name)


def main() -> None:
    """Parse command-line arguments and run the requested command."""
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "start":
        _start(args)
    elif args.command == "stop":
        _stop(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
