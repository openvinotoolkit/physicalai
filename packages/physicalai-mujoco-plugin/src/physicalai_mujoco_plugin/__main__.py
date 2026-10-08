# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""CLI entrypoint for the MuJoCo simulation plugin.

Usage:

    physicalai-mujoco start [--profile so101] [--scene <id>] [--bimanual] [options]
    physicalai-mujoco profiles
    physicalai-mujoco prefetch

Start a MuJoCo simulation as a zenoh robot owner, making it discoverable and
controllable from PhysicalAI Studio. ``profiles`` lists the robots with a
hand-written profile; any other MuJoCo Menagerie model name loads as an
unsupported profile, outside CI and Studio.
``prefetch`` downloads the profiles' models ahead of time, for offline use.
"""

from __future__ import annotations

import argparse
import contextlib
import http.client
import json
import os
import shlex
import signal
import socket
import sys
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, get_args

from loguru import logger

from physicalai.config import Config
from physicalai.robot.transport import SharedRobot
from physicalai_mujoco_plugin.compose import OverviewStyle, fetch_profile, scene_needs_robot
from physicalai_mujoco_plugin.constants import (
    DEFAULT_BIMANUAL_MUJOCO_OWNER_NAME,
    DEFAULT_MUJOCO_OWNER_NAME,
    MAX_SEED,
    default_owner_name,
)
from physicalai_mujoco_plugin.profiles import RobotProfile, get_profile, list_profiles
from physicalai_mujoco_plugin.studio_recorder import DEFAULT_STUDIO_URL, validate_studio_url
from physicalai_mujoco_plugin.viewer import ViewerTheme

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping

_CLI_NAME = "physicalai-mujoco"
_MAX_PORT = 65535
_START_PATTERNS = (f"{_CLI_NAME} start", "physicalai_mujoco_plugin start")
"""Command-line patterns of a ``start`` process: the console script and the ``-m`` module form."""
_CAMERA_START_TIMEOUT_S = 15.0
"""Longest ``start --status-json`` waits for every camera's first frame before ``ready``."""
_STARTUP_POLL_S = 0.05
_LOOPBACK_FOR_WILDCARD = {"0.0.0.0": "127.0.0.1", "::": "::1"}  # noqa: S104 - mapped, not bound  # nosec B104
"""Bind-all hosts, and the loopback address ``start`` connects to and reports instead."""

_STATUS_JSON_HELP = """\
Print startup events as JSON, one object per line on stdout (logs stay on stderr):
{"event": "phase", "phase": "fetch"} before the robot model download, then
{"event": "phase", "phase": "fetch", "bytes", "total"} while it downloads;
{"event": "phase", "phase": "connect"} before the owner process starts;
{"event": "phase", "phase": "load"} while the owner loads the scene, viewer and servers;
{"event": "phase", "phase": "cameras"} while it waits for each camera's first frame; then
{"event": "ready", "name", "pid", "profile", "scene", "arms", "overview_style", "http_url", "viewer_url",
"cameras"},
or {"event": "error", "message"} and a non-zero exit. Requires a name no running simulation uses"""


def _port_number(value: str) -> int:
    """Parse a TCP port for ``argparse``: ``0`` asks for a free port.

    Returns:
        The port.

    Raises:
        argparse.ArgumentTypeError: If *value* is not an integer from 0 to 65535.
    """
    try:
        port = int(value)
    except ValueError:
        port = -1
    if not 0 <= port <= _MAX_PORT:
        msg = f"expected a port from 0 to {_MAX_PORT} (0 picks a free one), got {value!r}"
        raise argparse.ArgumentTypeError(msg)
    return port


def _seed_number(value: str) -> int:
    """Parse a reset seed for ``argparse``, in the range ``POST /seed`` accepts.

    Returns:
        The seed.

    Raises:
        argparse.ArgumentTypeError: If *value* is not an integer from 0 to :data:`MAX_SEED`.
    """
    try:
        seed = int(value)
    except ValueError:
        seed = -1
    if not 0 <= seed <= MAX_SEED:
        msg = f"expected a seed from 0 to {MAX_SEED}, got {value!r}"
        raise argparse.ArgumentTypeError(msg)
    return seed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=_CLI_NAME,
        description="MuJoCo simulation of MuJoCo Menagerie robots for Physical AI Runtime and Studio",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    start = sub.add_parser("start", help="Start the MuJoCo simulation as a zenoh robot owner")
    start.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Scene XML to load instead of the registered scene's; its robot_mount frames get the profile's "
            "robot, and an XML without them is used as is"
        ),
    )
    start.add_argument(
        "--profile",
        type=str,
        default="so101",
        help="Robot profile (see `profiles`) or any MuJoCo Menagerie model name (default: so101)",
    )
    start.add_argument(
        "--name",
        type=str,
        default=None,
        help=(
            "Zenoh robot name (default: mujoco-<profile>-follow, or mujoco-<profile>-bimanual-follow "
            f"with two arms; {DEFAULT_MUJOCO_OWNER_NAME} for the SO-101)"
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
        default=None,
        help="MuJoCo simulation steps per control cycle (default: real time at --rate-hz, 10 for the SO-101 scenes)",
    )
    start.add_argument(
        "--unit",
        choices=["normalized", "degrees"],
        default=None,
        help=(
            "Joint units for observations and actions (default: the profile's; normalized for the SO-101, like "
            "the calibrated SO101 driver: body joints -100..100 and gripper 0..100 across each joint's range)"
        ),
    )
    start.add_argument(
        "--bimanual",
        action="store_true",
        default=False,
        help=(
            "Two arms in the chosen scene (any tabletop scene), named left_* and right_*. With --model, "
            "the XML needs left_robot_mount and right_robot_mount frames"
        ),
    )
    start.add_argument(
        "--scene",
        type=str,
        default=None,
        help="Scene id (default: the profile's default scene, single_pick_place for the SO-101)",
    )
    start.add_argument(
        "--seed",
        type=_seed_number,
        default=None,
        help=f"Fixed seed for scene resets, from 0 to {MAX_SEED}, so object layouts repeat (default: random)",
    )
    start.add_argument(
        "--overview",
        choices=get_args(OverviewStyle),
        default="shoulder",
        help=(
            "Where a tabletop scene's overview camera stands: shoulder, high behind the robot (BridgeData's "
            "over-the-shoulder view), or front, across the table facing the robots (robosuite/LIBERO agentview). "
            "Not saved: the viewer and POST /overview switch it while the simulation runs (default: shoulder)"
        ),
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
        type=_port_number,
        default=9090,
        help="Port for the browser-based 3D viewer; 0 picks a free one (default: 9090)",
    )
    start.add_argument(
        "--viewer-theme",
        choices=get_args(ViewerTheme),
        default="default",
        help=(
            "Browser viewer look: studio is dark with Physical AI Studio's accent colour and no Shutdown "
            "button, for a viewer embedded in Studio, which stops the simulation itself (default: default)"
        ),
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
        help="Disable camera rendering (HTTP streams and viewer previews)",
    )
    start.add_argument(
        "--http-host",
        type=str,
        default="127.0.0.1",
        help="Host for the camera/control HTTP server (default: 127.0.0.1)",
    )
    start.add_argument(
        "--http-port",
        type=_port_number,
        default=8080,
        help="Port for the camera/control HTTP server; 0 picks a free one (default: 8080)",
    )
    start.add_argument(
        "--no-http",
        action="store_true",
        default=False,
        help="Disable the camera/control HTTP server",
    )
    start.add_argument("--status-json", action="store_true", default=False, help=_STATUS_JSON_HELP)
    start.add_argument(
        "--exit-with-parent",
        action="store_true",
        default=False,
        help=(
            "Shut down when stdin reaches end of file, as on SIGTERM: a parent process keeps the write end of "
            "a pipe open and the simulation stops when the parent exits or crashes. The owner process also "
            "exits when this command is killed"
        ),
    )
    start.add_argument(
        "--studio-url",
        type=str,
        default=DEFAULT_STUDIO_URL,
        help=f"Physical AI Studio backend for automatic episode recording (default: {DEFAULT_STUDIO_URL})",
    )

    sub.add_parser("profiles", help="List registered robot profiles")
    sub.add_parser("prefetch", help="Download the supported robot models from MuJoCo Menagerie")

    stop = sub.add_parser("stop", help="Stop a running MuJoCo simulation owner")
    stop.add_argument(
        "--name",
        type=str,
        default=DEFAULT_MUJOCO_OWNER_NAME,
        help=(
            "Zenoh robot name to stop when the HTTP endpoint is unreachable "
            f"(default: {DEFAULT_MUJOCO_OWNER_NAME}; pass {DEFAULT_BIMANUAL_MUJOCO_OWNER_NAME} for --bimanual runs, "
            "mujoco-<profile>-[bimanual-]follow for other profiles)"
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


def _resolve_scene(args: argparse.Namespace, profile: RobotProfile) -> tuple[str | None, int]:
    """Return the scene id the driver will load and its number of robots, or exit with the reason.

    The arm count is 2 with ``--bimanual``, else 1; a custom ``--model``'s mount frames decide its own,
    and ``--bimanual`` requires exactly ``left_robot_mount`` and ``right_robot_mount``.

    Returns:
        The scene id (``None`` for a custom ``--model`` without ``--scene``) and the robot count.
    """
    from physicalai_mujoco_plugin.compose import mount_prefixes  # noqa: PLC0415
    from physicalai_mujoco_plugin.scene_registry import check_arm_count, check_model_mounts, get_scene  # noqa: PLC0415

    scene_id = args.scene
    num_arms = None
    if args.model is not None:
        path = Path(args.model).resolve()
        if not path.exists():
            logger.error("Model file not found: {}", path)
            sys.exit(1)
        args.model = str(path)
        try:
            check_model_mounts(profile, mount_prefixes(path), bimanual=args.bimanual)
        except ValueError as exc:
            logger.error("{}: {}", path, exc)
            sys.exit(1)
        num_arms = _xml_robot_count(path)
        if scene_id is None:
            return None, num_arms
    scene_id = scene_id or profile.default_scene
    if scene_id is None:
        logger.error("Profile {} has no default scene; pass --scene or --model", profile.name)
        sys.exit(1)
    try:
        scene = get_scene(scene_id)
    except KeyError as exc:
        logger.error("{}", exc.args[0])
        sys.exit(1)
    if not scene.supports(profile):
        logger.error("Scene {} does not support the {} profile", scene_id, profile.name)
        sys.exit(1)
    if num_arms is None:
        num_arms = 2 if args.bimanual else 1
        try:
            check_arm_count(scene, profile, num_arms)
        except ValueError as exc:
            logger.error("{}", exc)
            sys.exit(1)
    return scene_id, num_arms


def _check_overview(args: argparse.Namespace, scene_id: str | None) -> None:
    """Exit with the reason when the scene cannot place its overview camera in the ``--overview`` style."""
    from physicalai_mujoco_plugin.scene_registry import get_scene  # noqa: PLC0415

    if args.overview == "shoulder":
        return
    if args.model is not None or scene_id is None:
        logger.error("--overview {} needs a registered tabletop scene; a --model keeps its own camera", args.overview)
        sys.exit(1)
    if args.overview not in get_scene(scene_id).overview_styles:
        logger.error("Scene {} has no {} overview camera; it keeps its own cameras", scene_id, args.overview)
        sys.exit(1)


def _xml_robot_count(path: str | Path) -> int:
    """Return how many robots a ``--model`` XML attaches: its anchor frames, at least one.

    Returns:
        The number of anchor frames, or 1 for a robot-complete model without any.
    """
    import mujoco  # noqa: PLC0415

    from physicalai_mujoco_plugin.compose import anchor_prefixes  # noqa: PLC0415

    spec = mujoco.MjSpec.from_file(str(path))  # pyrefly: ignore [missing-attribute]
    return max(1, len(anchor_prefixes(spec)))


def _resolve_owner_name(args: argparse.Namespace, profile: str = "so101", num_arms: int | None = None) -> str:
    """Return the explicit ``--name``, or the default for this profile and arm count (CLI-3).

    Single-arm and bimanual simulations get distinct defaults so that running
    both at once does not have them fight over one zenoh name.

    Returns:
        The zenoh owner name to publish under.
    """
    if args.name is not None:
        return str(args.name)
    if num_arms is None:
        num_arms = 2 if args.bimanual else 1
    return default_owner_name(profile, num_arms)


def _fetch_robot_models(
    model_path: str | Path,
    profile: RobotProfile,
    progress: Callable[[int, int], None] | None = None,
) -> None:
    """Fetch the robot the model attaches before the owner starts, or exit with the reason.

    The owner subprocess must report ready within a fixed startup timeout; a
    first-run download inside it could exceed that and fail with an unrelated
    timeout error. Fetching here has no deadline and shows its progress.

    Args:
        model_path: The scene XML.
        profile: The robot profile.
        progress: Called with the bytes downloaded so far and the total during a download.
    """
    if not scene_needs_robot(model_path):
        return
    try:
        fetch_profile(profile, progress)
    except RuntimeError as exc:
        logger.error("{}", exc)
        sys.exit(1)


class _StatusWriter:
    """Startup events of ``start --status-json``: one JSON object per line on stdout.

    Logs stay on stderr, so a parent process (Physical AI Studio) can follow the startup by reading
    stdout line by line. Without ``--status-json`` every method is a no-op.
    """

    def __init__(self, *, enabled: bool) -> None:
        """Create a writer; a disabled one prints nothing."""
        self.enabled = enabled
        self._last_error: str | None = None

    def emit(self, event: str, **fields: object) -> None:
        """Print one event line and flush it.

        A reader that went away (a broken pipe) turns the writer off, and stdout goes to
        ``/dev/null`` so the interpreter does not fail flushing it at exit.
        """
        if not self.enabled:
            return
        try:
            sys.stdout.write(json.dumps({"event": event, **fields}) + "\n")
            sys.stdout.flush()
        except OSError:
            self.enabled = False
            with contextlib.suppress(OSError, ValueError):
                devnull = os.open(os.devnull, os.O_WRONLY)
                os.dup2(devnull, sys.stdout.fileno())
                os.close(devnull)

    def phase(self, phase: str) -> None:
        """Announce a startup phase (``fetch``, ``connect``)."""
        self.emit("phase", phase=phase)

    def download(self, done: int, total: int) -> None:
        """Report the robot model download: bytes so far and the total (0 when unknown)."""
        self.emit("phase", phase="fetch", bytes=done, total=total)

    def error(self, message: str) -> None:
        """Report why ``start`` failed."""
        self.emit("error", message=message)

    @contextlib.contextmanager
    def reporting_failures(self) -> Iterator[None]:
        """Turn a failed ``start`` into an ``error`` event, then let the failure continue.

        A non-zero ``SystemExit`` reports the last message logged at ``ERROR`` level, which is
        where ``start`` explains its exits; any other exception reports its own message.

        Yields:
            Nothing; failures inside the block are reported and re-raised.

        Raises:
            SystemExit: Re-raised after reporting a non-zero exit.
        """
        if not self.enabled:
            yield
            return
        sink = logger.add(self._remember_error, level="ERROR", format="{message}")
        try:
            yield
        except SystemExit as exc:
            if exc.code not in {0, None}:
                self.error(self._last_error or f"start exited with code {exc.code}")
            raise
        except Exception as exc:
            self.error(str(exc) or type(exc).__name__)
            raise
        finally:
            logger.remove(sink)

    def _remember_error(self, message: Any) -> None:  # noqa: ANN401 - loguru's Message
        self._last_error = str(message.record["message"])


def _start(args: argparse.Namespace) -> None:
    """Run ``start``; with ``--status-json``, a failure also prints an ``error`` event."""
    status = _StatusWriter(enabled=args.status_json)
    with status.reporting_failures():
        _run_start(args, status)


def _run_start(args: argparse.Namespace, status: _StatusWriter) -> None:  # noqa: PLR0914, PLR0915
    _use_egl_when_headless()
    shutdown = threading.Event()
    if args.exit_with_parent:
        _shut_down_on_stdin_eof(shutdown)
    # A supervising parent owns this simulation's lifetime: it must never reach a simulation
    # that someone else started under the same name.
    supervised = args.status_json or args.exit_with_parent

    from physicalai_mujoco_plugin.robot import MuJoCoRobot  # noqa: PLC0415
    from physicalai_mujoco_plugin.scene_registry import get_scene  # noqa: PLC0415

    profile = _profile_or_exit(args.profile)
    scene_id, num_arms = _resolve_scene(args, profile)
    _check_overview(args, scene_id)
    owner_name = _resolve_owner_name(args, profile.name, num_arms)
    xml_path = args.model if args.model is not None else get_scene(scene_id).scene_xml_path  # type: ignore[arg-type]
    status.phase("fetch")
    _fetch_robot_models(xml_path, profile, status.download if status.enabled else None)
    if shutdown.is_set():
        # Still a failed start: the events end with `error` (not silence) and the exit code is 1.
        logger.error("The parent process exited before the simulation started")
        sys.exit(1)
    if supervised:
        _require_unused_name(owner_name)

    http_enabled = not args.no_http
    viewer_enabled = not args.no_gui
    http_port, viser_port = _server_ports(args)
    robot = MuJoCoRobot(
        profile.name,
        scene=scene_id,
        bimanual=args.bimanual,
        model_path=args.model,
        unit=args.unit,
        substeps=args.substeps,
        rate_hz=args.rate_hz,
        seed=args.seed,
        # None streams the robot cameras and `overview`: wrist, overview (and right_wrist).
        cameras=[] if args.no_cameras else None,
        overview=args.overview,
        enable_viewer=viewer_enabled,
        viser_host=args.viser_host,
        viser_port=viser_port,
        viewer_theme=args.viewer_theme,
        http_host=args.http_host,
        http_port=http_port,
        owner_name=owner_name,
        studio_url=validate_studio_url(args.studio_url),
        # The owner runs in its own session; it watches this process so a hard kill stops it too.
        exit_with_pid=os.getpid() if args.exit_with_parent else None,
    )

    shared = SharedRobot.from_config(
        Config.from_instance(robot),
        name=owner_name,
        allow_remote=args.allow_remote,
        rate_hz=args.rate_hz,
        # Without HTTP nothing keeps an unwatched owner alive; with it, stream viewers do.
        idle_timeout=10.0 if args.idle_timeout is None and not http_enabled else args.idle_timeout,
    )

    logger.info("Connecting MuJoCo {} as zenoh owner '{}' ...", profile.display_name, owner_name)
    status.phase("connect")
    with _announcing_load(status, owner_name):
        shared.connect()
    owner_pid = _owner_pid(owner_name)
    # SharedRobot.connect attaches to an owner that already uses the name instead of spawning one
    # (a race past _require_unused_name); a supervised start must never stop such an owner.
    spawned = owner_pid is not None and _spawned_owner(shared)
    may_stop_owner = spawned or not supervised
    logger.info(
        "MuJoCo {} running (scene={}, rate={} Hz)",
        profile.display_name,
        scene_id or xml_path,
        args.rate_hz,
    )
    _log_addresses(args, http_port, viser_port)

    def _signal_handler(signum: int, frame: object) -> None:
        _ = frame, signum
        logger.info("Shutdown requested")
        shutdown.set()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    try:
        if supervised and not spawned:
            logger.error("Owner '{}' was not started by this command; leaving it running", owner_name)
            sys.exit(1)
        if status.enabled:
            viewer_url = _local_url(args.viser_host, viser_port) if viewer_enabled else None
            status.emit(
                "ready",
                name=owner_name,
                pid=owner_pid,
                profile=profile.name,
                scene=scene_id,
                arms=num_arms,
                overview_style=args.overview,
                **_ready_addresses(args, status, owner_name, http_port, viewer_url),
            )
        _wait_for_owner_shutdown(shutdown, owner_name, owner_pid)
    except KeyboardInterrupt:
        shutdown.set()
    except SystemExit:
        # A start that fails after spawning its owner (no HTTP server) takes the owner down too.
        if spawned:
            shutdown.set()
        raise
    finally:
        try:
            # An external stop (HTTP, viewer, or CLI) needs only subscriber
            # cleanup. Never forward shutdown to a replacement owner.
            if shutdown.is_set() and owner_pid is not None and may_stop_owner:
                stopped = http_enabled and _stop_owner_over_http(args.http_host, http_port, owner_name, owner_pid)
                if not stopped:
                    _stop_owner_by_signal(owner_name, owner_pid)
        finally:
            shared.disconnect()
        logger.info("MuJoCo {} stopped", profile.display_name)


def _profile_or_exit(name: str) -> RobotProfile:
    """Return the profile called *name*, or exit with the reason.

    Returns:
        The profile.
    """
    try:
        return get_profile(name)
    except KeyError as exc:
        logger.error("{}", exc.args[0])
        sys.exit(1)


def _log_addresses(args: argparse.Namespace, http_port: int, viser_port: int) -> None:
    """Log where the HTTP server and the viewer listen."""
    if not args.no_http:
        logger.info(
            "Camera/control HTTP server: http://{}:{} (camera streams: http://{}:{}/cameras)",
            args.http_host,
            http_port,
            args.http_host,
            http_port,
        )
    if not args.no_gui:
        logger.info("3D viewer: http://{}:{}", args.viser_host, viser_port)


def _require_unused_name(owner_name: str) -> None:
    """Exit when a local owner already uses *owner_name*, before ``start`` could attach to it."""
    pid = _owner_pid(owner_name)
    if pid is not None:
        logger.error(
            "A simulation named '{}' is already running (owner pid {}); stop it or pass another --name",
            owner_name,
            pid,
        )
        sys.exit(1)


def _spawned_owner(shared: SharedRobot) -> bool:
    """Return whether *shared*'s ``connect()`` spawned the owner, rather than attaching to a running one.

    ``SharedRobot`` keeps the owner handle only when its own connect won the spawn; an attacher,
    including one that lost a spawn race past :func:`_require_unused_name`, has none. Unlike
    ``waitpid`` on the owner's pid, this works on every platform.

    Returns:
        ``True`` if this process started the owner.
    """
    return getattr(shared, "_owner", None) is not None


@contextlib.contextmanager
def _announcing_load(status: _StatusWriter, owner_name: str) -> Iterator[None]:
    """Emit the ``load`` phase once the spawned owner holds its name, while ``connect`` waits.

    The owner takes the name lock right before it loads the scene and starts the viewer, cameras
    and HTTP server, so the lock marks the start of that phase. A fast owner gets the event right
    after ``connect`` returns, so ``load`` always comes before ``cameras`` and ``ready``.

    Args:
        status: The event writer; nothing runs when it is disabled.
        owner_name: The owner's name.

    Yields:
        Nothing; the block runs ``connect``.
    """
    if not status.enabled:
        yield
        return
    stop, announced = threading.Event(), threading.Event()

    def _poll() -> None:
        while not stop.is_set():
            if _owner_pid(owner_name) is not None:
                announced.set()
                status.phase("load")
                return
            stop.wait(_STARTUP_POLL_S)

    poller = threading.Thread(target=_poll, name="mujoco-start-load", daemon=True)
    poller.start()
    try:
        yield
    finally:
        stop.set()
        poller.join()
    if not announced.is_set():
        status.phase("load")


def _server_ports(args: argparse.Namespace) -> tuple[int, int]:
    """Return the HTTP and viewer ports, with a free port where ``0`` was asked for, or exit.

    Returns:
        The HTTP and viser ports; ``0`` for a disabled server, as ``MuJoCoRobot`` expects.
    """
    try:
        http_port, viser_port = _free_ports([
            (args.http_host, None if args.no_http else args.http_port),
            (args.viser_host, None if args.no_gui else args.viser_port),
        ])
    except OSError as exc:
        logger.error("Could not pick a free port: {}", exc)
        sys.exit(1)
    return http_port, viser_port


def _ready_addresses(
    args: argparse.Namespace,
    status: _StatusWriter,
    owner_name: str,
    http_port: int,
    viewer_url: str | None,
) -> dict[str, object]:
    """Return the ``ready`` event's addresses and working cameras, or exit without the HTTP server or viewer.

    The owner decides whether the viewer started and which cameras render, so with HTTP both come
    from the owner's server. A requested HTTP server that does not answer for this owner (the port
    was taken, or it failed to start) is an error: Studio could neither control the simulation nor
    show its cameras. So is a requested browser viewer that did not start (viser failed to import,
    bind or build its scene): Studio embeds it.

    Args:
        args: The ``start`` arguments.
        status: The event writer, for the ``cameras`` phase.
        owner_name: The owner the HTTP server must answer for.
        http_port: The HTTP server's port.
        viewer_url: The configured viewer URL; ``None`` without a viewer.

    Returns:
        ``http_url`` and ``viewer_url`` (``None`` when disabled) and ``cameras``.
    """
    if args.no_http:
        status.phase("cameras")
        return {"http_url": None, "viewer_url": viewer_url, "cameras": []}
    host = _LOOPBACK_FOR_WILDCARD.get(args.http_host, args.http_host)
    root = _http_json(host, http_port, "/")
    if root is None or root.get("service") != owner_name:
        logger.error("The camera/control HTTP server of '{}' is not answering at {}:{}", owner_name, host, http_port)
        sys.exit(1)
    if viewer_url is not None and root.get("viewer_url") is None:
        logger.error(
            "The browser viewer of '{}' did not start; the simulation's log says why. Pass --no-gui to run without it",
            owner_name,
        )
        sys.exit(1)
    status.phase("cameras")
    return {
        "http_url": _local_url(host, http_port),
        "viewer_url": root.get("viewer_url"),
        "cameras": _working_cameras(host, http_port),
    }


def _working_cameras(host: str, port: int, timeout_s: float = _CAMERA_START_TIMEOUT_S) -> list[str]:
    """Wait for every camera's first frame and return the cameras that stream.

    The camera thread creates its renderers and renders asynchronously, after the owner reported
    ready; a renderer can fail there even though the rendering probe passed. Cameras that failed,
    or have no frame within *timeout_s*, are left out with a warning.

    Returns:
        The names of the cameras with a frame, in the owner's order.
    """
    deadline = time.monotonic() + timeout_s
    while True:
        health = _http_json(host, port, "/health")
        cameras = health.get("cameras") if health is not None else None
        if isinstance(cameras, list):
            pending = [camera for camera in cameras if not camera.get("has_frame") and not camera.get("failed")]
            if not pending or time.monotonic() >= deadline:
                working = [
                    str(camera["name"]) for camera in cameras if camera.get("has_frame") and not camera.get("failed")
                ]
                for camera in cameras:
                    if str(camera["name"]) not in working:
                        logger.warning("Camera '{}' is not streaming; it is left out of 'ready'", camera["name"])
                return working
        elif time.monotonic() >= deadline:
            logger.warning("The camera status at {}:{} is unavailable; reporting no cameras", host, port)
            return []
        time.sleep(_STARTUP_POLL_S * 2)


def _local_url(host: str, port: int) -> str:
    """Return ``http://host:port`` as this machine reaches it: a bind-all host becomes loopback.

    Returns:
        The URL, with an IPv6 host in brackets.
    """
    host = _LOOPBACK_FOR_WILDCARD.get(host, host)
    return f"http://[{host}]:{port}" if ":" in host else f"http://{host}:{port}"


def _free_ports(requests: list[tuple[str, int | None]]) -> list[int]:
    """Return the requested ports, with each ``0`` replaced by a free port on its host.

    The sockets stay bound until every port is picked, so two requests never get the same one;
    they are released before the owner binds the ports, and another process could take one in
    between.

    Args:
        requests: ``(host, port)`` pairs; ``None`` (a disabled server) comes back as ``0``.

    Returns:
        One port per request.
    """
    with contextlib.ExitStack() as stack:
        ports = []
        for host, requested in requests:
            port = requested or 0
            if requested == 0:
                family, kind, proto, _, address = socket.getaddrinfo(host, 0, type=socket.SOCK_STREAM)[0]
                sock = stack.enter_context(socket.socket(family, kind, proto))
                sock.bind(address)
                port = int(sock.getsockname()[1])
            ports.append(port)
        return ports


def _shut_down_on_stdin_eof(shutdown: threading.Event) -> None:
    """Set *shutdown* once stdin reaches end of file (``--exit-with-parent``).

    A parent that keeps the write end of a pipe open makes this the parent-death signal on macOS
    and Linux alike: the pipe closes when the parent exits, however it exits. Anything written to
    stdin is ignored.
    """

    def _watch() -> None:
        try:
            fd = sys.stdin.fileno()
            while os.read(fd, 4096):
                pass
        except (AttributeError, OSError, ValueError):
            pass  # no readable stdin counts as end of file
        logger.info("stdin closed: shutting down (--exit-with-parent)")
        shutdown.set()

    threading.Thread(target=_watch, name="exit-with-parent", daemon=True).start()


def _headless_gl_env(platform: str, environ: Mapping[str, str]) -> dict[str, str]:
    """Return the environment that makes MuJoCo render with EGL on a headless Linux host.

    Without ``DISPLAY`` or ``WAYLAND_DISPLAY`` MuJoCo's default GLFW backend has no display to open.
    An explicit ``MUJOCO_GL`` always wins, and a host without an EGL library keeps the default:
    the rendering probe then turns the cameras off with its warning.

    Args:
        platform: ``sys.platform``.
        environ: The current environment.

    Returns:
        The variables to set; empty when nothing changes.
    """
    import ctypes.util  # noqa: PLC0415

    if not platform.startswith("linux") or environ.get("MUJOCO_GL"):
        return {}
    if environ.get("DISPLAY") or environ.get("WAYLAND_DISPLAY"):
        return {}
    if ctypes.util.find_library("EGL") is None:
        return {}
    env = {"MUJOCO_GL": "egl"}
    if not environ.get("PYOPENGL_PLATFORM"):
        env["PYOPENGL_PLATFORM"] = "egl"
    return env


def _use_egl_when_headless() -> None:
    """Select EGL rendering on a headless Linux host, before anything imports MuJoCo.

    The owner process inherits the environment, so its cameras render with EGL too.
    """
    env = _headless_gl_env(sys.platform, os.environ)
    if env:
        os.environ.update(env)
        logger.info("No display found: rendering with EGL ({})", " ".join(f"{k}={v}" for k, v in env.items()))


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


def _http_json(host: str, port: int, path: str) -> dict[str, Any] | None:
    """Return the JSON object at *path* of the sim listening on *host*:*port* (``GET``).

    Returns:
        The decoded JSON object, or ``None`` when unreachable or not a JSON object.
    """
    connection: http.client.HTTPConnection | None = None
    try:
        connection = http.client.HTTPConnection(host, port, timeout=5)
        connection.request("GET", path)
        payload = json.loads(connection.getresponse().read())
    except (OSError, http.client.HTTPException, ValueError):
        return None
    finally:
        if connection is not None:
            connection.close()
    return payload if isinstance(payload, dict) else None


def _http_owner_name(host: str, port: int) -> str | None:
    """Return the owner name reported by the sim listening on *host*:*port*.

    Returns:
        The ``service`` field from its root endpoint, or ``None`` when
        unreachable or the response is not from this CLI's HTTP server.
    """
    payload = _http_json(host, port, "/")
    service = payload.get("service") if payload is not None else None
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

    Parses ``--name``, ``--profile``, ``--model`` and ``--bimanual`` out of
    the process's own command line and resolves them through
    :func:`_resolve_owner_name`, as ``start`` does, so the pgrep fallback in
    :func:`_stop` can filter matches down to the requested owner instead of
    killing every ``start`` process on the machine. The arm count is 2 with
    ``--bimanual``, else 1, whatever the scene; a custom ``--model`` without
    ``--bimanual`` counts its mount frames instead, as ``start`` does
    (:func:`_model_robot_count`).

    Returns:
        The resolved owner name, or ``None`` when the command line for *pid*
        can't be read or its ``--model`` arm count is unknown (never a match).
    """
    command_line = _pid_command_line(pid)
    if command_line is None:
        return None
    try:
        tokens = shlex.split(command_line)
    except ValueError:
        return None
    values = _start_flag_values(tokens)
    bimanual = "--bimanual" in tokens
    if "--name" in values:
        return values["--name"]
    profile = values.get("--profile", "so101")
    num_arms = 2 if bimanual else 1
    # ``start --model --bimanual`` runs only with two mount frames, so only a model without the flag is counted.
    if "--model" in values and not bimanual:
        num_arms = _model_robot_count(pid, values["--model"])
        if num_arms is None:
            return None  # unknown arm count: never match, so stop cannot signal the wrong owner
    return _resolve_owner_name(argparse.Namespace(name=None, bimanual=bimanual), profile, num_arms)


def _start_flag_values(tokens: list[str]) -> dict[str, str]:
    """Return the ``--name``/``--profile``/``--model`` values of a ``start`` command line.

    Returns:
        The flags present, in both ``--flag value`` and ``--flag=value`` forms, mapped to their values.
    """
    values: dict[str, str] = {}
    for i, arg in enumerate(tokens):
        flag, has_value, value = arg.partition("=")
        if flag in {"--name", "--profile", "--model"}:
            if has_value:
                values[flag] = value
            elif i + 1 < len(tokens):
                values[flag] = tokens[i + 1]
    return values


def _pid_cwd(pid: int) -> Path | None:
    """Return the working directory of *pid*: ``/proc`` on Linux, ``lsof`` elsewhere (macOS).

    Returns:
        The directory, or ``None`` when it can't be read.
    """
    proc_cwd = Path(f"/proc/{pid}/cwd")
    if proc_cwd.exists():
        return proc_cwd.resolve()
    # The subprocess is limited to local process inspection (fixed argv, no shell); it never
    # executes a user-provided command or shell script.
    import subprocess  # noqa: PLC0415, S404  # nosec B404

    try:
        # A fixed command that only reads this local process's working directory.
        result = subprocess.run(  # nosec B603, B607  # noqa: S603
            ["lsof", "-a", "-p", str(pid), "-d", "cwd", "-Fn"],  # noqa: S607
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    for line in result.stdout.splitlines():
        if line.startswith("n/"):
            return Path(line[1:])
    return None


def _model_robot_count(pid: int, model: str) -> int | None:
    """Count the robots a ``start --model`` at *pid* loads, as :func:`_resolve_scene` does.

    Returns:
        The number of mount frames (at least 1), or ``None`` when the model file can't be found
        or parsed: a relative path whose process directory is unknown, or a moved file.
    """
    path = Path(model)
    if not path.is_absolute():
        cwd = _pid_cwd(pid)
        if cwd is None:
            return None
        path = cwd / path
    try:
        return _xml_robot_count(path)
    except Exception:  # noqa: BLE001 - any unreadable model means "unknown", never a match
        return None


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
    # both name-based and HTTP-based lookups have failed. Both launch forms
    # count: the console script, and `python -m physicalai_mujoco_plugin start`,
    # which Studio runs (SimulationLaunch argv).
    candidates = [pid for pattern in _START_PATTERNS for pid in _matching_pids(pattern)]
    for pid in dict.fromkeys(candidates):
        if _pid_owner_name(pid) != args.name:
            continue
        stopped = _terminate(pid, f"{_CLI_NAME} start") or stopped

    if not stopped:
        logger.warning("No running MuJoCo simulation found for name '{}'", args.name)


def _prefetch() -> None:
    for profile in list_profiles():
        try:
            path = fetch_profile(profile)
        except RuntimeError as exc:
            logger.error("{}", exc)
            sys.exit(1)
        logger.info("{} model ready at {}", profile.name, path)


def _profiles() -> None:
    """Print the hand-written profiles; any other Menagerie model name loads as an unsupported profile."""
    from physicalai_mujoco_plugin.scene_registry import supported_arm_counts  # noqa: PLC0415

    sys.stdout.write(f"{'PROFILE':<22} {'TIER':<12} {'ARMS':<6} {'MENAGERIE MODEL':<24} NAME\n")
    for profile in list_profiles():
        arms = ", ".join(map(str, supported_arm_counts(profile)))
        sys.stdout.write(
            f"{profile.name:<22} {profile.tier:<12} {arms:<6} {profile.menagerie_model:<24} {profile.display_name}"
            f"{' (provisional)' if profile.provisional else ''}\n"
        )
    sys.stdout.write(
        "ARMS: 2 runs two arms with start --bimanual, in any tabletop scene. "
        "Any other MuJoCo Menagerie model name loads as an unsupported profile, outside CI and Studio. "
        'Tendon or site torque actuators need MuJoCoRobot(torque_mode="raw").\n'
    )


def main() -> None:
    """Parse command-line arguments and run the requested command."""
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "start":
        _start(args)
    elif args.command == "profiles":
        _profiles()
    elif args.command == "prefetch":
        _prefetch()
    elif args.command == "stop":
        _stop(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
