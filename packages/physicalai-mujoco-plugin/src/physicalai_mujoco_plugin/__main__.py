# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""CLI entrypoint for the MuJoCo simulation plugin.

Usage:

    physicalai-mujoco start [--profile so101] [--scene <id>] [options]
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
import http.client
import json
import os
import shlex
import signal
import sys
import threading
from pathlib import Path

from loguru import logger

from physicalai.config import Config
from physicalai.robot.transport import SharedRobot
from physicalai_mujoco_plugin.compose import fetch_profile, scene_needs_robot
from physicalai_mujoco_plugin.constants import (
    DEFAULT_BIMANUAL_MUJOCO_OWNER_NAME,
    DEFAULT_MUJOCO_OWNER_NAME,
)
from physicalai_mujoco_plugin.profiles import PROFILES, RobotProfile, get_profile, list_profiles
from physicalai_mujoco_plugin.studio_recorder import DEFAULT_STUDIO_URL, validate_studio_url

_CLI_NAME = "physicalai-mujoco"


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
            f"for two-arm scenes; {DEFAULT_MUJOCO_OWNER_NAME} for the SO-101)"
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
        help="Deprecated: use --scene with a two-arm scene. Picks garment_fold for the SO-101",
    )
    start.add_argument(
        "--scene",
        type=str,
        default=None,
        help="Scene id (default: the profile's default scene, single_pick_place for the SO-101)",
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


def _resolve_scene(args: argparse.Namespace, profile: RobotProfile) -> tuple[str | None, int]:
    """Return the scene id the driver will load and its number of robots, or exit with the reason.

    Returns:
        The scene id (``None`` for a custom ``--model`` without ``--scene``) and the robot count.
    """
    from physicalai_mujoco_plugin.scene_registry import get_scene  # noqa: PLC0415

    scene_id = args.scene
    if scene_id is None and args.bimanual:
        if profile.name != "so101":
            logger.error("--bimanual only applies to the SO-101; pass --scene with a two-arm scene")
            sys.exit(1)
        scene_id = "garment_fold"
    if args.model is not None:
        path = Path(args.model).resolve()
        if not path.exists():
            logger.error("Model file not found: {}", path)
            sys.exit(1)
        args.model = str(path)
        if scene_id is None:
            import mujoco  # noqa: PLC0415

            from physicalai_mujoco_plugin.compose import anchor_prefixes  # noqa: PLC0415

            spec = mujoco.MjSpec.from_file(args.model)  # pyrefly: ignore [missing-attribute]
            return None, max(1, len(anchor_prefixes(spec)))
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
    return scene_id, scene.num_arms


def _resolve_owner_name(args: argparse.Namespace, profile: str = "so101", num_arms: int | None = None) -> str:
    """Return the explicit ``--name``, or the default for this profile and arm count (CLI-3).

    Single-arm and bimanual simulations get distinct defaults so that running
    both at once does not have them fight over one zenoh name.

    Returns:
        The zenoh owner name to publish under.
    """
    if args.name is not None:
        return str(args.name)
    bimanual = num_arms == 2 if num_arms is not None else args.bimanual  # noqa: PLR2004
    if profile == "so101":
        return DEFAULT_BIMANUAL_MUJOCO_OWNER_NAME if bimanual else DEFAULT_MUJOCO_OWNER_NAME
    return f"mujoco-{profile}-bimanual-follow" if bimanual else f"mujoco-{profile}-follow"


def _fetch_robot_models(model_path: str | Path, profile: RobotProfile) -> None:
    """Fetch the robot the model attaches before the owner starts, or exit with the reason.

    The owner subprocess must report ready within a fixed startup timeout; a
    first-run download inside it could exceed that and fail with an unrelated
    timeout error. Fetching here has no deadline and shows its progress.
    """
    if not scene_needs_robot(model_path):
        return
    try:
        fetch_profile(profile)
    except RuntimeError as exc:
        logger.error("{}", exc)
        sys.exit(1)


def _start(args: argparse.Namespace) -> None:
    from physicalai_mujoco_plugin.robot import MuJoCoRobot  # noqa: PLC0415
    from physicalai_mujoco_plugin.scene_registry import get_scene  # noqa: PLC0415

    try:
        profile = get_profile(args.profile)
    except KeyError as exc:
        logger.error("{}", exc.args[0])
        sys.exit(1)
    scene_id, num_arms = _resolve_scene(args, profile)
    owner_name = _resolve_owner_name(args, profile.name, num_arms)
    xml_path = args.model if args.model is not None else get_scene(scene_id).scene_xml_path  # type: ignore[arg-type]
    _fetch_robot_models(xml_path, profile)

    http_enabled = not args.no_http and args.http_port > 0
    robot = MuJoCoRobot(
        profile.name,
        scene=scene_id,
        model_path=args.model,
        unit=args.unit,
        substeps=args.substeps,
        rate_hz=args.rate_hz,
        # None streams the robot cameras and `overview`: wrist, overview (and right_wrist).
        cameras=[] if args.no_cameras else None,
        enable_viewer=not args.no_gui,
        viser_host=args.viser_host,
        viser_port=args.viser_port if not args.no_gui else 0,
        http_host=args.http_host,
        http_port=args.http_port if http_enabled else 0,
        owner_name=owner_name,
        studio_url=validate_studio_url(args.studio_url),
    )

    idle_timeout = args.idle_timeout
    if idle_timeout is None and not http_enabled:
        idle_timeout = 10.0

    shared = SharedRobot.from_config(
        Config.from_instance(robot),
        name=owner_name,
        allow_remote=args.allow_remote,
        rate_hz=args.rate_hz,
        idle_timeout=idle_timeout,
    )

    logger.info("Connecting MuJoCo {} as zenoh owner '{}' ...", profile.display_name, owner_name)
    shared.connect()
    owner_pid = _owner_pid(owner_name)
    logger.info(
        "MuJoCo {} running (scene={}, rate={} Hz)",
        profile.display_name,
        scene_id or xml_path,
        args.rate_hz,
    )
    if http_enabled:
        base_url = f"http://{args.http_host}:{args.http_port}"
        logger.info("Camera/control HTTP server: {} (camera streams: {}/cameras)", base_url, base_url)
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
            shared.disconnect()
        logger.info("MuJoCo {} stopped", profile.display_name)


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

    Parses ``--name``, ``--profile``, ``--scene``, ``--model`` and
    ``--bimanual`` out of the process's own command line and resolves them
    through :func:`_resolve_owner_name`, as ``start`` does, so the pgrep
    fallback in :func:`_stop` can filter matches down to the requested owner
    instead of killing every ``start`` process on the machine. A custom
    ``--model`` without ``--scene`` counts its mount frames, as ``start``
    does (:func:`_model_robot_count`).

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
    scene_id = values.get("--scene")
    if scene_id is None and bimanual and profile == "so101":
        scene_id = "garment_fold"
    if scene_id is None and "--model" not in values:
        registered = PROFILES.get(profile)
        scene_id = registered.default_scene if registered is not None else None
    num_arms = None
    if scene_id is not None:
        from physicalai_mujoco_plugin.scene_registry import get_scene  # noqa: PLC0415

        try:
            num_arms = get_scene(scene_id).num_arms
        except KeyError:
            num_arms = None
    elif "--model" in values:
        num_arms = _model_robot_count(pid, values["--model"])
        if num_arms is None:
            return None  # unknown arm count: never match, so stop cannot signal the wrong owner
    return _resolve_owner_name(argparse.Namespace(name=None, bimanual=bimanual), profile, num_arms)


def _start_flag_values(tokens: list[str]) -> dict[str, str]:
    """Return the ``--name``/``--profile``/``--scene``/``--model`` values of a ``start`` command line.

    Returns:
        The flags present, in both ``--flag value`` and ``--flag=value`` forms, mapped to their values.
    """
    values: dict[str, str] = {}
    for i, arg in enumerate(tokens):
        flag, has_value, value = arg.partition("=")
        if flag in {"--name", "--profile", "--scene", "--model"}:
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
        import mujoco  # noqa: PLC0415

        from physicalai_mujoco_plugin.compose import anchor_prefixes  # noqa: PLC0415

        spec = mujoco.MjSpec.from_file(str(path))  # pyrefly: ignore [missing-attribute]
        return max(1, len(anchor_prefixes(spec)))
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
    # both name-based and HTTP-based lookups have failed.
    for pid in _matching_pids(f"{_CLI_NAME} start"):
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
    sys.stdout.write(f"{'PROFILE':<10} {'TIER':<12} {'MENAGERIE MODEL':<24} NAME\n")
    for profile in list_profiles():
        sys.stdout.write(
            f"{profile.name:<10} {profile.tier:<12} {profile.menagerie_model:<24} {profile.display_name}\n"
        )
    sys.stdout.write(
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
