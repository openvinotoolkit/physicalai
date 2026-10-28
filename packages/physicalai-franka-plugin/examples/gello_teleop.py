# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Teleoperate one or two real Franka FR3 arms with a Franka GELLO (Duo).

Leader:   ``GelloLeader`` / ``BimanualGelloLeader`` (passive: torque disabled, read-only).
Follower: the plugin's ``FrankaFR3`` (1 kHz loop in its own process, torque-mode joint
impedance by default), composed by ``BimanualFrankaFR3``, through ``RobotRuntime`` +
``TeleopSource``. This script adds the teleoperation workflow around them: a slow
catch-up to the GELLO pose, joint-by-joint alignment, single-joint teleoperation, a
joint-by-joint ``--park`` mover and read-only checks.

Run it on the real-time PC (Python <= 3.12 with ``pylibfranka``). Typical sequence::

    # 1. Which USB port is which GELLO? (no robot connection)
    python gello_teleop.py --probe --left-port /dev/serial/by-id/usb-ROBOTIS_..A --right-port ...B

    # 2. Compare robot and GELLO poses without moving the robot
    python gello_teleop.py --dry-run --left-host 172.16.0.49 --left-port /dev/serial/by-id/...

    # 3. Teleoperate one joint first, then all (Desk: brakes unlocked, FCI active, workspace clear)
    python gello_teleop.py --align --joints 1 --left-host 172.16.0.49 --left-port /dev/serial/by-id/...
    python gello_teleop.py --align --left-host 172.16.0.49 --left-port /dev/serial/by-id/...

``--align`` guides the GELLO onto the robot's pose joint by joint (read-only). After Enter
the robot moves in joint space to the (live) GELLO pose at ``--catch-up-deg-s`` (default
10 deg/s), then tracks it; teleop refuses to start beyond ``--max-offset-deg`` (30). Do
not start from the GELLO's folded rest pose: the FR3 cannot reach it (self-collision).
Ctrl+C stops; the robot holds its last pose.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import multiprocessing
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from physicalai_franka_plugin.constants import (
    FR3_GRIPPER_DEFAULT_SPEED_M_S,
    VALID_CONTROL_MODES,
    fr3_joint_limits_deg,
)
from physicalai_franka_plugin.robot import (
    FrankaFR3,
    _gripper_process,  # noqa: PLC2701 - also used by --gripper-only
)

GRIPPER_OPEN_M = 0.08
TRIGGER_CLOSE_BELOW = 0.35  # GELLO trigger: 1 = released, 0 = fully pressed
TRIGGER_OPEN_ABOVE = 0.65
GLITCH_DEG = 45.0  # a GELLO joint jumping further than this within one tick is held


JOINT_LIMIT_CHOICES = ("hardware", "suggested", "plugin")


def _plugin_limit_kind(kind: str) -> str:
    return "urdf" if kind == "plugin" else kind


def joint_limits_deg(kind: str = "hardware", margin_deg: float = 5.0) -> np.ndarray:
    """The plugin's FR3 joint limits (7 x 2, deg); ``plugin`` means the bundled URDF limits."""
    return fr3_joint_limits_deg(_plugin_limit_kind(kind), margin_deg)


def use_gello_limits(limits_deg: np.ndarray) -> None:
    """Make every ``GelloLeader`` clip (and wrap) its readings to ``limits_deg`` instead of the plugin's."""
    from physicalai_franka_plugin.gello import GelloLeader  # noqa: PLC0415

    GelloLeader._LIMITS_RAD = np.radians(np.asarray(limits_deg, dtype=np.float64))  # noqa: SLF001


class GelloToFranka:
    """Map GELLO values (per arm: 7 joints in degrees + trigger 0..1) to the follower action.

    After :meth:`start` with the robot's pose, the arm targets first *catch up*: they move
    from the robot pose toward the (live) GELLO pose at no more than ``catch_up_deg_s``.
    Once a catch-up step lands exactly on the GELLO pose (every joint closer than one step),
    targets follow the GELLO directly and the
    trigger starts driving the hand (it stays open during the catch-up).
    """

    def __init__(
        self,
        num_arms: int,
        *,
        gripper: bool,
        catch_up_deg_s: float = 10.0,
        fps: float = 25.0,
        follow_joints: list[int] | None = None,
    ) -> None:
        self._num_arms = num_arms
        self._gripper = gripper
        # Joints (0-based) that follow the GELLO; the others hold the robot pose from start().
        self._follow = np.zeros(7, dtype=bool)
        self._follow[list(range(7)) if follow_joints is None else follow_joints] = True
        self._hold_pose: np.ndarray | None = None
        self._previous: np.ndarray | None = None
        self._held = np.zeros((num_arms, 7), dtype=bool)
        self._hand = [GRIPPER_OPEN_M] * num_arms
        self._catch_up_step = catch_up_deg_s / fps
        self._commanded: np.ndarray | None = None
        self.aligned = True  # until start() is called, targets pass straight through
        self.verbose = False
        self._last_status = 0.0

    def start(self, robot_arm_deg: np.ndarray) -> None:
        """Begin the catch-up from the robot's current arm joints (degrees, ``num_arms`` x 7)."""
        self._commanded = np.asarray(robot_arm_deg, dtype=np.float64).reshape(self._num_arms, 7).copy()
        self._hold_pose = self._commanded.copy()
        self.aligned = False

    def __call__(self, observation: Any) -> np.ndarray:  # noqa: ANN401
        values = np.asarray(observation.joint_positions, dtype=np.float64).reshape(self._num_arms, 8)
        joints = values[:, :7].copy()
        if not np.all(np.isfinite(joints)):
            raise ValueError("GELLO returned non-finite joint values")
        if self._previous is not None:
            # A joint that jumps (e.g. the GELLO wrist turned past +-180 deg wraps from one FR3
            # limit to the other, or a bad read) is held at its last value until the GELLO
            # returns near it. Only that joint pauses; the others keep tracking.
            jumped = np.abs(joints - self._previous) > GLITCH_DEG
            for arm, joint in zip(*np.nonzero(jumped & ~self._held), strict=True):
                logger.warning(
                    "Holding arm {} J{}: GELLO jumped {:.0f} deg (past its range?). Move it back to resume.",
                    arm,
                    joint + 1,
                    abs(joints[arm, joint] - self._previous[arm, joint]),
                )
            for arm, joint in zip(*np.nonzero(self._held & ~jumped), strict=True):
                logger.info("Arm {} J{} follows the GELLO again", arm, joint + 1)
            self._held = jumped
            joints[jumped] = self._previous[jumped]
        self._previous = joints.copy()
        if self._hold_pose is not None:
            joints[:, ~self._follow] = self._hold_pose[:, ~self._follow]
        if not self.aligned and self._commanded is not None:
            gap = joints - self._commanded
            if np.all(np.abs(gap) <= self._catch_up_step):  # final step lands on the GELLO: no jump
                self._commanded = joints.copy()
                self.aligned = True
                logger.info("Robot reached the GELLO pose; teleoperation active")
            else:
                self._commanded += np.clip(gap, -self._catch_up_step, self._catch_up_step)
            joints = self._commanded.copy()
        parts = []
        for arm in range(self._num_arms):
            parts.append(joints[arm])
            if self._gripper:
                trigger = values[arm, 7]
                if self.aligned and trigger < TRIGGER_CLOSE_BELOW:
                    self._hand[arm] = 0.0
                elif trigger > TRIGGER_OPEN_ABOVE:
                    self._hand[arm] = GRIPPER_OPEN_M
                # between the thresholds (or a non-finite reading) the hand keeps its state
                parts.append([self._hand[arm]])
        now = time.monotonic()
        if self.verbose and now - self._last_status > 0.5:  # noqa: PLR2004
            self._last_status = now
            if not self.aligned:
                offset = float(np.max(np.abs(values[:, :7] - joints)[:, self._follow]))
                print(f"catching up: {offset:5.1f} deg to the GELLO pose (hold it still)", flush=True)
            else:
                hands = "  ".join(
                    f"trig {values[arm, 7]:.2f} -> {'CLOSE' if self._hand[arm] == 0.0 else 'open '}"
                    for arm in range(self._num_arms)
                )
                print(f"tracking | {hands}" if self._gripper else "tracking", flush=True)
        return np.concatenate(parts)


def make_teleop_source(leader: Any, mapping: GelloToFranka, *, gripper: bool) -> Any:  # noqa: ANN401
    """``TeleopSource`` that hands the robot's first observed pose to ``mapping.start``."""
    from physicalai.runtime.action_sources.teleop import TeleopSource  # noqa: PLC0415

    class CatchUpTeleopSource(TeleopSource):
        def __init__(self) -> None:
            super().__init__(leader, to_action=mapping)
            self._started = False

        def update(self, robot_state: Any, camera_frames: Any, step: int) -> np.ndarray:  # noqa: ANN401
            if not self._started:
                stride = 8 if gripper else 7
                state = np.asarray(robot_state.joint_positions, dtype=np.float64).reshape(-1, stride)
                mapping.start(state[:, :7])
                self._started = True
            return super().update(robot_state, camera_frames, step)

    return CatchUpTeleopSource()


def load_calibration(path: Path | None) -> str | None:
    """Return a calibration JSON string from ``physicalai-gello-calibrate``, or ``None`` for the factory default."""
    if path is None:
        return None
    from physicalai_franka_plugin.gello import GelloCalibration  # noqa: PLC0415

    text = path.read_text(encoding="utf-8")
    calibration = GelloCalibration.from_json(text)  # validates
    logger.info(
        "{}: offsets {} signs {}", path.name, np.round(calibration.assembly_offsets_rad, 3), calibration.joint_signs
    )
    return json.dumps(json.loads(text))


class ResilientLeader:
    """Wrap a GELLO leader so a short USB/serial dropout holds the last pose instead of ending the run.

    A failed read returns the last good observation (the robot keeps its target, i.e. holds).
    After ``reconnect_after`` failures in a row the port is reopened. If no read succeeds for
    ``max_outage_s``, the error propagates and teleoperation stops.
    """

    def __init__(self, leader: Any, *, max_outage_s: float = 1.0, reconnect_after: int = 3) -> None:  # noqa: ANN401
        self._leader = leader
        self._max_outage_s = max_outage_s
        self._reconnect_after = reconnect_after
        self._last: Any = None
        self._failures = 0
        self._outage_since: float | None = None

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401 - delegate joint_names, device_ids, ...
        return getattr(self._leader, name)

    def connect(self) -> None:
        for attempt in range(3):
            try:
                self._leader.connect()
                return
            except (ConnectionError, OSError) as exc:
                logger.warning("GELLO connect failed ({}); retrying", exc)
                with contextlib.suppress(Exception):
                    self._leader.disconnect()
                if attempt == 2:  # noqa: PLR2004
                    raise
                time.sleep(0.5)

    def disconnect(self) -> None:
        self._leader.disconnect()

    def is_connected(self) -> bool:
        return self._leader.is_connected()

    def get_observation(self) -> Any:  # noqa: ANN401
        try:
            observation = self._leader.get_observation()
        except (ConnectionError, OSError) as exc:
            now = time.monotonic()
            self._outage_since = self._outage_since or now
            self._failures += 1
            if self._last is None or now - self._outage_since > self._max_outage_s:
                raise
            logger.warning("GELLO read failed ({}); holding the last pose", exc)
            if self._failures % self._reconnect_after == 0:
                with contextlib.suppress(Exception):
                    self._leader.disconnect()
                    self._leader.connect()
                    logger.info("GELLO port reopened")
            return self._last
        if self._failures:
            logger.info("GELLO reads recovered after {} failures", self._failures)
        self._failures = 0
        self._outage_since = None
        self._last = observation
        return observation


def build_leader(args: argparse.Namespace, arms: list[str]) -> Any:  # noqa: ANN401
    from physicalai_franka_plugin.gello import BimanualGelloLeader, GelloLeader  # noqa: PLC0415

    ports = {arm: getattr(args, f"{arm}_port") for arm in arms}
    calibrations = {arm: load_calibration(getattr(args, f"{arm}_calibration")) for arm in arms}
    if len(arms) == 2:  # noqa: PLR2004
        return ResilientLeader(
            BimanualGelloLeader(
                ports["left"],
                ports["right"],
                left_calibration=calibrations["left"],
                right_calibration=calibrations["right"],
            )
        )
    return ResilientLeader(GelloLeader(ports[arms[0]], calibration=calibrations[arms[0]]))


def format_joints(values: np.ndarray) -> str:
    return " ".join(f"{value:7.1f}" for value in values)


ALIGN_STABLE_TICKS = 8  # consecutive reads inside the tolerance before the next joint (as RLinf align-sequential)
_ALIGN_BAR_SPAN_DEG = 45.0


def _wrap_deg(value: float) -> float:
    return (value + 180.0) % 360.0 - 180.0


def _align_bar(delta_deg: float, tol_deg: float, width: int = 31) -> str:
    """``[----===#===----]``: ``=`` is the tolerance band, ``#`` the GELLO, clipped at +-45 deg."""
    centre = width // 2
    cells = ["-"] * width
    zone = max(1, round(tol_deg / _ALIGN_BAR_SPAN_DEG * centre))
    for index in range(centre - zone, centre + zone + 1):
        cells[index] = "="
    clipped = float(np.clip(delta_deg, -_ALIGN_BAR_SPAN_DEG, _ALIGN_BAR_SPAN_DEG))
    cells[round((clipped + _ALIGN_BAR_SPAN_DEG) / (2 * _ALIGN_BAR_SPAN_DEG) * (width - 1))] = "#"
    return "[" + "".join(cells) + "]"


def align(  # noqa: PLR0913
    leader: Any,  # noqa: ANN401
    robot_deg: np.ndarray,
    arms: list[str],
    tol_deg: float,
    period_s: float = 0.1,
    joints: list[int] | None = None,
) -> None:
    """Guide the operator joint by joint until each GELLO matches its (static) robot pose. Read-only.

    Modelled on RLinf's ``test_gello.py align-sequential``: one joint at a time, live delta and a
    bar, advancing once the joint stays within ``tol_deg`` for ``ALIGN_STABLE_TICKS`` reads.
    """
    num_arms = len(arms)
    for index, arm in enumerate(arms):
        robot = robot_deg[index * 7 : index * 7 + 7]
        print(f"\nAlign the {arm} GELLO with its robot, one joint at a time (within +-{tol_deg:g} deg).")
        print("Move only the joint shown; the others may drift a little, the final check catches that.")
        for joint in range(7) if joints is None else joints:
            stable = 0
            while stable < ALIGN_STABLE_TICKS:
                gello = float(gello_joints_deg(leader, num_arms)[index * 7 + joint])
                delta = _wrap_deg(gello - robot[joint])
                inside = abs(delta) <= tol_deg
                stable = stable + 1 if inside else 0
                hint = "hold still" if inside else ("turn so it decreases" if delta > 0 else "turn so it increases")
                print(
                    f"\r  {arm} J{joint + 1}: robot {robot[joint]:7.1f}  GELLO {gello:7.1f}  "
                    f"delta {delta:+7.1f}  {_align_bar(delta, tol_deg)}  {hint:<21}",
                    end="",
                    flush=True,
                )
                time.sleep(period_s)
            print(f"\r  {arm} J{joint + 1}: aligned (delta {delta:+.1f} deg)" + " " * 70)


def probe(args: argparse.Namespace) -> None:
    """Print live GELLO readings per port, to identify left/right and check the calibration."""
    from physicalai_franka_plugin.gello import GelloLeader  # noqa: PLC0415

    leaders = {}
    for arm in ("left", "right"):
        port = getattr(args, f"{arm}_port")
        if port:
            leaders[arm] = GelloLeader(port, calibration=load_calibration(getattr(args, f"{arm}_calibration")))
            leaders[arm].connect()
    if not leaders:
        raise SystemExit("--probe needs --left-port and/or --right-port")
    print("Move one GELLO at a time; Ctrl+C to stop. Columns: J1..J7 (deg), trigger (1 = released)")
    try:
        while True:
            for arm, leader in leaders.items():
                values = leader.get_observation().joint_positions
                print(f"{arm:5s} {format_joints(values[:7])}  trig {values[7]:.2f}")
            print()
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        for leader in leaders.values():
            leader.disconnect()


def read_robot_poses_deg(hosts: list[str]) -> np.ndarray:
    """Read each arm's joint positions once without starting a motion (brakes may be locked)."""
    import pylibfranka  # noqa: PLC0415

    poses = []
    for host in hosts:
        robot = pylibfranka.Robot(host, pylibfranka.RealtimeConfig.kIgnore)
        poses.append(np.degrees(np.asarray(robot.read_once().q, dtype=np.float64)))
        del robot
    return np.concatenate(poses)


def gello_joints_deg(leader: Any, num_arms: int) -> np.ndarray:  # noqa: ANN401
    values = np.asarray(leader.get_observation().joint_positions, dtype=np.float64).reshape(num_arms, 8)
    return values[:, :7].reshape(-1)


def report_offset(
    leader: Any,  # noqa: ANN401
    robot_deg: np.ndarray,
    num_arms: int,
    catch_up_deg_s: float,
    joints: list[int] | None = None,
) -> float:
    """Print the per-joint GELLO-robot offset; return the largest one (of ``joints``, 0-based) in degrees."""
    error = gello_joints_deg(leader, num_arms) - robot_deg
    for arm in range(num_arms):
        print(f"arm {arm} GELLO - robot (deg): {format_joints(error[arm * 7 : arm * 7 + 7])}")
    followed = list(range(7)) if joints is None else joints
    worst = float(np.max(np.abs(error.reshape(num_arms, 7)[:, followed])))
    print(f"largest offset {worst:.1f} deg -> catch-up takes about {worst / catch_up_deg_s:.0f} s", flush=True)
    return worst


def gripper_only(args: argparse.Namespace, arms: list[str]) -> None:
    """Drive only the Franka Hand(s) from the GELLO trigger(s); the arms are never commanded."""
    hosts = [getattr(args, f"{arm}_host") for arm in arms]
    leader = build_leader(args, arms)
    leader.connect()
    context = multiprocessing.get_context("spawn")
    hands = []
    for host in hosts:
        commands, width = context.Queue(), context.Value("d", float("nan"), lock=False)
        process = context.Process(
            target=_gripper_process,
            args=(host, commands, width, FR3_GRIPPER_DEFAULT_SPEED_M_S * 2, args.gripper_force_n),
            daemon=True,
        )
        process.start()
        hands.append((host, commands, width, process))
    mapping = GelloToFranka(len(arms), gripper=True)  # never started: no catch-up, trigger active at once
    sent: list[bool | None] = [None] * len(arms)
    print("Gripper only: press the GELLO trigger to grasp, release to open. Ctrl+C to stop.")
    try:
        last_print = 0.0
        while True:
            observation = leader.get_observation()
            action = mapping(observation)
            triggers = np.asarray(observation.joint_positions, dtype=np.float64).reshape(len(arms), 8)[:, 7]
            for index, (host, commands, _width, process) in enumerate(hands):
                if not process.is_alive():
                    raise SystemExit(f"Gripper process for {host} exited; see the warning above")
                closed = bool(action[index * 8 + 7] < GRIPPER_OPEN_M / 2)
                if closed != sent[index]:
                    commands.put(0.0 if closed else GRIPPER_OPEN_M)  # widths: closing grasps with force
                    sent[index] = closed
            now = time.monotonic()
            if now - last_print > 0.25:  # noqa: PLR2004
                status = "  ".join(
                    f"{arm}: trig {trigger:.2f} -> {'CLOSE' if closed else 'open '} (width {hand[2].value:.3f} m)"
                    for arm, trigger, closed, hand in zip(arms, triggers, sent, hands, strict=True)
                )
                print(status, flush=True)
                last_print = now
            time.sleep(1.0 / args.fps)
    except KeyboardInterrupt:
        pass
    finally:
        for _host, commands, _width, process in hands:
            with contextlib.suppress(Exception):
                commands.put(None)
            process.join(timeout=5.0)
        leader.disconnect()


# Wrist rolls first (they mostly turn in place), then the bends, the upper-arm roll, the base last.
PARK_ORDER = (6, 4, 5, 3, 1, 2, 0)
# Franka's home pose (deg). ``--home`` goes there in reverse PARK_ORDER, retracing a park.
HOME_DEG = np.array([0.0, 0.0, 0.0, -90.0, 0.0, 90.0, 0.0])


def move_joint(follower: Any, command: np.ndarray, joint: int, target_deg: float, speed_deg_s: float) -> None:  # noqa: ANN401
    """Ramp one joint of ``command`` (deg, modified in place) to ``target_deg``; the others hold."""
    step = speed_deg_s / 25.0
    while abs(target_deg - command[joint]) > 1e-9:
        command[joint] += float(np.clip(target_deg - command[joint], -step, step))
        follower.send_action(command)
        time.sleep(0.04)
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:  # let the tracker settle
        if abs(follower.get_observation().joint_positions[joint] - target_deg) < 0.5:  # noqa: PLR2004
            return
        time.sleep(0.05)


def park(args: argparse.Namespace, arms: list[str], hosts: list[str], leader: Any) -> None:  # noqa: ANN401
    """Move each robot, one joint at a time and with confirmation, to its GELLO's current joint angles.

    For GELLOs that only rest in their stand: brings the robots into the GELLO configuration so
    teleoperation starts with a near-zero catch-up. Only one joint of one arm moves at a time.
    """
    home = getattr(args, "home", False)
    pose = getattr(args, "pose", None)
    if home:
        targets = np.tile(HOME_DEG, (len(arms), 1))
    elif pose is not None:
        if len(pose) != 7:  # noqa: PLR2004
            raise SystemExit(f"--pose needs 7 joint angles, got {len(pose)}")
        targets = np.tile(np.asarray(pose, dtype=np.float64), (len(arms), 1))
    else:
        targets = gello_joints_deg(leader, len(arms)).reshape(len(arms), 7)
    order = PARK_ORDER[::-1] if home else PARK_ORDER
    cpus = [int(cpu) for cpu in args.arm_cpus.split(",") if cpu.strip()] if args.arm_cpus else []
    followers = [
        FrankaFR3(
            host,
            gripper_enabled=False,
            realtime_config=args.realtime_config,
            dynamics_factor=args.dynamics_factor,
            cpu=cpus[index] if index < len(cpus) else None,
            joint_limits=_plugin_limit_kind(args.joint_limits),
            limit_margin_deg=args.limit_margin_deg,
            control_mode=args.control,
        )
        for index, host in enumerate(hosts)
    ]
    try:
        for follower in followers:
            follower.prepare()
        for follower in followers:
            follower.connect()
        for arm, follower, target in zip(arms, followers, targets, strict=True):
            command = np.asarray(follower.get_observation().joint_positions, dtype=np.float64).copy()
            goal = np.clip(target, follower.limits_deg[:, 0], follower.limits_deg[:, 1])
            print(f"\n{arm} arm ({follower.host}): robot {format_joints(command)}")
            label = " home" if home else " pose" if pose is not None else "GELLO"
            print(f"{'':{len(arm) + 4}}{'':{len(follower.host) + 4}}{label} {format_joints(goal)}")
            for joint in order:
                travel = goal[joint] - command[joint]
                if abs(travel) < 1.0:
                    continue
                answer = (
                    input(
                        f"{arm} J{joint + 1}: {command[joint]:7.1f} -> {goal[joint]:7.1f} deg "
                        f"({abs(travel):.0f} deg, ~{abs(travel) / args.catch_up_deg_s:.0f} s). "
                        "Enter = move, s = skip, q = quit: "
                    )
                    .strip()
                    .lower()
                )
                if answer == "q":
                    return
                if answer == "s":
                    continue
                move_joint(follower, command, joint, float(goal[joint]), args.catch_up_deg_s)
            print(f"{arm} arm parked at {format_joints(np.asarray(follower.get_observation().joint_positions))}")
        print("\nDone." if home or pose is not None else "\nDone. Hold the GELLOs in this pose and start teleop.")
    except KeyboardInterrupt:
        print("\nAborted; the arms hold their current pose.")
    finally:
        for follower in followers:
            follower.disconnect()


def _joint_list(text: str) -> list[int]:
    """``"1,3"`` -> ``[0, 2]`` (0-based), rejecting anything outside J1-J7."""
    joints = sorted({int(part) for part in text.split(",") if part.strip()})
    if not joints or joints[0] < 1 or joints[-1] > 7:  # noqa: PLR2004
        raise argparse.ArgumentTypeError(f"joints must be 1-7, got {text!r}")
    return [joint - 1 for joint in joints]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for arm in ("left", "right"):
        parser.add_argument(f"--{arm}-host", help=f"FCI IP of the {arm} arm's Control")
        parser.add_argument(f"--{arm}-port", help=f"Serial port of the {arm} GELLO (prefer /dev/serial/by-id/...)")
        parser.add_argument(
            f"--{arm}-calibration", type=Path, help="physicalai-gello-calibrate JSON (default: factory)"
        )
    parser.add_argument("--probe", action="store_true", help="Only print GELLO readings; no robot connection")
    parser.add_argument("--dry-run", action="store_true", help="Compare GELLO and robot poses; never moves the robot")
    parser.add_argument("--gripper", action="store_true", help="Drive the Franka Hand from the GELLO trigger")
    parser.add_argument("--gripper-only", action="store_true", help="Drive only the hand(s); never command the arm")
    parser.add_argument("--park", action="store_true", help="Move the robots joint by joint to the GELLO pose")
    parser.add_argument(
        "--home", action="store_true", help="With --park: go to Franka's home pose instead, in reverse park order"
    )
    parser.add_argument(
        "--align",
        action="store_true",
        help="Before teleop/dry-run: guide the GELLO joint by joint onto the robot's current pose (read-only)",
    )
    parser.add_argument("--align-tol-deg", type=float, default=5.0, help="Per-joint tolerance for --align")
    parser.add_argument(
        "--joints",
        type=_joint_list,
        default=None,
        help="Teleoperate only these joints (1-7, comma-separated, e.g. 1 or 1,2); the others hold their pose",
    )
    parser.add_argument(
        "--pose",
        type=lambda text: [float(v) for v in text.split(",")],
        help="With --park: go to these 7 joint angles (deg, comma-separated) instead of the GELLO pose",
    )
    parser.add_argument(
        "--joint-limits",
        choices=JOINT_LIMIT_CHOICES,
        default="hardware",
        help="Limits for GELLO readings and robot commands: FR3 hardware minus a margin (default), "
        "Franka's suggested limits, or the plugin's URDF limits",
    )
    parser.add_argument("--limit-margin-deg", type=float, default=5.0, help="Margin inside the hardware limits")
    parser.add_argument("--gripper-force-n", type=float, default=30.0, help="Grasp force in newtons")
    parser.add_argument("--fps", type=float, default=25.0, help="GELLO read / action rate (Franka uses 25 Hz)")
    parser.add_argument("--dynamics-factor", type=float, default=0.3, help="Fraction of FR3 vel/acc/jerk limits")
    parser.add_argument("--catch-up-deg-s", type=float, default=10.0, help="Joint speed while moving to the GELLO pose")
    parser.add_argument(
        "--max-offset-deg",
        type=float,
        default=30.0,
        help="Refuse to start beyond this offset: hold the GELLO in the robot's pose so the catch-up stays short",
    )
    parser.add_argument("--yes", action="store_true", help="Start the catch-up without asking for Enter")
    parser.add_argument("--realtime-config", choices=("kEnforce", "kIgnore"), default="kEnforce")
    parser.add_argument(
        "--control",
        choices=sorted(VALID_CONTROL_MODES),
        default="impedance",
        help="impedance: torques with Franka's GELLO joint-impedance gains (default); position: joint positions",
    )
    parser.add_argument("--arm-cpus", default="10,11", help="CPU cores for the left,right arm loops ('' = none)")
    parser.add_argument("--duration-s", type=float, default=None)
    args = parser.parse_args()
    limits = joint_limits_deg(args.joint_limits, args.limit_margin_deg)
    use_gello_limits(limits)
    print(
        f"Joint limits ({args.joint_limits}): "
        + "  ".join(f"J{i + 1} {lo:.0f}..{hi:.0f}" for i, (lo, hi) in enumerate(limits))
    )

    if args.probe:
        probe(args)
        return

    arms = [arm for arm in ("left", "right") if getattr(args, f"{arm}_host")]
    if not arms:
        parser.error("give --left-host and/or --right-host")
    for arm in arms:
        if not getattr(args, f"{arm}_port"):
            parser.error(f"--{arm}-port is required with --{arm}-host")
    hosts = [getattr(args, f"{arm}_host") for arm in arms]
    num_arms = len(arms)
    if args.gripper_only:
        gripper_only(args, arms)
        return
    sys.setswitchinterval(0.0005)  # keep the GELLO and runtime threads responsive

    leader = build_leader(args, arms)
    leader.connect()
    try:
        if args.park:
            park(args, arms, hosts, leader)
            return
        robot_deg = read_robot_poses_deg(hosts)
        print(f"Arms {arms} at {hosts}")
        follow = args.joints
        if follow is not None:
            print(f"Only J{', J'.join(str(j + 1) for j in follow)} follow the GELLO; the other joints hold their pose.")
        if args.align:
            try:
                align(leader, robot_deg, arms, args.align_tol_deg, joints=follow)
            except KeyboardInterrupt:
                raise SystemExit("\nAlignment aborted; nothing was commanded.") from None
            print("\nAligned. Keep holding the GELLO still.")
        worst = report_offset(leader, robot_deg, num_arms, args.catch_up_deg_s, follow)
        if args.dry_run:
            print("Dry run: nothing is commanded. Live offsets below; Ctrl+C to stop.\n")
            with contextlib.suppress(KeyboardInterrupt):
                while True:
                    time.sleep(0.5)
                    report_offset(leader, robot_deg, num_arms, args.catch_up_deg_s)
                    print()
            return
        if worst > args.max_offset_deg:
            raise SystemExit(f"Offset {worst:.0f} deg exceeds --max-offset-deg {args.max_offset_deg:.0f}; move closer")
        if not args.yes:
            input(
                f"The robot will move in joint space to the GELLO pose at {args.catch_up_deg_s:.0f} deg/s. Hold the "
                "GELLO still, check the path is clear, then press Enter (Ctrl+C aborts). "
            )

        from physicalai_franka_plugin.bimanual import BimanualFrankaFR3  # noqa: PLC0415

        from physicalai.runtime import RobotRuntime  # noqa: PLC0415

        cpus = [int(cpu) for cpu in args.arm_cpus.split(",") if cpu.strip()] if args.arm_cpus else []
        followers = [
            FrankaFR3(
                host,
                gripper_enabled=args.gripper,
                realtime_config=args.realtime_config,
                dynamics_factor=args.dynamics_factor,
                gripper_force_n=args.gripper_force_n,
                cpu=cpus[index] if index < len(cpus) else None,
                joint_limits=_plugin_limit_kind(args.joint_limits),
                limit_margin_deg=args.limit_margin_deg,
                control_mode=args.control,
            )
            for index, host in enumerate(hosts)
        ]
        # Bring every arm and hand process up first; only then start the 1 kHz loops together.
        try:
            for follower in followers:
                follower.prepare()
        except Exception:
            for follower in followers:
                follower.disconnect()
            raise
        robot = followers[0] if num_arms == 1 else BimanualFrankaFR3(left=followers[0], right=followers[1])
        mapping = GelloToFranka(
            num_arms, gripper=args.gripper, catch_up_deg_s=args.catch_up_deg_s, fps=args.fps, follow_joints=follow
        )
        mapping.verbose = True
        source = make_teleop_source(leader, mapping, gripper=args.gripper)
        runtime = RobotRuntime(robot=robot, action_source=source, fps=args.fps)
        runtime.connect()
        try:
            print("Moving to the GELLO pose, then teleoperating. Ctrl+C to stop.")
            runtime.run(duration_s=args.duration_s)
        finally:
            runtime.disconnect()
            overruns = [follower.loop_overruns for follower in followers]
            print(f"Stopped ({runtime.last_run_reason}); late 1 kHz cycles per arm: {overruns}")
    finally:
        if leader.is_connected():
            leader.disconnect()


if __name__ == "__main__":
    main()
