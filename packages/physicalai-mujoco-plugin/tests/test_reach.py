# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Reachability (TST-6): every arm of a reach-scaled scene reaches its spawn area from above.

For each fixed-base profile that a ``layout="reach"`` scene lists, a generic damped least-squares IK
(``mj_jac`` on the profile's end effector) puts the end effector on sample points of the scaled
spawn arc, at object height, with its approach axis as close to straight down as the arm allows.
Position comes first; the approach axis uses the null space that is left.

Thresholds: position within 1 cm, approach axis within 15 degrees of straight down. The SO-101's own
spawn arc (the reference, whose layout must stay as it is) reaches 0.36 m out and needs 12 degrees
of tilt there. ``RobotProfile.reach`` uses the same tolerance: the farthest point at object height,
toward the spawn area, that the end effector reaches within 5 mm and 15 degrees. The test measures
it for every arm and checks the stored value to 1 cm, so the stored values cannot rot.

Robot-complete models with several arms (ALOHA) check each sample point with the arm whose base is
nearest, so each arm covers its own side of the spawn area. Two copies of a one-arm profile
(``bimanual``, laid out side by side) check the points on the arc's axis, where the arms hand over,
with both arms, and the other points with the nearest arm; their reach is measured with one arm.

The test is deselected by default (marker ``mujoco_smoke``) and renders nothing. Run this file as a
script to print the measured reaches and the errors at every sample point::

    uv run python packages/physicalai-mujoco-plugin/tests/test_reach.py [profile ...]
"""

# MuJoCo's bindings are supplied by the C extension at runtime.
# pyrefly: ignore-errors [missing-attribute]

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco
import numpy as np
import pytest

from physicalai_mujoco_plugin.compose import compose_scene
from physicalai_mujoco_plugin.profiles import get_profile
from physicalai_mujoco_plugin.scene_registry import list_scenes, supported_arm_counts
from physicalai_mujoco_plugin.spawn import FREEJOINT_SPAWN_Z

if TYPE_CHECKING:
    from physicalai_mujoco_plugin.profiles import EndEffector, RobotProfile
    from physicalai_mujoco_plugin.scene_registry import SceneConfig

pytestmark = [pytest.mark.mujoco_smoke, pytest.mark.slow, pytest.mark.requires_download]

POSITION_TOLERANCE = 0.01
"""Largest end-effector distance from a spawn sample point, in metres."""
TILT_TOLERANCE_DEG = 15.0
"""Largest angle between the approach axis and straight down at a spawn sample point."""
REACH_TOLERANCE = 0.01
"""Largest difference between a profile's stored reach and the measured one, in metres."""
REACH_POSITION_TOLERANCE = 0.005
REACH_TILT_DEG = TILT_TOLERANCE_DEG
"""What counts as reached while measuring ``RobotProfile.reach``: the test's own tilt tolerance."""
REACH_BISECTIONS = 10

GOOD_TILT_DEG = 1.0
"""A start pose that reaches a spawn point this close to straight down ends the search there."""
IK_ITERATIONS = 200
IK_RANDOM_SEEDS = 10
POSITION_DAMPING = 1e-3
TILT_DAMPING = 1e-2
MAX_STEP_RAD = 0.2
DOWN = np.array([0.0, 0.0, -1.0])


@dataclass
class Arm:
    """One arm of a composed scene: its end effector, joints and IK start poses."""

    name: str
    model: mujoco.MjModel
    data: mujoco.MjData
    effector: EndEffector
    site: int | None
    body: int
    qpos_adr: np.ndarray
    dof_adr: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    seeds: list[np.ndarray]
    base_xy: np.ndarray

    def tool(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Put the arm at *q* and return the end effector's point and approach axis in the world.

        Returns:
            The tool point and the unit approach axis.
        """
        self.data.qpos[self.qpos_adr] = q
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        axis = np.asarray(self.effector.axis, dtype=float) / np.linalg.norm(self.effector.axis)
        if self.site is not None:
            rotation = self.data.site_xmat[self.site].reshape(3, 3)
            return self.data.site_xpos[self.site].copy(), rotation @ axis
        rotation = self.data.xmat[self.body].reshape(3, 3)
        return self.data.xpos[self.body] + rotation @ np.asarray(self.effector.pos), rotation @ axis


def reach_profiles() -> list[str]:
    """Return every profile that a ``layout="reach"`` scene lists by name, in registry order."""
    names = (
        name
        for scene in list_scenes().values()
        if scene.layout == "reach" and scene.robots != "*"
        for name in scene.robots
    )
    return list(dict.fromkeys(names))


def reach_scene(profile: RobotProfile) -> SceneConfig:
    """Return the first reach-scaled scene that lists *profile*."""
    return next(scene for scene in list_scenes().values() if scene.layout == "reach" and profile.name in scene.robots)


def reach_cases() -> list[tuple[str, int]]:
    """Return every reach profile with one arm, then those that also run two, with two."""
    names = reach_profiles()
    return [(name, 1) for name in names] + [
        (name, 2) for name in names if 2 in supported_arm_counts(get_profile(name))  # noqa: PLR2004
    ]


def build_arms(scene: SceneConfig, profile: RobotProfile, rng: np.random.Generator, arms: int = 1) -> list[Arm]:
    """Compose *scene* with *arms* of *profile*'s robot and return one :class:`Arm` per robot and end effector.

    Returns:
        The arms, in anchor order and then the profile's end-effector order.
    """
    composed = compose_scene(scene.scene_xml_path, profile, scene_layout=scene.layout_for(profile, arms))
    model = composed.model
    data = mujoco.MjData(model)
    arms = []
    for binding in composed.robots:
        for effector in profile.end_effectors:
            if effector.site is not None:
                site = model.site(f"{binding.prefix}{effector.site}").id
                body = int(model.site_bodyid[site])
            else:
                site, body = None, model.body(f"{binding.prefix}{effector.body}").id
            joints = _chain_joints(model, body)
            lower = np.where(model.jnt_limited[joints], model.jnt_range[joints, 0], -math.pi)
            upper = np.where(model.jnt_limited[joints], model.jnt_range[joints, 1], math.pi)
            qpos_adr = model.jnt_qposadr[joints]
            home = model.qpos0.copy()
            for joint, values in binding.layout.home_qpos.items():
                address = model.joint(joint).qposadr[0]
                home[address : address + len(values)] = values
            seeds = [home[qpos_adr], model.qpos0[qpos_adr].copy()]
            seeds += [rng.uniform(lower, upper) for _ in range(IK_RANDOM_SEEDS)]
            root = body
            while model.body_parentid[root] != 0:
                root = int(model.body_parentid[root])
            mujoco.mj_kinematics(model, data)
            arms.append(
                Arm(
                    name=effector.site or effector.body or "",
                    model=model,
                    data=data,
                    effector=effector,
                    site=site,
                    body=body,
                    qpos_adr=qpos_adr,
                    dof_adr=model.jnt_dofadr[joints],
                    lower=lower,
                    upper=upper,
                    seeds=[np.clip(seed, lower, upper) for seed in seeds],
                    base_xy=data.xpos[root][:2].copy(),
                ),
            )
    return arms


def _chain_joints(model: mujoco.MjModel, body: int) -> np.ndarray:
    """Return the hinge and slide joints from the root down to *body*: the arm, without the fingers."""
    joints: list[int] = []
    while body > 0:
        start, count = model.body_jntadr[body], model.body_jntnum[body]
        joints[:0] = [
            j
            for j in range(start, start + count)
            if int(model.jnt_type[j]) in {int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)}
        ]
        body = int(model.body_parentid[body])
    return np.asarray(joints, dtype=int)


def solve(arm: Arm, target: np.ndarray, *, good_tilt_deg: float) -> tuple[float, float]:
    """Reach *target* with the approach axis as close to straight down as the arm allows.

    Damped least squares on the position, and on the tilt within the position's null space; the
    tilt rows drop the rotation about the approach axis, which the task leaves free. A joint at a
    limit that a step would push further drops out of the step (an active set). Near the edge of
    the workspace the tilt task can hold the position back, so a position-only pass ends each try.
    Every start pose runs until one reaches the target within 1 mm and *good_tilt_deg*.

    Returns:
        The best position error in metres and its tilt in degrees.
    """
    best = (math.inf, math.inf)
    for seed in arm.seeds:
        q = seed.copy()
        for _ in range(IK_ITERATIONS):
            dq = _step(arm, q, target, with_tilt=True)
            q = np.clip(q + dq, arm.lower, arm.upper)
            if float(np.max(np.abs(dq))) < 1e-7:  # noqa: PLR2004
                break
        for _ in range(IK_ITERATIONS):
            dq = _step(arm, q, target, with_tilt=False)
            q = np.clip(q + dq, arm.lower, arm.upper)
            if float(np.max(np.abs(dq))) < 1e-7:  # noqa: PLR2004
                break
        point, axis = arm.tool(q)
        result = (float(np.linalg.norm(target - point)), math.degrees(math.acos(np.clip(axis @ DOWN, -1.0, 1.0))))
        if _better(result, best):
            best = result
        if best[0] <= 1e-3 and best[1] <= good_tilt_deg:  # noqa: PLR2004
            break
    return best


def _step(arm: Arm, q: np.ndarray, target: np.ndarray, *, with_tilt: bool) -> np.ndarray:
    """Return one damped least-squares step toward *target*, tilting the approach axis down if *with_tilt*."""
    model, data = arm.model, arm.data
    jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    point, axis = arm.tool(q)
    position_error = target - point
    cross = np.cross(axis, DOWN)
    sine = float(np.linalg.norm(cross))
    tilt_error = cross / sine * math.atan2(sine, float(axis @ DOWN)) if sine > 1e-12 else np.zeros(3)  # noqa: PLR2004
    mujoco.mj_jac(model, data, jacp, jacr, point, arm.body)
    position_jac = jacp[:, arm.dof_adr]
    tilt_jac = (np.eye(3) - np.outer(axis, axis)) @ jacr[:, arm.dof_adr]
    free = np.ones(len(q), dtype=bool)
    dq = np.zeros(len(q))
    for _ in range(len(q)):
        jp, jr = position_jac * free, tilt_jac * free
        jp_pinv = jp.T @ np.linalg.inv(jp @ jp.T + POSITION_DAMPING**2 * np.eye(3))
        dq = jp_pinv @ position_error
        if with_tilt:
            null = np.eye(len(q)) - jp_pinv @ jp
            jr_null = jr @ null
            dq += null @ (jr_null.T @ np.linalg.solve(jr_null @ jr_null.T + TILT_DAMPING * np.eye(3), tilt_error - jr @ dq))
        dq *= free
        blocked = free & (((q >= arm.upper - 1e-9) & (dq > 0)) | ((q <= arm.lower + 1e-9) & (dq < 0)))
        if not blocked.any():
            break
        free &= ~blocked
    largest = float(np.max(np.abs(dq)))
    return dq * (MAX_STEP_RAD / largest) if largest > MAX_STEP_RAD else dq


def _better(result: tuple[float, float], best: tuple[float, float]) -> bool:
    """Prefer a result within the position tolerance with less tilt, else the smaller position error."""
    reached, best_reached = result[0] <= POSITION_TOLERANCE, best[0] <= POSITION_TOLERANCE
    if reached != best_reached:
        return reached
    return result[1] < best[1] if reached else result[0] < best[0]


def spawn_points(scene: SceneConfig) -> dict[str, np.ndarray]:
    """Return sample points of a laid-out scene's spawn arc at object height: its middle and its extremes.

    Returns:
        Points by label: ``inner``, ``middle`` and ``outer`` along the arc's axis, and its four corners.
    """
    center = np.asarray(scene.spawn_center)
    half = math.radians(scene.spawn_angle_half_deg)
    middle_r = (scene.spawn_min_r + scene.spawn_max_r) / 2
    polar = {
        "inner": (scene.spawn_min_r, 0.0),
        "middle": (middle_r, 0.0),
        "outer": (scene.spawn_max_r, 0.0),
        "inner+": (scene.spawn_min_r, half),
        "inner-": (scene.spawn_min_r, -half),
        "outer+": (scene.spawn_max_r, half),
        "outer-": (scene.spawn_max_r, -half),
    }
    return {
        label: np.array([*(center + r * np.array([math.cos(theta), math.sin(theta)])), FREEJOINT_SPAWN_Z])
        for label, (r, theta) in polar.items()
    }


def measure_reach(arm: Arm, toward: np.ndarray, stored: float) -> float:
    """Bisect the arm's reach from its base toward *toward*, between half and 1.5 times *stored*.

    Returns:
        The farthest horizontal distance reached within ``REACH_POSITION_TOLERANCE`` and ``REACH_TILT_DEG``.

    Raises:
        AssertionError: If even half the stored reach is out of reach.
    """
    direction = (toward[:2] - arm.base_xy) / np.linalg.norm(toward[:2] - arm.base_xy)

    def reached(distance: float) -> bool:
        target = np.array([*(arm.base_xy + distance * direction), FREEJOINT_SPAWN_Z])
        error, tilt = solve(arm, target, good_tilt_deg=REACH_TILT_DEG)
        return error <= REACH_POSITION_TOLERANCE and tilt <= REACH_TILT_DEG

    low, high = 0.5 * stored, 1.5 * stored
    assert reached(low), f"{arm.name} cannot reach {low:.3f} m from its base from above"
    for _ in range(REACH_BISECTIONS):
        middle = (low + high) / 2
        low, high = (middle, high) if reached(middle) else (low, middle)
    return low


@dataclass
class ReachReport:
    """What one profile's check measured."""

    reaches: list[tuple[str, float]]
    """Measured reach per arm."""
    points: list[tuple[str, str, float, float]]
    """Per sample point: its label, the arm that checked it, the position error and the tilt."""


AXIS_POINTS = ("inner", "middle", "outer")
"""Sample points on the spawn arc's axis, between two side-by-side arms: both must reach them."""


def check_profile(name: str, arms: int = 1) -> ReachReport:
    """Measure one profile's reach and solve every sample point of its scaled spawn arc with *arms* arms.

    Returns:
        The measurements; with two arms, no reach (it is the one-arm measurement).
    """
    profile = get_profile(name)
    scene = reach_scene(profile)
    laid_out = scene.for_profile(profile, arms)
    built = build_arms(scene, profile, np.random.default_rng(0), arms)
    points = spawn_points(laid_out)
    assert profile.reach is not None
    reaches = [] if arms > 1 else [(arm.name, measure_reach(arm, points["middle"], profile.reach)) for arm in built]
    solved = []
    for label, point in points.items():
        nearest = min(built, key=lambda candidate: float(np.linalg.norm(point[:2] - candidate.base_xy)))
        for arm in built if arms > 1 and label in AXIS_POINTS else [nearest]:
            error, tilt = solve(arm, point, good_tilt_deg=GOOD_TILT_DEG)
            solved.append((label, f"{arm.name}@{arm.base_xy.round(3).tolist()}", error, tilt))
    return ReachReport(reaches, solved)


@pytest.mark.parametrize(("name", "arms"), reach_cases())
def test_arm_reaches_its_scaled_spawn_area_from_above(name: str, arms: int) -> None:
    profile = get_profile(name)
    assert profile.end_effectors, f"{name} lists no end effector"
    assert profile.reach is not None, f"{name} has no reach"

    report = check_profile(name, arms)

    for arm, measured in report.reaches:
        assert abs(measured - profile.reach) <= REACH_TOLERANCE, (
            f"{name} {arm}: measured reach {measured:.3f} m, stored {profile.reach:.3f} m"
        )
    for label, arm, error, tilt in report.points:
        assert error <= POSITION_TOLERANCE, f"{name} {arm} misses {label} by {error * 1000:.1f} mm"
        assert tilt <= TILT_TOLERANCE_DEG, f"{name} {arm} reaches {label} tilted {tilt:.1f} degrees"


def _main(names: list[str]) -> None:
    for name, arms in reach_cases():
        if names and name not in names:
            continue
        report = check_profile(name, arms)
        stored = get_profile(name).reach
        reaches = ", ".join(f"{arm} {measured:.3f}" for arm, measured in report.reaches)
        print(f"{name} ({arms} arm(s)): reach stored {stored:.3f}, measured {reaches or '-'}")  # noqa: T201
        for label, arm, error, tilt in report.points:
            print(f"    {label:7s} {arm:32s} {error * 1000:6.2f} mm {tilt:5.1f} deg")  # noqa: T201


if __name__ == "__main__":
    _main(sys.argv[1:])
