# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""The bucket_sort scene and its demonstrator on the six-joint arms: the WidowX AI (alone and as a
pair) and the reBot B601 (no display).

``test_bucket_sort_robots.py`` covers the Franka arms; this file checks what the six-joint arms add:
their own layout (closer in, objects and buckets full size), the hand leaning outward to reach the
buckets, their grippers, and whole episodes in both autopilot modes.
"""

from __future__ import annotations

import argparse
from dataclasses import replace

import mujoco
import numpy as np
import pytest

from physicalai_mujoco_plugin import __main__ as cli
from physicalai_mujoco_plugin import bucket_sort_demo as bd
from physicalai_mujoco_plugin.bucket_sort import BUCKET_COLORS, BUCKET_SIZE, OBJECTS, bucket_footprint
from physicalai_mujoco_plugin.bucket_sort_arms import SIX_JOINT_LAYOUT, WXAI_PAIR_LAYOUT, bucket_sort_arms
from physicalai_mujoco_plugin.profiles import get_profile
from physicalai_mujoco_plugin.robot import MuJoCoRobot
from physicalai_mujoco_plugin.scene_registry import get_scene
from physicalai_mujoco_plugin.studio_catalog import list_catalog_entries

TICK_S = 0.02
SETUPS = [("trossen_wxai", False), ("trossen_wxai", True), ("rebot_b601", False)]


@pytest.fixture(scope="module", params=SETUPS, ids=["wxai", "wxai-pair", "rebot"])
def robot(request: pytest.FixtureRequest) -> MuJoCoRobot:
    profile, bimanual = request.param
    sim = MuJoCoRobot(profile, scene="bucket_sort", bimanual=bimanual, cameras=[], http_port=0, seed=3)
    sim.connect()
    yield sim
    sim.disconnect()


@pytest.fixture
def scene(robot: MuJoCoRobot) -> MuJoCoRobot:
    """The robot with the autopilot off and a fresh, seeded layout."""
    robot._automation.set_autopilot("off")
    robot._set_seed(3)
    robot._reset()
    robot._set_seed(None)
    return robot


def _demo(robot: MuJoCoRobot) -> bd.BucketSortDemonstrator:
    demo = robot._automation.autopilot.demonstrator
    assert isinstance(demo, bd.BucketSortDemonstrator)
    return demo


class TestScene:
    @pytest.mark.parametrize(("profile", "bimanual"), SETUPS)
    def test_start_accepts_the_arm_in_bucket_sort(self, profile: str, bimanual: bool) -> None:  # noqa: FBT001
        args = argparse.Namespace(model=None, scene="bucket_sort", bimanual=bimanual)
        assert cli._resolve_scene(args, get_profile(profile)) == ("bucket_sort", 2 if bimanual else 1)

    def test_studio_lists_bucket_sort_for_the_six_joint_followers(self) -> None:
        offered = {
            (entry.profile.name, len(entry.prefixes)): [scene.id for scene in entry.simulation_launch().scenes]
            for entry in list_catalog_entries()
        }
        for profile, bimanual in SETUPS:
            assert "bucket_sort" in offered[profile, 2 if bimanual else 1], profile
        assert "bucket_sort" not in offered["rebot_b601", 2]
        assert "bucket_sort" not in offered["so101", 1]

    def test_the_layout_moves_closer_in_but_keeps_the_sizes(self, scene: MuJoCoRobot) -> None:
        task = scene._sim.on_reset
        arms = len(_demo(scene).arms)
        expected = WXAI_PAIR_LAYOUT if arms == 2 else SIX_JOINT_LAYOUT  # noqa: PLR2004
        assert task.area == expected
        model, data = scene._sim.model, scene._sim.data
        for seed in range(5):
            scene._set_seed(seed)
            scene._reset()
            for name in OBJECTS:
                x, y = data.xpos[model.body(name).id][:2]
                assert expected.object_x[0] <= x <= expected.object_x[1]
                assert expected.object_y[0] <= y <= expected.object_y[1]
            for color in BUCKET_COLORS:
                pose = task.layout.buckets[color]
                corners = bucket_footprint(pose)
                sides = np.linalg.norm(np.diff(np.vstack([corners, corners[:1]]), axis=0), axis=1)
                np.testing.assert_allclose(sorted(sides), sorted([*BUCKET_SIZE[:2], *BUCKET_SIZE[:2]]), atol=1e-9)
        scene._set_seed(None)

    def test_the_pair_stands_on_the_table_edge_side_by_side(self, scene: MuJoCoRobot) -> None:
        demo = _demo(scene)
        if len(demo.arms) == 1:
            pytest.skip("one arm")
        bases = [planner.ik.base_xy for planner in demo.planners]
        np.testing.assert_allclose(bases, WXAI_PAIR_LAYOUT.bases, atol=1e-9)
        # At home each turns away from the other, leaving room for it to reach across.
        left, right = (planner.ik.fk(planner.ik.home)[0] for planner in demo.planners)
        assert left[1] > bases[0][1] and right[1] < bases[1][1]
        # The rear pole camera stands behind the pair's middle, not behind the left arm.
        layout = get_scene("bucket_sort").layout_for(get_profile("trossen_wxai"), 2)
        assert layout is not None and layout.overview_rig is not None
        assert layout.overview_rig.pos == (-0.17, 0.0, 0.90)

    def test_a_reset_puts_every_arm_at_home_clear_of_everything(self, scene: MuJoCoRobot) -> None:
        model, data = scene._sim.model, scene._sim.data
        demo = _demo(scene)
        assert demo.arms == bucket_sort_arms(scene._profile.name, len(scene._sim.bindings))
        mujoco.mj_forward(model, data)
        for planner in demo.planners:
            np.testing.assert_allclose(data.qpos[planner.ik.qpos_adr], planner.ik.home, atol=1e-9)
            assert planner.ik.tcp(data)[2] > 0.15  # noqa: PLR2004 - above the buckets' rims
        robot_bodies = np.isin(np.asarray(model.body_rootid), scene._sim.robot_roots)
        for contact in data.contact[: data.ncon]:
            bodies = model.geom_bodyid[[contact.geom1, contact.geom2]]
            assert not robot_bodies[bodies].any() or robot_bodies[bodies].all(), "an arm touches the scene at home"


class TestDemonstrator:
    def test_the_hand_leans_to_reach_the_buckets_and_grasps_nearly_fingers_down(self, scene: MuJoCoRobot) -> None:
        demo = _demo(scene)
        data = scene._sim.data
        demo.step(data, TICK_S, apply=False)
        task = scene._sim.on_reset
        params = bd.DemoParams.for_heights(demo.active_arm.heights)
        snapshot = bd.SceneSnapshot.take(data)
        if len(demo.arms) > 1:
            plan = demo._plan_with_an_arm(snapshot, task.instruction, params, TICK_S, np.random.default_rng(0))
        else:
            plan = demo.plan_attempt(snapshot, demo._q, demo._start_state(), task.instruction, params, TICK_S)
        assert plan is not None and plan.ok, plan and plan.problems
        tilt = {phase: np.degrees(plan.cartesian[k, 5]) for k, phase in enumerate(plan.phases)}
        carry = [np.degrees(plan.cartesian[k, 5]) for k, phase in enumerate(plan.phases) if phase in {"carry", "lower"}]
        assert max(carry) > 0.0  # leaning out over the buckets
        assert max(carry) <= np.degrees(demo.active_arm.max_tilt) + 1e-6
        assert tilt["close"] <= 15.0  # noqa: PLR2004 - the grasp from nearly straight above

    def test_the_widowx_servos_hold_the_targets_the_leader_publishes(self, scene: MuJoCoRobot) -> None:
        model, data = scene._sim.model, scene._sim.data
        if scene._profile.name != "trossen_wxai":
            assert not model.body_gravcomp.any()  # the WidowX AI's soft servos only
            return
        demo = _demo(scene)
        for _ in range(round(1.0 / TICK_S)):
            scene.get_observation()
        leader = np.asarray(scene._automation.leader_snapshot()["joint_positions"])
        for i, planner in enumerate(demo.planners):
            ik = planner.ik
            # The leader (and so Studio's actions) holds the nominal home, without any gravity offset...
            np.testing.assert_allclose(leader[7 * i : 7 * i + 6], np.degrees(ik.home), atol=1e-9)
            # ... and the arm stands there under gravity: the plant compensates its weight.
            np.testing.assert_allclose(data.qpos[ik.qpos_adr], ik.home, atol=np.radians(0.1))
        scene._automation.set_autopilot("drive")
        for _ in range(round(4.0 / TICK_S)):
            scene.get_observation()
            leader = np.asarray(scene._automation.leader_snapshot()["joint_positions"])
            arm = demo._active
            np.testing.assert_allclose(leader[7 * arm : 7 * arm + 6], np.degrees(demo._q), atol=1e-9)

    def test_a_plan_that_may_run_beats_a_better_scored_one_run_refuses(
        self, scene: MuJoCoRobot, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        demo = _demo(scene)
        data = scene._sim.data
        demo.step(data, TICK_S, apply=False)
        planner, made = demo.planner, []
        solve = planner.checked

        def checked(*args: object, **kwargs: object) -> bd.JointPlan:
            plan = solve(*args, **kwargs)  # type: ignore[arg-type]
            if not made:  # 21 mm off its path: _run refuses it
                plan = replace(plan, pos_error=0.021, problems=["IK residual 21.0 mm"])
            else:  # 19 mm off and 5 mm short of the bucket clearance: it may run
                short = {**plan.clearance, "bucket": bd.CLEARANCE["bucket"] - 0.005}
                plan = replace(plan, pos_error=0.019, clearance=short, problems=["bucket clearance"])
            made.append(plan)
            return plan

        monkeypatch.setattr(planner, "checked", checked)
        params = bd.DemoParams.for_heights(demo.active_arm.heights)
        instruction = scene._sim.on_reset.instruction
        snapshot = bd.SceneSnapshot.take(data)
        plan = demo.plan_attempt(snapshot, demo._q, demo._start_state(), instruction, params, TICK_S, report=False)
        assert bd._severity(made[0]) < bd._severity(made[1])  # the refused one scores better ...
        assert not demo._may_run(made[0])
        assert plan is made[1]  # ... but the one that may run is picked
        demo._run(plan, data)
        assert demo.status()["failure"] is None

    def test_no_lean_short_of_reaching_runs(self, scene: MuJoCoRobot) -> None:
        planner = _demo(scene).planner
        far = np.array([1.2, 0.0, 0.1])
        assert planner.ik.least_tilt(planner.ik.home, far, 0.0) is None
        planner.unreached = False
        assert planner.tilt_at(far, 0.0, grasp=True) == pytest.approx(bd.GRASP_TILT)
        assert planner.unreached

    def test_drive_writes_both_grippers_the_target_the_leader_publishes(self, scene: MuJoCoRobot) -> None:
        model, data = scene._sim.model, scene._sim.data
        scene._automation.set_autopilot("drive")
        for _ in range(round(1.0 / TICK_S)):
            scene.get_observation()
        leader = scene._automation.leader_snapshot()
        assert leader["joint_names"] == list(scene.joint_names)
        if scene._profile.name == "rebot_b601":  # both fingers take the gripper's target
            assert data.ctrl[model.actuator("joint_left").id] == pytest.approx(data.ctrl[model.actuator("joint_right").id])


@pytest.mark.slow
class TestEpisodes:
    """Whole episodes through the real control loop (MuJoCoRobot.get_observation)."""

    @staticmethod
    def _episode(robot: MuJoCoRobot, seed: int, mode: str) -> dict[str, object]:
        robot._set_seed(seed)
        robot._reset()
        robot._set_seed(None)
        model, data = robot._sim.model, robot._sim.data
        warnings = int(data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number)
        robot._automation.set_autopilot(mode)
        demo = _demo(robot)
        demo._rng = np.random.default_rng(seed)
        instruction = robot._sim.on_reset.instruction
        body = model.body(instruction.object).id
        count = robot._http_status()["episode"]["episode_count"]
        succeeded, jump = False, 0.0
        previous = data.xpos[body].copy()
        for _ in range(round(45.0 / TICK_S)):
            robot.get_observation()
            if mode == "leader":
                robot.send_action(np.asarray(robot._automation.leader_snapshot()["joint_positions"], dtype=np.float32))
            episode = robot._http_status()["episode"]
            if episode["episode_count"] != count:
                break
            jump = max(jump, float(np.linalg.norm(data.xpos[body] - previous)))
            previous = data.xpos[body].copy()
            if episode["success"]:
                succeeded = episode["in_bucket"][instruction.object] == instruction.bucket
        return {
            "success": succeeded,
            "outcome": demo.status()["outcome"],
            "failure": demo.status()["failure"],
            "jump": jump,
            "badqacc": int(data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number) - warnings,
        }

    @pytest.mark.parametrize("mode", ["drive", "leader"])
    def test_the_named_object_lands_in_the_named_bucket(self, robot: MuJoCoRobot, mode: str) -> None:
        result = self._episode(robot, 1, mode)
        assert result["success"], result
        assert result["badqacc"] == 0, result
        # Moved by contacts and gravity only: let go 0.16 m up, it falls into the bucket at up to 1.8 m/s,
        # 3.6 cm a tick; nothing teleports it further.
        assert result["jump"] < 0.04, result  # noqa: PLR2004


def test_the_rebot_grasps_lean_15_degrees_at_most() -> None:
    """The review's case (layout and demonstrator seed 36): the grasp once leaned the full 50 degrees."""
    robot = MuJoCoRobot("rebot_b601", scene="bucket_sort", cameras=[], http_port=0, seed=3)
    robot.connect()
    try:
        robot._set_seed(36)
        robot._reset()
        robot._set_seed(None)
        robot._automation.set_autopilot("drive")
        demo = _demo(robot)
        demo._rng = np.random.default_rng(36)
        plans: list[bd.JointPlan] = []
        run = demo._run

        def record(plan: bd.JointPlan, data: object, brake: np.ndarray | None = None) -> None:
            plans.append(plan)
            run(plan, data, brake)

        demo._run = record  # type: ignore[method-assign]
        for _ in range(round(45.0 / TICK_S)):
            robot.get_observation()
            if demo.status()["outcome"] in {"success", "failed", "stalled"}:
                break
        assert demo.status()["outcome"] == "success", demo.status()
        attempts = [plan for plan in plans if plan.kind == "attempt"]
        assert attempts
        for plan in attempts:
            assert not plan.unreached
            grasp = [k for k, phase in enumerate(plan.phases) if phase in {"descend", "close"}]
            assert np.degrees(plan.cartesian[grasp, 5]).max() <= np.degrees(bd.GRASP_TILT) + 1e-6
            assert plan.pos_error <= bd.OUT_OF_REACH
    finally:
        robot.disconnect()
