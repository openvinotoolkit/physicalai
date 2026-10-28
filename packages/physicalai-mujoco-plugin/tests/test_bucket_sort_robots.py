# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""The bucket_sort scene and its demonstrator on the Franka Panda and the FR3 Duo (no display).

``test_bucket_sort_demo.py`` covers the planner on the FR3; this file checks what differs per robot:
the scene's support and home pose, the demonstrator's channel layout and units, the Duo's arm that
holds still, and whole episodes in both autopilot modes.
"""

from __future__ import annotations

import argparse

import mujoco
import numpy as np
import pytest

from physicalai_mujoco_plugin import __main__ as cli
from physicalai_mujoco_plugin import bucket_sort_demo as bd
from physicalai_mujoco_plugin.bucket_sort import OBJECTS
from physicalai_mujoco_plugin.bucket_sort_arms import BUCKET_SORT_ARMS, FRANKA_GRIP_OPEN
from physicalai_mujoco_plugin.profiles import get_profile
from physicalai_mujoco_plugin.robot import MuJoCoRobot
from physicalai_mujoco_plugin.scene_registry import get_scene, list_scenes_naming
from physicalai_mujoco_plugin.studio_catalog import list_catalog_entries

TICK_S = 0.02
NEW_PROFILES = ("franka_panda", "franka_duo")


@pytest.fixture(scope="module", params=NEW_PROFILES)
def robot(request: pytest.FixtureRequest) -> MuJoCoRobot:
    sim = MuJoCoRobot(request.param, scene="bucket_sort", cameras=[], http_port=0, seed=3)
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


def _arm_qpos(robot: MuJoCoRobot, arm: int) -> np.ndarray:
    demo = robot._automation.autopilot.demonstrator
    return np.array(robot._sim.data.qpos[demo.planners[arm].ik.qpos_adr])


class TestScene:
    @pytest.mark.parametrize("profile", NEW_PROFILES)
    def test_start_accepts_the_profile_in_bucket_sort(self, profile: str) -> None:
        args = argparse.Namespace(model=None, scene="bucket_sort", bimanual=False)
        assert cli._resolve_scene(args, get_profile(profile)) == ("bucket_sort", 1)
        assert "bucket_sort" in list_scenes_naming(profile)

    def test_studio_lists_bucket_sort_for_the_franka_followers(self) -> None:
        offered = {entry.profile.name: entry.simulation_launch() for entry in list_catalog_entries()}
        for profile in ("franka_fr3", *NEW_PROFILES):
            assert "bucket_sort" in [scene.id for scene in offered[profile].scenes], profile
        assert "bucket_sort" not in [scene.id for scene in offered["trossen_wxai"].scenes]

    def test_the_widowx_ai_is_not_offered(self) -> None:
        assert not get_scene("bucket_sort").supports(get_profile("trossen_wxai"))

    def test_a_reset_puts_every_arm_at_its_home_fingers_down_and_open(self, scene: MuJoCoRobot) -> None:
        model, data = scene._sim.model, scene._sim.data
        demo = scene._automation.autopilot.demonstrator
        assert demo.arms == BUCKET_SORT_ARMS[scene._profile.name]
        for planner in demo.planners:
            ik = planner.ik
            np.testing.assert_allclose(data.qpos[ik.qpos_adr], ik.home, atol=1e-9)
            np.testing.assert_allclose(data.qpos[ik.finger_adr], FRANKA_GRIP_OPEN / 2, atol=1e-9)
            assert data.xmat[ik.hand].reshape(3, 3)[2, 2] == pytest.approx(-1.0, abs=1e-3)  # fingers down
            assert ik.margin(ik.home) > bd.JOINT_MARGIN
            assert ik.tcp(data)[2] > 0.3  # the hand waits well above the buckets' rims
        mujoco.mj_forward(model, data)
        robot_bodies = np.isin(np.asarray(model.body_rootid), scene._sim.robot_roots)
        for contact in data.contact[: data.ncon]:
            bodies = model.geom_bodyid[[contact.geom1, contact.geom2]]
            assert not robot_bodies[bodies].any() or robot_bodies[bodies].all(), "an arm touches the scene at home"

    def test_no_object_is_attached_to_a_robot(self, scene: MuJoCoRobot) -> None:
        model = scene._sim.model
        objects = {int(model.body(name).id) for name in OBJECTS}
        for eq in range(model.neq):
            assert not {int(model.eq_obj1id[eq]), int(model.eq_obj2id[eq])} & objects


class TestDemonstrator:
    def test_targets_follow_the_channels_and_their_units(self, scene: MuJoCoRobot) -> None:
        demo = scene._automation.autopilot.demonstrator
        data = scene._sim.data
        targets = demo.step(data, TICK_S, apply=False)
        assert len(targets) == len(scene.joint_names) == 8 * len(demo.arms)
        np.testing.assert_allclose(targets, scene._arm_targets(), atol=1e-9)  # at home, the reset's targets
        scene._automation.set_autopilot("drive")
        for _ in range(round(1.0 / TICK_S)):
            scene.get_observation()
        leader = scene._automation.leader_snapshot()
        assert leader["joint_names"] == list(scene.joint_names)
        np.testing.assert_allclose(scene._arm_targets(), demo.step(data, TICK_S, apply=False), atol=0.05)
        if scene._profile.name == "franka_panda":
            # actuator8 takes Menagerie's 0 to 255; its target is the finger tendon, half the opening.
            assert data.ctrl[scene._sim.model.actuator("actuator8").id] == pytest.approx(255.0)
            assert leader["joint_positions"][7] == pytest.approx(FRANKA_GRIP_OPEN / 2)

    def test_the_duo_arm_that_does_not_work_holds_its_home(self, scene: MuJoCoRobot) -> None:
        if scene._profile.name != "franka_duo":
            pytest.skip("one arm")
        scene._automation.set_autopilot("drive")
        demo = scene._automation.autopilot.demonstrator
        for _ in range(round(12.0 / TICK_S)):
            scene.get_observation()
            if demo.phase in {"lift", "carry"}:
                break
        assert demo.phase in {"lift", "carry"}
        idle = 1 - demo._active
        targets = scene._arm_targets()[8 * idle : 8 * idle + 7]
        np.testing.assert_array_equal(targets, demo.planners[idle].ik.home)
        np.testing.assert_allclose(_arm_qpos(scene, idle), demo.planners[idle].ik.home, atol=0.02)

    def test_a_replan_into_the_idle_arm_stops_instead_of_running(self, scene: MuJoCoRobot) -> None:
        """A dragged object re-plans the approach; a plan that reaches into the arm holding still never runs."""
        if scene._profile.name != "franka_duo":
            pytest.skip("one arm")
        model, data = scene._sim.model, scene._sim.data
        scene._automation.set_autopilot("drive")
        demo = scene._automation.autopilot.demonstrator
        for _ in range(round(10.0 / TICK_S)):
            scene.get_observation()
            if demo.phase == "approach":
                break
        assert demo.phase == "approach"
        idle = demo.planners[1 - demo._active].ik
        below = idle.fk(idle.home)[0]
        name = scene._sim.on_reset.instruction.object
        adr = int(model.joint(f"{name}:joint").qposadr[0])
        data.qpos[adr : adr + 3] = (below[0], below[1], 0.02)  # right under the idle arm's hand
        mujoco.mj_forward(model, data)
        arms = [{int(b) for b in range(model.nbody) if model.body(b).name.startswith(side)} for side in ("left_", "right_")]
        episode, failure, resets = demo.status()["episode"], None, scene._sim.on_reset.reset_count
        for _ in range(round(4.0 / TICK_S)):
            scene.get_observation()
            for contact in data.contact[: data.ncon]:
                bodies = {int(model.geom_bodyid[contact.geom1]), int(model.geom_bodyid[contact.geom2])}
                assert not (bodies & arms[0] and bodies & arms[1]), "the arms touched"
            status = demo.status()
            if status["episode"] != episode:
                break  # the failed episode's reset started the next one
            failure = status["failure"]
        assert demo.stats.replans >= 1
        assert failure is not None and "other arm" in failure, failure
        assert scene._sim.on_reset.reset_count == resets + 1  # and the scene reset after the hold

    def test_braking_into_the_idle_arm_stops_on_the_last_target_instead(self, scene: MuJoCoRobot) -> None:
        """Braking extrapolates the targets' speed; when that would reach the idle arm, the arm stops at once."""
        if scene._profile.name != "franka_duo":
            pytest.skip("one arm")
        data = scene._sim.data
        scene._automation.set_autopilot("drive")
        demo = scene._automation.autopilot.demonstrator
        for _ in range(round(10.0 / TICK_S)):
            scene.get_observation()
            if demo.phase == "approach":
                break
        assert demo.phase == "approach"
        planner, ik = demo.planner, demo.ik
        idle = demo.planners[1 - demo._active].ik
        snapshot = bd.SceneSnapshot.take(data)
        # A line in joint space from the current targets toward the idle arm's hand.
        into, _, _ = ik.solve(demo._q, idle.fk(idle.home)[0], idle.home_yaw, iters=60)
        line = [demo._q + s * (into - demo._q) for s in np.linspace(0.0, 1.0, 41)]
        gaps = [planner.arm_clearance(q, snapshot) for q in line]
        clear = max(i for i, gap in enumerate(gaps) if gap >= 0.045 and all(g >= 0.045 for g in gaps[: i + 1]))
        touching = next(i for i, gap in enumerate(gaps) if i > clear and gap < 0.005)
        # Moving along that line fast enough that braking (half the speed times BRAKE_S) ends touching.
        velocity = (line[touching] - line[clear]) / (bd.BRAKE_S / 2.0)
        demo._q, demo._q_prev = line[clear].copy(), line[clear] - velocity * TICK_S
        demo._replan(data, TICK_S, "a test move", attempt=True)
        assert demo.phase == "failed"
        assert "braking" in demo.status()["failure"]
        assert planner.arm_clearance(demo._plan.q, snapshot) >= bd.CLEARANCE["arm"]
        np.testing.assert_array_equal(demo._plan.q[-1], line[clear])

    def test_a_plan_that_may_run_beats_one_that_scores_better_but_may_not(self, scene: MuJoCoRobot, monkeypatch) -> None:
        """29 mm from the other arm scores 1, a 5 mm miss of the path 2: the miss may run, so it is the fallback."""
        if scene._profile.name != "franka_duo":
            pytest.skip("one arm")
        data = scene._sim.data
        demo = scene._automation.autopilot.demonstrator
        demo.reset()
        demo.step(data, TICK_S, apply=False)

        def fake(pos_error: float = 0.0, arm: float = 0.05) -> bd.JointPlan:
            plan = bd.JointPlan(
                kind="attempt", dt=TICK_S, q=np.zeros((2, 7)), grip=np.zeros(2), phases=["approach"] * 2,
                cartesian=np.zeros((2, 5)),
            )
            plan.pos_error, plan.clearance, plan.problems = pos_error, {"arm": arm}, ["a check"]
            return plan

        close, missing = fake(arm=0.029), fake(pos_error=0.005)
        assert bd._severity(close) < bd._severity(missing)
        snapshot, instruction = bd.SceneSnapshot.take(data), scene._sim.on_reset.instruction
        # Within one arm's variants ...
        plans = iter([close, missing, *(fake(pos_error=0.5) for _ in range(20))])
        monkeypatch.setattr(demo.planner, "checked", lambda *_args, **_kwargs: next(plans))
        best = demo.plan_attempt(snapshot, demo._q, demo._start_state(), instruction, bd.DemoParams(), TICK_S)
        assert best is missing
        # ... and across the arms.
        by_arm = {0: close, 1: missing}
        monkeypatch.setattr(demo, "plan_attempt", lambda *_args, **_kwargs: by_arm[demo._active])
        chosen = demo._plan_with_an_arm(snapshot, instruction, bd.DemoParams(), TICK_S, np.random.default_rng(0))
        assert chosen is missing and demo._active == 1

    def test_an_arm_the_probe_misses_still_plans_with_the_other_yaw_or_a_turned_release(self, robot: MuJoCoRobot) -> None:
        """Seed 1019: the probe misses both arms, yet the left arm's plan passes every check."""
        if robot._profile.name != "franka_duo":
            pytest.skip("one arm")
        robot._automation.set_autopilot("off")
        robot._set_seed(1019)
        robot._reset()
        robot._set_seed(None)
        data = robot._sim.data
        demo = robot._automation.autopilot.demonstrator
        demo.reset()
        demo.step(data, TICK_S, apply=False)
        snapshot, instruction = bd.SceneSnapshot.take(data), robot._sim.on_reset.instruction
        params = bd.DemoParams()
        assert all(demo._arm_preference(arm, snapshot, instruction, params)[0] for arm in range(2))
        plan = demo._plan_with_an_arm(snapshot, instruction, params, TICK_S, np.random.default_rng(0))
        assert plan is not None and plan.ok, plan and plan.problems
        assert demo.active_arm.label == "left arm"

    def test_an_instruction_no_arm_reaches_fails_the_episode_and_asks_for_a_reset(self, scene: MuJoCoRobot) -> None:
        if scene._profile.name != "franka_duo":
            pytest.skip("one arm")
        model, data = scene._sim.model, scene._sim.data
        name = scene._sim.on_reset.instruction.object
        adr = int(model.joint(f"{name}:joint").qposadr[0])
        data.qpos[adr : adr + 3] = (0.80, 0.0, 0.03)  # the table's far edge, past both arms' reach
        mujoco.mj_forward(model, data)
        scene._automation.set_autopilot("drive")
        demo = scene._automation.autopilot.demonstrator
        for _ in range(round(4.0 / TICK_S)):
            scene.get_observation()
            if demo.status()["outcome"] == "failed":
                break
        assert demo.status()["outcome"] == "failed"
        assert demo.status()["failure"].startswith("no arm reaches")


@pytest.mark.slow
class TestEpisodes:
    """Whole episodes through the real control loop (MuJoCoRobot.get_observation)."""

    @staticmethod
    def _episode(robot: MuJoCoRobot, seed: int, mode: str) -> dict[str, object]:
        """Run one episode from a seeded layout; in ``leader`` mode the leader's targets come back as actions.

        Returns:
            Whether it succeeded, the outcome, the largest jump of the named object in one tick,
            the Duo's idle arm's largest move, and MuJoCo's bad-acceleration warnings.
        """
        robot._set_seed(seed)
        robot._reset()
        robot._set_seed(None)
        model, data = robot._sim.model, robot._sim.data
        warnings = int(data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number)
        robot._automation.set_autopilot(mode)
        demo = robot._automation.autopilot.demonstrator
        demo._rng = np.random.default_rng(seed)
        instruction = robot._sim.on_reset.instruction
        body = model.body(instruction.object).id
        count = robot._http_status()["episode"]["episode_count"]
        succeeded, jump, idle_move = False, 0.0, 0.0
        previous = data.xpos[body].copy()
        homes = [_arm_qpos(robot, arm) for arm in range(len(demo.arms))]
        for _ in range(round(45.0 / TICK_S)):
            robot.get_observation()
            if mode == "leader":
                robot.send_action(np.asarray(robot._automation.leader_snapshot()["joint_positions"], dtype=np.float32))
            episode = robot._http_status()["episode"]
            if episode["episode_count"] != count:
                break  # the success dwell's reset laid out the next episode
            jump = max(jump, float(np.linalg.norm(data.xpos[body] - previous)))
            previous = data.xpos[body].copy()
            for arm in range(len(demo.arms)):
                if arm != demo._active:
                    idle_move = max(idle_move, float(np.abs(_arm_qpos(robot, arm) - homes[arm]).max()))
            if episode["success"]:
                succeeded = episode["in_bucket"][instruction.object] == instruction.bucket
        return {
            "success": succeeded,
            "outcome": demo.status()["outcome"],
            "jump": jump,
            "idle_move": idle_move,
            "badqacc": int(data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number) - warnings,
        }

    @pytest.mark.parametrize("mode", ["drive", "leader"])
    @pytest.mark.parametrize("seed", [1, 3])
    def test_the_named_object_lands_in_the_named_bucket(self, robot: MuJoCoRobot, mode: str, seed: int) -> None:
        result = self._episode(robot, seed, mode)
        assert result["success"], result
        assert result["badqacc"] == 0, result
        assert result["jump"] < 0.03, result  # moved by contacts only: nothing teleports it
        assert result["idle_move"] < 0.02, result  # the Duo's other arm held still
