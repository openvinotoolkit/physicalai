# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""The bucket_sort demonstrator: motion profiles, grasp choice, IK, seeding and its episode lifecycle (no display)."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from physicalai_mujoco_plugin import bucket_sort_demo as bd
from physicalai_mujoco_plugin.bucket_sort_demo import (
    FR3_DDQ_MAX,
    FR3_DQ_MAX,
    HOME_Q,
    BucketSortDemonstrator,
    DemoParams,
    MotionScript,
    SceneSnapshot,
    hand_rotation,
    joint_smoothness,
    min_jerk,
    move_duration,
    wrap_near,
)
from physicalai_mujoco_plugin.http_server import SetAutopilotCommand
from physicalai_mujoco_plugin.robot import MuJoCoRobot

TICK_S = 0.02


@pytest.fixture(scope="module")
def robot() -> MuJoCoRobot:
    sim = MuJoCoRobot("franka_fr3", scene="bucket_sort", cameras=[], http_port=0, seed=3)
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


def _demo(robot: MuJoCoRobot, seed: int = 0) -> BucketSortDemonstrator:
    return BucketSortDemonstrator(robot._sim.model, robot._sim.on_reset, rng=np.random.default_rng(seed))


def _plan(robot: MuJoCoRobot, demo: BucketSortDemonstrator, params: DemoParams) -> bd.JointPlan:
    data = robot._sim.data
    demo._q = np.array(data.qpos[demo.ik.qpos_adr])
    demo._grip = bd.GRIP_OPEN
    return demo.plan_attempt(
        SceneSnapshot.take(data), demo._q, demo._start_state(), robot._sim.on_reset.instruction, params, TICK_S
    )


def _place(robot: MuJoCoRobot, name: str, xy: tuple[float, float], yaw: float, roll: float = 0.0) -> None:
    model, data = robot._sim.model, robot._sim.data
    joint = model.joint(f"{name}:joint")
    adr = int(joint.qposadr[0])
    quat_yaw = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
    quat_roll = np.zeros(4)
    mujoco.mju_axisAngle2Quat(quat_roll, np.eye(3)[bd.object_shape(model, name).long_axis], roll)
    quat = np.zeros(4)
    mujoco.mju_mulQuat(quat, quat_yaw, quat_roll)
    data.qpos[adr : adr + 3] = (*xy, 0.02)
    data.qpos[adr + 3 : adr + 7] = quat
    mujoco.mj_forward(model, data)


class TestMotion:
    def test_min_jerk_starts_and_ends_at_rest(self) -> None:
        s = np.linspace(0.0, 1.0, 101)
        profile = min_jerk(s)
        assert profile[0] == 0.0 and profile[-1] == pytest.approx(1.0)
        velocity = np.gradient(profile, s)
        assert velocity[0] == pytest.approx(0.0, abs=1e-3) and velocity[-1] == pytest.approx(0.0, abs=1e-3)
        assert float(velocity.max()) == pytest.approx(1.875, rel=1e-2)
        assert min_jerk(-1.0) == 0.0 and min_jerk(2.0) == 1.0

    def test_move_duration_keeps_the_peak_speed_and_acceleration(self) -> None:
        assert move_duration(0.3, 0.5) == pytest.approx(1.125)
        assert move_duration(0.01, 0.5, minimum=0.4) == 0.4
        duration = move_duration(0.3, 10.0, peak_accel=1.0)
        assert 5.77 * 0.3 / duration**2 == pytest.approx(1.0)

    def test_blended_moves_round_the_corner_without_a_speed_jump(self) -> None:
        script = MotionScript(np.zeros(5))
        script.add("lift", pos=np.array([0.0, 0.0, 0.1]), duration=1.0)
        script.add("carry", pos=np.array([0.2, 0.0, 0.1]), duration=1.0, blend=0.3)
        assert script.t_end == pytest.approx(1.7)
        t = np.arange(0.0, script.t_end + 1e-9, 0.002)
        state = script.at(t)
        np.testing.assert_allclose(state[-1], [0.2, 0.0, 0.1, 0.0, 0.0], atol=1e-9)
        velocity = np.diff(state[:, :3], axis=0) / 0.002
        assert np.abs(np.diff(velocity, axis=0)).max() < 0.01  # no jump anywhere
        assert velocity[int(1.0 / 0.002), 0] > 0.0  # the carry started before the lift ended
        assert script.phases(np.array([0.5, 0.8, 1.5])) == ["lift", "carry", "carry"]

    def test_pause_and_stretch(self) -> None:
        script = MotionScript(np.zeros(5), slow={"close": 2.0})
        script.add("close", grip=0.08, duration=0.5)
        script.pause("release", 0.25)
        assert script.t_end == pytest.approx(1.25)
        assert script.first_time("release") == pytest.approx(1.0)

    def test_hand_rotation_points_the_fingers_down(self) -> None:
        for yaw in (-2.0, 0.0, 1.3):
            rotation = hand_rotation(yaw)
            np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
            assert np.linalg.det(rotation) == pytest.approx(1.0)
            np.testing.assert_allclose(rotation[:, 2], [0.0, 0.0, -1.0])
            np.testing.assert_allclose(rotation[:, 1], [np.cos(yaw), np.sin(yaw), 0.0])
        assert wrap_near(0.1, 6.0) == pytest.approx(0.1 + 2 * np.pi)
        assert wrap_near(3.0, 0.0, np.pi) == pytest.approx(3.0 - np.pi)


class TestGrasp:
    @pytest.mark.parametrize("name", ["resistor", "battery", "cpu"])
    def test_hand_closes_across_the_short_horizontal_side(self, scene: MuJoCoRobot, name: str) -> None:
        demo = _demo(scene)
        planner = demo.planner
        widths = {"resistor": 0.0346, "battery": 0.026, "cpu": 0.038}
        for yaw in (0.0, 0.7, 2.5):
            _place(scene, name, (0.35, 0.0), yaw)
            planner.load(SceneSnapshot.take(scene._sim.data))
            position, rotation = planner.object_pose(name)
            closing, along, width = planner.grasp_axes(name, rotation)
            assert width == pytest.approx(widths[name], abs=0.002)
            assert abs(float(closing @ along)) < 1e-9
            long_axis = rotation[:, bd.object_shape(scene._sim.model, name).long_axis]
            assert abs(float(closing @ long_axis)) < 1e-6
            point = planner.grasp_point(name, position, rotation, DemoParams())
            assert point[2] >= bd.MIN_TCP_Z
            np.testing.assert_allclose(point[:2], position[:2], atol=1e-9)

    def test_rolled_resistor_is_still_graspable_but_one_on_end_is_not(self, scene: MuJoCoRobot) -> None:
        shape = _demo(scene).planner.shapes["resistor"]
        _place(scene, "resistor", (0.35, 0.0), 0.4, roll=1.0)
        rotation = scene._sim.data.xmat[scene._sim.model.body("resistor").id].reshape(3, 3)
        axes = shape.grasp_axes(rotation)
        assert axes is not None
        assert axes[2] < bd.GRIP_OPEN
        standing = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])  # long axis vertical
        assert shape.grasp_axes(standing) is None

    def test_two_symmetric_yaws_half_a_turn_apart_nearest_first(self, scene: MuJoCoRobot) -> None:
        planner = _demo(scene).planner
        candidates = planner.yaw_candidates(0.3, bd.HOME_YAW, np.array([0.35, 0.0]))
        assert len(candidates) == 2
        assert abs(candidates[1] - candidates[0]) == pytest.approx(np.pi)
        assert abs(candidates[0] - bd.HOME_YAW) <= abs(candidates[1] - bd.HOME_YAW)
        for yaw in candidates:
            q7 = planner.predicted_q7(yaw, np.array([0.35, 0.0]))
            assert planner.ik.lower[6] < q7 < planner.ik.upper[6]


class TestIK:
    def test_home_pose_is_the_ready_pose_fingers_down(self, scene: MuJoCoRobot) -> None:
        ik = _demo(scene).ik
        tcp, yaw = ik.fk(HOME_Q)
        np.testing.assert_allclose(tcp, [0.307, 0.0, 0.486], atol=2e-3)
        assert yaw == pytest.approx(bd.HOME_YAW, abs=1e-3)
        assert ik.continuous_yaw(HOME_Q) == pytest.approx(bd.HOME_YAW, abs=1e-3)

    def test_a_straight_line_is_followed_continuously_inside_the_limits(self, scene: MuJoCoRobot) -> None:
        ik = _demo(scene).ik
        start, end = np.array([0.30, -0.2, 0.25]), np.array([0.66, 0.22, 0.14])
        previous, _, _ = ik.solve(HOME_Q, start, -1.2, iters=60)
        for s in np.linspace(0.0, 1.0, 200):
            q, e_pos, e_rot = ik.solve(previous, start + s * (end - start), -1.2 + s, iters=6)
            assert e_pos < 2e-3 and e_rot < np.radians(1.0)
            assert np.abs(q - previous).max() < 0.06  # no jump to another branch
            assert np.all(q >= ik.lower - 1e-9) and np.all(q <= ik.upper + 1e-9)
            previous = q

    def test_optimize_moves_away_from_the_limits_without_losing_the_task(self, scene: MuJoCoRobot) -> None:
        ik = _demo(scene).ik
        target = np.array([0.30, 0.05, 0.10])
        plain, _, _ = ik.solve(HOME_Q, target, -2.0, iters=40)
        better, e_pos, e_rot = ik.optimize(plain, target, -2.0)
        assert e_pos < 1e-3 and e_rot < 1e-2
        assert ik.margin(better) >= ik.margin(plain) - 1e-6


class TestPlan:
    def test_plan_passes_its_checks_and_matches_the_franka_channels(self, scene: MuJoCoRobot) -> None:
        scene._set_seed(2)
        scene._reset()
        scene._set_seed(None)
        demo = _demo(scene)
        plan = _plan(scene, demo, DemoParams())
        assert plan.ok, plan.problems
        assert plan.q.shape[1] == 7 and len(plan.grip) == len(plan.q)
        assert np.all(plan.grip >= 0.0) and np.all(plan.grip <= bd.GRIP_OPEN)
        smooth = joint_smoothness(plan.q, TICK_S)
        assert np.all(smooth["velocity"] <= bd.SPEED_FRACTION * FR3_DQ_MAX + 1e-9)
        assert np.all(smooth["acceleration"] <= bd.ACCEL_FRACTION * FR3_DDQ_MAX + 1e-9)
        assert plan.limit_margin >= bd.JOINT_MARGIN - 1e-6
        np.testing.assert_allclose(plan.q[-1], HOME_Q, atol=1e-9)
        assert plan.grip[-1] == pytest.approx(bd.GRIP_OPEN)
        order = [p for i, p in enumerate(plan.phases) if i == 0 or plan.phases[i - 1] != p]
        assert order[:2] == ["approach", "descend"] and order[-2:] == ["retreat", "return"]
        assert {"close", "lift", "carry", "lower", "release"} <= set(order)
        assert plan.lift_check is not None and plan.hold_until is not None and plan.carry_start is not None

    def test_the_same_seed_repeats_the_plan_and_another_seed_varies_it(self, scene: MuJoCoRobot) -> None:
        def plan(seed: int) -> bd.JointPlan:
            return _plan(scene, _demo(scene), DemoParams.sample(np.random.default_rng(seed)))

        assert DemoParams.sample(np.random.default_rng(5)) == DemoParams.sample(np.random.default_rng(5))
        first, again, other = plan(5), plan(5), plan(6)
        np.testing.assert_array_equal(first.q, again.q)
        np.testing.assert_array_equal(first.grip, again.grip)
        assert len(other.q) != len(first.q) or not np.allclose(other.q, first.q)

    def test_sampled_parameters_stay_in_range(self) -> None:
        rng = np.random.default_rng(0)
        for _ in range(200):
            p = DemoParams.sample(rng)
            assert 0.20 <= p.carry_z <= 0.27 and 0.12 <= p.release_z <= 0.18
            assert 0.0 <= p.blend <= 0.35 and abs(p.grasp_offset) <= 0.006


class TestLifecycle:
    def test_autopilot_is_available_and_reports_the_instruction(self, scene: MuJoCoRobot) -> None:
        autopilot = scene._automation.autopilot
        assert autopilot.available
        assert isinstance(autopilot.demonstrator, BucketSortDemonstrator)
        off = scene._http_status()["autopilot"]
        assert off["mode"] == "off" and off["instruction"] is None and off["attempt"] is None
        scene._commands.put(SetAutopilotCommand(mode="drive"))
        scene.get_observation()
        status = scene._http_status()["autopilot"]
        assert status["mode"] == "drive"
        assert status["instruction"] == scene._sim.on_reset.instruction.text
        assert status["phase"] == "settle" and status["outcome"] == "running" and status["clean"] is True

    def test_drive_writes_the_targets_the_leader_publishes(self, scene: MuJoCoRobot) -> None:
        scene._automation.set_autopilot("drive")
        for _ in range(round(1.5 / TICK_S)):
            scene.get_observation()
        leader = scene._automation.leader_snapshot()
        assert leader["joint_names"] == list(scene.joint_names)
        assert leader["unit"] == "degrees"
        targets = scene._arm_targets()
        np.testing.assert_allclose(leader["joint_positions"][:7], np.degrees(targets[:7]), atol=1e-6)
        assert leader["joint_positions"][7] == pytest.approx(targets[7])
        hold = np.zeros(8, dtype=np.float32)
        scene.send_action(hold)  # ignored while the autopilot drives
        np.testing.assert_allclose(scene._arm_targets(), targets)

    def test_a_scene_reset_starts_a_new_episode(self, scene: MuJoCoRobot) -> None:
        scene._automation.set_autopilot("drive")
        scene.get_observation()
        first = scene._http_status()["autopilot"]["episode"]
        scene._reset()
        scene.get_observation()
        status = scene._http_status()["autopilot"]
        assert status["episode"] == first + 1
        assert status["phase"] == "settle" and status["instruction"] == scene._sim.on_reset.instruction.text

    def test_leader_mode_without_a_client_stalls_instead_of_looping(self, scene: MuJoCoRobot) -> None:
        scene._automation.set_autopilot("leader")
        before = scene._arm_targets().copy()
        seen = set()
        for _ in range(round(15.0 / TICK_S)):
            scene.get_observation()
            seen.add(scene._http_status()["autopilot"]["phase"])
            if "stalled" in seen:
                break
        np.testing.assert_allclose(scene._arm_targets(), before)  # the autopilot moved nothing itself
        status = scene._http_status()["autopilot"]
        assert status["phase"] == "stalled" and status["outcome"] == "stalled"
        assert not seen & {"lift", "carry"}  # stopped before the grasp check could fail the episode
        for _ in range(50):
            scene.get_observation()
        assert scene._http_status()["autopilot"]["phase"] == "stalled"
        assert scene._http_status()["episode"]["episode_count"] == 0 or not scene._automation.take_reset_request()


@pytest.mark.slow
class TestEpisodes:
    """Whole episodes through the real control loop (MuJoCoRobot.get_observation)."""

    @staticmethod
    def _run(robot: MuJoCoRobot, seconds: float, until: object = None) -> dict[str, object]:
        count = robot._http_status()["episode"]["episode_count"]
        for _ in range(round(seconds / TICK_S)):
            robot.get_observation()
            if robot._http_status()["episode"]["episode_count"] != count:
                break
            if until is not None and until(robot):  # type: ignore[operator]
                break
        return robot._http_status()

    @pytest.mark.parametrize("seed", [1, 2, 3])
    def test_drive_mode_puts_the_named_object_in_the_named_bucket(self, robot: MuJoCoRobot, seed: int) -> None:
        robot._set_seed(seed)
        robot._reset()
        robot._set_seed(None)
        robot._automation.set_autopilot("drive")
        robot._automation.autopilot.demonstrator._rng = np.random.default_rng(seed)
        instruction = robot._sim.on_reset.instruction
        succeeded = []

        def success(r: MuJoCoRobot) -> bool:
            if r._http_status()["episode"]["success"]:
                succeeded.append(dict(r._http_status()["episode"]["in_bucket"]))
            return False

        status = self._run(robot, 40.0, success)
        assert succeeded, status["autopilot"]
        assert succeeded[-1][instruction.object] == instruction.bucket
        assert status["episode"]["episode_count"] >= 1  # the dwell started the next episode

    def test_leader_mode_succeeds_when_the_targets_come_back_as_actions(self, robot: MuJoCoRobot) -> None:
        robot._set_seed(4)
        robot._reset()
        robot._set_seed(None)
        robot._automation.set_autopilot("leader")
        count = robot._http_status()["episode"]["episode_count"]
        for _ in range(round(40.0 / TICK_S)):
            robot.get_observation()
            leader = robot._automation.leader_snapshot()
            robot.send_action(np.asarray(leader["joint_positions"], dtype=np.float32))
            if robot._http_status()["episode"]["episode_count"] != count:
                break
        assert robot._http_status()["episode"]["episode_count"] == count + 1

    @pytest.mark.parametrize("phase", ["lift", "lower"])
    def test_a_bucket_moved_before_the_release_is_planned_for_again(self, robot: MuJoCoRobot, phase: str) -> None:
        robot._set_seed(5)
        robot._reset()
        robot._set_seed(None)
        robot._automation.set_autopilot("drive")
        demo = robot._automation.autopilot.demonstrator
        model, data = robot._sim.model, robot._sim.data
        instruction = robot._sim.on_reset.instruction
        joint = model.joint(f"bucket_{instruction.bucket}:joint").id
        adr, dof = int(model.jnt_qposadr[joint]), int(model.jnt_dofadr[joint])
        moved = False
        succeeded = False
        count = robot._http_status()["episode"]["episode_count"]
        for _ in range(round(40.0 / TICK_S)):
            robot.get_observation()
            if not moved and demo.phase == phase:
                data.qpos[adr] += 0.04  # 4 cm further away while the object rises or is lowered into it
                data.qvel[dof : dof + 6] = 0.0
                moved = True
            succeeded |= robot._http_status()["episode"]["success"]
            if robot._http_status()["episode"]["episode_count"] != count:
                break
        assert moved
        assert demo.stats.replans >= 1
        assert succeeded

    def test_a_lost_object_is_retried_then_failed_and_the_scene_reset(self, robot: MuJoCoRobot) -> None:
        robot._set_seed(6)
        robot._reset()
        robot._set_seed(None)
        robot._automation.set_autopilot("drive")
        demo = robot._automation.autopilot.demonstrator
        model, data = robot._sim.model, robot._sim.data
        name = robot._sim.on_reset.instruction.object
        adr = int(model.joint(f"{name}:joint").qposadr[0])
        first_episode = None
        reasons: list[str] = []
        for _ in range(round(60.0 / TICK_S)):
            robot.get_observation()
            status = robot._http_status()["autopilot"]
            first_episode = first_episode or status["episode"]
            if status["episode"] not in {first_episode, None}:
                break  # the requested reset started the next episode
            if demo.phase == "carry":
                # Snatch the object out of the hand onto a far corner of the table, out of reach.
                data.qpos[adr : adr + 3] = (0.75, 0.85, 0.03)
                dof = int(model.joint(f"{name}:joint").dofadr[0])
                data.qvel[dof : dof + 6] = 0.0
                mujoco.mj_forward(model, data)
            if demo._episode is not None:
                reasons = list(demo._episode.reasons)
        assert reasons and reasons[0] == "the object slipped out of the hand"
        assert demo.stats.failures >= 1
        assert robot._http_status()["autopilot"]["episode"] == first_episode + 1


def test_autopilot_markdown_shows_the_attempt_and_the_outcome() -> None:
    from physicalai_mujoco_plugin.viser_controls import _autopilot_markdown  # noqa: PLC0415

    running = {"mode": "drive", "phase": "carry", "target": "cpu", "attempt": 1, "max_attempts": 2, "outcome": "running"}
    assert _autopilot_markdown(running) == "**Autopilot:** carry (cpu), attempt 1/2"
    failed = {**running, "phase": "failed", "attempt": 2, "outcome": "failed", "failure": "the object slipped"}
    assert _autopilot_markdown(failed) == "**Autopilot:** failed (cpu), attempt 2/2, failed: the object slipped"
