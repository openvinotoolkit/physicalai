# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""The bucket_sort demonstrator plans off the control tick (no display).

A plan takes up to seconds (an attempt of a leaning arm, or of two arms). Planned inside the tick,
it stopped the simulation: the virtual leader's ``seq`` stood still, Studio took the leader for
failed after 3 s and held the follower, and the autopilot stalled. These tests drive whole episodes
through the real control loop (``MuJoCoRobot.get_observation``) and check that no tick waits for a
plan, that the leader advances meanwhile, and that a reset discards a plan in flight.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from physicalai_mujoco_plugin import bucket_sort_demo as bd
from physicalai_mujoco_plugin.robot import MuJoCoRobot

TICK_S = 0.02
WORST_TICK_S = 0.15
"""The longest a control tick may take while the demonstrator plans.

A tick takes about 25 ms on a laptop: 1 to 5 ms of physics, and while it plans up to
:data:`~.bucket_sort_demo.PLAN_WAIT_S` (20 ms) waiting for the plan in a loop that runs faster than
real time. 150 ms leaves room for a loaded CI machine yet stays far below the plans themselves (the
reBot's first attempt takes 0.2 to 3 s, an FR3's 0.1 to 0.5 s) and below the 3 s after which
Studio gives up on the leader."""


def _robot(profile: str) -> MuJoCoRobot:
    sim = MuJoCoRobot(profile, scene="bucket_sort", cameras=[], http_port=0, seed=3)
    sim.connect()
    return sim


@pytest.fixture(scope="module")
def fr3() -> MuJoCoRobot:
    sim = _robot("franka_fr3")
    yield sim
    sim.disconnect()


@pytest.fixture
def scene(fr3: MuJoCoRobot) -> MuJoCoRobot:
    """The FR3 with the autopilot off and a fresh, seeded layout."""
    fr3._automation.set_autopilot("off")
    fr3._set_seed(3)
    fr3._reset()
    fr3._set_seed(None)
    return fr3


def _demo(robot: MuJoCoRobot) -> bd.BucketSortDemonstrator:
    demo = robot._automation.autopilot.demonstrator
    assert isinstance(demo, bd.BucketSortDemonstrator)
    return demo


def _tick(robot: MuJoCoRobot) -> float:
    """Run one control tick in ``leader`` mode, the leader's targets sent back as Studio does.

    Returns:
        How long ``get_observation`` took, seconds.
    """
    start = time.perf_counter()
    robot.get_observation()
    elapsed = time.perf_counter() - start
    robot.send_action(np.asarray(robot._automation.leader_snapshot()["joint_positions"], dtype=np.float32))
    return elapsed


@pytest.mark.slow
@pytest.mark.parametrize(
    ("profile", "seeds", "longest_plan_s"),
    [
        # The layouts and demonstrator draws in this order include the reBot's slowest plans
        # measured (seed 10's first attempt took 3 s inside a tick, seed 14's 1.9 s).
        ("rebot_b601", range(11), 1.0),
        ("trossen_wxai", range(4), 0.15),
        ("franka_fr3", range(4), 0.05),
    ],
    ids=["rebot", "wxai", "fr3"],
)
def test_no_tick_waits_for_a_plan(profile: str, seeds: range, longest_plan_s: float) -> None:
    """Whole episodes in ``leader`` mode: the worst tick stays short however long the plans take."""
    robot = _robot(profile)
    try:
        worst, longest = 0.0, 0.0
        for seed in seeds:
            robot._automation.set_autopilot("off")
            robot._set_seed(seed)
            robot._reset()
            robot._set_seed(None)
            robot._automation.set_autopilot("leader")
            count = robot._http_status()["episode"]["episode_count"]
            since = None
            for _ in range(round(45.0 / TICK_S)):
                worst = max(worst, _tick(robot))
                planning = robot._http_status()["autopilot"]["phase"] == "planning"
                if planning and since is None:
                    since = time.perf_counter()
                elif not planning and since is not None:
                    longest, since = max(longest, time.perf_counter() - since), None
                if robot._http_status()["episode"]["episode_count"] != count:
                    break
        assert worst < WORST_TICK_S, f"a tick took {worst * 1000:.0f} ms (plans took up to {longest:.1f} s)"
        # Planned inside a tick, the worst tick was the longest plan.
        assert longest > longest_plan_s, "the episodes did not plan as long as they should test"
    finally:
        robot.disconnect()


def _slow_planner(
    monkeypatch: pytest.MonkeyPatch, demo: bd.BucketSortDemonstrator, release: threading.Event | None = None
) -> list[bd.JointPlan]:
    """Make each attempt plan take 0.5 s on the planning thread (or until *release*).

    Returns:
        The plans made, appended once each is done.
    """
    plan_attempt, made = demo.plan_attempt, []
    checkpoint = getattr(bd, "_checkpoint", lambda: None)

    def slow(*args: object, **kwargs: object) -> bd.JointPlan:
        deadline = time.monotonic() + (10.0 if release is not None else 0.5)
        while time.monotonic() < deadline and not (release is not None and release.is_set()):
            checkpoint()  # as the planner's loops do: a cancelled plan stops here
            time.sleep(0.005)
        plan = plan_attempt(*args, **kwargs)  # type: ignore[arg-type]
        made.append(plan)
        return plan

    monkeypatch.setattr(demo, "plan_attempt", slow)
    return made


def _until_planning(robot: MuJoCoRobot) -> None:
    demo = _demo(robot)
    for _ in range(round(5.0 / TICK_S)):
        _tick(robot)
        if demo.planning:
            return
    pytest.fail(f"the demonstrator never planned (phase {demo.phase})")


def test_the_leader_advances_every_tick_while_it_plans(scene: MuJoCoRobot, monkeypatch: pytest.MonkeyPatch) -> None:
    """A 0.5 s plan: every tick returns at once with the next ``seq``, the arm holding, until the plan runs."""
    demo = _demo(scene)
    made = _slow_planner(monkeypatch, demo)
    scene._automation.set_autopilot("leader")
    _tick(scene)
    held = scene._automation.leader_snapshot()
    seq, planning = int(held["seq"]), 0
    for _ in range(round(5.0 / TICK_S)):
        elapsed = _tick(scene)
        leader = scene._automation.leader_snapshot()
        assert int(leader["seq"]) == seq + 1, "the leader did not advance this tick"
        assert elapsed < WORST_TICK_S, f"a tick took {elapsed * 1000:.0f} ms"
        seq = int(leader["seq"])
        phase = scene._http_status()["autopilot"]["phase"]
        if phase == "planning":
            planning += 1
            # The arm holds where it was: Studio sends the same targets back.
            np.testing.assert_allclose(leader["joint_positions"], held["joint_positions"], atol=1e-9)
        elif made:
            break
    assert planning >= 10  # a 0.5 s plan held the arm for many ticks  # noqa: PLR2004
    assert len(made) == 1 and demo.last_plan is made[0]
    assert demo.phase == "approach" and demo.status()["outcome"] == "running"
    for _ in range(round(2.0 / TICK_S)):  # and then the arm follows the plan, not stalled
        _tick(scene)
    assert demo.status()["outcome"] == "running" and demo.phase != "stalled"


@pytest.mark.parametrize("how", ["reset", "off", "mode"])
def test_a_reset_discards_the_plan_in_flight(scene: MuJoCoRobot, monkeypatch: pytest.MonkeyPatch, how: str) -> None:
    demo = _demo(scene)
    release = threading.Event()
    made = _slow_planner(monkeypatch, demo, release)
    scene._automation.set_autopilot("leader")
    _until_planning(scene)
    episode = demo.status()["episode"]
    job = demo._job
    assert job is not None
    if how == "reset":
        scene._reset()  # the scene's reset resets the autopilot too
    elif how == "off":
        scene._automation.set_autopilot("off")
    else:
        scene._automation.set_autopilot("drive")
    started = time.monotonic()
    job.thread.join(1.0)  # the plan stops at its next checkpoint, cancelled
    assert not job.thread.is_alive() and time.monotonic() - started < 0.5  # noqa: PLR2004
    assert made == []  # it never got as far as a plan
    if how == "off":
        scene._automation.set_autopilot("leader")
    for _ in range(3):
        _tick(scene)
    assert not demo.planning
    assert demo.plans == [] and demo.phase == "settle"  # a fresh episode, from the arm's pose
    assert demo.status()["episode"] == episode + 1
    release.set()


def test_camera_edits_wait_for_a_plan_switched_off_only_until_it_stops(
    scene: MuJoCoRobot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Model edits wait while the planner reads the model; the autopilot switched off, only until its thread stops.

    Nothing steps the demonstrator once the autopilot is off, so the cancelled plan must not keep
    blocking the overview switch and the scene XML camera edits for good.
    """
    demo = _demo(scene)
    watcher = scene._watcher
    assert watcher is not None
    model, data = scene._sim.model, scene._sim.data
    camera = model.camera("overview").id
    shoulder = data.cam_xpos[camera].copy()
    release = threading.Event()
    _slow_planner(monkeypatch, demo, release)
    scene._automation.set_autopilot("leader")
    _until_planning(scene)
    try:
        assert scene._set_overview("front")
        _tick(scene)
        np.testing.assert_array_equal(data.cam_xpos[camera], shoulder)  # the planner reads the model: it waits
        job = demo._job
        scene._automation.set_autopilot("off")
        job.thread.join(1.0)  # cancelled: it stops at its next checkpoint
        assert not job.thread.is_alive()
        for _ in range(3):
            _tick(scene)
        assert not np.allclose(data.cam_xpos[camera], shoulder), "the overview switch never applied"
        applied = []
        apply = watcher.apply
        monkeypatch.setattr(watcher, "apply", lambda *args: applied.append(apply(*args)))
        monkeypatch.setattr(watcher, "_snapshot", lambda: {"scene.xml": 1.0})  # a camera edit on disk
        watcher._next_check = 0.0
        _tick(scene)
        assert applied, "the camera edit never applied"
        assert not demo.planning and demo._job is None and demo.phase == "idle"
    finally:
        release.set()
        monkeypatch.undo()
        scene._set_overview("shoulder")
        _tick(scene)


def test_a_finished_plan_a_reset_overtook_is_dropped(scene: MuJoCoRobot, monkeypatch: pytest.MonkeyPatch) -> None:
    """A plan that finished while nothing collected it (the autopilot off) never runs."""
    demo = _demo(scene)
    release = threading.Event()
    made = _slow_planner(monkeypatch, demo, release)
    scene._automation.set_autopilot("leader")
    _until_planning(scene)
    job = demo._job
    release.set()
    job.thread.join(5.0)
    assert len(made) == 1  # planned, but not collected yet
    scene._automation.set_autopilot("off")
    scene._automation.set_autopilot("leader")
    _tick(scene)
    assert made[0] not in demo.plans and demo.phase == "settle"


def test_the_plan_on_the_thread_is_the_plan_made_in_the_tick(scene: MuJoCoRobot) -> None:
    """Same layout and draws: planning off the tick does not change the plan."""
    demo = _demo(scene)
    plans = []
    for background in (False, True):
        scene._automation.set_autopilot("off")
        scene._set_seed(5)
        scene._reset()
        scene._set_seed(None)
        demo._rng = np.random.default_rng(5)
        demo.background_planning = background
        scene._automation.set_autopilot("drive")
        for _ in range(round(10.0 / TICK_S)):
            scene.get_observation()
            if demo.last_plan is not None and not demo.planning:
                break
        plans.append(demo.last_plan)
    demo.background_planning = True
    inline, threaded = plans
    assert inline is not None and threaded is not None
    # Equal but for rounding: replanning the same layout in the tick differs by up to 1e-11 rad too
    # (the IK's scratch data starts where the last plan left it).
    np.testing.assert_allclose(inline.q, threaded.q, rtol=0.0, atol=1e-9)
    np.testing.assert_allclose(inline.grip, threaded.grip, rtol=0.0, atol=1e-9)


def test_a_replan_brakes_while_it_plans_and_runs_on_from_there(scene: MuJoCoRobot, monkeypatch) -> None:
    """The targets brake to a stop along the plan's own braking: no jump when the plan arrives."""
    demo = _demo(scene)
    scene._automation.set_autopilot("drive")
    for _ in range(round(10.0 / TICK_S)):
        scene.get_observation()
        if demo.phase == "approach" and demo._tick > 20:  # noqa: PLR2004 - moving
            break
    assert demo.phase == "approach"
    _slow_planner(monkeypatch, demo)
    data = scene._sim.data
    speed = float(np.abs(demo._q - demo._q_prev).max())  # rad per tick, as the replan finds the arm
    demo._deferring = True  # as within step(): the plan goes to its thread
    try:
        demo._replan(data, TICK_S, "a test move", attempt=True)
    finally:
        demo._deferring = False
    assert demo.planning and demo.phase == "planning"
    hold = demo._job.hold
    outputs = [hold[0]]
    while demo.planning:
        outputs.append(demo.step(data, TICK_S, apply=False))
    outputs += [demo.step(data, TICK_S, apply=False) for _ in range(5)]
    q = np.array(outputs)[:, :7]
    brake = hold[1:, :7]
    assert len(q) > len(hold) + 5  # it braked to a stop, held, then ran on  # noqa: PLR2004
    np.testing.assert_allclose(q[: len(hold)], hold[:, :7], atol=1e-12)  # the braking, as planned
    np.testing.assert_allclose(q[len(hold) : -6], np.repeat(brake[-1:], len(q) - len(hold) - 6, axis=0), atol=1e-12)
    assert np.abs(np.diff(q, axis=0)).max() <= speed + 1e-6  # never faster than when it braked: no jump
    assert demo.phase == "approach" and demo.last_plan.brake_ticks == len(brake)


def test_a_stuck_start_planned_on_the_thread_matches_one_planned_in_the_tick(scene: MuJoCoRobot) -> None:
    demo = _demo(scene)
    data = scene._sim.data
    inline = demo.stuck_start(data, TICK_S, np.random.default_rng(2))
    demo.begin_stuck_start(data, TICK_S, np.random.default_rng(2))
    for _ in range(round(10.0 / TICK_S)):
        planning, move = demo.poll_stuck_start(TICK_S)
        if not planning:
            break
    assert (move is None) == (inline is None)
    if inline is not None:
        np.testing.assert_array_equal(move.targets, inline.targets)


@pytest.mark.parametrize("background", [True, False])
def test_a_reset_discards_a_stuck_start_not_polled_yet(scene: MuJoCoRobot, background: bool) -> None:  # noqa: FBT001
    """Planned for the layout before the reset, the move must not come back from a later poll."""
    demo = _demo(scene)
    data = scene._sim.data
    demo.background_planning = background
    try:
        demo.begin_stuck_start(data, TICK_S, np.random.default_rng(2))
        job = demo._job
        if job is not None:
            job.thread.join(10.0)
            assert job.done.is_set()  # done, and not polled
        demo.reset()
        assert demo.poll_stuck_start(TICK_S) == (False, None)
    finally:
        demo.background_planning = True
