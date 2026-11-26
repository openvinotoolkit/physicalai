"""The autopilot protocol's episode book and routes, in memory (no MuJoCo).

The book is driven the way the simulation thread drives it (``view``, ``mark_ready``, ``latch``, ``end``)
and the way Physical AI Studio does over ``/autopilot/v1`` and ``GET /leader``, with a fake clock for
liveness.
"""

from __future__ import annotations

import queue

import pytest
from fastapi.testclient import TestClient
from physicalai_studio_plugin.autopilot import (
    AUTOPILOT_PATH,
    LEADER_TIMEOUT_S,
    AutopilotError,
    AutopilotInfo,
    Episode,
    EpisodeOptions,
    EpisodeRequest,
)

from physicalai_mujoco_plugin.autopilot import (
    NATIVE_VIEWER_REASON,
    REPLACED_REASON,
    STOPPED_READING_REASON,
    STUDIO_ABORT_REASON,
    AutopilotRefusal,
    EpisodeBook,
    SceneSupport,
)
from physicalai_mujoco_plugin.http_server import build_app

BUCKET_SORT = SceneSupport(
    scene="bucket_sort",
    supported=True,
    reason=None,
    options=("max_attempts", "recovery_fraction"),
    joint_names=("joint1", "gripper"),
)
CONVEYOR = SceneSupport(scene="conveyor_sort", supported=True, reason=None, joint_names=("shoulder_pan",))


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def book(clock: Clock) -> EpisodeBook:
    book = EpisodeBook(clock=clock)
    book.set_scene(BUCKET_SORT)
    return book


def _ready(book: EpisodeBook, options: EpisodeOptions | None = None) -> Episode:
    """POST an episode, and lay it out as the simulation thread would."""
    episode = book.create(EpisodeRequest(options=options or EpisodeOptions()))
    view = book.view()
    assert view is not None
    assert (view.id, view.phase) == (episode.id, "preparing")
    assert book.mark_ready(episode.id, "Put the CPU in the red bucket.")
    return book.get(episode.id)


# ----------------------------------------------------------------------
# The book
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("outcome", "clean", "reason"),
    [
        ("success", True, None),
        ("success", False, "Succeeded after a retry."),
        ("failure", False, "1 wrong, 0 missed"),
        ("stalled", None, "The episode ran out of time (120 s simulated)."),
        ("aborted", None, "The scene did not reset for a new episode."),
    ],
)
def test_an_episode_goes_from_preparing_to_ready_to_running_to_ended(
    book: EpisodeBook, outcome: str, clean: bool | None, reason: str | None
) -> None:
    episode = book.create(EpisodeRequest())
    assert (episode.phase, episode.task, episode.outcome) == ("preparing", None, None)
    with pytest.raises(AutopilotRefusal, match="not ready") as refused:
        book.begin(episode.id)
    assert (refused.value.code, refused.value.status_code) == ("wrong_phase", 409)
    assert book.mark_ready(episode.id, "Put the CPU in the red bucket.")
    assert book.get(episode.id).task == "Put the CPU in the red bucket."
    running = book.begin(episode.id)
    assert running.phase == "running"
    assert book.begin(episode.id) == running  # a retried begin is the same
    assert book.may_step(episode.id)
    assert book.latch(episode.id, outcome, clean=clean, reason=reason)  # type: ignore[arg-type]
    assert not book.latch(episode.id, "failure")  # latched once
    ended = book.get(episode.id)
    assert (ended.phase, ended.outcome, ended.clean, ended.reason) == ("ended", outcome, clean, reason)
    assert not book.may_step(episode.id)
    assert book.abort(episode.id) == ended  # abort is idempotent: an ended episode is returned as is
    with pytest.raises(AutopilotRefusal) as refused:
        book.begin(episode.id)
    assert refused.value.code == "wrong_phase"


def test_studio_aborts_an_episode(book: EpisodeBook) -> None:
    episode = _ready(book)
    aborted = book.abort(episode.id)
    assert (aborted.phase, aborted.outcome, aborted.reason) == ("ended", "aborted", STUDIO_ABORT_REASON)


def test_a_new_post_replaces_the_episode_in_progress(book: EpisodeBook) -> None:
    first = _ready(book)
    second = book.create(EpisodeRequest())
    assert second.id != first.id
    assert second.phase == "preparing"
    with pytest.raises(AutopilotRefusal) as refused:
        book.get(first.id)  # only the current or last episode is kept
    assert (refused.value.code, refused.value.status_code) == ("not_found", 404)
    info = book.info()
    assert info.episode is not None
    assert info.episode.id == second.id


def test_the_replaced_episode_ends_aborted_before_the_new_one_starts(book: EpisodeBook) -> None:
    first = _ready(book)
    seen: list[Episode] = []
    original = book._end_locked  # noqa: SLF001

    def spy(reason: str) -> bool:
        record = book._episode  # noqa: SLF001
        ended = original(reason)
        if ended and record is not None:
            seen.append(record.model())
        return ended

    book._end_locked = spy  # type: ignore[method-assign]  # noqa: SLF001
    book.create(EpisodeRequest())
    assert [(e.id, e.outcome, e.reason) for e in seen] == [(first.id, "aborted", REPLACED_REASON)]


def test_ids_are_chosen_by_the_simulation_and_never_repeat(clock: Clock) -> None:
    ids = set()
    for _ in range(3):
        book = EpisodeBook(clock=clock)  # another robot in the same process
        book.set_scene(CONVEYOR)
        for _ in range(5):
            ids.add(book.create(EpisodeRequest()).id)
    assert len(ids) == 15  # noqa: PLR2004


def test_one_second_without_a_sign_of_life_aborts_the_episode(book: EpisodeBook, clock: Clock) -> None:
    episode = _ready(book)
    book.begin(episode.id)
    for _ in range(30):  # leader reads at a slower rate than Studio's, still alive
        clock.now += 0.5
        assert book.leader_fields() == {"episode_id": episode.id, "phase": "running"}
    clock.now += LEADER_TIMEOUT_S - 0.01
    assert book.view().phase == "running"  # type: ignore[union-attr] - the simulation thread is no sign of life
    clock.now += 0.02
    view = book.view()
    assert view is not None
    assert view.phase == "ended"
    ended = book.get(episode.id)
    assert (ended.outcome, ended.reason) == ("aborted", STOPPED_READING_REASON)


@pytest.mark.parametrize("phase", ["preparing", "ready"])
def test_liveness_covers_preparing_and_ready_and_any_request(book: EpisodeBook, clock: Clock, phase: str) -> None:
    episode = book.create(EpisodeRequest())
    if phase == "ready":
        book.mark_ready(episode.id, "task")
    for _ in range(5):
        clock.now += 0.9
        book.get(episode.id)  # any request under the path is a sign of life
    clock.now += 0.9
    book.info()
    assert book.get(episode.id).phase == phase
    clock.now += 1.5
    assert book.leader_fields() == {"episode_id": episode.id, "phase": "ended"}  # too late: it ended first
    assert book.get(episode.id).reason == STOPPED_READING_REASON


def test_an_ended_episode_is_not_timed_out(book: EpisodeBook, clock: Clock) -> None:
    episode = _ready(book)
    book.begin(episode.id)
    book.latch(episode.id, "success", clean=True)
    clock.now += 60.0
    assert book.get(episode.id).outcome == "success"


def test_a_change_to_the_scene_ends_the_episode_naming_it(book: EpisodeBook) -> None:
    episode = _ready(book)
    assert book.end("The belt speed changed.")
    assert not book.end("The scene was reset.")  # only the first change ends it
    ended = book.get(episode.id)
    assert (ended.phase, ended.outcome, ended.reason) == ("ended", "aborted", "The belt speed changed.")


def test_refusals(book: EpisodeBook) -> None:
    with pytest.raises(AutopilotRefusal, match="does not take") as refused:
        book.set_scene(CONVEYOR)
        book.create(EpisodeRequest(options=EpisodeOptions(recovery_fraction=0.2)))
    assert (refused.value.code, refused.value.status_code) == ("bad_option", 422)
    book.set_scene(SceneSupport(scene="tabletop_cube", supported=False, reason="Scene 'tabletop_cube' has none."))
    with pytest.raises(AutopilotRefusal, match="has none") as refused:
        book.create(EpisodeRequest())
    assert (refused.value.code, refused.value.status_code) == ("unsupported", 409)
    book.set_scene(BUCKET_SORT)
    book.set_busy("A replay is running; stop it first.")
    with pytest.raises(AutopilotRefusal, match="replay") as refused:
        book.create(EpisodeRequest())
    assert refused.value.code == "unsupported"
    book.set_busy(None)
    book.create(EpisodeRequest())
    with pytest.raises(AutopilotRefusal) as refused:
        book.abort("nope")
    assert refused.value.code == "not_found"


def test_the_native_viewer_makes_the_scene_unsupported_and_ends_the_episode(book: EpisodeBook) -> None:
    episode = _ready(book)
    book.set_native_viewer(True)  # noqa: FBT003
    assert book.get(episode.id).reason == NATIVE_VIEWER_REASON
    info = book.info()
    assert (info.supported, info.reason) == (False, NATIVE_VIEWER_REASON)
    with pytest.raises(AutopilotRefusal) as refused:
        book.create(EpisodeRequest())
    assert refused.value.code == "unsupported"
    book.set_native_viewer(False)  # noqa: FBT003
    assert book.info().supported


def test_the_book_answers_in_the_protocols_models(book: EpisodeBook) -> None:
    """The book's answers survive the SDK's JSON round trip."""
    info = book.info()
    assert AutopilotInfo.model_validate_json(info.model_dump_json()) == info
    assert info.options == ["max_attempts", "recovery_fraction"]
    refusal = AutopilotRefusal("Episode 'e' is not ready.", code="wrong_phase")
    assert AutopilotError.model_validate(refusal.detail()).code == "wrong_phase"


# ----------------------------------------------------------------------
# The routes
# ----------------------------------------------------------------------


@pytest.fixture
def served(book: EpisodeBook) -> tuple[TestClient, queue.Queue]:
    commands: queue.Queue = queue.Queue()
    leader = {"seq": 1, "mode": "leader", "joint_names": ["joint1", "gripper"], "joint_positions": [0.1, 0.04]}

    def read_leader() -> dict[str, object]:
        return {**leader, **book.leader_fields()}

    status = {"connected": True, "scene": "bucket_sort", "scenes": ["bucket_sort"], "cameras": []}
    app = build_app(
        service_name="sim",
        buffers={},
        commands=commands,
        get_status=lambda: status,
        get_leader=read_leader,
        autopilot=book,
    )
    return TestClient(app), commands


def test_the_routes_run_an_episode(served: tuple[TestClient, queue.Queue], book: EpisodeBook) -> None:
    client, _ = served
    info = AutopilotInfo.model_validate(client.get(AUTOPILOT_PATH).json())
    assert (info.protocol, info.scene, info.supported, info.episode) == (1, "bucket_sort", True, None)
    assert client.get("/leader").json()["episode_id"] is None
    response = client.post(f"{AUTOPILOT_PATH}/episodes", json={"options": {"max_attempts": 3}})
    assert response.status_code == 201
    episode = Episode.model_validate(response.json())
    assert episode.phase == "preparing"
    assert book.view().options.max_attempts == 3  # type: ignore[union-attr]  # noqa: PLR2004
    sample = client.get("/leader").json()
    assert (sample["episode_id"], sample["phase"]) == (episode.id, "preparing")
    begin = client.post(f"{AUTOPILOT_PATH}/episodes/{episode.id}/begin")
    assert (begin.status_code, begin.json()["detail"]["code"]) == (409, "wrong_phase")
    book.mark_ready(episode.id, "Put the CPU in the red bucket.")
    assert client.get(f"{AUTOPILOT_PATH}/episodes/{episode.id}").json()["phase"] == "ready"
    assert client.post(f"{AUTOPILOT_PATH}/episodes/{episode.id}/begin").json()["phase"] == "running"
    aborted = client.post(f"{AUTOPILOT_PATH}/episodes/{episode.id}/abort")
    assert aborted.status_code == 200
    assert Episode.model_validate(aborted.json()).outcome == "aborted"
    assert client.post(f"{AUTOPILOT_PATH}/episodes/{episode.id}/abort").json() == aborted.json()
    assert client.post(f"{AUTOPILOT_PATH}/episodes", content=b"").status_code == 201  # no body: default options


def test_the_routes_answer_errors_as_autopilot_errors(
    served: tuple[TestClient, queue.Queue], book: EpisodeBook
) -> None:
    client, commands = served
    cases = [
        (client.get(f"{AUTOPILOT_PATH}/episodes/unknown"), 404, "not_found"),
        (client.post(f"{AUTOPILOT_PATH}/episodes/unknown/begin"), 404, "not_found"),
        (client.post(f"{AUTOPILOT_PATH}/episodes/unknown/abort"), 404, "not_found"),
        (client.post(f"{AUTOPILOT_PATH}/episodes", json={"options": {"max_attempts": 9}}), 422, "bad_option"),
        (client.post(f"{AUTOPILOT_PATH}/episodes", json={"options": {"speed": 1}}), 422, "bad_option"),
        (client.post(f"{AUTOPILOT_PATH}/episodes", json={"run_id": "x"}), 422, "bad_option"),
    ]
    book.set_scene(CONVEYOR)
    cases.append((client.post(f"{AUTOPILOT_PATH}/episodes", json={"options": {"max_attempts": 2}}), 422, "bad_option"))
    book.set_scene(SceneSupport(scene="tabletop_cube"))
    cases.append((client.post(f"{AUTOPILOT_PATH}/episodes", json={}), 409, "unsupported"))
    for response, status, code in cases:
        assert response.status_code == status, response.json()
        error = AutopilotError.model_validate(response.json()["detail"])
        assert error.code == code
        assert error.message
    assert commands.empty()  # the protocol never queues a command
    # Outside the path, validation errors keep their usual shape.
    assert isinstance(client.post("/autopilot", json={"mode": "fly"}).json()["detail"], list)


def test_studio_recording_is_gone(served: tuple[TestClient, queue.Queue]) -> None:
    client, _ = served
    assert client.post("/studio/recording", json={"enabled": True}).status_code == 404
    endpoints = client.get("/").json()["endpoints"]
    assert endpoints["autopilot_episodes"] == AUTOPILOT_PATH
    assert "studio_recording" not in endpoints
