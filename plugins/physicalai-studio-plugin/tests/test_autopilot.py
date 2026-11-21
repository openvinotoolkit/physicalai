# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""The autopilot protocol: the descriptor, the models and the client."""

from __future__ import annotations

import asyncio
import copy
import json
from typing import Any, cast

import httpx
import pytest
from pydantic import BaseModel, ValidationError

from physicalai_studio_plugin import RobotCatalogDefinition, SimulationLaunch, SimulationScene, autopilot
from physicalai_studio_plugin.autopilot import (
    LeaderFactory,
    AUTOPILOT_PATH,
    AutopilotClient,
    AutopilotInfo,
    AutopilotRefusedError,
    AutopilotSpec,
    AutopilotUnreachableError,
    Episode,
    EpisodeOptions,
    EpisodeRequest,
    loopback_root,
)

# Payloads of protocol 1 as JSON decodes them, with the model each one is.
EXAMPLES: dict[str, tuple[str, dict[str, Any]]] = {
    # GET {path}: a scene with an autopilot, no episode yet.
    "info_supported": (
        "AutopilotInfo",
        {
            "protocol": 1,
            "scene": "bucket_sort",
            "supported": True,
            "reason": None,
            "options": ["max_attempts", "recovery_fraction"],
            "joint_names": ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "gripper"],
            "episode": None,
        },
    ),
    # GET {path}: a scene without a demonstrator.
    "info_unsupported": (
        "AutopilotInfo",
        {
            "protocol": 1,
            "scene": "tabletop_cube",
            "supported": False,
            "reason": "Scene 'tabletop_cube' has no autopilot; automatic recording needs conveyor or bucket_sort.",
            "options": [],
            "joint_names": [],
            "episode": None,
        },
    ),
    # POST {path}/episodes
    "episode_request": ("EpisodeRequest", {"options": {"max_attempts": 3, "recovery_fraction": 0.2}}),
    "episode_preparing": ("Episode", {"id": "e-0007", "phase": "preparing"}),
    "episode_ready": ("Episode", {"id": "e-0007", "phase": "ready", "task": "Put the red cube in the blue bucket."}),
    "episode_running": (
        "Episode",
        {"id": "e-0007", "phase": "running", "task": "Put the red cube in the blue bucket."},
    ),
    "episode_success_regrasp": (
        "Episode",
        {
            "id": "e-0007",
            "phase": "ended",
            "task": "Put the red cube in the blue bucket.",
            "outcome": "success",
            "reason": "Success after a regrasp (the cube slipped).",
            "clean": False,
        },
    ),
    "episode_aborted_by_liveness": (
        "Episode",
        {
            "id": "e-0008",
            "phase": "ended",
            "task": "Sort the items into the bin of their color.",
            "outcome": "aborted",
            "reason": "Physical AI Studio stopped reading the leader.",
        },
    ),
    "error_wrong_phase": ("AutopilotError", {"code": "wrong_phase", "message": "Episode 'e-0007' is not ready."}),
    "error_bad_option": (
        "AutopilotError",
        {"code": "bad_option", "message": "Scene 'conveyor_sort' does not take recovery_fraction."},
    ),
}


def example(name: str) -> dict[str, Any]:
    return copy.deepcopy(EXAMPLES[name][1])


@pytest.mark.parametrize("name", EXAMPLES)
def test_every_example_validates_and_round_trips(name: str) -> None:
    model: type[BaseModel] = getattr(autopilot, EXAMPLES[name][0])
    parsed = model.model_validate(example(name))
    assert model.model_validate_json(parsed.model_dump_json()) == parsed


def test_requests_refuse_unknown_fields_and_answers_ignore_them() -> None:
    with pytest.raises(ValidationError):
        EpisodeRequest.model_validate({"options": {"max_attempts": 2, "seed": 3}})
    with pytest.raises(ValidationError):
        EpisodeOptions.model_validate({"max_atempts": 2})
    episode = Episode.model_validate({**example("episode_running"), "added_later": 1})
    assert episode.phase == "running"
    assert AutopilotInfo.model_validate({**example("info_supported"), "added_later": True}).supported


@pytest.mark.parametrize("options", [{"max_attempts": 0}, {"max_attempts": 6}, {"recovery_fraction": 0.6}])
def test_options_are_bounded(options: dict) -> None:
    with pytest.raises(ValidationError):
        EpisodeOptions.model_validate(options)


def test_the_spec_checks_its_version_and_path_and_a_simulation_carries_it() -> None:
    leader = cast("LeaderFactory[object]", lambda follower_payload, http_url: object())
    spec = AutopilotSpec(leader_for=leader)
    assert (spec.protocol, spec.path) == (1, AUTOPILOT_PATH)
    bads: list[dict[str, Any]] = [{"protocol": 0}, {"path": "autopilot"}, {"path": "/autopilot/"}, {"path": "/a?b"}]
    for bad in bads:
        with pytest.raises(ValueError, match="protocol|path"):
            AutopilotSpec(leader_for=leader, **bad)
    launch = SimulationLaunch[BaseModel](
        scenes=(SimulationScene(id="bucket_sort", display_name="Bucket sort"),),
        default_scene="bucket_sort",
        payload_owner_name=lambda payload: "sim",
        build_argv=lambda scene, owner_name, seed: ["sim", scene],
        autopilot=spec,
    )
    assert launch.autopilot is spec
    assert SimulationLaunch.__dataclass_fields__["autopilot"].default is None
    assert "autopilot" not in RobotCatalogDefinition.__dataclass_fields__


@pytest.mark.parametrize(
    "url", ["http://192.0.2.1:8080", "https://127.0.0.1:8080", "http://127.0.0.1", "http://127.0.0.1:80/x"]
)
def test_the_client_talks_only_to_this_machine(url: str) -> None:
    with pytest.raises(ValueError):  # noqa: PT011 - the messages are for people
        loopback_root(url)
    with pytest.raises(ValueError):  # noqa: PT011
        AutopilotClient(url)


def test_the_client_ignores_proxies_and_does_not_follow_redirects(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HTTP_PROXY", "http://192.0.2.9:3128")
    monkeypatch.setenv("ALL_PROXY", "http://192.0.2.9:3128")

    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(307, headers={"location": "http://192.0.2.1/autopilot/v1"})

    client = AutopilotClient("http://127.0.0.1:8000", transport=httpx.MockTransport(handle))
    assert client._sync._mounts == {}  # noqa: SLF001 - no proxy mounted from the environment
    with pytest.raises(AutopilotRefusedError) as refused:
        client.info()
    assert refused.value.status == 307
    client.close()


def _simulation(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path == AUTOPILOT_PATH:
        return httpx.Response(200, json=example("info_supported"))
    if path == f"{AUTOPILOT_PATH}/episodes":
        assert EpisodeRequest.model_validate(json.loads(request.content)).options.max_attempts == 2
        return httpx.Response(201, json=example("episode_preparing"))
    if path.endswith("/begin"):
        return httpx.Response(409, json={"detail": example("error_wrong_phase")})
    if path.endswith("/abort"):
        return httpx.Response(200, json={**example("episode_aborted_by_liveness"), "id": "e-0007"})
    if path.endswith("/broken"):
        return httpx.Response(200, json={"not": "an episode"})
    return httpx.Response(200, json=example("episode_ready"))


def test_the_client_speaks_the_protocol_both_ways() -> None:
    client = AutopilotClient(
        "http://127.0.0.1:8000",
        transport=httpx.MockTransport(_simulation),
        async_transport=httpx.MockTransport(_simulation),
    )
    assert client.info().supported
    assert client.start_episode(EpisodeOptions(max_attempts=2)).phase == "preparing"
    assert client.episode("e-0007").task
    with pytest.raises(AutopilotRefusedError) as refused:
        client.begin("e-0007")
    assert (refused.value.status, refused.value.code) == (409, "wrong_phase")
    assert client.abort("e-0007").outcome == "aborted"

    async def asynchronously() -> None:
        assert (await client.ainfo()).scene == "bucket_sort"
        assert (await client.astart_episode(EpisodeOptions(max_attempts=2))).id == "e-0007"
        assert (await client.aepisode("e-0007")).phase == "ready"
        with pytest.raises(AutopilotRefusedError):
            await client.abegin("e-0007")
        assert (await client.aabort("e-0007")).phase == "ended"
        with pytest.raises(AutopilotUnreachableError):
            await client._acall("GET", f"{AUTOPILOT_PATH}/episodes/broken", Episode)  # noqa: SLF001
        await client.aclose()

    asyncio.run(asynchronously())


def test_the_client_reports_a_silent_simulation_as_unreachable() -> None:
    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    client = AutopilotClient("http://127.0.0.1:8000", transport=httpx.MockTransport(down))
    with pytest.raises(AutopilotUnreachableError):
        client.info()
    client.close()


@pytest.mark.parametrize("path", ["http://192.0.2.1:9000/autopilot/v1", "//192.0.2.1/autopilot/v1", "autopilot/v1", "/a b"])
def test_the_client_refuses_a_path_that_could_leave_this_machine(path: str) -> None:
    with pytest.raises(ValueError, match="path"):
        AutopilotClient("http://127.0.0.1:8000", path=path)


@pytest.mark.parametrize(("episode_id", "valid"), [("1a2b3c4d-7", True), ("ep_7.x~1", True), ("e#0007", False), ("a/b", False), ("a b", False)])
def test_an_episode_id_is_a_url_safe_path_segment(episode_id: str, valid: bool) -> None:
    payload = {"id": episode_id, "phase": "preparing"}
    if valid:
        assert Episode.model_validate(payload).id == episode_id
    else:
        with pytest.raises(ValidationError):
            Episode.model_validate(payload)
