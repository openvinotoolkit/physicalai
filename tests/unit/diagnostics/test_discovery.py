# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from pydantic import BaseModel

from physicalai.config import Config
from physicalai.diagnostics.discovery import DiscoverySession, Prompts
from physicalai.diagnostics.models import Alias, Device, Endpoint, Inventory, VideoMode
from physicalai.diagnostics.plugins import Catalog


class Payload(BaseModel):
    connection_string: str
    serial_number: str = ""


def inventory() -> Inventory:
    return Inventory(platform="linux", devices=[
        Device(id="arm", name="Serial candidate", vendor_id="1a86", product_id="7523", endpoints=[
            Endpoint(id="serial", path="/dev/ttyUSB0", kind="serial", aliases=[
                Alias(path="/dev/serial/by-path/arm", binding="port", target="/dev/ttyUSB0"),
            ]),
        ]),
        Device(id="camera", name="Camera", serial="SERIAL", endpoints=[
            Endpoint(id="video", path="/dev/video40", kind="video", role="capture", aliases=[
                Alias(path="/dev/v4l/by-path/wrist", binding="port", target="/dev/video40"),
            ], modes=[VideoMode(width=640, height=480, fps=30, pixel_format="MJPG")]),
        ]),
    ])


def prompts(answers: list[str]) -> Prompts:
    iterator = iter(answers)
    return Prompts(read=lambda _: next(iterator), write=lambda _: None)


def catalog() -> Catalog:
    def export(robot):
        return Config("tests.unit.cli.test_cli.FakeRobot", {"port": robot.payload.connection_string})

    return Catalog(definitions={"test_arm": cast(Any, SimpleNamespace(
        type="test_arm", display_name="Test follower", role="follower", robot_payload=Payload, config_exporter=export,
    ))})


def test_complete_policy_wizard_exports_real_runtime_shape_without_constructing(tmp_path: Path) -> None:
    policy = tmp_path / "policy"
    policy.mkdir()
    (policy / "manifest.json").write_text("{}")
    session = DiscoverySession(inventory(), catalog(), prompts([
        "1", "1", "1", "yes",  # Robot model, endpoint, port binding, startup approval.
        "yes", "wrist", "1", "1", "1", "no",  # One camera, slot and mode.
        "1", "", str(policy), "1", "",  # Policy source, Runtime FPS, export, execution, optional task.
    ]))
    document = session.build(components_only=False)
    runtime = document["runtime"]
    assert isinstance(runtime, dict)
    assert runtime["robot"]["init_args"]["port"] == "/dev/serial/by-path/arm"
    camera = runtime["cameras"]["wrist"]["init_args"]
    assert camera["device"] == "/dev/v4l/by-path/wrist"
    assert camera["backend_options"]["pixel_format"] == "mjpeg"
    assert isinstance(camera["fps"], int)
    assert runtime["action_source"]["class_path"] == "physicalai.runtime.PolicySource"
    for selection in session.selections:
        selection.validate(inventory())
    from physicalai.cli.run import build_parser
    # Parsing signatures must work even though manifest.json is deliberately not
    # a real policy. No InferenceModel is constructed and no device is opened.
    parsed = build_parser().parse_object(document)
    assert parsed.runtime.fps == 30


def test_components_camera_only_and_no_hidden_motion() -> None:
    session = DiscoverySession(inventory(), Catalog(), prompts([
        "no", "yes", "wrist", "1", "1", "1", "no",
    ]))
    document = session.build(components_only=True)
    assert document["kind"] == "physicalai.components"
    assert document["robots"] == {}
    assert "runtime" not in document


def test_revalidation_rejects_replaced_or_missing_device() -> None:
    session = DiscoverySession(inventory(), Catalog(), prompts(["1", "1", "1"]))
    session.camera()
    changed = inventory()
    changed.devices[1].serial = "DIFFERENT"
    with pytest.raises(ValueError, match="Hardware changed"):
        session.selections[0].validate(changed)
    with pytest.raises(ValueError, match="0 endpoints"):
        session.selections[0].validate(Inventory(platform="linux"))


def test_unstable_binding_requires_separate_acceptance() -> None:
    snapshot = inventory()
    snapshot.devices[1].endpoints[0].aliases.clear()
    session = DiscoverySession(snapshot, Catalog(), prompts(["1", "1", "no"]))
    with pytest.raises(ValueError, match="Unstable connection rejected"):
        session.camera()


def test_prompt_rejects_invalid_selection_and_confirmation() -> None:
    ui = prompts(["bad", "0", "3", "2", "maybe", "yes"])
    assert ui.choose("Pick", ["a", "b"]) == 1
    assert ui.confirm("Continue") is True


def test_replug_locator_uses_passive_snapshot_diffs() -> None:
    original = inventory()
    unplugged = inventory()
    unplugged.devices = [unplugged.devices[0]]
    returned = inventory()
    endpoint = returned.devices[1].endpoints[0]
    endpoint.path = "/dev/video44"
    endpoint.aliases[0].target = endpoint.path
    refreshes = iter([unplugged, returned])
    session = DiscoverySession(original, Catalog(), prompts([
        "2", "", "", "1", "1", "1",  # Locator, unplug/replug, endpoint, port binding, mode.
    ]), refresh=lambda: next(refreshes))
    recipe = session.camera()
    assert recipe.init_args["device"] == "/dev/v4l/by-path/wrist"
    session.selections[0].validate(returned)


def test_numeric_settings_reprompt_until_valid() -> None:
    ui = prompts(["oops", "nan", "-1", "1.5", "30"])
    assert ui.positive_number("FPS", "30", integer=True) == 30


def test_refresh_can_discover_a_camera_connected_after_startup() -> None:
    session = DiscoverySession(Inventory(platform="linux"), Catalog(), prompts(["1", "1", "1", "1"]), refresh=inventory)
    assert session.camera().init_args["device"] == "/dev/v4l/by-path/wrist"


def test_policy_paths_stay_absolute_when_loading_from_another_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    policy = tmp_path / "policy"
    policy.mkdir()
    (policy / "manifest.json").write_text("{}")
    monkeypatch.chdir(tmp_path)
    session = DiscoverySession(inventory(), Catalog(), prompts(["policy", "1", ""]))
    recipe = session._policy_source(30)
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.chdir(other)
    model = recipe.init_args["model"]
    assert isinstance(model, dict)
    arguments = model["init_args"]
    assert isinstance(arguments, dict)
    assert arguments["export_dir"] == str(policy)
