# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from typing import Literal

import pytest
import yaml

from physicalai.config import Config
from physicalai.diagnostics.config_check import inspect_config
from physicalai.diagnostics.export import component_bundle, encode_document, output_format, runtime_document, write_document
from physicalai.diagnostics.models import Inventory


@pytest.mark.parametrize("format", ["yaml", "json"])
def test_runtime_roundtrip_is_passive_and_existing_shape(tmp_path: Path, format: Literal["yaml", "json"]) -> None:
    document = runtime_document(Config("does.not.ImportRobot", {"port": "/dev/ttyUSB0"}), {}, Config("does.not.ImportSource"), fps=30)
    text = encode_document(document, format=format)
    assert yaml.safe_load(text)["runtime"]["action_source"]["class_path"] == "does.not.ImportSource"
    assert yaml.safe_load(text) == document
    path = tmp_path / f"runtime.{format}"
    write_document(path, text)
    assert path.read_text() == text
    with pytest.raises(FileExistsError):
        write_document(path, "replacement")
    assert path.read_text() == text
    write_document(path, "replacement", overwrite=True)
    assert path.read_text() == "replacement"
    assert list(tmp_path.iterdir()) == [path]


def test_component_bundle_is_distinct_and_recipes_reload() -> None:
    bundle = component_bundle({"arm": Config("operator.selected.Robot", {"port": "COM3"})}, {})
    assert bundle["kind"] == "physicalai.components"
    assert "runtime" not in bundle
    robots = bundle["robots"]
    assert isinstance(robots, dict)
    recipe = Config.from_dict(robots["arm"])
    assert recipe.init_args["port"] == "COM3"


def test_output_format_conflicts_and_missing_extension(tmp_path: Path) -> None:
    assert output_format(tmp_path / "config.yml") == "yaml"
    assert output_format(tmp_path / "config", "json") == "json"
    with pytest.raises(ValueError, match="conflicts"):
        output_format(tmp_path / "config.json", "yaml")
    with pytest.raises(ValueError, match="specify"):
        output_format(tmp_path / "config")


def test_config_inspection_never_imports_and_rejects_cycles(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(json.dumps({"runtime": {"robot": {"class_path": "untrusted.import.IsNeverLoaded", "init_args": {"port": "/dev/ttyUSB99"}}}}))
    findings = inspect_config(path, Inventory(platform="linux"))
    assert any(f.check_id == "config.selector" and f.severity == "FAIL" for f in findings)
    path.write_text("x: &x\n  recurse: *x\n")
    with pytest.raises(ValueError, match="cyclic"):
        inspect_config(path, Inventory(platform="linux"))


def test_generated_runtime_constructs_with_fake_robots_without_connecting(monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.unit.cli.test_cli import FakeRobot
    from physicalai.runtime import RobotRuntime

    monkeypatch.setattr(FakeRobot, "connect", lambda self: pytest.fail("construction must not connect"))
    robot = Config("tests.unit.cli.test_cli.FakeRobot", {"port": "fake-follower"})
    leader = Config("tests.unit.cli.test_cli.FakeRobot", {"port": "fake-leader"})
    source = Config("physicalai.runtime.TeleopSource", {"leader": leader.to_dict()})
    document = runtime_document(robot, {}, source, fps=30)
    values = yaml.safe_load(encode_document(document, format="yaml"))["runtime"]
    runtime = Config("physicalai.runtime.RobotRuntime", values).instantiate()
    assert isinstance(runtime, RobotRuntime)


def test_alias_expansion_and_metadata_selectors_are_checked_without_imports(tmp_path: Path) -> None:
    from physicalai.diagnostics.models import Device, Endpoint
    path = tmp_path / "metadata.yaml"
    path.write_text("runtime:\n  cameras:\n    wrist:\n      class_path: physicalai.capture.UVCCamera\n      init_args:\n        device: /dev/video43\n")
    snapshot = Inventory(platform="linux", devices=[Device(id="camera", name="camera", endpoints=[
        Endpoint(id="metadata", path="/dev/video43", kind="video", role="metadata"),
    ])])
    findings = inspect_config(path, snapshot)
    assert any(f.severity == "FAIL" and "metadata" in f.summary for f in findings)
    aliases = ["a0: &a0 [leaf]"]
    aliases.extend(f"a{i}: &a{i} [*a{i-1}, *a{i-1}]" for i in range(1, 16))
    path.write_text("\n".join(aliases))
    with pytest.raises(ValueError, match="node limit"):
        inspect_config(path, snapshot)
