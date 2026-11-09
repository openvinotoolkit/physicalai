# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# ruff:file-ignore[undocumented-public-module, undocumented-public-class, undocumented-public-method, undocumented-public-init, assert, magic-value-comparison]

"""Construction round-trips for robot ``@export_config`` wiring."""

from __future__ import annotations

import json
from typing import Any

import pytest

from physicalai.config import Config
from physicalai.robot import Robot


def _assert_construction_round_trip(robot: object) -> dict[str, Any]:
    assert Config.is_exportable(robot)
    assert isinstance(robot, Robot)
    config = Config.from_instance(robot)
    wire: dict[str, Any] = json.loads(json.dumps(config.to_dict()))
    restored = Config.from_dict(wire).instantiate()
    assert type(restored) is type(robot)
    assert Config.from_instance(restored) == wire
    assert not restored.is_connected()  # type: ignore[union-attr]
    return wire


# ---------------------------------------------------------------------------
# Trossen compatibility aliases
# ---------------------------------------------------------------------------


class TestTrossenCompatibilityAliases:
    def test_legacy_root_class_path_instantiates_plugin_driver(self) -> None:
        pytest.importorskip("trossen_arm", reason="trossen-arm SDK is not installed")

        from physicalai_trossen_plugin import WidowXAI as PluginWidowXAI

        robot = Config("physicalai.robot.WidowXAI", {"ip": "192.168.1.10"}).instantiate()

        assert isinstance(robot, PluginWidowXAI)
        assert Config.from_instance(robot).to_dict()["class_path"] == "physicalai_trossen_plugin.WidowXAI"

    def test_legacy_trossen_package_exports_plugin_driver(self) -> None:
        pytest.importorskip("trossen_arm", reason="trossen-arm SDK is not installed")

        from physicalai.robot.trossen import WidowXAI as LegacyWidowXAI
        from physicalai_trossen_plugin import WidowXAI as PluginWidowXAI

        assert LegacyWidowXAI is PluginWidowXAI


def test_missing_robot_plugin_names_the_install_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    import physicalai.robot.plugins as plugins

    def missing_plugin(_: str) -> object:
        raise ModuleNotFoundError(name="physicalai_so101_plugin")

    monkeypatch.setattr(plugins, "import_module", missing_plugin)

    with pytest.raises(ModuleNotFoundError, match=r"physicalai\[so101\]"):
        plugins.load_robot_plugin("physicalai_so101_plugin", "so101")


def test_robot_plugin_rejects_unapproved_module() -> None:
    import physicalai.robot.plugins as plugins

    with pytest.raises(ValueError, match="Unsupported robot plugin module"):
        plugins.load_robot_plugin("untrusted.plugin", "so101")


# ---------------------------------------------------------------------------
# SharedRobot (construction recipe only — no session / connection state)
# ---------------------------------------------------------------------------


class TestSharedRobotConfig:
    def test_spawn_recipe_round_trip(self) -> None:
        from physicalai.robot import SharedRobot

        robot = SharedRobot(
            "follower-arm",
            robot={
                "class_path": "tests.unit.robot.transport.fake.FakeRobot",
                "init_args": {"port": "/dev/fake-shared", "device_ids": ["fake:follower"]},
            },
            allow_remote=True,
            rate_hz=50.0,
            idle_timeout=2.5,
            connect_timeout=3.0,
        )
        wire = _assert_construction_round_trip(robot)
        assert wire["class_path"] == "physicalai.robot.SharedRobot"
        assert wire["init_args"]["name"] == "follower-arm"
        assert wire["init_args"]["allow_remote"] is True
        assert wire["init_args"]["rate_hz"] == 50.0
        assert wire["init_args"]["idle_timeout"] == 2.5
        assert wire["init_args"]["connect_timeout"] == 3.0
        nested = wire["init_args"]["robot"]
        assert isinstance(nested, dict)
        assert nested["class_path"] == "tests.unit.robot.transport.fake.FakeRobot"
        assert nested["init_args"]["port"] == "/dev/fake-shared"
        assert nested["init_args"]["device_ids"] == ["fake:follower"]
        # Run-state / transport handles are never part of the recipe.
        assert "_session" not in wire["init_args"]
        assert "_connected" not in wire["init_args"]
        assert "session" not in wire["init_args"]

    def test_defining_module_robot_path_round_trips_as_given(self) -> None:
        from physicalai.robot import SharedRobot

        defining = "physicalai_so101_plugin.SO101"
        robot = SharedRobot(
            "canonical-arm",
            robot={
                "class_path": defining,
                "init_args": {"port": "/dev/ttyUSB0"},
            },
        )
        wire = _assert_construction_round_trip(robot)
        nested = wire["init_args"]["robot"]
        assert isinstance(nested, dict)
        # Stored as written: the subscriber never imports the driver to canonicalize it.
        assert nested["class_path"] == defining

    def test_attach_only_round_trip(self) -> None:
        from physicalai.robot import SharedRobot

        robot = SharedRobot("leader-arm")
        wire = _assert_construction_round_trip(robot)
        assert wire["class_path"] == "physicalai.robot.SharedRobot"
        assert wire["init_args"] == {"name": "leader-arm"}
        assert "robot" not in wire["init_args"]

    def test_explicit_robot_none_round_trips(self) -> None:
        from physicalai.robot import SharedRobot

        robot = SharedRobot("attach-null", robot=None)
        wire = _assert_construction_round_trip(robot)
        assert wire["init_args"]["name"] == "attach-null"
        assert wire["init_args"]["robot"] is None

    def test_live_session_arg_fails_at_from_instance(self) -> None:
        from physicalai.config import ConfigError
        from physicalai.robot import SharedRobot

        robot = SharedRobot("sess", _session=object())
        with pytest.raises(ConfigError, match=r"init_args\._session"):
            Config.from_instance(robot)

    def test_from_config_omits_null_session(self) -> None:
        from physicalai.robot import SharedRobot

        robot = SharedRobot.from_config(
            {
                "class_path": "tests.unit.robot.transport.fake.FakeRobot",
                "init_args": {"port": "/dev/fake-from-config", "device_ids": ["fake:follower"]},
            },
            name="from-config-arm",
        )
        wire = _assert_construction_round_trip(robot)
        assert "_session" not in wire["init_args"]

    def test_constructor_recipe_omits_null_session(self) -> None:
        from physicalai.robot import SharedRobot

        robot = SharedRobot(
            "from-robot-arm",
            robot={
                "class_path": "tests.unit.robot.transport.fake.FakeRobot",
                "init_args": {"port": "/dev/fake-from-robot", "device_ids": ["fake:follower"]},
            },
        )
        wire = _assert_construction_round_trip(robot)
        assert "_session" not in wire["init_args"]

    def test_attach_omits_null_session(self) -> None:
        from physicalai.robot import SharedRobot

        robot = SharedRobot.attach("attach-arm")
        wire = _assert_construction_round_trip(robot)
        assert "_session" not in wire["init_args"]

    def test_classmethod_live_session_still_fails_at_from_instance(self) -> None:
        from physicalai.config import ConfigError
        from physicalai.robot import SharedRobot

        session = object()
        for robot in (
            SharedRobot.from_config(
                {
                    "class_path": "tests.unit.robot.transport.fake.FakeRobot",
                    "init_args": {"port": "/dev/fake-live", "device_ids": ["fake:follower"]},
                },
                name="live-from-config",
                _session=session,
            ),
            SharedRobot.attach("live-attach", _session=session),
        ):
            with pytest.raises(ConfigError, match=r"init_args\._session"):
                Config.from_instance(robot)

    def test_nested_recipe_is_not_instantiated(self) -> None:
        from physicalai.robot import SharedRobot

        config = Config(
            "physicalai.robot.SharedRobot",
            {
                "name": "nested-arm",
                "robot": {
                    "class_path": "tests.unit.robot.transport.fake.FakeRobot",
                    "init_args": {"port": "/dev/fake-nested", "device_ids": ["fake:follower"]},
                },
            },
        )
        restored = config.instantiate()
        assert isinstance(restored, SharedRobot)
        # The declared config arg stays a mapping — no driver is built here.
        assert restored._robot == config.init_args["robot"]
