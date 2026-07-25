# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# ruff:file-ignore[undocumented-public-module, undocumented-public-class, undocumented-public-method, undocumented-public-init, assert, magic-value-comparison]

"""Construction round-trips for robot ``@export_config`` wiring."""

from __future__ import annotations

import json
import sys
from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from physicalai.config import ComponentConfig, instantiate, is_config_exportable, to_config
from physicalai.robot import Robot

# SO101 imports scservo_sdk at module load; keep unit tests hardware-free.
sys.modules.setdefault("scservo_sdk", MagicMock())

SAMPLE_CALIBRATION: dict[str, Any] = {
    "shoulder_pan": {"id": 1, "drive_mode": 0, "homing_offset": 2048, "range_min": 707, "range_max": 3439},
    "shoulder_lift": {"id": 2, "drive_mode": 1, "homing_offset": 1024, "range_min": 669, "range_max": 3292},
    "elbow_flex": {"id": 3, "drive_mode": 0, "homing_offset": 2048, "range_min": 846, "range_max": 3069},
    "wrist_flex": {"id": 4, "drive_mode": 0, "homing_offset": 2048, "range_min": 956, "range_max": 3311},
    "wrist_roll": {"id": 5, "drive_mode": 0, "homing_offset": 2048, "range_min": 59, "range_max": 3946},
    "gripper": {"id": 6, "drive_mode": 0, "homing_offset": 2048, "range_min": 2026, "range_max": 3074},
}


def _assert_construction_round_trip(robot: object) -> dict[str, Any]:
    assert is_config_exportable(robot)
    assert isinstance(robot, Robot)
    config = to_config(robot)
    wire: dict[str, Any] = json.loads(json.dumps(config))
    restored = instantiate(wire)
    assert type(restored) is type(robot)
    assert to_config(restored) == wire
    assert not restored.is_connected()  # type: ignore[union-attr]
    return wire


# ---------------------------------------------------------------------------
# SO101
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_scservo_sdk() -> Generator[MagicMock, None, None]:
    sdk = MagicMock()
    with patch.dict(sys.modules, {"scservo_sdk": sdk}):
        # Ensure SO101 can be imported under the mock (idempotent if already loaded).
        yield sdk


class TestSO101ComponentConfig:
    def test_dict_calibration_round_trip(self, mock_scservo_sdk: MagicMock) -> None:
        from physicalai.robot import SO101

        robot = SO101(port="/dev/ttyUSB0", calibration=SAMPLE_CALIBRATION)
        wire = _assert_construction_round_trip(robot)
        assert wire["class_path"] == "physicalai.robot.SO101"
        assert wire["init_args"] == {
            "port": "/dev/ttyUSB0",
            "calibration": SAMPLE_CALIBRATION,
        }

    def test_calibration_object_encodes_to_dict(self, mock_scservo_sdk: MagicMock) -> None:
        from physicalai.robot import SO101
        from physicalai.robot.so101 import SO101Calibration

        calibration = SO101Calibration.from_dict(SAMPLE_CALIBRATION)
        robot = SO101(port="/dev/ttyACM0", calibration=calibration)
        wire = _assert_construction_round_trip(robot)
        assert wire["init_args"]["calibration"] == SAMPLE_CALIBRATION

    def test_path_calibration_keeps_relative_string(
        self, mock_scservo_sdk: MagicMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from physicalai.robot import SO101

        calib_path = tmp_path / "calibration.json"
        calib_path.write_text(json.dumps(SAMPLE_CALIBRATION), encoding="utf-8")
        monkeypatch.chdir(tmp_path)

        robot = SO101(port="/dev/ttyUSB0", calibration="calibration.json")
        wire = _assert_construction_round_trip(robot)
        assert wire["init_args"]["calibration"] == "calibration.json"

    def test_path_object_keeps_relative_as_given(
        self, mock_scservo_sdk: MagicMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from physicalai.robot import SO101

        calib_path = tmp_path / "calib.json"
        calib_path.write_text(json.dumps(SAMPLE_CALIBRATION), encoding="utf-8")
        monkeypatch.chdir(tmp_path)

        robot = SO101(port="/dev/ttyUSB0", calibration=Path("calib.json"))
        wire = _assert_construction_round_trip(robot)
        assert wire["init_args"]["calibration"] == "calib.json"

    def test_absolute_path_calibration_stays_absolute(
        self, mock_scservo_sdk: MagicMock, tmp_path: Path
    ) -> None:
        from physicalai.robot import SO101

        calib_path = (tmp_path / "absolute-calib.json").resolve()
        calib_path.write_text(json.dumps(SAMPLE_CALIBRATION), encoding="utf-8")

        robot = SO101(port="/dev/ttyUSB0", calibration=calib_path)
        wire = _assert_construction_round_trip(robot)
        assert wire["init_args"]["calibration"] == str(calib_path)
        assert Path(str(wire["init_args"]["calibration"])).is_absolute()

    def test_uncalibrated_export_round_trips(self, mock_scservo_sdk: MagicMock) -> None:
        from jsonargparse import ArgumentParser

        from physicalai.robot import SO101

        robot = SO101.uncalibrated(port="/dev/ttyUSB0")
        wire = _assert_construction_round_trip(robot)
        assert wire["init_args"]["calibration"] is None
        assert wire["init_args"]["allow_uncalibrated"] is True
        assert wire["init_args"]["unit"] == "ticks"

        # Public allow_uncalibrated + defaulted calibration make the export
        # loadable by jsonargparse (the physicalai run --config path); a
        # required calibration with no default rejects calibration: null.
        parser = ArgumentParser()
        parser.add_class_arguments(SO101, "robot")
        ns = parser.parse_object({"robot": wire["init_args"]})
        assert ns.robot.calibration is None
        assert ns.robot.allow_uncalibrated is True

    def test_protocol_still_holds(self, mock_scservo_sdk: MagicMock) -> None:
        from physicalai.robot import SO101

        robot = SO101(port="/dev/ttyUSB0", calibration=SAMPLE_CALIBRATION)
        assert isinstance(robot, Robot)
        assert robot.to_config() == to_config(robot)  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# WidowXAI / BimanualWidowXAI
# ---------------------------------------------------------------------------


def _make_mock_trossen_arm() -> MagicMock:
    module = MagicMock()
    driver = MagicMock()
    driver.get_is_configured.return_value = True
    module.TrossenArmDriver.return_value = driver
    module.Model.wxai_v0 = MagicMock(name="Model.wxai_v0")
    module.StandardEndEffector.wxai_v0_follower = MagicMock(name="wxai_v0_follower")
    module.StandardEndEffector.wxai_v0_leader = MagicMock(name="wxai_v0_leader")
    module.Mode.position = MagicMock(name="Mode.position")
    module.Mode.external_effort = MagicMock(name="Mode.external_effort")
    return module


@pytest.fixture
def mock_trossen_arm() -> Generator[MagicMock, None, None]:
    """Inject a mock ``trossen_arm`` so WidowX drivers import without the SDK."""
    mock_module = _make_mock_trossen_arm()
    # Drop cached modules so import picks up the mock even if a prior import failed.
    for name in (
        "physicalai.robot.trossen.widowxai",
        "physicalai.robot.trossen.bimanual_widowxai",
        "physicalai.robot.trossen",
    ):
        sys.modules.pop(name, None)
    with patch.dict(sys.modules, {"trossen_arm": mock_module}):
        yield mock_module


class TestWidowXAIComponentConfig:
    def test_default_role_omitted(self, mock_trossen_arm: MagicMock) -> None:
        from physicalai.robot import WidowXAI

        robot = WidowXAI(ip="192.168.1.2")
        wire = _assert_construction_round_trip(robot)
        assert wire["class_path"] == "physicalai.robot.WidowXAI"
        assert wire["init_args"] == {"ip": "192.168.1.2"}

    def test_explicit_role_round_trip(self, mock_trossen_arm: MagicMock) -> None:
        from physicalai.robot import WidowXAI

        robot = WidowXAI(ip="10.0.0.1", role="leader")
        wire = _assert_construction_round_trip(robot)
        assert wire["init_args"] == {"ip": "10.0.0.1", "role": "leader"}

    def test_protocol_still_holds(self, mock_trossen_arm: MagicMock) -> None:
        from physicalai.robot import WidowXAI

        robot = WidowXAI(ip="192.168.1.2")
        assert isinstance(robot, Robot)
        assert robot.to_config() == to_config(robot)  # type: ignore[attr-defined]


class TestBimanualWidowXAIComponentConfig:
    def test_nested_arms_round_trip(self, mock_trossen_arm: MagicMock) -> None:
        from physicalai.robot import BimanualWidowXAI, WidowXAI

        left = WidowXAI(ip="192.168.1.10", role="follower")
        right = WidowXAI(ip="192.168.1.11", role="follower")
        robot = BimanualWidowXAI(left=left, right=right)
        wire = _assert_construction_round_trip(robot)
        assert wire["class_path"] == "physicalai.robot.BimanualWidowXAI"
        left_cfg = wire["init_args"]["left"]
        right_cfg = wire["init_args"]["right"]
        assert isinstance(left_cfg, dict)
        assert isinstance(right_cfg, dict)
        assert left_cfg["class_path"] == "physicalai.robot.WidowXAI"
        assert right_cfg["class_path"] == "physicalai.robot.WidowXAI"
        assert left_cfg["init_args"] == {"ip": "192.168.1.10", "role": "follower"}
        assert right_cfg["init_args"] == {"ip": "192.168.1.11", "role": "follower"}


# ---------------------------------------------------------------------------
# SharedRobot (construction recipe only — no session / connection state)
# ---------------------------------------------------------------------------


class TestSharedRobotComponentConfig:
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

        defining = "physicalai.robot.so101.so101.SO101"
        robot = SharedRobot(
            "canonical-arm",
            robot={
                "class_path": defining,
                "init_args": {
                    "port": "/dev/ttyUSB0",
                    "calibration": SAMPLE_CALIBRATION,
                },
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

    def test_live_session_arg_fails_at_to_config(self) -> None:
        from physicalai.config import ComponentConfigError
        from physicalai.robot import SharedRobot

        robot = SharedRobot("sess", _session=object())
        with pytest.raises(ComponentConfigError, match=r"init_args\._session"):
            to_config(robot)

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

    def test_classmethod_live_session_still_fails_at_to_config(self) -> None:
        from physicalai.config import ComponentConfigError
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
            with pytest.raises(ComponentConfigError, match=r"init_args\._session"):
                to_config(robot)

    def test_nested_recipe_is_not_instantiated(self) -> None:
        from physicalai.config import instantiate
        from physicalai.robot import SharedRobot

        config: ComponentConfig = {
            "class_path": "physicalai.robot.SharedRobot",
            "init_args": {
                "name": "nested-arm",
                "robot": {
                    "class_path": "tests.unit.robot.transport.fake.FakeRobot",
                    "init_args": {"port": "/dev/fake-nested", "device_ids": ["fake:follower"]},
                },
            },
        }
        restored = instantiate(config)
        assert isinstance(restored, SharedRobot)
        # The declared config arg stays a mapping — no driver is built here.
        assert restored._robot == config["init_args"]["robot"]
