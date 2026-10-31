# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from physicalai.robot import RobotCalibrationError
from physicalai_lerobot_plugin._calibration import build_device, connect_device, disconnect_device
from physicalai_lerobot_plugin.lerobot_adapter import LeRobotAdapter, LeRobotTeleoperatorAdapter


class _Device:
    def __init__(self, path: Path, *, calibrated: bool = True) -> None:
        self.calibration_fpath = path
        self.calibration = {"joint": object()} if path.is_file() else {}
        self.bus = SimpleNamespace(calibration=self.calibration, is_connected=False, disconnect=Mock())
        self.is_connected = False
        self.is_calibrated = calibrated
        self.action_features = {"joint.pos": float}
        self.observation_features = self.action_features
        self.get_action = Mock(return_value={"joint.pos": 0.0})
        self.get_observation = Mock(return_value={"joint.pos": 0.0})
        self.calibrate = Mock(side_effect=AssertionError("Calibration must not run"))
        self.calibrate_arguments: list[bool] = []

    def connect(self, calibrate: bool = True) -> None:  # noqa: FBT001, FBT002
        self.calibrate_arguments.append(calibrate)
        self.is_connected = True
        self.bus.is_connected = True
        if calibrate and not self.is_calibrated:
            self.calibrate()

    def disconnect(self) -> None:
        self.is_connected = False
        self.bus.is_connected = False


@pytest.fixture
def calibration_path(tmp_path: Path) -> Path:
    path = tmp_path / "leader.json"
    path.write_text('{"joint": {}}')
    return path


@pytest.mark.parametrize("leader", [False, True])
@pytest.mark.parametrize("reason", ["missing", "invalid", "mismatch"])
def test_adapter_preserves_calibration_error_and_never_observes(
    tmp_path: Path,
    calibration_path: Path,
    *,
    leader: bool,
    reason: str,
) -> None:
    path = tmp_path / "absent.json" if reason == "missing" else calibration_path
    device = _Device(path, calibrated=reason != "mismatch")
    if reason == "invalid":
        device.calibration.clear()
    adapter = (
        LeRobotTeleoperatorAdapter("so101_leader", {}, _teleoperator=device)
        if leader
        else LeRobotAdapter("so101_follower", {}, _robot=device)
    )

    with pytest.raises(RobotCalibrationError) as caught:
        adapter.connect()

    assert caught.value.reason == reason
    assert caught.value.calibration_path == str(path)
    assert not device.is_connected
    assert not device.bus.is_connected
    device.get_action.assert_not_called()
    device.get_observation.assert_not_called()
    device.calibrate.assert_not_called()
    assert device.calibrate_arguments == ([False] if reason == "mismatch" else [])


@pytest.mark.parametrize("leader", [False, True])
def test_matching_calibration_connects_without_prompt(calibration_path: Path, *, leader: bool) -> None:
    device = _Device(calibration_path)
    adapter = (
        LeRobotTeleoperatorAdapter("so101_leader", {}, _teleoperator=device)
        if leader
        else LeRobotAdapter("so101_follower", {}, _robot=device)
    )
    adapter.connect()
    adapter.connect()
    assert adapter.is_connected()
    assert adapter.joint_names == ["joint"]
    assert device.calibrate_arguments == [False]
    device.calibrate.assert_not_called()
    adapter.disconnect()


def test_calibration_free_device_uses_argument_free_connect() -> None:
    class Keyboard:
        is_calibrated = True

        def connect(self) -> None:
            self.connected = True

    keyboard = Keyboard()
    connect_device(keyboard)
    assert keyboard.connected


def test_composite_reports_child_path_before_connecting(calibration_path: Path, tmp_path: Path) -> None:
    missing = tmp_path / "right.json"
    composite = SimpleNamespace(
        left_arm=_Device(calibration_path),
        right_arm=_Device(missing),
        connect=Mock(),
    )
    with pytest.raises(RobotCalibrationError) as caught:
        connect_device(composite)
    assert caught.value.calibration_path == str(missing)
    composite.connect.assert_not_called()


def test_partial_connection_cleanup_continues_after_resource_failure() -> None:
    bus = SimpleNamespace(is_connected=True, disconnect=Mock(side_effect=OSError("failed")))
    camera = SimpleNamespace(is_connected=True, disconnect=Mock())
    arm = SimpleNamespace(is_connected=False, bus=bus, cameras={"camera": camera}, connect=Mock())
    composite = SimpleNamespace(is_connected=False, left_arm=arm)
    disconnect_device(composite)
    bus.disconnect.assert_called_once()
    camera.disconnect.assert_called_once()


def test_malformed_calibration_during_construction_is_typed() -> None:
    factory = Mock(side_effect=json.JSONDecodeError("invalid", "{", 1))
    with pytest.raises(RobotCalibrationError) as caught:
        build_device(factory, None)
    assert caught.value.reason == "invalid"
    assert isinstance(caught.value.__cause__, json.JSONDecodeError)


def test_unrelated_construction_error_is_not_reclassified() -> None:
    error = ValueError("bad device configuration")
    with pytest.raises(ValueError, match="bad device configuration") as caught:
        build_device(Mock(side_effect=error), None)
    assert caught.value is error


def test_so101_uses_lerobots_resolved_calibration_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from lerobot.teleoperators import teleoperator
    from lerobot.teleoperators.so_leader import so_leader
    from lerobot.teleoperators.so_leader.config_so_leader import SO101LeaderConfig

    monkeypatch.setattr(teleoperator, "HF_LEROBOT_CALIBRATION", tmp_path)
    bus_factory = Mock(side_effect=lambda **kwargs: SimpleNamespace(calibration=kwargs["calibration"]))
    monkeypatch.setattr(so_leader, "FeetechMotorsBus", bus_factory)
    device = build_device(so_leader.SOLeader, SO101LeaderConfig(port="fake", id="leader"))
    with pytest.raises(RobotCalibrationError) as caught:
        connect_device(device)
    assert caught.value.reason == "missing"
    assert caught.value.calibration_path == str(tmp_path / "teleoperators" / "so_leader" / "leader.json")


@pytest.mark.parametrize("contents", ["{", '{"joint": {"unknown": 1}}'])
def test_so101_invalid_file_is_reported_during_construction(tmp_path: Path, contents: str) -> None:
    from lerobot.teleoperators.so_leader.config_so_leader import SO101LeaderConfig
    from lerobot.teleoperators.so_leader.so_leader import SOLeader

    (tmp_path / "leader.json").write_text(contents)
    config = SO101LeaderConfig(port="fake", id="leader", calibration_dir=tmp_path)
    with pytest.raises(RobotCalibrationError) as caught:
        build_device(SOLeader, config)
    assert caught.value.reason == "invalid"
