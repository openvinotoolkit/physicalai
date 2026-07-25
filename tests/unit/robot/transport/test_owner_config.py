# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import pickle
import subprocess
import sys
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest

from physicalai.config import ComponentConfigError, ComponentImportError
from physicalai.robot.transport._owner import RobotOwner
from physicalai.robot.transport._owner_config import (
    RobotOwnerConfig,
    normalize_robot_class,
    normalize_robot_config,
    validate_owner_config,
)

from .conftest import FAKE_ROBOT_CLASS
from .fake import FakeRobot

# SO101 imports scservo_sdk at module load; keep unit tests hardware-free.
sys.modules.setdefault("scservo_sdk", MagicMock())


def _fake_robot(**init_args: object) -> dict[str, object]:
    return {"class_path": FAKE_ROBOT_CLASS, "init_args": dict(init_args)}


@pytest.fixture
def mock_scservo_sdk() -> Any:
    sdk = MagicMock()
    with patch.dict(sys.modules, {"scservo_sdk": sdk}):
        yield sdk


class _Outer:
    class Inner: ...


class TestNormalizeRobotClass:
    def test_class_object(self) -> None:
        assert normalize_robot_class(FakeRobot) == FAKE_ROBOT_CLASS

    def test_string_path_passes_through_without_importing(self) -> None:
        # No scservo_sdk mock: a string path is trusted and never imported here,
        # so the owner envelope can be built where the driver is not installed.
        defining = "physicalai.robot.so101.so101.SO101"
        assert normalize_robot_class(defining) == defining

    def test_public_path_passes_through(self) -> None:
        assert normalize_robot_class("physicalai.robot.SO101") == "physicalai.robot.SO101"

    def test_nested_qualname(self) -> None:
        path = normalize_robot_class(_Outer.Inner)
        assert path.endswith(".test_owner_config._Outer.Inner")

    def test_local_class_raises(self) -> None:
        class _Local: ...

        with pytest.raises(ValueError, match="local class"):
            normalize_robot_class(_Local)

    def test_non_class_non_string_raises(self) -> None:
        with pytest.raises(TypeError, match="must be a class or a dotted path string"):
            normalize_robot_class(123)  # type: ignore[arg-type]

    @pytest.mark.parametrize("ref", ["", "   ", "NotDotted"])
    def test_non_dotted_string_raises(self, ref: str) -> None:
        with pytest.raises(ValueError, match="must be a nonempty dotted path"):
            normalize_robot_class(ref)


class TestRobotOwnerConfig:
    def test_picklable(self) -> None:
        config = RobotOwnerConfig(name="left-arm", robot=_fake_robot(port="/dev/ttyUSB0"))
        restored = pickle.loads(pickle.dumps(config))
        assert restored == config

    def test_json_roundtrip(self) -> None:
        config = RobotOwnerConfig(name="left-arm", robot=_fake_robot(ip="10.0.0.2"))
        assert RobotOwnerConfig.from_json_dict(json.loads(json.dumps(config.to_json_dict()))) == config

    def test_defaults(self) -> None:
        config = RobotOwnerConfig(name="left-arm", robot=_fake_robot())
        assert config.robot == {"class_path": FAKE_ROBOT_CLASS, "init_args": {}}
        assert config.robot_class == FAKE_ROBOT_CLASS
        assert config.allow_remote is False
        assert config.rate_hz == 100.0
        assert config.idle_timeout == 10.0

    def test_persistent_idle_timeout_json_roundtrip(self) -> None:
        config = RobotOwnerConfig(name="left-arm", robot=_fake_robot(), idle_timeout=None)
        restored = RobotOwnerConfig.from_json_dict(json.loads(json.dumps(config.to_json_dict())))
        assert restored == config
        assert restored.idle_timeout is None

    def test_invalid_name_raises(self) -> None:
        with pytest.raises(ValueError, match="invalid robot name"):
            RobotOwnerConfig(name="left/arm", robot=_fake_robot())

    @pytest.mark.parametrize("rate_hz", [float("nan"), True, "100"])
    def test_invalid_rate_types_raise(self, rate_hz: object) -> None:
        with pytest.raises(ValueError, match="rate_hz must be finite"):
            RobotOwnerConfig(name="left-arm", robot=_fake_robot(), rate_hz=rate_hz)  # type: ignore[arg-type]

    @pytest.mark.parametrize("idle_timeout", [0.0, -1.0, float("inf"), float("nan"), True, "10"])
    def test_invalid_idle_timeout_raises(self, idle_timeout: object) -> None:
        with pytest.raises(ValueError, match="idle_timeout must be finite"):
            RobotOwnerConfig(
                name="left-arm",
                robot=_fake_robot(),
                idle_timeout=cast(Any, idle_timeout),
            )

    def test_zero_rate_raises(self) -> None:
        with pytest.raises(ValueError, match="rate_hz must be finite"):
            RobotOwnerConfig(name="left-arm", robot=_fake_robot(), rate_hz=0.0)

    def test_negative_rate_raises(self) -> None:
        with pytest.raises(ValueError, match="rate_hz must be finite"):
            RobotOwnerConfig(name="left-arm", robot=_fake_robot(), rate_hz=-1.0)

    def test_infinite_rate_raises(self) -> None:
        with pytest.raises(ValueError, match="rate_hz must be finite"):
            RobotOwnerConfig(name="left-arm", robot=_fake_robot(), rate_hz=float("inf"))

    def test_non_serializable_init_args_raises(self) -> None:
        with pytest.raises(ValueError, match="JSON-serializable"):
            RobotOwnerConfig(
                name="left-arm",
                robot={"class_path": FAKE_ROBOT_CLASS, "init_args": {"calibration": object()}},
            )

    def test_build_known_class(self) -> None:
        config = RobotOwnerConfig(name="left-arm", robot=_fake_robot(port="/dev/ttyUSB0"))
        robot = config.build()
        assert isinstance(robot, FakeRobot)

    def test_build_non_class_path_raises(self) -> None:
        config = RobotOwnerConfig(
            name="left-arm",
            robot={"class_path": "physicalai.robot.transport._ids.KEY_PREFIX", "init_args": {}},
        )
        with pytest.raises(ComponentImportError, match="does not resolve to a class"):
            config.build()

    def test_build_unknown_path_raises(self) -> None:
        config = RobotOwnerConfig(
            name="left-arm",
            robot={"class_path": "totally.unknown.module.Cls", "init_args": {}},
        )
        with pytest.raises(ComponentImportError, match="cannot import class_path"):
            config.build()

    def test_flat_stdin_rejected_before_import(self) -> None:
        flat = {
            "name": "left-arm",
            "robot_class": "totally.unknown.module.Cls",
            "robot_kwargs": {"port": "/dev/ttyUSB0"},
        }
        with pytest.raises(ValueError, match="unknown owner config keys") as exc_info:
            RobotOwnerConfig.from_json_dict(flat)
        assert "robot_class" in str(exc_info.value)
        assert "robot_kwargs" in str(exc_info.value)

    def test_flat_robot_kwargs_alone_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown owner config keys"):
            RobotOwnerConfig.from_json_dict(
                {
                    "name": "left-arm",
                    "robot": _fake_robot(),
                    "robot_kwargs": {},
                },
            )

    def test_unknown_keys_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown owner config keys") as exc_info:
            RobotOwnerConfig.from_json_dict(
                {
                    "name": "left-arm",
                    "robot": _fake_robot(),
                    "extra_field": 1,
                },
            )
        assert "extra_field" in str(exc_info.value)

    def test_missing_robot_rejected(self) -> None:
        with pytest.raises(ValueError, match="missing required 'robot'"):
            RobotOwnerConfig.from_json_dict({"name": "left-arm"})

    def test_validate_owner_config_shared_helper(self) -> None:
        robot = validate_owner_config(
            {
                "name": "left-arm",
                "robot": _fake_robot(port="/dev/ttyUSB0"),
                "allow_remote": True,
                "rate_hz": 50.0,
                "idle_timeout": 2.5,
            },
        )
        assert robot["class_path"] == FAKE_ROBOT_CLASS
        init_args = robot["init_args"]
        assert isinstance(init_args, dict)
        assert init_args["port"] == "/dev/ttyUSB0"

    def test_malformed_robot_rejected_before_import(self) -> None:
        with pytest.raises(ComponentConfigError):
            RobotOwnerConfig.from_json_dict(
                {
                    "name": "left-arm",
                    "robot": {"class_path": "totally.unknown.module.Cls", "extra": 1},
                },
            )

    def test_defining_module_path_stored_as_given(self) -> None:
        # No scservo_sdk mock: storing the recipe must not import the driver.
        defining = "physicalai.robot.so101.so101.SO101"
        config = RobotOwnerConfig(
            name="left-arm",
            robot={
                "class_path": defining,
                "init_args": {"port": "/dev/ttyUSB0", "calibration": "./cal.json"},
            },
        )
        assert config.robot["class_path"] == defining
        assert config.robot_class == defining
        assert config.to_json_dict()["robot"]["class_path"] == defining

    def test_relative_path_survives_json_and_popen_inherits_cwd(self, monkeypatch: pytest.MonkeyPatch) -> None:
        config = RobotOwnerConfig(
            name="left-arm",
            robot=_fake_robot(calibration="./calibration.json"),
        )
        payload = config.to_json_dict()
        assert payload["robot"]["init_args"]["calibration"] == "./calibration.json"

        captured: dict[str, object] = {}

        class _FakeProc:
            stdin = None
            stdout = None

            def __init__(self, *_args: object, **kwargs: object) -> None:
                captured.update(kwargs)
                self.stdin = _FakeStdin()
                self.stdout = _FakeStdout()

            def poll(self) -> int | None:
                return 0

            def terminate(self) -> None:
                return None

            def kill(self) -> None:
                return None

            def wait(self, timeout: float | None = None) -> int:  # noqa: ARG002
                return 0

        class _FakeStdin:
            def write(self, _data: bytes) -> None:
                return None

            def close(self) -> None:
                return None

        class _FakeStdout:
            def readline(self) -> bytes:
                return b"READY\n"

        monkeypatch.setattr(subprocess, "Popen", _FakeProc)
        monkeypatch.setattr(RobotOwner, "_read_stdout_line", lambda _self, _timeout: "READY")
        owner = RobotOwner(config)
        owner.start(timeout=1.0)
        assert "cwd" not in captured


class TestNormalizeRobotConfig:
    def test_normalize_robot_config_rejects_malformed(self) -> None:
        with pytest.raises(ComponentConfigError):
            normalize_robot_config({"class_path": FAKE_ROBOT_CLASS, "extra": True})
