# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``physicalai`` CLI entry points."""

from __future__ import annotations

import io
import logging
import sys
import types
from contextlib import redirect_stdout
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from jsonargparse import ArgumentParser

from physicalai.cli import main as main_module
from physicalai.cli import run as run_module
from physicalai.cli._discovery import discover_subcommands  # noqa: PLC2701
from physicalai.cli._spec import Dispatch, SubcommandSpec  # noqa: PLC2701
from physicalai.cli.main import main
from physicalai.robot.interface import RobotObservation
from physicalai.runtime import PolicyRuntime, RunStats

if TYPE_CHECKING:
    from pathlib import Path


class _FakeObservation:
    joint_positions: np.ndarray
    timestamp: float
    sensor_data: dict[str, np.ndarray] | None
    images: dict | None

    def __init__(self) -> None:
        self.joint_positions = np.zeros(2, dtype=np.float32)
        self.timestamp = 0.0
        self.sensor_data = None
        self.images = None

    @property
    def state(self) -> np.ndarray:
        return self.joint_positions


class FakeRobot:
    """Minimal Robot Protocol implementation usable as a class_path target."""

    def __init__(self, port: str = "/dev/null") -> None:
        self.port = port
        self._connected = False

    @property
    def joint_names(self) -> list[str]:
        return ["j0", "j1"]

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    def get_observation(self) -> RobotObservation:
        return _FakeObservation()

    def send_action(self, action: np.ndarray, *, goal_time: float = 0.1) -> None: ...


_FAKE_ROBOT = f"{__name__}.FakeRobot"
_REAL_SYNC = "physicalai.runtime.execution.SyncExecution"
_REAL_MODEL = "physicalai.inference.model.InferenceModel"

_MINIMAL_ARGV: tuple[str, ...] = (
    f"--runtime.robot={_FAKE_ROBOT}",
    "--runtime.robot.port=/dev/null",
    f"--runtime.model={_REAL_MODEL}",
    "--runtime.model.export_dir=/tmp/fake",  # noqa: S108
    f"--runtime.execution={_REAL_SYNC}",
    "--runtime.fps=30",
)


def _fake_ep(name: str, dist_name: str = "third-party") -> MagicMock:
    ep = MagicMock()
    ep.name = name
    ep.dist = MagicMock()
    ep.dist.name = dist_name
    ep.dist.metadata = {"Summary": f"{dist_name} subcommands"}
    return ep


class TestSubcommandSpec:
    """``SubcommandSpec`` contract (RT-2)."""

    def test_is_frozen_dataclass(self) -> None:
        import dataclasses

        spec = run_module.register()
        with pytest.raises(dataclasses.FrozenInstanceError):
            spec.name = "other"  # type: ignore[misc]

    def test_required_fields_present(self) -> None:
        spec = run_module.register()
        assert spec.name == "run"
        assert isinstance(spec.parser, ArgumentParser)
        assert callable(spec.dispatch)
        assert spec.help

    def test_dispatch_protocol_runtime_checkable(self) -> None:
        assert callable(Dispatch.__call__)


class TestDiscovery:
    """``discover_subcommands`` behaviour (RT-3, RT-4, RT-5)."""

    def test_builtin_excluded_from_discovery(self) -> None:
        with patch("physicalai.cli._discovery.entry_points", return_value=[]):
            assert discover_subcommands(frozenset({"run"})) == {}

    def test_collision_with_builtin_logs_warning(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        ep = _fake_ep("run", dist_name="rogue-pkg")
        with (
            patch("physicalai.cli._discovery.entry_points", return_value=[ep]),
            caplog.at_level(logging.WARNING, logger="physicalai.cli._discovery"),
        ):
            result = discover_subcommands(frozenset({"run"}))
        assert result == {}
        assert "rogue-pkg" in caplog.text
        assert "built-in" in caplog.text

    def test_collision_between_third_parties_first_wins(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        a = _fake_ep("fit", dist_name="pkg-a")
        b = _fake_ep("fit", dist_name="pkg-b")
        with (
            patch("physicalai.cli._discovery.entry_points", return_value=[a, b]),
            caplog.at_level(logging.WARNING, logger="physicalai.cli._discovery"),
        ):
            result = discover_subcommands(frozenset({"run"}))
        assert result == {"fit": a}
        assert "pkg-a" in caplog.text
        assert "pkg-b" in caplog.text

    def test_third_party_subcommand_discovered(self) -> None:
        ep = _fake_ep("fit", dist_name="studio")
        with patch("physicalai.cli._discovery.entry_points", return_value=[ep]):
            assert discover_subcommands(frozenset({"run"})) == {"fit": ep}


class TestRunParser:
    """``physicalai run`` parser behaviour."""

    def test_help_exits_cleanly(self) -> None:
        parser = run_module.build_parser()
        with pytest.raises(SystemExit) as exc:
            parser.parse_args(["--help"])
        assert exc.value.code == 0

    def test_missing_required_args_errors(self) -> None:
        parser = run_module.build_parser()
        with pytest.raises(SystemExit) as exc:
            parser.parse_args([])
        assert exc.value.code != 0

    def test_print_config_outputs_runtime_skeleton(self) -> None:
        parser = run_module.build_parser()
        buf = io.StringIO()
        with redirect_stdout(buf), pytest.raises(SystemExit) as exc:
            parser.parse_args([*_MINIMAL_ARGV, "--print_config"])
        assert exc.value.code == 0
        output = buf.getvalue()
        assert "runtime:" in output
        assert "fps: 30" in output
        assert "FakeRobot" in output

    def test_parses_minimal_argv(self) -> None:
        parser = run_module.build_parser()
        cfg = parser.parse_args(list(_MINIMAL_ARGV))
        assert cfg.runtime.fps == 30
        assert cfg.runtime.robot.class_path == _FAKE_ROBOT
        assert cfg.runtime.execution.class_path == _REAL_SYNC

    def test_config_file_loads_yaml(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "runtime.yaml"
        cfg_file.write_text(
            "runtime:\n"
            "  fps: 30\n"
            "  robot:\n"
            f"    class_path: {_FAKE_ROBOT}\n"
            "    init_args:\n"
            "      port: /dev/null\n"
            "  model:\n"
            f"    class_path: {_REAL_MODEL}\n"
            "    init_args:\n"
            "      export_dir: /tmp/fake\n"
            "  execution:\n"
            f"    class_path: {_REAL_SYNC}\n"
            "run:\n"
            "  duration_s: 5\n",
        )
        parser = run_module.build_parser()
        cfg = parser.parse_args([f"--config={cfg_file}"])
        assert cfg.runtime.fps == 30
        assert cfg.run.duration_s == 5
        assert cfg.runtime.robot.class_path == _FAKE_ROBOT

    def test_cli_overrides_config_file(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "runtime.yaml"
        cfg_file.write_text(
            "runtime:\n"
            "  fps: 30\n"
            "  robot:\n"
            f"    class_path: {_FAKE_ROBOT}\n"
            "    init_args:\n"
            "      port: /dev/null\n"
            "  model:\n"
            f"    class_path: {_REAL_MODEL}\n"
            "    init_args:\n"
            "      export_dir: /tmp/fake\n"
            "  execution:\n"
            f"    class_path: {_REAL_SYNC}\n",
        )
        parser = run_module.build_parser()
        cfg = parser.parse_args([f"--config={cfg_file}", "--runtime.fps=60"])
        assert cfg.runtime.fps == 60


class TestRunDispatcher:
    """``physicalai run`` dispatcher: parser.instantiate + runtime.run."""

    @staticmethod
    def _fake_runtime(stats: RunStats) -> MagicMock:
        fake = MagicMock(spec=PolicyRuntime)
        fake.__enter__ = MagicMock(return_value=fake)
        fake.__exit__ = MagicMock(return_value=None)
        fake.run.return_value = stats
        return fake

    def test_invokes_runtime_run_with_method_args(self) -> None:
        parser = run_module.build_parser()
        cfg = parser.parse_args([*_MINIMAL_ARGV, "--run.duration_s=7"])
        fake = self._fake_runtime(
            RunStats(steps=10, total_pops=10, total_holds=0, inference_count=2),
        )

        with patch.object(parser, "instantiate") as inst:
            inst.return_value = MagicMock(runtime=fake)
            exit_code = run_module.run(parser, cfg)

        assert exit_code == 0
        inst.assert_called_once_with(cfg)
        fake.run.assert_called_once_with(duration_s=7)
        fake.__enter__.assert_called_once()
        fake.__exit__.assert_called_once()

    def test_defaults_to_none_duration(self) -> None:
        parser = run_module.build_parser()
        cfg = parser.parse_args(list(_MINIMAL_ARGV))
        fake = self._fake_runtime(
            RunStats(steps=0, total_pops=0, total_holds=0, inference_count=0),
        )

        with patch.object(parser, "instantiate") as inst:
            inst.return_value = MagicMock(runtime=fake)
            run_module.run(parser, cfg)

        fake.run.assert_called_once_with(duration_s=None)


class TestMainDispatch:
    """End-to-end ``main()`` dispatch."""

    def test_builtin_run_dispatches(self) -> None:
        captured: dict[str, object] = {}

        def fake_dispatch(parser: ArgumentParser, cfg: object) -> int:
            captured["parser"] = parser
            captured["cfg"] = cfg
            return 0

        fake_spec = SubcommandSpec(
            name="run",
            parser=run_module.build_parser(),
            dispatch=fake_dispatch,
            help="x",
        )
        with patch.dict(main_module._BUILTINS, {"run": lambda: fake_spec}, clear=False):  # noqa: SLF001
            exit_code = main(["run", *_MINIMAL_ARGV, "--run.duration_s=1"])
        assert exit_code == 0
        sub_cfg = cast(Any, captured["cfg"])
        assert sub_cfg.runtime.fps == 30
        assert sub_cfg.run.duration_s == 1

    def test_third_party_subcommand_dispatches(self) -> None:
        captured: dict[str, object] = {}

        def fake_dispatch(parser: ArgumentParser, cfg: object) -> int:  # noqa: ARG001
            captured["foo"] = cast(Any, cfg).foo
            return 0

        ext_parser = ArgumentParser(prog="physicalai fit")
        ext_parser.add_argument("--foo", type=int, required=True)
        spec = SubcommandSpec(name="fit", parser=ext_parser, dispatch=fake_dispatch, help="x")

        ep = _fake_ep("fit", dist_name="studio")
        ep.load = MagicMock(return_value=lambda: spec)
        with patch("physicalai.cli.main.discover_subcommands", return_value={"fit": ep}):
            exit_code = main(["fit", "--foo", "7"])
        assert exit_code == 0
        assert captured["foo"] == 7

    def test_help_exits_zero(self) -> None:
        with pytest.raises(SystemExit) as exc:
            main(["--help"])
        assert exc.value.code == 0

    def test_help_does_not_expose_raw_print_completion(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as exc:
            main(["--help"])
        assert exc.value.code == 0
        output = capsys.readouterr().out
        assert "--print_completion" not in output

    def test_completion_subcommand_without_shell_prints_help(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        with patch("sys.argv", ["pai", "completion"]):
            exit_code = main()
        assert exit_code == 0
        output = capsys.readouterr().out
        assert "usage: pai completion {bash,fish,zsh}" in output
        assert "Print a shell completion script." in output

    def test_completion_subcommand_uses_invoked_program_name(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        with patch("sys.argv", ["pai", "completion", "bash"]):
            exit_code = main()
        assert exit_code == 0
        output = capsys.readouterr().out
        assert "complete -o filenames -F _shtab_pai pai" in output

    def test_zsh_completion_script_is_safe_to_source(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        with patch("sys.argv", ["pai", "completion", "zsh"]):
            exit_code = main()
        assert exit_code == 0
        output = capsys.readouterr().out
        assert "compdef _shtab_pai -N pai" in output
        assert '_shtab_pai "$@"' not in output

    def test_run_help_exits_zero(self) -> None:
        exit_code = main(["run", "--help"])
        assert exit_code == 0

    def test_run_help_uses_fast_help_without_building_parser(self, capsys: pytest.CaptureFixture[str]) -> None:
        with patch.object(run_module, "build_parser", side_effect=AssertionError("should not build parser")):
            exit_code = main(["run", "--help"])
        assert exit_code == 0
        assert "usage:" in capsys.readouterr().out

    def test_third_party_help_uses_fast_help_without_registering(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        module = types.ModuleType("fake_cli_fit")

        def print_help(prog: str) -> None:
            print(f"fast help for {prog}")

        def register() -> SubcommandSpec:
            raise AssertionError("should not register")

        register.__module__ = module.__name__
        setattr(module, "print_help", print_help)
        setattr(module, "register", register)
        sys.modules[module.__name__] = module
        ep = _fake_ep("fit", dist_name="studio")
        ep.load = MagicMock(return_value=register)
        try:
            with patch("physicalai.cli.main.discover_subcommands", return_value={"fit": ep}):
                exit_code = main(["fit", "--help"])
        finally:
            sys.modules.pop(module.__name__, None)
        assert exit_code == 0
        assert "fast help for pytest fit" in capsys.readouterr().out

    def test_builtins_contain_run_only(self) -> None:
        assert list(main_module._BUILTINS) == ["run"]  # noqa: SLF001

    def test_unknown_subcommand_errors(self) -> None:
        with pytest.raises(SystemExit) as exc:
            main(["bogus"])
        assert exc.value.code != 0

    def test_invalid_spec_name_raises(self) -> None:
        wrong = SubcommandSpec(
            name="other",
            parser=ArgumentParser(),
            dispatch=lambda parser, cfg: 0,
            help="",
        )
        ep = _fake_ep("fit")
        ep.load = MagicMock(return_value=lambda: wrong)
        with (
            patch("physicalai.cli.main.discover_subcommands", return_value={"fit": ep}),
            pytest.raises(ValueError, match="name mismatch"),
        ):
            main(["fit"])
