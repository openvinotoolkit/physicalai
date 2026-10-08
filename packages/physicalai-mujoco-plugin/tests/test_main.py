from __future__ import annotations

import argparse
import contextlib
import functools
import http.server
import io
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from physicalai_mujoco_plugin import __main__ as cli
from physicalai_mujoco_plugin.__main__ import _stop_owner_over_http
from physicalai_mujoco_plugin.constants import (
    DEFAULT_BIMANUAL_MUJOCO_OWNER_NAME,
    DEFAULT_MUJOCO_OWNER_NAME,
)

if TYPE_CHECKING:
    from collections.abc import Callable


class TestHttpOwnerName:
    @staticmethod
    def _response(body: bytes) -> MagicMock:
        response = MagicMock()
        response.__enter__.return_value = response
        response.read.return_value = body
        return response

    def test_returns_service_field(self) -> None:
        response = self._response(b'{"service": "mujoco-so101-bimanual"}')
        connection = MagicMock()
        connection.getresponse.return_value = response
        with patch("physicalai_mujoco_plugin.__main__.http.client.HTTPConnection", return_value=connection):
            assert cli._http_owner_name("127.0.0.1", 8080) == "mujoco-so101-bimanual"  # noqa: SLF001
        connection.request.assert_called_once_with("GET", "/")
        connection.close.assert_called_once_with()

    def test_unreachable_returns_none(self) -> None:
        with patch(
            "physicalai_mujoco_plugin.__main__.http.client.HTTPConnection",
            side_effect=OSError("refused"),
        ):
            assert cli._http_owner_name("127.0.0.1", 8080) is None  # noqa: SLF001

    def test_non_json_response_returns_none(self) -> None:
        response = self._response(b"not json")
        connection = MagicMock()
        connection.getresponse.return_value = response
        with patch("physicalai_mujoco_plugin.__main__.http.client.HTTPConnection", return_value=connection):
            assert cli._http_owner_name("127.0.0.1", 8080) is None  # noqa: SLF001


class TestPidOwnerName:
    @staticmethod
    def _ps_output(cmdline: str) -> SimpleNamespace:
        return SimpleNamespace(args=["ps"], returncode=0, stdout=f"{cmdline}\n", stderr="")

    def test_explicit_name_flag(self) -> None:
        cmdline = "physicalai-mujoco start --model x.xml --name my-sim --bimanual"
        with patch("subprocess.run", return_value=self._ps_output(cmdline)):
            assert cli._pid_owner_name(1234) == "my-sim"  # noqa: SLF001

    def test_bimanual_default_without_explicit_name(self) -> None:
        cmdline = "physicalai-mujoco start --model x.xml --bimanual"
        with patch("subprocess.run", return_value=self._ps_output(cmdline)):
            assert cli._pid_owner_name(1234) == DEFAULT_BIMANUAL_MUJOCO_OWNER_NAME  # noqa: SLF001

    @pytest.mark.parametrize(
        ("scene_id", "expected"),
        [("single_pick_place", DEFAULT_MUJOCO_OWNER_NAME), ("garment_fold", DEFAULT_BIMANUAL_MUJOCO_OWNER_NAME)],
    )
    def test_a_custom_model_counts_its_mount_frames_like_start(self, scene_id: str, expected: str) -> None:
        """``start --model`` takes its arm count from the XML, so the stop fallback must too."""
        from physicalai_mujoco_plugin.scene_registry import get_scene  # noqa: PLC0415

        cmdline = f"physicalai-mujoco start --model {get_scene(scene_id).scene_xml_path}"
        with patch("subprocess.run", return_value=self._ps_output(cmdline)):
            assert cli._pid_owner_name(1234) == expected  # noqa: SLF001

    @pytest.mark.parametrize("model", ["/no/such/scene.xml", "relative/scene.xml"])
    def test_a_custom_model_with_an_unknown_arm_count_matches_no_owner(self, model: str) -> None:
        cmdline = f"physicalai-mujoco start --model {model}"
        with (
            patch("subprocess.run", return_value=self._ps_output(cmdline)),
            patch.object(cli, "_pid_cwd", return_value=None),
        ):
            assert cli._pid_owner_name(1234) is None  # noqa: SLF001

    def test_a_relative_custom_model_resolves_in_the_start_directory(self) -> None:
        from physicalai_mujoco_plugin.scene_registry import get_scene  # noqa: PLC0415

        scene_xml = get_scene("garment_fold").scene_xml_path
        cmdline = f"physicalai-mujoco start --model {scene_xml.parent.name}/{scene_xml.name}"
        with (
            patch("subprocess.run", return_value=self._ps_output(cmdline)),
            patch.object(cli, "_pid_cwd", return_value=scene_xml.parent.parent),
        ):
            assert cli._pid_owner_name(1234) == DEFAULT_BIMANUAL_MUJOCO_OWNER_NAME  # noqa: SLF001

    def test_pid_cwd_reads_this_process(self) -> None:
        assert cli._pid_cwd(os.getpid()) == Path.cwd().resolve()  # noqa: SLF001

    @pytest.mark.parametrize(
        ("arguments", "expected"),
        [
            ("--profile ur5e", "mujoco-ur5e-follow"),
            ("--profile=ur5e", "mujoco-ur5e-follow"),
            ("--scene garment_fold", DEFAULT_MUJOCO_OWNER_NAME),
            ("--scene=garment_fold --bimanual", DEFAULT_BIMANUAL_MUJOCO_OWNER_NAME),
            ("--scene conveyor_sort --bimanual", DEFAULT_BIMANUAL_MUJOCO_OWNER_NAME),
            ("--profile ur5e --scene no_such_scene", "mujoco-ur5e-follow"),
            ("--profile no_such_model", "mujoco-no_such_model-follow"),
            ("--profile trossen_wxai --bimanual", "mujoco-trossen_wxai-bimanual-follow"),
            ("--profile rebot_b601", "mujoco-rebot_b601-follow"),
        ],
    )
    def test_profile_and_scene_resolve_like_start(self, arguments: str, expected: str) -> None:
        cmdline = f"physicalai-mujoco start {arguments}"
        with patch("subprocess.run", return_value=self._ps_output(cmdline)):
            assert cli._pid_owner_name(1234) == expected  # noqa: SLF001

    def test_unreadable_command_line_returns_none(self) -> None:
        with patch("subprocess.run", side_effect=FileNotFoundError):
            assert cli._pid_owner_name(1234) is None  # noqa: SLF001


class TestStopOwnerOverHttp:
    def test_posts_shutdown(self) -> None:
        with (
            patch.object(cli, "_http_owner_name", return_value="my-sim"),
            patch.object(cli, "_owner_pid", return_value=1234),
            patch("physicalai_mujoco_plugin.__main__.http.client.HTTPConnection") as factory,
        ):
            stopped = cli._stop_owner_over_http("127.0.0.1", 8080, "my-sim", 1234)  # noqa: SLF001
        assert stopped is True
        connection = factory.return_value
        factory.assert_called_once_with("127.0.0.1", 8080, timeout=5)
        connection.request.assert_called_once_with("POST", "/shutdown")
        connection.close.assert_called_once_with()

    def test_unreachable_is_silent(self) -> None:
        connection = MagicMock()
        connection.request.side_effect = OSError("refused")
        with patch(
            "physicalai_mujoco_plugin.__main__.http.client.HTTPConnection",
            return_value=connection,
        ):
            stopped = cli._request_http_shutdown("127.0.0.1", 8080)  # noqa: SLF001
        assert stopped is False

    def test_connection_error_is_silent(self) -> None:
        connection = MagicMock()
        connection.request.side_effect = ConnectionError("refused")
        with patch(
            "physicalai_mujoco_plugin.__main__.http.client.HTTPConnection",
            return_value=connection,
        ):
            stopped = cli._request_http_shutdown("127.0.0.1", 8080)  # noqa: SLF001
        assert stopped is False

    @pytest.mark.parametrize("name,pid", [("other-sim", 1234), ("my-sim", 5678), ("my-sim", None)])
    def test_skips_wrong_endpoint_or_replaced_owner(self, name, pid) -> None:
        with (
            patch.object(cli, "_http_owner_name", return_value=name),
            patch.object(cli, "_owner_pid", return_value=pid),
            patch.object(cli, "_request_http_shutdown") as post,
        ):
            _stop_owner_over_http("127.0.0.1", 8080, "my-sim", 1234)
        post.assert_not_called()


class TestStopOwnerBySignal:
    def test_signals_original_owner_after_rechecking_its_identity(self) -> None:
        with (
            patch.object(cli, "_owner_pid", return_value=1234),
            patch.object(cli, "_terminate", return_value=True) as terminate,
        ):
            assert cli._stop_owner_by_signal("my-sim", 1234) is True  # noqa: SLF001

        terminate.assert_called_once_with(1234, "owner 'my-sim'")

    @pytest.mark.parametrize("current_pid", [None, 5678])
    def test_does_not_signal_if_owner_changed_or_exited(self, current_pid: int | None) -> None:
        with patch.object(cli, "_owner_pid", return_value=current_pid), patch("os.kill") as kill:
            assert cli._stop_owner_by_signal("my-sim", 1234) is False  # noqa: SLF001

        kill.assert_not_called()


class TestLauncherLifecycle:
    @pytest.mark.parametrize("next_pid", [None, 5678])
    def test_wait_returns_when_original_owner_exits(self, next_pid) -> None:
        shutdown = MagicMock(spec=threading.Event)
        shutdown.wait.return_value = False
        with patch.object(cli, "_owner_pid", side_effect=[1234, next_pid]):
            cli._wait_for_owner_shutdown(shutdown, "my-sim", 1234)
        assert shutdown.wait.call_count == 2
        assert all(call.kwargs["timeout"] == 0.25 for call in shutdown.wait.call_args_list)
        shutdown.set.assert_not_called()

    def test_wait_returns_on_operator_signal(self) -> None:
        shutdown = threading.Event()
        shutdown.set()
        with patch.object(cli, "_owner_pid") as lookup:
            cli._wait_for_owner_shutdown(shutdown, "my-sim", 1234)
        lookup.assert_not_called()

    def test_missing_initial_owner_does_not_wait(self) -> None:
        shutdown = MagicMock(spec=threading.Event)
        cli._wait_for_owner_shutdown(shutdown, "my-sim", None)
        shutdown.wait.assert_not_called()

    @pytest.mark.parametrize("operator_signal,http_enabled", [(False, True), (True, True), (True, False)])
    def test_start_only_requests_shutdown_on_operator_signal(self, operator_signal, http_enabled) -> None:
        args = cli._build_parser().parse_args(["start", "--name", "my-sim", "--no-cameras", "--no-gui"])
        args.no_http = not http_enabled

        def wait(shutdown, name, pid):
            assert (name, pid) == ("my-sim", 1234)
            if operator_signal:
                shutdown.set()

        with (
            patch.object(cli.SharedRobot, "from_config") as factory,
            patch.object(cli, "_owner_pid", return_value=1234),
            patch.object(cli, "_wait_for_owner_shutdown", side_effect=wait),
            patch.object(cli, "_stop_owner_over_http", return_value=True) as stop,
            patch.object(cli, "_stop_owner_by_signal") as signal_stop,
            patch.object(cli.signal, "signal"),
        ):
            cli._start(args)
        factory.return_value.disconnect.assert_called_once()
        if operator_signal and http_enabled:
            stop.assert_called_once_with("127.0.0.1", 8080, "my-sim", 1234)
            signal_stop.assert_not_called()
        elif operator_signal:
            stop.assert_not_called()
            signal_stop.assert_called_once_with("my-sim", 1234)
        else:
            stop.assert_not_called()
            signal_stop.assert_not_called()

    def test_start_falls_back_to_signal_when_http_shutdown_fails(self) -> None:
        args = cli._build_parser().parse_args(["start", "--name", "my-sim", "--no-cameras", "--no-gui"])

        def wait(shutdown, name, pid) -> None:
            assert (name, pid) == ("my-sim", 1234)
            shutdown.set()

        with (
            patch.object(cli.SharedRobot, "from_config") as factory,
            patch.object(cli, "_owner_pid", return_value=1234),
            patch.object(cli, "_wait_for_owner_shutdown", side_effect=wait),
            patch.object(cli, "_stop_owner_over_http", return_value=False) as http_stop,
            patch.object(cli, "_stop_owner_by_signal", return_value=True) as signal_stop,
            patch.object(cli.signal, "signal"),
        ):
            cli._start(args)

        http_stop.assert_called_once_with("127.0.0.1", 8080, "my-sim", 1234)
        signal_stop.assert_called_once_with("my-sim", 1234)
        factory.return_value.disconnect.assert_called_once()

    def test_start_uses_signal_fallback_when_http_is_disabled(self) -> None:
        args = cli._build_parser().parse_args(
            ["start", "--name", "my-sim", "--no-cameras", "--no-gui", "--no-http"],
        )

        def wait(shutdown, name, pid) -> None:
            assert (name, pid) == ("my-sim", 1234)
            shutdown.set()

        with (
            patch.object(cli.SharedRobot, "from_config") as factory,
            patch.object(cli, "_owner_pid", return_value=1234),
            patch.object(cli, "_wait_for_owner_shutdown", side_effect=wait),
            patch.object(cli, "_stop_owner_over_http") as http_stop,
            patch.object(cli, "_stop_owner_by_signal", return_value=True) as signal_stop,
            patch.object(cli.signal, "signal"),
        ):
            cli._start(args)

        http_stop.assert_not_called()
        signal_stop.assert_called_once_with("my-sim", 1234)
        factory.return_value.disconnect.assert_called_once()

    def test_viser_host_defaults_to_loopback_and_can_be_overridden(self) -> None:
        parser = cli._build_parser()
        default_args = parser.parse_args(["start"])
        remote_args = parser.parse_args(["start", "--viser-host", "0.0.0.0"])

        assert default_args.viser_host == "127.0.0.1"
        assert remote_args.viser_host == "0.0.0.0"

    def test_keyboard_interrupt_requests_identity_checked_shutdown(self) -> None:
        args = cli._build_parser().parse_args(["start", "--name", "my-sim", "--no-cameras", "--no-gui"])
        with (
            patch.object(cli.SharedRobot, "from_config") as factory,
            patch.object(cli, "_owner_pid", return_value=1234),
            patch.object(cli, "_wait_for_owner_shutdown", side_effect=KeyboardInterrupt),
            patch.object(cli, "_stop_owner_over_http") as stop,
            patch.object(cli.signal, "signal"),
        ):
            cli._start(args)
        stop.assert_called_once_with("127.0.0.1", 8080, "my-sim", 1234)
        factory.return_value.disconnect.assert_called_once()


class TestRobotModelFetch:
    def test_start_fetches_the_robot_before_the_owner_starts(self) -> None:
        args = cli._build_parser().parse_args(["start", "--name", "my-sim", "--no-cameras", "--no-gui"])
        calls: list[str] = []

        def wait(shutdown, _name, _pid) -> None:
            shutdown.set()

        with (
            patch.object(cli, "fetch_profile", lambda profile, _progress: calls.append(f"fetch {profile.name}")),
            patch.object(
                cli.SharedRobot, "from_config", side_effect=lambda *_a, **_k: calls.append("owner") or MagicMock()
            ),
            patch.object(cli, "_owner_pid", return_value=None),
            patch.object(cli, "_wait_for_owner_shutdown", side_effect=wait),
            patch.object(cli.signal, "signal"),
        ):
            cli._start(args)
        assert calls == ["fetch so101", "owner"]

    def test_start_exits_without_an_owner_when_the_fetch_fails(self) -> None:
        args = cli._build_parser().parse_args(["start", "--no-cameras", "--no-gui"])
        with (
            patch.object(cli, "fetch_profile", side_effect=RuntimeError("offline")),
            patch.object(cli.SharedRobot, "from_config") as factory,
            pytest.raises(SystemExit) as exit_info,
        ):
            cli._start(args)
        assert exit_info.value.code == 1
        factory.assert_not_called()

    def test_custom_model_without_mounts_skips_the_fetch(self, tmp_path) -> None:
        path = tmp_path / "custom.xml"
        path.write_text("<mujoco><worldbody/></mujoco>")
        with patch.object(cli, "fetch_profile") as fetch:
            cli._fetch_robot_models(str(path), cli.get_profile("so101"))  # noqa: SLF001
        fetch.assert_not_called()

    def test_prefetch_fetches_every_registered_profile(self) -> None:
        profiles = (cli.get_profile("so101"), cli.get_profile("ur5e"))
        with (
            patch.object(cli.sys, "argv", ["physicalai-mujoco", "prefetch"]),
            patch.object(cli, "list_profiles", return_value=profiles),
            patch.object(cli, "fetch_profile", return_value="cached.xml") as fetch,
        ):
            cli.main()
        assert [call.args[0] for call in fetch.call_args_list] == list(profiles)

    def test_profiles_list_the_arm_counts(self, capsys: pytest.CaptureFixture[str]) -> None:
        with patch.object(cli.sys, "argv", ["physicalai-mujoco", "profiles"]):
            cli.main()
        rows = {line.split()[0]: line.split()[2:4] for line in capsys.readouterr().out.splitlines()[1:] if line}
        assert rows["so101"] == ["1,", "2"]
        assert rows["koch"] == ["1,", "2"]
        assert rows["aloha"][0] == rows["unitree_go2"][0] == "1"

    def test_prefetch_exits_on_failure(self) -> None:
        with (
            patch.object(cli.sys, "argv", ["physicalai-mujoco", "prefetch"]),
            patch.object(cli, "fetch_profile", side_effect=RuntimeError("offline")),
            pytest.raises(SystemExit) as exit_info,
        ):
            cli.main()
        assert exit_info.value.code == 1


class TestStartRecipe:
    @staticmethod
    def _recipe(argv: list[str]) -> dict[str, object]:
        args = cli._build_parser().parse_args(["start", *argv])
        configs = []

        def factory(config, **kwargs):
            configs.append((config, kwargs))
            return MagicMock()

        with (
            patch.object(cli, "fetch_profile"),
            patch.object(cli.SharedRobot, "from_config", side_effect=factory),
            patch.object(cli, "_owner_pid", return_value=None),
            patch.object(cli, "_wait_for_owner_shutdown"),
            patch.object(cli.signal, "signal"),
        ):
            cli._start(args)
        config, kwargs = configs[0]
        return {"class_path": config["class_path"], **config["init_args"], "name": kwargs["name"]}

    def test_default_is_the_so101_in_single_pick_place(self) -> None:
        recipe = self._recipe(["--no-gui"])
        assert recipe["class_path"] == "physicalai_mujoco_plugin.robot.MuJoCoRobot"
        assert (recipe["profile"], recipe["scene"], recipe["name"]) == ("so101", "single_pick_place", "mujoco-so101-follow")
        assert recipe["cameras"] is None  # the robot's cameras, then the overview

    def test_bimanual_flag_keeps_the_default_scene_and_takes_the_bimanual_name(self) -> None:
        recipe = self._recipe(["--bimanual", "--no-gui", "--no-cameras"])
        assert (recipe["scene"], recipe["bimanual"], recipe["name"]) == (
            "single_pick_place",
            True,
            "mujoco-so101-bimanual-follow",
        )
        assert recipe["cameras"] == []

    @pytest.mark.parametrize(
        ("profile", "scene"),
        [("trossen_wxai", "single_pick_place"), ("rebot_b601", "single_pick_place"), ("so101", "conveyor_sort")],
    )
    def test_bimanual_runs_in_any_tabletop_scene(self, profile: str, scene: str) -> None:
        recipe = self._recipe(["--profile", profile, "--scene", scene, "--bimanual", "--no-gui"])
        assert (recipe["scene"], recipe["bimanual"], recipe["name"]) == (scene, True, f"mujoco-{profile}-bimanual-follow")

    @pytest.mark.parametrize("profile", ["aloha", "unitree_go2"])
    def test_bimanual_is_refused_for_two_arm_models_and_floating_bases(self, profile: str) -> None:
        with pytest.raises(SystemExit) as exit_info:
            self._recipe(["--profile", profile, "--bimanual", "--no-gui"])
        assert exit_info.value.code == 1

    def test_garment_fold_runs_one_arm_without_bimanual(self) -> None:
        recipe = self._recipe(["--scene", "garment_fold", "--no-gui"])
        assert (recipe.get("bimanual", False), recipe["name"]) == (False, "mujoco-so101-follow")

    def test_a_custom_model_with_two_mounts_is_bimanual(self) -> None:
        from physicalai_mujoco_plugin.scene_registry import get_scene  # noqa: PLC0415

        recipe = self._recipe(["--model", str(get_scene("garment_fold").scene_xml_path), "--no-gui"])
        assert recipe["name"] == "mujoco-so101-bimanual-follow"

    @pytest.mark.parametrize(
        "argv",
        [
            ["--profile", "aloha"],
            ["--profile", "aloha", "--bimanual"],
            ["--profile", "unitree_g1", "--bimanual"],
        ],
    )
    def test_a_two_mount_model_refuses_single_robot_profiles(self, argv: list[str]) -> None:
        from physicalai_mujoco_plugin.scene_registry import get_scene  # noqa: PLC0415

        with pytest.raises(SystemExit) as exit_info:
            self._recipe([*argv, "--model", str(get_scene("garment_fold").scene_xml_path), "--no-gui"])
        assert exit_info.value.code == 1

    def test_bimanual_needs_the_left_and_right_mount_frames(self, tmp_path) -> None:
        model = tmp_path / "other.xml"
        model.write_text(
            '<mujoco><worldbody><frame name="a_robot_mount"/><frame name="b_robot_mount" pos="0 0.3 0"/>'
            "</worldbody></mujoco>"
        )
        with pytest.raises(SystemExit) as exit_info:
            self._recipe(["--model", str(model), "--bimanual", "--no-gui"])
        assert exit_info.value.code == 1

    def test_bimanual_with_a_one_mount_model_exits(self) -> None:
        from physicalai_mujoco_plugin.scene_registry import get_scene  # noqa: PLC0415

        with pytest.raises(SystemExit) as exit_info:
            self._recipe(["--model", str(get_scene("yahtzee").scene_xml_path), "--bimanual", "--no-gui"])
        assert exit_info.value.code == 1

    def test_other_profiles_get_their_own_name(self) -> None:
        recipe = self._recipe(["--profile", "ur5e", "--no-gui"])
        assert (recipe["profile"], recipe["scene"], recipe["name"]) == ("ur5e", "single_pick_place", "mujoco-ur5e-follow")

    @pytest.mark.parametrize(
        "argv",
        [["--profile", "no_such_robot"], ["--scene", "nope"], ["--profile", "ur5e", "--scene", "conveyor_sort"]],
    )
    def test_bad_profile_or_scene_exits(self, argv: list[str]) -> None:
        with pytest.raises(SystemExit) as exit_info:
            self._recipe(argv)
        assert exit_info.value.code == 1


class TestResolveOwnerName:
    def test_other_profiles(self) -> None:
        args = argparse.Namespace(name=None, bimanual=False)
        assert cli._resolve_owner_name(args, "ur5e", 1) == "mujoco-ur5e-follow"  # noqa: SLF001
        assert cli._resolve_owner_name(args, "ur5e", 2) == "mujoco-ur5e-bimanual-follow"  # noqa: SLF001

    def test_single_arm_default(self) -> None:
        args = argparse.Namespace(name=None, bimanual=False)
        assert cli._resolve_owner_name(args) == DEFAULT_MUJOCO_OWNER_NAME  # noqa: SLF001

    def test_bimanual_gets_its_own_default(self) -> None:
        args = argparse.Namespace(name=None, bimanual=True)
        assert cli._resolve_owner_name(args) == DEFAULT_BIMANUAL_MUJOCO_OWNER_NAME  # noqa: SLF001

    def test_explicit_name_wins(self) -> None:
        args = argparse.Namespace(name="my-sim", bimanual=True)
        assert cli._resolve_owner_name(args) == "my-sim"  # noqa: SLF001

    def test_defaults_differ(self) -> None:
        assert DEFAULT_MUJOCO_OWNER_NAME != DEFAULT_BIMANUAL_MUJOCO_OWNER_NAME


class TestMatchingPids:
    @staticmethod
    def _pgrep_output(stdout: str) -> SimpleNamespace:
        return SimpleNamespace(args=["pgrep"], returncode=0, stdout=stdout, stderr="")

    def test_own_process_and_parent_are_excluded(self) -> None:
        stdout = f"{os.getpid()}\n{os.getppid()}\n424242\n"
        with patch("subprocess.run", return_value=self._pgrep_output(stdout)):
            assert cli._matching_pids("physicalai-mujoco start") == [424242]  # noqa: SLF001

    def test_pattern_is_passed_to_pgrep(self) -> None:
        with patch("subprocess.run", return_value=self._pgrep_output("")) as run:
            cli._matching_pids("physicalai-mujoco start")  # noqa: SLF001

        assert run.call_args.args[0] == ["pgrep", "-f", "physicalai-mujoco start"]

    def test_non_numeric_output_is_ignored(self) -> None:
        with patch("subprocess.run", return_value=self._pgrep_output("nope\n7\n")):
            assert cli._matching_pids("x") == [7]  # noqa: SLF001


class TestStop:
    @staticmethod
    def _args(name: str = DEFAULT_MUJOCO_OWNER_NAME) -> argparse.Namespace:
        return argparse.Namespace(http_host="127.0.0.1", http_port=8080, name=name)

    def test_http_shutdown_used_when_no_local_owner(self) -> None:
        with (
            patch.object(cli, "_owner_pid", return_value=None),
            patch.object(cli, "_http_owner_name", return_value=DEFAULT_MUJOCO_OWNER_NAME),
            patch.object(cli, "_request_http_shutdown", return_value=True),
            patch.object(cli, "_matching_pids") as matching,
            patch("os.kill") as kill,
        ):
            cli._stop(self._args())  # noqa: SLF001

        matching.assert_not_called()
        kill.assert_not_called()

    def test_http_shutdown_skipped_when_owner_name_mismatches(self) -> None:
        """The HTTP port might belong to a different named owner; don't shut it down."""
        with (
            patch.object(cli, "_owner_pid", return_value=None),
            patch.object(cli, "_http_owner_name", return_value="some-other-owner"),
            patch.object(cli, "_request_http_shutdown") as http_shutdown,
            patch.object(cli, "_matching_pids", return_value=[]),
            patch("os.kill") as kill,
        ):
            cli._stop(self._args())  # noqa: SLF001

        http_shutdown.assert_not_called()
        kill.assert_not_called()

    def test_named_owner_is_preferred_over_http(self) -> None:
        """A local owner matching --name is stopped directly; HTTP is never tried."""
        with (
            patch.object(cli, "_owner_pid", return_value=4242),
            patch.object(cli, "_request_http_shutdown") as http_shutdown,
            patch.object(cli, "_matching_pids") as matching,
            patch("os.kill") as kill,
        ):
            cli._stop(self._args())  # noqa: SLF001

        http_shutdown.assert_not_called()
        matching.assert_not_called()
        kill.assert_called_once_with(4242, signal.SIGTERM)

    def test_named_owner_is_signalled(self) -> None:
        with (
            patch.object(cli, "_request_http_shutdown", return_value=False),
            patch.object(cli, "_owner_pid", return_value=4242),
            patch.object(cli, "_matching_pids", return_value=[]),
            patch("os.kill") as kill,
        ):
            cli._stop(self._args())  # noqa: SLF001

        kill.assert_called_once_with(4242, signal.SIGTERM)

    def test_pgrep_fallback_skips_mismatched_owner_names(self) -> None:
        """Two concurrent sims: stopping one by name must not kill the other."""
        with (
            patch.object(cli, "_owner_pid", return_value=None),
            patch.object(cli, "_http_owner_name", return_value=None),
            patch.object(cli, "_request_http_shutdown", return_value=False),
            patch.object(cli, "_matching_pids", return_value=[111, 222]),
            patch.object(cli, "_pid_owner_name", side_effect=lambda pid: "other-sim" if pid == 111 else "mujoco-so101"),
            patch("os.kill") as kill,
        ):
            cli._stop(self._args(name="mujoco-so101"))  # noqa: SLF001

        kill.assert_called_once_with(222, signal.SIGTERM)

    def test_only_start_invocations_are_matched(self) -> None:
        patterns = []

        def record(pattern: str) -> list[int]:
            patterns.append(pattern)
            return []

        with (
            patch.object(cli, "_request_http_shutdown", return_value=False),
            patch.object(cli, "_http_owner_name", return_value=None),
            patch.object(cli, "_owner_pid", return_value=None),
            patch.object(cli, "_matching_pids", side_effect=record),
            patch("os.kill") as kill,
        ):
            cli._stop(self._args())  # noqa: SLF001

        assert patterns == ["physicalai-mujoco start"]
        assert all("_owner_worker" not in pattern for pattern in patterns)
        kill.assert_not_called()

    def test_unsignalable_pid_is_reported_not_raised(self) -> None:
        with (
            patch.object(cli, "_request_http_shutdown", return_value=False),
            patch.object(cli, "_owner_pid", return_value=4242),
            patch.object(cli, "_matching_pids", return_value=[]),
            patch("os.kill", side_effect=ProcessLookupError),
        ):
            cli._stop(self._args())  # noqa: SLF001


class TestOwnerPid:
    def test_missing_lock_file_returns_none(self) -> None:
        assert cli._owner_pid("no-such-owner-name-xyz") is None  # noqa: SLF001

    def test_released_lock_is_not_a_live_owner(self, monkeypatch, tmp_path) -> None:
        from physicalai.robot.transport import _lock

        monkeypatch.setattr(_lock, "_lock_dir", lambda: tmp_path)
        locks = _lock.acquire_locks("mujoco-stale-lock-test", ())
        try:
            assert cli._owner_pid("mujoco-stale-lock-test") == os.getpid()
        finally:
            locks.release_all()
        # The diagnostic PID still exists, but no owner holds the lock anymore.
        assert cli._owner_pid("mujoco-stale-lock-test") is None


@pytest.mark.parametrize("error", [FileNotFoundError(), subprocess.TimeoutExpired("ps", 5)])
def test_process_lookup_failures_are_bounded(error: Exception) -> None:
    with patch("subprocess.run", side_effect=error) as run:
        assert cli._pid_command_line(1234) is None
        assert cli._matching_pids("physicalai-mujoco start") == []
    assert all(call.kwargs["timeout"] == 5 for call in run.call_args_list)


ROOT = {"service": "my-sim", "cameras": ["wrist", "overview"], "viewer_url": "http://127.0.0.1:9090"}
HEALTH = {
    "cameras": [
        {"name": "wrist", "has_frame": True, "failed": False},
        {"name": "overview", "has_frame": True, "failed": False},
    ]
}


class _Launch(SimpleNamespace):
    """What a mocked ``start`` did: recipe, transport kwargs, events and the owner calls."""

    @property
    def events(self) -> list[dict[str, object]]:
        return [json.loads(line) for line in self.stdout.getvalue().splitlines()]


def _launch(  # noqa: PLR0913
    argv: list[str],
    *,
    root: dict | None = ROOT,
    health: dict | None = HEALTH,
    owner_pid: int | None = 1234,
    existing_pid: int | None = None,
    spawned: bool = True,
    on_connect: Callable[[], None] | None = None,
    fetch: bool = False,
    stdout: io.StringIO | None = None,
) -> _Launch:
    """Run ``start`` with the owner mocked.

    Args:
        argv: ``start`` arguments after ``--name my-sim``.
        root: The owner's ``GET /``.
        health: The owner's ``GET /health``.
        owner_pid: The owner holding the name once ``connect`` ran.
        existing_pid: The owner holding the name before ``connect`` (another simulation).
        spawned: Whether that owner is a child of this process.
        on_connect: Runs inside ``connect``, after the owner took its name.
        fetch: Run the real ``fetch_profile`` instead of a no-op.
        stdout: Where the events go (a new buffer by default).
    """
    args = cli._build_parser().parse_args(["start", "--name", "my-sim", *argv])
    launch = _Launch(stdout=stdout or io.StringIO(), connected=False, exit_code=None)
    shared = MagicMock()

    def connect() -> None:
        launch.connected = True
        if on_connect is not None:
            on_connect()

    shared.connect.side_effect = connect

    def factory(config, **kwargs):
        launch.init_args, launch.kwargs = config["init_args"], kwargs
        return shared

    responses = {"/": root, "/health": health}
    with contextlib.ExitStack() as stack:
        if not fetch:
            stack.enter_context(patch.object(cli, "fetch_profile"))
        launch.factory = stack.enter_context(patch.object(cli.SharedRobot, "from_config", side_effect=factory))
        stack.enter_context(
            patch.object(cli, "_owner_pid", side_effect=lambda _name: owner_pid if launch.connected else existing_pid)
        )
        stack.enter_context(patch.object(cli, "_spawned_by_this_process", return_value=spawned))
        launch.http_json = stack.enter_context(
            patch.object(cli, "_http_json", side_effect=lambda _host, _port, path: responses[path])
        )
        launch.wait = stack.enter_context(patch.object(cli, "_wait_for_owner_shutdown"))
        launch.http_stop = stack.enter_context(patch.object(cli, "_stop_owner_over_http", return_value=True))
        launch.signal_stop = stack.enter_context(patch.object(cli, "_stop_owner_by_signal", return_value=True))
        stack.enter_context(patch.object(cli.signal, "signal"))
        stack.enter_context(patch.object(cli.sys, "stdout", launch.stdout))
        try:
            cli._start(args)  # noqa: SLF001
        except SystemExit as exc:
            launch.exit_code = exc.code
    launch.shared = shared
    return launch


PHASES = [
    {"event": "phase", "phase": "fetch"},
    {"event": "phase", "phase": "connect"},
    {"event": "phase", "phase": "load"},
    {"event": "phase", "phase": "cameras"},
]


class TestStatusJson:
    """``start --status-json``: startup events on stdout for a supervising process (P4)."""

    def test_phases_then_ready_with_the_owners_addresses_and_cameras(self) -> None:
        events = _launch(["--status-json", "--no-gui"]).events
        assert events == [
            *PHASES,
            {
                "event": "ready",
                "name": "my-sim",
                "pid": 1234,
                "profile": "so101",
                "scene": "single_pick_place",
                "arms": 1,
                "overview_style": "shoulder",
                "http_url": "http://127.0.0.1:8080",
                "viewer_url": "http://127.0.0.1:9090",
                "cameras": ["wrist", "overview"],
            },
        ]
        assert list(events[-1]) == [
            "event", "name", "pid", "profile", "scene", "arms", "overview_style", "http_url", "viewer_url", "cameras",
        ]  # fmt: skip

    def test_ready_reports_the_overview_style(self) -> None:
        assert _launch(["--status-json", "--no-gui", "--overview", "front"]).events[-1]["overview_style"] == "front"

    def test_load_is_announced_while_the_owner_loads(self) -> None:
        """The owner takes its name, then loads; ``load`` must not wait for ``connect`` to return."""
        stdout = io.StringIO()
        seen: list[str] = []

        def owner_loading() -> None:
            deadline = time.monotonic() + 5.0
            while '"load"' not in stdout.getvalue() and time.monotonic() < deadline:
                time.sleep(0.01)
            seen.append(stdout.getvalue())

        _launch(["--status-json", "--no-gui"], on_connect=owner_loading, stdout=stdout)
        assert '"load"' in seen[0]
        assert '"cameras"' not in seen[0]

    def test_without_the_flag_stdout_stays_empty(self) -> None:
        launch = _launch(["--no-gui"])
        assert launch.stdout.getvalue() == ""
        launch.http_json.assert_not_called()

    def test_without_http_there_are_no_streams_and_the_viewer_url_is_the_configured_one(self) -> None:
        launch = _launch(["--status-json", "--no-http", "--viser-port", "9191"])
        assert launch.events[:-1] == PHASES
        ready = launch.events[-1]
        assert (ready["http_url"], ready["viewer_url"], ready["cameras"]) == (None, "http://127.0.0.1:9191", [])
        launch.http_json.assert_not_called()

    @pytest.mark.parametrize("root", [None, {**ROOT, "service": "other-sim"}])
    def test_requested_http_that_does_not_answer_is_an_error_and_stops_the_owner(self, root) -> None:
        """A failed bind or a port answering for another owner: Studio could not use the sim."""
        launch = _launch(["--status-json", "--no-gui"], root=root)
        assert launch.exit_code == 1
        assert launch.events[-1]["event"] == "error"
        assert "HTTP server of 'my-sim' is not answering" in str(launch.events[-1]["message"])
        assert "ready" not in [event["event"] for event in launch.events]
        launch.http_stop.assert_called_once_with("127.0.0.1", 8080, "my-sim", 1234)
        launch.shared.disconnect.assert_called_once_with()

    def test_a_requested_viewer_that_did_not_start_is_an_error_and_stops_the_owner(self) -> None:
        """Studio embeds the viewer; the owner publishes no viewer URL when viser failed to start."""
        launch = _launch(["--status-json"], root={**ROOT, "viewer_url": None})
        assert launch.exit_code == 1
        assert "browser viewer of 'my-sim' did not start" in str(launch.events[-1]["message"])
        assert "ready" not in [event["event"] for event in launch.events]
        launch.http_stop.assert_called_once_with("127.0.0.1", 8080, "my-sim", 1234)

    def test_without_a_requested_viewer_none_is_expected(self) -> None:
        launch = _launch(["--status-json", "--no-gui"], root={**ROOT, "viewer_url": None})
        assert launch.events[-1]["event"] == "ready"
        assert launch.events[-1]["viewer_url"] is None

    def test_a_bind_all_host_is_reached_and_reported_on_loopback(self) -> None:
        launch = _launch(["--status-json", "--no-gui", "--http-host", "0.0.0.0"])  # noqa: S104
        assert launch.events[-1]["http_url"] == "http://127.0.0.1:8080"
        launch.http_json.assert_any_call("127.0.0.1", 8080, "/")

    def test_a_start_error_is_reported_with_its_message_and_exits_non_zero(self) -> None:
        launch = _launch(["--status-json", "--scene", "nope"])
        assert launch.exit_code == 1
        assert launch.events[-1]["event"] == "error"
        assert "Unknown scene 'nope'" in str(launch.events[-1]["message"])

    def test_an_owner_failure_is_reported_and_raised(self) -> None:
        def fail() -> None:
            raise RuntimeError("robot owner did not become ready")

        stdout = io.StringIO()
        with pytest.raises(RuntimeError):
            _launch(["--status-json", "--no-gui"], on_connect=fail, stdout=stdout)
        last = json.loads(stdout.getvalue().splitlines()[-1])
        assert last == {"event": "error", "message": "robot owner did not become ready"}

    def test_a_closed_reader_turns_the_writer_off(self) -> None:
        stdout = MagicMock()
        stdout.write.side_effect = BrokenPipeError
        stdout.fileno.side_effect = ValueError
        writer = cli._StatusWriter(enabled=True)  # noqa: SLF001
        with patch.object(cli.sys, "stdout", stdout):
            writer.phase("fetch")
            writer.phase("connect")
        assert writer.enabled is False
        stdout.write.assert_called_once()

    def test_each_event_is_one_flushed_line(self) -> None:
        stdout = MagicMock()
        with patch.object(cli.sys, "stdout", stdout):
            cli._StatusWriter(enabled=True).error("boom")  # noqa: SLF001
        stdout.write.assert_called_once_with('{"event": "error", "message": "boom"}\n')
        stdout.flush.assert_called_once_with()


class TestNameCollision:
    """A supervised start never attaches to, or stops, a simulation that another command started."""

    @pytest.mark.parametrize("flag", ["--status-json", "--exit-with-parent"])
    def test_a_running_owner_with_the_name_is_an_error_before_spawning(self, flag: str) -> None:
        with patch.object(cli, "_shut_down_on_stdin_eof"):
            launch = _launch([flag, "--no-gui"], existing_pid=4242)
        assert launch.exit_code == 1
        launch.factory.assert_not_called()
        launch.http_stop.assert_not_called()
        launch.signal_stop.assert_not_called()

    def test_an_owner_attached_in_a_race_is_never_stopped_even_on_eof(self) -> None:
        shutdown_on_eof: list[threading.Event] = []
        with patch.object(cli, "_shut_down_on_stdin_eof", side_effect=shutdown_on_eof.append):
            launch = _launch(
                ["--status-json", "--exit-with-parent", "--no-gui"],
                spawned=False,
                on_connect=lambda: shutdown_on_eof[0].set(),
            )
        assert launch.exit_code == 1
        assert "was not started by this command" in str(launch.events[-1]["message"])
        launch.http_stop.assert_not_called()
        launch.signal_stop.assert_not_called()
        launch.shared.disconnect.assert_called_once_with()

    def test_an_unsupervised_start_keeps_its_old_behaviour(self) -> None:
        """Without the flags, a second ``start`` still attaches and its Ctrl+C stops the owner."""
        launch = _launch(["--no-gui"], existing_pid=4242, spawned=False)
        assert launch.exit_code is None
        launch.factory.assert_called_once()

    def test_only_a_running_child_counts_as_spawned_by_this_process(self) -> None:
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])  # noqa: S603
        try:
            assert cli._spawned_by_this_process(child.pid) is True  # noqa: SLF001
            # The owner of a simulation another command started is no child of this process.
            assert cli._spawned_by_this_process(os.getppid()) is False  # noqa: SLF001
        finally:
            child.kill()
            child.wait()
        assert cli._spawned_by_this_process(child.pid) is False  # noqa: SLF001


class TestCameraReadiness:
    """``ready`` lists only cameras that stream (P4 review)."""

    @staticmethod
    def _health(*cameras: tuple[str, bool, bool]) -> dict:
        return {"cameras": [{"name": n, "has_frame": f, "failed": x} for n, f, x in cameras]}

    def test_waits_for_first_frames_and_drops_failed_cameras(self) -> None:
        replies = [
            None,  # the owner is still answering its first requests
            self._health(("wrist", False, False), ("overview", False, False)),
            self._health(("wrist", True, False), ("overview", False, True)),
        ]
        with patch.object(cli, "_http_json", side_effect=replies) as http_json:
            assert cli._working_cameras("127.0.0.1", 8080) == ["wrist"]  # noqa: SLF001
        assert http_json.call_count == 3

    def test_cameras_without_a_frame_by_the_deadline_are_left_out(self) -> None:
        pending = self._health(("wrist", True, False), ("overview", False, False))
        with patch.object(cli, "_http_json", return_value=pending):
            assert cli._working_cameras("127.0.0.1", 8080, timeout_s=0.2) == ["wrist"]  # noqa: SLF001

    def test_an_unreachable_status_reports_no_cameras(self) -> None:
        with patch.object(cli, "_http_json", return_value=None):
            assert cli._working_cameras("127.0.0.1", 8080, timeout_s=0.2) == []  # noqa: SLF001

    def test_ready_lists_the_working_cameras(self) -> None:
        launch = _launch(["--status-json", "--no-gui"], health=self._health(("wrist", True, False), ("overview", False, True)))
        assert launch.events[-1]["cameras"] == ["wrist"]


class _ArchiveServer:
    """Serve one Menagerie-style archive over local HTTP, as a fake ``mujoco_menagerie`` robot."""

    def __init__(self, tmp_path: Path, size: int) -> None:
        import hashlib  # noqa: PLC0415
        import tarfile  # noqa: PLC0415

        from mujoco_menagerie._registry import EntryPoint, Robot  # noqa: PLC0415, PLC2701

        tree = tmp_path / "src" / "fake_arm"
        tree.mkdir(parents=True)
        (tree / "fake_arm.xml").write_text("<mujoco/>")
        (tree / "mesh.bin").write_bytes(os.urandom(size))
        served = tmp_path / "served"
        served.mkdir()
        archive = served / "fake_arm.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(tree, arcname="fake_arm")
        data = archive.read_bytes()
        self.size = len(data)
        self.robot = Robot(
            name="fake_arm",
            display_name="Fake arm",
            category="arm",
            license="MIT",
            oid="0" * 40,
            asset=archive.name,
            sha256=hashlib.sha256(data).hexdigest(),
            download_size=len(data),
            installed_size=size,
            entry_points=(EntryPoint(name="so101", kind="robot", file="fake_arm.xml"),),
            default_model="so101",
            default_scene=None,
        )
        handler = functools.partial(_QuietHandler, directory=str(served))
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *_args: object) -> None:
        pass


class TestDownloadProgress:
    """``fetch`` events carry the Menagerie download's bytes and total; a cached model sends none."""

    @pytest.fixture
    def menagerie(self, tmp_path):
        import mujoco_menagerie  # noqa: PLC0415

        server = _ArchiveServer(tmp_path, size=3 * 2**20)
        env = {"MENAGERIE_CACHE_DIR": str(tmp_path / "cache"), "MENAGERIE_BASE_URL": server.url}
        with (
            patch.dict(os.environ, env),
            patch.object(mujoco_menagerie, "get", return_value=server.robot),
        ):
            os.environ.pop("MENAGERIE_ROOT", None)
            yield server
        server.close()

    def test_a_real_download_reports_progress_in_order_and_a_cached_one_does_not(self, menagerie) -> None:
        launch = _launch(["--status-json", "--no-gui"], fetch=True)
        events = launch.events
        progress = [event for event in events if "bytes" in event]
        assert events[0] == {"event": "phase", "phase": "fetch"}
        assert events[1 : 1 + len(progress)] == progress
        assert events[1 + len(progress) :][:3] == PHASES[1:]
        assert len(progress) >= 2  # the archive spans several 1 MiB chunks
        assert [event["bytes"] for event in progress] == sorted(event["bytes"] for event in progress)
        assert progress[-1]["bytes"] == progress[-1]["total"] == menagerie.size

        cached = _launch(["--status-json", "--no-gui"], fetch=True).events
        assert [event for event in cached if "bytes" in event] == []
        assert cached[:4] == PHASES


class TestPorts:
    """``--http-port 0`` / ``--viser-port 0`` pick free ports; ``--no-http`` is what disables HTTP (P4)."""

    def test_zero_picks_distinct_free_ports_and_ready_reports_them(self) -> None:
        launch = _launch(["--status-json", "--http-port", "0", "--viser-port", "0"])
        http_port, viser_port = launch.init_args["http_port"], launch.init_args["viser_port"]
        assert 0 < http_port != viser_port > 0
        assert launch.kwargs["idle_timeout"] is None  # HTTP stays on, so no idle exit
        launch.http_json.assert_any_call("127.0.0.1", http_port, "/")
        assert launch.events[-1]["http_url"] == f"http://127.0.0.1:{http_port}"

    def test_disabled_servers_get_port_zero(self) -> None:
        init_args = _launch(["--no-http", "--no-gui", "--http-port", "0", "--viser-port", "0"]).init_args
        assert (init_args["http_port"], init_args["viser_port"]) == (0, 0)

    def test_fixed_ports_are_kept(self) -> None:
        init_args = _launch(["--http-port", "8123", "--viser-port", "9123"]).init_args
        assert (init_args["http_port"], init_args["viser_port"]) == (8123, 9123)

    def test_free_ports_keeps_each_socket_until_all_are_picked(self) -> None:
        ports = cli._free_ports([("127.0.0.1", 0), ("127.0.0.1", 0), ("127.0.0.1", None), ("127.0.0.1", 8080)])  # noqa: SLF001
        assert ports[0] != ports[1]
        assert ports[0] > 0
        assert ports[2:] == [0, 8080]

    def test_an_unbindable_host_exits(self) -> None:
        with patch.object(cli, "_free_ports", side_effect=OSError("cannot assign requested address")):
            launch = _launch(["--http-port", "0"])
        assert launch.exit_code == 1
        launch.factory.assert_not_called()

    @pytest.mark.parametrize("flag", ["--http-port", "--viser-port"])
    @pytest.mark.parametrize("value", ["-1", "65536", "http"])
    def test_out_of_range_ports_are_rejected(self, flag: str, value: str) -> None:
        with pytest.raises(SystemExit) as exit_info:
            cli._build_parser().parse_args(["start", flag, value])
        assert exit_info.value.code == 2

    @pytest.mark.parametrize(("url_host", "expected"), [("0.0.0.0", "127.0.0.1"), ("::", "[::1]"), ("::1", "[::1]")])  # noqa: S104
    def test_local_url_reaches_bind_all_hosts_on_loopback(self, url_host: str, expected: str) -> None:
        assert cli._local_url(url_host, 80) == f"http://{expected}:80"  # noqa: SLF001


class TestSeed:
    """``start --seed`` fixes the reset seed, in the range ``POST /seed`` accepts."""

    @pytest.mark.parametrize(("argv", "seed"), [([], None), (["--seed", "0"], 0), (["--seed", "4294967295"], 2**32 - 1)])
    def test_the_seed_reaches_the_robot(self, argv: list[str], seed: int | None) -> None:
        assert _launch(argv).init_args["seed"] == seed

    @pytest.mark.parametrize("value", ["-1", "4294967296", "1.5", "random"])
    def test_out_of_range_seeds_are_rejected(self, value: str) -> None:
        with pytest.raises(SystemExit) as exit_info:
            cli._build_parser().parse_args(["start", "--seed", value])
        assert exit_info.value.code == 2


class TestExitWithParent:
    """``--exit-with-parent``: stdin EOF stops the sim, and the owner follows ``start`` (P5)."""

    def test_the_owner_follows_this_process(self) -> None:
        with patch.object(cli, "_shut_down_on_stdin_eof"):
            assert _launch(["--exit-with-parent"]).init_args["exit_with_pid"] == os.getpid()
        assert _launch([]).init_args["exit_with_pid"] is None

    def test_stdin_eof_requests_shutdown_but_input_does_not(self) -> None:
        read_fd, write_fd = os.pipe()
        shutdown = threading.Event()
        with os.fdopen(read_fd) as stdin, patch.object(cli.sys, "stdin", stdin):
            cli._shut_down_on_stdin_eof(shutdown)  # noqa: SLF001
            os.write(write_fd, b"keep going\n")
            assert not shutdown.wait(0.2)
            os.close(write_fd)
            assert shutdown.wait(5.0)

    def test_a_missing_stdin_counts_as_eof(self) -> None:
        shutdown = threading.Event()
        with patch.object(cli.sys, "stdin", None):
            cli._shut_down_on_stdin_eof(shutdown)  # noqa: SLF001
            assert shutdown.wait(5.0)

    def test_eof_stops_the_owner_this_command_spawned_like_sigterm(self) -> None:
        shutdown_on_eof: list[threading.Event] = []
        with patch.object(cli, "_shut_down_on_stdin_eof", side_effect=shutdown_on_eof.append):
            launch = _launch(["--no-gui", "--exit-with-parent"], on_connect=lambda: shutdown_on_eof[0].set())
        launch.http_stop.assert_called_once_with("127.0.0.1", 8080, "my-sim", 1234)
        launch.shared.disconnect.assert_called_once_with()

    def test_eof_during_the_download_starts_no_owner(self) -> None:
        with patch.object(cli, "_shut_down_on_stdin_eof", side_effect=lambda shutdown: shutdown.set()):
            launch = _launch(["--no-gui", "--exit-with-parent"])
        launch.factory.assert_not_called()

    def test_viewer_theme_reaches_the_driver(self) -> None:
        assert _launch(["--viewer-theme", "studio"]).init_args["viewer_theme"] == "studio"
        assert _launch([]).init_args["viewer_theme"] == "default"

    def test_overview_reaches_the_driver(self) -> None:
        assert _launch(["--overview", "front"]).init_args["overview"] == "front"
        assert _launch([]).init_args["overview"] == "shoulder"

    @pytest.mark.parametrize(
        ("argv", "message"),
        [
            (["--profile", "unitree_go2"], "Scene floor_flat has no front overview camera"),
            (["--model", "MODEL"], "needs a registered tabletop scene"),
        ],
    )
    def test_front_overview_needs_a_tabletop_scene(self, argv: list[str], message: str, tmp_path: Path) -> None:
        model = tmp_path / "model.xml"
        model.write_text('<mujoco><worldbody><frame name="robot_mount"/></worldbody></mujoco>')
        argv = [str(model) if arg == "MODEL" else arg for arg in argv]
        launch = _launch(["--status-json", *argv, "--overview", "front"])
        assert launch.exit_code == 1
        assert not launch.connected
        assert message in str(launch.events[-1]["message"])


class TestHeadlessGl:
    """``start`` renders with EGL on a headless Linux host (P8)."""

    @pytest.mark.parametrize(
        ("platform", "environ", "egl", "expected"),
        [
            ("linux", {}, "libEGL.so.1", {"MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl"}),
            ("linux", {"PYOPENGL_PLATFORM": "egl"}, "libEGL.so.1", {"MUJOCO_GL": "egl"}),
            ("linux", {"DISPLAY": ":0"}, "libEGL.so.1", {}),
            ("linux", {"WAYLAND_DISPLAY": "wayland-0"}, "libEGL.so.1", {}),
            ("linux", {"MUJOCO_GL": "osmesa"}, "libEGL.so.1", {}),
            ("linux", {}, None, {}),
            ("darwin", {}, "libEGL.dylib", {}),
            ("win32", {}, "EGL.dll", {}),
        ],
    )
    def test_egl_only_without_a_display_or_a_choice(self, platform, environ, egl, expected) -> None:
        with patch("ctypes.util.find_library", return_value=egl):
            assert cli._headless_gl_env(platform, environ) == expected  # noqa: SLF001

    def test_start_sets_it_before_anything_loads_a_scene(self) -> None:
        seen: list[tuple[str | None, str | None]] = []

        def resolve_scene(*_args: object) -> None:
            seen.append((os.environ.get("MUJOCO_GL"), os.environ.get("PYOPENGL_PLATFORM")))
            raise SystemExit(1)

        with (
            patch.dict(os.environ),
            patch.object(cli.sys, "platform", "linux"),
            patch("ctypes.util.find_library", return_value="libEGL.so.1"),
            patch.object(cli, "_resolve_scene", side_effect=resolve_scene),
            pytest.raises(SystemExit),
        ):
            for name in ("MUJOCO_GL", "PYOPENGL_PLATFORM", "DISPLAY", "WAYLAND_DISPLAY"):
                os.environ.pop(name, None)
            cli._start(cli._build_parser().parse_args(["start"]))  # noqa: SLF001
        assert seen == [("egl", "egl")]
