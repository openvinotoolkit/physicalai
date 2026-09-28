from __future__ import annotations

import argparse
import os
import signal
import subprocess
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from physicalai_mujoco_so101_plugin import __main__ as cli
from physicalai_mujoco_so101_plugin.__main__ import _stop_owner_over_http
from physicalai_mujoco_so101_plugin.constants import (
    DEFAULT_BIMANUAL_MUJOCO_OWNER_NAME,
    DEFAULT_MUJOCO_OWNER_NAME,
)


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
        with patch("physicalai_mujoco_so101_plugin.__main__.http.client.HTTPConnection", return_value=connection):
            assert cli._http_owner_name("127.0.0.1", 8080) == "mujoco-so101-bimanual"  # noqa: SLF001
        connection.request.assert_called_once_with("GET", "/")
        connection.close.assert_called_once_with()

    def test_unreachable_returns_none(self) -> None:
        with patch(
            "physicalai_mujoco_so101_plugin.__main__.http.client.HTTPConnection",
            side_effect=OSError("refused"),
        ):
            assert cli._http_owner_name("127.0.0.1", 8080) is None  # noqa: SLF001

    def test_non_json_response_returns_none(self) -> None:
        response = self._response(b"not json")
        connection = MagicMock()
        connection.getresponse.return_value = response
        with patch("physicalai_mujoco_so101_plugin.__main__.http.client.HTTPConnection", return_value=connection):
            assert cli._http_owner_name("127.0.0.1", 8080) is None  # noqa: SLF001


class TestPidOwnerName:
    @staticmethod
    def _ps_output(cmdline: str) -> SimpleNamespace:
        return SimpleNamespace(args=["ps"], returncode=0, stdout=f"{cmdline}\n", stderr="")

    def test_explicit_name_flag(self) -> None:
        cmdline = "physicalai-mujoco-so101 start --model x.xml --name my-sim --bimanual"
        with patch("subprocess.run", return_value=self._ps_output(cmdline)):
            assert cli._pid_owner_name(1234) == "my-sim"  # noqa: SLF001

    def test_bimanual_default_without_explicit_name(self) -> None:
        cmdline = "physicalai-mujoco-so101 start --model x.xml --bimanual"
        with patch("subprocess.run", return_value=self._ps_output(cmdline)):
            assert cli._pid_owner_name(1234) == DEFAULT_BIMANUAL_MUJOCO_OWNER_NAME  # noqa: SLF001

    def test_single_arm_default_without_explicit_name(self) -> None:
        cmdline = "physicalai-mujoco-so101 start --model x.xml"
        with patch("subprocess.run", return_value=self._ps_output(cmdline)):
            assert cli._pid_owner_name(1234) == DEFAULT_MUJOCO_OWNER_NAME  # noqa: SLF001

    def test_unreadable_command_line_returns_none(self) -> None:
        with patch("subprocess.run", side_effect=FileNotFoundError):
            assert cli._pid_owner_name(1234) is None  # noqa: SLF001


class TestStopOwnerOverHttp:
    def test_posts_shutdown(self) -> None:
        with (
            patch.object(cli, "_http_owner_name", return_value="my-sim"),
            patch.object(cli, "_owner_pid", return_value=1234),
            patch("physicalai_mujoco_so101_plugin.__main__.http.client.HTTPConnection") as factory,
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
            "physicalai_mujoco_so101_plugin.__main__.http.client.HTTPConnection",
            return_value=connection,
        ):
            stopped = cli._request_http_shutdown("127.0.0.1", 8080)  # noqa: SLF001
        assert stopped is False

    def test_connection_error_is_silent(self) -> None:
        connection = MagicMock()
        connection.request.side_effect = ConnectionError("refused")
        with patch(
            "physicalai_mujoco_so101_plugin.__main__.http.client.HTTPConnection",
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


class TestResolveOwnerName:
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
            assert cli._matching_pids("physicalai-mujoco-so101 start") == [424242]  # noqa: SLF001

    def test_pattern_is_passed_to_pgrep(self) -> None:
        with patch("subprocess.run", return_value=self._pgrep_output("")) as run:
            cli._matching_pids("physicalai-mujoco-so101 start")  # noqa: SLF001

        assert run.call_args.args[0] == ["pgrep", "-f", "physicalai-mujoco-so101 start"]

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

        assert patterns == ["physicalai-mujoco-so101 start"]
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
        assert cli._matching_pids("physicalai-mujoco-so101 start") == []
    assert all(call.kwargs["timeout"] == 5 for call in run.call_args_list)
