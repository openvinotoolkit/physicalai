# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""CLI dispatch uses the same public report as application consumers."""

import json
import sys
from unittest.mock import Mock

import pytest

import physicalai.diagnostics as diagnostics
from physicalai.cli import doctor
from physicalai.cli.main import main
from physicalai.diagnostics import Device, Endpoint, Finding, Inventory, Report


@pytest.fixture
def report() -> Report:
    return Report(
        Inventory([
            Device(
                "1-1",
                "Test camera",
                speed_mbps=480,
                video_interface=True,
                endpoints=[Endpoint("/dev/video0", "video", "capture", True, True)],
            )
        ]),
        [],
    )


def test_doctor_dispatches_to_public_api(
    report: Report,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    diagnose = Mock(return_value=report)
    monkeypatch.setattr(diagnostics, "diagnose", diagnose)
    assert main(["doctor", "/dev/video0"]) == 0
    diagnose.assert_called_once_with("/dev/video0")
    assert "Test camera" in capsys.readouterr().out


def test_json_is_the_unstyled_api_document(
    report: Report,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report.inventory.usbfs_memory_mb = 256
    monkeypatch.setattr(diagnostics, "diagnose", lambda selector: report)
    assert main(["doctor", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == report.to_dict()


@pytest.mark.parametrize("memory", [None, 0, 16, 127, 128, 256])
def test_usbfs_memory_is_informational(
    report: Report, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], memory: int | None
) -> None:
    report.inventory.usbfs_memory_mb = memory
    monkeypatch.setattr(diagnostics, "diagnose", lambda selector: report)
    assert main(["doctor", "--no-ansi"]) == 0
    output = capsys.readouterr().out
    assert ("usbcore.usbfs_memory_mb" in output) == (memory is None or memory < 128)
    assert "WARNING" not in output and "ERROR" not in output


def test_warning_is_success_but_access_error_fails(report: Report, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(diagnostics, "diagnose", lambda selector: report)
    report.findings = [Finding("ambiguous_identity", "warning", ["1-1"], "Duplicate identity")]
    assert main(["doctor"]) == 0
    report.findings = [Finding("access", "error", ["1-1"], "Permission denied")]
    assert main(["doctor"]) == 1


def test_collection_errors_fail(report: Report, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(diagnostics, "diagnose", lambda selector: report)
    report.inventory.errors.append("sysfs unavailable")
    assert main(["doctor"]) == 1


@pytest.mark.parametrize("error", [ValueError("ambiguous path"), NotImplementedError("Linux only")])
def test_api_errors_become_cli_usage_errors(error: Exception, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(diagnostics, "diagnose", Mock(side_effect=error))
    with pytest.raises(SystemExit) as exc:
        main(["doctor"])
    assert exc.value.code == 2


def test_help_never_enumerates_hardware(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(diagnostics, "diagnose", Mock(side_effect=AssertionError("No hardware during help")))
    with pytest.raises(SystemExit) as exc:
        main(["doctor", "--help"])
    assert exc.value.code == 0


def test_only_small_passive_command_surface_is_exposed() -> None:
    help_text = doctor.build_parser().format_help()
    assert "--tree" in help_text and "--json" in help_text
    assert all(option not in help_text for option in ("--stream", "--config", "--watch", "--strict"))


@pytest.mark.parametrize(
    "tty,no_color,term,args,styled",
    [
        (True, "", "xterm-256color", [], True),
        (False, "", "xterm-256color", [], False),
        (True, "1", "xterm-256color", [], False),
        (True, "0", "xterm-256color", [], False),
        (True, "", "dumb", [], False),
        (True, "", "xterm-256color", ["--no-ansi"], False),
    ],
)
def test_terminal_styling_detection(
    report: Report,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tty: bool,
    no_color: str,
    term: str,
    args: list[str],
    styled: bool,
) -> None:
    monkeypatch.setattr(diagnostics, "diagnose", lambda selector: report)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: tty)
    monkeypatch.setenv("NO_COLOR", no_color)
    monkeypatch.setenv("TERM", term)
    assert main(["doctor", *args]) == 0
    assert ("\x1b[" in capsys.readouterr().out) is styled


@pytest.mark.parametrize("args", [[], ["--no-ansi"]])
def test_json_stays_plain_even_on_a_color_terminal(
    report: Report,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    args: list[str],
) -> None:
    monkeypatch.setattr(diagnostics, "diagnose", lambda selector: report)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    assert main(["doctor", "--json", *args]) == 0
    output = capsys.readouterr().out
    assert "\x1b" not in output
    assert json.loads(output) == report.to_dict()
