# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""CLI dispatch uses the same public report as application consumers."""

import json
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
    monkeypatch.setattr(diagnostics, "diagnose", lambda selector: report)
    assert main(["doctor", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == report.to_dict()


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
