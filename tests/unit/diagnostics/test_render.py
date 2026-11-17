# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Compact human views; data and findings remain independent of presentation."""

import re
from typing import Literal

import pytest

from physicalai.diagnostics import Device, Inventory, Report, assess
from physicalai.diagnostics.render import render_report


def test_default_output_is_compact_and_relevant(inventory: Inventory) -> None:
    text = render_report(Report(inventory, assess(inventory)))
    assert "/dev/video0" in text and "/dev/ttyACM0" in text
    assert "/dev/video1" not in text
    assert "Keyboard" not in text
    assert "not necessarily a fault" in text
    assert all(term not in text for term in ("SKIP", "PASS", "MB/s", "usbfs", "Getting (payload)"))
    assert len(text.splitlines()) < 25


def test_tree_only_shows_relevant_branches(inventory: Inventory) -> None:
    text = render_report(Report(inventory, assess(inventory)), tree=True)
    assert "Controller" in text and "Inner hub" in text
    assert "/dev/video0" in text
    assert "/dev/video1" not in text and "Keyboard" not in text
    assert "root ports are separate links" in text


def test_inspection_prints_a_usable_port_path(inventory: Inventory) -> None:
    report = Report(inventory, assess(inventory), selected_device_id="1-1.1.1", selected_endpoint="/dev/video0")
    text = render_report(report)
    assert "Unique port-bound path: /dev/v4l/by-path/port-1.1" in text
    assert "Unverified/ambiguous alias: /dev/v4l/by-id/" in text
    assert "Serial A" in text and "Serial B" in text
    assert "Keyboard" not in text
    assert "Port-bound paths follow the slot" in text


def test_metadata_inspection_explains_wrong_node(inventory: Inventory) -> None:
    report = Report(inventory, [], selected_device_id="1-1.1.1", selected_endpoint="/dev/video1")
    assert "Metadata only; select an image endpoint" in render_report(report)


def test_terminal_controls_are_escaped_without_mutating_data(inventory: Inventory) -> None:
    inventory.devices[4].name = "Camera\x1b[31m\n\u202eevil"
    report = Report(inventory, [])
    text = render_report(report)
    assert "\x1b" not in text and "\u202e" not in text
    assert r"Camera\u001b[31m\u000a\u202eevil" in text
    assert report.inventory.devices[4].name.startswith("Camera\x1b")


def test_no_devices_and_incomplete_collection_are_explained() -> None:
    text = render_report(Report(Inventory(errors=["sysfs unavailable"]), []))
    assert "No USB camera/serial endpoints found" in text
    assert "Collection limitations" in text and "sysfs unavailable" in text


def test_cyclic_topology_remains_visible(inventory: Inventory) -> None:
    inventory.devices[2].parent_id = "1-1.1"
    text = render_report(Report(inventory, []), tree=True)
    assert "[unresolved ancestry]" in text
    assert text.count("/dev/video0") == 1


def test_usb3_and_serial_speeds_are_not_mislabeled() -> None:
    inventory = Inventory([
        Device("1-1", "Camera", speed_mbps=5000, video_interface=True),
        Device("1-2", "Another camera", speed_mbps=None, video_interface=True),
    ])
    text = render_report(Report(inventory, []))
    assert "5000 Mb/s (USB 3.x)" in text
    assert "unknown" in text


@pytest.mark.parametrize("tree", [False, True])
def test_styling_preserves_text_and_table_alignment(inventory: Inventory, tree: bool) -> None:
    report = Report(inventory, assess(inventory))
    plain = render_report(report, tree=tree)
    styled = render_report(report, tree=tree, ansi=True)
    assert "\x1b[1mUSB connection diagnostics (Linux)\x1b[0m" in styled
    assert "\x1b[1;33mWARNING:\x1b[0m" in styled
    assert re.sub(r"\x1b\[[\d;]*m", "", styled) == plain
    assert "\x1b" not in plain


def test_usb_generations_are_colored_but_serial_speeds_are_neutral() -> None:
    inventory = Inventory([
        Device("1-1", "USB 3 camera", speed_mbps=5000, video_interface=True),
        Device("1-2", "USB 2 camera", speed_mbps=480, video_interface=True),
        Device("1-3", "Slow device", speed_mbps=12, video_interface=True),
    ])
    text = render_report(Report(inventory, []), ansi=True)
    assert "\x1b[1;36m5000 Mb/s (USB 3.x)\x1b[0m" in text
    assert "\x1b[1;33m480 Mb/s (USB 2)\x1b[0m" in text
    assert "\x1b[1m12 Mb/s\x1b[0m" in text


def test_styled_inspection_keeps_untrusted_text_escaped(inventory: Inventory) -> None:
    inventory.devices[4].name = "Camera\x1b[31m\n\u202eevil"
    report = Report(inventory, [], selected_device_id="1-1.1.1", selected_endpoint="/dev/video0")
    text = render_report(report, ansi=True)
    assert "\x1b[31m" not in text and "\u202e" not in text
    assert r"Camera\u001b[31m\u000a\u202eevil" in text
    assert re.sub(r"\x1b\[[\d;]*m", "", text) == render_report(report)


def test_styling_does_not_change_api_data(inventory: Inventory) -> None:
    report = Report(inventory, assess(inventory))
    original = report.to_dict()
    render_report(report, ansi=True)
    assert report.to_dict() == original


@pytest.mark.parametrize("severity,code", [("error", "1;31"), ("info", "1;34")])
def test_finding_severities_are_colored(inventory: Inventory, severity: Literal["error", "info"], code: str) -> None:
    report = Report(inventory, assess(inventory))
    report.findings[0].severity = severity
    assert f"\x1b[{code}m{severity.upper()}:\x1b[0m" in render_report(report, ansi=True)
