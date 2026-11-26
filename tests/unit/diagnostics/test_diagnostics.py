# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Public API and pure checks, without hardware or SDK dependencies."""

from __future__ import annotations

import copy
import json
import subprocess
import sys

import pytest
from pydantic import TypeAdapter

from physicalai.diagnostics import Alias, Device, Endpoint, Inventory, Report, assess, collect_inventory, diagnose
from physicalai.diagnostics import api


def test_nested_hubs_and_duplicate_aliases_are_not_repeated(inventory: Inventory) -> None:
    findings = assess(inventory)
    assert [f.code for f in findings] == ["ambiguous_identity", "shared_usb_link"]
    assert all(f.severity == "warning" for f in findings)
    shared = findings[1]
    assert shared.hub_id == "1-1.1"
    assert shared.devices == ["1-1.1.1", "1-1.1.2"]
    assert "does not prove congestion" in shared.hint
    assert not Report(inventory, findings).has_errors


def test_usb2_alone_and_serial_speeds_are_not_faults(inventory: Inventory) -> None:
    inventory.devices = [d for d in inventory.devices if d.id != "1-1.1.2"]
    assert assess(inventory) == []


def test_root_ports_are_not_a_shared_uplink(inventory: Inventory) -> None:
    for device in inventory.devices:
        if device.video_interface:
            device.parent_id = "usb1"
    assert not any(f.code == "shared_usb_link" for f in assess(inventory))


def test_slowest_shared_uplink_is_reported_once(inventory: Inventory) -> None:
    inventory.devices[3].speed_mbps = 5000
    finding = next(f for f in assess(inventory) if f.code == "shared_usb_link")
    assert finding.hub_id == "1-1"
    assert finding.severity == "warning"


@pytest.mark.parametrize("speed", [5000, None])
def test_shared_fast_or_unknown_hubs_are_context(inventory: Inventory, speed: float | None) -> None:
    for device in inventory.devices:
        if device.kind == "hub":
            device.speed_mbps = speed
    finding = next(f for f in assess(inventory) if f.code == "shared_usb_link")
    assert finding.severity == "info"


def test_metadata_node_is_not_a_second_camera(inventory: Inventory) -> None:
    inventory.devices = [d for d in inventory.devices if d.id != "1-1.1.2"]
    device, endpoint = inventory.resolve("/dev/video1")
    assert endpoint is not None and endpoint.role == "metadata"
    assert device.id == "1-1.1.1"
    assert not any(f.code == "shared_usb_link" for f in assess(inventory))


def test_access_failure_is_scoped_to_relevant_nodes(inventory: Inventory) -> None:
    _, image = inventory.resolve("/dev/video0")
    _, metadata = inventory.resolve("/dev/video1")
    assert image is not None and metadata is not None
    image.writable = False
    metadata.readable = False
    errors = [f for f in assess(inventory) if f.severity == "error"]
    assert len(errors) == 1
    assert "/dev/video0" in errors[0].message
    assert "/dev/video1" not in errors[0].message


def test_video_interface_without_os_endpoint_is_not_assumed_absent() -> None:
    inventory = Inventory([Device("1-1", "Camera", video_interface=True)])
    (finding,) = assess(inventory)
    assert finding.code == "video_endpoint_missing"
    assert finding.severity == "warning"
    assert "vendor SDK" in finding.hint


def test_metadata_only_device_reports_missing_image_endpoint() -> None:
    inventory = Inventory([
        Device(
            "1-1",
            "Camera",
            video_interface=True,
            endpoints=[Endpoint("/dev/video1", "video", "metadata", True, True)],
        )
    ])
    (finding,) = assess(inventory)
    assert finding.code == "video_endpoint_missing"
    assert "no image endpoint" in finding.message


def test_query_failure_is_unknown_not_healthy(inventory: Inventory) -> None:
    _, endpoint = inventory.resolve("/dev/video0")
    assert endpoint is not None
    endpoint.role = "unknown"
    endpoint.query_error = "Not a supported V4L2 query"
    assert any(f.code == "video_role_unknown" for f in assess(inventory))


def test_colliding_alias_is_rejected_even_for_the_symlink_winner(inventory: Inventory) -> None:
    with pytest.raises(ValueError, match="matches 2 endpoints"):
        inventory.resolve("/dev/v4l/by-id/usb-camera-SN0001-video-index0")


@pytest.mark.parametrize("claimed", [False, True])
def test_duplicate_identity_rejects_by_id_with_only_winner_alias(inventory: Inventory, claimed: bool) -> None:
    loser, loser_endpoint = inventory.resolve("/dev/video0")
    winner, winner_endpoint = inventory.resolve("/dev/video2")
    assert loser_endpoint is not None and winner_endpoint is not None
    loser_endpoint.aliases = [a for a in loser_endpoint.aliases if "/by-id/" not in a.path]
    shared = winner_endpoint.aliases[0]
    shared.claimed = claimed

    with pytest.raises(ValueError, match="duplicate device identity"):
        inventory.resolve(shared.path)

    assert shared not in inventory.unique_aliases(winner, winner_endpoint)
    for device, endpoint, port in ((loser, loser_endpoint, "1.1"), (winner, winner_endpoint, "1.2")):
        assert inventory.resolve(device.id) == (device, None)
        assert inventory.resolve(endpoint.path) == (device, endpoint)
        assert inventory.resolve(f"/dev/v4l/by-path/port-{port}") == (device, endpoint)


def test_unique_by_id_identity_still_resolves(inventory: Inventory) -> None:
    device, endpoint = inventory.resolve("/dev/video2")
    assert endpoint is not None
    inventory.devices = [d for d in inventory.devices if d.id != "1-1.1.1"]
    assert inventory.resolve(endpoint.aliases[0].path) == (device, endpoint)


@pytest.mark.parametrize("selector", ["missing", "/dev/video999"])
def test_missing_selector_is_rejected(inventory: Inventory, selector: str) -> None:
    with pytest.raises(ValueError, match="matches 0 endpoints"):
        inventory.resolve(selector)


def test_unique_path_and_device_id_resolve(inventory: Inventory) -> None:
    device, endpoint = inventory.resolve("/dev/v4l/by-path/port-1.1")
    assert endpoint is not None and endpoint.path == "/dev/video0"
    assert inventory.resolve(device.id) == (device, None)


def test_stale_alias_is_rejected(inventory: Inventory) -> None:
    device, endpoint = inventory.resolve("/dev/video0")
    assert endpoint is not None
    endpoint.aliases = [Alias("/dev/v4l/by-path/stale", "/dev/video9")]
    with pytest.raises(ValueError, match="no longer points"):
        inventory.resolve("/dev/v4l/by-path/stale")
    assert inventory.unique_aliases(device, endpoint) == []


def test_duplicate_serial_uses_port_bound_paths(inventory: Inventory) -> None:
    device, endpoint = inventory.resolve("/dev/video2")
    assert endpoint is not None
    aliases = inventory.unique_aliases(device, endpoint)
    assert [a.path for a in aliases] == ["/dev/v4l/by-path/port-1.2"]
    # This still holds if udev claim metadata is missing for the first camera.
    inventory.devices[4].endpoints[0].aliases.clear()
    assert inventory.unique_aliases(device, endpoint) == aliases


def test_by_id_needs_claims_and_unique_reported_identity(inventory: Inventory) -> None:
    device, endpoint = inventory.resolve("/dev/ttyACM0")
    assert endpoint is not None
    alias = Alias("/dev/serial/by-id/arm-a", endpoint.path)
    endpoint.aliases = [alias]
    assert inventory.unique_aliases(device, endpoint) == []
    alias.claimed = True
    assert inventory.unique_aliases(device, endpoint) == [alias]


def test_duplicate_serial_is_detected_without_alias_claims(inventory: Inventory) -> None:
    for device in inventory.devices:
        for endpoint in device.endpoints:
            endpoint.aliases.clear()
    finding = next(f for f in assess(inventory) if f.code == "ambiguous_identity")
    assert "same model and serial" in finding.message


def test_pure_checks_do_not_mutate_evidence(inventory: Inventory) -> None:
    previous = copy.deepcopy(inventory)
    assess(inventory)
    assert inventory == previous


@pytest.mark.parametrize("memory", [None, 0, 16, 256])
def test_usbfs_memory_does_not_infer_a_problem(inventory: Inventory, memory: int | None) -> None:
    expected = assess(inventory)
    inventory.usbfs_memory_mb = memory
    assert assess(inventory) == expected
    assert not Report(inventory, expected).has_errors


def test_cycles_do_not_hang_ancestry_or_checks(inventory: Inventory) -> None:
    inventory.devices[2].parent_id = "1-1.1"
    assert len(inventory.ancestors(inventory.devices[4])) == 2
    assert len(assess(inventory)) == 2


def test_public_api_returns_typed_data_without_printing(
    inventory: Inventory,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(api, "collect_inventory", lambda: inventory)
    report = diagnose()
    assert report.inventory is inventory
    assert report.schema_version == 1
    assert report.platform == "linux"
    assert capsys.readouterr().out == ""
    document = report.to_dict()
    assert json.loads(json.dumps(document)) == document
    serialized = json.loads(json.dumps(document))
    assert serialized["inventory"]["devices"][4]["endpoints"][0]["role"] == "capture"
    assert serialized["findings"][1]["hub_id"] == "1-1.1"


def test_selected_api_retains_topology_but_limits_findings(
    inventory: Inventory, monkeypatch: pytest.MonkeyPatch
) -> None:
    inventory.usbfs_memory_mb = 16
    inventory.devices[-1].endpoints = [Endpoint("/dev/ttyUSB9", "serial")]
    monkeypatch.setattr(api, "collect_inventory", lambda: inventory)
    report = diagnose("/dev/video0")
    assert report.selected_device_id == "1-1.1.1"
    assert report.selected_endpoint == "/dev/video0"
    assert report.inventory is inventory
    assert report.inventory.usbfs_memory_mb == 16
    assert all("1-1.1.1" in f.devices for f in report.findings)
    assert not report.has_errors
    assert diagnose("/dev/ttyUSB9").has_errors


@pytest.mark.parametrize("platform", ["darwin", "win32"])
@pytest.mark.parametrize("selector", [None, "/dev/video0"])
def test_api_preserves_platform_for_a_future_collector(
    inventory: Inventory, monkeypatch: pytest.MonkeyPatch, platform: str, selector: str | None
) -> None:
    # Fake only collection: platform dispatch remains Linux-only in production.
    monkeypatch.setattr(api.sys, "platform", platform)
    monkeypatch.setattr(api, "collect_inventory", lambda: inventory)
    report = diagnose(selector)
    assert report.platform == platform
    assert report.to_dict()["platform"] == platform
    assert TypeAdapter(Report).validate_python(report.to_dict()).platform == platform


@pytest.mark.parametrize("memory", [None, 0, 16, 256])
def test_report_can_be_a_pydantic_or_fastapi_response(inventory: Inventory, memory: int | None) -> None:
    inventory.usbfs_memory_mb = memory
    report = Report(inventory, assess(inventory))
    adapter = TypeAdapter(Report)
    assert adapter.validate_python(report.to_dict()).to_dict() == report.to_dict()
    assert adapter.json_schema()["title"] == "Report"


def test_reports_without_usbfs_memory_remain_readable(inventory: Inventory) -> None:
    document = json.loads(json.dumps(Report(inventory, []).to_dict()))
    del document["inventory"]["usbfs_memory_mb"]
    report = TypeAdapter(Report).validate_python(document)
    assert report.inventory.usbfs_memory_mb is None


def test_serialized_report_is_detached_from_evidence(inventory: Inventory) -> None:
    report = Report(inventory, assess(inventory))
    document = report.to_dict()
    report.inventory.devices.clear()
    assert json.loads(json.dumps(document))["inventory"]["devices"]


def test_alias_collision_between_endpoints_of_one_device_is_detected(inventory: Inventory) -> None:
    device, endpoint = inventory.resolve("/dev/video0")
    assert endpoint is not None
    metadata = device.endpoints[1]
    metadata.aliases = [Alias("/dev/v4l/by-path/port-1.1", endpoint.path, claimed=True)]
    collisions = [f for f in assess(inventory) if f.code == "ambiguous_identity"]
    assert any(f.devices == [device.id] for f in collisions)


def test_collection_failure_is_not_an_empty_healthy_report(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api, "collect_inventory", lambda: Inventory(errors=["sysfs unavailable"]))
    report = diagnose()
    assert report.has_errors
    assert report.inventory.errors == ["sysfs unavailable"]


@pytest.mark.parametrize("platform", ["darwin", "win32"])
def test_unsupported_hosts_fail_explicitly(platform: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api.sys, "platform", platform)
    with pytest.raises(NotImplementedError, match="Linux only"):
        collect_inventory()


def test_import_does_not_load_collectors_renderers_or_sdks() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
import physicalai.diagnostics
for name in ('fcntl', 'cv2', 'torch', 'pynokhwa', 'physicalai.capture', 'physicalai.robot',
             'physicalai.diagnostics.linux', 'physicalai.diagnostics.render'):
    assert name not in sys.modules, name
""",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
