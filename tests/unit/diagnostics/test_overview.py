# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from physicalai.diagnostics.checks import assess
from physicalai.diagnostics.models import Alias, Device, Endpoint, Finding, Inventory, VideoMode
from physicalai.diagnostics.overview import render_payloads, render_shared_links, render_tree
from physicalai.diagnostics.render import render_inspection, render_inventory, render_report
from physicalai.diagnostics.topology import USBTopology


def inventory() -> Inventory:
    mode = VideoMode(width=640, height=480, fps=30, pixel_format="YUYV")
    return Inventory(platform="linux", devices=[
        Device(id="controller", name="Controller", kind="controller"),
        Device(id="root", name="Root hub", kind="hub", parent_id="controller", speed_mbps=480,
               properties={"root_hub": "true"}),
        Device(id="outer", name="Outer hub", kind="hub", parent_id="root", speed_mbps=480,
               properties={"usb_path": "3-4"}),
        Device(id="inner", name="Inner hub", kind="hub", parent_id="outer", speed_mbps=480,
               properties={"usb_path": "3-4.1"}),
        Device(id="camera", name="Color camera", parent_id="inner", speed_mbps=480, serial="CAMERA",
               properties={"usb_path": "3-4.1.2"}, endpoints=[
                   Endpoint(id="capture", path="/dev/video38", kind="video", role="capture", current_mode=mode,
                            aliases=[Alias(path="/dev/v4l/by-path/color", binding="port", target="/dev/video38")]),
                   Endpoint(id="meta", path="/dev/video39", kind="video", role="metadata"),
               ]),
        Device(id="robot", name="Robot adapter", parent_id="inner", speed_mbps=12,
               properties={"usb_path": "3-4.1.4"}, endpoints=[Endpoint(id="serial", path="/dev/ttyUSB0", kind="serial")]),
        Device(id="mouse", name="Mouse", parent_id="inner", speed_mbps=12, properties={"usb_path": "3-4.1.3"}),
        Device(id="adapter", name="Other adapter", parent_id="outer", speed_mbps=12, properties={"usb_path": "3-4.2"}),
        Device(id="other-port", name="Different root-port hub", kind="hub", parent_id="root", speed_mbps=480,
               properties={"usb_path": "3-1"}),
        Device(id="other-camera", name="Other camera", parent_id="other-port", speed_mbps=480),
    ])


def sample(subject: str = "/dev/video38", check_id: str = "stream.concurrent") -> Finding:
    return Finding(check_id=check_id, severity="PASS", subject=subject, summary="streamed", details={
        "requested": {"width": 640, "height": 480, "fps": 30, "pixel_format": "mjpeg"},
        "metrics": {"received_fps": 20, "payload_bytes_per_second": 1_000_000},
        "payload_demand_bytes_per_second": 1_500_000,
    })


def test_default_tree_has_real_branches_ports_and_shared_physical_device_counts() -> None:
    text = render_inventory(inventory())
    assert "├──" in text and "└──" in text and "│" in text
    assert "Port 2: Color camera" in text
    assert "3 devices share uplink" in text  # Camera + robot + mouse, NOT camera metadata.
    assert "root ports independent" in text
    assert "/dev/video39" not in text
    assert "18.43 MB/s est." in text and "not measured" in text
    assert "60 MB/s" in text


def test_inspection_explains_all_upstream_hubs_and_only_their_peers() -> None:
    snapshot = inventory()
    topology = USBTopology(snapshot)
    assert [device.id for device in topology.route("camera")] == ["controller", "root", "outer", "inner", "camera"]
    assert {d.id for d in topology.shared_devices("inner", excluding="camera")} == {"robot", "mouse"}
    peers = render_shared_links(snapshot, "camera")
    assert "Robot adapter" in peers and "Mouse" in peers and "Other adapter" in peers
    assert "Other camera" not in peers  # Different root port is NOT the same shared external uplink.
    selected = snapshot.devices[4]
    text = render_inspection(snapshot, selected, selected.endpoints[0])
    assert "<selected>" in text and "Shares with:" in text
    assert "Other camera" not in text


def test_measured_view_uses_encoded_payload_and_keeps_signaling_separate() -> None:
    snapshot = inventory()
    finding = sample()
    text = render_payloads(snapshot, [finding])
    assert "1.50 MB/s est." in text and "20.0 FPS; 1.00 MB/s" in text
    assert "60 MB/s" in text and "NOT usable/allocated" in text
    tree = render_tree(snapshot, findings=[finding])
    assert "need 1.50 MB/s est.; getting 1.00 MB/s" in tree
    assert "other traffic unmeasured" in tree


def test_never_sum_individual_trials_from_different_time_windows() -> None:
    rows = [sample(check_id="stream.individual"), sample("other", "stream.individual")]
    assert "tested stream(s)" not in render_tree(inventory(), findings=rows)


def test_compact_report_omits_routine_pass_and_skip_dumps_but_verbose_retains_them() -> None:
    report = assess(inventory())
    report.findings.append(Finding(check_id="collection.error", severity="SKIP", subject="private-node",
                                   summary="very detailed diagnostic error"))
    short = render_report(report)
    assert "very detailed diagnostic error" not in short
    assert "very detailed diagnostic error" in render_report(report, verbose=True)
    assert "Needs attention" in short


def test_cycles_and_terminal_controls_do_not_break_tree_rendering() -> None:
    snapshot = inventory()
    snapshot.devices[0].parent_id = "inner"
    snapshot.devices[4].name = "camera\x1b[31m"
    text = render_tree(snapshot)
    assert "[unresolved ancestry]" in text and "\x1b" not in text
    assert "\\x1b" in text


def test_serial_inspection_does_not_invent_uart_traffic_or_show_a_camera_table() -> None:
    snapshot = inventory()
    serial = snapshot.devices[5].endpoints[0]
    text = render_payloads(snapshot, endpoint=serial)
    assert "protocol/workload dependent" in text and "Getting: not measured" in text
    assert "Camera throughput" not in text
