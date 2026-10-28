# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Pure diagnostic rules over collected evidence; no hardware or subprocesses."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from physicalai.diagnostics.identity import alias_claims
from physicalai.diagnostics.models import Finding, Report

if TYPE_CHECKING:
    from physicalai.diagnostics.models import Device, Inventory


def assess(inventory: Inventory) -> Report:
    """Explain confirmed faults and risks without equating unknown with healthy.

    Returns:
        A report containing independent, evidence-backed findings.
    """
    return Report(
        inventory=inventory,
        findings=[
            *_alias_findings(inventory),
            *_device_findings(inventory),
            *_link_findings(inventory),
            *_function_findings(inventory),
            *_host_findings(inventory),
        ],
    )


def _alias_findings(inventory: Inventory) -> list[Finding]:
    findings: list[Finding] = []
    for path, endpoints in sorted(alias_claims(inventory).items()):
        if len(endpoints) > 1:
            findings.append(
                Finding(
                    check_id="identity.alias_collision",
                    severity="WARN",
                    subject=path,
                    summary=f"{len(endpoints)} endpoints claim the same alias.",
                    evidence=[ep.path for ep in endpoints],
                    action="Use a unique physical-port/stream selector; do not rely on the current symlink winner.",
                    verification="Inspect each endpoint's alias claims after reconnecting.",
                )
            )
    return findings


def _device_findings(inventory: Inventory) -> list[Finding]:
    findings: list[Finding] = []
    serials: dict[tuple[str, str, str], list[Device]] = defaultdict(list)
    for device in inventory.devices:
        relevant = [
            ep for ep in device.endpoints if device.kind == "device" and (ep.kind == "serial" or ep.role == "capture")
        ]
        if device.serial and device.kind == "device":
            serials[device.vendor_id, device.product_id, device.serial].append(device)
        if relevant and not device.serial and device.kind == "device":
            findings.append(
                Finding(
                    check_id="identity.no_serial",
                    severity="WARN",
                    subject=device.id,
                    summary="No reported device serial; physical identity cannot be verified.",
                    evidence=[ep.path for ep in relevant],
                    action="Choose a port-bound selector and label the slot. "
                    "Identical replacements remain unverifiable.",
                    verification="Reconnect in the same slot and inspect the port selector.",
                )
            )
        findings.extend(
            Finding(
                check_id="access.endpoint",
                severity="FAIL",
                subject=endpoint.id,
                summary=f"Current user cannot read/write {endpoint.path}.",
                evidence=[f"readable={endpoint.readable}, writable={endpoint.writable}"],
                action="Review device ACLs/group membership and vendor rules; do not run the application as root.",
                verification="Check access as the same user that runs Runtime.",
            )
            for endpoint in relevant
            if endpoint.readable is False or endpoint.writable is False
        )
        findings.extend(
            Finding(
                check_id="access.in_use",
                severity="WARN",
                subject=endpoint.id,
                summary=f"{endpoint.path} is open by visible processes; exclusive capture/serial access can conflict.",
                evidence=[f"pid={holder.pid} name={holder.name}" for holder in endpoint.holders],
                action="Close the existing owner's connection only when appropriate; "
                "doctor never stops other processes.",
                verification="Inspect again after the owner has released the endpoint.",
            )
            for endpoint in relevant
            if endpoint.holders
        )
        accessible = [ep.path for ep in relevant if ep.readable is True and ep.writable is True]
        if accessible:
            findings.append(
                Finding(
                    check_id="access.available",
                    severity="PASS",
                    subject=device.id,
                    summary="Current user has read/write access to the relevant endpoints.",
                    evidence=accessible,
                    verification="Permissions do not establish streaming or robot-protocol health.",
                )
            )
        if any(ep.kind == "video" for ep in device.endpoints) and device.speed_mbps == _USB2_SPEED:
            findings.append(
                Finding(
                    check_id="usb.usb2_camera",
                    severity="INFO",
                    subject=device.id,
                    summary="Camera is using USB 2 (480 Mb/s); supported modes and concurrency can be limited.",
                    evidence=[device.location],
                    action="Check advertised modes; use a USB 3 path if the device supports it.",
                    verification="Inspect link speed and advertised modes after reconnecting.",
                )
            )
        findings.extend(_zed_findings(inventory, device))
    findings.extend(
        Finding(
            check_id="identity.duplicate_serial",
            severity="WARN",
            subject="|".join(sorted(d.id for d in devices)),
            summary="Multiple physical devices report the same model/serial identity.",
            evidence=[d.id for d in devices],
            action="Use distinct port-bound selectors.",
            verification="Inspect identities with both devices attached.",
        )
        for devices in serials.values()
        if len(devices) > 1
    )
    return findings


_USB2_SPEED = 480
_COMMON_UART_IDS = {("1a86", "7523"), ("1a86", "55d4"), ("10c4", "ea60"), ("0403", "6001"), ("0403", "6015")}


def _zed_findings(inventory: Inventory, device: Device) -> list[Finding]:
    if (device.vendor_id, device.product_id) != ("2b03", "f681"):
        return []
    videos = [
        d
        for d in inventory.devices
        if d.vendor_id == "2b03"
        and (
            any(ep.role == "capture" for ep in d.endpoints)
            or "0e" in d.properties.get("interface_classes", "").split(",")
        )
    ]
    if device.serial and any(video.serial == device.serial for video in videos):
        return []
    if videos:
        return [
            Finding(
                check_id="usb.zed_video_unpaired",
                severity="WARN",
                subject=device.id,
                summary="Stereolabs video is present, but its pairing with this ZED HID interface is unverified.",
                evidence=[video.id for video in videos],
                action="Compare serials and port chains; "
                "another camera's video must not hide an unpaired HID interface.",
                verification="Inspect this unit's HID and SuperSpeed video functions together.",
            )
        ]
    return [
        Finding(
            check_id="usb.zed_video_missing",
            severity="FAIL",
            subject=device.id,
            summary="ZED Mini HID is present, but no Stereolabs video interface was found.",
            evidence=[device.location],
            action="Try a direct USB 3 port and verified SuperSpeed cable, bypassing USB 2 hubs.",
            verification="Look for a 5000 Mb/s Stereolabs video device and a capture endpoint.",
        )
    ]


def _function_findings(inventory: Inventory) -> list[Finding]:
    findings: list[Finding] = []
    for device in inventory.devices:
        if device.properties.get("authorized") == "0":
            findings.append(
                Finding(
                    check_id="usb.authorization",
                    severity="FAIL",
                    subject=device.id,
                    summary="USB device is not authorized by the OS.",
                    evidence=[device.location],
                    action="Review USB authorization policy with the administrator; doctor never authorizes devices.",
                    verification="Inspect authorization state and function endpoints after policy review.",
                )
            )
        elif (device.vendor_id, device.product_id) in _COMMON_UART_IDS and not any(
            ep.kind == "serial" for ep in device.endpoints
        ):
            findings.append(
                Finding(
                    check_id="usb.serial_endpoint_missing",
                    severity="WARN",
                    subject=device.id,
                    summary="A common USB UART adapter is present but no serial endpoint is visible.",
                    evidence=[
                        device.location,
                        device.properties.get("interface_drivers", "No interface driver recorded"),
                    ],
                    action="Inspect serial driver binding and container/device visibility. "
                    "The adapter does not prove robot model.",
                    verification="Look for its tty endpoint and rerun device discovery.",
                )
            )
        if "0e" in device.properties.get("interface_classes", "").split(",") and not any(
            ep.kind == "video" for ep in device.endpoints
        ):
            findings.append(
                Finding(
                    check_id="usb.video_endpoint_missing",
                    severity="WARN",
                    subject=device.id,
                    summary="USB video interfaces are present but no OS video endpoint is visible.",
                    evidence=[
                        device.location,
                        device.properties.get("interface_drivers", "No interface driver recorded"),
                    ],
                    action="Check uvcvideo binding/container visibility "
                    "and whether a vendor SDK intentionally owns the interface.",
                    verification="Inspect expected backend discovery; "
                    "absence of V4L2 alone does not prove the device unusable.",
                )
            )
    return findings


def _link_findings(inventory: Inventory) -> list[Finding]:
    devices = {device.id: device for device in inventory.devices}
    shared: dict[str, set[str]] = defaultdict(set)
    for device in inventory.devices:
        if not any(ep.role == "capture" for ep in device.endpoints):
            continue
        parent = device.parent_id
        visited: set[str] = set()
        while parent is not None and parent in devices and parent not in visited:
            visited.add(parent)
            ancestor = devices[parent]
            if ancestor.kind == "hub" and ancestor.properties.get("root_hub") != "true":
                shared[parent].add(device.id)
            parent = ancestor.parent_id
    return [
        Finding(
            check_id="usb.shared_uplink",
            severity="WARN" if devices[key].speed_mbps == _USB2_SPEED else "INFO",
            subject=key,
            summary=f"{len(cameras)} cameras share this hub's upstream link; "
            "congestion risk is not measured utilization.",
            evidence=[devices[key].location, f"link={devices[key].speed_mbps} Mb/s", *sorted(cameras)],
            action="Compare individual/concurrent streams. "
            "If they degrade, distribute cameras across separate root links/controllers.",
            verification="Use the same wire modes for both tests; "
            "a powered hub does not increase its uplink bandwidth.",
        )
        for key, cameras in shared.items()
        if len(cameras) > 1
    ]


def _host_findings(inventory: Inventory) -> list[Finding]:
    findings: list[Finding] = []
    if "usbfs_memory_mb" in inventory.settings:
        findings.append(
            Finding(
                check_id="usb.usbfs_memory",
                severity="INFO",
                subject="host",
                summary=f"usbfs buffer allocation limit: {inventory.settings['usbfs_memory_mb']} MiB "
                "(0 means kernel maximum).",
                action="Only investigate tuning with usbfs/libusb allocation-failure evidence; "
                "this is not bus bandwidth.",
                verification="Compare the same concurrent workload before/after any operator-approved change.",
            )
        )
    for name, capability in inventory.capabilities.items():
        if capability.status != "available":
            findings.append(
                Finding(
                    check_id=f"capability.{name}",
                    severity="SKIP",
                    subject="host",
                    summary=capability.detail,
                    evidence=[capability.status],
                )
            )
    findings.extend(
        Finding(
            check_id="collection.error",
            severity="SKIP",
            subject=error.source,
            summary=error.message,
            action="Resolve the collection limitation and rerun diagnostics.",
        )
        for error in inventory.errors
    )
    if not inventory.devices:
        findings.append(
            Finding(
                check_id="inventory.empty",
                severity="INFO",
                subject="host",
                summary="No devices were enumerated; this does not verify expected hardware is absent.",
            )
        )
    return findings
