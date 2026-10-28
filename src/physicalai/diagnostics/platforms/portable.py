# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Portable baseline for platforms without Linux sysfs/udev."""

from __future__ import annotations

import json

from physicalai.diagnostics.models import Capability, CollectionError, Device, Endpoint, Inventory


class PortableCollector:
    """Enumerate serial and UVC candidates without opening ports or streams."""

    def __init__(self, platform: str) -> None:
        """Choose the reported native platform without importing native APIs."""
        self.platform = platform

    def collect(self) -> Inventory:
        """Return portable evidence and explicitly disclose native enrichment gaps."""
        result = Inventory(platform=self.platform)
        result.capabilities = {
            "usb_topology": Capability(status="unsupported", detail="Native topology enrichment is not implemented."),
            "system_logs": Capability(status="unsupported", detail="Native system-log collection is not implemented."),
            "device_events": Capability(status="available", detail="Polling; brief unplug/replug can be missed."),
            "video_formats": Capability(status="unavailable", detail="No passive native mode query is available."),
        }
        self._serial(result)
        self._cameras(result)
        return result

    @staticmethod
    def _serial(result: Inventory) -> None:
        try:
            from serial.tools.list_ports import comports  # noqa: PLC0415

            for port in comports():
                result.devices.append(
                    Device(
                        id=f"serial:{port.device}",
                        name=port.description or port.device,
                        serial=port.serial_number,
                        vendor_id=f"{port.vid:04x}" if port.vid is not None else "",
                        product_id=f"{port.pid:04x}" if port.pid is not None else "",
                        location=port.location or "",
                        endpoints=[Endpoint(id=f"serial:{port.device}", path=port.device, kind="serial")],
                        properties={"identity_source": "pyserial; native path persistence is not verified"},
                    )
                )
            result.capabilities["serial_inventory"] = Capability(
                status="available", detail="pyserial enumeration only."
            )
        except Exception as exc:  # noqa: BLE001 - optional SDK boundary; retain failure as evidence
            result.errors.append(CollectionError(source="serial", message=str(exc), essential=True))
            result.capabilities["serial_inventory"] = Capability(
                status="unavailable", detail="Install physicalai[doctor]."
            )

    @staticmethod
    def _cameras(result: Inventory) -> None:
        try:
            import pynokhwa  # noqa: PLC0415

            for info in pynokhwa.query(only_usable=False):
                identity = info.device_meta_json or {}
                result.devices.append(
                    Device(
                        id=f"camera:{info.index}",
                        name=info.name,
                        serial=str(identity["serial"]) if identity.get("serial") else None,
                        endpoints=[
                            Endpoint(
                                id=f"camera:{info.index}",
                                path=f"index:{info.index}",
                                kind="video",
                                role="capture",
                                properties={"backend_identity": json.dumps(identity), "backend": "omnicamera"},
                            )
                        ],
                        properties={"identity_source": "pynokhwa; physical-device grouping is not verified"},
                    )
                )
            result.capabilities["camera_inventory"] = Capability(
                status="available", detail="pynokhwa enumeration only."
            )
        except Exception as exc:  # noqa: BLE001 - optional native SDK boundary; preserve enumeration failure
            result.errors.append(CollectionError(source="cameras", message=str(exc), essential=True))
            result.capabilities["camera_inventory"] = Capability(status="unavailable", detail=str(exc))
