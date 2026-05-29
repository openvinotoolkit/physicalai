# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Factory convenience functions for config-driven camera creation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from physicalai.capture.camera import CameraType

if TYPE_CHECKING:
    from physicalai.capture.camera import Camera


def create_camera(camera_type: str, *, shared: bool = False, **kwargs) -> Camera:  # noqa: ANN003
    """Create a camera by type name.

    Args:
        camera_type: Camera type — one of ``"uvc"``, ``"ip"``,
            ``"realsense"``, ``"basler"``, ``"genicam"``.
            Case-insensitive.
        shared: If True, wrap the camera in a :class:`SharedCamera`
            (iceoryx2 shared-memory transport). Requires the
            ``transport`` extra.
        **kwargs: Forwarded to the camera constructor.

    Returns:
        A new camera instance.

    Raises:
        ValueError: If *camera_type* is not a recognised name.
    """
    camera_type = camera_type.lower()

    if shared:
        from physicalai.capture.transport import SharedCamera  # noqa: PLC0415

        return SharedCamera(camera_type, **kwargs)

    if camera_type == CameraType.UVC:
        from physicalai.capture.cameras.uvc import UVCCamera  # noqa: PLC0415

        return UVCCamera(**kwargs)

    if camera_type == CameraType.IP:
        from physicalai.capture.cameras.ip import IPCamera  # noqa: PLC0415

        return IPCamera(**kwargs)

    if camera_type == CameraType.REALSENSE:
        from physicalai.capture.cameras.realsense import RealSenseCamera  # noqa: PLC0415

        return RealSenseCamera(**kwargs)

    if camera_type == CameraType.BASLER:
        from physicalai.capture.cameras.basler import BaslerCamera  # noqa: PLC0415

        return BaslerCamera(**kwargs)

    if camera_type == CameraType.GENICAM:
        from physicalai.capture.cameras.genicam import GenicamCamera  # noqa: PLC0415

        return GenicamCamera(**kwargs)

    msg = f"Unknown camera type {camera_type!r}. Expected one of: {', '.join(CameraType)}"
    raise ValueError(msg)


# ─── Multi-camera construction ────────────────────────────────────────────────


def select_cameras_interactive(
    width: int,
    height: int,
    fps: int,
) -> dict[str, Camera]:
    """Discover cameras and let the user pick interactively via stdin.

    Uses :func:`~physicalai.capture.discover_all` to enumerate available
    devices, then presents a numbered menu.  The user selects cameras
    one at a time and assigns each a name.

    Args:
        width: Requested frame width.
        height: Requested frame height.
        fps: Requested frame rate.

    Returns:
        Dict mapping user-chosen names to SharedCamera instances.
        Empty dict if no cameras found or none selected.
    """
    from physicalai.capture.discovery import discover_all  # noqa: PLC0415

    logger.info("Discovering cameras...")
    all_devices = discover_all()

    flat: list[tuple[str, str, str]] = []
    for driver, devices in all_devices.items():
        flat.extend((driver, dev.device_id, f"{driver}: {dev.name or dev.device_id}") for dev in devices)

    if not flat:
        logger.warning("No cameras found. Continuing without cameras.")
        return {}

    logger.info("Available cameras:")
    for i, (_, _, display) in enumerate(flat):
        logger.info("  [{}] {}", i, display)

    cameras: dict[str, Camera] = {}
    while True:
        try:
            choice = input("Select camera index (or 'done' to finish): ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if choice.lower() in {"done", "d", ""}:
            break
        try:
            idx = int(choice)
            if idx < 0 or idx >= len(flat):
                logger.warning("  Invalid index. Choose 0-{}.", len(flat) - 1)
                continue
        except ValueError:
            logger.warning("  Enter a number or 'done'.")
            continue

        try:
            name = input("  Name for this camera (e.g. overhead, arm, front): ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not name:
            name = f"camera_{len(cameras)}"

        driver, device_id, _ = flat[idx]
        kwargs: dict = {"width": width, "height": height, "fps": fps}
        if driver == "realsense":
            kwargs["serial_number"] = device_id
        else:
            kwargs["device"] = device_id
        cameras[name] = create_camera(driver, shared=True, **kwargs)
        logger.info("  Added '{}' ({}:{})", name, driver, device_id)

    return cameras
