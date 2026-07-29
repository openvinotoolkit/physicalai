# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Factory convenience functions for config-driven camera creation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger

from physicalai.capture.camera import CameraType

if TYPE_CHECKING:
    from physicalai.capture.camera import Camera

_SHARED_TRANSPORT_KEYS = frozenset({
    "zero_copy",
    "validate_on_connect",
    "overwrite_settings",
    "idle_timeout",
    "service_name",
    "color_mode",
})

# CameraType token → public class_path for ``create_camera(..., shared=True)``.
# Static by necessity: subscriber hosts have no vendor SDK, so nothing here may
# import a driver. Stub types (ip, genicam) are absent so shared spawn cannot
# claim phantom coverage; test_factory.py checks values against @export_config.
_SHAREABLE_CLASS_PATHS: dict[str, str] = {
    "uvc": "physicalai.capture.UVCCamera",
    "realsense": "physicalai.capture.RealSenseCamera",
    "basler": "physicalai.capture.BaslerCamera",
}


def create_camera(camera_type: str, *, shared: bool = False, **kwargs: Any) -> Camera:  # noqa: ANN401
    """Create a camera by type name.

    Args:
        camera_type: Camera type — one of ``"uvc"``, ``"ip"``,
            ``"realsense"``, ``"basler"``, ``"genicam"``.
            Case-insensitive.
        shared: If True, wrap the camera in a :class:`SharedCamera`
            (iceoryx2 shared-memory transport). Requires the
            ``transport`` extra. Only backends with a real shared registry
            entry (``uvc``, ``realsense``, ``basler``) support derived
            ``service_name``; stub types must use
            :meth:`SharedCamera.from_config` with an explicit
            ``service_name`` once a driver exists.
        **kwargs: Forwarded to the camera constructor. When *shared* is
            True, SharedCamera transport knobs (``zero_copy``,
            ``validate_on_connect``, ``overwrite_settings``,
            ``idle_timeout``, ``service_name``, ``color_mode``) are peeled
            off for the subscriber; remaining kwargs become
            ``camera.init_args``.

    Returns:
        A new camera instance.

    Raises:
        ValueError: If *camera_type* is not a recognised name, or *shared*
            is True for a type without shared service-name derivation.
    """
    camera_type = camera_type.lower()

    if shared:
        from physicalai.capture.transport import SharedCamera  # noqa: PLC0415

        class_path = _SHAREABLE_CLASS_PATHS.get(camera_type)
        if class_path is None:
            if camera_type in {t.value for t in CameraType}:
                shareable = ", ".join(sorted(_SHAREABLE_CLASS_PATHS))
                msg = (
                    f"camera type {camera_type!r} does not support shared=True "
                    f"(no shareable driver for service-name derivation); "
                    f"shareable types: {shareable}. "
                    "Use SharedCamera.from_config(..., service_name=...) once a "
                    "driver exists, or create_camera without shared."
                )
                raise ValueError(msg)
            msg = f"Unknown camera type {camera_type!r}. Expected one of: {', '.join(CameraType)}"
            raise ValueError(msg)

        transport: dict[str, Any] = {}
        init_args = dict(kwargs)
        for key in _SHARED_TRANSPORT_KEYS:
            if key in init_args:
                transport[key] = init_args.pop(key)

        return SharedCamera(
            camera={"class_path": class_path, "init_args": init_args},
            **transport,
        )

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
    *,
    shared: bool = True,
) -> dict[str, Camera]:
    """Discover cameras and let the user pick interactively via stdin.

    Uses :func:`~physicalai.capture.discover_all` to enumerate available
    devices, then presents a numbered menu.  The user selects cameras
    one at a time and assigns each a name.

    Args:
        width: Requested frame width.
        height: Requested frame height.
        fps: Requested frame rate.
        shared: Wrap each camera in :class:`SharedCamera` when ``True``.

    Returns:
        Dict mapping user-chosen names to camera instances.
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
        cameras[name] = create_camera(driver, shared=shared, **kwargs)
        logger.info("  Added '{}' ({}:{})", name, driver, device_id)

    return cameras
