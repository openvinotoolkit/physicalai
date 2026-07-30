# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import sys
import time
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, cast

import numpy as np
import pynokhwa as omni_camera  # rename omni_camera references

from physicalai.capture.camera import Camera, ColorMode
from physicalai.capture.cameras.uvc._camera_setting import CameraSetting  # noqa: PLC2701
from physicalai.capture.errors import CaptureError, CaptureTimeoutError, NotConnectedError
from physicalai.capture.frame import Frame

if TYPE_CHECKING:
    from physicalai.capture.discovery import DeviceInfo


_MISSING_DEP_PKG = "omni_camera"
_MISSING_DEP_EXTRA = "capture"

_SYSFS_V4L2 = Path("/sys/class/video4linux")


class _UsbIdentity(NamedTuple):
    """USB identity of the device owning a V4L2 node."""

    devpath: str
    """USB device path, e.g. ``3-6.1``. Unique per physical port."""

    model_key: tuple[str, str, str]
    """``(idVendor, idProduct, serial)``.

    An unreported serial is empty, which is itself a collision key: udev omits
    it from the by-id name, so every unit of such a model claims one path.
    """


def _usb_identity(index: int) -> _UsbIdentity | None:
    """Read the USB identity of a V4L2 node from sysfs.

    This is used purely as *evidence about* the ``/dev/v4l/by-id`` paths and
    never as an identifier itself. Those paths are derived from the USB
    iSerial, so they cannot witness their own uniqueness: when a vendor ships
    one serial for every unit of a model, two cameras claim the same by-id
    name and udev materialises it for only one of them.

    Args:
        index: V4L2 node index, i.e. the ``N`` in ``/dev/videoN``.

    Returns:
        Identity of the owning USB device, or None when it cannot be read
        (non-Linux, non-USB device, or unreadable sysfs).
    """
    try:
        path = (_SYSFS_V4L2 / f"video{index}" / "device").resolve(strict=True)
        while not (path / "idVendor").exists():
            if path.parent == path:
                return None
            path = path.parent
        attrs = tuple(
            (path / name).read_text().strip() if (path / name).exists() else ""
            for name in ("idVendor", "idProduct", "serial")
        )
    except OSError:
        return None
    return _UsbIdentity(devpath=path.name, model_key=cast("tuple[str, str, str]", attrs))


class OmniCamera(Camera):
    _POLL_INTERVAL_S = 0.001

    def __init__(
        self,
        *,
        device_id: int | str = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        color_mode: ColorMode = ColorMode.RGB,
    ) -> None:
        super().__init__(color_mode=color_mode)
        self._device_id_raw = device_id
        self._width = width
        self._height = height
        self._fps = fps
        self._color_mode = color_mode
        self._connected = False
        self._sequence = 0
        self._cam: omni_camera.Camera | None = None
        self._last_frame: np.ndarray | None = None

    @staticmethod
    def _resolve_device_info(infos: list[omni_camera.CameraInfo], device_id: int | str) -> omni_camera.CameraInfo:
        # Try unique_id match first for string identifiers.
        if isinstance(device_id, str) and device_id:
            match = next((c for c in infos if c.unique_id and c.unique_id == device_id), None)
            if match is not None:
                return match

        # Fall back to index-based resolution.
        normalized_device_id: int
        if isinstance(device_id, str):
            if device_id.isdecimal():
                normalized_device_id = int(device_id)
            elif device_id.startswith("/dev/video"):
                suffix = device_id.removeprefix("/dev/video")
                if not suffix.isdecimal():
                    msg = f"Invalid device path: {device_id}"
                    raise ValueError(msg)
                normalized_device_id = int(suffix)
            else:
                msg = (
                    "OmniCamera backend does not support device path strings on this platform. "
                    "Use an integer camera index or a stable unique_id instead."
                )
                raise ValueError(msg)
        else:
            normalized_device_id = device_id

        info = next((candidate for candidate in infos if candidate.index == normalized_device_id), None)
        if info is None:
            msg = f"No camera found at index {normalized_device_id}"
            raise CaptureError(msg)
        return info

    def _resolve_format(self) -> omni_camera.CameraFormat:
        if self._cam is None:
            msg = "Camera cannot be opened"
            raise CaptureError(msg)

        fmts = self._cam.get_format_options()
        if not fmts:
            msg = (
                "Camera reports no supported formats. This typically means the device "
                "only outputs formats unsupported by the nokhwa backend (e.g. BGRA from "
                "a virtual camera like OBS Virtual Camera)."
            )
            raise CaptureError(msg)

        for f in fmts:
            if f.width == self._width and f.height == self._height and round(f.frame_rate) == round(self._fps):
                return f

        available = sorted({(f.width, f.height, int(f.frame_rate)) for f in fmts})
        available_str = ", ".join(f"{w}x{h}@{fps}" for w, h, fps in available)
        msg = (
            f"No camera format matching {self._width}x{self._height}@{self._fps}fps. Available formats: {available_str}"
        )
        raise CaptureError(msg)

    def connect(self, timeout: float = 5.0) -> None:
        # On macOS, nokhwa_initialize() fires an async AVFoundation permission
        # request at module import. If we query before that callback resolves,
        # the camera list may be empty. Retry briefly to give the TCC
        # callback time to deliver.
        # We use only_usable=False so that hardware indices match those
        # returned by discover(). Unsupported devices (e.g. BGRA-only
        # virtual cameras) are caught later in _resolve_format().
        query_deadline = time.monotonic() + 2.0
        infos = omni_camera.query(only_usable=False)
        while not infos and time.monotonic() < query_deadline:
            time.sleep(0.1)
            infos = omni_camera.query(only_usable=False)
        info = self._resolve_device_info(infos, self._device_id_raw)

        try:
            self._cam = omni_camera.Camera(info)
            fmt = self._resolve_format()

            self._cam.open(fmt)
        except RuntimeError as exc:
            if "FourCharCode" in str(exc):
                msg = (
                    f"Camera at index {self._device_id_raw} uses an unsupported pixel "
                    "format. This typically indicates a virtual or utility camera "
                    "(e.g. Nikon Webcam Utility, OBS Virtual Camera) that is not "
                    "compatible with the nokhwa backend."
                )
                raise CaptureError(msg) from exc
            raise

        frame_data = None
        seq = self._sequence
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            # result could be None if camera is no connected yet
            result = None
            with contextlib.suppress(Exception):
                result = self._cam.poll_frame_np_with_seq()
            if result is not None:
                frame_data, seq = result
                break
            time.sleep(1.0 / self._fps)

        if frame_data is None:
            self._do_disconnect()
            msg = f"Timed out waiting for first frame after {timeout}s"
            raise CaptureTimeoutError(msg)

        self._last_frame = frame_data
        self._connected = True
        self._sequence = seq

    def _do_disconnect(self) -> None:
        if self._cam is not None:
            self._cam.close()
        self._cam = None
        self._connected = False
        self._last_frame = None

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def device_id(self) -> str:
        return str(self._device_id_raw)

    def read(self, timeout: float = 2.0) -> Frame:
        if not self._connected or self._cam is None:
            err = NotConnectedError()
            raise err

        deadline = time.monotonic() + timeout
        while True:
            try:
                frame_data, seq = self._cam.poll_frame_np_with_seq()
                if frame_data is not None and seq != self._sequence:
                    converted = self._convert_color(frame_data)
                    self._sequence = seq
                    self._last_frame = frame_data
                    return Frame(data=converted, timestamp=time.monotonic(), sequence=self._sequence)
                last_error = None
            except Exception as exc:  # noqa: BLE001
                last_error = exc

            if time.monotonic() >= deadline:
                if last_error is not None:
                    self._do_disconnect()
                    msg = f"Failed to read frame from device {self.device_id} within {timeout}s: {last_error}"
                    raise CaptureError(msg) from last_error
                msg = f"Timed out waiting for frame after {timeout}s"
                raise CaptureTimeoutError(msg)

            time.sleep(self._POLL_INTERVAL_S)

    def read_latest(self) -> Frame:
        if not self._connected or self._cam is None:
            err = NotConnectedError()
            raise err

        try:
            frame_data, seq = self._cam.poll_frame_np_with_seq()
            if frame_data is not None:
                converted = self._convert_color(frame_data)
                self._sequence = seq
                self._last_frame = frame_data
                return Frame(data=converted, timestamp=time.monotonic(), sequence=self._sequence)
        except Exception as exc:
            self._do_disconnect()
            msg = f"Failed to read frame from device: {self.device_id}"
            raise CaptureError(msg) from exc

        if self._last_frame is not None:
            return Frame(
                data=self._convert_color(self._last_frame),
                timestamp=time.monotonic(),
                sequence=self._sequence,
            )

        msg = "No frame available"
        raise CaptureError(msg)

    def _convert_color(self, frame: np.ndarray) -> np.ndarray:
        if self._color_mode == ColorMode.RGB:
            return frame
        if self._color_mode == ColorMode.BGR:
            return frame[:, :, ::-1]
        if self._color_mode == ColorMode.GRAY:
            return np.dot(frame[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
        return frame

    @classmethod
    def discover(cls, *, only_usable: bool = True) -> list[DeviceInfo]:
        from physicalai.capture.discovery import DeviceInfo  # noqa: PLC0415

        infos = omni_camera.query(only_usable=only_usable)
        on_linux = sys.platform.startswith("linux")
        usb = {info.index: _usb_identity(info.index) for info in infos} if on_linux else {}

        if on_linux:
            # V4L2 exposes several /dev/videoN nodes per physical camera
            # (capture, metadata, ...). Every node of one camera resolves to
            # the same USB device path and no two cameras ever share one, so
            # it groups nodes correctly even when by-id paths collide. Keep
            # the lowest node index per device: that is the capture node.
            phys_best: dict[str, omni_camera.CameraInfo] = {}
            for info in infos:
                identity = usb.get(info.index)
                phys_key = identity.devpath if identity else str(info.index)
                if phys_key not in phys_best or info.index < phys_best[phys_key].index:
                    phys_best[phys_key] = info
            infos = list(phys_best.values())

        # Some vendors bake the same USB descriptors into every unit of a
        # model. Both units then claim one /dev/v4l/by-id name, so the by-id
        # that does exist denotes either camera and must not be trusted. Count
        # physical devices per model to detect that; counting after dedup
        # keeps the extra nodes of a single camera from looking like a twin.
        model_counts = Counter(identity.model_key for info in infos if (identity := usb.get(info.index)) is not None)
        # Backends without sysfs evidence (macOS, Windows) can only report the
        # weaker signal of two devices literally advertising the same id.
        duplicate_ids = {
            uid for uid, count in Counter(info.unique_id for info in infos if info.unique_id).items() if count > 1
        }

        devices: list[DeviceInfo] = []
        for info in infos:
            identity = usb.get(info.index)
            has_collision = (identity is not None and model_counts[identity.model_key] > 1) or (
                info.unique_id in duplicate_ids
            )
            stable = bool(info.id_stable and info.unique_id and not has_collision)
            devices.append(
                DeviceInfo(
                    device_id=info.unique_id if stable else str(info.index),
                    index=info.index,
                    name=info.name,
                    driver="uvc",
                    hardware_id=info.unique_id or None,
                    id_stable=stable,
                    manufacturer="",
                    model=info.name,
                    metadata={
                        "description": info.description,
                        "misc": info.misc,
                        "backend": "omnicamera",
                        "unique_id": info.unique_id or "",
                        "serial_collision": has_collision,
                    },
                )
            )
        return devices

    @classmethod
    def query_formats(cls, device_id: str) -> list[tuple[int, int, int]]:
        """Query supported formats for a device without opening a stream.

        Args:
            device_id: Device index or unique_id string.

        Returns:
            Sorted list of ``(width, height, fps)`` tuples.
        """
        infos = omni_camera.query(only_usable=False)
        resolved_id: int | str = int(device_id) if device_id.isdecimal() else device_id
        info = cls._resolve_device_info(infos, resolved_id)
        cam = omni_camera.Camera(info)
        fmts = cam.get_format_options()
        return sorted({(f.width, f.height, int(f.frame_rate)) for f in fmts})

    def get_settings(self) -> list[CameraSetting]:
        if not self._connected or self._cam is None:
            raise NotConnectedError

        get_controls = getattr(self._cam, "get_controls", None)
        if not callable(get_controls):
            msg = "get_settings is not available for this OmniCamera build."
            raise NotImplementedError(msg)

        raw_controls = get_controls()
        if not isinstance(raw_controls, dict):
            raw_controls = dict(cast("Any", raw_controls))

        controls: list[CameraSetting] = []
        for name, ctrl in raw_controls.items():
            vr = ctrl.value_range
            has_range = len(vr) > 0

            controls.append(
                CameraSetting(
                    id=name,
                    name=name,
                    setting_type="integer",
                    min=vr.start if has_range else None,
                    max=vr[-1] if has_range else None,
                    step=vr.step if has_range else None,
                    default=None,
                    value=None,
                    inactive=not ctrl.is_active,
                    read_only=False,
                )
            )
        return controls

    def apply_settings(self, settings: CameraSetting | list[CameraSetting]) -> None:
        """Apply one or more camera settings.

        Read-only, inactive, and valueless settings are silently skipped.
        """
        raise NotImplementedError
