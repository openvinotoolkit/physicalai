# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import sys
import time
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
    """Return USB identity for the device behind ``/dev/video<index>``.

    On Linux, walk from ``/sys/class/video4linux/video<index>/device`` to the
    owning USB node and read ``idVendor``, ``idProduct``, and ``serial``.

    Returns:
        The owning device's identity, or None on non-Linux platforms and when
        sysfs does not expose the required attributes.
    """
    if not sys.platform.startswith("linux"):
        return None
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


def _device_key(info: omni_camera.CameraInfo, usb: dict[int, _UsbIdentity | None]) -> str:
    """Key an entry by the camera behind it: its USB device path, or its own index.

    Returns:
        A key two entries share only when they are the same camera.
    """
    identity = usb.get(info.index)
    return identity.devpath if identity else str(info.index)


class OmniCamera(Camera):
    _POLL_INTERVAL_S = 0.001

    def __init__(
        self,
        *,
        device_id: int | str | dict = 0,
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
    def _physical_cameras(infos: list[omni_camera.CameraInfo]) -> list[omni_camera.CameraInfo]:
        """Reduce a query list to one entry per physical camera.

        V4L2 lists a camera several times, once per ``/dev/videoN`` it exposes
        (capture, metadata, ...). Grouping those by USB device rather than by
        by-id path is what keeps twins that share a by-id name from collapsing
        into one camera.

        Args:
            infos: Raw query list.

        Returns:
            The lowest-indexed entry of each camera, which is its capture device.
        """
        usb = {info.index: _usb_identity(info.index) for info in infos}
        by_device: dict[str, omni_camera.CameraInfo] = {}
        for info in infos:
            device = _device_key(info, usb)
            if device not in by_device or info.index < by_device[device].index:
                by_device[device] = info
        return list(by_device.values())

    @staticmethod
    def _best_matches(infos: list[omni_camera.CameraInfo], device_id: str) -> list[omni_camera.CameraInfo]:
        """Score each camera's match against an opaque identity string.

        ``device_meta_json`` carries whatever fields a device happens to
        report -- a serial, a sensor port, a bus path, or something else
        entirely -- with no field guaranteed present, so *device_id* is
        matched against the reported name and against every value in that
        dict rather than a fixed set of keys.

        Args:
            infos: Raw query list.
            device_id: The identity string to match.

        Returns:
            The cameras with the highest match score, empty if none match at
            all. More than one entry means the identity does not single out
            a camera.
        """
        scored: list[tuple[int, omni_camera.CameraInfo]] = []
        for c in infos:
            meta_values = {str(v) for v in (c.device_meta_json or {}).values() if v is not None and str(v)}
            score = (device_id == c.name) + (device_id in meta_values)
            if score > 0:
                scored.append((score, c))

        if not scored:
            return []
        best = max(score for score, _ in scored)
        return [c for score, c in scored if score == best]

    @staticmethod
    def _resolve_open_target(
        infos: list[omni_camera.CameraInfo], device_id: int | str | dict, *, resolve_strict: bool = False
    ) -> omni_camera.CameraInfo | int:
        """Resolve *device_id* to something safe to hand to ``omni_camera.Camera``.

        Args:
            infos: Raw query list.
            device_id: Video index, ``/dev/videoN`` path, ``index:N``, an
                opaque identity string (matched against the reported name and
                ``device_meta_json`` values), or a ``device_meta_json``
                identity dict.
            resolve_strict: If device id is a dict, whether to match strict or to find the best match.

        Returns:
            A bare index when *device_id* names a video index -- opening it
            directly is what keeps a colliding camera's identity from
            resolving to the wrong twin. The matching ``CameraInfo`` when
            *device_id* names the camera's reported identity instead, so the
            backend can re-resolve it after the video indices have shifted.

        Raises:
            CaptureError: The requested index does not exist, or the
                requested identity is shared by more than one connected
                device.
            ValueError: *device_id* is a device path string this platform
                does not support.
        """
        if isinstance(device_id, dict):
            matches = omni_camera.resolve_camera_for_device_meta_json(device_id)
            if len(matches) > 1:
                msg = (
                    f"Camera identity {device_id!r} is shared by more than one connected device "
                    "(duplicate or absent identity), so it cannot select a specific camera. "
                    "Open the camera by video index (e.g. device=0 or device='/dev/video0') instead."
                )
                raise CaptureError(msg)
            if len(matches) == 1 and (not resolve_strict or matches[0].device_meta_json == device_id):
                return matches[0]

            msg = f"No camera found matching {device_id!r}"
            raise CaptureError(msg)

        if isinstance(device_id, int):
            if not any(candidate.index == device_id for candidate in infos):
                msg = f"No camera found at index {device_id}"
                raise CaptureError(msg)
            return device_id

        # ``index:N`` is the backend's own spelling of a video index; it is
        # what a persisted device_id falls back to for a camera that reports
        # no identity of its own, so it can reach us again here.
        stripped = device_id.removeprefix("index:")
        if stripped.isdecimal():
            index = int(stripped)
        elif device_id.startswith("/dev/video"):
            suffix = device_id.removeprefix("/dev/video")
            if not suffix.isdecimal():
                msg = f"Invalid device path: {device_id}"
                raise ValueError(msg)
            index = int(suffix)
        else:
            matches = OmniCamera._best_matches(infos, device_id)
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                msg = (
                    f"Camera identity {device_id!r} is shared by more than one connected device "
                    "(duplicate or absent identity), so it cannot select a specific camera. "
                    "Open the camera by video index (e.g. device=0 or device='/dev/video0') instead."
                )
                raise CaptureError(msg)
            msg = (
                "OmniCamera backend does not support device path strings on this platform. "
                "Use an integer camera index or a matching camera identity instead."
            )
            raise CaptureError(msg)

        if not any(candidate.index == index for candidate in infos):
            msg = f"No camera found at index {index}"
            raise CaptureError(msg)
        return index

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
        infos_all = omni_camera.query(only_usable=False)
        infos_usable = omni_camera.query(only_usable=True)
        while not infos_all and time.monotonic() < query_deadline:
            time.sleep(0.1)
            infos_all = omni_camera.query(only_usable=False)
            infos_usable = omni_camera.query(only_usable=True)

        # Try to find target in usable devices first
        try:
            target = self._resolve_open_target(infos_usable, self._device_id_raw, resolve_strict=True)
        except CaptureError:
            target = self._resolve_open_target(infos_all, self._device_id_raw, resolve_strict=False)

        try:
            self._cam = omni_camera.Camera(target)
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

        return [
            DeviceInfo(
                device_id=str((info.device_meta_json or {}).get("serial") or f"index:{info.index}"),
                index=info.index,
                name=info.name,
                driver="uvc",
                hardware_payload=info.device_meta_json,
                manufacturer="",
                model=info.name,
                metadata={
                    "description": info.description,
                    "backend": "omnicamera",
                },
            )
            for info in cls._physical_cameras(infos)
        ]

    @classmethod
    def query_formats(cls, device_id: str | int | dict) -> list[tuple[int, int, int]]:
        """Query supported formats for a device without opening a stream.

        Args:
            device_id: Device index or unique_id string.

        Returns:
            Sorted list of ``(width, height, fps)`` tuples.
        """
        infos = omni_camera.query(only_usable=False)
        if isinstance(device_id, dict):
            cam = omni_camera.Camera(cls._resolve_open_target(infos, device_id))
        else:
            resolved_id: int | str = int(device_id) if str(device_id).isdecimal() else device_id
            cam = omni_camera.Camera(cls._resolve_open_target(infos, resolved_id))
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
