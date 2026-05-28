# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import re
import sys
import time
from typing import TYPE_CHECKING, Any, cast

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

        if sys.platform.startswith("linux"):
            # V4L2 exposes multiple /dev/videoN nodes per physical camera
            # (e.g. capture + metadata with distinct by-id paths like
            # ...-video-index0 and ...-video-index1). Keep only the lowest-
            # index node per physical device (index0 = capture, index1+ = metadata).
            phys_best: dict[str, omni_camera.CameraInfo] = {}
            for info in infos:
                uid = info.unique_id or ""
                # Only group by stripped key when the V4L2 multi-node
                # suffix is present (e.g. ...-video-index0 / -video-index1).
                # Cameras without that suffix keep their own index key so
                # genuinely separate devices sharing a serial are not collapsed.
                if uid and re.search(r"-video-index\d+$", uid):
                    phys_key = re.sub(r"-video-index\d+$", "", uid)
                else:
                    phys_key = str(info.index)
                if phys_key not in phys_best or info.index < phys_best[phys_key].index:
                    phys_best[phys_key] = info
            infos = list(phys_best.values())

        # Some vendors/models bake the same USB iSerial into every
        # unit of a model. When a unique_id appears more than once it cannot
        # identify a specific device, so demote those entries to index-based
        # fingerprints and let the user manage cable-to-config mapping.
        unique_id_counts: dict[str, int] = {}
        for info in infos:
            if info.unique_id:
                unique_id_counts[info.unique_id] = unique_id_counts.get(info.unique_id, 0) + 1
        colliding_ids = {uid for uid, count in unique_id_counts.items() if count > 1}

        devices: list[DeviceInfo] = []
        for info in infos:
            has_collision = bool(info.unique_id) and info.unique_id in colliding_ids
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
