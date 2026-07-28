# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Shared-memory camera subscriber transport based on iceoryx2.

Construction is :class:`~physicalai.config.ComponentConfig`-only via
``camera=`` or :meth:`SharedCamera.from_config`. ``camera`` is declared as an
``@export_config(config_args=...)`` argument, so a nested
:func:`~physicalai.config.instantiate` hands over the recipe instead of
building a camera this process would immediately discard. Pass ``camera=None``
with ``service_name`` (or use :meth:`from_publisher`) for attach-only. Flat
``camera_type`` / ``camera_kwargs`` are unsupported. The publisher owns the
device exclusively — do not keep a direct camera open on the same hardware
while sharing.
"""

from __future__ import annotations

import contextlib
import ctypes
import time
from collections.abc import Mapping
from importlib import import_module
from typing import TYPE_CHECKING, Any, cast

from loguru import logger

from physicalai.capture.camera import Camera, ColorMode
from physicalai.capture.errors import CaptureError, CaptureTimeoutError, NotConnectedError
from physicalai.config import export_config

from ._header import FrameHeader, decode_header, decode_rgb
from ._spec import derive_service_name, normalize_camera_config, validate_reconfigure_request

if TYPE_CHECKING:
    from physicalai.capture.frame import Frame
    from physicalai.capture.transport._publisher import CameraPublisher
    from physicalai.config import ComponentConfig


_SERVICE_NAME_EXPECTED_PARTS = 5


def _probe_service(service_name: str) -> bool:
    """Check if a publisher is serving *service_name*.

    Returns:
        ``True`` if a publisher is reachable, ``False`` otherwise.
    """
    try:
        iox2 = cast("Any", import_module("iceoryx2"))

        node = iox2.NodeBuilder.new().create(iox2.ServiceType.Ipc)
        try:
            svc = (
                node
                .service_builder(
                    iox2.ServiceName.new(service_name),
                )
                .publish_subscribe(iox2.Slice[ctypes.c_uint8])
                .open()
            )
        except Exception:  # noqa: BLE001
            return False
        else:
            del svc
            return True
        finally:
            del node
    except Exception:  # noqa: BLE001
        return False


def _probe_with_retry(service_name: str, timeout: float, interval: float = 0.1) -> bool:
    """Poll ``_probe_service`` until it returns True or *timeout* elapses.

    Returns:
        ``True`` if a publisher is reachable within timeout, ``False`` otherwise.
    """
    deadline = time.monotonic() + timeout
    while True:
        if _probe_service(service_name):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(interval)


@export_config(class_path="physicalai.capture.SharedCamera", config_args=("camera",))
class SharedCamera(Camera):
    """Camera subscriber that reads frames from shared memory via iceoryx2.

    Connects to a publisher process that owns the physical camera device.
    Multiple SharedCamera instances can subscribe to the same publisher
    for zero-copy fan-out.

    Prefer :meth:`from_config` when sharing; never keep a direct camera open
    while sharing. The constructor takes ``camera: ComponentConfig`` to spawn,
    or ``camera=None`` + ``service_name`` for attach-only
    (:meth:`from_publisher` is the explicit form).

    Opted into :func:`~physicalai.config.export_config` as a **construction
    recipe** only (nested ``camera`` ComponentConfig, ``service_name``,
    ``color_mode``, transport knobs). Publisher / iceoryx2 session / frame
    state is never part of :func:`~physicalai.config.to_config`.

    The publisher subprocess owns the device exclusively. Another connected
    holder of the same hardware will cause open to fail; this API does not
    hand off an already-open device into the child.

    Args:
        camera: Camera :class:`~physicalai.config.ComponentConfig` from local
            config input (same boundary as CLI/app args), used to spawn if no
            publisher exists yet for the derived or explicit ``service_name``.
            ``None`` means attach-only. Declared as an ``@export_config``
            config arg, so nested :func:`~physicalai.config.instantiate`
            passes the recipe through without constructing the camera here.
        color_mode: Pixel format preference for this subscriber.
        zero_copy: If True, returned frames reference the iceoryx2 SHM
            buffer directly (read-only). Otherwise, frames are copied.
        service_name: iceoryx2 service name. Derived for built-in
            ``class_path`` values when omitted; required for third-party
            cameras and for attach-only.
        validate_on_connect: If True and an existing publisher's frame
            dimensions do not match ``width``/``height`` in
            ``camera.init_args``, :meth:`connect` raises
            :class:`CaptureError`. If False (default), the mismatch is
            logged as a warning and the existing publisher's resolution
            is used.
        overwrite_settings: If True, attempt to reconfigure the publisher
            to match requested settings on config mismatch. Requires
            the publisher to support the control channel (phase 2+).
        idle_timeout: Seconds with zero subscribers before the publisher
            self-exits.  Lower values (e.g. 0.5) suit preview streams
            where resolution may change frequently; higher values
            (e.g. 5.0) suit recording sessions.
    """

    def __init__(
        self,
        *,
        camera: ComponentConfig | Mapping[str, object] | None = None,
        color_mode: ColorMode | str = ColorMode.RGB,
        zero_copy: bool = False,
        service_name: str | None = None,
        validate_on_connect: bool = False,
        overwrite_settings: bool = False,
        idle_timeout: float = 5.0,
    ) -> None:
        if camera is None and service_name is None:
            msg = "must provide camera ComponentConfig or service_name"
            raise ValueError(msg)

        recipe = None if camera is None else normalize_camera_config(camera)
        if recipe is not None:
            service_name = derive_service_name(recipe, service_name=service_name)
        elif service_name is None:
            msg = "service_name must be provided if camera is None"
            raise ValueError(msg)

        super().__init__(color_mode=color_mode)
        self._camera = recipe
        self._service_name: str = service_name
        self._zero_copy = zero_copy
        self._validate_on_connect = validate_on_connect
        self._overwrite_settings = overwrite_settings
        self._idle_timeout = idle_timeout
        self._publisher: CameraPublisher | None = None
        self._connected = False
        self._latest: Frame | None = None
        self._last_header: FrameHeader | None = None
        self._config_warned = False
        self._held_sample: Any = None
        self._node: Any | None = None
        self._subscriber: Any | None = None
        self._listener: Any | None = None

    @classmethod
    def _physicalai_normalize_captured_init_args(cls, supplied: dict[str, object]) -> None:
        camera = supplied.get("camera")
        if isinstance(camera, Mapping):
            supplied["camera"] = normalize_camera_config(camera)

    @classmethod
    def from_config(
        cls,
        config: ComponentConfig | Mapping[str, object],
        *,
        service_name: str | None = None,
        color_mode: ColorMode | str = ColorMode.RGB,
        zero_copy: bool = False,
        validate_on_connect: bool = False,
        overwrite_settings: bool = False,
        idle_timeout: float = 5.0,
    ) -> SharedCamera:
        """Primary API: spawn/attach from a local camera ComponentConfig.

        Args:
            config: Local ``class_path`` + ``init_args`` for the camera.
            service_name: Explicit iceoryx2 name; derived for built-ins when omitted.
            color_mode: Subscriber pixel-format preference.
            zero_copy: Whether frames reference SHM directly.
            validate_on_connect: Raise on resolution mismatch at connect.
            overwrite_settings: Reconfigure publisher on mismatch.
            idle_timeout: Idle self-exit timeout for a spawned publisher.

        Returns:
            A ``SharedCamera`` that stores the normalized ComponentConfig and
            writes only the new publisher stdin shape on spawn.
        """
        return cls(
            camera=config,
            service_name=service_name,
            color_mode=color_mode,
            zero_copy=zero_copy,
            validate_on_connect=validate_on_connect,
            overwrite_settings=overwrite_settings,
            idle_timeout=idle_timeout,
        )

    @classmethod
    def from_publisher(
        cls,
        service_name: str,
        *,
        color_mode: ColorMode | str = ColorMode.RGB,
        zero_copy: bool = False,
        validate_on_connect: bool = False,
        overwrite_settings: bool = False,
        idle_timeout: float = 5.0,
    ) -> SharedCamera:
        """Attach-only construction: subscribe to an existing publisher by name.

        Never spawns a publisher — :meth:`connect` times out if none is
        reachable.

        Args:
            service_name: iceoryx2 service name of an existing publisher.
            color_mode: Subscriber pixel-format preference.
            zero_copy: Whether frames reference SHM directly.
            validate_on_connect: Raise on resolution mismatch at connect.
            overwrite_settings: Reconfigure publisher on mismatch.
            idle_timeout: Unused for attach-only (no spawn); kept for API parity.

        Returns:
            An attach-only ``SharedCamera``.
        """
        return cls(
            camera=None,
            color_mode=color_mode,
            zero_copy=zero_copy,
            service_name=service_name,
            validate_on_connect=validate_on_connect,
            overwrite_settings=overwrite_settings,
            idle_timeout=idle_timeout,
        )

    def connect(self, timeout: float = 5.0) -> None:
        if self._connected:
            return

        if self._camera is not None and not _probe_service(self._service_name):
            from ._publisher import CameraPublisher  # noqa: PLC0415
            from ._spec import CameraPublisherConfig  # noqa: PLC0415

            spec = CameraPublisherConfig(camera=self._camera)
            publisher = CameraPublisher(spec, self._service_name, idle_timeout=self._idle_timeout)
            try:
                publisher.start()
            except Exception as exc:
                if _probe_with_retry(self._service_name, timeout=3.0):
                    logger.debug(f"Lost publisher race for {self._service_name} — using existing")
                else:
                    msg = (
                        f"failed to start camera publisher for {self._service_name}: {exc}. "
                        "The publisher opens the device exclusively; another connected "
                        "holder of the same hardware will cause open to fail."
                    )
                    raise CaptureError(msg) from exc
            else:
                self._publisher = publisher

        iox2 = cast("Any", import_module("iceoryx2"))

        # Silence iceoryx2's kHz-rate ``FailedToDeliverSignal`` warnings.
        # They fire whenever a listener's unix-datagram wake-up socket is
        # full — which happens for any subscriber using non-blocking reads
        # (e.g. ``read_latest``) under camera-rate notifications. The
        # pub-sub payload itself still delivers reliably.
        with contextlib.suppress(Exception):
            iox2.set_log_level(iox2.LogLevel.Error)

        self._node = iox2.NodeBuilder.new().create(iox2.ServiceType.Ipc)

        pub_sub = (
            self._node
            .service_builder(iox2.ServiceName.new(self._service_name))
            .publish_subscribe(iox2.Slice[ctypes.c_uint8])
            .open()
        )
        self._subscriber = pub_sub.subscriber_builder().create()

        event_svc = self._node.service_builder(iox2.ServiceName.new(f"{self._service_name}/notify")).event().open()
        self._listener = event_svc.listener_builder().create()

        logger.debug(f"Connecting SharedCamera subscriber to {self._service_name}")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            sample = self._subscriber.receive()
            if sample is not None:
                header, frame = self._decode_sample(sample)
                self._last_header = header
                self._latest = frame
                self._negotiate_config(header, timeout=deadline - time.monotonic())
                self._connected = True
                return

            remaining = deadline - time.monotonic()
            if remaining > 0:
                self._listener.timed_wait_one(
                    iox2.Duration.from_secs_f64(min(remaining, 0.5)),
                )

        logger.debug(f"SharedCamera connect timeout for {self._service_name}")
        self._do_disconnect()
        msg = "no publisher responded within timeout"
        raise CaptureTimeoutError(msg)

    def read(self, timeout: float = 2.0) -> Frame:
        if not self._connected or self._subscriber is None or self._listener is None:
            msg = "shared camera is not connected"
            raise NotConnectedError(msg)

        iox2 = cast("Any", import_module("iceoryx2"))

        wait_timeout = timeout
        event = self._listener.timed_wait_one(iox2.Duration.from_secs_f64(wait_timeout))
        if event is None:
            msg = "timed out waiting for frame"
            raise CaptureTimeoutError(msg)

        self._held_sample = None  # release previous borrow before draining
        newest_sample = None
        while True:
            sample = self._subscriber.receive()
            if sample is None:
                break
            newest_sample = sample
            # Explicitly clear 'sample' to release reference. Otherwise, Python keeps
            # it alive during the next receive() call, triggering iceoryx2 subscriber warnings/errors,
            # especially when run in an IDE where debuggers hold onto local frames.
            sample = None

        if newest_sample is not None:
            header, frame = self._decode_sample(newest_sample)
            self._last_header = header
            self._latest = frame
            self._check_config_match(header)
            # Explicitly clear 'newest_sample' to release the iceoryx2 sample borrow under IDE/debugger.
            newest_sample = None

        if self._latest is None:
            msg = "no frame available"
            raise CaptureTimeoutError(msg)

        return self._latest

    def read_latest(self) -> Frame:
        if not self._connected or self._subscriber is None:
            msg = "shared camera is not connected"
            raise NotConnectedError(msg)

        self._held_sample = None  # release previous borrow before draining
        newest_sample = None
        # Bounded queue: iceoryx2's default subscriber buffer is small (typically 1-2 samples).
        # The publisher sends one frame per camera tick, and the subscriber drains on every read_latest() call.
        while True:
            sample = self._subscriber.receive()
            if sample is None:
                break
            newest_sample = sample
            # Explicitly clear 'sample' to release reference. Otherwise, Python keeps
            # it alive during the next receive() call, triggering iceoryx2 subscriber warnings/errors,
            # especially when run in an IDE where debuggers hold onto local frames.
            sample = None

        if newest_sample is not None:
            header, frame = self._decode_sample(newest_sample)
            self._last_header = header
            self._latest = frame
            self._check_config_match(header)
            # Explicitly clear 'newest_sample' to release the iceoryx2 sample borrow under IDE/debugger.
            newest_sample = None

        if self._latest is None:
            msg = "no frame available"
            raise CaptureTimeoutError(msg)

        return self._latest

    def _decode_sample(self, sample: Any) -> tuple[FrameHeader, Frame]:  # noqa: ANN401
        import ctypes as _ct  # noqa: PLC0415

        slc = sample.payload()
        buf = (_ct.c_uint8 * slc.number_of_elements).from_address(slc.data_ptr)
        header = decode_header(bytes(buf))
        if self._zero_copy:
            from ._header import decode_rgb_view  # noqa: PLC0415

            self._held_sample = sample
            # ``toreadonly()`` marks the buffer read-only at the buffer-protocol
            # level. ``np.frombuffer`` then produces an array whose ``writeable``
            # flag cannot be flipped back to True, so consumer writes — including
            # accidental in-place ops like ``arr += 1`` or OpenCV drawing calls —
            # fail fast instead of silently corrupting the iceoryx2 SHM segment
            # (which would be seen by every other subscriber). Bounds metadata on
            # the memoryview also prevents out-of-bounds indexing past the
            # payload. Consumers that need to modify the frame must ``.copy()``.
            return header, decode_rgb_view(header, memoryview(buf).toreadonly())
        return header, decode_rgb(header, bytes(buf))

    def _do_disconnect(self) -> None:
        self._held_sample = None
        self._publisher = None
        self._subscriber = None
        self._listener = None
        self._node = None
        self._connected = False
        self._latest = None
        self._last_header = None

    def _build_reconfigure_request(self) -> dict[str, object]:
        """Build and validate the settings-only peer control request.

        Returns:
            A validated `RECONFIGURE` request.

        Raises:
            CaptureError: If this camera has no valid reconfigurable settings.
        """
        if self._camera is None:
            msg = "reconfigure requires a camera ComponentConfig (attach-only SharedCamera has none)"
            raise CaptureError(msg)

        init_args = self._camera["init_args"]
        settings = {key: value for key in ("width", "height", "fps") if (value := init_args.get(key)) is not None}
        request: dict[str, object] = {
            "kind": "RECONFIGURE",
            "settings": settings,
        }
        try:
            validate_reconfigure_request(request)
        except (TypeError, ValueError) as exc:
            msg = f"camera config has no valid reconfigure settings: {exc}"
            raise CaptureError(msg) from exc
        return request

    def _request_reconfigure(self, timeout: float = 5.0) -> dict:
        """Send a RECONFIGURE request to the publisher's control channel.

        Opens a one-shot request_response client on
        ``{service_name}/control``, sends a JSON reconfigure payload,
        and polls for the response within *timeout*.

        Args:
            timeout: Max seconds to wait for the response.

        Returns:
            Response dict (``{"ok": True}`` or ``{"ok": False, "error": ...}``).

        Raises:
            CaptureError: If the control service does not exist (v1 publisher)
                or the response times out.
        """
        import json  # noqa: PLC0415

        request = self._build_reconfigure_request()

        iox2 = cast("Any", import_module("iceoryx2"))
        control_name = f"{self._service_name}/control"

        try:
            node = iox2.NodeBuilder.new().create(iox2.ServiceType.Ipc)
            control_svc = (
                node
                .service_builder(iox2.ServiceName.new(control_name))
                .request_response(iox2.Slice[ctypes.c_uint8], iox2.Slice[ctypes.c_uint8])
                .open()
            )
            client = control_svc.client_builder().initial_max_slice_len(4096).create()
        except Exception as exc:
            msg = f"publisher does not support reconfigure (no control service at {control_name})"
            raise CaptureError(msg) from exc

        request_payload = json.dumps(request).encode()

        try:
            sample = client.loan_slice_uninit(len(request_payload))
            req_ptr = sample.payload().data_ptr
            ctypes.memmove(req_ptr, request_payload, len(request_payload))
            pending = sample.assume_init().send()
        except Exception as exc:
            msg = f"failed to send reconfigure request to {control_name}"
            raise CaptureError(msg) from exc

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            response = pending.receive()
            if response is not None:
                resp_slc = response.payload()
                resp_buf = (ctypes.c_uint8 * resp_slc.number_of_elements).from_address(resp_slc.data_ptr)
                return json.loads(bytes(resp_buf))
            time.sleep(0.05)

        msg = f"reconfigure request to {control_name} timed out after {timeout}s"
        raise CaptureError(msg)

    def _negotiate_config(self, header: FrameHeader, timeout: float = 5.0) -> None:
        """Connect-time config negotiation (called once, not per-frame).

        If config matches or no constraints specified, returns silently.
        If mismatch and ``overwrite_settings=True``, attempts reconfigure
        via the control channel. Otherwise defers to
        validate_on_connect/warn logic.

        Args:
            header: First received frame header.
            timeout: Remaining connect timeout for reconfigure response.

        Raises:
            CaptureError: When validate_on_connect=True and mismatch cannot
                be resolved.
        """
        mismatch = self._detect_mismatch(header)
        if mismatch is None:
            return

        want_str, actual_str = mismatch

        if self._overwrite_settings:
            try:
                result = self._request_reconfigure(timeout=timeout)
            except CaptureError as exc:
                if self._validate_on_connect:
                    self._do_disconnect()
                    raise
                logger.warning(
                    f"camera {self.device_id}: reconfigure failed ({exc}). Using existing config {actual_str}.",
                )
                return

            if result.get("ok"):
                logger.info(f"camera {self.device_id}: publisher reconfigured to ({want_str}).")
                return

            error_msg = result.get("error", "unknown error")
            if self._validate_on_connect:
                self._do_disconnect()
                msg = f"Cannot use SharedCamera for {self.device_id}: reconfigure failed: {error_msg}"
                raise CaptureError(msg)
            logger.warning(
                f"camera {self.device_id}: reconfigure failed ({error_msg}). Using existing config {actual_str}.",
            )
            return

        if self._validate_on_connect:
            self._do_disconnect()
            msg = (
                f"Cannot use SharedCamera for {self.device_id}: "
                f"existing publisher config {actual_str} does not match "
                f"requested ({want_str}). "
                f"Set validate_on_connect=False to attach to existing feed, "
                f"or overwrite_settings=True to reconfigure publisher."
            )
            raise CaptureError(msg)

        self._config_warned = True
        logger.warning(
            f"camera {self.device_id}: requested ({want_str}), "
            f"got {actual_str} from existing publisher. Using existing config.",
        )

    def _check_config_match(self, header: FrameHeader) -> None:
        """Per-frame config validation (hot path, no reconfigure).

        Called on every received frame. If publisher was reconfigured by
        another subscriber, detect the change and warn once. Connect-time
        validation has already decided whether this subscriber can attach.
        """
        mismatch = self._detect_mismatch(header)
        if mismatch is None:
            return

        want_str, actual_str = mismatch

        if not self._config_warned:
            self._config_warned = True
            logger.warning(
                f"camera {self.device_id}: requested ({want_str}), "
                f"got {actual_str} from existing publisher. Using existing config.",
            )

    def _detect_mismatch(self, header: FrameHeader) -> tuple[str, str] | None:
        """Compare header against requested camera init_args.

        Returns:
            ``(want_str, actual_str)`` if mismatch found, ``None`` otherwise.
        """
        if self._camera is None:
            return None
        init_args = self._camera.get("init_args", {})
        if not isinstance(init_args, dict):
            return None
        want_w = init_args.get("width")
        want_h = init_args.get("height")
        want_fps = init_args.get("fps")
        if want_w is None and want_h is None and want_fps is None:
            return None

        actual_w = header.width
        actual_h = header.height
        actual_fps = header.fps

        w_ok = want_w is None or actual_w == want_w
        h_ok = want_h is None or actual_h == want_h
        fps_ok = want_fps is None or actual_fps in {0, want_fps}
        if w_ok and h_ok and fps_ok:
            return None

        want_parts = []
        if want_w is not None:
            want_parts.append(f"w={want_w}")
        if want_h is not None:
            want_parts.append(f"h={want_h}")
        if want_fps is not None:
            want_parts.append(f"fps={want_fps}")
        return ", ".join(want_parts), f"{actual_w}x{actual_h}@{actual_fps}"

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def actual_width(self) -> int | None:
        return self._last_header.width if self._last_header is not None else None

    @property
    def actual_height(self) -> int | None:
        return self._last_header.height if self._last_header is not None else None

    @property
    def actual_fps(self) -> int | None:
        return self._last_header.fps if self._last_header is not None else None

    @property
    def service_name(self) -> str:
        return self._service_name

    @property
    def device_id(self) -> str:
        # service_name format: physicalai/camera/<class_name>/<device_id>/frame
        parts = self._service_name.split("/")
        if (
            len(parts) >= _SERVICE_NAME_EXPECTED_PARTS
            and parts[0] == "physicalai"
            and parts[1] == "camera"
            and parts[4] == "frame"
        ):
            return parts[3]
        # Custom service_name (from_publisher) — no embedded device id.
        return ""
