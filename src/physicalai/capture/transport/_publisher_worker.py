# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Standalone publisher worker (``python -m physicalai.capture.transport._publisher_worker``)."""

from __future__ import annotations

import contextlib
import ctypes
import importlib
import json
import os
import signal
import sys
import threading
import time
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from types import FrameType

    from physicalai.capture.camera import Camera

_MAX_CONSECUTIVE_FAILURES = 5
_CONTROL_MAX_SLICE_LEN = 4096

shutdown = threading.Event()


def sigterm_handler(_signum: int, _frame: FrameType | None) -> None:
    shutdown.set()


def signal_ready() -> None:
    sys.stdout.write("READY\n")
    sys.stdout.flush()
    sys.stdout.close()


def signal_error(msg: str) -> None:
    sys.stdout.write(f"ERROR:{json.dumps(msg)}\n")
    sys.stdout.flush()
    sys.stdout.close()


def suppress_stdout() -> int:
    """Redirect fd 1 to /dev/null.

    Operates at the OS file-descriptor level (not ``sys.stdout``) so that
    native libraries writing directly to the C stdout fd (e.g.
    ``omni_camera``/Nokhwa via ``printf``) are also silenced. This keeps
    the single-line ``READY``/``ERROR`` IPC protocol on stdout uncorrupted
    during camera startup.

    Returns:
        The saved original fd 1, to be passed to :func:`restore_stdout`.
    """
    saved_fd = os.dup(1)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull_fd, 1)
    os.close(devnull_fd)
    return saved_fd


def restore_stdout(saved_fd: int) -> None:
    """Restore fd 1 from *saved_fd* and rewrap ``sys.stdout``.

    Undoes :func:`suppress_stdout` by pointing fd 1 back at the original
    stdout and rebuilding the Python ``sys.stdout`` text wrapper so that
    subsequent ``print`` calls reach the parent process again.

    Args:
        saved_fd: The fd returned by :func:`suppress_stdout`.
    """
    os.dup2(saved_fd, 1)
    os.close(saved_fd)
    sys.stdout = os.fdopen(1, "w")


def build_camera(config: dict) -> Camera:
    """Instantiate a camera from a JSON config dict.

    Args:
        config: Configuration dict with camera_type, camera_kwargs, and
            optional _factory_override.

    Returns:
        Connected camera instance.
    """
    factory_override = config.get("_factory_override")
    if factory_override:
        module_path, _, attr = factory_override.rpartition(":")
        mod = importlib.import_module(module_path)
        factory = getattr(mod, attr)
        return factory(**config.get("camera_kwargs", {}))

    from physicalai.capture.transport._spec import CameraSpec  # noqa: PLC0415, PLC2701

    spec = CameraSpec.from_json_dict(config)
    return spec.build()


def _camera_fps_from_config(config: dict[str, object]) -> int:
    """Extract fps from camera config when camera_kwargs is mapping-like.

    Returns:
        Requested fps value, or 0 when camera_kwargs is absent or invalid.
    """
    camera_kwargs = config.get("camera_kwargs")
    if not isinstance(camera_kwargs, dict):
        return 0

    fps = camera_kwargs.get("fps", 0)
    return int(fps)


class _PublisherState:
    """Mutable holder for camera/publisher resources shared between threads.

    Protected by ``lock`` — both the main publish loop and the control
    listener thread must acquire it before touching camera/publisher.
    """

    def __init__(self, camera: Camera, publisher: Any, camera_fps: int, config: dict) -> None:  # noqa: ANN401
        self.lock = threading.Lock()
        self.camera = camera
        self.publisher = publisher
        self.camera_fps = camera_fps
        self.config = config


def _control_listener(
    state: _PublisherState,
    service_name: str,
    iox2: Any,  # noqa: ANN401
    node: Any,  # noqa: ANN401
) -> None:
    """Control channel thread — serves reconfigure requests via request_response.

    Runs until ``shutdown`` is set. Opens ``{service_name}/control`` as a
    request_response server and polls for requests.

    Args:
        state: Shared mutable publisher state (guarded by state.lock).
        service_name: Base service name (control channel is ``/control`` suffix).
        iox2: The iceoryx2 module.
        node: The iceoryx2 node for this worker process.
    """
    control_name = f"{service_name}/control"
    try:
        control_service = (
            node
            .service_builder(iox2.ServiceName.new(control_name))
            .request_response(iox2.Slice[ctypes.c_uint8], iox2.Slice[ctypes.c_uint8])
            .max_servers(1)
            .max_clients(4)
            .open_or_create()
        )
        server = control_service.server_builder().initial_max_slice_len(_CONTROL_MAX_SLICE_LEN).create()
    except Exception:  # noqa: BLE001
        logger.exception(f"Failed to create control channel {control_name}")
        return

    logger.debug(f"Control channel listening on {control_name}")

    while not shutdown.is_set():
        active_request = server.receive()
        if active_request is None:
            time.sleep(0.05)
            continue

        try:
            req_slc = active_request.payload()
            req_buf = (ctypes.c_uint8 * req_slc.number_of_elements).from_address(req_slc.data_ptr)
            request = json.loads(bytes(req_buf))
        except Exception as exc:  # noqa: BLE001
            _respond_json(active_request, {"ok": False, "error": f"malformed request: {exc}"})
            continue

        kind = request.get("kind")
        if kind == "RECONFIGURE":
            response = _handle_reconfigure(state, request, service_name)
        else:
            response = {"ok": False, "error": f"unknown request kind: {kind!r}"}

        _respond_json(active_request, response)

    with contextlib.suppress(Exception):
        del server, control_service


def _respond_json(active_request: Any, payload: dict) -> None:  # noqa: ANN401
    """Send a JSON response via the active request handle."""
    response_bytes = json.dumps(payload).encode()
    sample = active_request.loan_slice_uninit(len(response_bytes))
    resp_ptr = sample.payload().data_ptr
    ctypes.memmove(resp_ptr, response_bytes, len(response_bytes))
    sample.assume_init().send()


def _handle_reconfigure(state: _PublisherState, request: dict, service_name: str) -> dict:
    """Process a RECONFIGURE request under the state lock.

    Args:
        state: Shared publisher state.
        request: Parsed JSON request with ``spec`` key.
        service_name: For logging.

    Returns:
        Response dict with ``ok`` and optional ``error``.
    """
    spec_data = request.get("spec")
    if not spec_data or not isinstance(spec_data, dict):
        return {"ok": False, "error": "missing or invalid 'spec' in request"}

    with state.lock:
        old_config = state.config.copy()
        old_camera = state.camera
        old_fps = state.camera_fps

        new_config = {
            "camera_type": spec_data.get("camera_type", old_config.get("camera_type")),
            "camera_kwargs": spec_data.get("camera_kwargs", {}),
            "service_name": service_name,
        }
        # Preserve factory override if present in original config
        if "_factory_override" in old_config:
            new_config["_factory_override"] = old_config["_factory_override"]

        try:
            old_camera.disconnect()
        except Exception:  # noqa: BLE001
            logger.warning(f"Old camera disconnect failed during reconfigure for {service_name}")

        try:
            new_camera = build_camera(new_config)
            new_camera.connect()
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Reconfigure failed for {service_name}: {exc}. Attempting restore.")
            try:
                restored_camera = build_camera(old_config)
                restored_camera.connect()
                state.camera = restored_camera
                state.config = old_config
                state.camera_fps = old_fps
            except Exception:  # noqa: BLE001
                logger.critical(
                    f"Cannot restore old config for {service_name} — shutting down.",
                )
                shutdown.set()
                return {"ok": False, "error": f"reconfigure failed and restore failed: {exc}"}
            return {"ok": False, "error": str(exc)}

        new_fps = _camera_fps_from_config(new_config)
        state.camera = new_camera
        state.config = new_config
        state.camera_fps = new_fps
        logger.info(f"Reconfigured publisher for {service_name} with {spec_data}")
        return {"ok": True}


def main() -> int:  # noqa: C901, PLR0912, PLR0914, PLR0915
    """Entry point for the publisher worker process.

    Returns:
        Exit code: 0 on success, 1 on startup failure.
    """
    signal.signal(signal.SIGTERM, sigterm_handler)

    raw = sys.stdin.read()
    sys.stdin.close()
    try:
        config = json.loads(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        signal_error(f"invalid JSON config: {exc}")
        return 1

    service_name: str = config["service_name"]
    idle_timeout: float = config.get("idle_timeout", 5.0)
    max_subscribers: int = config.get("max_subscribers", 32)
    camera_fps = _camera_fps_from_config(config)

    saved_stdout_fd: int | None = suppress_stdout()

    camera = None
    try:
        iox2 = importlib.import_module("iceoryx2")

        with contextlib.suppress(Exception):
            iox2.set_log_level(iox2.LogLevel.Error)

        camera = build_camera(config)
        camera.connect()

        node = iox2.NodeBuilder.new().create(iox2.ServiceType.Ipc)
        max_nodes = max_subscribers + 2
        service = (
            node
            .service_builder(iox2.ServiceName.new(service_name))
            .publish_subscribe(iox2.Slice[ctypes.c_uint8])
            .max_publishers(1)
            .max_subscribers(max_subscribers)
            .max_nodes(max_nodes)
            .open_or_create()
        )

        from physicalai.capture.transport._header import HEADER_SIZE, encode_frame  # noqa: PLC0415, PLC2701

        first_frame = camera.read_latest()
        max_slice_len = HEADER_SIZE + first_frame.data.nbytes
        publisher = (
            service
            .publisher_builder()
            .initial_max_slice_len(max_slice_len)
            .allocation_strategy(iox2.AllocationStrategy.PowerOfTwo)
            .create()
        )

        event_service = (
            node
            .service_builder(iox2.ServiceName.new(f"{service_name}/notify"))
            .event()
            .max_listeners(max_subscribers)
            .max_nodes(max_nodes)
            .open_or_create()
        )
        notifier = event_service.notifier_builder().create()
    except Exception as exc:  # noqa: BLE001
        if saved_stdout_fd is not None:
            restore_stdout(saved_stdout_fd)
            saved_stdout_fd = None
        signal_error(str(exc))
        if camera is not None:
            try:
                camera.disconnect()
            except Exception:  # noqa: BLE001
                logger.exception("camera disconnect failed during error cleanup")
        return 1
    finally:
        if saved_stdout_fd is not None:
            restore_stdout(saved_stdout_fd)

    signal_ready()

    state = _PublisherState(camera=camera, publisher=publisher, camera_fps=camera_fps, config=config)

    control_thread = threading.Thread(
        target=_control_listener,
        args=(state, service_name, iox2, node),
        daemon=True,
        name="control-listener",
    )
    control_thread.start()

    from physicalai.capture.errors import CaptureError  # noqa: PLC0415

    node_check_interval = max(0.1, idle_timeout / 5)

    try:
        idle_since: float | None = None
        last_node_check = 0.0
        consecutive_failures = 0
        while not shutdown.is_set():
            with state.lock:
                current_camera = state.camera
                current_fps = state.camera_fps

            try:
                frame = current_camera.read(timeout=1.0)
            except CaptureError:
                with state.lock:
                    if state.camera is not current_camera:
                        consecutive_failures = 0
                        continue
                consecutive_failures += 1
                if consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
                    logger.error(
                        f"{consecutive_failures} consecutive read failures -- "
                        f"shutting down publisher for {service_name}",
                    )
                    break
                continue
            consecutive_failures = 0

            header, payload_bytes = encode_frame(frame, current_camera.color_mode, fps=current_fps)
            header_bytes = bytes(header)
            total_size = HEADER_SIZE + len(payload_bytes)

            with state.lock:
                sample = state.publisher.loan_slice_uninit(total_size)
                shm_ptr = sample.payload().data_ptr
                ctypes.memmove(shm_ptr, header_bytes, HEADER_SIZE)
                ctypes.memmove(shm_ptr + HEADER_SIZE, payload_bytes, len(payload_bytes))
                sample.assume_init().send()

            with contextlib.suppress(Exception):
                notifier.notify_with_custom_event_id(iox2.EventId.new(0))

            # Yield CPU until next frame is likely ready
            if current_fps > 0:
                sleep_percentage = 0.85
                time.sleep(sleep_percentage / current_fps)

            now = time.monotonic()
            if now - last_node_check >= node_check_interval:
                last_node_check = now
                sub_count = max(0, len(service.nodes) - 1)

                if sub_count == 0:
                    if idle_since is None:
                        idle_since = now
                    elif now - idle_since > idle_timeout:
                        logger.info(
                            f"No subscribers for {idle_timeout}s -- shutting down publisher for {service_name}",
                        )
                        break
                else:
                    idle_since = None
    except Exception:  # noqa: BLE001
        logger.exception(f"publisher loop failed for service {service_name}")
    finally:
        shutdown.set()
        control_thread.join(timeout=2.0)
        try:
            state.camera.disconnect()
        except Exception:  # noqa: BLE001
            logger.exception(f"camera disconnect failed for service {service_name}")
        with contextlib.suppress(NameError):
            del publisher, service, event_service, notifier, node

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
