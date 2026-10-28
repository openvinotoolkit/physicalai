# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Opt-in camera-only diagnostics with killable, isolated SDK worker processes."""

from __future__ import annotations

import contextlib
import math
import multiprocessing
import sys
import time
from typing import TYPE_CHECKING, Any

from pydantic import Field

from physicalai.diagnostics.cameras import supported_modes
from physicalai.diagnostics.identity import resolve_endpoint
from physicalai.diagnostics.models import EvidenceModel, Finding

if TYPE_CHECKING:
    from multiprocessing.connection import Connection
    from multiprocessing.synchronize import Event

    from physicalai.capture.camera import Camera
    from physicalai.diagnostics.models import Inventory

_STARTUP_TIMEOUT = 15.0


class StreamRequest(EvidenceModel):
    """Only a camera request can cross this worker boundary, never robot recipes."""

    selector: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    fps: int = Field(gt=0)
    pixel_format: str


def measure(camera: Camera, duration: float, *, stop: Event | None = None) -> dict[str, float | int | None]:
    """Measure host-received frames, not inferred USB bus utilization.

    Returns:
        Frame count, elapsed time, received FPS, and largest inter-frame gap.
    """
    started = time.monotonic()
    previous: float | None = None
    largest_gap = 0.0
    frames = 0
    first_captured: float | None = None
    last_captured: float | None = None
    while time.monotonic() - started < duration and (stop is None or not stop.is_set()):
        frame = camera.read(timeout=min(2.0, duration))
        if first_captured is None:
            first_captured = frame.timestamp
        last_captured = frame.timestamp
        now = time.monotonic()
        if previous is not None:
            largest_gap = max(largest_gap, now - previous)
        previous = now
        frames += 1
    elapsed = time.monotonic() - started
    return {
        "frames": frames,
        "elapsed_s": elapsed,
        "received_fps": frames / elapsed if elapsed else 0,
        "largest_gap_s": largest_gap,
        "timestamp_fps": (frames - 1) / (last_captured - first_captured)
        if first_captured is not None and last_captured is not None and last_captured > first_captured
        else None,
    }


def _worker(request: StreamRequest, connection: Connection, start: Event, stop: Event, duration: float) -> None:
    # Redirect SDK output so workers cannot corrupt the parent's JSON stdout.
    with contextlib.redirect_stdout(sys.stderr):
        camera = None
        previous = None
        result: dict[str, Any] = {}
        try:
            from physicalai.capture import UVCCamera  # noqa: PLC0415
            from physicalai.diagnostics.platforms.video import VideoState  # noqa: PLC0415

            previous = VideoState.read(request.selector)
            result["previous_mode"] = previous.describe()
            camera = UVCCamera(
                device=request.selector,
                backend="v4l2",
                width=request.width,
                height=request.height,
                fps=request.fps,
                backend_options={"pixel_format": request.pixel_format},
            )
            camera.connect(timeout=5)
            result["negotiated_mode"] = VideoState.read(request.selector).describe()
            connection.send({"ready": True})
            if not start.wait(timeout=_STARTUP_TIMEOUT):
                result["error"] = "Other cameras did not become ready before the test deadline."
            elif stop.is_set():
                result["error"] = "Camera test cancelled before measurement."
            else:
                camera.read_latest()  # Discard frames queued while the other workers were connecting.
                result["metrics"] = measure(camera, duration, stop=stop)
        except Exception as exc:  # noqa: BLE001 - contain optional camera SDK failures in the child process
            result["error"] = str(exc) or type(exc).__name__
            result["error_type"] = type(exc).__name__
        finally:
            if camera is not None:
                with contextlib.suppress(Exception):
                    camera.disconnect()
            if previous is not None:
                try:
                    previous.restore(request.selector)
                    restored_mode = type(previous).read(request.selector).describe()
                    result["restored"] = restored_mode == result["previous_mode"]
                    if not result["restored"]:
                        result["restore_error"] = (
                            f"Driver returned a different format after restoration: {restored_mode}"
                        )
                except OSError as exc:
                    result["restore_error"] = str(exc)
            connection.send(result)
            connection.close()


def _stop_workers(workers: list[tuple[Any, Any, Any]]) -> None:
    for process, parent, child in workers:
        if process.pid is not None:
            process.join(timeout=3)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
            if process.is_alive():
                process.kill()
                process.join(timeout=2)
            process.close()
        parent.close()
        child.close()


def _trial(requests: list[StreamRequest], duration: float) -> list[dict[str, Any]]:
    ctx = multiprocessing.get_context("spawn")
    start = ctx.Event()
    stop = ctx.Event()
    workers = []
    results: dict[int, dict[str, Any]] = {}
    ready: set[int] = set()
    deadline = time.monotonic() + duration + _STARTUP_TIMEOUT
    try:
        for request in requests:
            parent, child = ctx.Pipe(duplex=False)
            process = ctx.Process(target=_worker, args=(request, child, start, stop, duration), daemon=True)
            workers.append((process, parent, child))
            process.start()
            child.close()
        while len(results) < len(workers) and time.monotonic() < deadline:
            for index, (process, parent, _) in enumerate(workers):
                if index in results:
                    continue
                if parent.poll():
                    try:
                        message = parent.recv()
                    except EOFError:
                        message = {"error": f"Camera worker exited unexpectedly ({process.exitcode})."}
                    if message.get("ready"):
                        ready.add(index)
                    else:
                        results[index] = message
                elif not process.is_alive():
                    results[index] = {"error": f"Camera worker exited without a result ({process.exitcode})."}
            if len(ready | results.keys()) == len(workers):
                start.set()
            time.sleep(0.01)
        return [
            results.get(index, {"error": "Worker exceeded the bounded capture deadline; it was stopped."})
            for index in range(len(requests))
        ]
    finally:
        stop.set()  # Let cooperative workers disconnect/restore before resorting to termination.
        start.set()
        _stop_workers(workers)


_MAX_CAMERAS = 8


def _next_action(error: str | None) -> str:
    if error:
        message = error.lower()
        if "turbojpeg" in message:
            return (
                "Install/check the native libjpeg-turbo decoder required by V4L2 MJPEG capture. "
                "This is not USB congestion."
            )
        if "busy" in message:
            return "Close the application's existing camera stream before testing; doctor will not stop another owner."
        if "permission" in message:
            return "Review device ACLs/group membership and vendor rules as the Runtime user."
    return (
        "Compare individual/concurrent results, shared links, and fresh kernel logs; decoding and CPU also limit rates."
    )


def _stream_findings(request: StreamRequest, label: str, result: dict[str, Any]) -> list[Finding]:
    error = result.get("error")
    metrics = result.get("metrics", {})
    low_rate = metrics.get("received_fps", 0) < request.fps * 0.8
    actual = result.get("negotiated_mode", {})
    changed_mode = bool(actual) and (actual.get("width"), actual.get("height")) != (request.width, request.height)
    outcome = error or f"{metrics.get('received_fps', 0):.1f} received FPS"
    if changed_mode:
        outcome += " (negotiated dimensions differ from request)"
    findings = [
        Finding(
            check_id=f"stream.{label}",
            subject=request.selector,
            severity="FAIL" if error else "WARN" if low_rate or changed_mode else "PASS",
            summary=f"{label.capitalize()} camera test: {outcome}",
            evidence=[
                str(request.model_dump()),
                str(metrics),
                f"negotiated={actual}",
                f"previous={result.get('previous_mode')}",
                f"restored={result.get('restored', False)}",
            ],
            details={
                "requested": request.model_dump(mode="json"),
                "metrics": metrics,
                "negotiated_mode": actual,
                "previous_mode": result.get("previous_mode"),
                "restored": bool(result.get("restored")),
                "error_type": result.get("error_type"),
            },
            action=_next_action(error),
            verification="Repeat the same selected modes and topology; these measurements are not USB utilization.",
        )
    ]
    if result.get("restore_error"):
        findings.append(
            Finding(
                check_id="stream.restore",
                severity="WARN",
                subject=request.selector,
                summary="The previous camera format could not be restored after the test.",
                evidence=[str(result["restore_error"])],
                action="Review the device's current format and reapply the intended camera configuration.",
                verification="Inspect the camera format before starting another application.",
            )
        )
    return findings


def _comparison(request: StreamRequest, alone: dict[str, Any], concurrent: dict[str, Any]) -> Finding | None:
    if alone.get("error") or concurrent.get("error"):
        return None
    before = alone.get("metrics", {}).get("received_fps", 0)
    after = concurrent.get("metrics", {}).get("received_fps", 0)
    if before <= 0 or after <= 0:
        return None
    degraded = after < before * 0.8
    already_slow = before < request.fps * 0.8
    if degraded:
        summary = "Concurrent capture delivered over 20% fewer frames per second than individual capture."
        action = (
            "Inspect shared links and fresh logs, but also check CPU/decode load; this does not prove USB congestion."
        )
    elif already_slow:
        summary = (
            "Low frame rate was already present alone; adding the other selected cameras did not cause a 20% drop."
        )
        action = (
            "Inspect exposure/frame timing, firmware, format, and decode load; repeat before attributing the cause."
        )
    else:
        summary = "No additional 20% frame-rate drop observed during concurrent capture at these modes."
        action = "This result applies only to the tested modes, duration, and connected topology."
    return Finding(
        check_id="stream.comparison",
        severity="WARN" if degraded else "INFO" if already_slow else "PASS",
        subject=request.selector,
        summary=summary,
        evidence=[f"individual_fps={before}", f"concurrent_fps={after}"],
        details={"individual_fps": before, "concurrent_fps": after, "ratio": after / before, "drop_threshold": 0.2},
        action=action,
        verification="Repeat with the same modes and a longer duration for intermittent problems.",
    )


def test_streams(inventory: Inventory, selectors: list[str], *, duration: float) -> list[Finding]:
    """Test selected cameras individually, then concurrently, with identical modes.

    Returns:
        Per-camera findings for each trial, including the selected wire modes.

    Raises:
        ValueError: If a selector, platform, or supported wire mode is unavailable.
    """
    if inventory.platform != "linux":
        msg = "Bounded streaming tests currently support Linux V4L2 only; "
        msg += "passive inventory works on all supported platforms."
        raise ValueError(msg)
    if not math.isfinite(duration) or duration <= 0:
        msg = "Streaming duration must be finite and positive."
        raise ValueError(msg)
    if not 1 <= len(selectors) <= _MAX_CAMERAS:
        msg = "Select between one and eight camera endpoints for an explicit stream test."
        raise ValueError(msg)
    requests = []
    endpoint_ids: set[str] = set()
    for selector in selectors:
        device, endpoint = resolve_endpoint(inventory, selector)
        if endpoint.holders or any(ep.kind == "usb" and ep.holders for ep in device.endpoints):
            msg = f"{selector!r} is open by visible processes; release the existing owner before testing."
            raise ValueError(msg)
        modes = supported_modes(endpoint)
        if endpoint.role != "capture" or not modes or endpoint.id in endpoint_ids:
            msg = f"Select distinct supported capture endpoints; {selector!r} is duplicate or has no supported modes."
            raise ValueError(msg)
        endpoint_ids.add(endpoint.id)
        mode = min(
            modes, key=lambda m: (abs(m.width - 640) + abs(m.height - 480), abs(m.fps - 30), m.pixel_format != "MJPG")
        )
        requests.append(
            StreamRequest(
                selector=endpoint.path if selector == endpoint.id else selector,
                width=mode.width,
                height=mode.height,
                fps=int(mode.fps),
                pixel_format="mjpeg" if mode.pixel_format == "MJPG" else "yuyv",
            )
        )
    findings: list[Finding] = []
    trials = [("individual", [request]) for request in requests]
    if len(requests) > 1:
        trials.append(("concurrent", requests))
    individual: dict[str, dict[str, Any]] = {}
    for label, group in trials:
        results = _trial(group, duration)
        complete = all(not result.get("error") and result.get("metrics") for result in results)
        for request, result in zip(group, results, strict=True):
            findings.extend(_stream_findings(request, label, result))
            if label == "individual":
                individual[request.selector] = result
            elif complete and (comparison := _comparison(request, individual[request.selector], result)) is not None:
                findings.append(comparison)
        if label == "concurrent" and not complete:
            findings.append(
                Finding(
                    check_id="stream.comparison",
                    severity="SKIP",
                    subject="concurrent-workload",
                    summary="Not every selected camera completed concurrent capture; "
                    "no valid rate comparison is available.",
                    action="Resolve failed camera tests before comparing individual and concurrent performance.",
                    verification="Repeat with every selected stream completing its bounded test.",
                )
            )
    return findings
