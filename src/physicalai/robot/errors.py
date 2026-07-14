# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Robot error hierarchy.

All robot-related exceptions inherit from :class:`RobotError`, which
itself extends :class:`RuntimeError`. This allows callers to catch broad
(``except RobotError``) or narrow (``except RobotNameConflict``). Mirrors
the :class:`~physicalai.capture.errors.CaptureError` hierarchy.
"""

from __future__ import annotations


class RobotError(RuntimeError):
    """Base error for robot failures."""


class RobotNotConnectedError(RobotError):
    """Raised when read/write methods are called before connect()."""


class RobotTransportError(RobotError):
    """Raised when the shared-robot transport fails (spawn, handshake, wire).

    Carries structured diagnostics from the owner worker so callers (or
    :class:`SharedRobot` internally) can distinguish failure phases without
    string-matching the message:

    Attributes:
        phase: Stable failure-phase code from the worker (e.g.
            ``"construction_failed"``, ``"name_lock_contention"``,
            ``"device_lock_contention"``, ``"connection_failed"``,
            ``"endpoint_collision"``), or ``None`` for parent-side failures
            that never reached the worker.
        device_ids: The failing worker's own resolved device ids, present
            for lock-contention phases so the caller can compare them
            against a winner's advertised metadata.
    """

    def __init__(
        self,
        message: str,
        *,
        phase: str | None = None,
        device_ids: tuple[str, ...] | None = None,
    ) -> None:
        """Initialize with the message plus optional structured diagnostics."""
        super().__init__(message)
        self.phase = phase
        self.device_ids = device_ids


class RobotNameConflict(RobotTransportError):  # noqa: N818 — conflict condition, not a generic error
    """Raised when a concurrent owner claimed the same name for different devices."""


class RobotDeviceAlreadyOwned(RobotTransportError):  # noqa: N818 — conflict condition, not a generic error
    """Raised when a requested device is already locked under another name."""


class RobotProtocolMismatch(RobotTransportError):  # noqa: N818 — conflict condition, not a generic error
    """Raised when an existing owner speaks an unsupported transport protocol version."""
