# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Robot error hierarchy.

All robot-related exceptions inherit from :class:`RobotError`, which
itself extends :class:`RuntimeError`. This allows callers to catch broad
(``except RobotError``) or narrow (``except RobotIdConflict``). Mirrors
the :class:`~physicalai.capture.errors.CaptureError` hierarchy.
"""


class RobotError(RuntimeError):
    """Base error for robot failures."""


class RobotIdConflict(RobotError):  # noqa: N818 — conflict condition, not a generic error; name locked in design D19
    """Raised when a robot id is already claimed by different hardware."""


class RobotNotConnectedError(RobotError):
    """Raised when read/write methods are called before connect()."""


class RobotTransportError(RobotError):
    """Raised when the shared-robot transport fails (spawn, handshake, wire)."""
