# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Parent-side spawn and handshake for the robot owner subprocess."""

from __future__ import annotations

import json
import select
import subprocess  # noqa: S404 # nosec: B404
import sys
from typing import TYPE_CHECKING, Self

from loguru import logger

from physicalai.robot.errors import RobotDeviceAlreadyOwned, RobotTransportError

if TYPE_CHECKING:
    from physicalai.robot.transport._owner_config import RobotOwnerConfig

_DEFAULT_START_TIMEOUT = 30.0
"""Generous by design: WidowXAI.connect() blocks ~2s on a homing move, and
serial enumeration can be slow — a blind short timeout would misfire."""


class RobotOwner:
    """Spawns and supervises the robot owner subprocess.

    The subprocess owns the hardware connection, publishes state, and
    consumes actions over Zenoh. It detaches into its own session so it
    survives when the spawning parent exits; it self-terminates via idle
    timeout when zero subscribers remain.

    Args:
        config: Robot construction and transport-scope configuration.
    """

    def __init__(self, config: RobotOwnerConfig) -> None:
        self._config = config
        self._process: subprocess.Popen[bytes] | None = None

    def start(self, timeout: float = _DEFAULT_START_TIMEOUT) -> None:
        """Start the owner subprocess and wait for the READY handshake.

        On an ``ERROR:{json}`` response, delegates to
        :meth:`_raise_from_error_line`, which may raise the unambiguous
        :class:`~physicalai.robot.errors.RobotDeviceAlreadyOwned` directly
        instead of the generic :exc:`RobotTransportError` below.

        Args:
            timeout: Maximum seconds to wait for the subprocess to report
                readiness (hardware connect may block for seconds).

        Raises:
            RobotTransportError: If the subprocess never becomes ready, or
                reports an error phase other than device-already-owned
                (including name-lock contention: ``exc.phase ==
                "name_lock_contention"``, ``exc.device_ids`` set — the
                caller resolves that by re-probing ``/metadata``).
        """
        if self.is_alive:
            return

        # B603 suppressed: the argv list is static — sys.executable (the
        # active interpreter) plus a hardcoded internal module path.
        # shell=True is not used, so there is no shell-injection risk.
        # Configuration is delivered to the worker via stdin as JSON, not as
        # argv arguments.
        self._process = subprocess.Popen(  # nosec: B603
            [sys.executable, "-m", "physicalai.robot.transport._owner_worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            # Detach into a new session so the owner survives when the
            # parent exits / Ctrl+C's. Subsequent subscribers re-attach
            # by name.
            start_new_session=True,
        )
        assert self._process.stdin is not None  # noqa: S101
        self._process.stdin.write(json.dumps(self._config.to_json_dict()).encode())
        self._process.stdin.close()

        line = self._read_stdout_line(timeout)
        if line is None:
            self.stop()
            msg = f"robot owner did not become ready within {timeout:.1f}s"
            raise RobotTransportError(msg)
        if line.startswith("ERROR:"):
            self.stop()
            self._raise_from_error_line(line)
        if line != "READY":
            self.stop()
            msg = f"unexpected owner response: {line!r}"
            raise RobotTransportError(msg)

    @staticmethod
    def _raise_from_error_line(line: str) -> None:
        """Parse a worker ``ERROR:{json}`` line and raise the matching exception.

        Raises:
            RobotDeviceAlreadyOwned: On the unambiguous device-conflict phase.
            RobotTransportError: For every other phase, carrying ``phase``
                and ``device_ids`` for the caller to inspect.
        """
        try:
            payload = json.loads(line[len("ERROR:") :])
        except json.JSONDecodeError as exc:
            msg = f"failed to start robot owner (malformed ERROR payload): {line!r}"
            raise RobotTransportError(msg) from exc

        err = payload.get("msg", "<unknown>")
        tb = payload.get("traceback")
        phase = payload.get("phase")
        device_ids = tuple(payload.get("device_ids") or ())
        msg = f"failed to start robot owner: {err}"
        if tb:
            msg = f"{msg}\n--- worker traceback ---\n{tb}"

        if phase == "device_lock_contention":
            raise RobotDeviceAlreadyOwned(msg, phase=phase, device_ids=device_ids)
        raise RobotTransportError(msg, phase=phase, device_ids=device_ids or None)

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the owner subprocess.

        Sends SIGTERM (the worker disconnects the driver into its safe
        state), waits up to *timeout* seconds, then SIGKILL.

        Args:
            timeout: Maximum seconds to wait for graceful shutdown.
        """
        proc = self._process
        if proc is None or proc.poll() is not None:
            self._process = None
            return

        proc.terminate()
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            logger.warning(f"Owner subprocess did not exit within {timeout:.1f}s, killing")
            proc.kill()
            proc.wait(timeout=1)

        self._process = None

    @property
    def is_alive(self) -> bool:
        """Whether the owner subprocess is running."""
        return self._process is not None and self._process.poll() is None

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.stop()

    def _read_stdout_line(self, timeout: float) -> str | None:
        """Read one line from the subprocess stdout with a timeout.

        Args:
            timeout: Maximum seconds to wait for a line.

        Returns:
            The stripped line, or ``None`` on timeout / EOF.
        """
        proc = self._process
        if proc is None or proc.stdout is None:
            return None

        # select.select doesn't work on Windows pipes;
        # a thread-based fallback would be needed for Windows support.
        readable, _, _ = select.select([proc.stdout], [], [], timeout)
        if not readable:
            return None

        raw = proc.stdout.readline()
        if not raw:
            return None
        return raw.decode().strip()
