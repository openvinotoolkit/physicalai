# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from physicalai.robot.transport import SharedRobotClient


class _Session:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class TestSharedRobotClient:
    def test_requires_explicit_remote_opt_in(self) -> None:
        with pytest.raises(ValueError, match="allow_remote=True"):
            SharedRobotClient()

    def test_discovery_reuses_one_session(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import physicalai.robot.transport._client as client_module

        clock = _Clock()
        session = _Session()
        sessions_opened = 0
        discovery_calls = 0

        def _open_session(*, allow_remote: bool) -> _Session:
            nonlocal sessions_opened
            assert allow_remote
            sessions_opened += 1
            return session

        def _discover(*, session: _Session, timeout: float) -> list[dict[str, str]]:
            nonlocal discovery_calls
            assert session is not None
            assert timeout > 0
            discovery_calls += 1
            return [{"name": "left-arm"}] if discovery_calls >= 2 else []

        monkeypatch.setattr(client_module, "open_session", _open_session)
        monkeypatch.setattr(client_module, "discover_robots", _discover)
        monkeypatch.setattr(client_module.time, "monotonic", clock.monotonic)
        monkeypatch.setattr(client_module.time, "sleep", clock.sleep)

        with SharedRobotClient(allow_remote=True) as client:
            assert client.discover(timeout=0.5) == [{"name": "left-arm"}]
            assert client.discover(timeout=0.1) == [{"name": "left-arm"}]

        assert sessions_opened == 1
        assert session.closed

    def test_attach_borrows_session_and_close_disconnects_robot(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import physicalai.robot.transport._client as client_module

        session = _Session()
        received_session: _Session | None = None
        messages: list[str] = []

        class Robot:
            disconnected = False
            name = "left-arm"

            def disconnect(self) -> None:
                self.disconnected = True

        robot = Robot()

        def _attach(
            _name: str,
            *,
            allow_remote: bool,
            connect_timeout: float,
            _session: _Session,
        ) -> Robot:
            nonlocal received_session
            assert allow_remote
            assert connect_timeout == 3.0
            received_session = _session
            return robot

        monkeypatch.setattr(client_module, "open_session", lambda *, allow_remote: session)
        monkeypatch.setattr(client_module.SharedRobot, "attach", _attach)
        monkeypatch.setattr(client_module.logger, "info", messages.append)

        client = SharedRobotClient(allow_remote=True)
        assert client.attach("left-arm", connect_timeout=3.0) is robot
        client.close()

        assert received_session is session
        assert robot.disconnected
        assert session.closed
        assert messages == ["Disconnecting SharedRobot 'left-arm' as SharedRobotClient closes"]