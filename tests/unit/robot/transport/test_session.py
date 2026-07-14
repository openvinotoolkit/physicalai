# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

from physicalai.robot.transport._ids import derive_endpoint_port
from physicalai.robot.transport._session import open_session

from .conftest import requires_zenoh

if TYPE_CHECKING:
    import zenoh


@requires_zenoh
class TestOpenSession:
    def test_default_local_only_opens(self) -> None:
        session = open_session("left-arm")
        try:
            assert session is not None
        finally:
            session.close()

    def test_local_only_owner_listen(self) -> None:
        session = open_session("left-arm-owner", listen=True)
        try:
            assert session is not None
        finally:
            session.close()

    def test_allow_remote_owner_listen(self) -> None:
        session = open_session("left-arm-remote", listen=True, allow_remote=True)
        try:
            assert session is not None
        finally:
            session.close()

    def test_discovery_session_without_name(self) -> None:
        session = open_session()
        try:
            assert session is not None
        finally:
            session.close()


@requires_zenoh
class TestSessionConfig:
    """Assert on the exact Zenoh config values, not just that open() succeeds."""

    def _captured_config(self, *, name: str | None = None, listen: bool = False, allow_remote: bool = False) -> zenoh.Config:
        import zenoh

        original_open = zenoh.open
        captured: dict[str, zenoh.Config] = {}

        def _capture_and_open(config: zenoh.Config):  # type: ignore[no-untyped-def]
            captured["config"] = config
            return original_open(config)

        zenoh.open = _capture_and_open  # type: ignore[assignment]
        try:
            session = open_session(name, listen=listen, allow_remote=allow_remote)
            session.close()
        finally:
            zenoh.open = original_open  # type: ignore[assignment]
        return captured["config"]

    def test_local_only_disables_scouting(self) -> None:
        config = self._captured_config(name="left-arm", listen=True, allow_remote=False)
        assert config.get_json("scouting/multicast/enabled") == "false"
        assert config.get_json("scouting/gossip/enabled") == "false"

    def test_allow_remote_enables_scouting_defaults(self) -> None:
        config = self._captured_config(name="left-arm", listen=True, allow_remote=True)
        assert config.get_json("scouting/multicast/enabled") != "false"
        assert config.get_json("scouting/gossip/enabled") != "false"

    def test_local_only_binds_loopback(self) -> None:
        port = derive_endpoint_port("left-arm")
        config = self._captured_config(name="left-arm", listen=True, allow_remote=False)
        assert config.get_json("listen/endpoints") == f'["tcp/127.0.0.1:{port}"]'

    def test_allow_remote_binds_wildcard(self) -> None:
        port = derive_endpoint_port("left-arm")
        config = self._captured_config(name="left-arm", listen=True, allow_remote=True)
        assert config.get_json("listen/endpoints") == f'["tcp/0.0.0.0:{port}"]'

    def test_subscriber_always_connects_loopback(self) -> None:
        port = derive_endpoint_port("left-arm")
        config = self._captured_config(name="left-arm", listen=False, allow_remote=True)
        assert config.get_json("connect/endpoints") == f'["tcp/127.0.0.1:{port}"]'
