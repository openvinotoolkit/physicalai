# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import time
import uuid
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

    def test_subscriber_retries_its_connection_quickly(self) -> None:
        config = self._captured_config(name="left-arm", listen=False)
        assert json.loads(config.get_json("connect/retry")) == {
            "period_init_ms": 100,
            "period_max_ms": 500,
            "period_increase_factor": 1.5,
        }


@requires_zenoh
def test_subscriber_reaches_an_owner_that_starts_listening_later() -> None:
    """A subscriber opened before its owner listens must reach it before Zenoh's default retry would.

    With Zenoh's default connect backoff (1 s, 2 s, 4 s, ...) an owner that starts listening
    between two retries stays unreachable for up to seconds, so ``SharedRobot.connect`` could
    time out on ``/metadata`` or the owner's idle timeout could stop it first.
    """
    name = f"late-owner-{uuid.uuid4().hex}"
    subscriber = open_session(name)
    owner = None
    try:
        time.sleep(3.2)  # past Zenoh's default 1 s and 2 s retries, into its 4 s period
        owner = open_session(name, listen=True)
        key = f"test/{name}/ping"
        queryable = owner.declare_queryable(key, lambda query: query.reply(key, b"pong"))
        start = time.monotonic()
        reply = None
        while reply is None and time.monotonic() - start < 5.0:
            replies = [r for r in subscriber.get(key, timeout=0.5) if r.ok is not None]
            reply = replies[0] if replies else None
            if reply is None:
                time.sleep(0.05)
        elapsed = time.monotonic() - start
        queryable.undeclare()
        assert reply is not None
        # Worst case with the fast retry: one 500 ms retry period plus one empty 0.5 s get.
        # Zenoh's default backoff would not retry again for seconds.
        assert elapsed < 2.0, f"first reply after {elapsed:.2f}s"
    finally:
        subscriber.close()
        if owner is not None:
            owner.close()
