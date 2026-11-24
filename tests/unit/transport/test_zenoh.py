# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any
from unittest.mock import Mock

import pytest

from physicalai.transport import make_zenoh_config, open_zenoh_session

if TYPE_CHECKING:
    from collections.abc import Callable

    import zenoh

pytest.importorskip("zenoh")


@pytest.fixture(params=[False, True], ids=["make-config", "open-session"])
def build_config(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Callable[..., zenoh.Config]:
    import zenoh

    captured: list[zenoh.Config] = []

    def capture(config: zenoh.Config) -> Mock:
        captured.append(config)
        return Mock()

    # Even explicitly requested wildcard configurations must not open real sockets.
    monkeypatch.setattr(zenoh, "open", capture)

    def build(mode: str = "peer", **kwargs: Any) -> zenoh.Config:
        if request.param:
            open_zenoh_session(mode, **kwargs)
            return captured[-1]
        return make_zenoh_config(mode, **kwargs)

    return build


@pytest.mark.parametrize("mode", ["peer", "client", "router"])
def test_defaults_disable_listeners_and_scouting(build_config: Callable[..., zenoh.Config], mode: str) -> None:
    config = build_config(mode)

    assert json.loads(config.get_json("mode")) == mode
    assert json.loads(config.get_json("listen/endpoints")) == []
    assert json.loads(config.get_json("scouting/multicast/enabled")) is False
    assert json.loads(config.get_json("scouting/gossip/enabled")) is False


def test_defaults_disable_listeners_in_supplied_config(build_config: Callable[..., zenoh.Config]) -> None:
    import zenoh

    supplied = zenoh.Config()
    supplied.insert_json5("listen/endpoints", '["tcp/127.0.0.1:20001"]')
    config = build_config(config=supplied)

    assert config is supplied
    assert json.loads(config.get_json("listen/endpoints")) == []


@pytest.mark.parametrize("host", ["127.0.0.1", "0.0.0.0"])
def test_explicit_listener_endpoints_are_kept(build_config: Callable[..., zenoh.Config], host: str) -> None:
    endpoints = [f"tcp/{host}:20001"]
    config = build_config(listen_endpoints=endpoints)

    assert json.loads(config.get_json("listen/endpoints")) == endpoints


def test_explicit_none_preserves_supplied_listener(build_config: Callable[..., zenoh.Config]) -> None:
    import zenoh

    supplied = zenoh.Config()
    endpoints = ["tcp/127.0.0.1:20001"]
    supplied.insert_json5("listen/endpoints", json.dumps(endpoints))
    config = build_config(config=supplied, listen_endpoints=None)

    assert config is supplied
    assert json.loads(config.get_json("listen/endpoints")) == endpoints


def test_explicit_none_preserves_zenoh_defaults(build_config: Callable[..., zenoh.Config]) -> None:
    import zenoh

    expected = zenoh.Config().get_json("listen/endpoints")
    config = build_config(listen_endpoints=None)

    assert config.get_json("listen/endpoints") == expected


def test_explicit_empty_endpoints_disable_supplied_listener(build_config: Callable[..., zenoh.Config]) -> None:
    import zenoh

    supplied = zenoh.Config()
    supplied.insert_json5("listen/endpoints", '["tcp/127.0.0.1:20001"]')
    config = build_config(config=supplied, listen_endpoints=[])

    assert config is supplied
    assert json.loads(config.get_json("listen/endpoints")) == []


def test_explicit_endpoints_replace_supplied_listener(build_config: Callable[..., zenoh.Config]) -> None:
    import zenoh

    supplied = zenoh.Config()
    supplied.insert_json5("listen/endpoints", '["tcp/127.0.0.1:20001"]')
    endpoints = ["tcp/127.0.0.1:20002"]
    config = build_config(config=supplied, listen_endpoints=endpoints)

    assert json.loads(config.get_json("listen/endpoints")) == endpoints


def test_connect_endpoints_and_other_config_settings_are_preserved(build_config: Callable[..., zenoh.Config]) -> None:
    import zenoh

    supplied = zenoh.Config()
    supplied.insert_json5("transport/link/rx/max_message_size", "1024")
    endpoints = ["tcp/127.0.0.1:20001"]
    config = build_config(config=supplied, connect_endpoints=endpoints)

    assert config is supplied
    assert json.loads(config.get_json("connect/endpoints")) == endpoints
    assert json.loads(config.get_json("transport/link/rx/max_message_size")) == 1024
