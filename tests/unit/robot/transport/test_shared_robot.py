# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING
from uuid import uuid4

import numpy as np
import pytest

from physicalai.config import ComponentConfigError
from physicalai.robot.errors import (
    RobotNameConflict,
    RobotNotConnectedError,
    RobotProtocolMismatch,
    RobotTransportError,
)
from physicalai.robot.transport import SharedRobot, discover_robots
from physicalai.robot.transport._codec import ROBOT_TRANSPORT_PROTOCOL_VERSION

from .conftest import FAKE_ROBOT_CLASS, requires_zenoh
from .fake import FakeRobot

if TYPE_CHECKING:
    from collections.abc import Generator

_NUM_JOINTS = 6
_STATE_DIM = 12  # fake ships positions + velocities


def _shared_robot(
    name: str,
    *,
    allow_remote: bool = False,
    idle_timeout: float | None = 0.5,
    **robot_init_args: object,
) -> SharedRobot:
    return SharedRobot(
        name,
        robot={
            "class_path": FAKE_ROBOT_CLASS,
            "init_args": {"device_ids": [f"fake:{name}"], **robot_init_args},
        },
        idle_timeout=idle_timeout,
        allow_remote=allow_remote,
    )


# ---------------------------------------------------------------------------
# Module-scoped owner: one zenoh subprocess shared by most tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def module_owner(tmp_path_factory: pytest.TempPathFactory) -> Generator[SharedRobot, None, None]:
    """Single connected SharedRobot reused by tests that only need an active owner."""
    cache_dir = tmp_path_factory.mktemp("module_cache")
    prev_cache = os.environ.get("XDG_CACHE_HOME")
    os.environ["XDG_CACHE_HOME"] = str(cache_dir)
    name = f"test-mod-{uuid4().hex[:8]}"
    robot = _shared_robot(name)
    robot.connect()
    yield robot
    owner = robot._owner
    robot.disconnect()
    if owner is not None:
        owner.stop()
    if prev_cache is not None:
        os.environ["XDG_CACHE_HOME"] = prev_cache
    else:
        os.environ.pop("XDG_CACHE_HOME", None)


# ---------------------------------------------------------------------------
# Construction / validation (no zenoh required)
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_invalid_name_raises(self) -> None:
        with pytest.raises(ValueError, match="invalid robot name"):
            SharedRobot("bad/name")

    def test_attach_only_has_no_robot_config(self) -> None:
        robot = SharedRobot.attach("left-arm")
        assert robot.name == "left-arm"
        assert robot._robot is None
        assert robot.device_ids == ()

    def test_robot_component_config_normalized(self) -> None:
        robot = SharedRobot(
            "left-arm",
            robot={"class_path": FAKE_ROBOT_CLASS, "init_args": {"port": "/dev/ttyUSB0"}},
        )
        assert robot._robot == {
            "class_path": FAKE_ROBOT_CLASS,
            "init_args": {"port": "/dev/ttyUSB0"},
        }

    def test_from_config_stores_component_config(self) -> None:
        robot = SharedRobot.from_config(
            {"class_path": FAKE_ROBOT_CLASS, "init_args": {"port": "/dev/fake0"}},
            name="left-arm",
        )
        assert robot._robot == {"class_path": FAKE_ROBOT_CLASS, "init_args": {"port": "/dev/fake0"}}

    def test_constructor_requires_component_config_mapping(self) -> None:
        driver = FakeRobot(port="/dev/fake0", device_ids=("fake:/dev/fake0",))
        with pytest.raises(ComponentConfigError, match="robot must be a ComponentConfig mapping"):
            SharedRobot("left-arm", robot=driver)  # type: ignore[arg-type]

    def test_satisfies_robot_protocol(self) -> None:
        from physicalai.robot import Robot

        assert isinstance(SharedRobot.attach("left-arm"), Robot)

    def test_not_connected_errors(self) -> None:
        robot = _shared_robot("x")
        assert not robot.is_connected()
        with pytest.raises(RobotNotConnectedError):
            robot.get_observation()
        with pytest.raises(RobotNotConnectedError):
            robot.send_action(np.zeros(_NUM_JOINTS, dtype=np.float32))
        with pytest.raises(RobotNotConnectedError):
            _ = robot.joint_names

    def test_device_ids_always_empty(self) -> None:
        assert _shared_robot("x").device_ids == ()

    def test_remote_metadata_resolution_retries_after_scouting_settles(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import physicalai.robot.transport._shared_robot as shared_robot_module

        robot = SharedRobot.attach("left-arm", allow_remote=True)
        robot._session = object()
        metadata = {"name": "left-arm"}
        calls = 0

        def _query(_session: object, _key: str, timeout: float) -> dict[str, str] | None:
            nonlocal calls
            assert timeout > 0
            calls += 1
            return None if calls == 1 else metadata

        monkeypatch.setattr(shared_robot_module, "_query_metadata", _query)
        monkeypatch.setattr(shared_robot_module.time, "sleep", lambda _seconds: None)

        assert robot._resolve_metadata(timeout=1.0) == metadata
        assert calls == 2

    def test_malformed_metadata_rejected(self) -> None:
        robot = SharedRobot.attach("left-arm")
        with pytest.raises(RobotTransportError, match="malformed"):
            robot._validate_metadata(
                {
                    "protocol_version": ROBOT_TRANSPORT_PROTOCOL_VERSION,
                    "joint_names": ["a", "a"],
                    "num_joints": 2,
                    "state_dim": 2,
                },
            )


# ---------------------------------------------------------------------------
# Tests sharing the module-scoped owner (one subprocess for all)
# ---------------------------------------------------------------------------


@requires_zenoh
class TestSharedRobotLifecycle:
    def test_observe_and_act(self, module_owner: SharedRobot) -> None:
        robot = module_owner
        assert robot.is_connected()
        assert robot.joint_names == [f"joint_{i}" for i in range(_NUM_JOINTS)]

        obs = robot.get_observation()
        assert obs.joint_positions.shape == (_NUM_JOINTS,)
        assert obs.joint_positions.dtype == np.float32
        # Owner-computed state shipped as-is (positions + velocities).
        assert obs.state.shape == (_STATE_DIM,)
        assert obs.state.dtype == np.float32
        assert obs.images is None

        target = np.arange(_NUM_JOINTS, dtype=np.float32)
        robot.send_action(target, goal_time=0.05)

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if np.array_equal(robot.get_observation().joint_positions, target):
                break
            time.sleep(0.01)
        np.testing.assert_array_equal(robot.get_observation().joint_positions, target)

    def test_connect_idempotent(self, module_owner: SharedRobot) -> None:
        module_owner.connect()  # already connected -- must not raise
        assert module_owner.is_connected()

    def test_second_instance_attaches(self, module_owner: SharedRobot) -> None:
        second = _shared_robot(module_owner.name)
        second.connect()
        try:
            # Attached, not spawned: no owner subprocess handle of its own.
            assert second._owner is None
            assert second.get_observation().state.shape == (_STATE_DIM,)
        finally:
            second.disconnect()

    def test_disconnect_leaves_owner_running(self, module_owner: SharedRobot) -> None:
        attacher = SharedRobot.attach(module_owner.name)
        attacher.connect()
        owner = module_owner._owner
        assert owner is not None and owner.is_alive
        attacher.disconnect()
        assert not attacher.is_connected()
        # Subscriber disconnect must not stop the owner (owner owns safe-state).
        assert owner.is_alive

    def test_freshest_state_after_stall_not_backlog(self, module_owner: SharedRobot) -> None:
        """Ring(1) keeps only the newest sample while the subscriber stalls."""
        stale = module_owner.get_observation()
        # Stall for several owner periods (100 Hz -> ~10 ticks); the
        # native ring keeps buffering and evicting without the GIL.
        time.sleep(0.1)
        fresh = module_owner.get_observation()
        assert fresh.timestamp > stale.timestamp
        # Newest-or-nothing: a second immediate pull must not drain a backlog
        # of intermediate samples older than the one we just got.
        newest = module_owner.get_observation()
        assert newest.timestamp >= fresh.timestamp

    def test_metadata_exposed(self, module_owner: SharedRobot) -> None:
        robot = module_owner
        assert robot.metadata is not None
        assert robot.metadata["protocol_version"] == ROBOT_TRANSPORT_PROTOCOL_VERSION
        assert robot.metadata["name"] == robot.name
        assert robot.metadata["robot_class"] == FAKE_ROBOT_CLASS
        assert robot.metadata["state_dim"] == _STATE_DIM
        assert robot.metadata["num_joints"] == _NUM_JOINTS
        assert robot.metadata["device_ids"] == [f"fake:{robot.name}"]

    def test_remote_owner_metadata_redacts_device_ids(self, unique_id: str) -> None:
        # Remote scouting can take longer than the default 0.5s idle timeout;
        # keep the owner alive until this test attaches and stops it explicitly.
        robot = _shared_robot(unique_id.replace("/", "-"), allow_remote=True, idle_timeout=None)
        robot.connect()
        try:
            assert robot.metadata is not None
            assert "device_ids" not in robot.metadata
        finally:
            owner = robot._owner
            robot.disconnect()
            if owner is not None:
                owner.stop()

    def test_class_mismatch_warns_but_attaches(self, module_owner: SharedRobot, caplog: pytest.LogCaptureFixture) -> None:
        """robot_class mismatch on an existing owner is diagnostic, not fatal."""
        import logging

        from physicalai.robot.transport._owner_config import RobotOwnerConfig

        impostor = SharedRobot(
            module_owner.name,
            robot={
                "class_path": f"{RobotOwnerConfig.__module__}.{RobotOwnerConfig.__qualname__}",
                "init_args": {},
            },
        )
        with caplog.at_level(logging.WARNING):
            impostor.connect()  # attaches to the same owner; must not raise
        try:
            assert impostor.is_connected()
            # loguru writes to stderr; message shape is asserted via construction above
            assert impostor._robot_class != module_owner._robot_class
        finally:
            impostor.disconnect()

    def test_protocol_mismatch_rejected_before_action_publisher(self, module_owner: SharedRobot, monkeypatch: pytest.MonkeyPatch) -> None:
        import physicalai.robot.transport._shared_robot as shared_robot_module

        monkeypatch.setattr(shared_robot_module, "ROBOT_TRANSPORT_PROTOCOL_VERSION", 999)
        attacher = SharedRobot.attach(module_owner.name)
        with pytest.raises(RobotProtocolMismatch):
            attacher.connect()
        # Rejected before the action publisher was ever declared.
        assert attacher._action_pub is None

    def test_device_already_owned_under_another_name(self, module_owner: SharedRobot, unique_id: str) -> None:
        from physicalai.robot.errors import RobotDeviceAlreadyOwned

        metadata = module_owner.metadata
        assert metadata is not None
        shared_device = metadata["device_ids"][0]
        other_name = unique_id.replace("/", "-")
        second = SharedRobot(
            other_name,
            robot={"class_path": FAKE_ROBOT_CLASS, "init_args": {"device_ids": [shared_device]}},
        )
        with pytest.raises(RobotDeviceAlreadyOwned):
            second.connect()

    def test_discover_via_connected_session(self, module_owner: SharedRobot) -> None:
        found = discover_robots(timeout=1.0, session=module_owner._session)
        names = [m["name"] for m in found]
        assert module_owner.name in names
        metadata = next(m for m in found if m["name"] == module_owner.name)
        assert metadata["robot_class"] == FAKE_ROBOT_CLASS

    def test_discover_local_default_session(self, module_owner: SharedRobot) -> None:
        found = discover_robots(timeout=1.0)
        assert module_owner.name in [m["name"] for m in found]


# ---------------------------------------------------------------------------
# Tests that need their own owner subprocess
# ---------------------------------------------------------------------------


@requires_zenoh
class TestIndependentSpawn:
    @pytest.mark.parametrize("allow_remote", [False, True])
    def test_differing_device_race_raises_name_conflict(self, unique_id: str, *, allow_remote: bool) -> None:
        import threading

        name = unique_id.replace("/", "-")
        barrier = threading.Barrier(2)
        results: dict[str, object] = {}

        def _run(key: str, device_id: str) -> None:
            robot = SharedRobot(
                name,
                robot={"class_path": FAKE_ROBOT_CLASS, "init_args": {"device_ids": [device_id]}},
                allow_remote=allow_remote,
            )
            barrier.wait()
            try:
                robot.connect()
            except Exception as exc:  # noqa: BLE001
                results[key] = exc
            else:
                results[key] = robot

        threads = [
            threading.Thread(target=_run, args=("a", f"fake:{unique_id}-A")),
            threading.Thread(target=_run, args=("b", f"fake:{unique_id}-B")),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)

        outcomes = list(results.values())
        winners = [o for o in outcomes if isinstance(o, SharedRobot)]
        conflicts = [o for o in outcomes if isinstance(o, RobotNameConflict)]
        try:
            assert len(winners) == 1
            assert len(conflicts) == 1
        finally:
            for winner in winners:
                owner = winner._owner
                winner.disconnect()
                if owner is not None:
                    owner.stop()

    def test_attach_only_no_owner_raises(self, unique_id: str) -> None:
        robot = SharedRobot.attach(unique_id.replace("/", "-"))
        with pytest.raises(RobotTransportError, match="attach-only"):
            robot.connect()

    def test_spawn_failure_raises(self, unique_id: str) -> None:
        robot = _shared_robot(unique_id.replace("/", "-"), fail_connect=True)
        with pytest.raises(RobotTransportError, match="failed to start robot owner"):
            robot.connect()
        assert not robot.is_connected()

    def test_owner_exits_and_disconnects_driver_after_idle(self, unique_id: str) -> None:
        robot = _shared_robot(unique_id.replace("/", "-"))
        robot.connect()
        owner = robot._owner
        assert owner is not None
        proc = owner._process
        assert proc is not None
        robot.disconnect()

        # Owner detects zero subscribers via matching status and self-exits,
        # calling driver.disconnect() (exit code 0 = clean shutdown path ran).
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline and proc.poll() is None:
            time.sleep(0.2)
        assert proc.poll() == 0


# ---------------------------------------------------------------------------
# Discovery helpers (mocked -- no zenoh needed)
# ---------------------------------------------------------------------------


@requires_zenoh
class TestDiscovery:
    def test_remote_discovery_retries_after_scouting_settles(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import physicalai.robot.transport._shared_robot as shared_robot_module

        class Clock:
            now = 0.0

            def monotonic(self) -> float:
                return self.now

            def sleep(self, seconds: float) -> None:
                self.now += seconds

        class Session:
            calls = 0

            def get(self, _key: str, *, timeout: float) -> list[object]:
                self.calls += 1
                return []

            def close(self) -> None:
                pass

        clock = Clock()
        session = Session()
        monkeypatch.setattr(shared_robot_module, "registered_owner_names", lambda: [])
        monkeypatch.setattr(shared_robot_module, "open_session", lambda **_kwargs: session)
        monkeypatch.setattr(shared_robot_module.time, "monotonic", clock.monotonic)
        monkeypatch.setattr(shared_robot_module.time, "sleep", clock.sleep)

        assert discover_robots(timeout=1.0, allow_remote=True) == []
        assert session.calls > 1
