# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Plugin handover declarations across direct, composite, and shared robots."""

from __future__ import annotations

import subprocess
import sys
from unittest.mock import MagicMock, call
from uuid import uuid4

import numpy as np
import pytest

from physicalai.robot import CompositeRobot, HandoverCapabilities, SharedRobot
from physicalai.robot.errors import RobotTransportError
from physicalai.robot.handover import validate_handover_capabilities
from physicalai.robot.transport._owner_config import RobotOwnerConfig
from physicalai.robot.transport._owner_worker import _build_metadata
from tests.unit.robot.transport.conftest import TORQUE_FAKE_ROBOT_CLASS, requires_zenoh
from tests.unit.robot.transport.fake import FakeObservation, FakeRobot, TorqueFakeRobot


def _capabilities(*, powered: bool = True, space: str = "example-arm.radians.v1") -> dict[str, object]:
    return {
        "position_space": space,
        "position_tolerance": [0.01] * 6,
        "max_position_step": [0.02] * 6,
        "supports_alignment": powered,
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("position_space", ""),
        ("position_space", "   "),
        ("position_space", True),
        ("position_tolerance", []),
        ("position_tolerance", [0.01]),
        ("position_tolerance", [float("nan")] * 6),
        ("max_position_step", [float("inf")] * 6),
        ("max_position_step", [0.0] * 6),
        ("max_position_step", [True] * 6),
        ("max_position_step", ["0.02"] * 6),
        ("max_position_step", [-0.1] * 6),
        ("supports_alignment", "false"),
    ],
)
def test_invalid_capabilities_are_rejected(field: str, value: object) -> None:
    payload = {**_capabilities(), field: value}
    with pytest.raises(ValueError):
        validate_handover_capabilities(payload, num_joints=6)


def test_capabilities_must_match_joint_count() -> None:
    with pytest.raises(ValueError, match="joint count"):
        validate_handover_capabilities(_capabilities(), num_joints=7)


def test_undeclared_owners_do_not_load_the_optional_schema() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; "
            "from physicalai.robot.transport._owner_config import RobotOwnerConfig; "
            "from physicalai.robot.transport._owner_worker import _build_metadata; "
            "from tests.unit.robot.transport.fake import FakeRobot; "
            "config = RobotOwnerConfig(name='arm', robot={'class_path': "
            "'tests.unit.robot.transport.fake.FakeRobot', 'init_args': {}}); "
            "_build_metadata(config, FakeRobot(), (), state_dim=12); "
            "assert 'physicalai.robot.handover' not in sys.modules",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def test_missing_declaration_remains_optional() -> None:
    assert validate_handover_capabilities(None, num_joints=6) is None
    assert SharedRobot.attach("undeclared-arm").handover_capabilities is None


@pytest.mark.parametrize("powered", [False, True])
def test_plugin_metadata_round_trips_without_constructor_data(powered: bool) -> None:
    driver = TorqueFakeRobot(handover_capabilities=_capabilities(powered=powered))
    config = RobotOwnerConfig(name="arm", robot={"class_path": TORQUE_FAKE_ROBOT_CLASS, "init_args": {}})
    metadata = _build_metadata(config, driver, driver.device_ids, state_dim=12)
    shared = SharedRobot.attach("arm")
    shared._validate_metadata(metadata)
    shared._metadata = metadata

    assert shared.handover_capabilities == driver.handover_capabilities
    assert metadata["handover_capabilities"] == _capabilities(powered=powered)
    assert "init_args" not in metadata
    assert "calibration" not in metadata
    shared._metadata.pop("handover_capabilities")
    assert shared.handover_capabilities is None


def test_powered_declaration_requires_driver_torque_method() -> None:
    driver = FakeRobot(handover_capabilities=_capabilities())
    config = RobotOwnerConfig(name="arm", robot={"class_path": TORQUE_FAKE_ROBOT_CLASS, "init_args": {}})
    with pytest.raises(ValueError, match="set_torque"):
        _build_metadata(config, driver, driver.device_ids, state_dim=12)


def test_shared_robot_rejects_malformed_capabilities() -> None:
    driver = TorqueFakeRobot(handover_capabilities=_capabilities())
    config = RobotOwnerConfig(name="arm", robot={"class_path": TORQUE_FAKE_ROBOT_CLASS, "init_args": {}})
    metadata = _build_metadata(config, driver, driver.device_ids, state_dim=12)
    metadata["handover_capabilities"]["max_position_step"] = [float("inf")] * 6
    with pytest.raises(RobotTransportError, match="handover capabilities"):
        SharedRobot.attach("arm")._validate_metadata(metadata)


def test_composite_timestamp_advances_only_when_every_component_advances() -> None:
    left = TorqueFakeRobot(handover_capabilities=_capabilities())
    right = TorqueFakeRobot(handover_capabilities=_capabilities())
    positions = np.zeros(6)
    left.get_observation = MagicMock(side_effect=[FakeObservation(positions, stamp) for stamp in (10, 11, 12, 13, 13)])
    right.get_observation = MagicMock(side_effect=[FakeObservation(positions, stamp) for stamp in (20, 20, 21, 21, 22)])
    robot = CompositeRobot({"left": left, "right": right})

    assert [robot.get_observation().timestamp for _ in range(5)] == [10, 10, 12, 12, 13]


def test_composite_preserves_component_conventions_and_limits() -> None:
    left = TorqueFakeRobot(handover_capabilities=_capabilities())
    right = TorqueFakeRobot(handover_capabilities=_capabilities(space="example-arm.degrees.v1"))
    robot = CompositeRobot({"left": left, "right": right})
    capability = robot.handover_capabilities

    assert isinstance(capability, HandoverCapabilities)
    assert len(capability.position_tolerance) == 12
    assert capability.supports_alignment
    assert "radians" in capability.position_space and "degrees" in capability.position_space
    swapped = CompositeRobot({"left": right, "right": left})
    swapped_capability = swapped.handover_capabilities
    assert swapped_capability is not None
    assert swapped_capability.position_space != capability.position_space
    right.handover_capabilities = None
    assert robot.handover_capabilities is None


def test_composite_passive_component_disables_powered_alignment() -> None:
    left = TorqueFakeRobot(handover_capabilities=_capabilities())
    right = FakeRobot(handover_capabilities=_capabilities(powered=False))
    robot = CompositeRobot({"left": left, "right": right})
    capability = robot.handover_capabilities
    assert capability is not None
    assert not capability.supports_alignment
    left.set_torque = MagicMock()
    with pytest.raises(NotImplementedError):
        robot.set_torque(enabled=True)
    left.set_torque.assert_not_called()


def test_composite_rolls_back_every_component_after_failed_enable() -> None:
    left = TorqueFakeRobot()
    right = TorqueFakeRobot()
    left.connect()
    right.connect()
    right.set_torque = MagicMock(side_effect=[RuntimeError("right failed"), None])
    robot = CompositeRobot({"left": left, "right": right})

    with pytest.raises(RuntimeError, match="right failed"):
        robot.set_torque(enabled=True)

    assert left._torque_enabled is False
    assert right.set_torque.call_args_list == [call(enabled=True), call(enabled=False)]


def test_composite_attempts_all_torque_off_writes() -> None:
    left = TorqueFakeRobot()
    right = TorqueFakeRobot()
    left.set_torque = MagicMock(side_effect=RuntimeError("left failed"))
    right.set_torque = MagicMock()
    robot = CompositeRobot({"left": left, "right": right})

    with pytest.raises(RuntimeError, match="left failed"):
        robot.set_torque(enabled=False)
    right.set_torque.assert_called_once_with(enabled=False)


@requires_zenoh
def test_non_so101_capabilities_cross_the_real_owner_transport() -> None:
    name = f"handover-{uuid4().hex[:8]}"
    shared = SharedRobot(
        name,
        robot={
            "class_path": TORQUE_FAKE_ROBOT_CLASS,
            "init_args": {"device_ids": [f"fake:{name}"], "handover_capabilities": _capabilities()},
        },
    )
    owner = None
    try:
        shared.connect()
        owner = shared._owner
        assert shared.handover_capabilities == HandoverCapabilities.model_validate(_capabilities())
        shared.set_torque(enabled=True)
        shared.set_torque(enabled=False)
    finally:
        shared.disconnect()
        if owner is not None:
            owner.stop()
