# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""LeRobot-derived numeric degree fixtures, without a LeRobot dependency."""

from types import SimpleNamespace

import numpy as np
import pytest

from physicalai.config import Config
from physicalai_so101_plugin import SO101Calibration
from physicalai_so101_plugin.constants import SO101_JOINT_ORDER


def calibration(range_max=3100):
    return SO101Calibration.from_dict({
        name: {"id": servo_id, "drive_mode": 0, "homing_offset": -100, "range_min": 100, "range_max": range_max}
        for servo_id, name in enumerate(SO101_JOINT_ORDER, 1)
    })


def robot(**kwargs):
    from physicalai_so101_plugin import SO101
    return SO101("fake", calibration=calibration(), **kwargs)


@pytest.mark.parametrize("role", ["leader", "follower"])
def test_degrees_uses_the_single_unit_setting(role):
    by_unit = robot(role=role, unit="degrees")
    assert by_unit.unit == "degrees"
    assert not hasattr(by_unit, "use_degrees")
    assert robot(role=role).unit == "normalized"


@pytest.mark.parametrize("tick,body,gripper", [
    (-20, -142.41758241758242, 0),
    (0, -140.65934065934067, 0),
    (100, -131.86813186813185, 0),
    (1600, 0, 50),
    (1601, 0.08791208791208792, 50.03333333333333),
    (3100, 131.86813186813185, 100),
    (4095, 219.34065934065933, 100),
])
def test_degree_read_golden_values_use_range_midpoint_and_4095(tick, body, gripper):
    arm = robot(unit="degrees")
    values = arm._ticks_to_unit(np.full(6, tick))
    assert values.dtype == np.float32
    np.testing.assert_allclose(values, [body] * 5 + [gripper], rtol=1e-6, atol=1e-6)
    assert values.shape == (6,)


@pytest.mark.parametrize("degree,body_tick,grip_value,grip_tick", [
    (0, 1600, 50, 1600),
    (0.1, 1601, 0.05, 101),
    (-0.1, 1598, -1, 100),
    (180, 3647, 101, 3100),
    (-180, -447, 0, 100),
])
def test_degree_write_truncates_without_recorded_body_range_clipping(degree, body_tick, grip_value, grip_tick):
    arm = robot(unit="degrees")
    ticks = arm._unit_to_ticks(np.array([degree] * 5 + [grip_value], dtype=np.float64))
    assert ticks.dtype == np.int32
    assert ticks.tolist() == [body_tick] * 5 + [grip_tick]


def test_odd_range_midpoint_and_nonzero_homing_not_double_applied():
    from physicalai_so101_plugin import SO101
    arm = SO101("fake", calibration=calibration(range_max=3101), unit="degrees")
    values = arm._ticks_to_unit(np.full(6, 1600))
    assert values[0] == pytest.approx(-0.04395604395604396)
    assert arm._unit_to_ticks(np.zeros(6))[0] == 1600  # midpoint 1600.5, truncated.


def test_degree_observation_and_follower_action_use_the_selected_units(monkeypatch):
    arm = robot(unit="degrees")
    monkeypatch.setattr(arm, "_read_joint_positions", lambda: np.full(6, 1600, dtype=np.int32))
    observation = arm.get_observation()
    np.testing.assert_array_equal(observation.joint_positions, [0] * 5 + [50])
    written = []
    monkeypatch.setattr(arm, "_write_joint_positions", lambda ticks: written.append(ticks.copy()))
    arm.send_action(np.array([180] * 5 + [50], dtype=np.float32))
    np.testing.assert_array_equal(written[0], [3647] * 5 + [1600])


def test_degree_reads_decode_signed_positions_without_changing_normalized_driver():
    def connect_fake(arm):
        group = SimpleNamespace(
            txRxPacket=lambda: 0, isAvailable=lambda *args: True, getData=lambda *args: 0x8014,
        )
        arm._connection = SimpleNamespace(group_sync_read=group)
    arm = robot(unit="degrees")
    connect_fake(arm)
    assert arm._read_joint_positions().tolist() == [-20] * 6
    existing = robot()
    connect_fake(existing)
    assert existing._read_joint_positions().tolist() == [0x8014] * 6


def test_redundant_boolean_alias_is_not_a_constructor_setting():
    with pytest.raises(TypeError, match="use_degrees"):
        robot(use_degrees=True)


def test_degrees_requires_calibration_and_still_rejects_free_leader_actions():
    from physicalai_so101_plugin import SO101
    with pytest.raises(ValueError, match="calibration"):
        SO101("fake", unit="degrees")
    with pytest.raises(ValueError, match="ticks"):
        SO101("fake", allow_uncalibrated=True, unit="degrees")
    with pytest.raises(RuntimeError, match="torque off"):
        robot(role="leader", unit="degrees").send_action(np.zeros(6))


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_degree_action_rejected(value):
    with pytest.raises(ValueError, match="finite"):
        robot(unit="degrees")._unit_to_ticks(np.full(6, value))


def test_extreme_finite_actions_do_not_wrap_encoder_safety_clamp():
    arm = robot(unit="degrees")
    assert arm._unit_to_ticks(np.full(6, 1e308))[0] == np.iinfo(np.int32).max
    assert arm._unit_to_ticks(np.full(6, -1e308))[0] == np.iinfo(np.int32).min


def test_degree_recipe_roundtrips_and_is_jsonargparse_loadable():
    from jsonargparse import ArgumentParser
    from physicalai_so101_plugin import SO101
    arm = robot(unit="degrees", role="leader")
    recipe = Config.from_instance(arm).to_dict()
    assert recipe["init_args"]["unit"] == "degrees"
    assert "use_degrees" not in recipe["init_args"]
    restored = Config.from_dict(recipe).instantiate()
    assert restored.unit == "degrees"
    assert restored.role == "leader" and not restored.is_connected()
    assert Config.from_instance(restored).to_dict() == recipe
    parser = ArgumentParser()
    parser.add_class_arguments(SO101, "robot")
    namespace = parser.parse_object({"robot": recipe["init_args"]})
    assert namespace.robot.unit == "degrees"


def test_default_normalized_gripper_and_body_behavior_remain_unchanged():
    arm = robot()
    np.testing.assert_array_equal(arm._ticks_to_unit(np.full(6, 1600)), [0] * 5 + [50])
    # The original normalized path rounds; only opting into degree mode uses
    # LeRobot-style integer truncation for its mixed degree/percentage actions.
    values = np.array([0] * 5 + [0.05])
    assert arm._unit_to_ticks(values)[-1] == 102
    assert robot(unit="degrees")._unit_to_ticks(values)[-1] == 101


def test_bimanual_degree_recipes_preserve_both_arm_modes():
    from physicalai_so101_plugin import BimanualSO101
    pair = BimanualSO101(
        left=robot(role="leader", unit="degrees"),
        right=robot(role="leader", unit="degrees"),
    )
    recipe = Config.from_instance(pair).to_dict()
    assert recipe["init_args"]["left"]["init_args"]["unit"] == "degrees"
    assert recipe["init_args"]["right"]["init_args"]["unit"] == "degrees"
    restored = Config.from_dict(recipe).instantiate()
    assert restored.left.unit == restored.right.unit == "degrees"


def test_degree_actions_keep_existing_final_encoder_safety_clamp():
    arm = robot(unit="degrees")
    parameters = {}
    def add_param(servo_id, data):
        parameters[servo_id] = int.from_bytes(bytes(data), "little")
        return True
    group = SimpleNamespace(clearParam=lambda: None, addParam=add_param, txPacket=lambda: 0)
    arm._connection = SimpleNamespace(group_sync_write=group)
    arm.send_action(np.array([-180] * 5 + [50], dtype=np.float64))
    assert list(parameters.values()) == [0] * 5 + [1600]
    arm.send_action(np.array([360] * 5 + [50], dtype=np.float64))
    assert list(parameters.values()) == [4095] * 5 + [1600]
