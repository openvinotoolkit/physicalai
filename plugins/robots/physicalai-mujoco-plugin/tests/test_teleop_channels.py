# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Which channels teleoperation drives, and the gripper targets of a geared tendon."""

# MuJoCo model/data attributes are supplied by the C extension at runtime.
# pyrefly: ignore-errors [missing-attribute]

from __future__ import annotations

import mujoco
import pytest

from physicalai_mujoco_plugin.channels import ArmChannels
from physicalai_mujoco_plugin.compose import RobotBinding
from physicalai_mujoco_plugin.profiles import EndEffector, RobotProfile
from physicalai_mujoco_plugin.profiles.derive import derive_profile
from physicalai_mujoco_plugin.teleop import build_arms

_ARM = """<mujoco><compiler angle="radian"/><worldbody>
  <body name="link"><joint name="hinge" axis="0 1 0" range="-2 2"/>
    <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02" mass="1"/>
    <body name="finger" pos="0.3 0 0"><joint name="finger" type="slide" axis="0 1 0"/>
      <geom type="box" size="0.01 0.01 0.01" mass="0.05"/></body></body></worldbody>
  <tendon><fixed name="grip"><joint joint="finger" coef="1"/></fixed></tendon>
  <actuator>HINGE<position name="grip" tendon="grip" kp="10" gear="GEAR" ctrlrange="0 0.04"/></actuator></mujoco>"""

_POSITION = '<position name="hinge" joint="hinge" kp="10"/>'
_MOTOR = '<motor name="hinge" joint="hinge" ctrlrange="-50 50"/>'


def _arms(*, gear: float = 1.0, hinge: str = _POSITION, torque_mode: str = "pd"):
    model = mujoco.MjSpec.from_string(_ARM.replace("HINGE", hinge).replace("GEAR", str(gear))).compile()
    data = mujoco.MjData(model)
    layout = derive_profile(model)
    profile = RobotProfile(
        name="test",
        display_name="Test",
        menagerie_model="inline",
        end_effectors=(EndEffector(body="link", gripper="grip", gripper_opens_up=True),),
    )
    channels = ArmChannels(model, data, layout, profile, torque_mode=torque_mode)
    return build_arms(model, (RobotBinding("", layout),), (channels,), profile)


@pytest.mark.parametrize(("gear", "closed", "opened"), [(1.0, 0.0, 0.04), (2.0, 0.0, 0.02), (-2.0, -0.02, 0.0)])
def test_a_tendon_grippers_targets_divide_its_control_range_by_the_gear(gear, closed, opened) -> None:
    (arm,) = _arms(gear=gear)
    assert arm.gripper is not None
    assert (arm.gripper.closed, arm.gripper.opened) == pytest.approx((closed, opened))


def test_an_arm_on_raw_torques_is_not_teleoperable() -> None:
    assert len(_arms(hinge=_MOTOR, torque_mode="pd")) == 1
    assert _arms(hinge=_MOTOR, torque_mode="raw") == ()
