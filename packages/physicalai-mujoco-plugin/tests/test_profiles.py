# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Model-derived profiles and public actuator channels, on small inline models (no download)."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from physicalai_mujoco_plugin.channels import ArmChannels
from physicalai_mujoco_plugin.compose import _rename
from physicalai_mujoco_plugin.profiles import (
    SO101_PROFILE,
    ChannelOverride,
    PDOverride,
    RobotProfile,
    get_profile,
    list_profiles,
    validate_profile,
)
from physicalai_mujoco_plugin.profiles.derive import derive_profile
from physicalai_mujoco_plugin.scene_registry import list_scenes_for

_MJCF = """<mujoco model="derive-test">
  <compiler angle="radian"/>
  <worldbody>
    <body name="base" pos="0 0 1">
      <freejoint name="root"/>
      <geom name="base_geom" type="sphere" size="0.05" mass="1"/>
      <body name="shoulder" pos="0 0 0.1">
        <joint name="hinge" type="hinge" axis="0 1 0" range="-1 1"/>
        <geom name="arm_geom" type="capsule" fromto="0 0 0 0 0 0.2" size="0.02"/>
        <camera name="wrist_cam" pos="0 0 0.1"/>
        <body name="slider" pos="0 0 0.2">
          <joint name="slide" type="slide" axis="0 0 1" range="0 0.1"/>
          <geom name="slider_geom" type="sphere" size="0.02" mass="0.1"/>
        </body>
      </body>
    </body>
  </worldbody>
  <tendon><fixed name="finger_tendon"><joint joint="hinge" coef="1"/></fixed></tendon>
  <actuator>
    <position name="hinge_pos" joint="hinge" kp="10" ctrlrange="-1 1"/>
    <velocity name="slide_vel" joint="slide" kv="2" ctrlrange="-1 1"/>
    <motor name="slide_motor" joint="slide"/>
    <position name="tendon_pos" tendon="finger_tendon" kp="4"/>
    <general name="hinge_raw" joint="hinge" dyntype="integrator"/>
  </actuator>
  <sensor><jointpos name="hinge_sensor" joint="hinge"/></sensor>
  <keyframe><key name="home" qpos="0 0 1 1 0 0 0 0.25 0.01" ctrl="0.25 0 0 0.1 0"/></keyframe>
</mujoco>"""

_TORQUE_ARM = """<mujoco><compiler angle="radian"/><option timestep="0.002"/><worldbody>
  <body name="link"><joint name="hinge" axis="0 1 0" range="-2 2"/>
  <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02" mass="1"/></body></worldbody>
  <actuator><motor name="hinge" joint="hinge" ctrlrange="-50 50"/></actuator></mujoco>"""


def _model(xml: str = _MJCF) -> mujoco.MjModel:
    return mujoco.MjSpec.from_string(xml).compile()


def _profile(*channels: ChannelOverride, **fields) -> RobotProfile:
    return RobotProfile(name="test", display_name="Test", menagerie_model="inline", channels=channels, **fields)


def _bound(xml: str = _MJCF, profile: RobotProfile | None = None, **kwargs) -> tuple[mujoco.MjModel, mujoco.MjData, ArmChannels]:
    model = _model(xml)
    data = mujoco.MjData(model)
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    return model, data, ArmChannels(model, data, derive_profile(model), profile or _profile(), **kwargs)


def test_derive_classifies_actuators_and_finds_robot_metadata() -> None:
    layout = derive_profile(_model())

    assert layout.joint_names == ("hinge_pos", "slide_vel", "slide_motor", "tendon_pos", "hinge_raw")
    assert [c.kind for c in layout.channels] == ["position", "velocity", "torque", "position", "raw"]
    assert [c.unit for c in layout.channels] == ["degrees", "metres", "metres", "metres", "degrees"]
    assert [c.joint for c in layout.channels] == ["hinge", "slide", "slide", None, "hinge"]
    assert layout.channels[0].range == (-1.0, 1.0)
    assert (layout.floating_base_joint, layout.base_body) == ("root", "base")
    assert layout.home_qpos["hinge"] == (0.25,)
    assert layout.home_qpos["root"] == (0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0)
    assert layout.home_ctrl["hinge_pos"] == 0.25
    assert [sensor.name for sensor in layout.sensors] == ["hinge_sensor"]
    assert [(camera.name, camera.body) for camera in layout.cameras] == [("wrist_cam", "shoulder")]


def test_without_a_keyframe_home_is_qpos0_held_by_the_position_actuators() -> None:
    xml = _MJCF.replace('<keyframe><key name="home" qpos="0 0 1 1 0 0 0 0.25 0.01" ctrl="0.25 0 0 0.1 0"/></keyframe>', "")
    layout = derive_profile(_model(xml))

    assert layout.home_qpos["hinge"] == (0.0,)
    assert layout.home_ctrl == {"hinge_pos": 0.0, "slide_vel": 0.0, "slide_motor": 0.0, "tendon_pos": 0.0, "hinge_raw": 0.0}


def test_registry_has_hand_written_profiles_and_any_menagerie_model() -> None:
    assert [profile.name for profile in list_profiles()] == ["so101", "ur5e"]
    profile = get_profile("unitree_go2")
    assert (profile.name, profile.menagerie_model, profile.tier) == ("unitree_go2", "unitree_go2", "unsupported")
    with pytest.raises(KeyError, match="Unknown robot profile"):
        get_profile("not-a-menagerie-model")


def test_scenes_list_the_profiles_they_support() -> None:
    assert set(list_scenes_for(get_profile("ur5e"))) == {"single_pick_place"}
    assert set(list_scenes_for(SO101_PROFILE, 2)) == {"garment_fold"}
    assert list_scenes_for(get_profile("unitree_go2")) == {}


def test_so101_profile_pins_the_earlier_ranges() -> None:
    validate_profile(SO101_PROFILE)
    assert SO101_PROFILE.default_unit == "normalized"
    assert [channel.name for channel in SO101_PROFILE.channels] == [
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    ]
    assert [channel.gripper for channel in SO101_PROFILE.channels] == [False] * 5 + [True]
    assert all(channel.unit is None for channel in SO101_PROFILE.channels)  # degrees stay available


@pytest.mark.parametrize(
    ("channels", "message"),
    [
        ((ChannelOverride("a", ("x",)), ChannelOverride("a", ("y",))), "unique"),
        ((ChannelOverride("a", ("x",)), ChannelOverride("b", ("x",))), "two channels"),
        ((ChannelOverride("a", ("x", "y"), member_scales=(1.0, 2.0)),), "member scales"),
        ((ChannelOverride("a", ("x",), scale=0.0),), "invalid scale"),
        ((ChannelOverride("a", ("x",), range=(1.0, -1.0)),), "invalid range"),
    ],
)
def test_validation_rejects_inconsistent_overrides(channels: tuple[ChannelOverride, ...], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        validate_profile(_profile(*channels))


def test_normalized_read_write_and_action_validation() -> None:
    _, data, channels = _bound(profile=_profile(ChannelOverride("hinge", ("hinge_pos",), unit="normalized")))

    assert channels.names == ("hinge",)
    assert channels.read_positions() == pytest.approx([25.0])  # 0.25 rad in (-1, 1)
    channels.write(np.asarray([100.0]))
    assert data.ctrl[0] == pytest.approx(1.0)
    channels.write(np.asarray([150.0]))  # clamped to the span
    assert data.ctrl[0] == pytest.approx(1.0)
    with pytest.raises(ValueError, match=r"Expected action shape \(1,\), got \(2,\)"):
        channels.write(np.asarray([0.0, 1.0]))
    with pytest.raises(ValueError, match="must be finite"):
        channels.write(np.asarray([np.nan]))


def test_degrees_mode_converts_hinges_and_keeps_slides_and_tendons_in_metres() -> None:
    model, data, channels = _bound()

    assert channels.units == ("degrees", "metres", "metres", "metres", "raw")
    positions = channels.read_positions()
    assert positions[0] == pytest.approx(np.degrees(0.25))
    assert positions[1] == pytest.approx(0.0)  # velocity channel reads its velocity
    assert positions[2] == pytest.approx(0.01)
    channels.write(np.asarray([30.0, 0.5, 0.05, 0.2, 0.7]))
    assert data.ctrl[0] == pytest.approx(np.radians(30.0))
    assert data.ctrl[1] == pytest.approx(0.5)  # kv / gain = 1 for <velocity>
    assert data.ctrl[3] == pytest.approx(0.2)  # tendon length target
    assert data.ctrl[4] == pytest.approx(0.7)  # raw passes through
    assert channels.model_targets()[0] == pytest.approx(np.radians(30.0))


def test_position_targets_clip_to_the_joint_range() -> None:
    _, data, channels = _bound()
    channels.write(np.asarray([90.0, 0.0, 0.0, 0.0, 0.0]))
    assert data.ctrl[0] == pytest.approx(1.0)


def test_unnormalizable_channel_is_reported_by_name() -> None:
    with pytest.raises(ValueError, match="'tendon_pos' has no finite joint range"):
        _bound(unit="normalized")


def test_override_names_a_missing_actuator() -> None:
    with pytest.raises(ValueError, match="no actuator or joint 'nope'"):
        _bound(profile=_profile(ChannelOverride("x", ("nope",))))


def test_groups_write_every_member_and_must_be_position_like() -> None:
    xml = """<mujoco><compiler angle="radian"/><worldbody><body><joint name="a" type="slide" range="0 0.1"/>
    <geom size="0.01"/><body><joint name="b" type="slide" range="-0.1 0"/><geom size="0.01"/></body></body></worldbody>
    <actuator><position name="a" joint="a" kp="10"/><position name="b" joint="b" kp="10"/>
    <motor name="m" joint="a"/></actuator></mujoco>"""
    _, data, channels = _bound(xml, _profile(ChannelOverride("gripper", ("a", "b"), member_scales=(-1.0,))))

    channels.write(np.asarray([0.04]))
    np.testing.assert_allclose(data.ctrl[:2], [0.04, -0.04])
    with pytest.raises(ValueError, match="only position actuators"):
        _bound(xml, _profile(ChannelOverride("bad", ("a", "m"))))


def test_scale_and_offset_map_public_values() -> None:
    profile = _profile(ChannelOverride("hinge", ("hinge_pos",), scale=-1.0, offset=10.0))
    _, data, channels = _bound(profile=profile)

    assert channels.read_positions()[0] == pytest.approx(10.0 - np.degrees(0.25))
    channels.write(np.asarray([0.0]))
    assert data.ctrl[0] == pytest.approx(np.radians(10.0))


def test_torque_channels_track_targets_through_the_software_pd() -> None:
    model, data, channels = _bound(_TORQUE_ARM, unit="degrees")
    assert channels.channels[0].first.kind == "torque"

    channels.write(np.asarray([30.0]))
    for _ in range(1000):
        channels.apply_pd()
        mujoco.mj_step(model, data)

    assert channels.read_positions()[0] == pytest.approx(30.0, abs=1.0)
    assert channels.model_targets()[0] == pytest.approx(np.radians(30.0))


@pytest.mark.parametrize("gear", [2.0, -1.0])
def test_software_pd_accounts_for_the_actuator_gear(gear: float) -> None:
    """The joint receives gear * gain * ctrl; a geared or reversed motor must still track its target."""
    model, data, channels = _bound(_TORQUE_ARM.replace('ctrlrange="-50 50"', f'gear="{gear}" ctrlrange="-50 50"'), unit="degrees")

    channels.write(np.asarray([30.0]))
    for _ in range(1000):
        channels.apply_pd()
        mujoco.mj_step(model, data)

    assert channels.read_positions()[0] == pytest.approx(30.0, abs=1.0)


def test_raw_torque_mode_passes_ctrl_through() -> None:
    _, data, channels = _bound(_TORQUE_ARM, torque_mode="raw")
    channels.write(np.asarray([3.0]))
    channels.apply_pd()
    assert data.ctrl[0] == 3.0


def test_pd_gains_can_be_overridden() -> None:
    profile = _profile(pd=PDOverride(kp={"hinge": 0.0}, kd={"hinge": 0.0}))
    model, data, channels = _bound(_TORQUE_ARM, profile)
    channels.write(np.asarray([30.0]))
    channels.apply_pd()
    assert data.ctrl[0] == 0.0


def test_renames_update_tendon_sensor_equality_and_exclusion_references(tmp_path) -> None:
    spec = mujoco.MjSpec.from_string(
        """<mujoco model="rename-test">
  <worldbody>
    <body name="body0">
      <joint name="joint0"/>
      <geom name="geom0" type="sphere" size="0.1"/>
      <site name="site0"/>
      <site name="site1"/>
    </body>
    <body name="other"><geom name="other_geom" type="sphere" size="0.1"/></body>
  </worldbody>
  <tendon><spatial name="t"><site site="site0"/><site site="site1"/></spatial></tendon>
  <actuator><motor name="motor" joint="joint0"/></actuator>
  <sensor><jointpos name="sensor" joint="joint0"/></sensor>
  <equality><weld name="w" body1="body0" body2="other"/></equality>
  <contact><exclude body1="body0" body2="other"/></contact>
</mujoco>""",
    )

    for old_name, new_name in (("joint0", "joint1"), ("body0", "body1"), ("site0", "site2")):
        spec = _rename(spec, old_name, new_name, tmp_path)

    assert spec.actuator("motor").target == "joint1"
    assert spec.sensor("sensor").objname == "joint1"
    assert spec.equalities[0].name1 == "body1"
    assert spec.excludes[0].bodyname1 == "body1"
    assert spec.tendons[0].path[0].target.name == "site2"
    assert spec.compile().nu == 1
    with pytest.raises(ValueError, match="Cannot rename 'missing'"):
        _rename(spec, "missing", "x", tmp_path)
