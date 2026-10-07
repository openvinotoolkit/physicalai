# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Model-derived profiles and public actuator channels, on small inline models (no download)."""

from __future__ import annotations

import dataclasses
import re
from math import degrees
from pathlib import Path
from uuid import uuid4

import mujoco
import numpy as np
import pytest

from physicalai_mujoco_plugin import compose
from physicalai_mujoco_plugin.channels import REPLAY_MODEL_LIMIT, ArmChannels
from physicalai_mujoco_plugin.compose import _rename
from physicalai_mujoco_plugin.profiles import (
    SO101_PROFILE,
    ChannelOverride,
    EndEffector,
    PDOverride,
    RobotProfile,
    get_profile,
    list_profiles,
    validate_profile,
)
from physicalai_mujoco_plugin.profiles.derive import derive_profile
from physicalai_mujoco_plugin.scene_registry import get_scene, list_scenes_for

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


_VELOCITY_ARM = """<mujoco><compiler angle="radian"/><option timestep="0.002" integrator="implicitfast"/><worldbody>
  <body name="link"><joint name="hinge" axis="0 0 1" range="-2 2"/>
  <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02" mass="1"/></body></worldbody>
  <actuator><velocity name="hinge" joint="hinge" kv="50" ctrlrange="-1 1"/></actuator>
  <keyframe><key qpos="0.7"/></keyframe></mujoco>"""


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
    assert (layout.base.joint, layout.base.body) == ("root", "base")
    assert (layout.base.qpos_adr, layout.base.dof_adr) == (0, 0)
    assert layout.base.home == (0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0)
    assert layout.home_qpos == {"hinge": (0.25,), "slide": (0.01,)}  # the base's pose is in layout.base
    assert layout.home_ctrl["hinge_pos"] == 0.25
    assert [sensor.name for sensor in layout.sensors] == ["hinge_sensor"]
    assert [(camera.name, camera.body) for camera in layout.cameras] == [("wrist_cam", "shoulder")]


def test_without_a_keyframe_home_is_qpos0_held_by_the_position_actuators() -> None:
    xml = _MJCF.replace('<keyframe><key name="home" qpos="0 0 1 1 0 0 0 0.25 0.01" ctrl="0.25 0 0 0.1 0"/></keyframe>', "")
    layout = derive_profile(_model(xml))

    assert layout.home_qpos["hinge"] == (0.0,)
    assert layout.home_ctrl == {"hinge_pos": 0.0, "slide_vel": 0.0, "slide_motor": 0.0, "tendon_pos": 0.0, "hinge_raw": 0.0}


def test_registry_has_hand_written_profiles_and_any_menagerie_model() -> None:
    assert [profile.name for profile in list_profiles()] == [
        "so101",
        "trossen_wxai",
        "rebot_b601",
        "ur5e",
        "aloha",
        "so_arm100",
        "koch",
        "piper",
        "franka_fr3",
        "franka_panda",
        "xarm7",
        "kinova_gen3",
        "unitree_g1",
        "unitree_go2",
        "boston_dynamics_spot",
    ]
    assert {get_profile(name).tier for name in ("unitree_g1", "unitree_go2", "boston_dynamics_spot")} == {
        "experimental"
    }
    profile = get_profile("unitree_go1")
    assert (profile.name, profile.menagerie_model, profile.tier) == ("unitree_go1", "unitree_go1", "unsupported")
    with pytest.raises(KeyError, match="Unknown robot profile"):
        get_profile("not-a-menagerie-model")


def test_dataset_tier_arms_run_in_single_pick_place() -> None:
    dataset = [profile for profile in list_profiles() if profile.tier == "dataset"]

    assert [profile.name for profile in dataset] == [
        "ur5e",
        "aloha",
        "so_arm100",
        "koch",
        "piper",
        "franka_fr3",
        "franka_panda",
        "xarm7",
        "kinova_gen3",
    ]
    for profile in dataset:
        validate_profile(profile)
        assert profile.default_scene == "single_pick_place"
        assert get_scene("single_pick_place").supports(profile)


@pytest.mark.requires_download
def test_scenes_list_the_profiles_they_support(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert set(list_scenes_for(get_profile("ur5e"))) == {"single_pick_place"}
    assert set(list_scenes_for(SO101_PROFILE, 2)) == {"single_pick_place", "yahtzee", "conveyor_sort", "garment_fold"}
    assert set(list_scenes_for(get_profile("trossen_wxai"), 2)) == {"single_pick_place", "garment_fold"}
    assert list_scenes_for(get_profile("aloha"), 2) == {}
    assert list_scenes_for(get_profile("unitree_go2"), 2) == {}
    # Floor scenes take any floating-base robot, and only those (SCN-5); the unregistered robot is inline.
    assert set(list_scenes_for(get_profile("unitree_go2"))) == {"floor_flat"}
    robot = tmp_path / "floating.xml"
    robot.write_text(_MJCF)
    monkeypatch.setattr(compose, "fetch_profile", lambda _profile: robot)
    floating = RobotProfile(name=f"inline-floating-{uuid4().hex}", display_name="Floating", menagerie_model="inline")
    assert set(list_scenes_for(floating)) == {"floor_flat"}
    assert "floor_flat" not in list_scenes_for(SO101_PROFILE)


def test_derive_finds_a_floating_base_driven_through_a_site() -> None:
    """Drones act through site transmissions; their root free joint is still the floating base (PRF-3)."""
    xml = """<mujoco><worldbody><body name="drone" pos="0 0 1"><freejoint/>
      <geom type="box" size="0.1 0.1 0.02" mass="1"/><site name="rotor"/></body>
      <body name="crate" pos="1 0 0"><freejoint name="crate"/><geom type="box" size="0.1 0.1 0.1"/></body></worldbody>
      <actuator><motor name="thrust" site="rotor" gear="0 0 1 0 0 0"/></actuator></mujoco>"""
    layout = derive_profile(_model(xml))

    assert (layout.base.body, layout.base.joint) == ("drone", "")
    assert layout.base.home == (0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0)
    assert layout.home_qpos == {"crate": (1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)}


def test_derive_leaves_unactuated_free_bodies_alone() -> None:
    xml = """<mujoco><worldbody><body name="arm"><joint name="j" axis="0 0 1"/><geom size="0.05"/></body>
      <body name="cube" pos="1 0 0"><freejoint/><geom type="box" size="0.1 0.1 0.1"/></body></worldbody>
      <actuator><position name="j" joint="j" kp="1"/></actuator></mujoco>"""
    assert derive_profile(_model(xml)).base is None


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


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"end_effectors": (EndEffector(),)}, "exactly one of a site or a body"),
        ({"end_effectors": (EndEffector(site="tcp", body="hand"),)}, "exactly one of a site or a body"),
        ({"end_effectors": (EndEffector(body="hand", pos=(0.0, float("nan"), 0.1)),)}, "invalid position"),
        ({"end_effectors": (EndEffector(site="tcp", axis=(0.0, 0.0, 0.0)),)}, "invalid approach axis"),
        ({"reach": 0.0}, "invalid reach"),
        ({"reach": float("inf")}, "invalid reach"),
    ],
)
def test_validation_rejects_invalid_end_effectors_and_reach(fields: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        validate_profile(_profile(**fields))


@pytest.mark.requires_download
@pytest.mark.parametrize(("name", "base"), [("koch", "base_link"), ("so_arm100", "Base")])
def test_turned_dataset_arms_face_x_with_their_joints_unchanged(name: str, base: str) -> None:
    """The Koch and SO-ARM100 turn about their base only: dataset joint names, ranges and zero pose stay."""
    profile = get_profile(name)
    turned = compose.load_robot_spec(profile).compile()
    original = compose.load_robot_spec(dataclasses.replace(profile, customize=None)).compile()
    tips = []
    for model in (turned, original):
        data = mujoco.MjData(model)
        mujoco.mj_kinematics(model, data)
        effector = profile.end_effectors[0]
        tips.append(data.xpos[model.body(effector.body).id][:2] - data.xpos[model.body(base).id][:2])

    assert [turned.joint(i).name for i in range(turned.njnt)] == [original.joint(i).name for i in range(original.njnt)]
    np.testing.assert_array_equal(turned.jnt_range, original.jnt_range)
    np.testing.assert_array_equal(turned.qpos0, original.qpos0)
    # At the zero pose the arm now points along +x; Menagerie's points along -x (Koch) or -y (SO-ARM100).
    turned_heading, original_heading = (tip[0] / np.linalg.norm(tip) for tip in tips)
    assert turned_heading > 0.9  # noqa: PLR2004
    assert original_heading < 0.1  # noqa: PLR2004


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
    assert positions[1] == pytest.approx(0.01)  # velocity channel reads its joint position
    assert positions[2] == pytest.approx(0.01)
    channels.write(np.asarray([30.0, 0.5, 0.05, 0.2, 0.7]))
    assert data.ctrl[0] == pytest.approx(np.radians(30.0))
    assert channels.model_targets()[1] == pytest.approx(0.1)  # a position, clipped to the slide's range
    # kv / gain = 1 for <velocity>; the weak slide (kv 2) caps its tracking gain below 10 / s.
    gain = channels._velocity_gains_by_index[1]  # noqa: SLF001
    assert 0.0 < gain < 10.0
    assert data.ctrl[1] == pytest.approx(gain * (0.1 - 0.01))
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


def test_placing_a_group_homes_every_member() -> None:
    xml = """<mujoco><compiler angle="radian"/><worldbody><body><joint name="a" type="slide" range="0 0.1"/>
    <geom size="0.01"/><body><joint name="b" type="slide" range="-0.1 0"/><geom size="0.01"/></body></body></worldbody>
    <actuator><position name="a" joint="a" kp="10"/><position name="b" joint="b" kp="10"/></actuator></mujoco>"""
    model, data, channels = _bound(xml, _profile(ChannelOverride("gripper", ("a", "b"), member_scales=(-1.0,))))
    data.qpos[:] = [0.05, -0.05]

    channels.place({"a": 0.02, "b": -0.02})

    np.testing.assert_allclose(data.qpos, [0.02, -0.02])
    np.testing.assert_allclose(data.ctrl, [0.02, -0.02])


def test_setting_positions_moves_the_joints_but_keeps_the_targets() -> None:
    joint_channels = (
        ChannelOverride("hinge_pos", ("hinge_pos",)),
        ChannelOverride("slide_vel", ("slide_vel",)),
        ChannelOverride("hinge_raw", ("hinge_raw",)),
    )
    model, data, channels = _bound(profile=_profile(*joint_channels))
    targets = channels.model_targets()
    data.qvel[model.joint("slide").dofadr[0]] = 0.3
    # Past the hinge's range of 1 rad: recorded positions replay unclipped.
    positions = np.asarray([np.degrees(1.2), 0.05, 1.2])

    channels.set_positions(positions)
    mujoco.mj_forward(model, data)

    assert channels.unpositionable == ()
    np.testing.assert_allclose(channels.read_positions(), positions)
    np.testing.assert_array_equal(channels.model_targets(), targets)
    assert data.qvel[model.joint("slide").dofadr[0]] == 0.0


def test_tendon_channels_cannot_be_placed() -> None:
    model, data, channels = _bound()
    qpos = data.qpos.copy()

    assert channels.unpositionable == ("tendon_pos",)
    with pytest.raises(ValueError, match=r"\['tendon_pos'\] drive no hinge or slide joint directly"):
        channels.set_positions(np.zeros(len(channels)))
    np.testing.assert_array_equal(data.qpos, qpos)


def test_setting_positions_moves_every_group_member() -> None:
    xml = """<mujoco><compiler angle="radian"/><worldbody><body><joint name="a" type="slide" range="0 0.1"/>
    <geom size="0.01"/><body><joint name="b" type="slide" range="-0.1 0"/><geom size="0.01"/></body></body></worldbody>
    <actuator><position name="a" joint="a" kp="10"/><position name="b" joint="b" kp="10"/></actuator></mujoco>"""
    _, data, channels = _bound(xml, _profile(ChannelOverride("gripper", ("a", "b"), member_scales=(-1.0,))))

    channels.set_positions(np.asarray([0.03]))

    np.testing.assert_allclose(data.qpos, [0.03, -0.03])
    np.testing.assert_allclose(channels.read_positions(), [0.03])


def test_pd_targets_clip_to_the_joint_range() -> None:
    _, _, channels = _bound(_TORQUE_ARM, unit="degrees")
    channels.write(np.asarray([500.0]))
    assert channels.model_targets()[0] == pytest.approx(2.0)  # the joint's upper limit, not 500 degrees


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
    assert channels.read_positions()[0] == pytest.approx(0.0)  # the joint, not the torque command


def test_velocity_and_raw_channels_observe_positions_not_commands() -> None:
    model, data, channels = _bound()
    channels.write(np.asarray([np.degrees(0.25), 0.5, 0.01, 0.1, 0.7]))
    for _ in range(20):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)

    positions = channels.read_positions()
    assert positions[1] == pytest.approx(data.qpos[model.jnt_qposadr[model.joint("slide").id]])
    assert positions[1] != pytest.approx(data.actuator_velocity[1])
    assert positions[4] == pytest.approx(data.qpos[model.jnt_qposadr[model.joint("hinge").id]])  # raw: radians
    assert positions[4] != pytest.approx(data.ctrl[4])
    assert channels.read_velocities()[1] == pytest.approx(data.qvel[model.jnt_dofadr[model.joint("slide").id]])


def _run(model: mujoco.MjModel, data: mujoco.MjData, channels: ArmChannels, steps: int) -> list[float]:
    ctrls = []
    for _ in range(steps):
        channels.apply_pd()
        ctrls.append(float(data.ctrl[0]))
        mujoco.mj_step(model, data)
    return ctrls


def test_echoing_a_velocity_channels_position_holds_the_joint() -> None:
    model, data, channels = _bound(_VELOCITY_ARM, unit="degrees")
    assert channels.channels[0].first.kind == "velocity"
    start = channels.read_positions()

    for _ in range(50):
        channels.write(channels.read_positions())
        _run(model, data, channels, 10)

    assert channels.read_positions()[0] == pytest.approx(start[0], abs=0.01)
    assert channels.read_velocities()[0] == pytest.approx(0.0, abs=0.01)


@pytest.mark.parametrize("kv", [0.05, 50.0])
@pytest.mark.parametrize("transmission", ["joint", "tendon"])
def test_velocity_tracking_does_not_overshoot_a_weak_actuator(kv: float, transmission: str) -> None:
    """A weak velocity servo lags its command; the position loop's gain is capped so it doesn't ring."""
    xml = _VELOCITY_ARM.replace('kv="50" ctrlrange="-1 1"', f'kv="{kv}"')
    step = 30.0  # degrees on the hinge
    if transmission == "tendon":
        xml = xml.replace('joint="hinge" kv', 'tendon="drive" kv').replace(
            "<actuator>", '<tendon><fixed name="drive"><joint joint="hinge" coef="1"/></fixed></tendon><actuator>'
        )
        step = 0.5  # metres of tendon = radians of the hinge
    model, data, channels = _bound(xml, unit="degrees")
    start = channels.read_positions()[0]
    channels.write(np.asarray([start + step]))

    peak = start
    for _ in range(4000):  # 8 s
        channels.apply_pd()
        mujoco.mj_step(model, data)
        peak = max(peak, channels.read_positions()[0])

    assert peak <= start + step * 1.002
    assert channels.read_positions()[0] == pytest.approx(start + step, abs=step * 0.03)


def test_velocity_channels_drive_the_joint_to_a_position_target_within_ctrlrange() -> None:
    model, data, channels = _bound(_VELOCITY_ARM, unit="degrees")

    channels.write(np.asarray([-30.0]))
    ctrls = _run(model, data, channels, 1000)

    assert channels.read_positions()[0] == pytest.approx(-30.0, abs=0.5)
    assert channels.model_targets()[0] == pytest.approx(np.radians(-30.0))
    assert min(ctrls) == pytest.approx(-1.0)  # 70 degrees away commands more than the 1 rad/s ctrlrange
    assert all(-1.0 <= ctrl <= 1.0 for ctrl in ctrls)


def test_pd_torque_mode_rejects_a_torque_actuator_without_a_joint() -> None:
    xml = _TORQUE_ARM.replace(
        '<actuator><motor name="hinge" joint="hinge" ctrlrange="-50 50"/></actuator>',
        '<tendon><fixed name="cable"><joint joint="hinge" coef="1"/></fixed></tendon>'
        '<actuator><motor name="cable_motor" tendon="cable" ctrlrange="-50 50"/></actuator>',
    )
    with pytest.raises(ValueError, match=r"'cable_motor' is a torque actuator without a hinge or slide joint.*torque_mode='raw'"):
        _bound(xml)
    _, data, channels = _bound(xml, torque_mode="raw")
    channels.write(np.asarray([3.0]))
    assert data.ctrl[0] == 3.0


def test_prefixed_pd_gains_use_the_unprefixed_override_names() -> None:
    xml = _TORQUE_ARM.replace('"hinge"', '"left_hinge"')
    profile = _profile(ChannelOverride("hinge", ("hinge",)), pd=PDOverride(kp={"hinge": 0.0}, kd={"hinge": 0.0}))
    _, data, channels = _bound(xml, profile, prefix="left_")
    assert channels.names == ("left_hinge",)
    channels.write(np.asarray([30.0]))
    channels.apply_pd()
    assert data.ctrl[0] == 0.0


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
  <sensor><jointpos name="sensor" joint="joint0"/><actuatorfrc name="force" actuator="motor"/></sensor>
  <equality><weld name="w" body1="body0" body2="other"/></equality>
  <contact><exclude body1="body0" body2="other"/></contact>
</mujoco>""",
    )

    for old_name, new_name in (("joint0", "joint1"), ("body0", "body1"), ("site0", "site2"), ("motor", "motor1")):
        spec = _rename(spec, old_name, new_name, tmp_path)

    assert spec.actuator("motor1").target == "joint1"
    assert spec.sensor("force").objname == "motor1"
    assert spec.sensor("sensor").objname == "joint1"
    assert spec.equalities[0].name1 == "body1"
    assert spec.excludes[0].bodyname1 == "body1"
    assert spec.tendons[0].path[0].target.name == "site2"
    assert spec.compile().nu == 1
    with pytest.raises(ValueError, match="Cannot rename 'missing'"):
        _rename(spec, "missing", "x", tmp_path)


def test_replay_refuses_normalized_positions_it_would_clamp() -> None:
    _, data, channels = _bound(_TORQUE_ARM, unit="normalized")
    qpos = data.qpos.copy()

    with pytest.raises(ValueError, match=r"frame 0: hinge = 150.0 is outside its normalized range"):
        channels.set_positions(np.asarray([150.0]))
    np.testing.assert_array_equal(data.qpos, qpos)
    for value in (-100.0, 100.0):
        channels.set_positions(np.asarray([value]))
        np.testing.assert_allclose(channels.read_positions(), [value])


def test_replay_refuses_a_joint_beyond_the_model_limit() -> None:
    _, data, channels = _bound(_TORQUE_ARM, unit="degrees")
    limit = degrees(REPLAY_MODEL_LIMIT)

    channels.set_positions(np.asarray([limit * (1 - 1e-12)]))
    for value in (limit * (1 + 1e-9), 1e40, -1e40):
        with pytest.raises(ValueError, match=re.escape(f"puts hinge beyond {REPLAY_MODEL_LIMIT:g} model units")):
            channels.set_positions(np.asarray([value]))


def _show(channels: ArmChannels, data: mujoco.MjData, value: float | np.ndarray) -> np.ndarray:
    """Place one frame of public values, then read every channel back as the float32 observation does."""
    channels.set_positions(np.atleast_1d(np.asarray(value, dtype=np.float64)))
    assert np.isfinite(data.qpos).all()
    return channels.read_positions().astype(np.float32)


@pytest.mark.parametrize(
    ("override", "unit"),
    [
        # The extremes a profile may use: the observation still tells the whole range apart.
        (ChannelOverride("hinge", ("hinge",), scale=1e4, offset=1e7), "normalized"),
        (ChannelOverride("hinge", ("hinge",), scale=-1e-3, offset=1.0), "normalized"),
        (ChannelOverride("hinge", ("hinge",), scale=1e-3, offset=-1.0), "degrees"),
        (ChannelOverride("hinge", ("hinge",), scale=-1e4), "degrees"),
    ],
)
def test_validated_scales_keep_replayed_positions_distinct(override: ChannelOverride, unit: str) -> None:
    validate_profile(_profile(override))
    _, data, channels = _bound(_TORQUE_ARM, _profile(override), unit=unit)
    # Joint range -2..2 rad: the ends and the middle in public units, as the robot reports them.
    ends = []
    for qpos in (-2.0, 0.0, 2.0):
        data.qpos[0] = qpos
        ends.append(channels.read_positions()[0])

    shown = [float(_show(channels, data, value)[0]) for value in ends]

    assert len(set(shown)) == 3  # float32 observations still tell them apart
    np.testing.assert_allclose(shown, ends, rtol=1e-6, atol=1e-6 * abs(override.scale))


def test_every_frame_is_checked_before_any_joint_moves() -> None:
    xml = """<mujoco><compiler angle="radian"/><worldbody><body><joint name="a" type="slide"/>
    <geom size="0.01"/><body><joint name="b" type="slide"/><geom size="0.01"/></body></body></worldbody>
    <actuator><position name="a" joint="a" kp="10"/><position name="b" joint="b" kp="10"/></actuator></mujoco>"""
    profile = _profile(ChannelOverride("gripper", ("a", "b"), member_scales=(-1e3,)))
    validate_profile(profile)
    _, data, channels = _bound(xml, profile, unit="metres")
    edge = REPLAY_MODEL_LIMIT / 1e3

    placements = channels.replay_placements(np.asarray([[edge], [np.nextafter(edge, 0.0)]]))
    assert [member.joint for member, _ in placements] == ["a", "b"]
    channels.set_positions(np.asarray([edge]))
    np.testing.assert_allclose(data.qpos, [edge, -REPLAY_MODEL_LIMIT])
    qpos = data.qpos.copy()
    with pytest.raises(ValueError, match=r"frame 1: gripper = .* puts b beyond"):
        channels.replay_placements(np.asarray([[0.0], [np.nextafter(edge, np.inf)]]))
    with pytest.raises(ValueError, match="puts b beyond"):
        channels.set_positions(np.asarray([np.nextafter(edge, np.inf)]))
    np.testing.assert_array_equal(data.qpos, qpos)  # nothing moved, not even the first finger


@pytest.mark.parametrize(
    ("override", "message"),
    [
        (ChannelOverride("hinge", ("hinge",), scale=4e36), r"\|scale\| <= 10000"),
        (ChannelOverride("hinge", ("hinge",), scale=1e-290), r"0.001 <= \|scale\|"),
        (ChannelOverride("hinge", ("hinge",), scale=1e-3, offset=2.0), r"\|offset\| <= 1000 \* \|scale\|"),
        (ChannelOverride("hinge", ("hinge",), offset=1e30), r"\|offset\| <= 1000 \* \|scale\|"),
        (ChannelOverride("g", ("a", "b"), member_scales=(1e308,)), r"\|member scale\| <= 1000"),
        (ChannelOverride("g", ("a", "b"), member_scales=(0.0,)), r"0.001 <= \|member scale\|"),
    ],
)
def test_profiles_keep_unit_conversions_within_float32(override: ChannelOverride, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        validate_profile(_profile(override))


@pytest.mark.parametrize(
    "override",
    [
        ChannelOverride("hinge", ("hinge",), scale=1e-3, offset=1.0),  # smallest scale, largest offset for it
        ChannelOverride("hinge", ("hinge",), scale=-1e-3, offset=-1.0),
        ChannelOverride("hinge", ("hinge",), scale=1e4, offset=1e7),
        ChannelOverride("hinge", ("hinge",)),
    ],
)
def test_recorded_float32_ends_of_a_normalized_range_replay(override: ChannelOverride) -> None:
    validate_profile(_profile(override))
    _, data, channels = _bound(_TORQUE_ARM, _profile(override), unit="normalized")
    for end in (-2.0, 2.0):  # the joint range
        data.qpos[0] = end
        recorded = channels.read_positions().astype(np.float32)  # as MuJoCoRobot observes it

        shown = _show(channels, data, float(recorded[0]))

        np.testing.assert_array_equal(shown, recorded)
        assert data.qpos[0] == pytest.approx(end)
    low, high = (override.offset + override.scale * end for end in (-100.0, 100.0))
    for value in (high + 0.01 * (high - low), low - 0.01 * (high - low)):  # 1 % past either end
        with pytest.raises(ValueError, match="is outside its normalized range"):
            channels.set_positions(np.asarray([value]))


def test_channels_sharing_a_joint_must_agree() -> None:
    xml = """<mujoco><compiler angle="radian"/><worldbody><body><joint name="hinge" range="-1 1"/>
    <geom size="0.01"/></body></worldbody><actuator><position name="motor_a" joint="hinge" kp="10"/>
    <position name="motor_b" joint="hinge" kp="10"/></actuator></mujoco>"""
    _, data, channels = _bound(xml, unit="normalized")
    qpos = data.qpos.copy()

    with pytest.raises(ValueError, match="frame 0: motor_a and motor_b place joint hinge at different positions"):
        channels.set_positions(np.asarray([0.0, 100.0]))
    np.testing.assert_array_equal(data.qpos, qpos)
    channels.set_positions(np.asarray([50.0, 50.0]))  # what a recording of this robot contains
    np.testing.assert_allclose(channels.read_positions(), [50.0, 50.0])


_SHARED_HINGE = """<mujoco><compiler angle="radian"/><worldbody><body><joint name="hinge" range="{low} {high}"/>
    <geom size="0.01"/></body></worldbody><actuator><position name="motor_a" joint="hinge" kp="10"/>
    <position name="motor_b" joint="hinge" kp="10"/></actuator></mujoco>"""


@pytest.mark.parametrize("qpos", [-0.9, 0.01, 0.1, 0.2, 0.4, 0.5, 0.99])
def test_a_recording_of_channels_in_different_units_on_one_joint_replays(qpos: float) -> None:
    profile = _profile(
        ChannelOverride("norm", ("motor_a",), unit="normalized"), ChannelOverride("deg", ("motor_b",), unit="degrees")
    )
    validate_profile(profile)
    _, data, channels = _bound(_SHARED_HINGE.format(low=-1, high=1), profile)
    data.qpos[0] = qpos
    recorded = channels.read_positions().astype(np.float32)  # as MuJoCoRobot observes it
    data.qpos[0] = 0.0

    shown = _show(channels, data, recorded)

    np.testing.assert_array_equal(shown, recorded)


def test_a_small_but_visible_disagreement_on_a_shared_joint_is_refused() -> None:
    # A 0.1 mrad range: 0 and 0.001 normalized are 5e-10 rad apart, yet observably different.
    _, data, channels = _bound(_SHARED_HINGE.format(low=0.9999, high=1.0), unit="normalized")
    qpos = data.qpos.copy()

    with pytest.raises(ValueError, match="frame 0: motor_a and motor_b place joint hinge at different positions"):
        channels.set_positions(np.asarray([0.0, 0.001]))
    np.testing.assert_array_equal(data.qpos, qpos)
    channels.set_positions(np.asarray([0.001, 0.001]))
