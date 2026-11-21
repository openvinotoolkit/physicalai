from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pytest

from physicalai_rebot_b601_plugin import get_urdf_path
from physicalai_rebot_b601_plugin.constants import (
    REBOT_B601_DM_GRAVITY_URDF,
    REBOT_B601_RS_GRAVITY_URDF,
    REBOT_B601_URDF_ARM_JOINTS,
)
from physicalai_rebot_b601_plugin.gravity import STANDARD_GRAVITY, GravityModel

if TYPE_CHECKING:
    from pathlib import Path

_TWO_LINK = """<?xml version="1.0"?>
<robot name="two_link">
  <link name="base"/>
  <link name="upper">
    <inertial><origin xyz="0.2 0 0"/><mass value="2.0"/><inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>
  <link name="lower">
    <inertial><origin xyz="0.15 0 0"/><mass value="1.0"/><inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>
  <link name="tool">
    <inertial><origin xyz="0 0 0"/><mass value="0.5"/><inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>
  <joint name="shoulder" type="revolute">
    <parent link="base"/><child link="upper"/><origin xyz="0 0 0.1" rpy="1.5707963267948966 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-3" upper="3" effort="1" velocity="1"/>
  </joint>
  <joint name="elbow" type="revolute">
    <parent link="upper"/><child link="lower"/><origin xyz="0.4 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-3" upper="3" effort="1" velocity="1"/>
  </joint>
  <joint name="mount" type="fixed">
    <parent link="lower"/><child link="tool"/><origin xyz="0.3 0 0"/>
  </joint>
</robot>
"""


def _two_link_urdf(tmp_path: Path) -> Path:
    path = tmp_path / "two_link.urdf"
    path.write_text(_TWO_LINK)
    return path


@pytest.mark.parametrize(("shoulder", "elbow"), [(0.0, 0.0), (0.7, -1.2), (-0.4, 2.0)])
def test_matches_the_planar_two_link_formula(tmp_path: Path, shoulder: float, elbow: float) -> None:
    # The shoulder frame is rolled +90 degrees, so both joints turn in the vertical x-z plane and
    # positive angles lift the links: holding torque = g * sum(m * horizontal lever arm).
    model = GravityModel(_two_link_urdf(tmp_path), ["shoulder", "elbow"])
    g = STANDARD_GRAVITY
    c1, c12 = math.cos(shoulder), math.cos(shoulder + elbow)
    elbow_torque = g * (1.0 * 0.15 * c12 + 0.5 * 0.3 * c12)
    shoulder_torque = g * (2.0 * 0.2 * c1 + 1.0 * (0.4 * c1 + 0.15 * c12) + 0.5 * (0.4 * c1 + 0.3 * c12))

    np.testing.assert_allclose(model.torques([shoulder, elbow]), [shoulder_torque, elbow_torque], atol=1e-9)
    assert model.total_mass == pytest.approx(3.5)


def test_rejects_unknown_or_fixed_joints(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no revolute joint named 'wrist'"):
        GravityModel(_two_link_urdf(tmp_path), ["shoulder", "wrist"])
    with pytest.raises(ValueError, match="no revolute joint named 'mount'"):
        GravityModel(_two_link_urdf(tmp_path), ["mount"])


def test_rejects_wrong_joint_count(tmp_path: Path) -> None:
    model = GravityModel(_two_link_urdf(tmp_path), ["shoulder", "elbow"])

    with pytest.raises(ValueError, match="Expected 2 joint positions"):
        model.torques([0.0])


# Reference torques from MuJoCo's inverse dynamics (``qfrc_bias`` at rest) on the same URDFs,
# with g = 9.80665 m/s^2. Poses are public-frame radians: the folded rest (zero) pose, Menagerie's
# "raised" keyframe, and a forward reach.
_POSES = {
    "rest": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "raised": [0.0, -0.7, -1.1, 0.0, 0.0, 0.0],
    "reach": [0.3, -1.5, -0.5, 0.4, 0.2, 0.1],
}
_MUJOCO_REFERENCE = {
    ("dm", "rest"): [0.0, -1.283, -7.181, -1.98, 0.0, 0.0],
    ("dm", "raised"): [0.0, -0.462, -5.986, -1.732, 0.0, 0.0],
    ("dm", "reach"): [0.0, 3.975, -4.534, -0.566, 0.139, 0.0],
    ("rs", "rest"): [0.0, -1.544, -6.761, -2.0, 0.0, 0.001],
    ("rs", "raised"): [0.0, -0.928, -5.391, -1.757, 0.0, 0.001],
    ("rs", "reach"): [0.0, 4.213, -4.749, -0.553, 0.165, 0.0],
}


@pytest.mark.parametrize(("variant", "pose"), sorted(_MUJOCO_REFERENCE))
def test_bundled_urdfs_match_mujoco(variant: str, pose: str) -> None:
    urdf = REBOT_B601_DM_GRAVITY_URDF if variant == "dm" else REBOT_B601_RS_GRAVITY_URDF
    model = GravityModel(get_urdf_path() / urdf, REBOT_B601_URDF_ARM_JOINTS)

    np.testing.assert_allclose(model.torques(_POSES[pose]), _MUJOCO_REFERENCE[variant, pose], atol=0.01)


def test_rs_public_and_motor_frame_urdfs_agree() -> None:
    # The preview URDF reverses every arm axis of the motor-frame URDF: q_public = -q_motor, so the
    # holding torques are mirrored too. This pins the sign convention the driver relies on.
    urdf_dir = get_urdf_path() / "rebot-b601-rs" / "urdf"
    public = GravityModel(urdf_dir / "00-arm-rs_asm-v3_joint_frame.urdf", REBOT_B601_URDF_ARM_JOINTS)
    motor = GravityModel(urdf_dir / "00-arm-rs_asm-v3.urdf", REBOT_B601_URDF_ARM_JOINTS)
    rng = np.random.default_rng(0)
    for q in rng.uniform(-2.0, 0.0, size=(20, 6)):
        np.testing.assert_allclose(public.torques(q), -motor.torques(-q), atol=1e-9)
