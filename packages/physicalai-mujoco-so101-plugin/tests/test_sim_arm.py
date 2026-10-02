"""SimArm: one profile's arm, attached under a prefix, read and driven through a compiled model."""

# MuJoCo's bindings are supplied by the C extension at runtime.
# pyrefly: ignore-errors [missing-attribute]

from pathlib import Path

import mujoco
import numpy as np
import pytest

from physicalai_mujoco_so101_plugin.robot_profile import ROBOT_MOUNT_FRAME, SO101_PROFILE, load_scene_model
from physicalai_mujoco_so101_plugin.sim_arm import SimArm, radians_to_normalized


@pytest.fixture(scope="module")
def two_arm_model(tmp_path_factory: pytest.TempPathFactory) -> mujoco.MjModel:
    """A model with the SO-101 attached twice, under ``a_`` and ``b_``."""
    path = Path(tmp_path_factory.mktemp("scene")) / "scene.xml"
    path.write_text(
        f"""<mujoco><worldbody>
        <frame name="a_{ROBOT_MOUNT_FRAME}" pos="-0.3 0 0"/>
        <frame name="b_{ROBOT_MOUNT_FRAME}" pos="0.3 0 0"/>
        </worldbody></mujoco>"""
    )
    return load_scene_model(path)


def _bound(model: mujoco.MjModel, prefix: str, unit: str = "normalized") -> SimArm:
    arm = SimArm(SO101_PROFILE, prefix, unit)  # type: ignore[arg-type]
    arm.bind(arm.resolve(model))
    return arm


def test_names_follow_the_profile_and_prefix() -> None:
    arm = SimArm(SO101_PROFILE, "left_")
    assert arm.joint_names == tuple(f"left_{name}" for name in SO101_PROFILE.joint_order)
    assert arm.wrist_camera == "left_wrist"
    assert arm.grippers == (False, False, False, False, False, True)


def test_each_arm_binds_to_its_own_joints(two_arm_model: mujoco.MjModel) -> None:
    a, b = _bound(two_arm_model, "a_"), _bound(two_arm_model, "b_")
    assert set(a.ctrl_indices).isdisjoint(b.ctrl_indices)
    for arm in (a, b):
        assert [two_arm_model.actuator(i).name for i in arm.ctrl_indices] == list(arm.joint_names)
        expected = np.array([(low, high) for _, low, high in SO101_PROFILE.joint_ranges])
        np.testing.assert_array_equal(arm.joint_limits, expected)


def test_a_missing_prefix_does_not_resolve(two_arm_model: mujoco.MjModel) -> None:
    assert SimArm(SO101_PROFILE, "c_").resolve(two_arm_model) is None


def test_write_only_drives_its_own_arm(two_arm_model: mujoco.MjModel) -> None:
    data = mujoco.MjData(two_arm_model)
    a, b = _bound(two_arm_model, "a_"), _bound(two_arm_model, "b_")

    a.write(data, np.array([50.0, -50.0, 0.0, 25.0, -25.0, 80.0]))

    assert np.any(a.targets(data) != 0.0)
    np.testing.assert_array_equal(b.targets(data), 0.0)


def test_read_converts_the_arm_state(two_arm_model: mujoco.MjModel) -> None:
    data = mujoco.MjData(two_arm_model)
    arm = _bound(two_arm_model, "b_")
    radians = np.array([0.3, -0.4, 0.5, -0.6, 0.7, 0.2])
    for name, value in zip(arm.joint_names, radians, strict=True):
        data.joint(name).qpos = value

    positions, _ = arm.read(data)
    np.testing.assert_allclose(positions, radians_to_normalized(radians, arm.joint_limits, arm.grippers))

    degrees = _bound(two_arm_model, "b_", unit="degrees")
    np.testing.assert_allclose(degrees.read(data)[0], np.degrees(radians))


def test_go_home_sets_pose_and_targets(two_arm_model: mujoco.MjModel) -> None:
    data = mujoco.MjData(two_arm_model)
    arm = _bound(two_arm_model, "a_")

    arm.go_home(two_arm_model, data, {"a_elbow_flex": 0.7, "a_wrist_flex": 9.0})

    assert data.joint("a_elbow_flex").qpos[0] == pytest.approx(0.7)
    assert data.joint("a_wrist_flex").qpos[0] == pytest.approx(SO101_PROFILE.joint_ranges[3][2])  # clipped
    np.testing.assert_allclose(arm.targets(data)[2], 0.7)


def test_unbound_arm_reports_not_connected() -> None:
    arm = SimArm(SO101_PROFILE)
    assert not arm.is_bound
    assert arm.ctrl_indices == ()
    assert arm.joint_limits is None
    with pytest.raises(ConnectionError):
        arm.to_units(np.zeros(6))
