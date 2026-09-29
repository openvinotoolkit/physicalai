"""Robot profile and scene composition: robot-free scenes with SO-101 arms attached at mount frames."""

# MuJoCo's bindings are supplied by the C extension at runtime.
# pyrefly: ignore-errors [missing-attribute]

import warnings
from pathlib import Path

import mujoco
import numpy as np
import pytest

from physicalai_mujoco_so101_plugin._urdf import get_urdf_path
from physicalai_mujoco_so101_plugin.constants import BIMANUAL_SO101_JOINT_ORDER, SO101_JOINT_ORDER
from physicalai_mujoco_so101_plugin.mujoco_robot import BiMuJoCoSO101, MuJoCoSO101
from physicalai_mujoco_so101_plugin.robot_profile import (
    ROBOT_MOUNT_FRAME,
    SO101_PROFILE,
    compose_scene_spec,
    load_scene_model,
    robot_mount_prefixes,
)
from physicalai_mujoco_so101_plugin.scene_registry import get_scene, list_scenes, list_scenes_for_arms

# Joint ranges (radians) of the plugin's SO-101 model before it moved to MuJoCo Menagerie's.
# Normalized units span these ranges, so trained policies depend on them staying put.
EARLIER_SO101_JOINT_RANGES = np.array([
    (-1.9198621771937616, 1.9198621771937634),
    (-1.7453292519943224, 1.7453292519943366),
    (-1.69, 1.69),
    (-1.6580628494556928, 1.6580627293335335),
    (-2.7438472969992493, 2.841206309382605),
    (-0.17453297762778586, 1.7453291995659765),
])


def test_profile_points_at_the_vendored_menagerie_model() -> None:
    assert SO101_PROFILE.mjcf_path == get_urdf_path() / "robots/so101/so101.xml"
    assert SO101_PROFILE.mjcf_path.is_file()
    assert (SO101_PROFILE.mjcf_path.parent / "LICENSE").is_file()
    assert SO101_PROFILE.joint_order == SO101_JOINT_ORDER
    assert tuple(name for name, _, _ in SO101_PROFILE.joint_ranges) == SO101_JOINT_ORDER


def test_robot_spec_keeps_the_public_names_and_units() -> None:
    model = SO101_PROFILE.load_spec().compile()

    assert tuple(model.joint(i).name for i in range(model.njnt)) == SO101_JOINT_ORDER
    assert tuple(model.actuator(i).name for i in range(model.nu)) == SO101_JOINT_ORDER
    assert [model.camera(i).name for i in range(model.ncam)] == [SO101_PROFILE.wrist_camera]
    np.testing.assert_array_equal(model.jnt_range, EARLIER_SO101_JOINT_RANGES)
    np.testing.assert_array_equal(model.actuator_forcerange, np.tile([-3.35, 3.35], (6, 1)))


def test_robot_spec_drops_menagerie_collision_meshes() -> None:
    """Grasp contacts come from the same primitives as before; no mesh geom collides."""
    model = SO101_PROFILE.load_spec().compile()
    for geom in range(model.ngeom):
        if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH:
            assert model.geom_contype[geom] == 0, model.geom(geom).name
            assert model.geom_conaffinity[geom] == 0, model.geom(geom).name


@pytest.mark.parametrize("scene_id", list(list_scenes()))
def test_every_scene_mounts_one_arm_per_declared_arm(scene_id: str) -> None:
    scene = get_scene(scene_id)
    prefixes = robot_mount_prefixes(mujoco.MjSpec.from_file(str(scene.scene_xml_path)))
    assert prefixes == (("left_", "right_") if scene.num_arms == 2 else ("",))


@pytest.mark.parametrize("scene_id", list(list_scenes()))
def test_scene_files_hold_no_robot(scene_id: str) -> None:
    model = mujoco.MjModel.from_xml_path(str(get_scene(scene_id).scene_xml_path))
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base") < 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_base") < 0


def test_attaching_the_arm_raises_no_warnings() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        get_scene("single_pick_place").load_model()


@pytest.mark.parametrize("scene_id", list(list_scenes_for_arms(1)))
def test_single_arm_scenes_attach_the_arm_at_the_world_origin(scene_id: str) -> None:
    model = get_scene(scene_id).load_model()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    for name in SO101_JOINT_ORDER:
        assert model.joint(name).name == name
        assert model.actuator(name).trnid[0] == model.joint(name).id
    np.testing.assert_allclose(data.xpos[model.body("base").id], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(data.xquat[model.body("base").id], [1.0, 0.0, 0.0, 0.0])
    assert model.opt.timestep == pytest.approx(0.002)


def test_bimanual_scene_prefixes_every_arm_name() -> None:
    model = get_scene("garment_fold").load_model()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    arm_joints = [model.joint(i).name for i in range(model.njnt) if model.joint(i).name in BIMANUAL_SO101_JOINT_ORDER]
    assert tuple(arm_joints) == BIMANUAL_SO101_JOINT_ORDER
    assert tuple(model.actuator(i).name for i in range(model.nu)) == BIMANUAL_SO101_JOINT_ORDER
    assert {model.camera(i).name for i in range(model.ncam)} == {"overview", "left_wrist", "right_wrist"}
    for prefix in ("left_", "right_"):
        for name in ("gripper", "camera_mount", "moving_jaw_so101_v1"):
            assert model.body(f"{prefix}{name}").id > 0
        assert model.site(f"{prefix}gripperframe").id >= 0
    np.testing.assert_allclose(data.xpos[model.body("left_base").id], [-0.23, -0.25, 0.40])
    np.testing.assert_allclose(data.xquat[model.body("left_base").id], [1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(data.xpos[model.body("right_base").id], [0.23, -0.25, 0.40])
    np.testing.assert_allclose(data.xquat[model.body("right_base").id], [0.0, 0.0, 0.0, 1.0])


@pytest.mark.parametrize(
    ("robot_cls", "scene_id", "joint_ranges"),
    [
        (MuJoCoSO101, "single_pick_place", EARLIER_SO101_JOINT_RANGES),
        (MuJoCoSO101, "conveyor_sort", EARLIER_SO101_JOINT_RANGES),
        (BiMuJoCoSO101, "garment_fold", np.vstack([EARLIER_SO101_JOINT_RANGES] * 2)),
    ],
)
def test_normalized_units_span_the_earlier_joint_ranges(
    robot_cls: type[MuJoCoSO101], scene_id: str, joint_ranges: np.ndarray
) -> None:
    robot = robot_cls(model_path=str(get_scene(scene_id).scene_xml_path))
    robot.connect()
    try:
        np.testing.assert_array_equal(robot._joint_limits, joint_ranges)
    finally:
        robot.disconnect()


def test_xml_without_mount_frames_compiles_unchanged(tmp_path: Path) -> None:
    xml = """<mujoco><worldbody><body name="arm"><joint name="hinge"/><geom size="0.1"/></body></worldbody></mujoco>"""
    path = tmp_path / "custom.xml"
    path.write_text(xml)
    assert robot_mount_prefixes(mujoco.MjSpec.from_file(str(path))) == ()
    model = load_scene_model(path)
    assert (model.nbody, model.njnt, model.nu) == (2, 1, 0)


def test_mount_frame_prefix_names_the_attached_arm(tmp_path: Path) -> None:
    xml = f"""<mujoco><worldbody><frame name="demo_{ROBOT_MOUNT_FRAME}" pos="0 0 1"/></worldbody></mujoco>"""
    path = tmp_path / "scene.xml"
    path.write_text(xml)
    model = compose_scene_spec(path).compile()
    assert tuple(model.actuator(i).name for i in range(model.nu)) == tuple(f"demo_{n}" for n in SO101_JOINT_ORDER)
    assert model.camera("demo_wrist").id >= 0
