"""Scene composition: robot-free scenes with the profile's robot attached at mount frames."""

# MuJoCo's bindings are supplied by the C extension at runtime.
# pyrefly: ignore-errors [missing-attribute]

import dataclasses
import warnings
from pathlib import Path

import mujoco
import mujoco_menagerie
import numpy as np
import pytest

from physicalai_mujoco_plugin._urdf import get_urdf_path
from physicalai_mujoco_plugin.constants import BIMANUAL_SO101_JOINT_ORDER, SO101_JOINT_ORDER
from physicalai_mujoco_plugin.robot import MuJoCoRobot
from physicalai_mujoco_plugin.compose import (
    ROBOT_MOUNT_FRAME,
    anchor_prefixes,
    compose_scene,
    compose_scene_spec,
    fetch_profile,
    load_robot_spec,
    load_scene_model,
    robot_layout,
    scene_anchors,
    scene_needs_robot,
)
from physicalai_mujoco_plugin.profiles import SO101_PROFILE, RobotProfile, get_profile
from physicalai_mujoco_plugin.scene_registry import (
    arm_prefixes,
    get_scene,
    list_scenes,
    list_scenes_for,
    list_scenes_for_arms,
    supported_arm_counts,
)
from physicalai_mujoco_plugin.sim import place_home, robot_roots


def _scene_profile(scene_id: str) -> RobotProfile:
    """The SO-101 for tabletop scenes, the Go2 for floor scenes (spawn anchors)."""
    return SO101_PROFILE if scene_id in list_scenes_for(SO101_PROFILE) else get_profile("unitree_go2")

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


def test_profile_names_the_menagerie_so101() -> None:
    robot = mujoco_menagerie.get(SO101_PROFILE.menagerie_model)
    assert SO101_PROFILE.menagerie_entry in robot.entry_names
    assert robot.license == "Apache-2.0"
    assert tuple(channel.name for channel in SO101_PROFILE.channels) == SO101_JOINT_ORDER


def test_robot_spec_keeps_the_public_names_and_units() -> None:
    model = load_robot_spec(SO101_PROFILE).compile()

    assert tuple(model.joint(i).name for i in range(model.njnt)) == SO101_JOINT_ORDER
    assert tuple(model.actuator(i).name for i in range(model.nu)) == SO101_JOINT_ORDER
    assert [model.camera(i).name for i in range(model.ncam)] == ["wrist"]
    np.testing.assert_array_equal(model.jnt_range, EARLIER_SO101_JOINT_RANGES)
    np.testing.assert_array_equal(model.actuator_forcerange, np.tile([-3.35, 3.35], (6, 1)))


def test_robot_spec_collides_through_the_earlier_primitives_only() -> None:
    """Grasp contacts come from the same 26 primitives as before; visual geoms never collide."""
    model = load_robot_spec(SO101_PROFILE).compile()
    collidable = [g for g in range(model.ngeom) if model.geom_contype[g] or model.geom_conaffinity[g]]

    assert len(collidable) == 26
    for geom in range(model.ngeom):
        if geom in collidable:
            assert model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH, model.geom(geom).name
            assert model.geom_group[geom] == 3, model.geom(geom).name
        else:
            assert model.geom_group[geom] == 2, model.geom(geom).name


def test_bundled_stl_meshes_are_not_git_lfs_pointers() -> None:
    """The URDF meshes for Studio's 3D view: a checkout or build without ``git lfs pull`` must fail here."""
    meshes = sorted(get_urdf_path().rglob("*.stl"))
    assert meshes
    for mesh in meshes:
        head = mesh.read_bytes()[:64]
        assert not head.startswith(b"version https://git-lfs"), mesh
        assert mesh.stat().st_size > 1024, mesh


@pytest.mark.parametrize("scene_id", list(list_scenes()))
def test_composed_scenes_have_no_keyframes(scene_id: str) -> None:
    """Keyframes store raw qpos; the arm's position in qpos depends on where it is attached."""
    assert get_scene(scene_id).load_model(_scene_profile(scene_id)).nkey == 0


@pytest.mark.parametrize("scene_id", list(list_scenes()))
def test_every_scene_mounts_one_arm_per_declared_arm(scene_id: str) -> None:
    scene = get_scene(scene_id)
    prefixes = anchor_prefixes(mujoco.MjSpec.from_file(str(scene.scene_xml_path)))
    assert prefixes == arm_prefixes(scene.written_arms)


@pytest.mark.parametrize("scene_id", list(list_scenes()))
def test_declared_anchors_match_the_scene_xml(scene_id: str) -> None:
    """SCN-4: ``SceneConfig.anchors`` and ``written_arms`` agree with the frames in the scene file."""
    scene = get_scene(scene_id)
    anchors = scene_anchors(mujoco.MjSpec.from_file(str(scene.scene_xml_path)))
    assert {kind for _, kind in anchors} == {scene.anchors}
    assert len(anchors) == scene.written_arms
    assert scene.written_arms in scene.arm_counts


@pytest.mark.parametrize("scene_id", list(list_scenes()))
def test_every_arm_count_has_a_layout(scene_id: str) -> None:
    """P1: an arm count the file does not lay out is pinned, or derived from a single mount at the origin facing +x."""
    scene = get_scene(scene_id)
    spec = mujoco.MjSpec.from_file(str(scene.scene_xml_path))  # keeps the frames alive
    frames = [frame for frame in spec.frames if frame.name.endswith(ROBOT_MOUNT_FRAME)]
    for arms in scene.arm_counts:
        if arms == scene.written_arms or arms in dict(scene.mounts):
            continue
        assert (arms, len(frames)) == (2, 1), f"{scene_id} cannot derive {arms} arms"
        np.testing.assert_array_equal(frames[0].pos, [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(frames[0].quat, [1.0, 0.0, 0.0, 0.0])
        assert frames[0].alt.type == mujoco.mjtOrientation.mjORIENTATION_QUAT
        assert frames[0].parent.name == "world"
    for arms, poses in scene.mounts:
        assert arms != scene.written_arms and len(poses) == arms


@pytest.mark.parametrize(("scene_id", "arms"), [(scene_id, arms) for scene_id in list_scenes_for(SO101_PROFILE) for arms in (1, 2)])
def test_scenes_attach_the_arms_at_their_layout(scene_id: str, arms: int) -> None:
    """P1: each arm's base sits where the layout puts its mount, with the prefixes of its arm count."""
    scene = get_scene(scene_id)
    model = scene.load_model(SO101_PROFILE, arms)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    layout = scene.layout_for(SO101_PROFILE, arms)
    if layout is None:  # the file's own mount frames
        spec = mujoco.MjSpec.from_file(str(scene.scene_xml_path))  # keeps the frames alive
        frames = {frame.name: frame for frame in spec.frames}
        poses = [(prefix, frames[f"{prefix}{ROBOT_MOUNT_FRAME}"]) for prefix in arm_prefixes(arms)]
        expected = [(prefix, frame.pos, frame.quat) for prefix, frame in poses]
    else:
        expected = [(prefix, pose.pos, pose.quat) for prefix, pose in layout.mounts]
    assert [prefix for prefix, _, _ in expected] == list(arm_prefixes(arms))
    for prefix, pos, quat in expected:
        np.testing.assert_allclose(data.xpos[model.body(f"{prefix}base").id], pos, atol=1e-12)
        np.testing.assert_allclose(data.xquat[model.body(f"{prefix}base").id], quat, atol=1e-12)
    names = {model.actuator(i).name for i in range(model.nu)} & set(BIMANUAL_SO101_JOINT_ORDER + SO101_JOINT_ORDER)
    assert names == set(BIMANUAL_SO101_JOINT_ORDER if arms == 2 else SO101_JOINT_ORDER)


def test_derived_two_arm_layouts() -> None:
    """P1: side by side at the mount's edge, or across the spawn centre; separation and centre scale with reach."""
    scene = get_scene("single_pick_place")
    trossen = get_profile("trossen_wxai")
    scale = scene.layout_scale(trossen)

    side = dict(scene.mount_poses(trossen, 2))
    assert side["left_"].pos == pytest.approx((0.0, 0.1 * scale, 0.0))
    assert side["right_"].pos == pytest.approx((0.0, -0.1 * scale, 0.0))
    assert side["left_"].quat == side["right_"].quat == (1.0, 0.0, 0.0, 0.0)
    assert scene.for_profile(trossen, 2).spawn_center == pytest.approx((0.2 * scale, 0.0))

    across = dict(dataclasses.replace(scene, two_arm_style="across").mount_poses(trossen, 2))
    assert across["left_"].pos == pytest.approx((0.2 * scale, 0.1 * scale, 0.0))
    assert across["right_"].pos == pytest.approx((0.2 * scale, -0.1 * scale, 0.0))
    np.testing.assert_allclose(across["left_"].quat, [0.5**0.5, 0.0, 0.0, -(0.5**0.5)])  # faces -y
    np.testing.assert_allclose(across["right_"].quat, [0.5**0.5, 0.0, 0.0, 0.5**0.5])  # faces +y

    assert dict(scene.mount_poses(get_profile("ur5e"), 2))["left_"].pos == pytest.approx((0.0, 0.325, 0.0))
    with pytest.raises(ValueError, match="no layout for 3"):
        scene.mount_poses(SO101_PROFILE, 3)


def test_a_layout_for_one_arm_unnames_the_files_second_mount() -> None:
    """MjSpec cannot delete a frame: garment_fold's one-arm layout renames the left mount and unnames the right."""
    scene = get_scene("garment_fold")
    spec = compose_scene_spec(scene.scene_xml_path, SO101_PROFILE, scene_layout=scene.layout_for(SO101_PROFILE, 1))
    assert anchor_prefixes(spec) == ("",)
    assert sum(1 for frame in spec.frames if frame.name == "") >= 1


@pytest.mark.parametrize("scene_id", list(list_scenes()))
def test_scene_files_hold_no_robot(scene_id: str) -> None:
    model = mujoco.MjModel.from_xml_path(str(get_scene(scene_id).scene_xml_path))
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base") < 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_base") < 0


def test_attaching_the_arm_raises_no_warnings() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        get_scene("single_pick_place").load_model()


@pytest.mark.parametrize("scene_id", list(list_scenes()))
def test_scene_options_win_over_the_robot_options(scene_id: str) -> None:
    scene = get_scene(scene_id)
    scene_option = mujoco.MjSpec.from_file(str(scene.scene_xml_path)).option
    opt = scene.load_model(_scene_profile(scene_id)).opt
    assert opt.timestep == scene_option.timestep
    assert opt.iterations == scene_option.iterations
    assert opt.solver == scene_option.solver
    np.testing.assert_array_equal(opt.gravity, scene_option.gravity)


@pytest.mark.parametrize("scene_id", [scene_id for scene_id in list_scenes_for_arms(1) if scene_id != "garment_fold"])
def test_single_arm_scenes_attach_the_arm_at_the_world_origin(scene_id: str) -> None:
    model = get_scene(scene_id).load_model(arms=1)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    for name in SO101_JOINT_ORDER:
        assert model.joint(name).name == name
        assert model.actuator(name).trnid[0] == model.joint(name).id
    np.testing.assert_allclose(data.xpos[model.body("base").id], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(data.xquat[model.body("base").id], [1.0, 0.0, 0.0, 0.0])
    assert model.opt.timestep == pytest.approx(0.002)


def test_bimanual_scene_prefixes_every_arm_name() -> None:
    model = get_scene("garment_fold").load_model(arms=2)
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


def test_bimanual_arms_get_their_own_default_classes() -> None:
    spec = compose_scene_spec(get_scene("garment_fold").scene_xml_path)
    for prefix in ("left_", "right_"):
        for name in ("so101", "sts3215", "visual", "collision", "collision_gripper"):
            assert spec.find_default(f"{prefix}{name}") is not None, f"{prefix}{name}"
        assert spec.joint(f"{prefix}wrist_roll").classname.name == f"{prefix}sts3215"
        assert spec.geom(f"{prefix}fixed_jaw_box1").classname.name == f"{prefix}collision_gripper"


@pytest.mark.parametrize(
    ("scene_id", "joint_ranges"),
    [
        ("single_pick_place", EARLIER_SO101_JOINT_RANGES),
        ("conveyor_sort", EARLIER_SO101_JOINT_RANGES),
        ("garment_fold", np.vstack([EARLIER_SO101_JOINT_RANGES] * 2)),
    ],
)
def test_normalized_units_span_the_earlier_joint_ranges(
    scene_id: str, joint_ranges: np.ndarray
) -> None:
    robot = MuJoCoRobot(scene=scene_id, bimanual=len(joint_ranges) == 12, cameras=[])
    robot.connect()
    try:
        ranges = [channel.range for channels in robot._sim.channels for channel in channels.channels]
        np.testing.assert_array_equal(ranges, joint_ranges)
        model = robot._model
        np.testing.assert_array_equal([model.joint(name).range for name in robot.joint_names], joint_ranges)
    finally:
        robot.disconnect()


def test_xml_without_mount_frames_compiles_unchanged(tmp_path: Path) -> None:
    xml = """<mujoco><worldbody><body name="arm"><joint name="hinge"/><geom size="0.1"/></body></worldbody></mujoco>"""
    path = tmp_path / "custom.xml"
    path.write_text(xml)
    assert anchor_prefixes(mujoco.MjSpec.from_file(str(path))) == ()
    model = load_scene_model(path)
    assert (model.nbody, model.njnt, model.nu) == (2, 1, 0)


def test_mount_frame_prefix_names_the_attached_arm(tmp_path: Path) -> None:
    xml = f"""<mujoco><worldbody><frame name="demo_{ROBOT_MOUNT_FRAME}" pos="0 0 1"/></worldbody></mujoco>"""
    path = tmp_path / "scene.xml"
    path.write_text(xml)
    model = compose_scene_spec(path).compile()
    assert tuple(model.actuator(i).name for i in range(model.nu)) == tuple(f"demo_{n}" for n in SO101_JOINT_ORDER)
    assert model.camera("demo_wrist").id >= 0


def test_missing_download_explains_how_to_get_the_model(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*_args: object) -> None:
        raise mujoco_menagerie.DownloadError("offline")

    monkeypatch.setattr(mujoco_menagerie.Robot, "path", fail)
    with pytest.raises(RuntimeError, match=r"physicalai-mujoco prefetch.*MENAGERIE_ROOT"):
        fetch_profile(SO101_PROFILE)
    with pytest.raises(RuntimeError, match="robotstudio_so101"):
        load_robot_spec(SO101_PROFILE)


def test_missing_menagerie_root_model_is_reported(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MENAGERIE_ROOT", str(tmp_path))
    with pytest.raises(RuntimeError, match="not found under MENAGERIE_ROOT"):
        fetch_profile(SO101_PROFILE)


def test_fetch_returns_the_entry_file() -> None:
    entry = mujoco_menagerie.get(SO101_PROFILE.menagerie_model).entry(SO101_PROFILE.menagerie_entry)
    path = fetch_profile(SO101_PROFILE)
    assert path.is_file()
    assert path.name == entry.file.split("/")[-1]


@pytest.mark.parametrize("scene_id", sorted(list_scenes()))
def test_every_scene_needs_the_robot(scene_id: str) -> None:
    assert scene_needs_robot(get_scene(scene_id).scene_xml_path)


def test_xml_without_mount_frames_needs_no_robot(tmp_path: Path) -> None:
    path = tmp_path / "custom.xml"
    path.write_text("<mujoco><worldbody/></mujoco>")
    assert not scene_needs_robot(path)


def test_anchors_are_mount_or_spawn_frames() -> None:
    xml = """<mujoco><worldbody><frame name="left_robot_mount"/><frame name="robot_spawn"/>
      <frame name="right_robot_spawn"/><frame name="spawn_marker"/></worldbody></mujoco>"""
    spec = mujoco.MjSpec.from_string(xml)
    assert scene_anchors(spec) == (("left_", "mount"), ("", "spawn"), ("right_", "spawn"))
    assert anchor_prefixes(spec) == ("left_", "", "right_")


def test_one_prefix_cannot_be_mounted_and_spawned() -> None:
    xml = '<mujoco><worldbody><frame name="robot_mount"/><frame name="robot_spawn"/></worldbody></mujoco>'
    with pytest.raises(ValueError, match="both robot_mount and robot_spawn"):
        scene_anchors(mujoco.MjSpec.from_string(xml))


def test_a_fixed_base_robot_cannot_spawn() -> None:
    with pytest.raises(ValueError, match="fixed base"):
        compose_scene(get_scene("floor_flat").scene_xml_path, SO101_PROFILE)


def test_composed_layout_is_bound_to_each_anchor() -> None:
    composed = compose_scene(get_scene("garment_fold").scene_xml_path, SO101_PROFILE)
    model = composed.model

    assert [binding.prefix for binding in composed.robots] == ["left_", "right_"]
    for binding in composed.robots:
        layout = binding.layout
        assert layout.joint_names == tuple(f"{binding.prefix}{name}" for name in SO101_JOINT_ORDER)
        for channel in layout.channels:
            assert model.actuator(channel.actuator_id).name == channel.actuator
            assert model.joint(channel.joint_id).name == channel.joint
            assert model.jnt_qposadr[channel.joint_id] == channel.qpos_adr
        assert [camera.name for camera in layout.cameras] == [f"{binding.prefix}wrist"]
        assert set(layout.home_qpos) == {f"{binding.prefix}{name}" for name in SO101_JOINT_ORDER}


def test_a_joint_and_its_differently_named_actuator_are_one_robot() -> None:
    from physicalai_mujoco_plugin.compose import model_prefixes
    from physicalai_mujoco_plugin.profiles.derive import derive_profile

    xml = "<mujoco><worldbody><body><joint name='shoulder_pan'/><geom size='0.1'/></body></worldbody>"
    xml += "<actuator><position name='motor_shoulder_pan' joint='shoulder_pan'/></actuator></mujoco>"
    layout = derive_profile(mujoco.MjModel.from_xml_string(xml))
    assert model_prefixes(layout, SO101_PROFILE) == ("",)


def test_robot_layout_is_cached_per_profile() -> None:
    assert robot_layout(SO101_PROFILE) is robot_layout(SO101_PROFILE)


def test_floating_base_robots_are_not_attached_at_mount_frames(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from physicalai_mujoco_plugin import compose
    from physicalai_mujoco_plugin.profiles import RobotProfile

    robot_xml = """<mujoco><worldbody><body name="base"><freejoint name="root"/><geom size="0.1"/>
    <body name="leg"><joint name="hinge"/><geom size="0.05"/></body></body></worldbody>
    <actuator><position name="hinge" joint="hinge" kp="10"/></actuator></mujoco>"""
    monkeypatch.setattr(compose, "_customized_robot_spec", lambda _profile: mujoco.MjSpec.from_string(robot_xml))
    monkeypatch.setattr(compose, "_LAYOUTS", {})
    path = tmp_path / "table.xml"
    path.write_text(f'<mujoco><worldbody><frame name="{ROBOT_MOUNT_FRAME}"/></worldbody></mujoco>')
    profile = RobotProfile(name="floating", display_name="Floating", menagerie_model="inline")

    with pytest.raises(ValueError, match=r"'floating' has a floating base \('root'\).*robot_spawn"):
        compose_scene_spec(path, profile)


def test_reach_layout_moves_the_target_and_the_overview_rig_with_the_arm() -> None:
    """SCN-6, CAM-4: the SO-101 compiles the scene as written; a longer arm gets it scaled about the mount."""
    scene = get_scene("single_pick_place")
    xml = mujoco.MjSpec.from_file(str(scene.scene_xml_path))
    ur5e = get_profile("ur5e")
    scale = scene.layout_scale(ur5e)

    so101 = scene.load_model(SO101_PROFILE)
    longer = scene.load_model(dataclasses.replace(ur5e, name="ur5e-without-a-rig-override"))

    for body in ("target", "overview_camera_rig", "block1"):
        written = np.asarray(xml.body(body).pos)
        np.testing.assert_array_equal(so101.body(body).pos, written)
        np.testing.assert_allclose(longer.body(body).pos, written * scale)
    assert scale > 2.0  # noqa: PLR2004


OVERVIEW_MARGIN_PX = 8
"""Closest a framed point may come to the edge of the 640x480 overview."""


def _robot_geometry_points(model: mujoco.MjModel, data: mujoco.MjData, bodies: set[int]) -> np.ndarray:
    """Return the world points that bound the visible geoms of *bodies*: mesh vertices, else box corners."""
    corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=float)
    points = []
    for geom in range(model.ngeom):
        if model.geom_bodyid[geom] not in bodies or model.geom_rgba[geom][3] == 0:
            continue
        rotation = data.geom_xmat[geom].reshape(3, 3)
        if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = model.geom_dataid[geom]
            start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            local = model.mesh_vert[start : start + count]
        else:
            local = model.geom_aabb[geom][:3] + corners * model.geom_aabb[geom][3:]
        points.append(local @ rotation.T + data.geom_xpos[geom])
    return np.concatenate(points)


@pytest.mark.parametrize(
    ("name", "arms"),
    [
        (name, arms)
        for arms in (1, 2)
        for name in get_scene("single_pick_place").robots
        if arms in supported_arm_counts(get_profile(name))
    ],
)
def test_overview_frames_the_arm_at_home_the_spawn_arc_and_the_target(name: str, arms: int) -> None:
    """CAM-4: in the reach-scaled single_pick_place, the 640x480 overview shows every arm and the work area.

    Projects the arms' visible geometry at home (mesh vertices, primitive bounding boxes), the spawn
    arc and the target through the camera's intrinsics; nothing is rendered.
    """
    scene = get_scene("single_pick_place")
    profile = get_profile(name)
    composed = compose_scene(scene.scene_xml_path, profile, scene_layout=scene.layout_for(profile, arms))
    assert len(composed.robots) == arms
    model, data = composed.model, mujoco.MjData(composed.model)
    for binding in composed.robots:
        place_home(model, data, binding)
    mujoco.mj_forward(model, data)
    scene_bodies = {body.name for body in mujoco.MjSpec.from_file(str(scene.scene_xml_path)).bodies}
    robot = {i for i in range(1, model.nbody) if model.body(i).name not in scene_bodies}
    laid_out = scene.for_profile(profile, arms)
    half = np.radians(laid_out.spawn_angle_half_deg)
    arc = [
        np.array([*(np.asarray(laid_out.spawn_center) + r * np.array([np.cos(a), np.sin(a)])), 0.02])
        for r in (laid_out.spawn_min_r, laid_out.spawn_max_r)
        for a in (-half, 0.0, half)
    ]
    points = np.vstack([_robot_geometry_points(model, data, robot), *arc, data.body("target").xpos])
    camera = model.camera("overview").id
    focal = 240 / np.tan(np.radians(model.cam_fovy[camera]) / 2)

    local = (points - data.cam_xpos[camera]) @ data.cam_xmat[camera].reshape(3, 3)
    assert np.all(local[:, 2] < 0), f"{name}: part of the arm or the work area is behind the overview camera"
    u, v = 320 + focal * local[:, 0] / -local[:, 2], 240 - focal * local[:, 1] / -local[:, 2]
    margin = OVERVIEW_MARGIN_PX
    assert margin <= u.min() and u.max() <= 640 - margin, f"{name}: x spans {u.min():.0f}..{u.max():.0f} px"
    assert margin <= v.min() and v.max() <= 480 - margin, f"{name}: y spans {v.min():.0f}..{v.max():.0f} px"


def test_robot_roots_lists_every_arm_tree_of_one_robot() -> None:
    """ALOHA's two arms hang from the world separately in one model: spawn resets avoid both."""
    scene = get_scene("single_pick_place")
    profile = get_profile("aloha")
    composed = compose_scene(scene.scene_xml_path, profile, scene_layout=scene.layout_for(profile))

    roots = robot_roots(composed.model, composed.robots)

    assert [composed.model.body(root).name for root in roots] == ["left/base_link", "right/base_link"]


@pytest.mark.parametrize("name", ["so101", "franka_fr3", "ur5e", "aloha"])
def test_a_live_xml_edit_keeps_the_reach_layout_of_the_overview_rig(name: str) -> None:
    """The scene watcher re-reads the unscaled XML; it scales and overrides the rig as compose did."""
    from physicalai_mujoco_plugin.scene_watch import SceneXmlWatcher  # noqa: PLC0415

    scene = get_scene("single_pick_place")
    profile = get_profile(name)
    layout = scene.layout_for(profile)
    model = compose_scene(scene.scene_xml_path, profile, scene_layout=layout).model
    data = mujoco.MjData(model)
    rig = model.body("overview_camera_rig").id
    composed = model.body_pos[rig].copy(), model.body_quat[rig].copy()

    SceneXmlWatcher(scene.scene_xml_path, layout).apply(model, data)

    np.testing.assert_allclose(model.body_pos[rig], composed[0], atol=1e-12)
    np.testing.assert_allclose(model.body_quat[rig], composed[1], atol=1e-12)
