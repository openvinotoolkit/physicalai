from __future__ import annotations

import dataclasses
from unittest.mock import MagicMock, patch

import mujoco
import numpy as np
import pytest

from physicalai_mujoco_plugin import scene_registry
from physicalai_mujoco_plugin.compose import OverviewRig, SceneLayout, apply_layout
from physicalai_mujoco_plugin.profiles import SO101_PROFILE, get_profile
from physicalai_mujoco_plugin.scene_registry import (
    SceneConfig,
    get_reset_fn,
    get_scene,
    list_scenes,
    list_scenes_for_arms,
)

# A registry entry used only to exercise the multi-object spawn reset.
_MULTI_BLOCK_SCENE = SceneConfig(
    scene_id="multi_block_test",
    display_name="Multi-block test",
    description="Three blocks and a target disc",
    scene_xml_relpath="unused.xml",
    free_joints=("block1:joint", "block2:joint", "block3:joint"),
    target_bodies=("target",),
    spawn_center=(0.24, 0.0),
    spawn_min_r=0.08,
    spawn_max_r=0.34,
    spawn_angle_half_deg=125.0,
)


class TestGarmentFoldScene:
    def test_scene_registered(self) -> None:
        scene = get_scene("garment_fold")
        assert scene.scene_id == "garment_fold"
        assert scene.scene_xml_relpath == "scenes/garment_fold/scene.xml"
        assert scene.free_joints == ()
        assert scene.target_bodies == ()

    def test_reset_fn_registered(self) -> None:
        assert get_reset_fn("garment_fold") is not None

    def test_list_scenes_contains_garment_fold(self) -> None:
        assert "garment_fold" in list_scenes()

    def test_unknown_scene_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown scene"):
            get_scene("nope")


def _mock_mujoco_for(name_to_id: dict[str, int]) -> MagicMock:
    mock_mujoco = MagicMock()
    mock_mujoco.mjtObj = MagicMock()
    mock_mujoco.mj_name2id.side_effect = lambda _model, _obj_type, name: name_to_id.get(name, -1)
    return mock_mujoco


class TestFreejointSpawnReset:
    def test_places_every_free_joint_clear_of_the_target(self) -> None:
        scene = _MULTI_BLOCK_SCENE
        joint_ids = {joint: i + 1 for i, joint in enumerate(scene.free_joints)}
        mock_mujoco = _mock_mujoco_for({"target": 0, **joint_ids})

        with (
            patch.dict("sys.modules", {"mujoco": mock_mujoco}),
            patch.dict(scene_registry._SCENES, {scene.scene_id: scene}),  # noqa: SLF001
        ):
            model = MagicMock()
            model.jnt_qposadr = [0, 0, 7, 14]
            model.jnt_dofadr = [0, 0, 6, 12]
            data = MagicMock()
            data.qpos = np.zeros(21)
            data.qvel = np.ones(18)
            data.xpos = np.array([[0.30, 0.0, 0.01]])

            fn = scene_registry._freejoint_spawn_reset(scene)  # noqa: SLF001
            fn(model, data, np.random.default_rng(0))

            placed = [data.qpos[adr : adr + 3] for adr in (0, 7, 14)]
            for xy in placed:
                assert xy[2] == pytest.approx(0.02)
                assert np.hypot(xy[0] - 0.30, xy[1]) >= scene.target_min_sep
            for i, first in enumerate(placed):
                for second in placed[i + 1 :]:
                    assert np.hypot(first[0] - second[0], first[1] - second[1]) >= scene.block_min_sep
            assert np.all(data.qvel == 0.0)
            mock_mujoco.mj_forward.assert_called_once_with(model, data)

    def test_orientation_is_a_unit_yaw_quaternion(self) -> None:
        mock_mujoco = _mock_mujoco_for({"target": 0, "block1:joint": 1})

        with patch.dict("sys.modules", {"mujoco": mock_mujoco}):
            model = MagicMock()
            model.jnt_qposadr = [0, 0]
            model.jnt_dofadr = [0, 0]
            data = MagicMock()
            data.qpos = np.zeros(7)
            data.qvel = np.ones(6)
            data.xpos = np.array([[0.30, 0.0, 0.01]])

            fn = get_reset_fn("single_pick_place")
            assert fn is not None
            fn(model, data, np.random.default_rng(3))

            quat = data.qpos[3:7]
            assert np.linalg.norm(quat) == pytest.approx(1.0)
            assert quat[1] == pytest.approx(0.0)
            assert quat[2] == pytest.approx(0.0)

    def test_missing_joint_is_skipped(self) -> None:
        mock_mujoco = _mock_mujoco_for({"target": 0})

        with patch.dict("sys.modules", {"mujoco": mock_mujoco}):
            model = MagicMock()
            data = MagicMock()
            data.xpos = np.array([[0.30, 0.0, 0.01]])

            fn = get_reset_fn("single_pick_place")
            assert fn is not None
            fn(model, data, np.random.default_rng(0))

            mock_mujoco.mj_forward.assert_called_once_with(model, data)


class TestSceneSpawnConfig:
    def test_single_pick_place_carries_its_own_spawn_arc(self) -> None:
        scene = get_scene("single_pick_place")
        assert scene.spawn_center == (0.22, 0.0)
        assert (scene.spawn_min_r, scene.spawn_max_r) == (0.05, 0.14)
        assert scene.spawn_angle_half_deg == 50.0

    def test_every_freejoint_scene_declares_its_free_joints(self) -> None:
        for scene_id in ("single_pick_place", "yahtzee"):
            assert get_scene(scene_id).free_joints


class TestArmCompatibility:
    def test_only_garment_fold_is_written_for_two_arms(self) -> None:
        assert {scene_id for scene_id, scene in list_scenes().items() if scene.written_arms == 2} == {"garment_fold"}

    def test_every_tabletop_scene_runs_one_or_two_arms(self) -> None:
        single, bimanual = list_scenes_for_arms(1), list_scenes_for_arms(2)
        # Floor scenes spawn floating-base robots, never SO-101 arms.
        assert set(single) == set(bimanual) == set(list_scenes()) - {"floor_flat"}
        assert get_scene("floor_flat").arm_counts == (1,)
        assert list_scenes_for_arms(3) == {}

    def test_garment_fold_pins_its_two_arm_home_and_has_a_one_arm_home(self) -> None:
        scene = get_scene("garment_fold")
        two = dict(scene.home_pose(2))
        assert two["left_shoulder_pan"] == pytest.approx(-1.1)
        assert two["right_shoulder_pan"] == pytest.approx(1.1)
        assert dict(scene.home_pose(1)) == {
            "shoulder_pan": 0.0,
            "shoulder_lift": 0.3,
            "elbow_flex": 0.8,
            "wrist_flex": 0.3,
        }
        assert get_scene("single_pick_place").home_pose(2) == ()

    def test_a_shared_home_applies_to_each_arm(self) -> None:
        scene = get_scene("conveyor_sort")
        assert scene.home_pose(1) == scene.home_qpos
        two = scene.home_pose(2)
        assert two == tuple((f"{p}{joint}", value) for p in ("left_", "right_") for joint, value in scene.home_qpos)

    def test_arm_prefixes(self) -> None:
        assert scene_registry.arm_prefixes(1) == ("",)
        assert scene_registry.arm_prefixes(2) == scene_registry.BIMANUAL_PREFIXES == ("left_", "right_")
        with pytest.raises(ValueError, match="not 3"):
            scene_registry.arm_prefixes(3)

    @pytest.mark.parametrize(
        ("profile", "scene_id", "message"),
        [
            ("aloha", "single_pick_place", "has 2 arms already"),
            ("unitree_go2", "floor_flat", "one floating-base robot"),
        ],
    )
    def test_bimanual_is_refused_where_it_cannot_work(self, profile: str, scene_id: str, message: str) -> None:
        with pytest.raises(ValueError, match=message):
            scene_registry.check_arm_count(get_scene(scene_id), get_profile(profile), 2)
        scene_registry.check_arm_count(get_scene(scene_id), get_profile(profile), 1)

    def test_supported_arm_counts(self) -> None:
        counts = {name: scene_registry.supported_arm_counts(get_profile(name)) for name in ("so101", "koch", "aloha", "unitree_g1")}
        assert counts == {"so101": (1, 2), "koch": (1, 2), "aloha": (1,), "unitree_g1": (1,)}


class TestGarmentFoldReset:
    def test_reset_sets_home_and_flex(self) -> None:
        mock_mujoco = MagicMock()
        mock_mujoco.mj_name2id.side_effect = lambda _model, _obj_type, name: {
            "left_shoulder_pan": 0,
            "left_shoulder_lift": 1,
            "left_elbow_flex": 2,
            "left_wrist_flex": 3,
            "right_shoulder_pan": 4,
            "right_shoulder_lift": 5,
            "right_elbow_flex": 6,
            "right_wrist_flex": 7,
        }.get(name, -1)
        mock_mujoco.mjtObj = MagicMock()

        with patch.dict("sys.modules", {"mujoco": mock_mujoco}):
            nflexvert = 196
            model = MagicMock()
            model.nflex = 1
            model.nflexvert = nflexvert
            model.njnt = 8 + 3 * nflexvert
            model.jnt_bodyid = [100, 101, 102, 103, 104, 105, 106, 107, *([200] * (3 * nflexvert))]
            model.jnt_qposadr = [0, 1, 2, 3, 4, 5, 6, 7, *range(8, 8 + 3 * nflexvert)]
            model.jnt_dofadr = [0, 1, 2, 3, 4, 5, 6, 7, *range(8, 8 + 3 * nflexvert)]
            model.flex_vertbodyid = [200] * nflexvert
            model.flex_vert = np.tile(np.array([0.0, 0.02, 0.322]), (nflexvert, 1))

            qpos = np.zeros(8 + 3 * nflexvert)
            qvel = np.ones(8 + 3 * nflexvert)
            ctrl = np.zeros(8)
            data = MagicMock()
            data.qpos = qpos
            data.qvel = qvel
            data.ctrl = ctrl

            fn = get_reset_fn("garment_fold", SO101_PROFILE, 2)
            assert fn is not None
            fn(model, data, np.random.default_rng(0))

            assert data.qpos[0] == pytest.approx(-1.1)
            assert data.qpos[3] == pytest.approx(0.3)
            assert data.qpos[4] == pytest.approx(1.1)
            assert data.qpos[7] == pytest.approx(0.3)
            np.testing.assert_allclose(data.qpos[8:], np.tile([0.0, 0.02, 0.322], nflexvert))
            assert np.all(data.qvel[8:] == 0.0)
            assert data.ctrl[0] == pytest.approx(-1.1)
            mock_mujoco.mj_forward.assert_called_once_with(model, data)

    def test_reset_without_jointed_flex_vertices_is_survivable(self) -> None:
        mock_mujoco = MagicMock()
        mock_mujoco.mjtObj = MagicMock()
        mock_mujoco.mj_name2id.return_value = -1

        with patch.dict("sys.modules", {"mujoco": mock_mujoco}):
            model = MagicMock()
            model.nflex = 1
            model.njnt = 2
            model.nflexvert = 4
            # No joint belongs to the flex vertex body (a pinned first vertex).
            model.jnt_bodyid = [1, 2]
            model.flex_vertbodyid = [200]
            data = MagicMock()

            fn = get_reset_fn("garment_fold")
            assert fn is not None
            fn(model, data, np.random.default_rng(0))

            mock_mujoco.mj_forward.assert_called_once_with(model, data)

    def test_reset_without_flex_is_noop(self) -> None:
        mock_mujoco = MagicMock()
        mock_mujoco.mjtObj = MagicMock()
        mock_mujoco.mj_name2id.return_value = -1

        with patch.dict("sys.modules", {"mujoco": mock_mujoco}):
            model = MagicMock()
            model.nflex = 0
            data = MagicMock()
            fn = get_reset_fn("garment_fold")
            assert fn is not None
            fn(model, data, np.random.default_rng(0))


class TestReachLayout:
    """``layout="reach"`` (SCN-6): the spawn arc and top-level bodies scale with reach / SO-101 reach."""

    def test_so101_keeps_the_scene_itself(self) -> None:
        scene = get_scene("single_pick_place")
        assert scene.layout == "reach"
        assert scene.layout_scale(SO101_PROFILE) == 1.0
        assert scene.for_profile(SO101_PROFILE) is scene

    def test_a_longer_arm_scales_the_spawn_arc_about_the_mount(self) -> None:
        scene = get_scene("single_pick_place")
        longer = dataclasses.replace(get_profile("ur5e"), reach=2 * SO101_PROFILE.reach)

        laid_out = scene.for_profile(longer)

        assert scene.layout_scale(longer) == pytest.approx(2.0)
        assert laid_out.spawn_center == pytest.approx((0.44, 0.0))
        assert (laid_out.spawn_min_r, laid_out.spawn_max_r) == pytest.approx((0.10, 0.28))
        assert laid_out.spawn_angle_half_deg == scene.spawn_angle_half_deg
        assert (laid_out.block_min_sep, laid_out.target_min_sep) == (scene.block_min_sep, scene.target_min_sep)

    def test_a_profile_spawn_center_replaces_the_scenes_before_scaling(self) -> None:
        scene = get_scene("single_pick_place")
        aloha = get_profile("aloha")
        scale = scene.layout_scale(aloha)

        center = scene.for_profile(aloha).spawn_center

        assert center == pytest.approx((-0.086 * scale, -0.01 * scale))

    def test_fixed_layouts_ignore_reach(self) -> None:
        assert get_scene("yahtzee").layout_scale(get_profile("ur5e")) == 1.0

    def test_a_reach_scene_refuses_a_profile_without_reach(self) -> None:
        scene = get_scene("single_pick_place")
        with pytest.raises(ValueError, match="set RobotProfile.reach"):
            scene.for_profile(dataclasses.replace(get_profile("ur5e"), reach=None))

    def test_every_profile_a_reach_scene_lists_has_a_reach_and_end_effectors(self) -> None:
        for scene in list_scenes().values():
            if scene.layout != "reach" or scene.robots == "*":
                continue
            for name in scene.robots:
                profile = get_profile(name)
                assert profile.reach is not None, name
                assert profile.end_effectors, name

    def test_the_spawn_reset_uses_the_profiles_layout(self) -> None:
        mock_mujoco = _mock_mujoco_for({"target": 0, "block1:joint": 1})
        longer = dataclasses.replace(get_profile("ur5e"), reach=2 * SO101_PROFILE.reach)
        laid_out = get_scene("single_pick_place").for_profile(longer)

        with patch.dict("sys.modules", {"mujoco": mock_mujoco}):
            model = MagicMock()
            model.jnt_qposadr = [0, 0]
            model.jnt_dofadr = [0, 0]
            data = MagicMock()
            data.qpos = np.zeros(7)
            data.qvel = np.zeros(6)
            data.xpos = np.array([[0.44, -0.60, 0.001]])
            fn = get_reset_fn("single_pick_place", longer)
            assert fn is not None
            for seed in range(20):
                fn(model, data, np.random.default_rng(seed))
                offset = np.hypot(*(data.qpos[:2] - np.asarray(laid_out.spawn_center)))
                assert laid_out.spawn_min_r - 1e-9 <= offset <= laid_out.spawn_max_r + 1e-9


class TestApplyLayout:
    _SCENE = """<mujoco><worldbody>
      <body name="target" pos="0.2 -0.3 0.001"><geom type="cylinder" size="0.05 0.001"/></body>
      <body name="rig" pos="-0.1 0.1 0.8" euler="0 0 1"><body name="tilt"><camera name="overview"/></body></body>
      <frame name="robot_mount" pos="{mount}"/>
    </worldbody></mujoco>"""

    def test_top_level_bodies_move_away_from_the_mount(self) -> None:
        spec = mujoco.MjSpec.from_string(self._SCENE.format(mount="0 0 0"))

        apply_layout(spec, SceneLayout(scale=2.0))

        np.testing.assert_allclose(spec.body("target").pos, [0.4, -0.6, 0.002])
        np.testing.assert_allclose(spec.body("rig").pos, [-0.2, 0.2, 1.6])
        assert spec.body("target").geoms[0].size[0] == pytest.approx(0.05)

    def test_an_overview_rig_override_places_the_cameras_top_level_body(self) -> None:
        spec = mujoco.MjSpec.from_string(self._SCENE.format(mount="0 0 0"))

        apply_layout(spec, SceneLayout(scale=2.0, overview_rig=OverviewRig(pos=(0.0, -0.7, 1.1), quat=(0.0, 0.0, 0.0, 1.0))))
        model = spec.compile()

        np.testing.assert_allclose(model.body("rig").pos, [0.0, -0.7, 1.1])
        np.testing.assert_allclose(model.body("rig").quat, [0.0, 0.0, 0.0, 1.0])
        np.testing.assert_allclose(model.body("target").pos, [0.4, -0.6, 0.002])

    def test_a_mount_off_the_origin_is_refused(self) -> None:
        spec = mujoco.MjSpec.from_string(self._SCENE.format(mount="0.1 0 0"))
        with pytest.raises(ValueError, match="robot mount at the origin"):
            apply_layout(spec, SceneLayout(scale=2.0))

    def test_the_so101_compiles_the_scene_as_written(self) -> None:
        assert get_scene("single_pick_place").layout_for(SO101_PROFILE) is None
