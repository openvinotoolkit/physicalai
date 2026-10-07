from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import numpy as np
from physicalai.config import Config
from physicalai.robot.errors import RobotNotConnectedError
from physicalai.robot.transport import RobotOwnerConfig, SharedRobot

from physicalai_mujoco_plugin.virtual_leader import MuJoCoVirtualLeader
from physicalai_mujoco_plugin.constants import (
    BIMANUAL_SO101_JOINT_ORDER,
    DEFAULT_BIMANUAL_MUJOCO_OWNER_NAME,
    DEFAULT_MUJOCO_OWNER_NAME,
    SO101_JOINT_ORDER,
)
from physicalai_mujoco_plugin.profiles import PROFILES, SO101_PROFILE
from physicalai_mujoco_plugin.studio_catalog import (
    CatalogEntry,
    MuJoCoRobotProbe,
    MuJoCoVirtualLeaderPayload,
    MuJoCoSO101BimanualPayload,
    MuJoCoSO101Payload,
    _definitions,
    _SharedMuJoCoRobot,
    list_catalog_entries,
    register_physicalai_studio_plugin,
)

SO101_ENTRY = CatalogEntry(SO101_PROFILE, ("",))


class TestMuJoCoSO101Payload:
    def test_default_payload(self) -> None:
        payload = MuJoCoSO101Payload()
        assert payload.name == DEFAULT_MUJOCO_OWNER_NAME
        assert payload.allow_remote is False
        assert payload.connect_timeout == 10.0
        assert payload.http_url == "http://127.0.0.1:8080"

    def test_payload_stored_before_http_url_still_validates(self) -> None:
        stored = {"name": "my-sim", "allow_remote": True, "connect_timeout": 5.0}
        payload = MuJoCoSO101Payload.model_validate(stored)
        assert payload.model_dump() == {**stored, "http_url": "http://127.0.0.1:8080"}

    def test_custom_payload(self) -> None:
        payload = MuJoCoSO101Payload(name="my-sim", allow_remote=True, connect_timeout=5.0)
        assert payload.name == "my-sim"
        assert payload.allow_remote is True
        assert payload.connect_timeout == 5.0

    def test_payload_model_rebuild(self) -> None:
        MuJoCoSO101Payload.model_rebuild(raise_errors=True)

    def test_name_str_validated(self) -> None:
        payload = MuJoCoSO101Payload(name="test-robot-1")
        assert payload.name == "test-robot-1"


class TestMuJoCoSO101BimanualPayload:
    def test_default_owner_name_differs_from_single_arm(self) -> None:
        payload = MuJoCoSO101BimanualPayload()
        assert payload.name == DEFAULT_BIMANUAL_MUJOCO_OWNER_NAME
        assert payload.name != MuJoCoSO101Payload().name

    def test_inherits_connection_settings(self) -> None:
        payload = MuJoCoSO101BimanualPayload(allow_remote=True, connect_timeout=3.0)
        assert payload.allow_remote is True
        assert payload.connect_timeout == 3.0

    def test_connection_settings_are_advanced(self) -> None:
        schema = MuJoCoSO101BimanualPayload.model_json_schema()
        for field in ("allow_remote", "connect_timeout", "http_url"):
            assert schema["properties"][field]["x-physicalai-ui"]["advanced_configuration"] is True

    def test_payload_model_rebuild(self) -> None:
        MuJoCoSO101BimanualPayload.model_rebuild(raise_errors=True)


class TestDefinitions:
    def test_definitions_return_list(self) -> None:
        defs = _definitions()
        assert [d.role for d in defs[:3]] == ["follower", "follower", "leader"]
        assert {d.role for d in defs[3:]} == {"follower"}

    def test_definition_contents(self) -> None:
        defs = _definitions()
        definition = defs[0]
        assert definition.type == "MuJoCo_SO101_Follower"
        assert definition.display_name == "MuJoCo SO-101 Follower"
        assert definition.role == "follower"
        assert definition.asset is not None
        assert definition.robot_payload is MuJoCoSO101Payload
        assert definition.probe is not None

    def test_bimanual_definition_contents(self) -> None:
        defs = _definitions()
        definition = defs[1]
        assert definition.type == "MuJoCo_SO101_Bimanual_Follower"
        assert definition.display_name == "MuJoCo SO-101 Bimanual Follower"
        assert definition.role == "follower"
        assert definition.asset is not None
        assert definition.robot_payload is MuJoCoSO101BimanualPayload
        assert definition.probe is not None

    def test_adapter_options(self) -> None:
        defs = _definitions()
        definition = defs[0]
        assert definition.adapter_options.include_velocities is False
        assert definition.adapter_options.external_effort_gain is None
        assert definition.adapter_options.goal_time_scale == 1.0

    def test_builder_callable(self) -> None:
        defs = _definitions()
        definition = defs[0]
        assert callable(definition.robot_builder)

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("index", "name", "joint_order"),
        [
            (0, DEFAULT_MUJOCO_OWNER_NAME, SO101_JOINT_ORDER),
            (1, DEFAULT_BIMANUAL_MUJOCO_OWNER_NAME, BIMANUAL_SO101_JOINT_ORDER),
        ],
    )
    async def test_builder_exports_owner_recipe(
        self,
        index: int,
        name: str,
        joint_order: tuple[str, ...],
    ) -> None:
        definition = _definitions()[index]
        container = MagicMock(payload={"allow_remote": True, "connect_timeout": 3.0})

        robot = await definition.robot_builder(container, MagicMock())
        recipe = Config.from_instance(robot)
        built = RobotOwnerConfig(name="studio-owner", robot=recipe).build()

        assert isinstance(built, _SharedMuJoCoRobot)
        assert built._shared_robot._name == name  # noqa: SLF001
        assert built._shared_robot._allow_remote is True  # noqa: SLF001
        assert built._shared_robot._connect_timeout == 3.0  # noqa: SLF001
        assert built._shared_robot._robot is None  # noqa: SLF001
        assert built.joint_names == list(joint_order)

    def test_payload_model_class(self) -> None:
        defs = _definitions()
        definition = defs[0]
        assert definition.robot_payload is MuJoCoSO101Payload

    def test_urdf_asset(self) -> None:
        defs = _definitions()
        definition = defs[0]
        assert definition.asset is not None
        assert str(definition.asset.urdf_relative_path) == "so101/so101_new_calib.urdf"
        assert "so101" in definition.asset.packages
        assert "shoulder_pan.pos" in definition.asset.joint_map
        assert definition.asset.root_resolver is not None
        assert (definition.asset.root_resolver() / definition.asset.urdf_relative_path).is_file()

    def test_bimanual_urdf_asset(self) -> None:
        defs = _definitions()
        definition = defs[1]
        assert definition.asset is not None
        assert str(definition.asset.urdf_relative_path) == "so101/so101_dual.urdf"
        assert "so101" in definition.asset.packages
        assert "left_shoulder_pan.pos" in definition.asset.joint_map
        assert "right_shoulder_pan.pos" in definition.asset.joint_map
        assert definition.asset.root_resolver is not None
        assert (definition.asset.root_resolver() / definition.asset.urdf_relative_path).is_file()


class TestSharedRobotAdapter:
    def test_has_no_owned_devices(self) -> None:
        robot = _SharedMuJoCoRobot(SharedRobot.attach(DEFAULT_MUJOCO_OWNER_NAME), SO101_JOINT_ORDER)
        assert robot.device_ids == ()
        assert robot.joint_names == list(SO101_JOINT_ORDER)

    def test_bimanual_joint_names(self) -> None:
        robot = _SharedMuJoCoRobot(SharedRobot.attach(DEFAULT_BIMANUAL_MUJOCO_OWNER_NAME), BIMANUAL_SO101_JOINT_ORDER)
        assert robot.joint_names == list(BIMANUAL_SO101_JOINT_ORDER)

    def test_exports_attach_only_shared_robot_recipe(self) -> None:
        robot = _SharedMuJoCoRobot(SharedRobot.attach(DEFAULT_MUJOCO_OWNER_NAME, connect_timeout=5.0), SO101_JOINT_ORDER)

        assert Config.from_instance(robot) == {
            "class_path": "physicalai_mujoco_plugin.studio_catalog._SharedMuJoCoRobot",
            "init_args": {
                "shared_robot": {
                    "class_path": "physicalai.robot.SharedRobot",
                    "init_args": {
                        "name": DEFAULT_MUJOCO_OWNER_NAME,
                        "allow_remote": False,
                        "connect_timeout": 5.0,
                    },
                },
                "joint_names": list(SO101_JOINT_ORDER),
            },
        }

    def test_owner_build_constructs_attach_only_shared_robot(self) -> None:
        recipe = Config.from_instance(
            _SharedMuJoCoRobot(
                SharedRobot.attach("mujoco-so101", connect_timeout=5.0),
                SO101_JOINT_ORDER,
            ),
        )

        built = RobotOwnerConfig(name="studio-owner", robot=recipe).build()

        assert isinstance(built, _SharedMuJoCoRobot)
        assert isinstance(built._shared_robot, SharedRobot)  # noqa: SLF001
        assert built._shared_robot._name == "mujoco-so101"  # noqa: SLF001
        assert built._shared_robot._robot is None  # noqa: SLF001
        assert built.joint_names == list(SO101_JOINT_ORDER)


class TestSharedRobotLifecycle:
    def test_real_shared_robot_disconnect_and_reconnect(self) -> None:
        shared = SharedRobot.attach("mujoco-lifecycle-test")
        robot = _SharedMuJoCoRobot(shared, SO101_JOINT_ORDER)
        sessions = [MagicMock(), MagicMock()]
        with (
            patch("physicalai.robot.transport._shared_robot.open_session", side_effect=sessions),
            patch.object(shared, "_resolve_metadata", return_value={}),
            patch.object(shared, "_validate_metadata"),
            patch.object(shared, "_attach"),
        ):
            robot.connect()
            assert robot.is_connected()
            robot.disconnect()
            assert not robot.is_connected()
            with pytest.raises(RobotNotConnectedError):
                robot.get_observation()
            with pytest.raises(RobotNotConnectedError):
                robot.send_action(np.zeros(6))
            robot.connect()
            assert robot.is_connected()
            robot.disconnect()
        for session in sessions:
            session.close.assert_called_once()

    def test_delegates_connection_to_shared_robot(self) -> None:
        shared = MagicMock(spec=SharedRobot)
        shared.is_connected.return_value = True
        robot = _SharedMuJoCoRobot(shared, SO101_JOINT_ORDER)

        robot.connect()
        assert robot.is_connected() is True
        robot.disconnect()

        shared.connect.assert_called_once()
        shared.is_connected.assert_called_once()
        shared.disconnect.assert_called_once()


class TestProbe:
    @pytest.mark.anyio
    async def test_discover(self) -> None:
        probe = MuJoCoRobotProbe(SO101_ENTRY)
        manager = AsyncMock()
        manager.robots = []
        result = await probe.discover(manager)
        assert result == []
        manager.find_robots.assert_awaited_once()

    @pytest.mark.anyio
    async def test_identify(self) -> None:
        probe = MuJoCoRobotProbe(SO101_ENTRY)
        payload = MuJoCoSO101Payload()
        await probe.identify(payload)

    @pytest.mark.anyio
    async def test_is_online_no_owner(self) -> None:
        from physicalai_mujoco_plugin import studio_catalog as sc

        probe = MuJoCoRobotProbe(SO101_ENTRY)
        payload = MuJoCoSO101Payload(name="nonexistent")

        with patch.object(sc, "_check_zenoh_robot_online", return_value=False):
            result = await probe.is_online(payload)
            assert result is False


class TestProbeJointCheck:
    @staticmethod
    def _owner(joint_names: tuple[str, ...]) -> MagicMock:
        owner = MagicMock()
        owner.joint_names = list(joint_names)
        return owner

    def test_matching_owner_is_online(self) -> None:
        from physicalai_mujoco_plugin import studio_catalog as sc

        owner = self._owner(SO101_JOINT_ORDER)
        with patch.object(sc.SharedRobot, "attach", return_value=owner):
            assert sc._check_zenoh_robot_online("sim", SO101_JOINT_ORDER)  # noqa: SLF001
        owner.disconnect.assert_called_once()

    def test_owner_with_other_joints_is_not_online(self) -> None:
        from physicalai_mujoco_plugin import studio_catalog as sc

        owner = self._owner(BIMANUAL_SO101_JOINT_ORDER)
        with patch.object(sc.SharedRobot, "attach", return_value=owner):
            assert not sc._check_zenoh_robot_online("sim", SO101_JOINT_ORDER)  # noqa: SLF001
        owner.disconnect.assert_called_once()

    @pytest.mark.anyio
    async def test_each_follower_entry_probes_for_its_own_joints(self) -> None:
        from physicalai_mujoco_plugin import studio_catalog as sc

        single, bimanual = _definitions()[:2]
        with patch.object(sc, "_check_zenoh_robot_online", return_value=True) as check:
            await single.probe.is_online(MuJoCoSO101Payload(name="a"))
            await bimanual.probe.is_online(MuJoCoSO101BimanualPayload(name="b"))
        assert [call.args for call in check.call_args_list] == [
            ("a", SO101_JOINT_ORDER),
            ("b", BIMANUAL_SO101_JOINT_ORDER),
        ]


class TestRegistration:
    def test_register_called(self) -> None:
        registry = MagicMock()
        register_physicalai_studio_plugin(registry)
        assert registry.register_robot.call_count == len(list_catalog_entries()) + 1  # + virtual leader


class TestVirtualLeaderDefinition:
    def test_definition_contents(self) -> None:
        definition = _definitions()[2]
        assert definition.type == "MuJoCo_SO101_Virtual_Leader"
        assert definition.role == "leader"
        assert definition.robot_payload is MuJoCoVirtualLeaderPayload
        assert definition.asset is not None
        assert definition.probe is not None

    def test_payload_bounds_the_port(self) -> None:
        assert MuJoCoVirtualLeaderPayload().http_port == 8080
        with pytest.raises(ValueError, match="less than or equal"):
            MuJoCoVirtualLeaderPayload(http_port=70000)

    @pytest.mark.anyio
    async def test_builder_exports_a_leader_recipe(self) -> None:
        definition = _definitions()[2]
        robot = await definition.robot_builder(MagicMock(payload={"http_port": 8123}), MagicMock())
        recipe = Config.from_instance(robot)
        built = RobotOwnerConfig(name="studio-leader", robot=recipe).build()

        assert isinstance(built, MuJoCoVirtualLeader)
        assert built.http_port == 8123
        assert built.host == "127.0.0.1"
        assert built.joint_names == list(SO101_JOINT_ORDER)

    @pytest.mark.anyio
    async def test_probe_reports_offline_without_a_simulation(self) -> None:
        probe = _definitions()[2].probe
        assert probe is not None
        assert await probe.discover(MagicMock()) == []
        # Port 1 is never a simulation.
        assert await probe.is_online(MuJoCoVirtualLeaderPayload(http_port=1)) is False


class TestGeneratedEntries:
    """STU-1/STU-2: one follower entry per supported profile and arm layout."""

    def test_types_follow_the_profiles(self) -> None:
        assert [(entry.type, entry.owner_name) for entry in list_catalog_entries()] == [
            ("MuJoCo_SO101_Follower", DEFAULT_MUJOCO_OWNER_NAME),
            ("MuJoCo_SO101_Bimanual_Follower", DEFAULT_BIMANUAL_MUJOCO_OWNER_NAME),
            ("MuJoCo_WidowXAI_Follower", "mujoco-trossen_wxai-follow"),
            ("MuJoCo_WidowXAI_Bimanual_Follower", "mujoco-trossen_wxai-bimanual-follow"),
            ("MuJoCo_reBotB601_Follower", "mujoco-rebot_b601-follow"),
            ("MuJoCo_UniversalRobotsUR5e_Follower", "mujoco-ur5e-follow"),
            ("MuJoCo_ALOHA_Follower", "mujoco-aloha-follow"),
            ("MuJoCo_SOARM100_Follower", "mujoco-so_arm100-follow"),
            ("MuJoCo_Koch_Follower", "mujoco-koch-follow"),
            ("MuJoCo_AgileXPiPER_Follower", "mujoco-piper-follow"),
            ("MuJoCo_FrankaResearch3_Follower", "mujoco-franka_fr3-follow"),
            ("MuJoCo_FrankaEmikaPanda_Follower", "mujoco-franka_panda-follow"),
            ("MuJoCo_UFACTORYxArm7_Follower", "mujoco-xarm7-follow"),
            ("MuJoCo_KinovaGen3_Follower", "mujoco-kinova_gen3-follow"),
            ("MuJoCo_UnitreeG1_Follower", "mujoco-unitree_g1-follow"),
            ("MuJoCo_UnitreeGo2_Follower", "mujoco-unitree_go2-follow"),
            ("MuJoCo_BostonDynamicsSpot_Follower", "mujoco-boston_dynamics_spot-follow"),
        ]
        assert len({definition.type for definition in _definitions()}) == len(_definitions())

    def test_every_supported_profile_has_a_single_robot_entry(self) -> None:
        """SUP-1/STU-1: twin, dataset and experimental profiles are all selectable in Studio."""
        supported = {name for name, profile in PROFILES.items() if profile.tier != "unsupported"}
        single = {entry.profile.name for entry in list_catalog_entries() if not entry.bimanual}

        assert supported == single
        assert {"aloha", "so_arm100", "koch", "piper", "franka_fr3", "franka_panda", "xarm7", "kinova_gen3"} <= single

    def test_display_names_mark_experimental_entries(self) -> None:
        names = {entry.type: entry.display_name for entry in list_catalog_entries()}
        assert names["MuJoCo_WidowXAI_Bimanual_Follower"] == "MuJoCo WidowX AI Bimanual Follower"
        assert names["MuJoCo_UnitreeG1_Follower"] == "MuJoCo Unitree G1 Follower (experimental)"

    def test_unsupported_profiles_get_no_entry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from physicalai_mujoco_plugin import studio_catalog as sc
        from physicalai_mujoco_plugin.profiles import RobotProfile

        unsupported = RobotProfile(
            name="unitree_go1", display_name="Go1", menagerie_model="unitree_go1", default_scene="floor_flat"
        )
        monkeypatch.setattr(sc, "PROFILES", {"so101": SO101_PROFILE, "unitree_go1": unsupported})
        assert [entry.profile.name for entry in list_catalog_entries()] == ["so101", "so101"]

    def test_generated_payloads_default_to_their_owner(self) -> None:
        for definition, entry in zip(_definitions()[3:], list_catalog_entries()[2:], strict=True):
            payload_model = definition.robot_payload
            payload = payload_model()
            assert (payload.name, payload.http_url) == (entry.owner_name, "http://127.0.0.1:8080")
            payload_model.model_rebuild(raise_errors=True)
            assert payload_model.model_validate({"name": "x"}).name == "x"

    def test_widowx_reuses_studios_urdf_and_others_rely_on_the_viewer(self) -> None:
        definitions = {definition.type: definition for definition in _definitions()}
        asset = definitions["MuJoCo_WidowXAI_Follower"].asset
        assert asset is not None
        assert asset.root_resolver is None  # Studio's built-in WidowX AI URDF
        assert str(asset.urdf_relative_path) == "widowx/urdf/generated/wxai/wxai_follower.urdf"
        assert set(asset.joint_map) == {f"{name}.pos" for name in list_catalog_entries()[2].joint_names()}
        for name in ("MuJoCo_WidowXAI_Bimanual_Follower", "MuJoCo_reBotB601_Follower", "MuJoCo_UnitreeG1_Follower"):
            assert definitions[name].asset is None

    def test_two_robot_entries_use_the_scenes_prefixes(self) -> None:
        import mujoco  # noqa: PLC0415

        from physicalai_mujoco_plugin.compose import anchor_prefixes  # noqa: PLC0415
        from physicalai_mujoco_plugin.scene_registry import list_scenes  # noqa: PLC0415
        from physicalai_mujoco_plugin.studio_catalog import BIMANUAL_PREFIXES  # noqa: PLC0415

        for scene in list_scenes().values():
            prefixes = anchor_prefixes(mujoco.MjSpec.from_file(str(scene.scene_xml_path)))
            assert len(prefixes) == scene.num_arms
            assert prefixes == (BIMANUAL_PREFIXES if scene.num_arms == 2 else ("",))

    @pytest.mark.anyio
    async def test_probe_checks_for_the_entrys_joints(self) -> None:
        from physicalai_mujoco_plugin import studio_catalog as sc

        definition = {d.type: d for d in _definitions()}["MuJoCo_WidowXAI_Bimanual_Follower"]
        with patch.object(sc, "_check_zenoh_robot_online", return_value=True) as check:
            assert await definition.probe.is_online(definition.robot_payload(name="w"))
        names = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_yaw", "wrist_roll", "gripper")
        check.assert_called_once_with("w", tuple(f"{side}_{name}" for side in ("left", "right") for name in names))

    @pytest.mark.anyio
    async def test_derived_profiles_take_their_names_from_the_model(self) -> None:
        from physicalai_mujoco_plugin import compose

        definition = {d.type: d for d in _definitions()}["MuJoCo_UniversalRobotsUR5e_Follower"]
        layout = MagicMock(joint_names=("shoulder_pan", "elbow"))
        with patch.object(compose, "robot_layout", return_value=layout):
            robot = await definition.robot_builder(MagicMock(payload={}), MagicMock())
        assert robot.joint_names == ["shoulder_pan", "elbow"]
        assert robot._shared_robot._name == "mujoco-ur5e-follow"  # noqa: SLF001

    @pytest.mark.anyio
    async def test_an_entry_whose_model_is_unavailable_is_offline(self) -> None:
        from physicalai_mujoco_plugin import compose
        from physicalai_mujoco_plugin import studio_catalog as sc

        definition = {d.type: d for d in _definitions()}["MuJoCo_UnitreeG1_Follower"]
        with (
            patch.object(compose, "robot_layout", side_effect=RuntimeError("offline")),
            patch.object(sc, "_check_zenoh_robot_online") as check,
        ):
            assert await definition.probe.is_online(definition.robot_payload()) is False
        check.assert_not_called()

    def test_import_needs_no_mujoco(self) -> None:
        import subprocess  # noqa: PLC0415
        import sys  # noqa: PLC0415

        code = (
            "import sys, physicalai_mujoco_plugin.studio_catalog as sc; sc._definitions(); "
            "sys.exit('mujoco' in sys.modules)"
        )
        assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0  # noqa: S603
