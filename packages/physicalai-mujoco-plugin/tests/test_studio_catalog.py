from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import numpy as np
from physicalai.config import Config
from physicalai.robot.errors import RobotNotConnectedError, RobotProtocolMismatch, RobotTransportError
from physicalai.robot.transport import RobotOwnerConfig, SharedRobot
from physicalai_studio_plugin import SimulationLaunch, SimulationScene

from physicalai_mujoco_plugin.virtual_leader import MuJoCoVirtualLeader
from physicalai_mujoco_plugin.constants import (
    BIMANUAL_SO101_JOINT_ORDER,
    DEFAULT_BIMANUAL_MUJOCO_OWNER_NAME,
    DEFAULT_MUJOCO_OWNER_NAME,
    MAX_SEED,
    SO101_JOINT_ORDER,
)
from physicalai_mujoco_plugin.profiles import PROFILES, SO101_PROFILE
from physicalai_mujoco_plugin.scene_registry import check_arm_count, get_scene
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


class TestNotRunningError:
    """P2: a follower whose simulation is not running says which one and how to start it."""

    START = "Start it with: uv run physicalai-mujoco start --profile trossen_wxai --bimanual"

    def _robot(self, error: Exception) -> _SharedMuJoCoRobot:
        shared = MagicMock(spec=SharedRobot)
        shared.name = "mujoco-trossen_wxai-bimanual-follow"
        shared.connect.side_effect = error
        return _SharedMuJoCoRobot(shared, ("joint_0",), self.START)

    def test_no_owner_names_the_simulation_and_its_start_command(self) -> None:
        missing = RobotTransportError(
            "no owner found for 'mujoco-trossen_wxai-bimanual-follow' (attach-only mode: robot config not provided)"
        )
        with pytest.raises(RobotTransportError) as raised:
            self._robot(missing).connect()

        assert type(raised.value) is RobotTransportError
        assert str(raised.value) == (
            f"No MuJoCo simulation named 'mujoco-trossen_wxai-bimanual-follow' is running. {self.START}"
        )
        assert raised.value.__cause__ is missing

    @pytest.mark.parametrize(
        "error",
        [
            RobotTransportError("no state received from owner of 'x' within 5.0s"),
            RobotProtocolMismatch("no owner found for 'x' speaks another protocol"),
            TimeoutError("zenoh"),
        ],
    )
    def test_other_errors_pass_through(self, error: Exception) -> None:
        with pytest.raises(type(error)) as raised:
            self._robot(error).connect()
        assert raised.value is error

    @pytest.mark.parametrize(
        ("entry_type", "name", "http_url", "hint"),
        [
            (
                "MuJoCo_SO101_Follower",
                DEFAULT_MUJOCO_OWNER_NAME,
                "http://127.0.0.1:8080",
                "Start it with: uv run physicalai-mujoco start --profile so101",
            ),
            (
                "MuJoCo_WidowXAI_Bimanual_Follower",
                "mujoco-trossen_wxai-bimanual-follow",
                "http://localhost:8123/",
                "Start it with: uv run physicalai-mujoco start --profile trossen_wxai --bimanual --http-port 8123",
            ),
            (
                "MuJoCo_Koch_Follower",
                "my koch",
                "http://[::1]:8080",
                "Start it with: uv run physicalai-mujoco start --profile koch --name='my koch'",
            ),
            (
                "MuJoCo_Koch_Follower",
                "-sim",
                "http://127.0.0.1:8080",
                "Start it with: uv run physicalai-mujoco start --profile koch --name=-sim",
            ),
            (
                "MuJoCo_SO101_Follower",
                DEFAULT_MUJOCO_OWNER_NAME,
                "http://10.0.0.5:8081",
                "Start it on 10.0.0.5 with: uv run physicalai-mujoco start --profile so101 "
                "--http-host 10.0.0.5 --http-port 8081 --allow-remote",
            ),
            (
                "MuJoCo_SO101_Follower",
                DEFAULT_MUJOCO_OWNER_NAME,
                "http://127.0.0.1:notaport",
                "Start it with: uv run physicalai-mujoco start --profile so101",
            ),
        ],
    )
    def test_start_hint_follows_the_entry_and_its_payload(
        self, entry_type: str, name: str, http_url: str, hint: str
    ) -> None:
        entry = {entry.type: entry for entry in list_catalog_entries()}[entry_type]
        assert entry.start_hint(name, http_url) == hint

    @pytest.mark.parametrize("name", ["-sim", "--profile", "my koch", "it's"])
    def test_the_start_command_parses_back_to_the_payload(self, name: str) -> None:
        """The CLI must read the suggested command as the follower's name and port, whatever the name."""
        import shlex

        from physicalai_mujoco_plugin import __main__ as cli

        entry = {entry.type: entry for entry in list_catalog_entries()}["MuJoCo_Koch_Follower"]
        hint = entry.start_hint(name, "http://127.0.0.1:8123")
        argv = shlex.split(hint.removeprefix("Start it with: "))
        assert argv[:3] == ["uv", "run", "physicalai-mujoco"]
        args = cli._build_parser().parse_args(argv[3:])
        assert (args.profile, args.name, args.http_port, args.bimanual) == ("koch", name, 8123, False)

    @pytest.mark.anyio
    async def test_builder_passes_the_payloads_start_hint(self) -> None:
        definition = {d.type: d for d in _definitions()}["MuJoCo_WidowXAI_Bimanual_Follower"]
        robot = await definition.robot_builder(MagicMock(payload={"http_url": "http://127.0.0.1:8123"}), MagicMock())

        hint = f"{self.START} --http-port 8123"
        assert robot._start_hint == hint  # noqa: SLF001
        assert Config.from_instance(robot)["init_args"]["start_hint"] == hint


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
            ("MuJoCo_reBotB601_Bimanual_Follower", "mujoco-rebot_b601-bimanual-follow"),
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

    def test_no_entry_offers_zero_calibration(self) -> None:
        """Simulated joints have no motor offset, so Studio must not offer the guided zero-pose step.

        It would call ``release``/``set_zero`` on the driver, which ``MuJoCoRobot`` doesn't have; this
        also covers the reBot B601 twin, whose real driver has the step.
        """
        assert [d.type for d in _definitions() if d.zero_calibration is not None] == []

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

    def test_only_twins_get_a_bimanual_entry(self) -> None:
        """D2: two-arm entries for the robots with real bimanual leaders; other arms run two from the CLI only."""
        from physicalai_mujoco_plugin.scene_registry import arm_prefixes  # noqa: PLC0415
        from physicalai_mujoco_plugin.studio_catalog import BIMANUAL_PREFIXES  # noqa: PLC0415

        bimanual = [entry.profile.name for entry in list_catalog_entries() if entry.bimanual]
        assert bimanual == ["so101", "trossen_wxai", "rebot_b601"]
        assert {PROFILES[name].tier for name in bimanual} == {"twin"}
        assert BIMANUAL_PREFIXES == arm_prefixes(2)

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


class TestSimulationLaunch:
    """P3: every follower entry tells Studio how to start its simulation."""

    def test_followers_offer_the_scenes_for_their_arm_count_and_the_leader_none(self) -> None:
        launches = {d.type: d.simulation for d in _definitions()}
        assert launches.pop("MuJoCo_SO101_Virtual_Leader") is None
        so101 = ["single_pick_place", "yahtzee", "conveyor_sort", "garment_fold"]
        expected = {
            "MuJoCo_SO101_Follower": so101,
            "MuJoCo_SO101_Bimanual_Follower": so101,
            "MuJoCo_WidowXAI_Follower": ["single_pick_place", "garment_fold"],
            "MuJoCo_WidowXAI_Bimanual_Follower": ["single_pick_place", "garment_fold"],
            **dict.fromkeys(
                ["MuJoCo_UnitreeG1_Follower", "MuJoCo_UnitreeGo2_Follower", "MuJoCo_BostonDynamicsSpot_Follower"],
                ["floor_flat"],
            ),
        }
        for entry in list_catalog_entries():
            launch = launches[entry.type]
            assert launch is not None
            scene_ids = [scene.id for scene in launch.scenes]
            assert scene_ids == expected.get(entry.type, ["single_pick_place"]), entry.type
            assert launch.default_scene == entry.profile.default_scene
            for scene_id in scene_ids:
                check_arm_count(get_scene(scene_id), entry.profile, len(entry.prefixes))

    def test_scenes_carry_the_registrys_names_and_descriptions(self) -> None:
        launch = SO101_ENTRY.simulation_launch()
        assert launch is not None
        conveyor = get_scene("conveyor_sort")
        assert launch.scenes[2] == SimulationScene(
            id="conveyor_sort", display_name=conveyor.display_name, description=conveyor.description
        )

    @pytest.mark.parametrize(
        ("entry_type", "seed", "extra"),
        [
            ("MuJoCo_SO101_Follower", None, []),
            ("MuJoCo_WidowXAI_Bimanual_Follower", 7, ["--bimanual"]),
            ("MuJoCo_UnitreeGo2_Follower", SimulationLaunch.max_seed, []),
        ],
    )
    def test_argv_starts_a_supervised_simulation_that_start_parses(
        self, entry_type: str, seed: int | None, extra: list[str]
    ) -> None:
        from physicalai_mujoco_plugin import __main__ as cli  # noqa: PLC0415

        entry = {entry.type: entry for entry in list_catalog_entries()}[entry_type]
        launch = entry.simulation_launch()
        assert launch is not None
        argv = launch.argv(launch.default_scene, "-studio sim", seed)
        assert argv[:6] == [sys.executable, "-m", "physicalai_mujoco_plugin", "start", "--profile", entry.profile.name]
        assert argv[6 : 6 + len(extra)] == extra
        args = cli._build_parser().parse_args(argv[3:])  # noqa: SLF001
        assert (args.profile, args.bimanual, args.scene, args.name, args.seed) == (
            entry.profile.name,
            entry.bimanual,
            launch.default_scene,
            "-studio sim",
            seed,
        )
        assert (args.status_json, args.exit_with_parent, args.http_port, args.viser_port) == (True, True, 0, 0)
        assert args.viewer_theme == "studio"

    def test_the_seed_range_matches_post_seed(self) -> None:
        assert SimulationLaunch.max_seed == MAX_SEED

    def test_the_module_runs_with_python_dash_m(self) -> None:
        import subprocess  # noqa: PLC0415

        command = [sys.executable, "-m", "physicalai_mujoco_plugin", "start", "--help"]
        result = subprocess.run(command, check=False, capture_output=True, text=True)  # noqa: S603
        assert result.returncode == 0
        assert "--seed" in result.stdout
