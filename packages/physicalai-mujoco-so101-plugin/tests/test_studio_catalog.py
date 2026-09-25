from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import numpy as np
from physicalai.config import Config
from physicalai.robot.errors import RobotNotConnectedError
from physicalai.robot.transport import RobotOwnerConfig, SharedRobot

from physicalai_mujoco_so101_plugin.constants import (
    BIMANUAL_SO101_JOINT_ORDER,
    DEFAULT_BIMANUAL_MUJOCO_OWNER_NAME,
    DEFAULT_MUJOCO_OWNER_NAME,
    SO101_JOINT_ORDER,
)
from physicalai_mujoco_so101_plugin.studio_catalog import (
    MuJoCoSO101BimanualPayload,
    MuJoCoSO101Payload,
    MuJoCoSO101Probe,
    _definitions,
    _SharedSO101Robot,
    register_physicalai_studio_plugin,
)


class TestMuJoCoSO101Payload:
    def test_default_payload(self) -> None:
        payload = MuJoCoSO101Payload()
        assert payload.name == DEFAULT_MUJOCO_OWNER_NAME
        assert payload.allow_remote is False
        assert payload.connect_timeout == 10.0

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
        for field in ("allow_remote", "connect_timeout"):
            assert schema["properties"][field]["x-physicalai-ui"]["advanced_configuration"] is True

    def test_payload_model_rebuild(self) -> None:
        MuJoCoSO101BimanualPayload.model_rebuild(raise_errors=True)


class TestDefinitions:
    def test_definitions_return_list(self) -> None:
        defs = _definitions()
        assert len(defs) == 2

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

        assert isinstance(built, _SharedSO101Robot)
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
        robot = _SharedSO101Robot(SharedRobot.attach(DEFAULT_MUJOCO_OWNER_NAME), SO101_JOINT_ORDER)
        assert robot.device_ids == ()
        assert robot.joint_names == list(SO101_JOINT_ORDER)

    def test_bimanual_joint_names(self) -> None:
        robot = _SharedSO101Robot(SharedRobot.attach(DEFAULT_BIMANUAL_MUJOCO_OWNER_NAME), BIMANUAL_SO101_JOINT_ORDER)
        assert robot.joint_names == list(BIMANUAL_SO101_JOINT_ORDER)

    def test_exports_attach_only_shared_robot_recipe(self) -> None:
        robot = _SharedSO101Robot(SharedRobot.attach(DEFAULT_MUJOCO_OWNER_NAME, connect_timeout=5.0), SO101_JOINT_ORDER)

        assert Config.from_instance(robot) == {
            "class_path": "physicalai_mujoco_so101_plugin.studio_catalog._SharedSO101Robot",
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
            _SharedSO101Robot(
                SharedRobot.attach("mujoco-so101", connect_timeout=5.0),
                SO101_JOINT_ORDER,
            ),
        )

        built = RobotOwnerConfig(name="studio-owner", robot=recipe).build()

        assert isinstance(built, _SharedSO101Robot)
        assert isinstance(built._shared_robot, SharedRobot)  # noqa: SLF001
        assert built._shared_robot._name == "mujoco-so101"  # noqa: SLF001
        assert built._shared_robot._robot is None  # noqa: SLF001
        assert built.joint_names == list(SO101_JOINT_ORDER)


class TestSharedRobotLifecycle:
    def test_real_shared_robot_disconnect_and_reconnect(self) -> None:
        shared = SharedRobot.attach("mujoco-lifecycle-test")
        robot = _SharedSO101Robot(shared, SO101_JOINT_ORDER)
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
        robot = _SharedSO101Robot(shared, SO101_JOINT_ORDER)

        robot.connect()
        assert robot.is_connected() is True
        robot.disconnect()

        shared.connect.assert_called_once()
        shared.is_connected.assert_called_once()
        shared.disconnect.assert_called_once()


class TestProbe:
    @pytest.mark.anyio
    async def test_discover(self) -> None:
        probe = MuJoCoSO101Probe()
        manager = AsyncMock()
        manager.robots = []
        result = await probe.discover(manager)
        assert result == []
        manager.find_robots.assert_awaited_once()

    @pytest.mark.anyio
    async def test_identify(self) -> None:
        probe = MuJoCoSO101Probe()
        payload = MuJoCoSO101Payload()
        await probe.identify(payload)

    @pytest.mark.anyio
    async def test_is_online_no_owner(self) -> None:
        from physicalai_mujoco_so101_plugin import studio_catalog as sc

        probe = MuJoCoSO101Probe()
        payload = MuJoCoSO101Payload(name="nonexistent")

        with patch.object(sc, "_check_zenoh_robot_online", return_value=False):
            result = await probe.is_online(payload)
            assert result is False


class TestRegistration:
    def test_register_called(self) -> None:
        registry = MagicMock()
        register_physicalai_studio_plugin(registry)
        assert registry.register_robot.call_count == 2
