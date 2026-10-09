from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from physicalai_studio_plugin import (
    PortScanner,
    RobotAdapterOptions,
    RobotAsset,
    RobotCatalogDefinition,
    RobotProbe,
    RobotZeroCalibration,
    SerialPortInfo,
    SimulationLaunch,
    SimulationScene,
    robot_field_ui,
    robot_payload_ui,
    validate_robot_payload_ui,
)


class TestPayload(BaseModel):
    serial_number: str = Field(...)
    connection_string: str = ""


class TestProbe:
    """Structurally implements RobotProbe[TestPayload]."""

    async def discover(self, manager: PortScanner) -> list[SerialPortInfo]:
        return []

    async def identify(
        self,
        payload: TestPayload,
        manager: PortScanner | None,
        joint: str | None = None,
    ) -> None:
        self._last_payload = payload

    async def is_online(
        self,
        payload: TestPayload,
        manager: PortScanner | None = None,
    ) -> bool:
        return payload.serial_number != ""


def test_probe_is_runtime_checkable() -> None:
    probe = TestProbe()
    assert isinstance(probe, RobotProbe)


def test_serial_port_info_allows_either_identifier() -> None:
    assert SerialPortInfo(serial_number="SN-001").connection_string is None
    assert SerialPortInfo(connection_string="/dev/ttyUSB0").serial_number is None


def test_typed_payload_reaches_identify() -> None:
    probe = TestProbe()
    payload = TestPayload(serial_number="SN-001")

    asyncio.run(probe.identify(payload, None))
    assert probe._last_payload is payload


def test_is_online_typed_payload() -> None:
    probe = TestProbe()
    payload = TestPayload(serial_number="SN-001")

    result = asyncio.run(probe.is_online(payload))
    assert result


def test_is_online_empty_payload() -> None:
    probe = TestProbe()
    payload = TestPayload(serial_number="")

    result = asyncio.run(probe.is_online(payload))
    assert not result


def test_definition_creation() -> None:
    asset = RobotAsset(
        urdf_relative_path=Path("test/model.urdf"),
        packages={"test": Path("test")},
        joint_map={"gripper.pos": ["gripper"]},
    )
    probe = TestProbe()
    definition = RobotCatalogDefinition[TestPayload](
        type="Test_Follower",
        display_name="Test Follower",
        role="follower",
        robot_payload=TestPayload,
        asset=asset,
        adapter_options=RobotAdapterOptions(include_velocities=True),
        probe=probe,
    )

    assert definition.type == "Test_Follower"
    assert definition.robot_payload is TestPayload
    assert definition.probe is probe


async def _noop_calibration_step(robot: object) -> None:
    _ = robot


def test_definition_defaults_to_no_zero_calibration() -> None:
    definition = RobotCatalogDefinition(type="Test_Follower", display_name="Test Follower", role="follower")

    assert definition.zero_calibration is None


def test_definition_accepts_zero_calibration() -> None:
    calibration = RobotZeroCalibration(instructions="Move to the rest pose.", set_zero=_noop_calibration_step)
    definition = RobotCatalogDefinition(
        type="Test_Follower", display_name="Test Follower", role="follower", zero_calibration=calibration
    )

    assert definition.zero_calibration is calibration
    assert calibration.release is None
    assert calibration.zero_tolerance_deg == 5.0


def test_zero_calibration_steps_receive_the_concrete_driver() -> None:
    class _Driver:
        zeroed = False

        def set_zero_position(self) -> None:
            self.zeroed = True

    async def set_zero(robot: _Driver) -> None:
        robot.set_zero_position()

    calibration = RobotZeroCalibration[Any](instructions="Move to the rest pose.", set_zero=set_zero)
    driver = _Driver()

    async def run() -> None:
        await calibration.set_zero(driver)

    asyncio.run(run())

    assert driver.zeroed


@pytest.mark.parametrize(
    ("instructions", "zero_tolerance_deg", "message"),
    [
        ("  ", 5.0, "instructions must not be empty"),
        ("Move to the rest pose.", 0.0, "zero_tolerance_deg must be a finite positive value"),
        ("Move to the rest pose.", float("nan"), "zero_tolerance_deg must be a finite positive value"),
    ],
)
def test_zero_calibration_rejects_invalid_values(instructions: str, zero_tolerance_deg: float, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        RobotZeroCalibration(
            instructions=instructions, set_zero=_noop_calibration_step, zero_tolerance_deg=zero_tolerance_deg
        )


_SCENES = (
    SimulationScene(id="pick", display_name="Pick & Place", description="One block and a target"),
    SimulationScene(id="fold", display_name="Fold"),
)


class SimPayload(BaseModel):
    """A simulated robot's payload whose owner name is derived, not a ``name`` field."""

    sim_id: str
    host: str = "127.0.0.1"


def _sim_argv(*, scene: str, owner_name: str, seed: int | None) -> list[str]:
    argv = ["sim", "start", "--scene", scene, f"--name={owner_name}"]
    return argv if seed is None else [*argv, "--seed", str(seed)]


def _launch(**kwargs: Any) -> SimulationLaunch[SimPayload]:
    options: dict[str, Any] = {
        "scenes": _SCENES,
        "default_scene": "pick",
        "payload_owner_name": lambda payload: f"sim-{payload.sim_id}",
        "build_argv": _sim_argv,
        "max_seed": 2**32 - 1,
    }
    return SimulationLaunch[SimPayload](**(options | kwargs))


def test_definition_defaults_to_no_simulation() -> None:
    definition = RobotCatalogDefinition(type="Test_Follower", display_name="Test Follower", role="follower")

    assert definition.simulation is None


def test_definition_accepts_a_simulation_launch() -> None:
    launch = _launch(default_scene="fold", labels=("2 arms", "twin"))
    definition = RobotCatalogDefinition[SimPayload](
        type="Sim_Follower", display_name="Sim Follower", role="follower", robot_payload=SimPayload, simulation=launch
    )

    assert definition.simulation is launch
    assert [scene.id for scene in launch.scenes] == ["pick", "fold"]
    assert launch.scenes[1].description == ""
    assert launch.labels == ("2 arms", "twin")


def test_simulation_launch_resolves_the_owner_name_from_the_payload() -> None:
    launch = _launch()
    payload = SimPayload(sim_id="7")

    assert launch.owner_name(payload) == "sim-7"
    assert launch.argv("fold", payload) == ["sim", "start", "--scene", "fold", "--name=sim-7"]
    assert launch.argv("pick", payload, seed=2**32 - 1)[-2:] == ["--seed", "4294967295"]
    assert launch.argv("pick", payload, seed=0)[-1] == "0"


def test_each_launch_has_its_own_seed_range() -> None:
    small, unseeded = _launch(max_seed=99), _launch(max_seed=None)
    payload = SimPayload(sim_id="7")

    assert small.argv("pick", payload, seed=99)[-1] == "99"
    with pytest.raises(ValueError, match="seed must be an integer from 0 to 99, got 100"):
        small.argv("pick", payload, seed=100)
    assert _launch().argv("pick", payload, seed=100)[-1] == "100"
    assert unseeded.argv("pick", payload) == ["sim", "start", "--scene", "pick", "--name=sim-7"]
    with pytest.raises(ValueError, match="this simulation takes no seed"):
        unseeded.argv("pick", payload, seed=0)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"scenes": ()}, "scenes must not be empty"),
        ({"scenes": (*_SCENES, SimulationScene(id="pick", display_name="Again"))}, r"duplicates \['pick'\]"),
        ({"default_scene": "stack"}, "default_scene 'stack' is not one of the scenes"),
        ({"max_seed": -1}, "max_seed must be a non-negative integer or None"),
        ({"max_seed": 1.5}, "max_seed must be a non-negative integer or None"),
        ({"labels": ("1 arm", " ")}, "labels must not be blank"),
    ],
)
def test_simulation_launch_rejects_invalid_definitions(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _launch(**kwargs)


@pytest.mark.parametrize(
    ("scene_id", "display_name", "message"),
    [(" ", "Pick", "scene id must not be empty"), ("pick", "", "scene 'pick' needs a display_name")],
)
def test_simulation_scene_rejects_blank_names(scene_id: str, display_name: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        SimulationScene(id=scene_id, display_name=display_name)


@pytest.mark.parametrize(
    ("scene", "sim_id", "seed", "message"),
    [
        ("stack", "7", None, "Unknown scene 'stack'"),
        ("pick", " ", None, "payload_owner_name returned an empty owner name"),
        ("pick", "7", -1, "seed must be an integer from 0 to 4294967295"),
        ("pick", "7", 2**32, "seed must be an integer from 0 to 4294967295"),
        ("pick", "7", True, "seed must be an integer"),
        ("pick", "7", 1.5, "seed must be an integer"),
    ],
)
def test_simulation_launch_argv_rejects_invalid_requests(scene: str, sim_id: str, seed: Any, message: str) -> None:
    launch = _launch(payload_owner_name=lambda payload: payload.sim_id)

    with pytest.raises(ValueError, match=message):
        launch.argv(scene, SimPayload(sim_id=sim_id), seed)


def test_simulation_launch_rejects_an_empty_argv() -> None:
    def no_argv(*, scene: str, owner_name: str, seed: int | None) -> list[str]:
        _ = scene, owner_name, seed
        return []

    launch = _launch(build_argv=no_argv)

    with pytest.raises(ValueError, match="no command line"):
        launch.argv("pick", SimPayload(sim_id="7"))


def test_generic_payload_linked_to_probe() -> None:
    asset = RobotAsset(
        urdf_relative_path=Path("test/model.urdf"),
        packages={"test": Path("test")},
        joint_map={"gripper.pos": ["gripper"]},
    )
    probe = TestProbe()
    definition = RobotCatalogDefinition[TestPayload](
        type="Test_Follower",
        display_name="Test Follower",
        role="follower",
        robot_payload=TestPayload,
        asset=asset,
        probe=probe,
    )

    payload_instance = TestPayload(serial_number="SN-002")
    assert definition.probe is not None
    asyncio.run(definition.probe.identify(payload_instance, None))
    assert probe._last_payload is payload_instance


def test_multiple_definitions() -> None:
    definitions: list[RobotCatalogDefinition] = [
        RobotCatalogDefinition[TestPayload](
            type="RobotA",
            display_name="Robot A",
            role="follower",
            robot_payload=TestPayload,
            probe=TestProbe(),
        ),
        RobotCatalogDefinition[TestPayload](
            type="RobotB",
            display_name="Robot B",
            role="leader",
            robot_payload=TestPayload,
            probe=TestProbe(),
        ),
    ]
    assert len(definitions) == 2
    assert definitions[0].type == "RobotA"
    assert definitions[1].role == "leader"


def test_valid_payload_passes_validation() -> None:
    asset = RobotAsset(
        urdf_relative_path=Path("test/model.urdf"),
        packages={"test": Path("test")},
        joint_map={"gripper.pos": ["gripper"]},
    )
    definition = RobotCatalogDefinition[TestPayload](
        type="Test_Follower",
        display_name="Test Follower",
        role="follower",
        robot_payload=TestPayload,
        asset=asset,
    )

    raw = {"serial_number": "SN-003", "connection_string": "/dev/ttyUSB0"}
    payload_model = definition.robot_payload
    assert payload_model is not None
    validated = payload_model.model_validate(raw)
    assert isinstance(validated, BaseModel)
    assert validated.serial_number == "SN-003"


def test_invalid_payload_raises() -> None:
    definition = RobotCatalogDefinition[TestPayload](
        type="Test_Follower",
        display_name="Test Follower",
        role="follower",
        robot_payload=TestPayload,
    )

    raw = {"connection_string": "/dev/ttyUSB0"}
    payload_model = definition.robot_payload
    assert payload_model is not None
    with pytest.raises(ValidationError):
        payload_model.model_validate(raw)


def test_no_payload_model_returns_raw_dict() -> None:
    definition = RobotCatalogDefinition[TestPayload](
        type="NoPayload",
        display_name="No Payload",
        role="follower",
        robot_payload=None,
    )

    raw = {"some": "data"}
    result = raw if definition.robot_payload is None else definition.robot_payload.model_validate(raw)
    assert result == raw


def test_robot_field_ui_supports_required_option() -> None:
    assert robot_field_ui({"required": True}) == {"x-physicalai-ui": {"required": True}}


def test_robot_field_ui_supports_advanced_configuration_option() -> None:
    assert robot_field_ui({"advanced_configuration": True}) == {
        "x-physicalai-ui": {"advanced_configuration": True},
    }


def test_robot_field_ui_supports_contextual_info() -> None:
    assert robot_field_ui(
        {
            "info": {
                "title": "Calibration file",
                "description": "Provide a calibration JSON exported from the robot.",
                "link_url": "https://example.com/calibration",
                "variant": "help",
            }
        }
    ) == {
        "x-physicalai-ui": {
            "info": {
                "title": "Calibration file",
                "description": "Provide a calibration JSON exported from the robot.",
                "link_url": "https://example.com/calibration",
                "variant": "help",
            }
        }
    }


def test_robot_payload_ui_supports_recursive_items() -> None:
    assert robot_payload_ui(
        [
            {
                "kind": "section",
                "id": "connection",
                "title": "Connection",
                "description": "Pick a detected device or enter one manually.",
                "items": [
                    {"kind": "info", "text": "USB hubs can rename ports after reboot.", "variant": "warning"},
                    {
                        "kind": "connection",
                        "bind": {"connection": "connection_string", "serial_number": "serial_number"},
                    },
                ],
            },
        ],
    ) == {
        "x-physicalai-ui": [
            {
                "kind": "section",
                "id": "connection",
                "title": "Connection",
                "description": "Pick a detected device or enter one manually.",
                "items": [
                    {"kind": "info", "text": "USB hubs can rename ports after reboot.", "variant": "warning"},
                    {
                        "kind": "connection",
                        "bind": {"connection": "connection_string", "serial_number": "serial_number"},
                    },
                ],
            },
        ],
    }


def test_robot_payload_ui_supports_ip_address_items() -> None:
    assert robot_payload_ui(
        [
            {
                "kind": "ip_address",
                "name": "connection_string",
                "identify": True,
                "identify_robot_type": "Trossen_WidowXAI_Follower",
            },
        ],
    ) == {
        "x-physicalai-ui": [
            {
                "kind": "ip_address",
                "name": "connection_string",
                "identify": True,
                "identify_robot_type": "Trossen_WidowXAI_Follower",
            },
        ],
    }


def test_robot_payload_ui_supports_calibration_items() -> None:
    assert robot_payload_ui(
        [
            {
                "kind": "calibration",
                "name": "calibration",
                "label": "Calibration",
            },
        ],
    ) == {
        "x-physicalai-ui": [
            {
                "kind": "calibration",
                "name": "calibration",
                "label": "Calibration",
            },
        ],
    }


def test_robot_payload_ui_supports_info_attribute_on_items() -> None:
    assert robot_payload_ui(
        [
            {
                "kind": "field",
                "name": "connection_string",
                "info": {
                    "description": "Set the robot endpoint address.",
                    "link_url": "https://example.com/network-setup",
                },
            },
        ],
    ) == {
        "x-physicalai-ui": [
            {
                "kind": "field",
                "name": "connection_string",
                "info": {
                    "description": "Set the robot endpoint address.",
                    "link_url": "https://example.com/network-setup",
                },
            },
        ],
    }


def test_validate_robot_payload_ui_accepts_field_level_info() -> None:
    class Payload(BaseModel):
        connection_string: str = Field(
            default="",
            json_schema_extra=robot_field_ui({"info": {"description": "Connection string for the robot."}}),
        )

    validate_robot_payload_ui(Payload)


def test_validate_robot_payload_ui_accepts_nested_item_lists() -> None:
    class ConnectionPayload(BaseModel):
        connection_string: str
        serial_number: str

        model_config = ConfigDict(
            json_schema_extra=robot_payload_ui(
                [
                    {
                        "kind": "connection",
                        "bind": {"connection": "connection_string", "serial_number": "serial_number"},
                    },
                ],
            ),
        )

    class RobotPayload(BaseModel):
        arm: ConnectionPayload

        model_config = ConfigDict(json_schema_extra=robot_payload_ui([{"kind": "field", "name": "arm"}]))

    validate_robot_payload_ui(RobotPayload)


def test_validate_robot_payload_ui_ignores_field_options() -> None:
    class Payload(BaseModel):
        id: str | None = Field(default=None, json_schema_extra=robot_field_ui({"required": True}))

    validate_robot_payload_ui(Payload)


@pytest.mark.parametrize(
    ("items", "message"),
    [
        ({"groups": {}}, "must be a list of items"),
        ([{"kind": "field", "name": "missing"}], "must reference an existing payload field"),
        ([{"kind": "connection", "bind": {"connection": "port"}}], "must reference a string payload field"),
        ([{"kind": "ip_address", "name": "missing"}], "must reference an existing payload field"),
        ([{"kind": "ip_address", "name": "port"}], "must reference a string payload field"),
        ([{"kind": "calibration", "name": "missing"}], "must reference an existing payload field"),
        ([{"kind": "calibration", "name": "connection_string"}], "must reference an object payload field"),
        ([{"kind": "field", "name": "connection_string", "info": "bad"}], "info must be an object"),
        (
            [{"kind": "field", "name": "connection_string", "info": {"title": "Info"}}],
            "info.description must be a non-empty string",
        ),
        (
            [
                {
                    "kind": "field",
                    "name": "connection_string",
                    "info": {"description": "text", "variant": "warning"},
                }
            ],
            "info.variant must be one of: info, help",
        ),
        (
            [
                {"kind": "field", "name": "connection_string"},
                {"kind": "connection", "bind": {"connection": "connection_string"}},
            ],
            "owned more than once",
        ),
        (
            [
                {"kind": "field", "name": "connection_string"},
                {"kind": "ip_address", "name": "connection_string"},
            ],
            "owned more than once",
        ),
        (
            [
                {"kind": "field", "name": "connection_string"},
                {"kind": "calibration", "name": "connection_string"},
            ],
            "must reference an object payload field",
        ),
        ([{"kind": "info", "text": "Hello", "variant": "error"}], "variant must be 'info' or 'warning'"),
        (
            [{"kind": "connection", "identify": "yes", "bind": {"connection": "connection_string"}}],
            "identify must be a boolean",
        ),
    ],
)
def test_validate_robot_payload_ui_rejects_invalid_metadata(items: object, message: str) -> None:
    class InvalidPayload(BaseModel):
        connection_string: str
        port: int
        calibration: dict[str, int]

        model_config = ConfigDict(json_schema_extra={"x-physicalai-ui": items})

    with pytest.raises(ValueError, match=message):
        validate_robot_payload_ui(InvalidPayload)


def test_validate_robot_payload_ui_rejects_invalid_field_info() -> None:
    class InvalidPayload(BaseModel):
        connection_string: str = Field(
            default="",
            json_schema_extra=robot_field_ui(cast(Any, {"info": {"title": "Broken"}})),
        )

    with pytest.raises(ValueError, match="info.description must be a non-empty string"):
        validate_robot_payload_ui(InvalidPayload)
