from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import BaseModel, ValidationError


class _FakeRegistry:
    def __init__(self) -> None:
        self.definitions: list = []

    def register_robot(self, definition: object) -> None:
        self.definitions.append(definition)

    def register_many(self, definitions: list[object]) -> None:
        self.definitions.extend(definitions)


class _StubFactory:
    def __init__(self, port: str | None = "/dev/ttyACM0") -> None:
        self._port = port

    async def find_port(self, port_info: object) -> str | None:
        _ = port_info
        return self._port


class _StubRobot:
    def __init__(self, payload: object) -> None:
        self.payload = payload


class TestRegistration:
    def test_register_plugin(self) -> None:
        from physicalai_bimanual_so101_plugin.studio_catalog import (
            register_physicalai_studio_plugin,
        )

        registry = _FakeRegistry()
        register_physicalai_studio_plugin(registry)
        assert len(registry.definitions) == 2

    def test_definitions_have_expected_types(self) -> None:
        from physicalai_bimanual_so101_plugin.studio_catalog import _definitions

        types = {d.type for d in _definitions()}
        assert types == {"BimanualSO101_Follower", "BimanualSO101_Leader"}

    def test_follower_structure(self) -> None:
        from physicalai_bimanual_so101_plugin.studio_catalog import BimanualSO101Payload, _definitions

        follower = next(d for d in _definitions() if d.type == "BimanualSO101_Follower")

        assert follower.display_name == "Bimanual SO-101 Follower"
        assert follower.role == "follower"
        assert follower.asset is not None
        assert follower.asset.urdf_relative_path == Path("so101_dual/so101_dual.urdf")
        assert follower.asset.root_resolver is not None
        assert (follower.asset.root_resolver() / follower.asset.urdf_relative_path).is_file()
        assert callable(follower.robot_builder)
        assert follower.robot_payload is BimanualSO101Payload
        assert follower.probe is not None
        assert follower.asset.packages == {"so101_dual": Path("so101_dual")}
        assert len(follower.asset.joint_map) == 12

    def test_leader_structure(self) -> None:
        from physicalai_bimanual_so101_plugin.studio_catalog import BimanualSO101Payload, _definitions

        leader = next(d for d in _definitions() if d.type == "BimanualSO101_Leader")

        assert leader.display_name == "Bimanual SO-101 Leader"
        assert leader.role == "leader"
        assert leader.asset is not None
        assert leader.asset.urdf_relative_path == Path("so101_dual/so101_dual.urdf")
        assert leader.asset.root_resolver is not None
        assert (leader.asset.root_resolver() / leader.asset.urdf_relative_path).is_file()
        assert callable(leader.robot_builder)
        assert leader.robot_payload is BimanualSO101Payload


class TestPayload:
    def test_payload_defaults(self) -> None:
        from physicalai_bimanual_so101_plugin.studio_catalog import (
            BimanualSO101Payload,
        )

        payload = BimanualSO101Payload(
            left_serial_number="SN-LEFT-001",
            right_serial_number="SN-RIGHT-001",
        )
        assert payload.left_serial_number == "SN-LEFT-001"
        assert payload.right_serial_number == "SN-RIGHT-001"
        assert payload.baudrate == 1_000_000
        assert payload.role == "follower"
        assert payload.disable_torque_on_disconnect is True
        assert payload.left_calibration is None
        assert payload.right_calibration is None

    def test_payload_requires_both_serials(self) -> None:
        from physicalai_bimanual_so101_plugin.studio_catalog import (
            BimanualSO101Payload,
        )

        with pytest.raises(ValidationError):
            BimanualSO101Payload()

    def test_payload_with_calibrations(self) -> None:
        from physicalai_bimanual_so101_plugin.studio_catalog import (
            BimanualSO101Payload,
        )

        payload = BimanualSO101Payload(
            left_serial_number="SN-L",
            right_serial_number="SN-R",
            left_calibration={
                "shoulder_pan": {"id": 1, "drive_mode": 0, "homing_offset": 0, "range_min": 0, "range_max": 4095},
                "shoulder_lift": {"id": 2, "drive_mode": 0, "homing_offset": 0, "range_min": 0, "range_max": 4095},
                "elbow_flex": {"id": 3, "drive_mode": 0, "homing_offset": 0, "range_min": 0, "range_max": 4095},
                "wrist_flex": {"id": 4, "drive_mode": 0, "homing_offset": 0, "range_min": 0, "range_max": 4095},
                "wrist_roll": {"id": 5, "drive_mode": 0, "homing_offset": 0, "range_min": 0, "range_max": 4095},
                "gripper": {"id": 6, "drive_mode": 0, "homing_offset": 0, "range_min": 0, "range_max": 4095},
            },
            right_calibration={
                "shoulder_pan": {"id": 7, "drive_mode": 0, "homing_offset": 0, "range_min": 0, "range_max": 4095},
                "shoulder_lift": {"id": 8, "drive_mode": 0, "homing_offset": 0, "range_min": 0, "range_max": 4095},
                "elbow_flex": {"id": 9, "drive_mode": 0, "homing_offset": 0, "range_min": 0, "range_max": 4095},
                "wrist_flex": {"id": 10, "drive_mode": 0, "homing_offset": 0, "range_min": 0, "range_max": 4095},
                "wrist_roll": {"id": 11, "drive_mode": 0, "homing_offset": 0, "range_min": 0, "range_max": 4095},
                "gripper": {"id": 12, "drive_mode": 0, "homing_offset": 0, "range_min": 0, "range_max": 4095},
            },
        )
        assert payload.left_calibration is not None
        assert payload.right_calibration is not None
        assert payload.left_calibration["shoulder_pan"].id == 1
        assert payload.right_calibration["gripper"].id == 12

    def test_payload_model_rebuild(self) -> None:
        from physicalai_bimanual_so101_plugin.studio_catalog import BimanualSO101Payload

        BimanualSO101Payload.model_rebuild(raise_errors=True)

    def test_payload_schema_exposes_two_calibration_upload_controls(self) -> None:
        from physicalai_studio_plugin import validate_robot_payload_ui

        from physicalai_bimanual_so101_plugin.studio_catalog import BimanualSO101Payload

        validate_robot_payload_ui(BimanualSO101Payload)
        schema = BimanualSO101Payload.model_json_schema()
        assert schema["x-physicalai-ui"] == [
            {
                "kind": "calibration",
                "name": "left_calibration",
                "info": {
                    "description": "Upload left-arm SO101 calibration JSON (joint-name keyed).",
                },
            },
            {
                "kind": "calibration",
                "name": "right_calibration",
                "info": {
                    "description": "Upload right-arm SO101 calibration JSON (joint-name keyed).",
                },
            },
        ]


class TestURDF:
    def test_urdf_path_exists(self) -> None:
        from physicalai_bimanual_so101_plugin import get_urdf_path

        path = get_urdf_path()
        assert path.exists()
        urdf_file = path / "so101_dual" / "so101_dual.urdf"
        assert urdf_file.exists(), f"URDF not found at {urdf_file}"
        assert urdf_file.stat().st_size > 0

    def test_asset_root_resolver(self) -> None:
        from physicalai_bimanual_so101_plugin.studio_catalog import (
            _get_bimanual_urdf_root,
        )

        root = _get_bimanual_urdf_root()
        assert isinstance(root, Path)
        assert root.exists()


class TestBuilder:
    @pytest.mark.anyio
    async def test_build_uncalibrated(self) -> None:
        from physicalai_bimanual_so101_plugin.studio_catalog import (
            BimanualSO101Payload,
            _build_bimanual_driver,
        )

        payload = BimanualSO101Payload(
            left_serial_number="SN-L",
            right_serial_number="SN-R",
        )
        robot = _StubRobot(payload)
        factory = _StubFactory(port="/dev/ttyACM0")
        driver = await _build_bimanual_driver(robot, factory)
        assert driver is not None
        assert driver.is_connected() is False

    @pytest.mark.anyio
    async def test_build_from_dict(self) -> None:
        from physicalai_bimanual_so101_plugin.studio_catalog import (
            _build_bimanual_driver,
        )

        payload = {
            "left_serial_number": "SN-L",
            "right_serial_number": "SN-R",
            "role": "follower",
        }
        robot = _StubRobot(payload)
        factory = _StubFactory(port="/dev/ttyACM1")
        driver = await _build_bimanual_driver(robot, factory)
        assert driver is not None

    @pytest.mark.anyio
    async def test_build_from_foreign_pydantic_model(self) -> None:
        from physicalai_bimanual_so101_plugin.studio_catalog import _build_bimanual_driver

        class _ForeignPayload(BaseModel):
            left_serial_number: str
            right_serial_number: str

        robot = _StubRobot(_ForeignPayload(left_serial_number="SN-L", right_serial_number="SN-R"))
        driver = await _build_bimanual_driver(robot, _StubFactory())
        assert cast(Any, driver).role == "follower"

    @pytest.mark.anyio
    async def test_catalog_builders_enforce_definition_roles(self) -> None:
        from physicalai_bimanual_so101_plugin.studio_catalog import BimanualSO101Payload, _definitions

        payload = BimanualSO101Payload(
            left_serial_number="SN-L",
            right_serial_number="SN-R",
            role="leader",
        )
        robot = _StubRobot(payload)
        factory = _StubFactory()
        definitions = {definition.role: definition for definition in _definitions()}
        follower_builder = definitions["follower"].robot_builder
        leader_builder = definitions["leader"].robot_builder
        assert follower_builder is not None
        assert leader_builder is not None

        follower = await follower_builder(robot, factory)
        leader = await leader_builder(robot, factory)

        assert cast(Any, follower).role == "follower"
        assert cast(Any, leader).role == "leader"

    @pytest.mark.anyio
    async def test_build_left_port_not_found(self) -> None:
        from physicalai_bimanual_so101_plugin.studio_catalog import (
            BimanualSO101Payload,
            _build_bimanual_driver,
        )

        payload = BimanualSO101Payload(
            left_serial_number="SN-MISSING",
            right_serial_number="SN-R",
        )
        robot = _StubRobot(payload)
        factory = _StubFactory(port=None)
        with pytest.raises(RuntimeError, match="Left arm not found"):
            await _build_bimanual_driver(robot, factory)

    @pytest.mark.anyio
    async def test_build_right_port_not_found(self) -> None:
        from physicalai_bimanual_so101_plugin.studio_catalog import (
            BimanualSO101Payload,
            _build_bimanual_driver,
        )

        class _SelectiveFactory:
            async def find_port(self, port_info: object) -> str | None:
                serial_number = getattr(port_info, "serial_number", None)
                return "/dev/ttyACM0" if serial_number == "SN-L" else None

            async def find_so101_port(self, robot: object) -> str:
                return "/dev/ttyACM0"

        payload = BimanualSO101Payload(
            left_serial_number="SN-L",
            right_serial_number="SN-MISSING",
        )
        robot = _StubRobot(payload)
        factory = _SelectiveFactory()
        with pytest.raises(RuntimeError, match="Right arm not found"):
            await _build_bimanual_driver(robot, factory)

    @pytest.mark.anyio
    async def test_build_mixed_calibration_raises(self) -> None:
        from physicalai_bimanual_so101_plugin.studio_catalog import (
            BimanualSO101Payload,
            _build_bimanual_driver,
        )

        payload = BimanualSO101Payload(
            left_serial_number="SN-L",
            right_serial_number="SN-R",
            left_calibration={
                "shoulder_pan": {"id": 1, "drive_mode": 0, "homing_offset": 0, "range_min": 0, "range_max": 4095},
            },
            right_calibration=None,
        )
        robot = _StubRobot(payload)
        factory = _StubFactory()
        with pytest.raises(ValueError, match="Both arms must have calibrations"):
            await _build_bimanual_driver(robot, factory)


class TestProbe:
    @pytest.mark.anyio
    async def test_is_online_no_manager(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from physicalai_bimanual_so101_plugin.studio_catalog import (
            BimanualSO101Payload,
            BimanualSO101Probe,
        )

        class _ListPortsStub:
            @staticmethod
            def comports() -> list[object]:
                return []

        monkeypatch.setattr(
            "physicalai_bimanual_so101_plugin.studio_catalog.list_ports",
            _ListPortsStub,
        )

        probe = BimanualSO101Probe()
        payload = BimanualSO101Payload(left_serial_number="SN-L", right_serial_number="SN-R")
        result = await probe.is_online(payload)
        assert result is False

    def test_joint_map(self) -> None:
        from physicalai_bimanual_so101_plugin.studio_catalog import (
            _BIMANUAL_SO101_TO_URDF,
        )

        assert len(_BIMANUAL_SO101_TO_URDF) == 12
        assert "left_shoulder_pan.pos" in _BIMANUAL_SO101_TO_URDF
        assert "right_shoulder_pan.pos" in _BIMANUAL_SO101_TO_URDF
        assert _BIMANUAL_SO101_TO_URDF["left_gripper.pos"] == ["left_gripper"]
        assert _BIMANUAL_SO101_TO_URDF["right_gripper.pos"] == ["right_gripper"]
