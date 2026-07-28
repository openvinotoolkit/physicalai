# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# ruff:file-ignore[undocumented-public-module, undocumented-public-class, undocumented-public-method, undocumented-public-init, assert, magic-value-comparison]

"""Construction round-trips for camera ``@export_config`` wiring."""

from __future__ import annotations

import json
import sys
from collections.abc import Generator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from physicalai.capture.camera import Camera, ColorMode
from physicalai.config import ComponentConfig, instantiate, is_config_exportable, to_config


def _assert_construction_round_trip(camera: object) -> dict[str, Any]:
    assert is_config_exportable(camera)
    assert isinstance(camera, Camera)
    config = to_config(camera)
    wire: dict[str, Any] = json.loads(json.dumps(config))
    restored = instantiate(wire)
    assert type(restored) is type(camera)
    assert to_config(restored) == wire
    assert not restored.is_connected  # type: ignore[union-attr]
    return wire


class TestUVCCameraComponentConfig:
    def test_v4l2_backend_round_trip(self) -> None:
        from physicalai.capture import UVCCamera

        camera = UVCCamera(device=0, width=640, height=480, fps=30, backend="v4l2")
        wire = _assert_construction_round_trip(camera)
        assert wire["class_path"] == "physicalai.capture.UVCCamera"
        assert wire["init_args"]["device"] == 0
        assert wire["init_args"]["backend"] == "v4l2"
        assert wire["init_args"]["width"] == 640

    def test_color_mode_enum_round_trips_as_string(self) -> None:
        from physicalai.capture import UVCCamera

        camera = UVCCamera(device=1, color_mode=ColorMode.BGR, backend="v4l2")
        wire = _assert_construction_round_trip(camera)
        assert wire["init_args"]["color_mode"] == "bgr"


@pytest.fixture
def mock_pyrealsense2() -> Generator[MagicMock, None, None]:
    mock_rs = MagicMock()
    with patch.dict(sys.modules, {"pyrealsense2": mock_rs}):
        # Force reload if already imported with real/missing SDK.
        sys.modules.pop("physicalai.capture.cameras.realsense._camera", None)
        sys.modules.pop("physicalai.capture.cameras.realsense", None)
        yield mock_rs


class TestRealSenseCameraComponentConfig:
    def test_round_trip(self, mock_pyrealsense2: MagicMock) -> None:
        from physicalai.capture.cameras.realsense import RealSenseCamera

        camera = RealSenseCamera(serial_number="SN123", width=640, height=480, fps=30)
        wire = _assert_construction_round_trip(camera)
        assert wire["class_path"] == "physicalai.capture.RealSenseCamera"
        assert wire["init_args"] == {
            "serial_number": "SN123",
            "width": 640,
            "height": 480,
            "fps": 30,
        }


@pytest.fixture
def mock_pypylon() -> Generator[None, None, None]:
    mock_pylon = MagicMock()
    mock_genicam = MagicMock()
    with (
        patch.dict(sys.modules, {"pypylon": MagicMock(), "pypylon.pylon": mock_pylon, "pypylon.genicam": mock_genicam}),
        patch.dict(sys.modules, {"cv2": MagicMock()}),
    ):
        sys.modules.pop("physicalai.capture.cameras.basler._camera", None)
        sys.modules.pop("physicalai.capture.cameras.basler", None)
        yield


class TestBaslerCameraComponentConfig:
    def test_round_trip(self, mock_pypylon: None) -> None:
        from physicalai.capture.cameras.basler import BaslerCamera

        camera = BaslerCamera(serial_number="BAS-1", fps=15, width=800, height=600)
        wire = _assert_construction_round_trip(camera)
        assert wire["class_path"] == "physicalai.capture.BaslerCamera"
        assert wire["init_args"]["serial_number"] == "BAS-1"
        assert wire["init_args"]["fps"] == 15
        assert wire["init_args"]["width"] == 800
        assert wire["init_args"]["height"] == 600


# ---------------------------------------------------------------------------
# SharedCamera (construction recipe only — no publisher / SHM state)
# ---------------------------------------------------------------------------


class TestSharedCameraComponentConfig:
    def test_spawn_recipe_round_trip(self) -> None:
        from physicalai.capture import SharedCamera

        camera = SharedCamera(
            camera={
                "class_path": "physicalai.capture.UVCCamera",
                "init_args": {"device": "/dev/video0", "width": 640, "height": 480, "fps": 30, "backend": "v4l2"},
            },
            color_mode=ColorMode.BGR,
            zero_copy=True,
            validate_on_connect=True,
            idle_timeout=1.5,
        )
        wire = _assert_construction_round_trip(camera)
        assert wire["class_path"] == "physicalai.capture.SharedCamera"
        assert wire["init_args"]["color_mode"] == "bgr"
        assert wire["init_args"]["zero_copy"] is True
        assert wire["init_args"]["validate_on_connect"] is True
        assert wire["init_args"]["idle_timeout"] == 1.5
        # service_name was derived, not caller-supplied — omitted from recipe.
        assert "service_name" not in wire["init_args"]
        nested = wire["init_args"]["camera"]
        assert isinstance(nested, dict)
        assert nested["class_path"] == "physicalai.capture.UVCCamera"
        assert nested["init_args"]["device"] == "/dev/video0"
        assert "_publisher" not in wire["init_args"]
        assert "_connected" not in wire["init_args"]
        assert "_node" not in wire["init_args"]

    def test_defining_module_camera_path_round_trips_as_given(self) -> None:
        from physicalai.capture import SharedCamera

        defining = "physicalai.capture.cameras.uvc._camera.UVCCamera"
        camera = SharedCamera(
            camera={
                "class_path": defining,
                "init_args": {"device": 0, "backend": "v4l2"},
            },
        )
        wire = _assert_construction_round_trip(camera)
        nested = wire["init_args"]["camera"]
        assert isinstance(nested, dict)
        # Stored as written; the terminal class name still derives one service name.
        assert nested["class_path"] == defining
        assert camera._service_name == "physicalai/camera/UVCCamera/0/frame"

    def test_attach_only_round_trip(self) -> None:
        from physicalai.capture import SharedCamera

        camera = SharedCamera(camera=None, service_name="physicalai/camera/UVCCamera/0/frame")
        wire = _assert_construction_round_trip(camera)
        assert wire["class_path"] == "physicalai.capture.SharedCamera"
        assert wire["init_args"]["camera"] is None
        assert wire["init_args"]["service_name"] == "physicalai/camera/UVCCamera/0/frame"

    def test_nested_recipe_is_not_instantiated(self) -> None:
        from physicalai.capture import SharedCamera

        config: ComponentConfig = {
            "class_path": "physicalai.capture.SharedCamera",
            "init_args": {
                "camera": {
                    "class_path": "physicalai.capture.UVCCamera",
                    "init_args": {"device": 0, "backend": "v4l2"},
                },
                "service_name": "physicalai/camera/UVCCamera/0/frame",
            },
        }
        restored = instantiate(config)
        assert isinstance(restored, SharedCamera)
        # The declared config arg stays a mapping — no UVCCamera is built here.
        assert restored._camera == config["init_args"]["camera"]
