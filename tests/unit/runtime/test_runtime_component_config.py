# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# ruff:file-ignore[undocumented-public-module, undocumented-public-class, undocumented-public-method, undocumented-public-init, assert, magic-value-comparison]

"""Construction round-trips for runtime ``@export_config`` wiring (steps 6–7)."""

from __future__ import annotations

import json
import sys
from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from physicalai.config import ComponentConfigError, instantiate, is_config_exportable, to_config

# SO101 imports scservo_sdk at module load; keep unit tests hardware-free.
# Use setdefault — not patch.dict(sys.modules) — because patch.dict restores by
# clearing sys.modules and drops modules imported while the patch is active
# (e.g. jsonargparse ``build_parser``), which poisons later YAML config loads.
sys.modules.setdefault("scservo_sdk", MagicMock())


def _assert_construction_round_trip(value: object) -> dict[str, Any]:
    assert is_config_exportable(value)
    config = to_config(value)
    wire: dict[str, Any] = json.loads(json.dumps(config))
    restored = instantiate(wire)
    assert type(restored) is type(value)
    assert to_config(restored) == wire
    return wire


def _make_export_dir(tmp_path: Path, *, backend: str = "openvino") -> Path:
    export_dir = tmp_path / "exports"
    export_dir.mkdir(exist_ok=True)
    artifact = "act.xml" if backend == "openvino" else f"act.{backend}"
    manifest = {
        "format": "policy_package",
        "version": "1.0",
        "policy": {
            "name": "act",
            "source": {"class_path": "physicalai.policies.act.ACT"},
        },
        "model": {
            "artifacts": {backend: artifact},
            "runner": {"class_path": "physicalai.inference.runners.SinglePass", "init_args": {}},
        },
    }
    with (export_dir / "manifest.json").open("w") as f:
        json.dump(manifest, f)
    (export_dir / artifact).touch()
    if artifact.endswith(".xml"):
        (export_dir / artifact.replace(".xml", ".bin")).touch()
    return export_dir


@pytest.fixture
def mock_adapter() -> MagicMock:
    adapter = MagicMock()
    adapter.input_names = []
    adapter.output_names = []
    adapter.default_device.return_value = "cpu"
    return adapter


@pytest.fixture
def _patch_adapter(mock_adapter: MagicMock) -> Generator[MagicMock, None, None]:
    with patch("physicalai.inference.model.get_adapter", return_value=mock_adapter):
        yield mock_adapter


@pytest.fixture
def inference_model(tmp_path: Path, _patch_adapter: MagicMock) -> Any:
    from physicalai.inference import InferenceModel

    return InferenceModel(export_dir=_make_export_dir(tmp_path), backend="openvino", device="cpu")


# ---------------------------------------------------------------------------
# Smoothers / queues / execution
# ---------------------------------------------------------------------------


class TestSmootherComponentConfig:
    def test_lerp_round_trip(self) -> None:
        from physicalai.runtime import LerpSmoother

        wire = _assert_construction_round_trip(LerpSmoother(duration_frames=7))
        assert wire["class_path"] == "physicalai.runtime.LerpSmoother"
        assert wire["init_args"] == {"duration_frames": 7}

    def test_lerp_default_omitted(self) -> None:
        from physicalai.runtime import LerpSmoother

        wire = _assert_construction_round_trip(LerpSmoother())
        assert wire["init_args"] == {}

    def test_replace_round_trip(self) -> None:
        from physicalai.runtime import ReplaceSmoother

        wire = _assert_construction_round_trip(ReplaceSmoother())
        assert wire["class_path"] == "physicalai.runtime.ReplaceSmoother"
        assert wire["init_args"] == {}


class TestActionQueueComponentConfig:
    def test_chunked_bare_restores_replace_smoother(self) -> None:
        from physicalai.runtime import ChunkedActionQueue, ReplaceSmoother

        queue = ChunkedActionQueue()
        wire = _assert_construction_round_trip(queue)
        assert wire["class_path"] == "physicalai.runtime.ChunkedActionQueue"
        assert wire["init_args"] == {}
        restored = instantiate(wire)
        assert isinstance(restored._smoother, ReplaceSmoother)  # type: ignore[attr-defined]

    def test_chunked_explicit_lerp_nested(self) -> None:
        from physicalai.runtime import ChunkedActionQueue, LerpSmoother

        queue = ChunkedActionQueue(smoother=LerpSmoother(duration_frames=3))
        wire = _assert_construction_round_trip(queue)
        assert wire["init_args"]["smoother"] == {
            "class_path": "physicalai.runtime.LerpSmoother",
            "init_args": {"duration_frames": 3},
        }

    def test_chunked_undecorated_smoother_fails(self) -> None:
        from physicalai.runtime import ChunkedActionQueue
        from physicalai.runtime.smoothers import ChunkSmoother

        class UndecoratedSmoother(ChunkSmoother):
            def merge(self, remaining: Any, incoming: Any) -> Any:  # noqa: ANN401
                return incoming

        queue = ChunkedActionQueue(smoother=UndecoratedSmoother())
        with pytest.raises(ComponentConfigError, match=r"init_args\.smoother"):
            to_config(queue)

    def test_rtc_action_queue_round_trip(self) -> None:
        from physicalai.runtime import RTCActionQueue

        wire = _assert_construction_round_trip(RTCActionQueue())
        assert wire["class_path"] == "physicalai.runtime.RTCActionQueue"
        assert wire["init_args"] == {}


class TestExecutionComponentConfig:
    def test_sync_default_omitted(self) -> None:
        from physicalai.runtime import SyncExecution

        wire = _assert_construction_round_trip(SyncExecution())
        assert wire["class_path"] == "physicalai.runtime.SyncExecution"
        assert wire["init_args"] == {}

    def test_sync_explicit_threshold(self) -> None:
        from physicalai.runtime import SyncExecution

        wire = _assert_construction_round_trip(SyncExecution(request_threshold=0.25))
        assert wire["init_args"] == {"request_threshold": 0.25}

    def test_async_round_trip(self) -> None:
        from physicalai.runtime import AsyncExecution

        wire = _assert_construction_round_trip(
            AsyncExecution(request_threshold=0.1, watchdog_timeout_s=10.0)
        )
        assert wire["class_path"] == "physicalai.runtime.AsyncExecution"
        assert wire["init_args"] == {"request_threshold": 0.1, "watchdog_timeout_s": 10.0}

    def test_rtc_scalar_args_round_trip(self) -> None:
        from physicalai.runtime import RTCExecution

        wire = _assert_construction_round_trip(
            RTCExecution(chunk_size=50, execution_horizon=10, fps=30.0, max_guidance_weight=3.0)
        )
        assert wire["class_path"] == "physicalai.runtime.RTCExecution"
        assert wire["init_args"] == {
            "chunk_size": 50,
            "execution_horizon": 10,
            "fps": 30.0,
            "max_guidance_weight": 3.0,
        }

    def test_rtc_live_latency_tracker_fails(self) -> None:
        from physicalai.runtime import RTCExecution

        execution = RTCExecution(latency_tracker=MagicMock())
        with pytest.raises(ComponentConfigError, match=r"init_args\.latency_tracker"):
            to_config(execution)

    def test_rtc_live_postprocessors_fail(self) -> None:
        from physicalai.runtime import RTCExecution

        execution = RTCExecution(postprocessors=[MagicMock()])
        with pytest.raises(ComponentConfigError, match=r"init_args\.postprocessors"):
            to_config(execution)


# ---------------------------------------------------------------------------
# PolicySource
# ---------------------------------------------------------------------------


class TestPolicySourceComponentConfig:
    def test_model_only_omits_execution_and_queue(
        self, inference_model: Any, _patch_adapter: MagicMock
    ) -> None:
        from physicalai.runtime import ChunkedActionQueue, LerpSmoother, PolicySource, SyncExecution

        source = PolicySource(model=inference_model, task="pick cube")
        wire = _assert_construction_round_trip(source)
        assert wire["class_path"] == "physicalai.runtime.PolicySource"
        assert "execution" not in wire["init_args"]
        assert "action_queue" not in wire["init_args"]
        assert wire["init_args"]["task"] == "pick cube"
        assert wire["init_args"]["model"]["class_path"] == "physicalai.inference.InferenceModel"

        restored = instantiate(wire)
        assert isinstance(restored._execution, SyncExecution)  # type: ignore[attr-defined]
        assert isinstance(restored._action_queue, ChunkedActionQueue)  # type: ignore[attr-defined]
        assert isinstance(restored._action_queue._smoother, LerpSmoother)  # type: ignore[attr-defined]

    def test_explicit_nested_graph(self, inference_model: Any, _patch_adapter: MagicMock) -> None:
        from physicalai.runtime import (
            AsyncExecution,
            ChunkedActionQueue,
            PolicySource,
            ReplaceSmoother,
        )

        source = PolicySource(
            model=inference_model,
            execution=AsyncExecution(request_threshold=0.2),
            action_queue=ChunkedActionQueue(smoother=ReplaceSmoother()),
            task="stack",
        )
        wire = _assert_construction_round_trip(source)
        assert wire["init_args"]["execution"]["class_path"] == "physicalai.runtime.AsyncExecution"
        assert wire["init_args"]["action_queue"]["init_args"]["smoother"]["class_path"] == (
            "physicalai.runtime.ReplaceSmoother"
        )
        assert wire["init_args"]["task"] == "stack"

    def test_rtc_policy_graph(self, inference_model: Any, _patch_adapter: MagicMock) -> None:
        from physicalai.runtime import PolicySource, RTCActionQueue, RTCExecution

        source = PolicySource(
            model=inference_model,
            execution=RTCExecution(chunk_size=40, fps=30.0),
            action_queue=RTCActionQueue(),
        )
        wire = _assert_construction_round_trip(source)
        assert wire["init_args"]["execution"]["class_path"] == "physicalai.runtime.RTCExecution"
        assert wire["init_args"]["action_queue"]["class_path"] == "physicalai.runtime.RTCActionQueue"


# ---------------------------------------------------------------------------
# TeleopSource
# ---------------------------------------------------------------------------


class TestTeleopSourceComponentConfig:
    def test_leader_nested_round_trip(self) -> None:
        from physicalai.robot import SO101
        from physicalai.runtime import TeleopSource

        calibration = {
            "shoulder_pan": {"id": 1, "drive_mode": 0, "homing_offset": 2048, "range_min": 707, "range_max": 3439},
            "shoulder_lift": {"id": 2, "drive_mode": 1, "homing_offset": 1024, "range_min": 669, "range_max": 3292},
            "elbow_flex": {"id": 3, "drive_mode": 0, "homing_offset": 2048, "range_min": 846, "range_max": 3069},
            "wrist_flex": {"id": 4, "drive_mode": 0, "homing_offset": 2048, "range_min": 956, "range_max": 3311},
            "wrist_roll": {"id": 5, "drive_mode": 0, "homing_offset": 2048, "range_min": 59, "range_max": 3946},
            "gripper": {"id": 6, "drive_mode": 0, "homing_offset": 2048, "range_min": 2026, "range_max": 3074},
        }
        leader = SO101(port="/dev/ttyUSB0", calibration=calibration, role="leader")
        source = TeleopSource(leader=leader)
        wire = _assert_construction_round_trip(source)
        assert wire["class_path"] == "physicalai.runtime.TeleopSource"
        assert "to_action" not in wire["init_args"]
        assert wire["init_args"]["leader"]["class_path"] == "physicalai.robot.SO101"

    def test_supplied_to_action_fails(self) -> None:
        from physicalai.robot import SO101
        from physicalai.runtime import TeleopSource

        calibration = {
            "shoulder_pan": {"id": 1, "drive_mode": 0, "homing_offset": 2048, "range_min": 707, "range_max": 3439},
            "shoulder_lift": {"id": 2, "drive_mode": 1, "homing_offset": 1024, "range_min": 669, "range_max": 3292},
            "elbow_flex": {"id": 3, "drive_mode": 0, "homing_offset": 2048, "range_min": 846, "range_max": 3069},
            "wrist_flex": {"id": 4, "drive_mode": 0, "homing_offset": 2048, "range_min": 956, "range_max": 3311},
            "wrist_roll": {"id": 5, "drive_mode": 0, "homing_offset": 2048, "range_min": 59, "range_max": 3946},
            "gripper": {"id": 6, "drive_mode": 0, "homing_offset": 2048, "range_min": 2026, "range_max": 3074},
        }
        leader = SO101(port="/dev/ttyUSB0", calibration=calibration, role="leader")
        source = TeleopSource(leader=leader, to_action=lambda obs: obs.joint_positions)
        with pytest.raises(ComponentConfigError, match=r"init_args\.to_action"):
            to_config(source)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


class TestCallbackComponentConfig:
    def test_console_round_trip(self) -> None:
        from physicalai.runtime import ConsoleCallback

        wire = _assert_construction_round_trip(ConsoleCallback(throttle_steps=15))
        assert wire["class_path"] == "physicalai.runtime.ConsoleCallback"
        assert wire["init_args"] == {"throttle_steps": 15}

    def test_low_pass_round_trip(self) -> None:
        from physicalai.runtime import LowPassFilterCallback

        wire = _assert_construction_round_trip(LowPassFilterCallback(alpha=0.3))
        assert wire["class_path"] == "physicalai.runtime.LowPassFilterCallback"
        assert wire["init_args"] == {"alpha": 0.3}

    def test_jsonl_temp_path_round_trip(self, tmp_path: Path) -> None:
        from physicalai.runtime import JsonlCallback

        path = tmp_path / "events.jsonl"
        cb = JsonlCallback(path=path, record_chunks=True)
        restored = None
        try:
            assert is_config_exportable(cb)
            config = to_config(cb)
            wire: dict[str, Any] = json.loads(json.dumps(config))
            restored = instantiate(wire)
            assert type(restored) is type(cb)
            assert to_config(restored) == wire
            assert wire["class_path"] == "physicalai.runtime.JsonlCallback"
            assert wire["init_args"]["path"] == str(path)
            assert wire["init_args"]["record_chunks"] is True
        finally:
            cb.close()
            if restored is not None:
                restored.close()  # type: ignore[attr-defined]

    def test_rerun_scalar_round_trip(self) -> None:
        from physicalai.runtime import RerunCallback

        rr = MagicMock()
        rr.__name__ = "rerun"
        with patch.dict(sys.modules, {"rerun": rr}):
            cb = RerunCallback(
                mode="connect",
                connect_addr="10.0.0.1:9876",
                application_id="test-app",
                image_decimation=5,
                log_images=False,
            )
            wire = _assert_construction_round_trip(cb)
        assert wire["class_path"] == "physicalai.runtime.RerunCallback"
        assert wire["init_args"]["mode"] == "connect"
        assert wire["init_args"]["connect_addr"] == "10.0.0.1:9876"
        assert wire["init_args"]["application_id"] == "test-app"
        assert wire["init_args"]["image_decimation"] == 5
        assert wire["init_args"]["log_images"] is False

    def test_async_nested_console_round_trip(self) -> None:
        from physicalai.runtime import AsyncCallback, ConsoleCallback

        cb = AsyncCallback(ConsoleCallback(throttle_steps=10), max_queue=64)
        try:
            wire = _assert_construction_round_trip(cb)
            assert wire["class_path"] == "physicalai.runtime.AsyncCallback"
            assert wire["init_args"]["max_queue"] == 64
            assert wire["init_args"]["inner"]["class_path"] == "physicalai.runtime.ConsoleCallback"
            restored = instantiate(wire)
            try:
                assert isinstance(restored._inner, ConsoleCallback)  # type: ignore[attr-defined]
            finally:
                restored.close()  # type: ignore[attr-defined]
        finally:
            cb.close()

    def test_async_undecorated_inner_fails(self) -> None:
        from physicalai.runtime import AsyncCallback

        class UndecoratedTickCallback:
            def on_tick(self, event: object) -> None:
                pass

        cb = AsyncCallback(UndecoratedTickCallback())
        try:
            with pytest.raises(ComponentConfigError, match=r"init_args\.inner"):
                to_config(cb)
        finally:
            cb.close()

    def test_async_rejects_action_hook_inner(self) -> None:
        from physicalai.runtime import AsyncCallback, LowPassFilterCallback

        with pytest.raises(TypeError, match="action hooks"):
            AsyncCallback(LowPassFilterCallback(alpha=0.5))


# ---------------------------------------------------------------------------
# RobotRuntime (step 7)
# ---------------------------------------------------------------------------


_SAMPLE_CALIBRATION: dict[str, Any] = {
    "shoulder_pan": {"id": 1, "drive_mode": 0, "homing_offset": 2048, "range_min": 707, "range_max": 3439},
    "shoulder_lift": {"id": 2, "drive_mode": 1, "homing_offset": 1024, "range_min": 669, "range_max": 3292},
    "elbow_flex": {"id": 3, "drive_mode": 0, "homing_offset": 2048, "range_min": 846, "range_max": 3069},
    "wrist_flex": {"id": 4, "drive_mode": 0, "homing_offset": 2048, "range_min": 956, "range_max": 3311},
    "wrist_roll": {"id": 5, "drive_mode": 0, "homing_offset": 2048, "range_min": 59, "range_max": 3946},
    "gripper": {"id": 6, "drive_mode": 0, "homing_offset": 2048, "range_min": 2026, "range_max": 3074},
}


def _make_so101(*, port: str = "/dev/ttyACM0") -> Any:
    from physicalai.robot import SO101

    return SO101(port=port, calibration=_SAMPLE_CALIBRATION)


def _make_full_runtime(inference_model: Any) -> Any:
    from physicalai.capture import UVCCamera
    from physicalai.runtime import ConsoleCallback, PolicySource, RobotRuntime, SyncExecution

    return RobotRuntime(
        robot=_make_so101(),
        action_source=PolicySource(
            model=inference_model,
            execution=SyncExecution(request_threshold=0.25),
            task="pick cube",
        ),
        fps=30.0,
        cameras={"wrist": UVCCamera(device="/dev/video0", width=640, height=480, fps=30, backend="v4l2")},
        callbacks=[ConsoleCallback(throttle_steps=15)],
    )


class TestRobotRuntimeComponentConfig:
    def test_complete_tree_instantiate_round_trip(
        self, inference_model: Any, _patch_adapter: MagicMock
    ) -> None:
        runtime = _make_full_runtime(inference_model)
        wire = _assert_construction_round_trip(runtime)
        assert wire["class_path"] == "physicalai.runtime.RobotRuntime"
        assert wire["init_args"]["fps"] == 30.0
        assert wire["init_args"]["robot"]["class_path"] == "physicalai.robot.SO101"
        assert wire["init_args"]["action_source"]["class_path"] == "physicalai.runtime.PolicySource"
        assert wire["init_args"]["cameras"]["wrist"]["class_path"] == "physicalai.capture.UVCCamera"
        assert wire["init_args"]["callbacks"][0]["class_path"] == "physicalai.runtime.ConsoleCallback"
        # Connection / session state are never part of the recipe.
        assert "session_id" not in wire["init_args"]
        assert "_connected" not in wire["init_args"]
        from physicalai.runtime import RobotRuntime

        restored = instantiate(wire)
        assert isinstance(restored, RobotRuntime)
        assert restored._connected is False  # noqa: SLF001
        assert restored._session_id == ""  # noqa: SLF001

    def test_complete_tree_jsonargparse_under_runtime(
        self,
        inference_model: Any,
        _patch_adapter: MagicMock,
        tmp_path: Path,
    ) -> None:
        from physicalai.capture import UVCCamera
        from physicalai.cli.run import build_parser
        from physicalai.robot import SO101
        from physicalai.runtime import ConsoleCallback, PolicySource, RobotRuntime

        runtime = _make_full_runtime(inference_model)
        wire = to_config(runtime)
        # CLI nests constructor args under ``runtime:`` (not the bare ComponentConfig).
        # Prefer JSON for the ActionConfigFile payload — avoids PyYAML C-loader
        # interaction with ``yaml.safe_dump`` that can poison later YAML parses in
        # the same process.
        cfg_path = tmp_path / "runtime.json"
        cfg_path.write_text(json.dumps({"runtime": wire["init_args"]}))

        parser = build_parser()
        ns = parser.parse_args(["--config", str(cfg_path)])
        restored = parser.instantiate(ns).runtime
        assert isinstance(restored, RobotRuntime)
        assert restored._connected is False  # noqa: SLF001
        assert restored._session_id == ""  # noqa: SLF001
        assert restored._fps == 30.0  # noqa: SLF001
        assert isinstance(restored.robot, SO101)
        assert isinstance(restored.action_source, PolicySource)
        assert set(restored.cameras) == {"wrist"}
        assert isinstance(restored.cameras["wrist"], UVCCamera)
        assert isinstance(restored._bus._callbacks[0], ConsoleCallback)  # noqa: SLF001
        # jsonargparse may supply omitted ctor defaults; check public class_path shape
        # after a JSON boundary (strips StrEnum identity / applied defaults noise).
        restored_wire = json.loads(json.dumps(to_config(restored)))
        assert restored_wire["class_path"] == "physicalai.runtime.RobotRuntime"
        assert restored_wire["init_args"]["fps"] == 30.0
        assert restored_wire["init_args"]["robot"]["class_path"] == "physicalai.robot.SO101"
        assert restored_wire["init_args"]["action_source"]["class_path"] == "physicalai.runtime.PolicySource"
        assert restored_wire["init_args"]["cameras"]["wrist"]["class_path"] == "physicalai.capture.UVCCamera"
        assert restored_wire["init_args"]["callbacks"][0]["class_path"] == "physicalai.runtime.ConsoleCallback"

        via_from_config = RobotRuntime.from_config(cfg_path)
        assert isinstance(via_from_config, RobotRuntime)
        assert via_from_config._connected is False  # noqa: SLF001
        assert via_from_config._fps == 30.0  # noqa: SLF001

    def test_bare_component_export_round_trips(
        self,
        inference_model: Any,
        _patch_adapter: MagicMock,
        tmp_path: Path,
    ) -> None:
        from physicalai.cli.run import build_parser
        from physicalai.runtime import RobotRuntime

        runtime = _make_full_runtime(inference_model)
        # The bare to_config shape (top-level class_path) — no manual rewrite
        # to ``runtime:`` needed on either load path.
        cfg_path = tmp_path / "runtime_export.json"
        cfg_path.write_text(json.dumps(to_config(runtime)))

        parser = build_parser()
        ns = parser.parse_args(["--config", str(cfg_path)])
        restored = parser.instantiate(ns).runtime
        assert isinstance(restored, RobotRuntime)
        assert restored._fps == 30.0  # noqa: SLF001
        assert restored._connected is False  # noqa: SLF001

        via_from_config = RobotRuntime.from_config(cfg_path)
        assert isinstance(via_from_config, RobotRuntime)
        assert via_from_config._fps == 30.0  # noqa: SLF001

    def test_bare_export_foreign_class_path_rejected(self, tmp_path: Path) -> None:
        from physicalai.config import ComponentConfigError
        from physicalai.runtime import RobotRuntime

        cfg_path = tmp_path / "not_runtime.json"
        cfg_path.write_text(json.dumps({"class_path": "physicalai.runtime.SyncExecution", "init_args": {}}))

        with pytest.raises(ComponentConfigError, match="does not resolve to"):
            RobotRuntime.from_config(cfg_path)

    def test_omitted_cameras_and_callbacks(
        self, inference_model: Any, _patch_adapter: MagicMock
    ) -> None:
        from physicalai.runtime import PolicySource, RobotRuntime

        runtime = RobotRuntime(
            robot=_make_so101(),
            action_source=PolicySource(model=inference_model),
            fps=20.0,
        )
        wire = _assert_construction_round_trip(runtime)
        assert "cameras" not in wire["init_args"]
        assert "callbacks" not in wire["init_args"]

    def test_empty_callbacks_stay_empty(
        self, inference_model: Any, _patch_adapter: MagicMock
    ) -> None:
        from physicalai.runtime import PolicySource, RobotRuntime

        runtime = RobotRuntime(
            robot=_make_so101(),
            action_source=PolicySource(model=inference_model),
            fps=20.0,
            callbacks=[],
        )
        wire = _assert_construction_round_trip(runtime)
        assert wire["init_args"]["callbacks"] == []

    def test_undecorated_robot_fails(
        self, inference_model: Any, _patch_adapter: MagicMock
    ) -> None:
        from physicalai.runtime import PolicySource, RobotRuntime

        class UndecoratedRobot:
            def __init__(self) -> None:
                pass

        runtime = RobotRuntime(
            robot=UndecoratedRobot(),  # type: ignore[arg-type]
            action_source=PolicySource(model=inference_model),
            fps=10.0,
        )
        with pytest.raises(ComponentConfigError, match=r"init_args\.robot"):
            to_config(runtime)

    def test_undecorated_action_source_fails(self) -> None:
        from physicalai.runtime import RobotRuntime

        class UndecoratedSource:
            def connect(self, **_kwargs: object) -> None: ...
            def disconnect(self) -> None: ...
            def update(self, *_args: object, **_kwargs: object) -> object:
                return None

        runtime = RobotRuntime(
            robot=_make_so101(),
            action_source=UndecoratedSource(),  # type: ignore[arg-type]
            fps=10.0,
        )
        with pytest.raises(ComponentConfigError, match=r"init_args\.action_source"):
            to_config(runtime)

    def test_undecorated_camera_fails(
        self, inference_model: Any, _patch_adapter: MagicMock
    ) -> None:
        from physicalai.runtime import PolicySource, RobotRuntime

        class UndecoratedCamera:
            def __init__(self) -> None:
                pass

        runtime = RobotRuntime(
            robot=_make_so101(),
            action_source=PolicySource(model=inference_model),
            fps=10.0,
            cameras={"wrist": UndecoratedCamera()},  # type: ignore[dict-item]
        )
        with pytest.raises(ComponentConfigError, match=r"init_args\.cameras\.wrist"):
            to_config(runtime)

    def test_undecorated_callback_fails(
        self, inference_model: Any, _patch_adapter: MagicMock
    ) -> None:
        from physicalai.runtime import PolicySource, RobotRuntime

        class UndecoratedCallback:
            def on_tick(self, event: object) -> None:
                pass

        runtime = RobotRuntime(
            robot=_make_so101(),
            action_source=PolicySource(model=inference_model),
            fps=10.0,
            callbacks=[UndecoratedCallback()],
        )
        with pytest.raises(ComponentConfigError, match=r"init_args\.callbacks\[0\]"):
            to_config(runtime)

    def test_jsonargparse_sees_original_ctor_signature(self) -> None:
        import inspect

        from physicalai.runtime import RobotRuntime

        params = list(inspect.signature(RobotRuntime.__init__).parameters)
        assert params == ["self", "robot", "action_source", "fps", "cameras", "callbacks"]

    def test_nested_shared_robot_and_shared_camera(
        self, inference_model: Any, _patch_adapter: MagicMock
    ) -> None:
        from physicalai.capture import SharedCamera
        from physicalai.robot import SharedRobot
        from physicalai.runtime import PolicySource, RobotRuntime, SyncExecution

        runtime = RobotRuntime(
            robot=SharedRobot(
                "follower",
                robot={
                    "class_path": "tests.unit.robot.transport.fake.FakeRobot",
                    "init_args": {"port": "/dev/fake-rt", "device_ids": ["fake:follower"]},
                },
            ),
            action_source=PolicySource(
                model=inference_model,
                execution=SyncExecution(),
            ),
            fps=15.0,
            cameras={
                "wrist": SharedCamera(
                    camera={
                        "class_path": "physicalai.capture.UVCCamera",
                        "init_args": {
                            "device": "/dev/video2",
                            "width": 320,
                            "height": 240,
                            "fps": 15,
                            "backend": "v4l2",
                        },
                    },
                ),
            },
        )
        wire = _assert_construction_round_trip(runtime)
        assert wire["init_args"]["robot"]["class_path"] == "physicalai.robot.SharedRobot"
        assert wire["init_args"]["robot"]["init_args"]["name"] == "follower"
        nested_robot = wire["init_args"]["robot"]["init_args"]["robot"]
        assert nested_robot["class_path"] == "tests.unit.robot.transport.fake.FakeRobot"
        assert wire["init_args"]["cameras"]["wrist"]["class_path"] == "physicalai.capture.SharedCamera"
        nested_cam = wire["init_args"]["cameras"]["wrist"]["init_args"]["camera"]
        assert nested_cam["class_path"] == "physicalai.capture.UVCCamera"
        restored = instantiate(wire)
        assert isinstance(restored, RobotRuntime)
        assert isinstance(restored.robot, SharedRobot)
        assert isinstance(restored.cameras["wrist"], SharedCamera)
        assert restored._connected is False  # noqa: SLF001
        assert not restored.robot.is_connected()
        assert not restored.cameras["wrist"].is_connected
