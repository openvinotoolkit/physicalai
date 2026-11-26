"""Tests for the motion/observation helpers in physicalai_rebot_b601_plugin.motion."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import patch

import numpy as np
import pytest
from physicalai.config import Config

from physicalai_rebot_b601_plugin.motion import HoldPoseSource, JointLogger, SineWaveSource

if TYPE_CHECKING:
    from physicalai.runtime.events import TickEvent


class _FakeObservation:
    def __init__(self, joint_positions: np.ndarray) -> None:
        self.joint_positions = joint_positions
        self.timestamp = 1.0
        self.sensor_data: dict[str, np.ndarray] | None = None
        self.images: dict[str, Any] | None = None

    @property
    def state(self) -> np.ndarray:
        return self.joint_positions


class _FakeEvent:
    def __init__(self, step: int, joint_positions: np.ndarray) -> None:
        self.step = step
        self.robot_state = _FakeObservation(joint_positions)


class TestSineWaveSource:
    def test_update_returns_sine_commands(self) -> None:
        source = SineWaveSource(amplitude=10.0, frequency=0.25)
        obs = _FakeObservation(np.zeros(7, dtype=np.float32))
        with patch("physicalai_rebot_b601_plugin.motion.time.monotonic", side_effect=[0.0, 0.5]):
            source.connect(bus=cast(Any, object()), session_id="s")
            action = source.update(cast(Any, obs), {}, 0)

        expected = np.array(
            [10.0 * math.sin(math.tau * 0.25 * 0.5 + i * math.tau / 7) for i in range(7)],
            dtype=np.float32,
        )
        np.testing.assert_allclose(action, expected)

    def test_joint_amplitudes_and_phase_offsets(self) -> None:
        source = SineWaveSource(
            frequency=0.5,
            joint_amplitudes=[1.0, 2.0, 0.0],
            phase_offsets=[0.0, math.pi / 2, 1.0],
        )
        obs = _FakeObservation(np.zeros(3, dtype=np.float32))
        with patch("physicalai_rebot_b601_plugin.motion.time.monotonic", side_effect=[0.0, 1.0]):
            source.connect(bus=cast(Any, object()), session_id="s")
            action = source.update(cast(Any, obs), {}, 0)

        expected = np.array(
            [1.0 * math.sin(math.tau * 0.5), 2.0 * math.sin(math.tau * 0.5 + math.pi / 2), 0.0],
            dtype=np.float32,
        )
        np.testing.assert_allclose(action, expected)

    def test_mismatched_lists_raise(self) -> None:
        source = SineWaveSource(joint_amplitudes=[1.0, 2.0])
        obs = _FakeObservation(np.zeros(3, dtype=np.float32))
        with pytest.raises(ValueError, match="match the robot"):
            source.update(cast(Any, obs), {}, 0)

    def test_constructor_rejects_non_positive_frequency(self) -> None:
        with pytest.raises(ValueError, match="frequency"):
            SineWaveSource(frequency=0.0)

    def test_export_config_roundtrip(self) -> None:
        cfg = Config.from_instance(SineWaveSource(frequency=0.5))
        assert cfg["class_path"] == "physicalai_rebot_b601_plugin.motion.SineWaveSource"
        assert cfg["init_args"]["frequency"] == 0.5


class TestHoldPoseSource:
    def test_returns_observation(self) -> None:
        source = HoldPoseSource()
        obs = _FakeObservation(np.array([1.0, 2.0, 3.0], dtype=np.float32))

        action = source.update(cast(Any, obs), {}, 0)

        np.testing.assert_allclose(action, [1.0, 2.0, 3.0])

    def test_connect_and_disconnect_are_noops(self) -> None:
        source = HoldPoseSource()
        source.connect(bus=cast(Any, object()), session_id="s")
        source.disconnect()

    def test_export_config_roundtrip(self) -> None:
        cfg = Config.from_instance(HoldPoseSource())
        assert cfg["class_path"] == "physicalai_rebot_b601_plugin.motion.HoldPoseSource"


class TestJointLogger:
    def test_logs_joint_positions(self, capsys: pytest.CaptureFixture[str]) -> None:
        logger = JointLogger(throttle_steps=1)
        logger.on_tick(cast("TickEvent", _FakeEvent(0, np.array([1.0, 2.0], dtype=np.float32))))
        out = capsys.readouterr().out
        assert "1.00" in out
        assert "2.00" in out

    def test_throttle_skips_steps(self, capsys: pytest.CaptureFixture[str]) -> None:
        logger = JointLogger(throttle_steps=5)
        logger.on_tick(cast("TickEvent", _FakeEvent(3, np.array([1.0], dtype=np.float32))))
        assert capsys.readouterr().out == ""
        logger.on_tick(cast("TickEvent", _FakeEvent(5, np.array([2.0], dtype=np.float32))))
        assert capsys.readouterr().out != ""

    def test_constructor_rejects_non_positive_throttle(self) -> None:
        with pytest.raises(ValueError, match="throttle"):
            JointLogger(throttle_steps=0)

    def test_export_config_roundtrip(self) -> None:
        cfg = Config.from_instance(JointLogger(throttle_steps=3))
        assert cfg["class_path"] == "physicalai_rebot_b601_plugin.motion.JointLogger"
        assert cfg["init_args"]["throttle_steps"] == 3
