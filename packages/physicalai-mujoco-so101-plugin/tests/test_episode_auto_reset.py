"""Tests for cube-on-plate episode auto-reset."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from physicalai_mujoco_so101_plugin.episode_auto_reset import (
    EpisodeAutoReset,
    EpisodeAutoResetConfig,
    _Runtime,
)


def _make_helper(*, dwell_s: float = 5.0) -> EpisodeAutoReset:
    config = EpisodeAutoResetConfig(
        success_dwell_s=dwell_s,
        target_radius=0.05,
        cube_half_size=0.02,
        cube_static_speed=0.08,
        spawn_center=(0.24, 0.0),
        spawn_min_r=0.08,
        spawn_max_r=0.30,
        target_min_sep=0.11,
    )
    helper = EpisodeAutoReset(config, rng=np.random.default_rng(0))
    helper._runtime = _Runtime(  # noqa: SLF001
        cube_body_id=1,
        target_body_id=2,
        cube_qpos_address=0,
        cube_dof_address=0,
    )
    return helper


def _fake_model_data(
    *,
    cube_xy: tuple[float, float],
    target_xy: tuple[float, float],
    z: float = 0.02,
    speed: float = 0.0,
) -> tuple[MagicMock, SimpleNamespace]:
    xpos = np.zeros((3, 3), dtype=np.float64)
    xpos[1, :2] = cube_xy
    xpos[1, 2] = z
    xpos[2, :2] = target_xy
    xpos[2, 2] = 0.001
    qpos = np.zeros(7, dtype=np.float64)
    qpos[:3] = [cube_xy[0], cube_xy[1], z]
    qpos[3] = 1.0
    qvel = np.zeros(6, dtype=np.float64)
    qvel[:3] = speed
    data = SimpleNamespace(xpos=xpos, qpos=qpos, qvel=qvel, time=0.0)
    model = MagicMock()
    return model, data


class TestEpisodeAutoReset:
    def test_status_idle_when_enabled(self) -> None:
        helper = _make_helper()
        status = helper.status()
        assert status["enabled"] is True
        assert status["phase"] == "idle"
        assert status["countdown_s"] is None

    def test_starts_countdown_when_cube_on_plate(self) -> None:
        helper = _make_helper(dwell_s=5.0)
        model, data = _fake_model_data(cube_xy=(0.22, -0.30), target_xy=(0.22, -0.30))
        data.time = 1.0
        helper.update(model, data)
        assert helper.status()["phase"] == "success_hold"
        assert helper.status()["countdown_s"] == 5.0

    def test_cancels_countdown_if_cube_leaves(self) -> None:
        helper = _make_helper(dwell_s=5.0)
        model, data = _fake_model_data(cube_xy=(0.22, -0.30), target_xy=(0.22, -0.30))
        data.time = 1.0
        helper.update(model, data)
        model, data = _fake_model_data(cube_xy=(0.40, 0.0), target_xy=(0.22, -0.30))
        data.time = 2.0
        helper.update(model, data)
        assert helper.status()["phase"] == "idle"
        assert helper.status()["countdown_s"] is None

    def test_respawns_cube_after_dwell_without_moving_plate(self) -> None:
        helper = _make_helper(dwell_s=5.0)
        model, data = _fake_model_data(cube_xy=(0.22, -0.30), target_xy=(0.22, -0.30))
        data.time = 0.0
        helper.update(model, data)
        data.time = 5.0
        plate_before = data.xpos[2].copy()

        with patch("mujoco.mj_forward"):
            helper.update(model, data)

        assert helper.status()["phase"] == "idle"
        assert helper.status()["episode_count"] == 1
        assert np.allclose(data.xpos[2], plate_before)
        assert not np.allclose(data.qpos[:2], [0.22, -0.30])
        assert np.allclose(data.qvel[:6], 0.0)

    def test_manual_reset_clears_countdown(self) -> None:
        helper = _make_helper()
        model, data = _fake_model_data(cube_xy=(0.22, -0.30), target_xy=(0.22, -0.30))
        data.time = 1.0
        helper.update(model, data)
        helper.notify_manual_reset()
        assert helper.status()["phase"] == "idle"
        assert helper.status()["countdown_s"] is None
