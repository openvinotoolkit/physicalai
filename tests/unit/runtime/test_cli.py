# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from physicalai.cli._config import RuntimeConfig, load_config

FIXTURES_DIR = Path(__file__).resolve().parents[2] / "fixtures"
VALID_CONFIG = FIXTURES_DIR / "runtime_config.yaml"
INVALID_CONFIG = FIXTURES_DIR / "runtime_config_invalid.yaml"


class TestLoadConfig:
    def test_load_config_valid(self) -> None:
        config = load_config(VALID_CONFIG)
        assert isinstance(config, RuntimeConfig)
        assert config.model.path == "./test_exports/act_policy"
        assert config.model.backend == "auto"
        assert config.fps == 10.0
        assert config.duration_s == 1.0
        assert config.execution.mode == "sync"
        assert "overhead" in config.cameras

    def test_load_config_missing_file(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.yaml")

    def test_load_config_invalid_yaml(self) -> None:
        with pytest.raises(ValueError, match="Invalid runtime config"):
            load_config(INVALID_CONFIG)

    def test_runtime_config_defaults(self) -> None:
        config = RuntimeConfig(model={"path": "./test"}, robot={"type": "mock"})
        assert config.fps == 30.0
        assert config.duration_s is None
        assert config.execution.mode == "async"
        assert config.smoother.type == "lerp"
        assert config.cameras == {}

    def test_runtime_config_camera_extra_fields(self) -> None:
        config = RuntimeConfig(
            model={"path": "./test"},
            robot={"type": "mock"},
            cameras={"cam0": {"type": "realsense", "device_id": "123", "resolution": [640, 480]}},
        )
        cam = config.cameras["cam0"]
        assert cam.type == "realsense"
        assert cam.model_dump()["resolution"] == [640, 480]


class TestCLI:
    def _run_cli(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "physicalai.cli", *args],
            capture_output=True,
            text=True,
            timeout=30,
        )

    def test_cli_run_help(self) -> None:
        result = self._run_cli("run", "--help")
        assert result.returncode == 0
        assert "--config" in result.stdout

    def test_cli_dry_run(self) -> None:
        result = self._run_cli("run", "--config", str(VALID_CONFIG), "--dry-run")
        assert result.returncode == 0
        assert "10.0" in result.stdout
        assert "Config loaded" in result.stdout

    def test_cli_missing_config(self) -> None:
        result = self._run_cli("run", "--config", "/nonexistent.yaml")
        assert result.returncode != 0
        assert "not found" in result.stderr.lower() or "error" in result.stderr.lower()

    def test_cli_no_command(self) -> None:
        result = self._run_cli()
        assert result.returncode == 2
