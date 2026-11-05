# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest
import yaml


def test_help_and_registration_are_lightweight():
    code = """
import sys
from importlib.metadata import entry_points
from physicalai.cli.main import main
assert any(ep.name == 'rosbag' for ep in entry_points(group='physicalai.cli.subcommands'))
try:
    main(['rosbag', '--help'])
except SystemExit as error:
    assert error.code == 0
assert not {'torch', 'lerobot', 'rosbags', 'cv2'} & sys.modules.keys()
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "convert" in result.stdout


def test_cli_reports_invalid_mapping(tmp_path):
    mapping = tmp_path / "invalid.yaml"
    mapping.write_text("fps: 0\n")
    executable = Path(sys.executable).parent / "physicalai"
    result = subprocess.run(
        [
            str(executable),
            "rosbag",
            "convert",
            "--input",
            str(tmp_path),
            "--output",
            str(tmp_path / "out"),
            "--mapping",
            str(mapping),
            "--repo-id",
            "local/test",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "conversion failed" in result.stderr
    assert not (tmp_path / "out").exists()


@pytest.mark.skipif(sys.platform != "linux", reason="POSIX signal cleanup")
@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM])
def test_cli_signal_cleans_staging(tmp_path, demo, signum):
    bag = tmp_path / "bag"
    demo["write_demo"](bag)
    mapping = tmp_path / "mapping.yaml"
    mapping.write_text(yaml.safe_dump(demo["demo_mapping"]()))
    marker = tmp_path / "started"
    output = tmp_path / "out"
    code = """
import sys,time
from pathlib import Path
from physicalai.cli.main import main
import physicalai_rosbag_plugin.converter as converter
marker=Path(sys.argv[1])
def probe(root):
    marker.write_text(str(root))
    time.sleep(60)
converter._encoder_probe=probe
raise SystemExit(main(sys.argv[2:]))
"""
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            code,
            str(marker),
            "rosbag",
            "convert",
            "--input",
            str(bag),
            "--output",
            str(output),
            "--mapping",
            str(mapping),
            "--repo-id",
            "local/test",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 20
        while not marker.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        assert marker.exists(), process.communicate(timeout=5)
        os.kill(process.pid, signum)
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == 130, (stdout, stderr)
        assert not output.exists()
        assert not Path(marker.read_text()).exists()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
