from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_import_isolate_does_not_load_heavy_frameworks_or_other_runtime_packages() -> None:
    source_root = Path(__file__).resolve().parents[3] / "src"
    python_path = os.pathsep.join(
        part for part in (str(source_root), os.environ.get("PYTHONPATH", "")) if part
    )
    child_env = {**os.environ, "PYTHONPATH": python_path}
    script = """
import sys
import physicalai.isolate
heavy = {"torch", "transformers"}.intersection(sys.modules)
other_runtime = {
    "physicalai.config",
    "physicalai.engine",
    "physicalai.inference",
    "physicalai.robot",
    "physicalai.runtime",
    "physicalai.studio",
}.intersection(sys.modules)
assert not heavy, sorted(heavy)
assert not other_runtime, sorted(other_runtime)
"""

    subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        env=child_env,
        shell=False,
    )
