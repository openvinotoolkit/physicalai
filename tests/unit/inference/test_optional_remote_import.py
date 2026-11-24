# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


@pytest.mark.parametrize("dependency", ["msgpack", "zenoh"])
def test_cli_and_robot_codec_import_without_transport_dependency(dependency: str) -> None:
    script = textwrap.dedent(
        f"""
        import builtins

        original_import = builtins.__import__
        blocked_dependency = {dependency!r}

        def block_optional(name, *args, **kwargs):
            if name == blocked_dependency or name.startswith(blocked_dependency + "."):
                raise ModuleNotFoundError("simulated missing optional dependency: " + blocked_dependency)
            return original_import(name, *args, **kwargs)

        builtins.__import__ = block_optional
        import physicalai.cli.main
        import physicalai.cli.robot
        from physicalai.robot.transport import _codec

        assert _codec.ROBOT_TRANSPORT_PROTOCOL_VERSION == 1
        """
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("dependency", ["msgpack", "zenoh"])
def test_remote_tests_skip_collection_without_transport_dependency(dependency: str) -> None:
    script = textwrap.dedent(
        f"""
        import importlib.abc
        import sys
        import pytest

        blocked_dependency = {dependency!r}

        class BlockOptional(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == blocked_dependency or fullname.startswith(blocked_dependency + "."):
                    raise ModuleNotFoundError("simulated missing optional dependency: " + fullname, name=fullname)
                return None

        for name in tuple(sys.modules):
            if name == blocked_dependency or name.startswith(blocked_dependency + "."):
                del sys.modules[name]
        sys.meta_path.insert(0, BlockOptional())
        raise SystemExit(pytest.main(["--collect-only", "-q", "-m", "ov_smoke", sys.argv[1]]))
        """
    )
    test_file = Path(__file__).parent / "remote" / "test_protocol_and_client.py"
    result = subprocess.run(
        [sys.executable, "-c", script, str(test_file)], capture_output=True, text=True, check=False,
    )

    assert result.returncode == pytest.ExitCode.NO_TESTS_COLLECTED, result.stdout + result.stderr
    assert f"could not import '{dependency}'" in result.stdout


def test_inference_package_imports_without_zenoh() -> None:
    script = textwrap.dedent(
        """
        import builtins

        original_import = builtins.__import__

        def block_zenoh(name, *args, **kwargs):
            if name == "zenoh" or name.startswith("zenoh."):
                raise ModuleNotFoundError("simulated missing optional zenoh dependency")
            return original_import(name, *args, **kwargs)

        builtins.__import__ = block_zenoh
        import physicalai.inference as inference
        import physicalai.runtime as runtime

        assert inference.InferenceModel is not None
        assert "RemoteInferenceModel" not in inference.__all__
        assert "InferenceServer" not in inference.__all__
        assert runtime.SyncExecution is not None
        assert "RemoteInferenceModel" not in runtime.__all__
        assert "InferenceServer" not in runtime.__all__
        assert not hasattr(runtime, "RemoteInferenceModel")
        assert not hasattr(runtime, "InferenceServer")
        """
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)

    assert result.returncode == 0, result.stderr
