# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for inference runners and the runner factory."""

from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from physicalai.inference.manifest import ComponentSpec
from physicalai.inference.runners import SinglePass, TwoStage, get_runner


def fake_adapter(outputs: dict[str, np.ndarray], input_names: list[str] | None = None) -> Mock:
    """Build an adapter stand-in whose predict returns fixed outputs.

    Args:
        outputs: What ``predict`` returns.
        input_names: Declared input names, empty when omitted.

    Returns:
        The mock adapter.
    """
    adapter = Mock()
    adapter.predict.return_value = outputs
    adapter.input_names = input_names or []
    return adapter


class TestSinglePass:
    """The default runner is a passthrough."""

    def test_returns_adapter_output(self) -> None:
        """Test SinglePass forwards the adapter's output unchanged."""
        adapter = fake_adapter({"action": np.zeros(3)})

        outputs = SinglePass().run(adapter, {"state": np.zeros(2)})

        assert set(outputs) == {"action"}
        adapter.predict.assert_called_once()


class TestTwoStage:
    """Chaining a second model onto the primary adapter."""

    @staticmethod
    def _runner(second: Mock, tmp_path: Path) -> TwoStage:
        """Build a runner whose second stage is a mock adapter.

        Args:
            second: The stand-in for the second adapter.
            tmp_path: Directory used as the artifact location.

        Returns:
            The configured runner.
        """
        runner = TwoStage(backend="openvino", artifact=str(tmp_path / "second.xml"))
        runner._second = second  # noqa: SLF001 - bypasses lazy loading in tests
        return runner

    def test_feeds_the_first_stage_output_into_the_second(self, tmp_path: Path) -> None:
        """Test the intermediate tensors are passed on by name."""
        first = fake_adapter({"noise": np.zeros(2), "state_embed": np.ones(2)})
        second = fake_adapter({"action": np.full(3, 7.0)}, input_names=["noise", "state_embed"])

        outputs = self._runner(second, tmp_path).run(first, {"state": np.zeros(2)})

        assert outputs["action"].tolist() == [7.0, 7.0, 7.0]
        passed = second.predict.call_args.args[0]
        assert set(passed) == {"noise", "state_embed"}

    def test_drops_intermediates_the_second_stage_does_not_want(self, tmp_path: Path) -> None:
        """Test extra outputs from the first stage are filtered out.

        The first stage is a whole model, so it may emit diagnostics the second
        graph has no input for; passing those on would fail inside the backend.
        """
        first = fake_adapter({"noise": np.zeros(2), "debug": np.zeros(1)})
        second = fake_adapter({"action": np.zeros(3)}, input_names=["noise"])

        self._runner(second, tmp_path).run(first, {})

        assert set(second.predict.call_args.args[0]) == {"noise"}

    def test_passes_everything_when_the_second_stage_declares_nothing(self, tmp_path: Path) -> None:
        """Test an adapter without declared inputs receives the whole dict."""
        first = fake_adapter({"noise": np.zeros(2), "extra": np.zeros(1)})
        second = fake_adapter({"action": np.zeros(3)})

        self._runner(second, tmp_path).run(first, {})

        assert set(second.predict.call_args.args[0]) == {"noise", "extra"}

    def test_missing_intermediate_is_reported(self, tmp_path: Path) -> None:
        """Test a naming mismatch is reported rather than left to the backend."""
        first = fake_adapter({"noise": np.zeros(2)})
        second = fake_adapter({"action": np.zeros(3)}, input_names=["noise", "state_embed"])

        with pytest.raises(KeyError, match="state_embed"):
            self._runner(second, tmp_path).run(first, {})

    def test_second_stage_is_loaded_lazily(self, tmp_path: Path) -> None:
        """Test constructing the runner does not compile the second graph.

        The runner is built while the manifest is read, before the export
        directory has necessarily been validated.
        """
        artifact = tmp_path / "second.xml"
        runner = TwoStage(backend="openvino", artifact=str(artifact))
        first = fake_adapter({"noise": np.zeros(2)})
        second = fake_adapter({"action": np.zeros(3)})

        with patch("physicalai.inference.runners.two_stage.get_adapter", return_value=second) as factory:
            factory.assert_not_called()
            runner.run(first, {})
            runner.run(first, {})

        factory.assert_called_once()
        second.load.assert_called_once_with(artifact)

    def test_reset_keeps_the_compiled_second_stage(self, tmp_path: Path) -> None:
        """Test a new episode does not force a graph recompile."""
        second = fake_adapter({"action": np.zeros(3)})
        runner = self._runner(second, tmp_path)

        runner.reset()

        assert runner._second is second  # noqa: SLF001 - asserting the cache survives

    def test_repr_names_the_second_stage(self, tmp_path: Path) -> None:
        """Test the representation is useful in a log line."""
        assert "openvino" in repr(TwoStage(backend="openvino", artifact=str(tmp_path / "second.xml")))


class TestGetRunner:
    """Runner selection from a manifest."""

    def test_defaults_to_single_pass(self) -> None:
        """Test a manifest without a runner spec gets the passthrough runner."""
        assert isinstance(get_runner({}), SinglePass)

    def test_builds_a_declared_runner(self) -> None:
        """Test a type-based spec resolves through the component registry."""
        runner = get_runner({"model": {"runner": {"type": "single_pass"}}})

        assert isinstance(runner, SinglePass)

    def test_rejects_a_component_that_is_not_a_runner(self) -> None:
        """Test a misconfigured manifest fails at load, not at first inference."""
        spec = {"model": {"runner": {"type": "normalize", "stats": {}}}}

        with pytest.raises(TypeError, match="not an InferenceRunner"):
            get_runner(spec)

    def test_resolves_the_runner_artifact_against_the_export_dir(self, tmp_path: Path) -> None:
        """Test a relative artifact becomes absolute, as it does for preprocessors.

        A manifest names artifacts relative to the export directory, so a runner
        that loads a model of its own cannot use the name as given.
        """
        spec = {"model": {"runner": {"type": "two_stage", "backend": "openvino", "artifact": "expert.xml"}}}

        runner = get_runner(spec, export_dir=tmp_path)

        assert isinstance(runner, TwoStage)
        assert runner.artifact == str(tmp_path.resolve() / "expert.xml")

    def test_resolves_a_class_path_runner_artifact(self, tmp_path: Path) -> None:
        """Test class_path specs resolve their artifact too.

        Exporters prefer class_path mode for runners that carry manifest metadata:
        in type mode every extra field is forwarded to the constructor, so a
        chunk_size annotation would be passed to the runner as an argument.
        """
        spec = {
            "model": {
                "runner": {
                    "class_path": "physicalai.inference.runners.TwoStage",
                    "init_args": {"backend": "openvino", "artifact": "expert.xml"},
                    "chunk_size": 8,
                },
            },
        }

        runner = get_runner(spec, export_dir=tmp_path)

        assert isinstance(runner, TwoStage)
        assert runner.artifact == str(tmp_path.resolve() / "expert.xml")

    def test_rejects_an_artifact_outside_the_export_dir(self, tmp_path: Path) -> None:
        """Test path traversal in a manifest is refused."""
        spec = {
            "model": {
                "runner": {"type": "two_stage", "backend": "openvino", "artifact": "../../etc/passwd"},
            },
        }

        with pytest.raises(ValueError, match="escapes the export directory"):
            get_runner(spec, export_dir=tmp_path)

    def test_manifest_object_path_resolves_too(self, tmp_path: Path) -> None:
        """Test the Manifest branch behaves like the dict branch."""
        from physicalai.inference.manifest import Manifest, ModelSpec

        manifest = Manifest(
            model=ModelSpec(
                runner=ComponentSpec(type="two_stage", backend="openvino", artifact="expert.xml"),
                artifacts={"torch": "policy.pt"},
            ),
        )

        runner = get_runner(manifest, export_dir=tmp_path)

        assert isinstance(runner, TwoStage)
        assert runner.artifact == str(tmp_path.resolve() / "expert.xml")
