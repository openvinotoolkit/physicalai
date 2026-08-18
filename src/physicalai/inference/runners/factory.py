# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Runner factory for selecting inference runners from a manifest."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from physicalai.inference.runners.base import InferenceRunner
from physicalai.inference.runners.single_pass import SinglePass

if TYPE_CHECKING:
    from pathlib import Path

    from physicalai.inference.manifest import ComponentSpec, Manifest


def get_runner(source: Manifest | dict[str, Any], export_dir: Path | None = None) -> InferenceRunner:
    """Select and instantiate a runner from a manifest.

    Supports three input formats:

    1. **Manifest object** — reads ``source.model.runner`` and
       instantiates via :func:`instantiate_component`.
    2. **Dict with runner spec** — raw manifest dict containing a
       runner component spec either under ``"model" -> "runner"``
       or at the top-level ``"runner"`` key.
    3. **Dict without runner spec** — falls back to a plain ``SinglePass`` runner.

    Args:
        source: A :class:`Manifest` instance or a raw manifest dict.
        export_dir: Directory the manifest was read from. When given, a relative
            ``artifact`` on the runner spec is resolved against it, the same way
            preprocessor artifacts are. Runners that load a model of their own —
            :class:`~physicalai.inference.runners.two_stage.TwoStage`, say — need
            this; the others ignore it.

    Returns:
        Configured runner instance.
    """
    from physicalai.inference.manifest import Manifest  # noqa: PLC0415

    if isinstance(source, Manifest):
        if source.model.runner is not None:
            return _instantiate_runner(source.model.runner, export_dir)
        return SinglePass()

    runner_spec = _extract_runner_spec(source)
    if runner_spec is not None:
        from physicalai.inference.manifest import ComponentSpec  # noqa: PLC0415

        return _instantiate_runner(ComponentSpec.model_validate(runner_spec), export_dir)

    return SinglePass()


def _instantiate_runner(spec: ComponentSpec, export_dir: Path | None) -> InferenceRunner:
    """Build a runner from a component spec.

    Args:
        spec: Runner component spec.
        export_dir: Directory used to resolve a relative ``artifact``, if any.

    Returns:
        The instantiated runner.

    Raises:
        TypeError: If the component is not an :class:`InferenceRunner`.
    """
    from physicalai.inference.component_factory import instantiate_component, resolve_artifact  # noqa: PLC0415

    if export_dir is not None:
        spec = resolve_artifact(spec, export_dir)

    runner = instantiate_component(spec)
    if not isinstance(runner, InferenceRunner):
        msg = f"Configured runner is not an InferenceRunner: {type(runner).__name__}"
        raise TypeError(msg)
    return runner


def _extract_runner_spec(manifest_dict: dict[str, Any]) -> dict[str, Any] | None:
    """Extract a runner spec dict from nested or flat manifest data.

    Returns:
        Runner spec dict if found, otherwise ``None``.
    """
    model_section = manifest_dict.get("model", {})
    if isinstance(model_section, dict):
        runner = model_section.get("runner")
        if isinstance(runner, dict) and ("class_path" in runner or "type" in runner):
            return runner

    runner = manifest_dict.get("runner")
    if isinstance(runner, dict) and ("class_path" in runner or "type" in runner):
        return runner

    return None
