# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Fuzz ComponentSpec resolution.

Four sub-targets: path traversal prevention, depth-limit enforcement,
flat-params injection, and symlink-bypass behaviour documentation.
"""
from __future__ import annotations

import os
import sys
import tempfile
from argparse import ArgumentError
from pathlib import Path

import atheris

with atheris.instrument_imports():
    from pydantic import ValidationError
    from physicalai.config import ConfigError
    from physicalai.inference.manifest import ComponentSpec
    from physicalai.inference.component_factory import instantiate_component, resolve_artifact

# Depth limit value — mirrors _MAX_CONFIG_DEPTH in physicalai.config._types
_MAX_DEPTH = 10

# Registry short names safe to instantiate for flat-params injection tests
_SAFE_TYPES = [
    "single_pass",
    "action_chunk_trimmer",
]


@atheris.instrument_func
def _sub_resolve_artifact(fdp: atheris.FuzzedDataProvider, export_dir: str) -> None:
    """Assert that resolve_artifact never returns a path outside export_dir."""
    artifact = fdp.ConsumeUnicodeNoSurrogates(256)
    try:
        spec = ComponentSpec.model_validate({"type": "normalize", "artifact": artifact})
    except ValidationError:
        return

    try:
        result = resolve_artifact(spec, Path(export_dir))
    except ValueError:
        return  # Expected for traversal attempts

    resolved = result.flat_params.get("artifact", "")
    if resolved and os.path.isabs(resolved):
        real_export = Path(os.path.realpath(export_dir))
        real_resolved = Path(os.path.normpath(resolved))
        assert real_resolved.is_relative_to(real_export), (
            f"PATH TRAVERSAL: artifact {artifact!r} resolved to {resolved!r}, "
            f"which escapes export_dir {export_dir!r}"
        )


@atheris.instrument_func
def _sub_depth_limit(fdp: atheris.FuzzedDataProvider) -> None:
    """Assert that nesting beyond _MAX_DEPTH raises before unbounded recursion."""
    depth = fdp.ConsumeIntInRange(1, _MAX_DEPTH + 8)

    # Build a chain: each level wraps the previous via init_args
    inner: dict = {
        "class_path": "physicalai.inference.runners.SinglePass",
        "init_args": {},
    }
    for _ in range(depth):
        inner = {
            "class_path": "physicalai.inference.postprocessors.ActionNormalizer",
            "init_args": {"_child": inner},
        }

    try:
        spec = ComponentSpec.model_validate(inner)
        instantiate_component(spec)
    except (ValueError, ConfigError) as exc:
        if depth > _MAX_DEPTH:
            message = str(exc).lower()
            assert "depth" in message or "nesting" in message, (
                f"Expected depth-limit error for depth={depth}, got: {exc}"
            )
    except (TypeError, AttributeError, ArgumentError):
        pass  # Constructor or parser errors are fine; what matters is no uncaught exit


@atheris.instrument_func
def _sub_flat_params(fdp: atheris.FuzzedDataProvider) -> None:
    """Pass unexpected flat kwargs to known-safe registered component types."""
    chosen = fdp.PickValueInList(_SAFE_TYPES)
    n_extra = fdp.ConsumeIntInRange(0, 8)
    extra = {
        fdp.ConsumeUnicodeNoSurrogates(16): fdp.ConsumeUnicodeNoSurrogates(32)
        for _ in range(n_extra)
    }
    spec_dict = {"type": chosen, **extra}
    try:
        spec = ComponentSpec.model_validate(spec_dict)
        instantiate_component(spec)
    except (TypeError, ValueError, AttributeError, ArgumentError):
        pass


@atheris.instrument_func
def _sub_symlink_bypass(export_dir: str) -> None:
    """Verify the documented symlink bypass: resolve_artifact() accepts intra-dir symlinks.

    The traversal check is intentionally lexical (normpath, not realpath) so that
    HuggingFace Hub snapshot symlinks pass.  This sub-target ensures that behaviour
    is stable — if it starts raising ValueError the design changed and needs review.
    """
    import tempfile
    from pathlib import Path

    # Create a file outside export_dir that the symlink will point to.
    with tempfile.NamedTemporaryFile(delete=False) as outside_file:
        outside_path = outside_file.name

    try:
        link_name = "symlinked_artifact"
        link_path = os.path.join(export_dir, link_name)
        try:
            os.symlink(outside_path, link_path)
        except (OSError, NotImplementedError):
            return  # symlinks unavailable in this sandbox

        try:
            spec = ComponentSpec.model_validate(
                {"type": "normalize", "artifact": link_name}
            )
        except ValidationError:
            return

        try:
            result = resolve_artifact(spec, Path(export_dir))
        except ValueError:
            # The bypass was closed — surface as assertion so it's not silently skipped
            raise AssertionError(
                f"resolve_artifact raised ValueError for intra-dir symlink "
                f"{link_name!r} → {outside_path!r}.  This closes the documented "
                f"lexical bypass.  Verify this is intentional before silencing."
            ) from None

        returned = result.flat_params.get("artifact", "")
        if returned:
            assert Path(os.path.normpath(returned)).is_relative_to(
                Path(os.path.realpath(export_dir))
            ), (
                f"Symlink artifact resolved to {returned!r} which escapes "
                f"export_dir {export_dir!r}"
            )
    finally:
        try:
            os.unlink(outside_path)
        except OSError:
            pass



def test_one_input(data: bytes) -> None:
    if len(data) < 4:
        return
    fdp = atheris.FuzzedDataProvider(data)
    sub = fdp.ConsumeIntInRange(0, 3)

    if sub == 0:
        with tempfile.TemporaryDirectory() as export_dir:
            _sub_resolve_artifact(fdp, export_dir)
    elif sub == 1:
        _sub_depth_limit(fdp)
    elif sub == 2:
        _sub_flat_params(fdp)
    else:
        with tempfile.TemporaryDirectory() as export_dir:
            _sub_symlink_bypass(export_dir)


def main() -> None:
    atheris.Setup(sys.argv, test_one_input)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
