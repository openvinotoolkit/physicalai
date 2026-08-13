# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Component registry and factory for dynamic instantiation.

The :class:`ComponentRegistry` maps short names (e.g. ``"single_pass"``)
to fully-qualified class paths so that manifests can use concise
identifiers instead of full dotted paths.  The :func:`instantiate_component`
factory resolves a :class:`~physicalai.inference.manifest.ComponentSpec`
to an object instance, supporting both ``type`` + flat params and
``class_path`` + ``init_args`` resolution modes.
"""

from __future__ import annotations

import os
from argparse import ArgumentError
from pathlib import Path

from jsonargparse import ArgumentParser

from physicalai.config import Config

from ._importing import import_dotted_path
from .manifest import ComponentSpec


class ComponentRegistry:
    """Name → class_path registry for dynamically instantiated components.

    Built-in entries are registered at module load time.  Domain layers
    can register additional entries via :meth:`register`.

    Examples:
        >>> registry = ComponentRegistry()
        >>> registry.register("my_runner", "myapp.runners.MyRunner")
        >>> registry.resolve("my_runner")
        'myapp.runners.MyRunner'
        >>> registry.resolve("myapp.runners.MyRunner")  # passthrough
        'myapp.runners.MyRunner'
    """

    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._entries: dict[str, str] = {}

    def register(self, name: str, class_path: str) -> None:
        """Register a short name → class_path mapping.

        Args:
            name: Short identifier (e.g. ``"single_pass"``).
            class_path: Fully-qualified class path.
        """
        self._entries[name] = class_path

    def resolve(self, name_or_path: str) -> str:
        """Resolve a short name to a class path, or pass through if already qualified.

        Args:
            name_or_path: Either a registered short name or a full class path.

        Returns:
            Fully-qualified class path.
        """
        return self._entries.get(name_or_path, name_or_path)

    def get_class(self, name_or_path: str) -> type:
        """Resolve a name to a class path, import the module, and return the class.

        Args:
            name_or_path: Either a registered short name or a full class path.

        Returns:
            The resolved class object.
        """
        return _import_class(self.resolve(name_or_path))

    def entries(self) -> dict[str, str]:
        """Return a copy of all registered entries.

        Returns:
            Dict mapping short names to class paths.
        """
        return dict(self._entries)

    def __contains__(self, name: str) -> bool:
        """Check if *name* is a registered short name.

        Returns:
            True if *name* is registered.
        """
        return name in self._entries

    def __repr__(self) -> str:
        """Return a developer-friendly representation."""
        return f"ComponentRegistry({self._entries!r})"


component_registry = ComponentRegistry()

# Runners
component_registry.register("single_pass", "physicalai.inference.runners.SinglePass")

# Preprocessors
component_registry.register("normalize", "physicalai.inference.preprocessors.StatsNormalizer")
component_registry.register("resize", "physicalai.inference.preprocessors.ResizePreprocessor")
component_registry.register("smolvla_resize", "physicalai.inference.preprocessors.ResizeSmolVLA")
component_registry.register("new_line", "physicalai.inference.preprocessors.NewLinePreprocessor")
component_registry.register("hf_tokenizer", "physicalai.inference.preprocessors.HFTokenizer")
component_registry.register("ov_tokenizer", "physicalai.inference.preprocessors.OVTokenizer")
component_registry.register("pi05", "physicalai.inference.preprocessors.Pi05Preprocessor")
component_registry.register("to_float_tensor", "physicalai.inference.preprocessors.ToFloatTensorPreprocessor")

# Postprocessors
component_registry.register("denormalize", "physicalai.inference.postprocessors.StatsDenormalizer")
component_registry.register("action_chunk_trimmer", "physicalai.inference.postprocessors.ActionChunkTrimmer")


def resolve_artifact(spec: ComponentSpec, export_dir: Path) -> ComponentSpec:
    """Resolve relative ``artifact`` paths to absolute paths.

    For type-based specs, resolves a relative ``artifact`` flat
    param to an absolute path.  For class_path-based specs,
    resolves a relative ``artifact`` in ``init_args``.

    Args:
        spec: Component descriptor that may contain a relative artifact path.
        export_dir: Base directory for resolving relative paths.

    Returns:
        The spec with resolved artifact path, or the original spec
        unchanged if no resolution is needed.
    """
    norm_export = Path(export_dir).resolve()

    def _resolve_artifact_path(artifact: str) -> str:
        # Reject manifest paths that escape the export directory via
        # "../" traversal (e.g. "../../etc/passwd").  The check is intentionally
        # lexical (normpath, no symlink following) so that HuggingFace Hub
        # snapshot symlinks which point from snapshot/ into a sibling blobs/
        # store are accepted without error.
        candidate = Path(os.path.normpath(norm_export / artifact))
        if not candidate.is_relative_to(norm_export):
            msg = f"artifact path {artifact!r} escapes the export directory"
            raise ValueError(msg)
        return str(candidate)

    flat = spec.flat_params
    if "artifact" in flat:
        # Absolute paths joining onto norm_export drop the left side (pathlib semantics),
        # so _resolve_artifact_path catches absolute escapes the same as relative traversal.
        new_params = {**flat, "artifact": _resolve_artifact_path(flat["artifact"])}
        return type(spec).model_validate({"type": spec.type, **new_params})

    if spec.class_path and "artifact" in spec.init_args:
        artifact = spec.init_args["artifact"]
        new_init_args = {**spec.init_args, "artifact": _resolve_artifact_path(artifact)}
        return type(spec).model_validate({"class_path": spec.class_path, "init_args": new_init_args})

    return spec


def _import_class(class_path: str) -> type:
    """Import and return a class from a fully-qualified dotted path.

    Returns:
        The imported class object.

    Raises:
        TypeError: If the resolved object is not a class.
    """
    obj = import_dotted_path(class_path)
    if not isinstance(obj, type):
        msg = f"{class_path!r} does not resolve to a class (got {type(obj).__name__})"
        raise TypeError(msg)
    return obj


def instantiate_component(
    base: type | ComponentSpec,
    spec: ComponentSpec | None = None,
    *,
    registry: ComponentRegistry | None = None,
) -> object:
    """Import the class described by *spec* and return a live instance.

    Supports two resolution modes:

    1. **class_path + init_args** — ``spec.class_path`` is resolved
       (via registry if it's a short name) and the class is
       instantiated with ``spec.init_args``.
    2. **type + flat params** — ``spec.type`` is resolved via the
       registry to a class path, and ``spec.flat_params`` are passed
       as keyword arguments.

    ``class_path`` takes precedence when both are present.

    Nested typed component values are instantiated by jsonargparse according
    to the resolved class constructor annotations.

    Args:
        base: Expected component base type, or a component spec for the
            deprecated one-argument compatibility form.
        spec: Component descriptor with type or class_path.
        registry: Optional registry for short-name resolution.
            Defaults to :data:`component_registry`.

    Returns:
        An instance of the resolved class.

    Raises:
        TypeError: If the resolved recipe does not describe a class recipe.
    """
    if spec is None:
        spec = base
        if not isinstance(spec, ComponentSpec):
            msg = "instantiate_component requires a component spec"
            raise TypeError(msg)
        base = _import_class(component_registry.resolve(spec.class_path or spec.type))
    reg = registry or component_registry
    canonical = _canonical_spec(spec, registry=reg)
    class_path = canonical["class_path"]
    init_args = canonical["init_args"]
    if not isinstance(class_path, str) or not isinstance(init_args, dict):
        msg = "Resolved component spec is not a valid class recipe"
        raise TypeError(msg)
    parser = ArgumentParser(exit_on_error=False)
    parser.add_subclass_arguments(base, "component", required=True)
    try:
        namespace = parser.parse_object(
            {"component": {"class_path": class_path, "init_args": init_args}},
            defaults=False,
        )
        return parser.instantiate(namespace).component
    except ArgumentError as exc:
        raise TypeError(str(exc)) from exc


def _canonical_spec(spec: ComponentSpec, *, registry: ComponentRegistry) -> dict[str, object]:
    """Resolve aliases and normalize a manifest component to constructor args.

    Returns:
        A canonical ``class_path`` and ``init_args`` mapping.
    """
    if spec.class_path:
        class_path = registry.resolve(spec.class_path)
        init_args = spec.init_args
    else:
        class_path = registry.resolve(spec.type)
        init_args = spec.flat_params
    return Config(class_path, init_args).to_dict()
