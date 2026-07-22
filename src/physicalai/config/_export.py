# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Opt-in constructor-config export for live components."""

from __future__ import annotations

import functools
import inspect
from typing import TypeVar

from ._errors import ComponentConfigError
from ._normalize import (
    normalize_value,
    snapshot_captured_value,
    validate_component_config,
)
from ._path import format_path
from ._types import (
    _CAPTURED_INIT_ARGS_ATTR,
    _CONFIG_CLASS_PATH_ATTR,
    _CONFIG_HOOK_NAME,
    _EXPORT_DEPTH_ATTR,
    _EXPORT_MARKER_ATTR,
    _MAX_CONFIG_DEPTH,
    ComponentConfig,
    JsonValue,
)
from .importing import import_dotted_path

_T = TypeVar("_T", bound=type)


def _has_export_marker(obj: object) -> bool:
    return bool(getattr(obj, _EXPORT_MARKER_ATTR, False))


def _has_config_hook(value: object) -> bool:
    hook = getattr(value, _CONFIG_HOOK_NAME, None)
    return callable(hook)


def is_config_exportable(value: object) -> bool:
    """Return whether *value* can export a :class:`ComponentConfig`.

    A value is exportable if and only if its most-derived class's effective
    ``__init__`` carries the ``@export_config`` marker, or the instance
    provides a callable ``__component_config__`` hook.
    """
    if _has_config_hook(value):
        return True
    init = type(value).__init__
    return _has_export_marker(init)


def _resolve_public_class_path(cls: type) -> str:
    explicit = getattr(cls, _CONFIG_CLASS_PATH_ATTR, None)
    path = explicit if isinstance(explicit, str) and explicit else f"{cls.__module__}.{cls.__qualname__}"

    if "<locals>" in cls.__qualname__:
        msg = f"local class {path!r} cannot export a stable class_path"
        raise ComponentConfigError(msg)

    try:
        resolved = import_dotted_path(path)
    except (ValueError, ImportError, AttributeError) as exc:
        msg = f"class_path {path!r} for {cls.__qualname__} is not importable: {exc}"
        raise ComponentConfigError(msg) from exc
    if resolved is not cls:
        msg = f"class_path {path!r} resolves to {resolved!r}, expected exactly {cls!r}"
        raise ComponentConfigError(msg)
    return path


def _component_path_prefix(value: object) -> str:
    cls = type(value)
    explicit = getattr(cls, _CONFIG_CLASS_PATH_ATTR, None)
    if isinstance(explicit, str) and explicit:
        return explicit
    return f"{cls.__module__}.{cls.__qualname__}"


def _arg_path(prefix: str, class_path: str, key: str) -> str:
    if prefix:
        return f"{prefix}.init_args.{key}"
    return f"{class_path}.init_args.{key}"


def to_config(
    value: object,
    *,
    _path: str = "",
    _depth: int = 0,
    _seen: set[int] | None = None,
) -> ComponentConfig:
    """Export an opted-in live component as JSON-safe ``class_path`` + ``init_args``.

    Args:
        value: An instance whose class uses ``@export_config``, or which
            implements ``__component_config__``.

    Returns:
        A :class:`ComponentConfig` describing the object as constructed.

    Raises:
        ComponentConfigError: If *value* is not exportable or captured values
            cannot be normalized.
    """
    if _depth > _MAX_CONFIG_DEPTH:
        msg = f"{format_path(_path)}: nesting depth exceeds {_MAX_CONFIG_DEPTH}"
        raise ComponentConfigError(msg)

    seen = _seen if _seen is not None else set()

    if _has_config_hook(value):
        hook = getattr(value, _CONFIG_HOOK_NAME)
        raw = hook()
        path_prefix = _path or _component_path_prefix(value)
        validated = validate_component_config(raw, path=path_prefix)
        normalized_args: dict[str, JsonValue] = {
            key: normalize_value(
                item,
                path=_arg_path(_path, validated["class_path"], key),
                depth=_depth + 1,
                seen=seen,
                to_config=to_config,
                is_exportable=is_config_exportable,
            )
            for key, item in validated["init_args"].items()
        }
        return {"class_path": validated["class_path"], "init_args": normalized_args}

    if not _has_export_marker(type(value).__init__):
        msg = (
            f"{_path or _component_path_prefix(value)}: not config-exportable; "
            "decorate the concrete class with @export_config or implement "
            "__component_config__"
        )
        raise ComponentConfigError(msg)

    captured = getattr(value, _CAPTURED_INIT_ARGS_ATTR, None)
    if captured is None:
        msg = (
            f"{_path or _component_path_prefix(value)}: no captured constructor "
            "arguments; the object was not constructed through the decorated __init__"
        )
        raise ComponentConfigError(msg)

    class_path = _resolve_public_class_path(type(value))
    init_args: dict[str, JsonValue] = {}
    for key, item in captured.items():
        init_args[key] = normalize_value(
            item,
            path=_arg_path(_path, class_path, key),
            depth=_depth + 1,
            seen=seen,
            to_config=to_config,
            is_exportable=is_config_exportable,
        )
    return {"class_path": class_path, "init_args": init_args}


def _instance_to_config(self: object) -> ComponentConfig:
    """Instance-method sugar for :func:`to_config`; prefer the module function.

    Returns:
        The component config for *self*.
    """
    return to_config(self)


def _validate_replayable_signature(cls: type, signature: inspect.Signature) -> None:
    for param in signature.parameters.values():
        if param.name == "self":
            continue
        if param.kind is inspect.Parameter.POSITIONAL_ONLY:
            msg = (
                f"@export_config on {cls.__qualname__}: positional-only parameter "
                f"{param.name!r} is not replayable through keyword init_args"
            )
            raise TypeError(msg)
        if param.kind is inspect.Parameter.VAR_POSITIONAL:
            msg = f"@export_config on {cls.__qualname__}: *args is not replayable through keyword init_args"
            raise TypeError(msg)


def export_config(cls: _T) -> _T:
    """Opt a concrete class into constructor-config export via :func:`to_config`.

    Remembers caller-supplied ``__init__`` arguments (not defaults). Rejects
    constructors that declare positional-only parameters or ``*args``. Injects
    an instance ``to_config()`` convenience method; library code should still
    call the module-level :func:`to_config`.

    Inheritance:

    - Decorate every concrete class that **overrides** ``__init__``.
    - An undecorated overriding ``__init__`` fails at :func:`to_config` (partial
      base recipes are not emitted).
    - A subclass that inherits a decorated constructor unchanged remains valid
      without re-decorating.
    - Do not apply ``@export_config`` to a subclass that does not define its
      own ``__init__`` (that would double-wrap the inherited wrapper).

    Args:
        cls: Class to decorate. Must define ``__init__`` in its own class body.

    Returns:
        The same class with a wrapped ``__init__``.

    Raises:
        TypeError: If *cls* is not a type, has no own ``__init__``, or its
            ``__init__`` is not replayable.
    """
    if not isinstance(cls, type):
        msg = f"@export_config expects a class, got {type(cls).__name__}"
        raise TypeError(msg)

    if "__init__" not in cls.__dict__:
        msg = (
            f"@export_config on {cls.__qualname__}: decorate only classes that "
            "define their own __init__; inherited decorated constructors remain "
            "exportable without re-decorating"
        )
        raise TypeError(msg)

    original_init = cls.__dict__["__init__"]
    if original_init is object.__init__:
        msg = f"@export_config on {cls.__qualname__}: class has no custom __init__"
        raise TypeError(msg)

    # Re-decorating the same class body is a no-op.
    if _has_export_marker(original_init):
        return cls

    signature = inspect.signature(original_init)
    _validate_replayable_signature(cls, signature)

    @functools.wraps(original_init)
    def wrapped_init(self: object, *args: object, **kwargs: object) -> None:
        bound = signature.bind(self, *args, **kwargs)
        # Do not apply_defaults — omit unsupplied arguments so reconstruction
        # uses current constructor defaults.
        supplied = {
            name: snapshot_captured_value(value, keep_by_reference=is_config_exportable)
            for name, value in bound.arguments.items()
            if name != "self"
        }
        # Flatten **kwargs mapping into init_args.
        var_kw_name = next(
            (name for name, param in signature.parameters.items() if param.kind is inspect.Parameter.VAR_KEYWORD),
            None,
        )
        if var_kw_name is not None and var_kw_name in supplied:
            extra = supplied.pop(var_kw_name)
            if not isinstance(extra, dict):
                msg = f"{cls.__qualname__}: **{var_kw_name} must be a mapping"
                raise TypeError(msg)
            for key, value in extra.items():
                if not isinstance(key, str):
                    msg = f"{cls.__qualname__}: **{var_kw_name} keys must be strings"
                    raise TypeError(msg)
                supplied[key] = snapshot_captured_value(value, keep_by_reference=is_config_exportable)

        depth = getattr(self, _EXPORT_DEPTH_ATTR, 0)
        setattr(self, _EXPORT_DEPTH_ATTR, depth + 1)
        try:
            original_init(self, *args, **kwargs)
            # Only the outermost successful decorated constructor commits.
            if getattr(self, _EXPORT_DEPTH_ATTR, 0) == 1:
                setattr(self, _CAPTURED_INIT_ARGS_ATTR, supplied)
        finally:
            setattr(self, _EXPORT_DEPTH_ATTR, depth)
            if depth == 0 and hasattr(self, _EXPORT_DEPTH_ATTR):
                delattr(self, _EXPORT_DEPTH_ATTR)

    setattr(wrapped_init, _EXPORT_MARKER_ATTR, True)
    cls.__init__ = wrapped_init  # type: ignore[method-assign]
    # Convenience method; type checkers do not see the injection — prefer module to_config.
    cls.to_config = _instance_to_config  # type: ignore[attr-defined]
    return cls
