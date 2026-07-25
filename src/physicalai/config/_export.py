# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Opt-in constructor-config export for live components."""

from __future__ import annotations

import functools
import inspect
from typing import TYPE_CHECKING, TypeVar, overload

from ._errors import ComponentConfigError
from ._normalize import (
    normalize_value,
    snapshot_captured_value,
)
from ._path import format_path
from ._types import (
    _CAPTURED_INIT_ARGS_ATTR,
    _CONFIG_ARGS_ATTR,
    _CONFIG_CLASS_PATH_ATTR,
    _EXPORT_DEPTH_ATTR,
    _EXPORT_MARKER_ATTR,
    _MAX_CONFIG_DEPTH,
    _NORMALIZE_CAPTURED_INIT_ARGS_ATTR,
    ComponentConfig,
    JsonValue,
)
from .importing import import_dotted_path

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

_T = TypeVar("_T", bound=type)


class _NonScalarVarKwarg:
    """Poison value so ``to_config`` rejects non-scalar ``**kwargs`` entries.

    Used when ``@export_config(scalar_var_kwargs=True)`` seals flattened
    var-keyword arguments to JSON scalars only.
    """

    __slots__ = ("_name",)

    def __init__(self, name: str) -> None:
        self._name = name

    def __repr__(self) -> str:
        return f"<non-scalar **kwargs value {self._name!r}>"


def _is_json_scalar(value: object) -> bool:
    """Return whether *value* is a :data:`~physicalai.config.JsonScalar`."""
    if value is None or isinstance(value, (bool, str)):
        return True
    if isinstance(value, int) and not isinstance(value, bool):
        return True
    return isinstance(value, float)


def _has_export_marker(obj: object) -> bool:
    return bool(getattr(obj, _EXPORT_MARKER_ATTR, False))


def _encode_domain_value(value: object) -> tuple[JsonValue] | None:
    """Encode a domain value via ``to_config_value()`` if present.

    Returns:
        ``(payload,)`` when a codec applies — the 1-tuple keeps a real JSON
        ``null`` payload distinct from "no codec" — or ``None`` when the value
        has no domain encoder. The payload is re-normalized by the caller.
    """
    encode = getattr(value, "to_config_value", None)
    if not callable(encode):
        return None
    return (encode(),)  # type: ignore[return-value]


def is_config_exportable(value: object) -> bool:
    """Return whether *value* can export a :class:`ComponentConfig`.

    A value is exportable if and only if its most-derived class's effective
    ``__init__`` carries the ``@export_config`` marker.
    """
    init = type(value).__init__
    return _has_export_marker(init)


def declared_config_args(cls: type) -> frozenset[str]:
    """Return init-arg names *cls* consumes as ComponentConfig data.

    Declared via ``@export_config(config_args=...)``. :func:`instantiate`
    passes these arguments through as plain mappings instead of building the
    nested component, so the component receives the recipe it needs to hand to
    another process rather than a live object it would immediately discard.

    Returns:
        The declared argument names, or an empty set when none are declared.
    """
    return getattr(cls.__init__, _CONFIG_ARGS_ATTR, frozenset())


def _class_path_override(cls: type) -> str | None:
    """Return the decorator ``class_path=`` only when *cls* owns the decorated ``__init__``.

    Returns:
        The override path, or ``None`` for inherited constructors (subclasses
        export ``__module__.__qualname__`` unless they re-decorate).
    """
    owned_init = cls.__dict__.get("__init__")
    if owned_init is None:
        return None
    explicit = getattr(owned_init, _CONFIG_CLASS_PATH_ATTR, None)
    if isinstance(explicit, str) and explicit:
        return explicit
    return None


def resolve_public_class_path(cls: type) -> str:
    """Resolve the stable public ``class_path`` for an opted-in component class.

    Uses the decorator ``class_path=`` override when present, otherwise
    ``cls.__module__ + "." + cls.__qualname__``. Verifies that importing the
    path yields exactly *cls*.

    Args:
        cls: The concrete class to resolve.

    Returns:
        The importable public dotted path.

    Raises:
        ComponentConfigError: If *cls* is local, the path is not importable, or
            the path resolves to a different object.
    """
    path = _class_path_override(cls) or f"{cls.__module__}.{cls.__qualname__}"

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
    return _class_path_override(cls) or f"{cls.__module__}.{cls.__qualname__}"


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

    Nested constructor values may be components (``@export_config``) or domain
    values with :meth:`~ConfigValue.to_config_value` (re-normalized; absence of
    the method means no codec; a returned ``None`` is JSON null).

    Args:
        value: An instance whose class uses ``@export_config``.

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

    if not _has_export_marker(type(value).__init__):
        msg = (
            f"{_path or _component_path_prefix(value)}: not config-exportable; "
            "decorate the concrete class with @export_config"
        )
        raise ComponentConfigError(msg)

    captured = getattr(value, _CAPTURED_INIT_ARGS_ATTR, None)
    if captured is None:
        msg = (
            f"{_path or _component_path_prefix(value)}: no captured constructor "
            "arguments; the object was not constructed through the decorated __init__"
        )
        raise ComponentConfigError(msg)

    class_path = resolve_public_class_path(type(value))
    init_args: dict[str, JsonValue] = {}
    for key, item in captured.items():
        init_args[key] = normalize_value(
            item,
            path=_arg_path(_path, class_path, key),
            depth=_depth + 1,
            seen=seen,
            to_config=to_config,
            is_exportable=is_config_exportable,
            domain_codec=_encode_domain_value,
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


def _flatten_var_kwargs(
    supplied: dict[str, object],
    *,
    cls_name: str,
    var_kw_name: str,
    scalar_var_kwargs: bool,
) -> None:
    """Move flattened ``**kwargs`` entries into *supplied*; optionally seal scalars.

    Raises:
        TypeError: If the var-keyword value is not a string-keyed mapping.
    """
    extra = supplied.pop(var_kw_name)
    if not isinstance(extra, dict):
        msg = f"{cls_name}: **{var_kw_name} must be a mapping"
        raise TypeError(msg)
    for key, value in extra.items():
        if not isinstance(key, str):
            msg = f"{cls_name}: **{var_kw_name} keys must be strings"
            raise TypeError(msg)
        if scalar_var_kwargs and not _is_json_scalar(value):
            # Seal so normalize fails at to_config (no silent JSON nest).
            supplied[key] = _NonScalarVarKwarg(key)
        else:
            supplied[key] = snapshot_captured_value(value)


def _validate_config_args(
    cls: type,
    signature: inspect.Signature,
    config_args: Sequence[str],
    var_kw_name: str | None,
) -> frozenset[str]:
    """Check that every declared config arg is a real keyword parameter.

    Returns:
        The validated names.

    Raises:
        TypeError: If a name is not a string, is not declared by ``__init__``,
            or names the ``**kwargs`` parameter.
    """
    names: set[str] = set()
    for name in config_args:
        if not isinstance(name, str) or not name:
            msg = f"@export_config on {cls.__qualname__}: config_args entries must be non-empty strings"
            raise TypeError(msg)
        if name == var_kw_name:
            msg = (
                f"@export_config on {cls.__qualname__}: config_args cannot name the "
                f"**{name} parameter; declare the individual arguments instead"
            )
            raise TypeError(msg)
        if name not in signature.parameters or name == "self":
            msg = f"@export_config on {cls.__qualname__}: config_args {name!r} is not an __init__ parameter"
            raise TypeError(msg)
        names.add(name)
    return frozenset(names)


def _decorate_export_config(
    cls: _T,
    *,
    class_path: str | None,
    scalar_var_kwargs: bool,
    config_args: Sequence[str] | None,
) -> _T:
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

    # Re-decorating the same class body is a no-op (keeps prior class_path if any).
    if _has_export_marker(original_init):
        return cls

    if class_path is not None and (not isinstance(class_path, str) or not class_path):
        msg = f"@export_config on {cls.__qualname__}: class_path must be a non-empty string"
        raise TypeError(msg)

    signature = inspect.signature(original_init)
    _validate_replayable_signature(cls, signature)

    var_kw_name = next(
        (name for name, param in signature.parameters.items() if param.kind is inspect.Parameter.VAR_KEYWORD),
        None,
    )
    if scalar_var_kwargs and var_kw_name is None:
        msg = f"@export_config on {cls.__qualname__}: scalar_var_kwargs=True requires a **kwargs parameter"
        raise TypeError(msg)

    declared = _validate_config_args(cls, signature, config_args or (), var_kw_name)

    @functools.wraps(original_init)
    def wrapped_init(self: object, *args: object, **kwargs: object) -> None:
        bound = signature.bind(self, *args, **kwargs)
        # Do not apply_defaults — omit unsupplied arguments so reconstruction
        # uses current constructor defaults.
        supplied = {name: snapshot_captured_value(value) for name, value in bound.arguments.items() if name != "self"}
        if var_kw_name is not None and var_kw_name in supplied:
            _flatten_var_kwargs(
                supplied,
                cls_name=cls.__qualname__,
                var_kw_name=var_kw_name,
                scalar_var_kwargs=scalar_var_kwargs,
            )

        depth = getattr(self, _EXPORT_DEPTH_ATTR, 0)
        setattr(self, _EXPORT_DEPTH_ATTR, depth + 1)
        try:
            original_init(self, *args, **kwargs)
            # Only the outermost successful decorated constructor commits.
            if getattr(self, _EXPORT_DEPTH_ATTR, 0) == 1:
                normalize_captured = getattr(self, _NORMALIZE_CAPTURED_INIT_ARGS_ATTR, None)
                if normalize_captured is not None:
                    if not callable(normalize_captured):
                        msg = f"{cls.__qualname__}: {_NORMALIZE_CAPTURED_INIT_ARGS_ATTR} must be callable"
                        raise TypeError(msg)
                    normalize_captured(supplied)
                setattr(self, _CAPTURED_INIT_ARGS_ATTR, supplied)
        finally:
            setattr(self, _EXPORT_DEPTH_ATTR, depth)
            if depth == 0 and hasattr(self, _EXPORT_DEPTH_ATTR):
                delattr(self, _EXPORT_DEPTH_ATTR)

    setattr(wrapped_init, _EXPORT_MARKER_ATTR, True)
    if class_path is not None:
        setattr(wrapped_init, _CONFIG_CLASS_PATH_ATTR, class_path)
    if declared:
        setattr(wrapped_init, _CONFIG_ARGS_ATTR, declared)
    cls.__init__ = wrapped_init  # type: ignore[method-assign]
    # Convenience method; type checkers do not see the injection — prefer module to_config.
    cls.to_config = _instance_to_config  # type: ignore[attr-defined]
    return cls


@overload
def export_config(cls: _T, /) -> _T: ...


@overload
def export_config(
    *,
    class_path: str | None = None,
    scalar_var_kwargs: bool = False,
    config_args: Sequence[str] | None = None,
) -> Callable[[_T], _T]: ...


def export_config(
    cls: _T | None = None,
    /,
    *,
    class_path: str | None = None,
    scalar_var_kwargs: bool = False,
    config_args: Sequence[str] | None = None,
) -> _T | Callable[[_T], _T]:
    """Opt a concrete class into constructor-config export via :func:`to_config`.

    Remembers caller-supplied ``__init__`` arguments (not defaults). Rejects
    constructors that declare positional-only parameters or ``*args``. Injects
    an instance ``to_config()`` convenience method; library code should still
    call the module-level :func:`to_config`.

    Usage::

        @export_config
        class MyRobot: ...

        @export_config(class_path="physicalai.robot.SO101")
        class SO101: ...

        @export_config(class_path="physicalai.inference.InferenceModel", scalar_var_kwargs=True)
        class InferenceModel: ...

    When ``class_path`` is omitted, export uses
    ``type(self).__module__ + "." + type(self).__qualname__``. Pass
    ``class_path=`` when the public import path differs from the defining
    module (for example a package re-export).

    Pass ``scalar_var_kwargs=True`` when flattened ``**kwargs`` must export as
    JSON scalars only (``None`` / ``bool`` / ``int`` / ``float`` / ``str``).
    Non-scalar var-keyword values then fail at :func:`to_config` instead of
    being normalized as nested JSON. Requires a ``**kwargs`` parameter.

    Nested non-component domain values (for example calibration objects) may
    implement :meth:`~ConfigValue.to_config_value` so they normalize to
    constructor-compatible JSON; that method's output is re-normalized.
    Absence of the method means no codec; a returned ``None`` is JSON null.

    Inheritance:

    - Decorate every concrete class that **overrides** ``__init__``.
    - An undecorated overriding ``__init__`` fails at :func:`to_config` (partial
      base recipes are not emitted).
    - A subclass that inherits a decorated constructor unchanged remains valid
      without re-decorating.
    - Do not apply ``@export_config`` to a subclass that does not define its
      own ``__init__`` (that would double-wrap the inherited wrapper).

    Args:
        cls: Class to decorate when used as ``@export_config``. Must define
            ``__init__`` in its own class body.
        class_path: Optional stable public import path for export. Verified on
            export to resolve exactly to the decorated class.
        scalar_var_kwargs: When ``True``, seal flattened ``**kwargs`` to JSON
            scalars so non-scalars fail at :func:`to_config`.
        config_args: Init-arg names the class consumes as ComponentConfig
            *data*. :func:`instantiate` passes these through as plain mappings
            instead of constructing the nested component — use it for spawn
            recipes that must cross a process boundary (for example
            ``SharedCamera(camera=...)``).

    Returns:
        The decorated class, or a decorator when keyword options are passed.
    """
    if cls is not None:
        return _decorate_export_config(
            cls,
            class_path=class_path,
            scalar_var_kwargs=scalar_var_kwargs,
            config_args=config_args,
        )

    def decorator(target: _T) -> _T:
        return _decorate_export_config(
            target,
            class_path=class_path,
            scalar_var_kwargs=scalar_var_kwargs,
            config_args=config_args,
        )

    return decorator
