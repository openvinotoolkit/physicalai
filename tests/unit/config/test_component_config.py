# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# ruff:file-ignore[undocumented-public-module, undocumented-public-class, undocumented-public-method, undocumented-public-init, undocumented-magic-method, bad-dunder-method-name, magic-value-comparison, no-self-use, assert, unused-method-argument, too-many-public-methods]

from __future__ import annotations

import inspect
import json
import math
import sys
from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from typing import Protocol, cast, runtime_checkable
from unittest.mock import MagicMock, patch

import pytest

from physicalai.config import (
    ComponentConfig,
    ComponentConfigError,
    ComponentImportError,
    JsonValue,
    export_config,
    import_dotted_path,
    instantiate,
    is_config_exportable,
    normalize_component_config,
    to_config,
)

# Matches physicalai.config._types._MAX_CONFIG_DEPTH
_MAX_CONFIG_DEPTH = 10


def _as_mapping(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return value


def _as_list(value: object) -> list[object]:
    assert isinstance(value, list)
    return value


class Color(Enum):
    RED = "red"
    BLUE = "blue"


@runtime_checkable
class Named(Protocol):
    @property
    def name(self) -> str: ...


@export_config
class Point:
    def __init__(self, x: int, y: int = 0) -> None:
        self.x = x
        self.y = y


@export_config
class Box:
    def __init__(self, origin: Point, label: str = "box") -> None:
        self.origin = origin
        self.label = label


@export_config
class Nest:
    def __init__(self, child: Nest | None = None) -> None:
        self.child = child


@export_config
class BadInner:
    def __init__(self, fn: object) -> None:
        self.fn = fn


@export_config
class Outer:
    def __init__(self, child: BadInner) -> None:
        self.child = child


@export_config
class BaseWidget:
    def __init__(self, name: str) -> None:
        self.name = name


@export_config
class DerivedWidget(BaseWidget):
    def __init__(self, name: str, size: int) -> None:
        super().__init__(name)
        self.size = size


class UndecoratedOverride(BaseWidget):
    def __init__(self, name: str, extra: int) -> None:
        super().__init__(name)
        self.extra = extra


class InheritsDecorated(BaseWidget):
    """Subclass that inherits the decorated constructor unchanged."""


@export_config
class WithExtras:
    def __init__(self, base: int, **kwargs: object) -> None:
        self.base = base
        self.kwargs = kwargs


@export_config(scalar_var_kwargs=True)
class ScalarVarKwargs:
    def __init__(self, base: int, **kwargs: object) -> None:
        self.base = base
        self.kwargs = kwargs


@export_config
class Leaf:
    def __init__(self, value: int) -> None:
        self.value = value


@export_config(config_args=("recipe",))
class Holder:
    def __init__(self, recipe: ComponentConfig, eager: object) -> None:
        self.recipe = recipe
        self.eager = eager


@export_config
class PathHolder:
    def __init__(self, path: str | Path) -> None:
        self.path = path


@export_config
class EnumHolder:
    def __init__(self, color: Color | str) -> None:
        self.color = Color(color) if not isinstance(color, Color) else color


@export_config
class MappingHolder:
    def __init__(self, data: Mapping[str, object]) -> None:
        self.data = dict(data)


@export_config
class ListHolder:
    def __init__(self, items: list[object]) -> None:
        self.items = list(items)


@export_config
class OptionalName:
    def __init__(self, name: str | None = "default") -> None:
        self.name = name


@export_config
class CanonicalName:
    normalize_calls = 0

    def __init__(self, name: str = "default") -> None:
        self.name = name

    @classmethod
    def _physicalai_normalize_captured_init_args(cls, supplied: dict[str, object]) -> None:
        cls.normalize_calls += 1
        if "name" in supplied:
            supplied["name"] = str(supplied["name"]).lower()


@export_config
class Boom:
    def __init__(self, x: int) -> None:
        msg = "nope"
        raise RuntimeError(msg)


@export_config
class CtorBoom:
    def __init__(self, x: int) -> None:
        msg = "boom"
        raise ValueError(msg)


class DomainPayload:
    """Domain value that encodes via ``to_config_value()`` (not a component)."""

    def __init__(self, amount: int) -> None:
        self.amount = amount

    def to_config_value(self) -> dict[str, int]:
        return {"amount": self.amount}


class BadNanDomain:
    def to_config_value(self) -> dict[str, float]:
        return {"x": math.nan}


class BadReservedDomain:
    def to_config_value(self) -> dict[str, object]:
        return {"class_path": "not.a.component", "other": 1}


class NullDomain:
    def to_config_value(self) -> None:
        return None


class CodecPeer:
    """Domain value that can point at another domain value for cycle tests."""

    def __init__(self) -> None:
        self.other: object | None = None

    def to_config_value(self) -> object:
        return self.other


@export_config
class DomainHolder:
    def __init__(self, payload: object) -> None:
        self.payload = payload


@export_config(class_path="tests.unit.config.test_component_config.ExportAlias")
class _HiddenExport:
    """Defining-module class with a public re-export alias (see ExportAlias)."""

    def __init__(self, x: int) -> None:
        self.x = x


ExportAlias = _HiddenExport


class InheritsAliasedExport(_HiddenExport):
    """Inherits decorated constructor; must not inherit ``class_path=`` override."""


class TestImportDottedPath:
    def test_resolves_nested_class(self) -> None:
        assert import_dotted_path("tests.unit.config.test_component_config.Point") is Point

    def test_real_import_failures_are_not_masked(self, tmp_path: Path) -> None:
        pkg = tmp_path / "mask_pkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("", encoding="utf-8")
        (pkg / "broken.py").write_text(
            "import totally_missing_dep_xyz\n\nclass Robot:\n    pass\n",
            encoding="utf-8",
        )
        (pkg / "raises_ie.py").write_text(
            "raise ImportError('boom from module body')\n",
            encoding="utf-8",
        )
        sys.path.insert(0, str(tmp_path))
        try:
            with pytest.raises(ModuleNotFoundError, match="totally_missing_dep_xyz"):
                import_dotted_path("mask_pkg.broken.Robot")
            with pytest.raises(ImportError, match="boom from module body"):
                import_dotted_path("mask_pkg.raises_ie.Robot")
        finally:
            sys.path.remove(str(tmp_path))
            sys.modules.pop("mask_pkg.broken", None)
            sys.modules.pop("mask_pkg.raises_ie", None)
            sys.modules.pop("mask_pkg", None)

    def test_unimportable_prefix_raises(self) -> None:
        with pytest.raises(ValueError, match="could not import"):
            import_dotted_path("totally.unknown.module.Cls")


class TestNormalizeComponentConfig:
    def test_rejects_nan(self) -> None:
        with pytest.raises(ValueError, match="JSON-serializable"):
            normalize_component_config(
                {"class_path": f"{__name__}.Point", "init_args": {"x": math.nan}},
                component_key="robot",
                class_label="robot_class",
            )

    def test_rejects_infinity(self) -> None:
        with pytest.raises(ValueError, match="JSON-serializable"):
            normalize_component_config(
                {"class_path": f"{__name__}.Point", "init_args": {"x": float("inf")}},
                component_key="camera",
                class_label="camera_class",
            )

    def test_rejects_non_serializable_object(self) -> None:
        with pytest.raises(ValueError, match="JSON-serializable"):
            normalize_component_config(
                {"class_path": f"{__name__}.Point", "init_args": {"x": object()}},
                component_key="robot",
                class_label="robot_class",
            )


class TestNormalizeAndInstantiate:
    def test_primitives_round_trip(self) -> None:
        point = Point(1, y=2)
        config = to_config(point)
        wire = json.loads(json.dumps(config))
        restored = instantiate(wire)
        assert isinstance(restored, Point)
        assert restored.x == 1
        assert restored.y == 2
        assert to_config(restored) == wire

    def test_omitted_defaults_stay_omitted(self) -> None:
        point = Point(3)
        config = to_config(point)
        assert config["init_args"] == {"x": 3}
        restored = cast(Point, instantiate(config))
        assert restored.y == 0

    def test_explicit_none_is_preserved(self) -> None:
        obj = OptionalName(None)
        config = to_config(obj)
        assert config["init_args"] == {"name": None}
        restored = cast(OptionalName, instantiate(config))
        assert restored.name is None

    def test_path_as_given(self) -> None:
        relative = PathHolder(Path("calib.json"))
        absolute = PathHolder(Path("/var/calib.json"))
        assert to_config(relative)["init_args"]["path"] == "calib.json"
        assert to_config(absolute)["init_args"]["path"] == "/var/calib.json"
        str_relative = PathHolder("./relative.json")
        assert to_config(str_relative)["init_args"]["path"] == "./relative.json"

    def test_enum_value(self) -> None:
        holder = EnumHolder(Color.RED)
        config = to_config(holder)
        assert config["init_args"]["color"] == "red"
        restored = cast(EnumHolder, instantiate(config))
        assert restored.color is Color.RED

    def test_non_finite_float_rejected(self) -> None:
        holder = MappingHolder({"x": math.nan})
        with pytest.raises(ComponentConfigError, match="non-finite"):
            to_config(holder)

    def test_nested_component(self) -> None:
        box = Box(Point(1, 2), label="b")
        config = to_config(box)
        origin = _as_mapping(config["init_args"]["origin"])
        assert cast(str, origin["class_path"]).endswith(".Point")
        assert origin["init_args"] == {"x": 1, "y": 2}
        restored = cast(Box, instantiate(json.loads(json.dumps(config))))
        assert restored.origin.x == 1
        assert to_config(restored) == json.loads(json.dumps(config))

    def test_list_and_mapping(self) -> None:
        holder = ListHolder([Point(1), {"a": 1}])
        config = to_config(holder)
        items = _as_list(config["init_args"]["items"])
        first = _as_mapping(items[0])
        assert _as_mapping(first["init_args"])["x"] == 1
        assert items[1] == {"a": 1}
        restored = cast(ListHolder, instantiate(config))
        assert isinstance(restored.items[0], Point)

    def test_mutable_container_snapshot(self) -> None:
        data: dict[str, object] = {"a": 1}
        holder = MappingHolder(data)
        data["a"] = 99
        assert _as_mapping(to_config(holder)["init_args"]["data"])["a"] == 1

    def test_cyclic_mapping_rejected(self) -> None:
        data: dict[str, object] = {}
        data["self"] = data
        holder = MappingHolder(data)
        with pytest.raises(ComponentConfigError, match="cyclic"):
            to_config(holder)

    def test_depth_limit_on_mappings(self) -> None:
        nested: dict[str, object] = {"leaf": 1}
        for _ in range(_MAX_CONFIG_DEPTH + 2):
            nested = {"child": nested}
        holder = MappingHolder(nested)
        with pytest.raises(ComponentConfigError, match="nesting depth"):
            to_config(holder)

    def test_nested_component_depth_symmetric(self) -> None:
        """Export and instantiate share the same nested-component depth limit.

        A Nest chain of length ``_MAX_CONFIG_DEPTH`` (depths 0..MAX-1 for
        components, leaf ``None`` at depth MAX) must round-trip. Length
        ``_MAX_CONFIG_DEPTH + 1`` must fail on both sides.
        """

        def nest_chain(length: int) -> Nest:
            node: Nest | None = None
            for _ in range(length):
                node = Nest(node)
            assert node is not None
            return node

        allowed = nest_chain(_MAX_CONFIG_DEPTH)
        config = to_config(allowed)
        restored = instantiate(config)
        assert isinstance(restored, Nest)
        assert to_config(restored) == config

        too_deep = nest_chain(_MAX_CONFIG_DEPTH + 1)
        with pytest.raises(ComponentConfigError, match="nesting depth"):
            to_config(too_deep)

        deeper: ComponentConfig = {
            "class_path": "tests.unit.config.test_component_config.Nest",
            "init_args": cast("dict[str, JsonValue]", {"child": config}),
        }
        with pytest.raises(ComponentConfigError, match="nesting depth"):
            instantiate(deeper)

    def test_nested_error_path_includes_parent(self) -> None:
        outer = Outer(BadInner(lambda: None))
        with pytest.raises(ComponentConfigError, match=r"Outer\.init_args\.child\.init_args\.fn"):
            to_config(outer)

    def test_malformed_nested_config_before_import(self) -> None:
        with pytest.raises(ComponentConfigError, match="unexpected keys"):
            instantiate({
                "class_path": "tests.unit.config.test_component_config.Point",
                "init_args": {},
                "extra": 1,
            })

    def test_null_init_args_rejected(self) -> None:
        with pytest.raises(ComponentConfigError, match="init_args"):
            instantiate({
                "class_path": "tests.unit.config.test_component_config.Point",
                "init_args": None,
            })

    def test_missing_class_path_rejected(self) -> None:
        with pytest.raises(ComponentConfigError, match="missing required"):
            instantiate({"init_args": {}})  # type: ignore[arg-type]

    def test_non_class_import_target(self) -> None:
        with pytest.raises(ComponentImportError, match="does not resolve to a class"):
            instantiate({"class_path": "os.path.join", "init_args": {}})

    def test_unimportable_class_path(self) -> None:
        with pytest.raises(ComponentImportError, match="cannot import"):
            instantiate({"class_path": "totally.unknown.module.Cls", "init_args": {}})

    def test_dict_with_class_path_is_reserved(self) -> None:
        holder = MappingHolder({"class_path": "not.a.component", "other": 1})
        with pytest.raises(ComponentConfigError, match="to_config_value"):
            to_config(holder)

    def test_unsupported_object_reports_path(self) -> None:
        holder = MappingHolder({"fn": lambda: None})
        with pytest.raises(ComponentConfigError, match=r"init_args\.data\.fn"):
            to_config(holder)

    def test_local_class_export_fails(self) -> None:
        @export_config
        class LocalPoint:
            def __init__(self, x: int) -> None:
                self.x = x

        obj = LocalPoint(1)
        with pytest.raises(ComponentConfigError, match="<locals>"):
            to_config(obj)

    def test_local_class_instantiate_fails(self) -> None:
        with pytest.raises(ComponentConfigError, match="<locals>"):
            instantiate({
                "class_path": "tests.unit.config.test_component_config.Local.<locals>.X",
                "init_args": {},
            })

    def test_constructor_failure_propagates_with_note(self) -> None:
        with pytest.raises(ValueError, match="boom") as info:
            instantiate({
                "class_path": "tests.unit.config.test_component_config.CtorBoom",
                "init_args": {"x": 1},
            })
        notes = getattr(info.value, "__notes__", [])
        assert any("constructor failed" in note for note in notes)

    def test_malformed_nested_config_rejected_before_any_import(self) -> None:
        config = {
            "class_path": "tests.unit.config.test_component_config.Point",
            "init_args": {
                "x": {
                    "class_path": "tests.unit.config.test_component_config.Point",
                    "init_args": None,
                },
            },
        }
        with (
            patch("physicalai.config._instantiate.import_dotted_path") as import_path,
            pytest.raises(ComponentConfigError, match="init_args"),
        ):
            instantiate(config)  # type: ignore[arg-type]
        import_path.assert_not_called()

    @pytest.mark.parametrize("value", [math.nan, math.inf, (1, 2), Path("config.json"), Color.RED, object()])
    def test_non_json_value_rejected_before_any_import(self, value: object) -> None:
        config = {
            "class_path": "tests.unit.config.test_component_config.DomainHolder",
            "init_args": {"payload": value},
        }
        with (
            patch("physicalai.config._instantiate.import_dotted_path") as import_path,
            pytest.raises(ComponentConfigError),
        ):
            instantiate(config)  # type: ignore[arg-type]
        import_path.assert_not_called()

    def test_non_string_plain_mapping_key_rejected_before_any_import(self) -> None:
        config = {
            "class_path": "tests.unit.config.test_component_config.MappingHolder",
            "init_args": {"data": {1: "value"}},
        }
        with (
            patch("physicalai.config._instantiate.import_dotted_path") as import_path,
            pytest.raises(ComponentConfigError, match="mapping keys must be strings"),
        ):
            instantiate(config)  # type: ignore[arg-type]
        import_path.assert_not_called()

    def test_cyclic_component_config_rejected_before_any_import(self) -> None:
        config: dict[str, object] = {
            "class_path": "tests.unit.config.test_component_config.Nest",
            "init_args": {},
        }
        cast("dict[str, object]", config["init_args"])["child"] = config
        with (
            patch("physicalai.config._instantiate.import_dotted_path") as import_path,
            pytest.raises(ComponentConfigError, match="cyclic component config"),
        ):
            instantiate(config)  # type: ignore[arg-type]
        import_path.assert_not_called()

    def test_plain_nested_mapping_remains_valid(self) -> None:
        config = cast("ComponentConfig", {
            "class_path": "tests.unit.config.test_component_config.MappingHolder",
            "init_args": {"data": {"nested": {"value": 1}}},
        })
        restored = cast(MappingHolder, instantiate(config))
        assert restored.data == {"nested": {"value": 1}}


class TestExportConfig:
    def test_positional_binds_to_names(self) -> None:
        point = Point(4, 5)
        assert to_config(point)["init_args"] == {"x": 4, "y": 5}

    def test_kwargs_flatten(self) -> None:
        obj = WithExtras(1, color="red", count=2)
        assert to_config(obj)["init_args"] == {"base": 1, "color": "red", "count": 2}

    def test_rejects_var_positional(self) -> None:
        with pytest.raises(TypeError, match=r"\*args"):

            @export_config
            class Bad:
                def __init__(self, *args: object) -> None:
                    self.args = args

    def test_rejects_positional_only(self) -> None:
        with pytest.raises(TypeError, match="positional-only"):

            @export_config
            class BadPosOnly:
                def __init__(self, x: int, /) -> None:
                    self.x = x

    def test_rejects_decorating_subclass_without_own_init(self) -> None:
        with pytest.raises(TypeError, match="define their own __init__"):

            @export_config
            class Redundant(BaseWidget):
                pass

    def test_outermost_super_wins(self) -> None:
        widget = DerivedWidget("w", 10)
        config = to_config(widget)
        assert config["class_path"].endswith(".DerivedWidget")
        assert config["init_args"] == {"name": "w", "size": 10}

    def test_undecorated_override_fails(self) -> None:
        obj = UndecoratedOverride("n", 1)
        assert not is_config_exportable(obj)
        with pytest.raises(ComponentConfigError, match="not config-exportable"):
            to_config(obj)

    def test_inherited_decorated_constructor(self) -> None:
        obj = InheritsDecorated("ok")
        assert is_config_exportable(obj)
        config = to_config(obj)
        assert config["class_path"].endswith(".InheritsDecorated")
        assert config["init_args"] == {"name": "ok"}

    def test_failed_constructor_does_not_capture(self) -> None:
        with pytest.raises(RuntimeError, match="nope"):
            Boom(1)

    def test_explicit_class_path_override(self) -> None:
        obj = _HiddenExport(3)
        assert is_config_exportable(obj)
        config = to_config(obj)
        assert config["class_path"] == "tests.unit.config.test_component_config.ExportAlias"
        assert config["init_args"] == {"x": 3}
        restored = instantiate(config)
        assert type(restored) is _HiddenExport
        assert restored.x == 3  # type: ignore[union-attr]

    def test_inherited_class_path_override_does_not_leak(self) -> None:
        obj = InheritsAliasedExport(9)
        assert is_config_exportable(obj)
        config = to_config(obj)
        assert config["class_path"] == (
            "tests.unit.config.test_component_config.InheritsAliasedExport"
        )
        assert config["init_args"] == {"x": 9}
        restored = instantiate(config)
        assert type(restored) is InheritsAliasedExport

    def test_domain_value_hook_encodes_to_json(self) -> None:
        holder = DomainHolder(DomainPayload(42))
        config = to_config(holder)
        assert config["init_args"]["payload"] == {"amount": 42}
        wire = json.loads(json.dumps(config))
        restored = cast(DomainHolder, instantiate(wire))
        assert restored.payload == {"amount": 42}
        assert to_config(restored) == wire

    def test_domain_value_none_is_json_null(self) -> None:
        holder = DomainHolder(NullDomain())
        config = to_config(holder)
        assert config["init_args"]["payload"] is None
        wire = json.loads(json.dumps(config))
        restored = cast(DomainHolder, instantiate(wire))
        assert restored.payload is None

    def test_domain_value_codec_cycle_raises(self) -> None:
        left = CodecPeer()
        right = CodecPeer()
        left.other = right
        right.other = left
        holder = DomainHolder(left)
        with pytest.raises(ComponentConfigError, match="cyclic to_config_value"):
            to_config(holder)

    def test_domain_value_hook_output_is_renormalized(self) -> None:
        holder = DomainHolder(BadNanDomain())
        with pytest.raises(ComponentConfigError, match="non-finite"):
            to_config(holder)

    def test_domain_value_hook_reserved_class_path_validated(self) -> None:
        holder = DomainHolder(BadReservedDomain())
        with pytest.raises(ComponentConfigError, match="to_config_value"):
            to_config(holder)

    def test_instance_to_config_sugar(self) -> None:
        point = Point(1, 2)
        assert point.to_config() == to_config(point)  # type: ignore[attr-defined]

    def test_injected_to_config_does_not_break_protocol(self) -> None:
        widget = BaseWidget("ok")
        assert isinstance(widget, Named)
        assert widget.to_config()["init_args"] == {"name": "ok"}  # type: ignore[attr-defined]

    def test_signature_preserved(self) -> None:
        sig = inspect.signature(Point.__init__)
        assert list(sig.parameters) == ["self", "x", "y"]

    def test_depth_attr_cleaned_after_construction(self) -> None:
        point = Point(1)
        assert is_config_exportable(point)
        assert "_physicalai_export_config_depth" not in vars(point)

    def test_private_capture_normalizer_canonicalizes_supplied_args_only(self) -> None:
        CanonicalName.normalize_calls = 0
        explicit = CanonicalName("LOUD")
        omitted = CanonicalName()
        assert to_config(explicit)["init_args"] == {"name": "loud"}
        assert to_config(omitted)["init_args"] == {}
        assert CanonicalName.normalize_calls == 2


class TestScalarVarKwargs:
    def test_scalar_var_kwargs_round_trip(self) -> None:
        obj = ScalarVarKwargs(1, count=2, flag=True, label="x", missing=None)
        config = to_config(obj)
        assert config["init_args"] == {
            "base": 1,
            "count": 2,
            "flag": True,
            "label": "x",
            "missing": None,
        }
        restored = cast(ScalarVarKwargs, instantiate(json.loads(json.dumps(config))))
        assert restored.base == 1
        assert restored.kwargs == {"count": 2, "flag": True, "label": "x", "missing": None}

    def test_non_scalar_dict_var_kwarg_fails(self) -> None:
        obj = ScalarVarKwargs(1, config_blob={"a": 1})
        with pytest.raises(ComponentConfigError, match=r"init_args\.config_blob"):
            to_config(obj)

    def test_non_scalar_list_var_kwarg_fails(self) -> None:
        obj = ScalarVarKwargs(1, tags=["x", "y"])
        with pytest.raises(ComponentConfigError, match=r"init_args\.tags"):
            to_config(obj)

    def test_named_mapping_still_exports_without_scalar_flag(self) -> None:
        # Default **kwargs flattening still accepts nested JSON.
        obj = WithExtras(1, nested={"a": 1})
        assert to_config(obj)["init_args"]["nested"] == {"a": 1}

    def test_scalar_var_kwargs_requires_var_keyword(self) -> None:
        with pytest.raises(TypeError, match="scalar_var_kwargs=True requires"):

            @export_config(scalar_var_kwargs=True)
            class NoVarKwargs:
                def __init__(self, x: int) -> None:
                    self.x = x


class TestConfigArgs:
    def test_declared_config_arg_is_not_instantiated(self) -> None:
        config: ComponentConfig = {
            "class_path": f"{__name__}.Holder",
            "init_args": {
                "recipe": {"class_path": f"{__name__}.Leaf", "init_args": {"value": 3}},
                "eager": {"class_path": f"{__name__}.Leaf", "init_args": {"value": 3}},
            },
        }
        holder = cast(Holder, instantiate(config))
        assert holder.recipe == config["init_args"]["recipe"]
        assert isinstance(holder.eager, Leaf)

    def test_declared_config_arg_round_trips(self) -> None:
        recipe: ComponentConfig = {
            "class_path": f"{__name__}.Leaf",
            "init_args": {"value": 3},
        }
        holder = Holder(recipe=recipe, eager=Leaf(value=3))
        config = to_config(holder)
        assert config["init_args"]["recipe"] == recipe
        wire = json.loads(json.dumps(config))
        restored = cast(Holder, instantiate(wire))
        assert to_config(restored) == wire

    def test_unknown_config_arg_name_rejected(self) -> None:
        with pytest.raises(TypeError, match="config_args 'missing' is not an __init__ parameter"):

            @export_config(config_args=("missing",))
            class Bad:
                def __init__(self, x: int) -> None:
                    self.x = x

    def test_config_args_cannot_name_var_keyword(self) -> None:
        with pytest.raises(TypeError, match=r"config_args cannot name the \*\*kwargs parameter"):

            @export_config(config_args=("kwargs",))
            class BadVarKw:
                def __init__(self, **kwargs: object) -> None:
                    self.kwargs = kwargs
