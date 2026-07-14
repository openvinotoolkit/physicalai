# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: D100, D101, D102, PLC0415, PLC2701, PLR2004, PLR6301, RUF043, S101

from __future__ import annotations

import pytest

from physicalai.inference._importing import import_dotted_path


class _Outer:
    class Inner:
        VALUE = 42


class TestImportDottedPath:
    def test_module_dot_class(self) -> None:
        obj = import_dotted_path("tests.unit.inference.test_importing._Outer")
        assert obj is _Outer

    def test_nested_qualname(self) -> None:
        obj = import_dotted_path("tests.unit.inference.test_importing._Outer.Inner")
        assert obj is _Outer.Inner

    def test_deeply_nested_attribute(self) -> None:
        obj = import_dotted_path("tests.unit.inference.test_importing._Outer.Inner.VALUE")
        assert obj == 42

    def test_module_only_no_dot_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one '.'"):
            import_dotted_path("os")

    def test_unimportable_prefix_raises(self) -> None:
        with pytest.raises(ValueError, match="could not import"):
            import_dotted_path("totally.unknown.module.Cls")

    def test_importable_module_bad_attribute_raises(self) -> None:
        with pytest.raises(AttributeError):
            import_dotted_path("os.path.NoSuchAttribute")

    def test_stdlib_function(self) -> None:
        obj = import_dotted_path("os.path.join")
        import os

        assert obj is os.path.join
