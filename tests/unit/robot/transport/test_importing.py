# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: D100, D103, PLC2701, S101

from __future__ import annotations

import pytest

from physicalai.robot.transport._importing import import_dotted_path


class _Outer:
    class Inner:
        VALUE = 42


def test_nested_qualname() -> None:
    resolved = import_dotted_path("tests.unit.robot.transport.test_importing._Outer.Inner")
    assert resolved is _Outer.Inner


def test_unimportable_prefix_raises() -> None:
    with pytest.raises(ValueError, match="could not import"):
        import_dotted_path("totally.unknown.module.Robot")


def test_importable_module_bad_attribute_raises() -> None:
    with pytest.raises(AttributeError):
        import_dotted_path("os.path.NoSuchRobot")
