from __future__ import annotations

import importlib.machinery
import sys

import pytest


def _ensure_module_spec(module_name: str) -> None:
    existing = sys.modules.get(module_name)
    if existing is None:
        return

    if not isinstance(getattr(existing, "__spec__", None), importlib.machinery.ModuleSpec):
        existing.__spec__ = importlib.machinery.ModuleSpec(module_name, loader=None)


@pytest.fixture(autouse=True)
def _ensure_optional_sdk_specs() -> None:
    # Other test modules install SDK mocks during collection, after this
    # conftest is imported. Normalize them before each LeRobot test instead.
    for module_name in ("scservo_sdk", "motorbridge", "motorbridge_smart_servo"):
        _ensure_module_spec(module_name)
