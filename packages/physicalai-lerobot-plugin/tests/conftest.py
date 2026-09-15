from __future__ import annotations

import importlib.machinery
import sys


def _ensure_module_spec(module_name: str) -> None:
    existing = sys.modules.get(module_name)
    if existing is None:
        return

    if not isinstance(getattr(existing, "__spec__", None), importlib.machinery.ModuleSpec):
        existing.__spec__ = importlib.machinery.ModuleSpec(module_name, loader=None)


for _module_name in ("scservo_sdk", "motorbridge", "motorbridge_smart_servo"):
    _ensure_module_spec(_module_name)
