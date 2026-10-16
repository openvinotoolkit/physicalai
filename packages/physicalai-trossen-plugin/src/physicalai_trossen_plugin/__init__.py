# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: PLC0415

"""Trossen WidowX AI robot integration for PhysicalAI."""

from __future__ import annotations

from typing import TYPE_CHECKING

from physicalai_trossen_plugin._urdf import get_urdf_path as get_urdf_path

if TYPE_CHECKING:
    from physicalai_trossen_plugin.bimanual_widowxai import BimanualWidowXAI as BimanualWidowXAI
    from physicalai_trossen_plugin.bimanual_widowxai import BimanualWidowXAIObservation as BimanualWidowXAIObservation
    from physicalai_trossen_plugin.widowxai import WidowXAI as WidowXAI
    from physicalai_trossen_plugin.widowxai import WidowXAIObservation as WidowXAIObservation

__all__ = [
    "BimanualWidowXAI",
    "BimanualWidowXAIObservation",
    "WidowXAI",
    "WidowXAIObservation",
    "get_urdf_path",
]


def __getattr__(name: str) -> object:
    """Lazily import robot drivers without loading the vendor SDK on package import.

    Returns:
        The requested robot driver or observation type.

    Raises:
        AttributeError: If ``name`` is not exported by this package.
    """
    if name in {"WidowXAI", "WidowXAIObservation"}:
        from physicalai_trossen_plugin.widowxai import WidowXAI, WidowXAIObservation

        return {"WidowXAI": WidowXAI, "WidowXAIObservation": WidowXAIObservation}[name]
    if name in {"BimanualWidowXAI", "BimanualWidowXAIObservation"}:
        from physicalai_trossen_plugin.bimanual_widowxai import BimanualWidowXAI, BimanualWidowXAIObservation

        return {
            "BimanualWidowXAI": BimanualWidowXAI,
            "BimanualWidowXAIObservation": BimanualWidowXAIObservation,
        }[name]
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
