# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Inference adapters for different backend runtimes.

Adapters are discovered through :class:`RuntimeAdapterRegistry`.  This
package ships the ``onnx`` and ``openvino`` backends only — they
self-register on import below.

Additional backends are contributed by other distributions sharing the
``physicalai`` namespace via the ``physicalai.inference.adapters``
:mod:`importlib.metadata` entry-point group.  Each such entry point names
a callable ``register(registry)`` that populates the shared
:data:`backend_registry`.  Lightweight registration entry points (no heavy
imports) let third parties expose their backends — and in particular
their file extensions for auto-detection — without forcing this package
to know about them.

Use :func:`get_adapter` to obtain an adapter instance for a given backend.
"""

from __future__ import annotations

from pkgutil import extend_path

# Allow `physicalai.inference.adapters` to be split across multiple
# distributions sharing the `physicalai` namespace.
__path__ = extend_path(__path__, __name__)

# Re-export get_adapter from the internal discovery module.
from physicalai.inference.adapters._discovery import get_adapter as get_adapter  # noqa: PLC2701
from physicalai.inference.adapters.base import RuntimeAdapter
from physicalai.inference.adapters.onnx import ONNXAdapter

# Eagerly import core adapters so they self-register.  Their runtime
# dependencies (onnxruntime, openvino) are part of the `inference` extra.
from physicalai.inference.adapters.openvino import OpenVINOAdapter
from physicalai.inference.adapters.registry import (
    RuntimeAdapterRegistry,
    adapter_registry,
)

__all__ = [
    "ONNXAdapter",
    "OpenVINOAdapter",
    "RuntimeAdapter",
    "RuntimeAdapterRegistry",
    "adapter_registry",
    "get_adapter",
]
