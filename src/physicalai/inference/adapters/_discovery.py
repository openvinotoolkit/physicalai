# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Adapter discovery and factory helpers."""

from __future__ import annotations

import logging
from importlib.metadata import entry_points
from typing import TYPE_CHECKING, Any

from physicalai.inference.adapters.registry import adapter_registry

if TYPE_CHECKING:
    from physicalai.inference.adapters.base import RuntimeAdapter

logger = logging.getLogger(__name__)

#: Entry-point group used by other distributions to contribute adapters.
#:
#: Each entry point must point at a callable with the signature
#: ``register(registry: RuntimeAdapterRegistry) -> None``.  The callable
#: should populate *registry* (typically via
#: :meth:`RuntimeAdapterRegistry.register_lazy_module`) without
#: triggering heavy imports.
ENTRY_POINT_GROUP = "physicalai.inference.adapters"


def _load_external_adapters() -> None:
    """Discover and run third-party adapter registrations.

    Imports every entry point in :data:`ENTRY_POINT_GROUP` and invokes it
    with the shared :data:`backend_registry`.  Failures are logged and
    swallowed so a single broken plugin cannot prevent the rest of the
    inference stack from importing.
    """
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        try:
            register_fn = ep.load()
        except Exception:
            logger.exception("Failed to load adapter provider entry point %r", ep.name)
            continue
        try:
            register_fn(adapter_registry)
        except Exception:
            logger.exception("Adapter provider %r raised during registration", ep.name)


_load_external_adapters()


def get_adapter(backend: str, **kwargs: Any) -> RuntimeAdapter:  # noqa: ANN401
    """Instantiate the adapter registered for *backend*.

    Args:
        backend: Backend identifier as a string (e.g. ``"onnx"``,
            ``"openvino"``).  Any object whose ``str()`` value matches a
            registered backend is accepted (e.g. a
            :class:`enum.StrEnum` member).
        **kwargs: Forwarded to the adapter constructor (e.g. ``device``).

    Returns:
        A ready-to-use :class:`RuntimeAdapter` instance.

    Examples:
        >>> adapter = get_adapter("openvino", device="CPU")
        >>> adapter = get_adapter("onnx")
    """
    name = str(backend)
    adapter_cls = adapter_registry.get_class(name)
    return adapter_cls(**kwargs)
