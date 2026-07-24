# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Serializable camera construction spec for transport endpoints.

Private publisher stdin is ``camera: ComponentConfig`` only. The publisher
envelope is validated schema-positively: required ``camera``, known transport
keys, and rejection of unknown keys (including legacy flat
``camera_type`` / ``camera_kwargs``) before import or hardware access. Public
``SharedCamera`` uses the same ``camera`` / :meth:`~SharedCamera.from_config`
shape.

Security: ``class_path`` is trusted local application/config input. It must
never originate from network-received data.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from physicalai.config import (
    ComponentConfig,
    instantiate,
    normalize_class_reference,
    normalize_component_config,
    validate_envelope,
)

from .builtin import builtin_type_for_class_path

if TYPE_CHECKING:
    from physicalai.capture.camera import Camera

# Allowed keys on publisher stdin and reconfigure ``spec`` payloads.
# Everything else (including legacy flat camera_type / camera_kwargs) is an
# unknown-key schema error. Keep this the single allowlist — stdin parse and
# reconfigure share :func:`validate_publisher_config`.
_PUBLISHER_ENVELOPE_KEYS = frozenset({
    "camera",
    "service_name",
    "idle_timeout",
    "max_subscribers",
    "_factory_override",
})


def validate_publisher_config(data: Mapping[str, Any]) -> Mapping[str, object]:
    """Validate a publisher stdin or reconfigure payload schema-positively.

    Returns:
        The validated ``camera`` ComponentConfig mapping (not yet
        public-path-normalized — :func:`normalize_camera_config` does that).
    """
    return validate_envelope(
        data,
        component_key="camera",
        allowed_keys=_PUBLISHER_ENVELOPE_KEYS,
        envelope_name="publisher",
    )


def normalize_camera_class(camera_class: type | str) -> str:
    """Normalize a camera class reference to its public import path.

    Returns:
        The normalized public dotted path.
    """
    return normalize_class_reference(camera_class, label="camera class_path")


def normalize_camera_config(camera: Mapping[str, object]) -> ComponentConfig:
    """Validate a camera ComponentConfig and normalize ``class_path``.

    Returns:
        A validated config whose ``class_path`` is the public import path.
    """
    return normalize_component_config(
        camera,
        component_key="camera",
        class_label="camera class_path",
    )


def derive_service_name(
    camera: Mapping[str, object],
    *,
    service_name: str | None = None,
) -> str:
    """Resolve iceoryx2 ``service_name`` for a camera ComponentConfig.

    Built-in public class paths derive ``physicalai/camera/{token}/{device_id}/frame``
    via the transport class-path → type-token map. Third-party / unknown
    class paths (including stub types without a shared registry entry) require
    an explicit *service_name*.

    Args:
        camera: Normalized or raw camera ComponentConfig.
        service_name: Explicit override; when set, returned unchanged.

    Returns:
        Concrete service name for the publisher envelope.

    Raises:
        ValueError: If *service_name* is omitted for a non-built-in class_path.
    """
    if service_name is not None:
        return service_name

    class_path = str(camera["class_path"])
    token = builtin_type_for_class_path(class_path)
    if token is None:
        msg = (
            f"camera {class_path!r} requires an explicit service_name; "
            "built-in derivation only covers shareable physicalai.capture "
            "backends (uvc, realsense, basler)"
        )
        raise ValueError(msg)

    init_args = camera.get("init_args", {})
    if not isinstance(init_args, Mapping):
        init_args = {}
    device_id = init_args.get("serial_number", init_args.get("device", 0))
    # Resolve symlinks so that /dev/v4l/by-id/... and /dev/videoN produce
    # the same service name for the same physical device.
    if isinstance(device_id, str) and device_id.startswith("/dev/"):
        device_id = Path(device_id).resolve().name
    return f"physicalai/camera/{token}/{device_id}/frame"


@dataclass(frozen=True)
class CameraPublisherConfig:
    """Config payload describing how to construct a camera instance.

    Attributes:
        camera: Trusted construction config (``class_path`` + ``init_args``).
    """

    camera: Mapping[str, object]

    def __post_init__(self) -> None:
        """Normalize ``camera`` to a public ComponentConfig."""
        object.__setattr__(self, "camera", normalize_camera_config(self.camera))

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize the construction fragment for the stdin handshake.

        Returns:
            Dictionary with ``camera`` only (transport fields are merged by
            the publisher).

        Raises:
            TypeError: If ``camera.init_args`` is not a mapping.
        """
        init_args = self.camera["init_args"]
        if not isinstance(init_args, dict):
            msg = f"camera.init_args must be a mapping, got {type(init_args).__name__}"
            raise TypeError(msg)
        return {
            "camera": {
                "class_path": self.camera["class_path"],
                "init_args": dict(init_args),
            },
        }

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> CameraPublisherConfig:
        """Deserialize from a JSON dictionary (full publisher envelope or fragment).

        Uses :func:`validate_publisher_config` so stdin and reconfigure share
        one schema (required ``camera``, known transport keys, unknown keys
        rejected).

        Args:
            data: Dictionary produced by :meth:`to_json_dict` or a publisher
                stdin envelope containing ``camera``.

        Returns:
            A new :class:`CameraPublisherConfig` instance.

        Raises:
            TypeError: If ``data`` is not a mapping.
        """
        if not isinstance(data, dict):
            msg = f"camera spec must be a mapping, got {type(data).__name__}"
            raise TypeError(msg)

        camera = validate_publisher_config(data)
        return cls(camera=camera)

    def build(self) -> Camera:
        """Instantiate the camera described by this spec.

        Uses :func:`physicalai.config.instantiate` on the trusted ``camera``
        ComponentConfig, then verifies the :class:`~physicalai.capture.camera.Camera`
        protocol. Does not route through :func:`~physicalai.capture.create_camera`,
        so third-party class paths work without a registry entry.

        Returns:
            A new, not-yet-connected camera instance.

        Raises:
            TypeError: If the instantiated object does not satisfy ``Camera``.
        """
        from physicalai.capture.camera import Camera  # noqa: PLC0415

        driver = instantiate(self.camera)  # type: ignore[arg-type]
        if not isinstance(driver, Camera):
            msg = f"{self.camera['class_path']!r} does not satisfy the Camera protocol (got {type(driver).__name__})"
            raise TypeError(msg)
        return driver
