# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Serializable camera construction spec for transport endpoints.

Private publisher stdin is ``camera: Config`` only. The publisher
envelope is validated schema-positively: required ``camera``, known transport
keys, and rejection of unknown keys (including legacy flat
``camera_type`` / ``camera_kwargs``) before import or hardware access. Public
``SharedCamera`` uses the same ``camera`` / :meth:`~SharedCamera.from_config`
shape.

Security: ``class_path`` is local application/config input. It must never
originate from network-received or control-channel data.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit, urlunsplit

from physicalai.config import (
    Config,
    normalize_config,
    validate_envelope,
)

if TYPE_CHECKING:
    from physicalai.capture.camera import Camera

# Allowed keys on the publisher stdin handshake. Everything else
# (including legacy flat camera_type / camera_kwargs) is an unknown-key
# schema error.
_PUBLISHER_ENVELOPE_KEYS = frozenset({
    "camera",
    "service_name",
    "idle_timeout",
    "max_subscribers",
    "_factory_override",
})
_RECONFIGURE_REQUEST_KEYS = frozenset({"kind", "settings"})
_RECONFIGURABLE_SETTINGS = frozenset({"width", "height", "fps"})


def validate_publisher_config(data: Mapping[str, Any]) -> Config:
    """Validate a publisher stdin payload schema-positively.

    Returns:
        The validated ``camera`` Config mapping (see
        :func:`normalize_camera_config` for the JSON-serializability check).
    """
    return validate_envelope(
        data,
        component_key="camera",
        allowed_keys=_PUBLISHER_ENVELOPE_KEYS,
        envelope_name="publisher",
    )


def validate_reconfigure_request(request: Mapping[str, Any]) -> dict[str, int]:
    """Validate a control-channel camera reconfigure request.

    The peer may change only scalar capture settings. The publisher keeps the
    local startup ``class_path`` and all other constructor arguments.

    Returns:
        Validated settings to patch into the local camera recipe.

    Raises:
        TypeError: If the request or a setting has the wrong type.
        ValueError: If required fields are missing, unknown, or out of range.
    """
    if not isinstance(request, Mapping):
        msg = f"reconfigure request must be a mapping, got {type(request).__name__}"
        raise TypeError(msg)

    unknown_request_keys = sorted(set(request) - _RECONFIGURE_REQUEST_KEYS)
    if unknown_request_keys:
        msg = f"unknown reconfigure request keys {unknown_request_keys}"
        raise ValueError(msg)
    if request.get("kind") != "RECONFIGURE":
        msg = "reconfigure request kind must be 'RECONFIGURE'"
        raise ValueError(msg)

    settings = request.get("settings")
    if not isinstance(settings, Mapping):
        msg = f"reconfigure 'settings' must be a mapping, got {type(settings).__name__}"
        raise TypeError(msg)
    if not settings:
        msg = "reconfigure 'settings' must not be empty"
        raise ValueError(msg)

    unknown_settings = sorted(set(settings) - _RECONFIGURABLE_SETTINGS)
    if unknown_settings:
        msg = f"unknown reconfigure settings {unknown_settings}; allowed: {sorted(_RECONFIGURABLE_SETTINGS)}"
        raise ValueError(msg)

    validated: dict[str, int] = {}
    for key, value in settings.items():
        if isinstance(value, bool) or not isinstance(value, int):
            msg = f"reconfigure setting {key!r} must be an integer, got {type(value).__name__}"
            raise TypeError(msg)
        if value <= 0:
            msg = f"reconfigure setting {key!r} must be greater than zero, got {value}"
            raise ValueError(msg)
        validated[key] = value
    return validated


def normalize_camera_config(camera: Config | Mapping[str, object]) -> Config:
    """Validate a camera Config without importing its ``class_path``.

    Returns:
        A validated config whose ``class_path`` is a dotted import path.
    """
    return normalize_config(
        camera,
        component_key="camera",
        class_label="camera class_path",
    )


def _url_token(url: str) -> str:
    """Credential-free, slash-free token identifying a stream URL.

    Used as the ``device_id`` segment of a derived service name for
    ``IPCamera`` (``init_args={"url": ...}``), whose identifying config is a
    URL rather than a ``serial_number``/``device`` — and may embed
    ``user:pass@`` credentials that must never end up in a service name
    (visible OS-wide) or a log line (``service_name`` is logged verbatim in
    ``SharedCamera``). Hashing the credential-stripped URL keeps distinct
    cameras (different host/path) on distinct services without leaking or
    even reversibly encoding the URL itself.

    Returns:
        First 12 hex chars of a sha256 hash of the URL with any userinfo removed.
    """
    parts = urlsplit(url)
    netloc = parts.netloc.rsplit("@", 1)[-1]
    sanitized = urlunsplit((parts.scheme, netloc, parts.path, parts.query, ""))
    return hashlib.sha256(sanitized.encode()).hexdigest()[:12]


def derive_service_name(
    camera: Config | Mapping[str, object],
    *,
    service_name: str | None = None,
) -> str:
    """Resolve iceoryx2 ``service_name`` for a camera Config.

    Derives ``physicalai/camera/{class_name}/{device_id}/frame`` from the
    terminal segment of ``class_path``, without importing it. The bare class
    name collapses every spelling of one driver onto one publisher; distinct
    classes sharing a name and device id need an explicit *service_name*.

    Args:
        camera: Normalized or raw camera Config.
        service_name: Explicit override; when set, returned unchanged.

    Returns:
        Concrete service name for the publisher envelope.
    """
    if service_name is not None:
        return service_name

    class_name = str(camera["class_path"]).rsplit(".", 1)[-1]

    init_args = camera.get("init_args", {})
    if not isinstance(init_args, Mapping):
        init_args = {}
    device_id = init_args.get("serial_number", init_args.get("device"))
    if device_id is None:
        # IPCamera has no serial_number/device — its identifying config is a
        # URL, which is sanitized
        url = init_args.get("url")
        device_id = _url_token(url) if isinstance(url, str) else 0
    # Resolve symlinks so that /dev/v4l/by-id/... and /dev/videoN produce
    # the same service name for the same physical device.
    if isinstance(device_id, str) and device_id.startswith("/dev/"):
        device_id = Path(device_id).resolve().name
    return f"physicalai/camera/{class_name}/{device_id}/frame"


@dataclass(frozen=True)
class CameraPublisherConfig:
    """Config payload describing how to construct a camera instance.

    Attributes:
        camera: Local construction config (``class_path`` + ``init_args``).
    """

    camera: Config | Mapping[str, object]

    def __post_init__(self) -> None:
        """Normalize ``camera`` to a public Config."""
        object.__setattr__(self, "camera", normalize_camera_config(self.camera))

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize the construction fragment for the stdin handshake.

        Returns:
            Dictionary with ``camera`` only (transport fields are merged by
            the publisher).

        """
        return {"camera": normalize_camera_config(self.camera).to_dict()}

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> CameraPublisherConfig:
        """Deserialize from a JSON dictionary (full publisher envelope or fragment).

        Uses :func:`validate_publisher_config` so publisher stdin requires
        ``camera``, accepts only known transport keys, and rejects unknown
        keys before construction.

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

        Uses :func:`physicalai.config.instantiate` on the ``camera``
        Config, then verifies the :class:`~physicalai.capture.camera.Camera`
        protocol. Does not route through :func:`~physicalai.capture.create_camera`,
        so third-party class paths work without a registry entry.

        Returns:
            A new, not-yet-connected camera instance.

        Raises:
            TypeError: If the instantiated object does not satisfy ``Camera``.
        """
        from physicalai.capture.camera import Camera  # noqa: PLC0415

        driver = normalize_camera_config(self.camera).instantiate()
        if not isinstance(driver, Camera):
            msg = f"{self.camera['class_path']!r} does not satisfy the Camera protocol (got {type(driver).__name__})"
            raise TypeError(msg)
        return driver
