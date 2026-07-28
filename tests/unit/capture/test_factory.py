# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the factory and discovery functions."""

import pytest

from physicalai.capture.discovery import discover_all
from physicalai.capture.factory import _SHAREABLE_CLASS_PATHS, create_camera


class TestCreateCamera:
    """create_camera() driver dispatch."""

    def test_unknown_driver_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown camera type"):
            create_camera("nonexistent")

    def test_case_insensitive(self) -> None:
        # Ensure camera type dispatch is case-insensitive.
        from physicalai.capture.cameras.uvc import UVCCamera

        cam = create_camera("UVC", backend="v4l2")
        assert isinstance(cam, UVCCamera)

    def test_shared_stub_types_rejected(self) -> None:
        for stub in ("ip", "genicam"):
            with pytest.raises(ValueError, match="does not support shared=True"):
                create_camera(stub, shared=True, device=0)

    def test_shared_builtin_uses_registry_class_path(self) -> None:
        cam = create_camera("uvc", shared=True, device=0, backend="v4l2")
        assert cam._camera is not None  # type: ignore[attr-defined]
        assert cam._camera["class_path"] == _SHAREABLE_CLASS_PATHS["uvc"]  # type: ignore[attr-defined]
        assert cam._service_name == "physicalai/camera/UVCCamera/0/frame"  # type: ignore[attr-defined]


class TestShareableClassPaths:
    """The static token → class_path table must track the drivers it names."""

    @pytest.mark.parametrize("token", sorted(_SHAREABLE_CLASS_PATHS))
    def test_matches_export_config_when_importable(self, token: str) -> None:
        """Keeps the hand-written table honest when the extra is installed."""
        from physicalai.config import import_dotted_path, resolve_public_class_path

        class_path = _SHAREABLE_CLASS_PATHS[token]
        driver: object = None
        try:
            driver = import_dotted_path(class_path)
        except (ImportError, AttributeError) as exc:  # optional camera extra absent
            pytest.skip(f"{token} driver is not installed: {exc}")
        assert isinstance(driver, type)
        assert resolve_public_class_path(driver) == class_path



class TestDiscoverAll:
    """discover_all() aggregation."""

    def test_returns_dict(self) -> None:
        # Without any camera SDKs installed, backends will fail to
        # import and be silently skipped.
        result = discover_all()
        assert isinstance(result, dict)

    def test_missing_backends_skipped(self) -> None:
        # Should not raise even when no backends are installed.
        result = discover_all()
        # Result may be empty or partial; the key invariant is no exception.
        assert isinstance(result, dict)
