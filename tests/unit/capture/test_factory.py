# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the factory and discovery functions."""

import pytest

from physicalai.capture.discovery import discover_all
from physicalai.capture.factory import create_camera
from physicalai.capture.transport.builtin import (
    builtin_class_path_for_type,
    builtin_class_paths_for_type,
    builtin_shared_type_tokens,
    builtin_type_for_class_path,
)


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
        assert cam._camera["class_path"] == builtin_class_path_for_type("uvc")  # type: ignore[attr-defined]
        assert cam._service_name == "physicalai/camera/uvc/0/frame"  # type: ignore[attr-defined]


class TestBuiltinSharedRegistry:
    """Single source of truth for shareable type ↔ class_path."""

    def test_shareable_tokens(self) -> None:
        assert builtin_shared_type_tokens() == frozenset({"uvc", "realsense", "basler"})

    def test_round_trip_uvc(self) -> None:
        path = builtin_class_path_for_type("uvc")
        assert path == "physicalai.capture.UVCCamera"
        assert builtin_type_for_class_path(path) == "uvc"

    def test_matches_export_config_when_importable(self) -> None:
        from physicalai.capture.cameras.uvc import UVCCamera
        from physicalai.config import resolve_public_class_path

        assert builtin_class_path_for_type("uvc") == resolve_public_class_path(UVCCamera)

    @pytest.mark.parametrize("token", ["uvc", "realsense", "basler"])
    def test_static_table_matches_drivers(self, token: str) -> None:
        """Every listed spelling must reach the driver the decorator declares."""
        from physicalai.config import import_dotted_path, resolve_public_class_path

        paths = builtin_class_paths_for_type(token)
        expected: object = None
        try:
            expected = import_dotted_path(paths[0])
        except (ImportError, AttributeError) as exc:  # optional camera extra absent
            pytest.skip(f"{token} driver is not installed: {exc}")
        assert isinstance(expected, type)
        assert resolve_public_class_path(expected) == paths[0]

        # Compare by defining name rather than identity: other tests reload
        # driver modules with a mocked SDK, which replaces the class object.
        defining = f"{expected.__module__}.{expected.__qualname__}"
        assert defining == paths[-1]
        for alias in paths[1:]:
            resolved = import_dotted_path(alias)
            assert isinstance(resolved, type)
            assert f"{resolved.__module__}.{resolved.__qualname__}" == defining

    @pytest.mark.parametrize("token", ["uvc", "realsense", "basler"])
    def test_every_spelling_maps_back_to_one_token(self, token: str) -> None:
        for path in builtin_class_paths_for_type(token):
            assert builtin_type_for_class_path(path) == token

    def test_no_phantom_ip_genicam(self) -> None:
        assert builtin_class_path_for_type("ip") is None
        assert builtin_class_path_for_type("genicam") is None
        assert builtin_type_for_class_path("physicalai.capture.IPCamera") is None
        assert builtin_type_for_class_path("physicalai.capture.GenicamCamera") is None


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
