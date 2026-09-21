# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# ruff: file-ignore[undocumented-public-module, undocumented-public-function, assert]

from __future__ import annotations

import physicalai.config
from physicalai.config import import_dotted_path


class _Target:
    pass


def test_import_dotted_path_remains_public() -> None:
    resolved = import_dotted_path("tests.unit.config.test_importing._Target")

    assert resolved is _Target
    assert "import_dotted_path" in physicalai.config.__all__


def test_deprecated_configuration_shims_are_not_restored() -> None:
    removed_names = {
        "FromConfig",
        "from_config",
        "import_class",
        "instantiate_obj",
        "load_yaml",
        "normalize_config",
        "save_yaml",
        "to_config",
        "to_yaml",
        "validate_envelope",
    }

    assert removed_names.isdisjoint(vars(physicalai.config))
