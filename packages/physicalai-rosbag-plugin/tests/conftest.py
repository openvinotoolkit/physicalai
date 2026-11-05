# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import runpy
from pathlib import Path

import pytest


@pytest.fixture
def demo():
    return runpy.run_path(str(Path(__file__).parents[1] / "examples/generate_demo.py"))
