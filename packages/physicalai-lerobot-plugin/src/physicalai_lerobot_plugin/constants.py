# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Hardware-agnostic constants for the LeRobot adapter plugin."""

from __future__ import annotations

from typing import Final

VALID_ROLES: Final = frozenset({"leader", "follower"})
DEFAULT_POS_SUFFIX: Final = ".pos"
