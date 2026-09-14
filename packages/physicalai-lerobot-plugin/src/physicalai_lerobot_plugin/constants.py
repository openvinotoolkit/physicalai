# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Hardware-agnostic constants for the LeRobot adapter plugin."""

from __future__ import annotations

import os
from typing import Final

VALID_ROLES: Final = frozenset({"leader", "follower"})
DEFAULT_POS_SUFFIX: Final = ".pos"
TRUST_UNVERIFIED_PLUGIN: Final = "TRUST_UNVERIFIED_PLUGIN"


def trust_unverified_plugins() -> bool:
    """Return whether installed third-party LeRobot plugins may be imported.

    Importing a plugin executes its package code. This is deliberately opt-in
    until Studio can present its unverified-plugin approval flow.
    """
    return os.environ.get(TRUST_UNVERIFIED_PLUGIN, "").lower() in {"1", "true", "yes", "on"}
