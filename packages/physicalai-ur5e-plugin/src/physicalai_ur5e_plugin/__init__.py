# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Universal Robots e-Series plugin; importing it does not load the vendor SDK."""

from .driver import UniversalRobot
from .profiles import JOINT_NAMES, SUPPORTED_MODELS, get_asset_root

__all__ = ["JOINT_NAMES", "SUPPORTED_MODELS", "UniversalRobot", "get_asset_root"]
