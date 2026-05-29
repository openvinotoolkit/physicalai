# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FakeRobotObservation:
    """Test double satisfying the RobotObservation protocol."""

    joint_positions: np.ndarray
    timestamp: float = 0.0
    sensor_data: dict[str, np.ndarray] | None = None
    images: dict | None = None

    @property
    def state(self) -> np.ndarray:
        return self.joint_positions
