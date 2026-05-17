# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Runtime telemetry observer — subscribes to zenoh pub-sub events."""

from __future__ import annotations

from physicalai.runtime.observer._subscriber import TelemetrySubscriber  # noqa: PLC2701

__all__ = ["TelemetrySubscriber"]
