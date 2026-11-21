# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Pydantic schemas shared by plugin protocols."""

from __future__ import annotations

from pydantic import BaseModel


class SerialPortInfo(BaseModel):
    """Connection metadata for a discovered serial or network robot."""

    connection_string: str | None = None
    serial_number: str | None = None
