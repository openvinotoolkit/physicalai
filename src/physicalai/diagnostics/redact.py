# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Best-effort support-report redaction that keeps alias relationships intact."""

from __future__ import annotations

import re
import socket
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

_Model = TypeVar("_Model", bound=BaseModel)
_SCHEMA_FIELDS = {"kind", "role", "status", "severity", "check_id", "binding", "platform", "pixel_format", "mode"}
_SUBSTRING_MIN_LENGTH = 4


class Redactor:
    """Keep pseudonyms consistent within one report/watch, not across sessions."""

    def __init__(self) -> None:
        """Initialize locally known host identifiers without collecting network data."""
        self.replacements = {str(Path.home()): "<HOME>", socket.gethostname(): "<HOST>"}
        self._next_id = 1

    def _register(self, value: object) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if (
                    key in {"serial", "ID_SERIAL_SHORT", "ID_USB_SERIAL_SHORT"}
                    and isinstance(child, str)
                    and child
                    and child not in self.replacements
                ):
                    self.replacements[child] = f"<DEVICE-{self._next_id}>"
                    self._next_id += 1
                self._register(child)
        elif isinstance(value, list):
            for child in value:
                self._register(child)

    def register(self, value: BaseModel) -> None:
        """Remember identifiers before hardware disappears from a later snapshot."""
        self._register(value.model_dump(mode="json"))

    def model(self, value: _Model) -> _Model:
        """Redact a report/inventory/event without mutating collection evidence.

        This is not a general secret scrubber. Review free-text device names and
        logs before publication. Redacted paths are not deployable selectors.

        Returns:
            A validated copy with session-consistent pseudonyms.
        """
        document = value.model_dump(mode="json")
        self._register(document)
        patterns = [
            re.escape(text)
            if len(text) >= _SUBSTRING_MIN_LENGTH
            else rf"(?<![A-Za-z0-9]){re.escape(text)}(?![A-Za-z0-9])"
            for text in sorted(self.replacements, key=len, reverse=True)
            if text
        ]
        pattern = re.compile("|".join(patterns))

        def replace(item: object, key: str = "") -> object:
            if isinstance(item, str) and key not in _SCHEMA_FIELDS:
                return pattern.sub(lambda match: self.replacements[match.group()], item)
            if isinstance(item, dict):
                result = {k: replace(v, k) for k, v in item.items()}
                if "redacted" in result:
                    result["redacted"] = True
                return result
            if isinstance(item, list):
                return [replace(child, key) for child in item]
            return item

        model_type: type[_Model] = value.__class__
        return model_type.model_validate(replace(document))
