# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._sources import ProjectSource

_PROFILE_ITEM_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")


@dataclass(frozen=True)
class SyncProfile:
    """Select the locked dependencies and Python version for an environment.

    Attributes:
        python: Python version or interpreter selector passed to ``uv sync``.
        extras: Project extras enabled by this profile.
        groups: Dependency groups enabled by this profile.
        no_default_groups: Whether to exclude the project's default dependency groups.
    """

    python: str | None = None
    extras: tuple[str, ...] = ()
    groups: tuple[str, ...] = ()
    no_default_groups: bool = False

    def __post_init__(self) -> None:
        """Normalize sequences and reject invalid uv selectors.

        Raises:
            TypeError: If a selector collection or flag has the wrong type.
            ValueError: If a selector is malformed or repeated.
        """
        if self.python is not None:
            if not isinstance(self.python, str):
                msg = "The Python selector must be a string."
                raise TypeError(msg)
            python = self.python.strip()
            object.__setattr__(self, "python", python)
            if not python or python.startswith("-"):
                msg = "The Python selector must be a non-empty value that does not start with '-'."
                raise ValueError(msg)
        if isinstance(self.extras, str) or isinstance(self.groups, str):
            msg = "Extras and dependency groups must be sequences of names, not strings."
            raise TypeError(msg)
        try:
            extras = tuple(self.extras)
            groups = tuple(self.groups)
        except TypeError as exc:
            msg = "Extras and dependency groups must be iterable sequences of names."
            raise TypeError(msg) from exc
        object.__setattr__(self, "extras", extras)
        object.__setattr__(self, "groups", groups)
        for values, label in ((extras, "extra"), (groups, "dependency group")):
            for value in values:
                if not isinstance(value, str) or _PROFILE_ITEM_RE.fullmatch(value) is None:
                    msg = f"Invalid {label} name {value!r}."
                    raise ValueError(msg)
            if len(values) != len(set(values)):
                msg = f"A sync profile cannot repeat a {label}."
                raise ValueError(msg)
        if not isinstance(self.no_default_groups, bool):
            msg = "no_default_groups must be a boolean."
            raise TypeError(msg)

    def identity(self) -> dict[str, object]:
        """Return canonical inputs selected by this profile.

        Returns:
            Python selector, extras, dependency groups, and default-group setting.
        """
        return {
            "python": self.python,
            "extras": sorted(self.extras),
            "groups": sorted(self.groups),
            "no_default_groups": self.no_default_groups,
        }


@dataclass(frozen=True)
class Environment:
    """A project source and the uv sync profile used to prepare it.

    Environment declarations intentionally do not bind a worker or entrypoint.
    Pass an entrypoint to :func:`physicalai.isolate.run` or
    :func:`physicalai.isolate.start` when launching a session.

    Attributes:
        source: A project source such as :class:`BundledProject` or :class:`GitProject`.
        profile: The uv sync profile to install from the project's lock.
    """

    source: ProjectSource
    profile: SyncProfile = field(default_factory=SyncProfile)

    def __post_init__(self) -> None:
        """Ensure declarations use a project source and sync profile.

        Raises:
            TypeError: If either declaration field has an unsupported type.
        """
        if not isinstance(self.profile, SyncProfile):
            msg = "Environment profile must be a SyncProfile."
            raise TypeError(msg)
        required_methods = ("identity", "materialize", "provenance")
        if any(not callable(getattr(self.source, method, None)) for method in required_methods):
            methods = ", ".join(f"{method}()" for method in required_methods)
            msg = f"Environment source must provide {methods} methods."
            raise TypeError(msg)

    def identity(self) -> dict[str, object]:
        """Return the source and profile inputs that determine the environment.

        Returns:
            JSON-compatible source identity and sync profile.
        """
        return {"format": 1, "source": self.source.identity(), "profile": self.profile.identity()}

    def key(self) -> str:
        """Return the SHA-256 cache key for this source and sync profile.

        Returns:
            A lowercase hexadecimal SHA-256 digest.
        """
        payload = json.dumps(self.identity(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()
