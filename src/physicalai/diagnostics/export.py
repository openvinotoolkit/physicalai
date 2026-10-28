# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Passive Runtime recipe assembly and crash-safe user-selected file output."""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import yaml

from physicalai.config import Config

if TYPE_CHECKING:
    from collections.abc import Mapping


def component_bundle(robots: Mapping[str, Config], cameras: Mapping[str, Config]) -> dict[str, object]:
    """Return named recipes for Python integration, not a ``pai run`` document."""
    return {
        "schema_version": 1,
        "kind": "physicalai.components",
        "robots": {name: recipe.to_dict() for name, recipe in robots.items()},
        "cameras": {name: recipe.to_dict() for name, recipe in cameras.items()},
    }


def runtime_document(
    robot: Config, cameras: Mapping[str, Config], action_source: Config, *, fps: float
) -> dict[str, object]:
    """Assemble the existing Runtime schema without importing configured classes.

    Returns:
        A complete CLI-shaped Runtime configuration document.

    Raises:
        ValueError: If the control-loop rate is invalid.
    """
    if not math.isfinite(fps) or fps <= 0:
        msg = "Runtime FPS must be finite and positive."
        raise ValueError(msg)
    recipe = Config(
        "physicalai.runtime.RobotRuntime",
        {
            "robot": robot.to_dict(),
            "cameras": {name: cam.to_dict() for name, cam in cameras.items()},
            "action_source": action_source.to_dict(),
            "fps": fps,
        },
    )
    return {"runtime": recipe.to_dict()["init_args"]}


def encode_document(document: Mapping[str, object], *, format: Literal["yaml", "json"]) -> str:  # noqa: A002 - public file-format option
    """Serialize data only; never resolve ``class_path`` or instantiate objects.

    Returns:
        UTF-8-ready serialized text with a final newline.
    """
    if format == "json":
        return json.dumps(document, indent=2, allow_nan=False, ensure_ascii=False) + "\n"
    return yaml.safe_dump(dict(document), sort_keys=False, allow_unicode=True)


def output_format(path: Path, explicit: Literal["yaml", "json"] | None = None) -> Literal["yaml", "json"]:
    """Infer format while rejecting misleading extension/format combinations.

    Returns:
        The chosen YAML or JSON serialization format.

    Raises:
        ValueError: If the extension conflicts or a format cannot be determined.
    """
    inferred = {".yaml": "yaml", ".yml": "yaml", ".json": "json"}.get(path.suffix.lower())
    if inferred and explicit and inferred != explicit:
        msg = f"The {path.suffix} extension conflicts with --format {explicit}."
        raise ValueError(msg)
    chosen = explicit or inferred
    if chosen not in {"yaml", "json"}:
        msg = "Choose a .yaml/.yml/.json destination or specify --format."
        raise ValueError(msg)
    return "json" if chosen == "json" else "yaml"


def write_document(path: Path, text: str, *, overwrite: bool = False) -> None:
    """Publish a complete file atomically, without clobbering unapproved files.

    Temporary files live beside the destination to stay on the same filesystem.
    A hard-link publication makes the no-clobber decision atomic; explicit
    replacements use ``os.replace``. The parent must already exist.

    """
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp = Path(temporary)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            temp.replace(path)
        else:
            os.link(temp, path)
    finally:
        temp.unlink(missing_ok=True)
