# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Robot models published outside MuJoCo Menagerie, downloaded into the Menagerie cache on first use.

OpenArm is not in Menagerie (google-deepmind/mujoco_menagerie#197). Enactic, who make it, publish its
MJCF as the ``openarm-mujoco`` package, and the package's source archive on PyPI holds the v2 model
files. The plugin downloads that archive when a profile first needs it (or on ``physicalai-mujoco
prefetch``) instead of depending on the package: a dependency would install 22 MB of meshes for
every user, into a ``share/`` data directory that the package finds through ``sysconfig`` and that
``pip install --target`` and bundled installs do not keep.

The pinned URL, size and digest fix every file, as the ``mujoco-menagerie`` pin does; PyPI never
replaces a published file. ``mujoco_menagerie.Cache`` checks the archive against the digest and
publishes it in the Menagerie cache directory (``MENAGERIE_CACHE_DIR``). An environment variable per
model (``OPENARM_MUJOCO_ROOT``) points at a local copy instead, for offline machines.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from collections.abc import Callable

    import mujoco_menagerie

_ARCHIVE_SUFFIX = ".tar.gz"


@dataclass(frozen=True)
class ExternalModel:
    """A robot model distributed as a source archive, fetched like a Menagerie model."""

    name: str
    """What a profile's ``menagerie_model`` names it by."""
    display_name: str
    license: str
    archive_url: str
    """A ``.tar.gz`` whose single top-level directory is the file name without the suffix."""
    sha256: str
    size: int
    """Archive size in bytes."""
    entries: tuple[tuple[str, str], ...]
    """Entry points: a name, and its XML file relative to the archive's top-level directory."""
    default_entry: str
    root_env: str
    """Environment variable naming a directory laid out like the archive's top level, used instead."""

    @property
    def archive_name(self) -> str:
        """The archive's file name."""
        return self.archive_url.rsplit("/", 1)[1]

    def entry_file(self, entry: str | None = None) -> str:
        """Return the XML file of *entry* (``None``: the default), relative to the archive's top level.

        Raises:
            RuntimeError: If the model has no such entry.
        """
        name = entry or self.default_entry
        files = dict(self.entries)
        if name not in files:
            msg = f"The {self.display_name} model has no entry {name!r}; it has {', '.join(files)}"
            raise RuntimeError(msg)
        return files[name]

    def menagerie_robot(self) -> mujoco_menagerie.Robot:
        """Describe the archive as a Menagerie model, so that ``mujoco_menagerie.Cache`` can fetch it.

        Returns:
            A robot whose name is the archive's top-level directory and whose tree id is its digest.
        """
        import mujoco_menagerie  # noqa: PLC0415

        return mujoco_menagerie.Robot(
            name=self.archive_name.removesuffix(_ARCHIVE_SUFFIX),
            display_name=self.display_name,
            category="arm",
            license=self.license,
            oid=self.sha256,
            asset=self.archive_name,
            sha256=self.sha256,
            download_size=self.size,
            installed_size=0,
            entry_points=tuple(mujoco_menagerie.EntryPoint(name, "robot", file) for name, file in self.entries),
            default_model=self.default_entry,
            default_scene=None,
        )


OPENARM_MUJOCO_VERSION = "2.3.0"
OPENARM_V2 = ExternalModel(
    name="openarm_v2",
    display_name="OpenArm v2",
    license="Apache-2.0",
    archive_url=(
        "https://files.pythonhosted.org/packages/d7/cf/d4ce0f428aef402af4adbdb0d2c22392b644c7d9853d066fb5652d757735/"
        f"openarm_mujoco-{OPENARM_MUJOCO_VERSION}.tar.gz"
    ),
    sha256="2e32cfac261d3f4dcbb5ebfa7e2ba2d7205dd5a9655798cff13170a0cf1b5339",
    size=6071364,
    entries=(("openarm_bimanual", "v2/openarm_bimanual.xml"),),
    default_entry="openarm_bimanual",
    root_env="OPENARM_MUJOCO_ROOT",
)
"""Enactic's OpenArm v2 MJCF, from the ``openarm-mujoco`` source archive on PyPI.

``OPENARM_MUJOCO_ROOT`` may point at a checkout of github.com/enactic/openarm_mujoco at the
same version, or at the unpacked archive.
"""

EXTERNAL_MODELS: dict[str, ExternalModel] = {model.name: model for model in (OPENARM_V2,)}
"""External models by the name a profile's ``menagerie_model`` gives."""


def _cache(model: ExternalModel) -> mujoco_menagerie.Cache:
    import mujoco_menagerie  # noqa: PLC0415

    cache = mujoco_menagerie.Cache(base_url=model.archive_url.rsplit("/", 1)[0])
    cache.root = None  # MENAGERIE_ROOT is a Menagerie checkout, which does not hold this archive
    return cache


def external_available(model: ExternalModel, entry: str | None = None) -> bool:
    """Return whether *model* loads without a download: from its local copy, or from the cache.

    Returns:
        ``True`` when ``root_env`` names a directory with the entry's file, or the archive is cached.
    """
    root = os.environ.get(model.root_env)
    if root:
        return (Path(root) / model.entry_file(entry)).is_file()
    return _cache(model).is_cached(model.menagerie_robot())


def fetch_external(
    model: ExternalModel,
    entry: str | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> Path:
    """Put *model* in the cache, downloading it if needed, and return *entry*'s XML file.

    Args:
        model: The external model.
        entry: An entry point name; ``None`` takes the default.
        progress: Called with the bytes downloaded so far and the total after each chunk.

    Returns:
        The entry's XML file.

    Raises:
        RuntimeError: If the local copy lacks the file, or the archive cannot be downloaded or checked.
    """
    import mujoco_menagerie  # noqa: PLC0415

    file = model.entry_file(entry)
    root = os.environ.get(model.root_env)
    if root:
        path = Path(root) / file
        if not path.is_file():
            msg = f"{model.root_env}={root} has no {file}; point it at an openarm_mujoco checkout or unpacked archive"
            raise RuntimeError(msg)
        return path
    cache = _cache(model)
    robot = model.menagerie_robot()
    if not cache.is_cached(robot):
        logger.info("Downloading the {} model ({:.1f} MB) into {}", model.display_name, model.size / 2**20, cache.dir)

    def report(name: str, done: int, total: int) -> None:
        mujoco_menagerie.print_progress(name, done, total)
        if progress is not None:
            progress(done, total)

    try:
        return robot.path(cache, report) / file
    except mujoco_menagerie.DownloadError as exc:
        msg = (
            f"Could not download the {model.display_name} model from {model.archive_url}: {exc}. "
            f"Connect to the internet and run `physicalai-mujoco prefetch`, "
            f"or set {model.root_env} to a local copy."
        )
        raise RuntimeError(msg) from exc
    except mujoco_menagerie.MenagerieError as exc:
        raise RuntimeError(str(exc)) from exc


__all__ = [
    "EXTERNAL_MODELS",
    "OPENARM_MUJOCO_VERSION",
    "OPENARM_V2",
    "ExternalModel",
    "external_available",
    "fetch_external",
]
