# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""How Studio starts the simulation that a simulated robot type attaches to."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Protocol

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True)
class SimulationScene:
    """A scene the user can pick when Studio starts a simulation.

    Attributes:
        id: Stable identifier passed to :attr:`SimulationLaunch.build_argv`.
        display_name: Human-readable name shown in Studio's scene picker.
        description: One-line description shown next to the name.
    """

    id: str
    display_name: str
    description: str = ""

    def __post_init__(self) -> None:
        """Validate the scene.

        Raises:
            ValueError: If ``id`` or ``display_name`` is blank.
        """
        if not self.id.strip():
            msg = "scene id must not be empty"
            raise ValueError(msg)
        if not self.display_name.strip():
            msg = f"scene {self.id!r} needs a display_name"
            raise ValueError(msg)


class SimulationArgvBuilder(Protocol):
    """Builds the command line that starts a simulation; see :class:`SimulationLaunch`."""

    def __call__(self, *, scene: str, owner_name: str, seed: int | None) -> Sequence[str]:
        """Return the argv, executable first, that starts *scene* under *owner_name*.

        Args:
            scene: The id of one of the launch's scenes.
            owner_name: Name the simulation publishes its robot under, the one the robot's
                payload attaches to.
            seed: Fixed seed for the scene's randomization, or ``None`` for random layouts.
        """
        ...


@dataclass(frozen=True)
class SimulationLaunch:
    """How Studio starts and supervises the simulation of a simulated robot type.

    Studio shows ``scenes`` in a start dialog, preselecting ``default_scene``, and runs
    :meth:`argv` as a child process with this contract:

    - stdin is a pipe that Studio never writes to; the simulation stops when it reaches end of
      file, which also happens when Studio exits or crashes;
    - stdout carries one JSON object per line: ``{"event": "phase", "phase": ...}`` while it starts,
      then ``{"event": "ready", "name", "http_url", "viewer_url", "cameras", ...}``, or
      ``{"event": "error", "message"}``;
    - the process exits with a non-zero code when it fails.

    Attributes:
        scenes: The scenes the user can pick, in the order to show them.
        default_scene: Id of the scene to preselect; one of ``scenes``.
        build_argv: Builds the command line for a scene, an owner name and an optional seed.
    """

    max_seed: ClassVar[int] = 2**32 - 1
    """Largest seed Studio passes; seeds go from 0 to this value."""

    scenes: tuple[SimulationScene, ...]
    default_scene: str
    build_argv: SimulationArgvBuilder

    def __post_init__(self) -> None:
        """Validate the scenes and the default scene.

        Raises:
            ValueError: If ``scenes`` is empty, two scenes share an id, or ``default_scene`` is not
                one of them.
        """
        if not self.scenes:
            msg = "scenes must not be empty"
            raise ValueError(msg)
        ids = [scene.id for scene in self.scenes]
        duplicates = sorted({scene_id for scene_id in ids if ids.count(scene_id) > 1})
        if duplicates:
            msg = f"scene ids must be unique, got duplicates {duplicates}"
            raise ValueError(msg)
        if self.default_scene not in ids:
            msg = f"default_scene {self.default_scene!r} is not one of the scenes {ids}"
            raise ValueError(msg)

    def argv(self, scene: str, owner_name: str, seed: int | None = None) -> list[str]:
        """Return the command line that starts *scene* under *owner_name*.

        Args:
            scene: The id of one of ``scenes``.
            owner_name: Name the simulation publishes its robot under.
            seed: Fixed seed from 0 to :attr:`max_seed`, or ``None`` for random layouts.

        Returns:
            The argv from ``build_argv``, executable first.

        Raises:
            ValueError: If the scene is unknown, the owner name is blank, the seed is out of range,
                or ``build_argv`` returns an empty command line.
        """
        if scene not in {known.id for known in self.scenes}:
            msg = f"Unknown scene {scene!r}; choose one of {[known.id for known in self.scenes]}"
            raise ValueError(msg)
        if not owner_name.strip():
            msg = "owner_name must not be empty"
            raise ValueError(msg)
        if seed is not None and (type(seed) is not int or not 0 <= seed <= self.max_seed):
            msg = f"seed must be an integer from 0 to {self.max_seed}, got {seed!r}"
            raise ValueError(msg)
        argv = list(self.build_argv(scene=scene, owner_name=owner_name, seed=seed))
        if not argv:
            msg = f"build_argv returned no command line for scene {scene!r}"
            raise ValueError(msg)
        return argv
