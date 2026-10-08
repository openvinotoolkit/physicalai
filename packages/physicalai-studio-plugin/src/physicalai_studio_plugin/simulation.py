# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""How Studio starts the simulation that a simulated robot type attaches to."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, Protocol, TypeVar

from pydantic import BaseModel

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

_PayloadT = TypeVar("_PayloadT", bound=BaseModel)


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
            seed: Fixed seed for the scene's randomization, from 0 to the launch's ``max_seed``, or
                ``None`` for random layouts.
        """
        ...


@dataclass(frozen=True)
class SimulationLaunch(Generic[_PayloadT]):
    """How Studio starts and supervises the simulation of a simulated robot type.

    Studio shows ``scenes`` in a start dialog, preselecting ``default_scene``, next to ``labels``,
    and runs :meth:`argv` for the robot's payload as a child process with this contract:

    - stdin is a pipe that Studio never writes to; the simulation stops when it reaches end of
      file, which also happens when Studio exits or crashes;
    - stdout carries one JSON object per line: ``{"event": "phase", "phase": ...}`` while it starts,
      then ``{"event": "ready", "name", "http_url", "viewer_url", "cameras", ...}``, or
      ``{"event": "error", "message"}``; every run ends with exactly one ``ready`` or ``error``;
      ``ready.name`` is :meth:`owner_name` of the payload;
    - Studio uses ``ready.http_url`` and ``ready.viewer_url``, not the payload's addresses: the
      simulation may pick free ports, so Studio copies them onto the robot's payload (or otherwise
      uses them for cameras and the embedded viewer) after ``ready``;
    - the process exits with a non-zero code when it fails.

    Parameterize it with the robot type's payload model, like ``RobotCatalogDefinition``.

    Attributes:
        scenes: The scenes the user can pick, in the order to show them.
        default_scene: Id of the scene to preselect; one of ``scenes``.
        payload_owner_name: Returns the name a robot built from a payload attaches to, which the
            simulation must publish its robot under.
        build_argv: Builds the command line for a scene, an owner name and an optional seed.
        max_seed: Largest fixed seed the simulation takes (seeds go from 0 to it), or ``None`` when
            it takes no seed.
        labels: Short read-only facts about the simulated robot, such as its arm count, that
            Studio shows as chips in the start dialog, in order.
    """

    scenes: tuple[SimulationScene, ...]
    default_scene: str
    payload_owner_name: Callable[[_PayloadT], str]
    build_argv: SimulationArgvBuilder
    max_seed: int | None = None
    labels: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate the scenes, the default scene, the seed range and the labels.

        Raises:
            ValueError: If ``scenes`` is empty, two scenes share an id, ``default_scene`` is not
                one of them, ``max_seed`` is negative, or a label is blank.
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
        if self.max_seed is not None and (type(self.max_seed) is not int or self.max_seed < 0):
            msg = f"max_seed must be a non-negative integer or None, got {self.max_seed!r}"
            raise ValueError(msg)
        if any(not label.strip() for label in self.labels):
            msg = f"labels must not be blank, got {list(self.labels)}"
            raise ValueError(msg)

    def owner_name(self, payload: _PayloadT) -> str:
        """Return the name the simulation for *payload* publishes its robot under.

        Studio compares it with ``ready.name`` before it attaches the robot.

        Returns:
            The name from ``payload_owner_name``.

        Raises:
            ValueError: If the name is blank.
        """
        name = self.payload_owner_name(payload)
        if not name.strip():
            msg = "payload_owner_name returned an empty owner name"
            raise ValueError(msg)
        return name

    def argv(self, scene: str, payload: _PayloadT, seed: int | None = None) -> list[str]:
        """Return the command line that starts *scene* for the robot configured by *payload*.

        Args:
            scene: The id of one of ``scenes``.
            payload: The robot's payload; the simulation runs under its :meth:`owner_name`.
            seed: Fixed seed from 0 to ``max_seed``, or ``None`` for random layouts.

        Returns:
            The argv from ``build_argv``, executable first.

        Raises:
            ValueError: If the scene is unknown, the owner name is blank, the seed is out of range
                (any seed when ``max_seed`` is ``None``), or ``build_argv`` returns an empty command
                line.
        """
        if scene not in {known.id for known in self.scenes}:
            msg = f"Unknown scene {scene!r}; choose one of {[known.id for known in self.scenes]}"
            raise ValueError(msg)
        if seed is not None:
            if self.max_seed is None:
                msg = "this simulation takes no seed"
                raise ValueError(msg)
            if type(seed) is not int or not 0 <= seed <= self.max_seed:
                msg = f"seed must be an integer from 0 to {self.max_seed}, got {seed!r}"
                raise ValueError(msg)
        argv = list(self.build_argv(scene=scene, owner_name=self.owner_name(payload), seed=seed))
        if not argv:
            msg = f"build_argv returned no command line for scene {scene!r}"
            raise ValueError(msg)
        return argv
