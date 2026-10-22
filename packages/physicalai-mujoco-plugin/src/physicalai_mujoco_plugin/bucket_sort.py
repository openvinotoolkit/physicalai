# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Layout, instructions and success for the ``bucket_sort`` scene.

Three buckets (black, blue, red) stand in a row of slots in front of the arm, and three objects
(a resistor, a 9 V battery and a CPU) lie between the arm and the buckets. Each episode names one
object and one bucket, "put the CPU in the red bucket". A reset:

- assigns the colours to the slots at random, so the bucket to use is told by its colour and not
  by where it stands, and jitters each bucket a little;
- drops the objects at random, non-overlapping positions and headings;
- samples the instruction from the allowed (object, bucket) pairs, all nine by default, so pairs
  can be held out of training (:func:`sample_instruction`, :class:`BucketSortTask`).

The episode succeeds when the named object rests inside the named bucket, out of the gripper,
and no other object is in a bucket (:meth:`BucketSortTask.evaluate`). :class:`BucketSortEpisode`
reports it in ``/health`` and the viewer, and starts a new episode after the success dwell.
"""

# MuJoCo model/data attributes are supplied by the C extension at runtime.
# pyrefly: ignore-errors [missing-attribute]

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from loguru import logger

from physicalai_mujoco_plugin.spawn import SPAWN_ATTEMPTS, sample_clear_position

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

OBJECTS = ("resistor", "battery", "cpu")
"""Object names: each is a body with the free joint ``<name>:joint``."""
OBJECT_LABELS = {"resistor": "resistor", "battery": "9 V battery", "cpu": "CPU"}
"""How an instruction names each object."""
BUCKET_COLORS = ("black", "blue", "red")
"""Bucket colours: each bucket is the mocap body ``bucket_<colour>``."""
ALL_PAIRS: tuple[tuple[str, str], ...] = tuple((obj, color) for obj in OBJECTS for color in BUCKET_COLORS)
"""Every (object, bucket colour) pair an instruction can name."""

BUCKET_SIZE = (0.150, 0.2025, 0.120)
"""The bucket's outer size in its own frame, in metres: across (x), long axis (y) and depth (z).

It lies on its long side with its opening up (``scripts/build_bucket_sort_meshes.py``)."""
BUCKET_YAW = 0.0
"""The buckets' turn about the vertical, in radians, before jitter: 0 lays their long axis along
the row (y), with the open end above the low lip at +y. Along the row, the buckets' far edge is
0.70 m from the base; turned along x, it would be 0.75 m, and the bucket centres 3 cm further out."""
SLOT_X = 0.62
"""Distance of the bucket row in front of the robot base, in metres: its near edge is 0.545 m away."""
SLOT_Y = (0.26, 0.0, -0.26)
"""Slot centres along y: left, centre and right, as seen from behind the robot."""
SLOT_NAMES = ("left", "centre", "right")
BUCKET_JITTER_XY = 0.01
"""Largest random shift of a bucket from its slot centre along x and along y, in metres."""
BUCKET_JITTER_YAW = float(np.radians(5.0))
"""Largest random turn of a bucket about the vertical, in radians. With the shifts, neighbouring
buckets (5.75 cm apart in their slots) stay at least 2.5 cm apart."""
BUCKET_INTERIOR = ((-0.069, 0.069), (-0.095, 0.095), (0.006, 0.120))
"""The space inside a bucket in its own frame, ``(x, y, z)`` ranges in metres: between its walls,
above its bottom, up to its rim."""

OBJECT_ZONE = ((0.29, 0.40), (-0.22, 0.22))
"""Where object centres spawn: ``(x, y)`` ranges in the world frame, in metres.

The near edge keeps the hand clear of the arm's joint limits 10 cm above the table. At the far
edge, a hand grasping from above with its fingers closing toward the buckets (it reaches 0.104 m
from the grasp point along that axis) stays 2 cm from a bucket at its largest jitter."""
OBJECT_GAP = 0.02
"""Smallest gap between the circles that enclose two objects at any heading, in metres."""
DROP_HEIGHT = 0.002
"""How far above the table an object is released at reset, in metres."""
SETTLE_SPEED = 0.02
"""Largest speed, in m/s, at which an object counts as resting."""

Pair = tuple[str, str]


@dataclass(frozen=True)
class Instruction:
    """One episode's instruction: put ``object`` in the ``bucket`` bucket."""

    object: str
    """One of :data:`OBJECTS`."""
    bucket: str
    """One of :data:`BUCKET_COLORS`."""

    @property
    def text(self) -> str:
        """The instruction as a sentence, e.g. ``put the CPU in the red bucket``."""
        return f"put the {OBJECT_LABELS[self.object]} in the {self.bucket} bucket"

    def as_dict(self) -> dict[str, str]:
        """Return the instruction for JSON: ``object``, ``bucket`` and ``text``."""
        return {"object": self.object, "bucket": self.bucket, "text": self.text}


def check_pairs(pairs: Iterable[Pair] | None) -> tuple[Pair, ...]:
    """Validate a set of allowed (object, bucket colour) pairs.

    Args:
        pairs: The allowed pairs; ``None`` allows all of :data:`ALL_PAIRS`.

    Returns:
        The pairs without repeats, in :data:`ALL_PAIRS` order.

    Raises:
        ValueError: If a pair names an unknown object or colour, or no pair is left.
    """
    if pairs is None:
        return ALL_PAIRS
    wanted = {(str(obj), str(color)) for obj, color in pairs}
    unknown = sorted(wanted - set(ALL_PAIRS))
    if unknown:
        msg = f"Unknown (object, bucket) pairs {unknown}; objects are {OBJECTS}, buckets {BUCKET_COLORS}"
        raise ValueError(msg)
    if not wanted:
        msg = "At least one (object, bucket) pair must be allowed"
        raise ValueError(msg)
    return tuple(pair for pair in ALL_PAIRS if pair in wanted)


def sample_instruction(rng: np.random.Generator, allowed_pairs: Iterable[Pair] | None = None) -> Instruction:
    """Draw an instruction uniformly from the allowed (object, bucket colour) pairs.

    Args:
        rng: The generator.
        allowed_pairs: The pairs to draw from (:func:`check_pairs`); ``None`` draws from all nine.

    Returns:
        The instruction.
    """
    pairs = check_pairs(allowed_pairs)
    obj, color = pairs[int(rng.integers(len(pairs)))]
    return Instruction(object=obj, bucket=color)


def _yaw_quat(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)])


@dataclass(frozen=True)
class TaskState:
    """Where the objects are relative to the buckets, and whether the instruction is done."""

    in_bucket: dict[str, str | None]
    """The bucket colour each object's centre is inside, or ``None``."""
    resting: dict[str, bool]
    """Whether each object is still and out of the robot's grasp."""
    success: bool
    """The named object rests in the named bucket, and no other object is in a bucket."""

    def as_dict(self) -> dict[str, object]:
        """Return the state for JSON."""
        return {"in_bucket": dict(self.in_bucket), "resting": dict(self.resting), "success": self.success}


class BucketSortTask:
    """The scene's reset and success test, with the current layout and instruction.

    Call it as the scene's reset callback, ``task(model, data, rng)``: it puts the arm at home,
    lays the buckets and objects out and samples the instruction. The allowed pairs can be
    changed at any time and apply from the next reset.
    """

    def __init__(
        self,
        *,
        allowed_pairs: Iterable[Pair] | None = None,
        place_home: Callable[[object, object], None] | None = None,
        robot_roots: tuple[int, ...] = (),
    ) -> None:
        """Create the task; nothing is laid out until the first reset.

        Args:
            allowed_pairs: The (object, bucket colour) pairs instructions are drawn from
                (:func:`check_pairs`); ``None`` allows all nine.
            place_home: Puts the arm at its home pose at rest; ``None`` leaves the arm where it is.
            robot_roots: Root body of every robot: objects touching them are held, not resting.
        """
        self._allowed = check_pairs(allowed_pairs)
        self._place_home = place_home
        self._robot_roots = tuple(robot_roots)
        self.instruction: Instruction | None = None
        """The current instruction; ``None`` before the first reset."""
        self.slot_colors: tuple[str, ...] = BUCKET_COLORS
        """The bucket colour in each slot, left to right (:data:`SLOT_Y`)."""

    @property
    def allowed_pairs(self) -> tuple[Pair, ...]:
        """The (object, bucket colour) pairs the next instructions are drawn from."""
        return self._allowed

    def set_allowed_pairs(self, pairs: Iterable[Pair] | None) -> None:
        """Draw later instructions from *pairs* (:func:`check_pairs`); ``None`` allows all nine.

        The current instruction stays until the next reset.
        """
        self._allowed = check_pairs(pairs)

    def __call__(self, model: object, data: object, rng: np.random.Generator) -> None:
        """Reset: the arm at home, the colours shuffled over the slots, the objects respawned, a new instruction."""
        import mujoco  # noqa: PLC0415

        if self._place_home is not None:
            self._place_home(model, data)
        order = rng.permutation(len(BUCKET_COLORS))
        self.slot_colors = tuple(BUCKET_COLORS[int(i)] for i in order)
        for color, slot_y in zip(self.slot_colors, SLOT_Y, strict=True):
            mocap = int(model.body_mocapid[_body(model, f"bucket_{color}")])
            dx, dy = rng.uniform(-BUCKET_JITTER_XY, BUCKET_JITTER_XY, size=2)
            data.mocap_pos[mocap] = (SLOT_X + dx, slot_y + dy, 0.0)
            yaw = BUCKET_YAW + float(rng.uniform(-BUCKET_JITTER_YAW, BUCKET_JITTER_YAW))
            data.mocap_quat[mocap] = _yaw_quat(yaw)
        _place_objects(model, data, rng)
        self.instruction = sample_instruction(rng, self._allowed)
        mujoco.mj_forward(model, data)
        logger.info("Bucket sort: {!r}; buckets left to right {}", self.instruction.text, ", ".join(self.slot_colors))

    def evaluate(self, model: object, data: object) -> TaskState:
        """Check where the objects are and whether the current instruction is done.

        Returns:
            The state; never a success before the first reset.
        """
        held = self._bodies_touching_robot(model, data)
        in_bucket: dict[str, str | None] = {}
        resting: dict[str, bool] = {}
        for name in OBJECTS:
            body = _body(model, name)
            dof = int(model.jnt_dofadr[_joint(model, f"{name}:joint")])
            in_bucket[name] = bucket_containing(model, data, data.xpos[body])
            speed = float(np.linalg.norm(data.qvel[dof : dof + 3]))
            resting[name] = speed <= SETTLE_SPEED and body not in held
        instruction = self.instruction
        success = (
            instruction is not None
            and in_bucket[instruction.object] == instruction.bucket
            and resting[instruction.object]
            and all(in_bucket[name] is None for name in OBJECTS if name != instruction.object)
        )
        return TaskState(in_bucket=in_bucket, resting=resting, success=success)

    def describe(self) -> dict[str, object]:
        """Return the instruction, the colour in each slot and the allowed pairs, for JSON."""
        return {
            "instruction": self.instruction.as_dict() if self.instruction is not None else None,
            "slots": [
                {"slot": slot, "y": y, "color": color}
                for slot, y, color in zip(SLOT_NAMES, SLOT_Y, self.slot_colors, strict=True)
            ],
            "allowed_pairs": [list(pair) for pair in self._allowed],
        }

    def _bodies_touching_robot(self, model: object, data: object) -> set[int]:
        ncon = int(data.ncon)
        if ncon == 0 or not self._robot_roots:
            return set()
        robot = np.isin(np.asarray(model.body_rootid), self._robot_roots)
        bodies = np.asarray(model.geom_bodyid)
        b1 = bodies[np.asarray(data.contact.geom1[:ncon])]
        b2 = bodies[np.asarray(data.contact.geom2[:ncon])]
        r1, r2 = robot[b1], robot[b2]
        return {int(b) for b in np.concatenate([b2[r1 & ~r2], b1[r2 & ~r1]])}


def _place_objects(model: object, data: object, rng: np.random.Generator) -> None:
    """Drop every object flat at a random heading in :data:`OBJECT_ZONE`, clear of the others."""
    placed: list[tuple[float, float, float]] = []
    (x_lo, x_hi), (y_lo, y_hi) = OBJECT_ZONE
    for name in OBJECTS:
        radius, half_height = object_extent(model, name)

        def draw() -> tuple[float, float]:
            return float(rng.uniform(x_lo, x_hi)), float(rng.uniform(y_lo, y_hi))

        def clear(xy: tuple[float, float], radius: float = radius) -> bool:
            return all(np.hypot(xy[0] - x, xy[1] - y) >= radius + r + OBJECT_GAP for x, y, r in placed)

        x, y = sample_clear_position(draw, clear_of_others=clear, attempts=SPAWN_ATTEMPTS)
        placed.append((x, y, radius))
        joint = _joint(model, f"{name}:joint")
        adr, dof = int(model.jnt_qposadr[joint]), int(model.jnt_dofadr[joint])
        data.qpos[adr : adr + 3] = (x, y, half_height + DROP_HEIGHT)
        data.qpos[adr + 3 : adr + 7] = _yaw_quat(float(rng.uniform(0.0, 2.0 * np.pi)))
        data.qvel[dof : dof + 6] = 0.0


def bucket_containing(model: object, data: object, point: np.ndarray) -> str | None:
    """Return the colour of the bucket whose interior (:data:`BUCKET_INTERIOR`) holds *point*.

    Returns:
        The colour, or ``None`` outside every bucket.
    """
    for color in BUCKET_COLORS:
        body = _body(model, f"bucket_{color}")
        local = data.xmat[body].reshape(3, 3).T @ (np.asarray(point) - data.xpos[body])
        if all(lo <= value <= hi for value, (lo, hi) in zip(local, BUCKET_INTERIOR, strict=True)):
            return color
    return None


def object_extent(model: object, name: str) -> tuple[float, float]:
    """Return an object's horizontal radius at any heading and its half height, lying as written.

    Both come from the bounding boxes of the object's collision geoms about its body origin.

    Returns:
        ``(radius, half_height)`` in metres.
    """
    import mujoco  # noqa: PLC0415

    body = _body(model, name)
    radius, half_height = 0.0, 0.0
    rotation = np.zeros(9)
    corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=np.float64)
    for geom in np.flatnonzero(np.asarray(model.geom_bodyid) == body):
        if int(model.geom_contype[geom]) == 0 and int(model.geom_conaffinity[geom]) == 0:
            continue
        mujoco.mju_quat2Mat(rotation, model.geom_quat[geom])
        points = (
            model.geom_pos[geom]
            + (model.geom_aabb[geom, :3] + corners * model.geom_aabb[geom, 3:]) @ rotation.reshape(3, 3).T
        )
        radius = max(radius, float(np.hypot(points[:, 0], points[:, 1]).max()))
        half_height = max(half_height, float(-points[:, 2].min()))
    return radius, half_height


Phase = Literal["idle", "success_hold"]


class BucketSortEpisode:
    """Report the task in ``/health`` and the viewer, and start a new episode after a success.

    It has the controls of the other episode helpers (``active``, the success dwell). Once the
    instruction has held for the dwell, it resets the scene as ``POST /reset`` does.
    """

    def __init__(
        self, task: BucketSortTask, rng: np.random.Generator, *, success_dwell_s: float = 5.0, active: bool = True
    ) -> None:
        """Watch *task*; *rng* is the scene's reset generator, used for the automatic resets."""
        self.task = task
        self._rng = rng
        self._dwell_s = float(success_dwell_s)
        self._active = active
        self._phase: Phase = "idle"
        self._success_since: float | None = None
        self._countdown_s: float | None = None
        self._episode_count = 0
        self._state: TaskState | None = None

    @property
    def active(self) -> bool:
        """Whether a success starts a new episode after the dwell."""
        return self._active

    def set_active(self, active: bool) -> None:  # noqa: FBT001
        """Pause or resume the automatic reset; pausing cancels a running countdown."""
        self._active = active
        if not active:
            self.notify_manual_reset()

    @property
    def dwell_s(self) -> float:
        """Seconds the instruction must stay done before the next episode starts."""
        return self._dwell_s

    def set_dwell(self, dwell_s: float) -> None:
        """Change the success dwell; a running countdown uses the new value."""
        self._dwell_s = float(dwell_s)

    def notify_manual_reset(self) -> None:
        """Clear the success countdown after an explicit reset."""
        self._phase = "idle"
        self._success_since = None
        self._countdown_s = None

    def status(self) -> dict[str, object]:
        """Return a JSON-friendly snapshot for ``/health`` and the viewer."""
        state = self._state
        return {
            "enabled": True,
            "kind": "bucket_sort",
            "active": self._active,
            "phase": self._phase,
            "episode_count": self._episode_count,
            "success_dwell_s": self._dwell_s,
            "countdown_s": self._countdown_s,
            **self.task.describe(),
            "success": state.success if state is not None else False,
            "in_bucket": dict(state.in_bucket) if state is not None else dict.fromkeys(OBJECTS),
        }

    def update(self, model: object, data: object) -> None:
        """Evaluate the task for one tick, and reset the scene once a success has held for the dwell."""
        state = self.task.evaluate(model, data)
        self._state = state
        if not self._active:
            return
        if not state.success:
            if self._phase == "success_hold":
                logger.info("Bucket sort: the instruction is no longer met; countdown cancelled")
            self.notify_manual_reset()
            return
        now = float(data.time)
        if self._success_since is None:
            self._success_since = now
            self._phase = "success_hold"
            logger.info("Bucket sort: success; next episode in {:.1f} s", self._dwell_s)
        self._countdown_s = max(0.0, self._dwell_s - (now - self._success_since))
        if self._countdown_s > 0.0:
            return
        self.task(model, data, self._rng)
        self._episode_count += 1
        self.notify_manual_reset()
        self._state = self.task.evaluate(model, data)


def _body(model: object, name: str) -> int:
    import mujoco  # noqa: PLC0415

    body = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name))
    if body < 0:
        msg = f"The bucket_sort model has no body {name!r}"
        raise ValueError(msg)
    return body


def _joint(model: object, name: str) -> int:
    import mujoco  # noqa: PLC0415

    joint = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name))
    if joint < 0:
        msg = f"The bucket_sort model has no joint {name!r}"
        raise ValueError(msg)
    return joint


__all__ = [
    "ALL_PAIRS",
    "BUCKET_COLORS",
    "BUCKET_INTERIOR",
    "BUCKET_SIZE",
    "BUCKET_YAW",
    "OBJECTS",
    "OBJECT_LABELS",
    "OBJECT_ZONE",
    "SLOT_X",
    "SLOT_Y",
    "BucketSortEpisode",
    "BucketSortTask",
    "Instruction",
    "TaskState",
    "bucket_containing",
    "check_pairs",
    "object_extent",
    "sample_instruction",
]
