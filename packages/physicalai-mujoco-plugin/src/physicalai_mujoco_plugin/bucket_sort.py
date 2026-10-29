# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Layout, instructions and success for the ``bucket_sort`` scene.

Three buckets (black, blue, red) stand in front of the arm, and three objects (a resistor, a 9 V
battery and a CPU) lie between the arm and the buckets. Each episode names one object and one
bucket, "put the CPU in the red bucket". A reset:

- lays the buckets out anywhere in their area (:data:`BUCKET_X`, :data:`BUCKET_Y`), each turned
  up to :data:`BUCKET_YAW_RANGE` away from facing the robot, at least :data:`BUCKET_GAP` apart and
  in the arm's reach, and gives them the colours in a random order, so neither the place nor the
  order tells which bucket is meant (:func:`sample_layout`). The buckets are free bodies: the arm
  can push them and the viewer or ``POST /objects/{joint}/pose`` can move them, and the success
  test follows them wherever they are;
- drops the objects at random positions and headings in their area (:data:`OBJECT_X`,
  :data:`OBJECT_Y`), apart from each other and far enough from the buckets that a hand grasping
  from above clears the bucket walls;
- samples the instruction from the allowed (object, bucket) pairs, all nine by default, so pairs
  can be held out of training (:func:`sample_instruction`, :class:`BucketSortTask`), and words it
  in one of many ways (:data:`PHRASES`): "Drop the 9V battery in the blue bin.", "Can you put the
  chip in the red box?".

The episode succeeds when the named object rests inside the named bucket, out of the gripper,
and no other object is in a bucket (:meth:`BucketSortTask.evaluate`). :class:`BucketSortEpisode`
reports it in ``/health`` and the viewer, and starts a new episode after the success dwell.
"""

# MuJoCo model/data attributes are supplied by the C extension at runtime.
# pyrefly: ignore-errors [missing-attribute]

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

OBJECTS = ("resistor", "battery", "cpu")
"""Object names: each is a body with the free joint ``<name>:joint``."""
OBJECT_LABELS = {"resistor": "resistor", "battery": "9 V battery", "cpu": "CPU"}
"""How the canonical instruction (:attr:`Instruction.canonical_text`) names each object."""
BUCKET_COLORS = ("black", "blue", "red")
"""Bucket colours: each bucket is the body ``bucket_<colour>`` with the free joint ``bucket_<colour>:joint``."""
BUCKET_JOINTS = tuple(f"bucket_{color}:joint" for color in BUCKET_COLORS)
"""The buckets' free joints, in :data:`BUCKET_COLORS` order."""
ALL_PAIRS: tuple[tuple[str, str], ...] = tuple((obj, color) for obj in OBJECTS for color in BUCKET_COLORS)
"""Every (object, bucket colour) pair an instruction can name."""

BUCKET_SIZE = (0.150, 0.2025, 0.120)
"""The bucket's outer size in its own frame, in metres: across (x), long axis (y) and depth (z).

It lies on its long side with its opening up (``scripts/build_bucket_sort_meshes.py``)."""
BUCKET_YAW = float(np.pi / 2)
"""The turn about the vertical, in radians (counter-clockwise seen from above), that points a
bucket's open end straight at the robot.

The bucket's open end, above its low lip, is at +y in its own frame; a quarter turn points it at
the robot (-x), with the long axis along x. A reset adds up to :data:`BUCKET_YAW_RANGE` to it."""
BUCKET_INTERIOR = ((-0.069, 0.069), (-0.095, 0.095), (0.006, 0.120))
"""The space inside a bucket in its own frame, ``(x, y, z)`` ranges in metres: between its walls,
above its bottom, up to its rim."""

# ----------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------
# Every range below is in the world frame, in metres (the robot base at the origin, facing +x).

TABLE_X = (-0.17, 0.83)
"""The table top's extent along x (``scene.xml``)."""
TABLE_Y = (-0.9, 0.9)
"""The table top's extent along y."""
TABLE_MARGIN = 0.03
"""Smallest distance from a bucket's footprint to the table's edges."""
BUCKET_X = (0.50, 0.72)
"""Where bucket centres are drawn along x, before the checks below."""
BUCKET_Y = (-0.40, 0.40)
"""Where bucket centres are drawn along y. The buckets are drawn one by one, each anywhere, so
they stand in a row, staggered, or two behind each other."""
BUCKET_YAW_RANGE = float(np.radians(25.0))
"""Largest turn of a bucket away from :data:`BUCKET_YAW`, either way, in radians: the open end
still faces the robot's side of the table."""
BUCKET_GAP = 0.03
"""Smallest gap between two buckets' footprints (their rotated outer rectangles)."""
OBJECT_X = (0.30, 0.48)
"""Where object centres are drawn along x."""
OBJECT_Y = (-0.34, 0.34)
"""Where object centres are drawn along y."""
OBJECT_GAP = 0.02
"""Smallest gap between the circles that enclose two objects at any heading."""
OBJECT_BUCKET_GAP = 0.06
"""Smallest gap between an object's enclosing circle and a bucket's footprint."""
REACH_RADIUS = (0.35, 0.76)
"""Horizontal distance from the base's axis, ``(nearest, farthest)``, at which the hand reaches
every object centre and every point of :data:`BUCKET_REACH_POINTS`.

Measured with damped least-squares IK of the FR3 with the fingers down, for four hand headings
(each with its half-turn twin), the null space pulling the joints away from their limits: at
0.35 m every joint stays at least 9 deg inside its limits from the table up to 0.16 m (the grasp
and the stop above it); nearer, the folded elbow comes within 7 deg. At 0.76 m the margin is over
45 deg up to 0.27 m (the carry over a bucket), and the reach ends at about 0.80 m. The nearest
radius also keeps the objects away from the arm's base."""
BUCKET_REACH_POINTS = ((0.0, -0.04), (0.0, 0.0), (0.0, 0.04))
"""Points in a bucket's frame, ``(x, y)``, that must be in :data:`REACH_RADIUS`: where a hand
over the opening lets an object go."""
HAND_HALF_SIZE = (0.110, 0.045)
"""Half the footprint of the Franka Hand about the grasp point, in metres: along the fingers'
closing axis and across it.

The hand's housing spans 0.106 m either way along the closing axis and 0.032 m across, from
0.03 m above the fingertips, so it is below a bucket's rim when the hand grasps an object on the
table. The margins cover a grasp turned 7 deg or shifted 6 mm along the object."""
HAND_BUCKET_GAP = 0.015
"""Smallest gap between the footprint of a hand grasping an object (:data:`HAND_HALF_SIZE`,
closing across the object's short side) and a bucket's footprint."""
FINGER_HALF_SIZE = (0.068, 0.012)
"""Half the footprint of the open fingers about the grasp point: along the closing axis and across."""
FINGER_OBJECT_GAP = 0.01
"""Smallest gap between the open fingers grasping one object and another object's enclosing circle."""
LAYOUT_DRAWS = 200
"""Draws per bucket or object before a layout is started over."""
LAYOUT_RESTARTS = 50
"""Layouts started over before :data:`FALLBACK_LAYOUT` is used (it never was in 100 000 seeds)."""
SLOT_NAMES = ("left", "centre", "right")
"""Names of the buckets from left to right as seen from behind the robot (decreasing y)."""
DROP_HEIGHT = 0.002
"""How far above the table an object is released at reset, in metres."""
SETTLE_SPEED = 0.02
"""Largest speed, in m/s, at which an object counts as resting."""

Pair = tuple[str, str]


@dataclass(frozen=True)
class Pose2D:
    """A pose on the table: the centre's ``x`` and ``y`` in metres and the ``yaw`` in radians."""

    x: float
    y: float
    yaw: float

    def as_dict(self) -> dict[str, float]:
        """Return the pose for JSON."""
        return {"x": self.x, "y": self.y, "yaw": self.yaw}


@dataclass(frozen=True)
class ObjectFootprint:
    """What the layout needs to know of an object lying flat."""

    radius: float
    """Horizontal radius at any heading, about the body origin."""
    half_height: float
    """Height of the body origin above the table."""
    closing_yaw: float
    """Heading in the body frame of the object's short horizontal side, across which a hand closes:
    0 for the body's x axis, a quarter turn for its y axis."""


@dataclass(frozen=True)
class Layout:
    """Where a reset puts the buckets and the objects."""

    buckets: dict[str, Pose2D]
    """Each bucket's pose by colour; the yaw includes :data:`BUCKET_YAW`."""
    objects: dict[str, Pose2D]
    """Each object's pose by name."""
    draws: int = 0
    """Candidate poses drawn to find this layout."""
    fallback: bool = False
    """Whether no layout was found and :data:`FALLBACK_LAYOUT` was used."""

    @property
    def colors_left_to_right(self) -> tuple[str, ...]:
        """The bucket colours by decreasing y: left to right as seen from behind the robot."""
        return tuple(sorted(self.buckets, key=lambda color: -self.buckets[color].y))


FALLBACK_LAYOUT = Layout(
    buckets={color: Pose2D(0.645, y, BUCKET_YAW) for color, y in zip(BUCKET_COLORS, (0.22, 0.0, -0.22), strict=True)},
    objects={name: Pose2D(0.40, y, 0.0) for name, y in zip(OBJECTS, (0.17, 0.0, -0.17), strict=True)},
    fallback=True,
)
"""A layout that passes every check, for a generator that finds none."""


def _rotation(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]])


def _rectangle(pose: Pose2D, half: tuple[float, float], yaw: float | None = None) -> np.ndarray:
    # The corners, counter-clockwise, of a rectangle of half sizes *half* (along its x and y) at *pose*.
    corners = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]]) * np.asarray(half)
    return np.array([pose.x, pose.y]) + corners @ _rotation(pose.yaw if yaw is None else yaw).T


def _segment_distances(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    # The distance from each of *points* (n, 2) to the boundary of *polygon* (m, 2).
    edges = np.roll(polygon, -1, axis=0) - polygon
    offset = points[:, None, :] - polygon[None, :, :]
    t = np.clip(np.einsum("nmi,mi->nm", offset, edges) / np.einsum("mi,mi->m", edges, edges), 0.0, 1.0)
    return np.linalg.norm(offset - t[..., None] * edges[None], axis=2).min(axis=1)


def _overlap(a: np.ndarray, b: np.ndarray) -> bool:
    # Whether two convex polygons overlap: the separating axis test.
    for polygon in (a, b):
        edges = np.roll(polygon, -1, axis=0) - polygon
        normals = np.stack([edges[:, 1], -edges[:, 0]], axis=1)
        pa, pb = a @ normals.T, b @ normals.T
        if np.any((pa.max(axis=0) < pb.min(axis=0)) | (pb.max(axis=0) < pa.min(axis=0))):
            return False
    return True


def polygon_gap(a: np.ndarray, b: np.ndarray) -> float:
    """Return the distance between two convex polygons (corners in order), or ``-1`` if they overlap.

    Returns:
        Metres.
    """
    if _overlap(a, b):
        return -1.0
    return float(min(_segment_distances(a, b).min(), _segment_distances(b, a).min()))


def circle_gap(centre: tuple[float, float], radius: float, polygon: np.ndarray) -> float:
    """Return the distance from a circle to a convex polygon (corners counter-clockwise), negative when they overlap.

    Returns:
        Metres.
    """
    point = np.asarray(centre, dtype=np.float64)
    edges = np.roll(polygon, -1, axis=0) - polygon
    inside = np.all(edges[:, 0] * (point[1] - polygon[:, 1]) - edges[:, 1] * (point[0] - polygon[:, 0]) >= 0.0)
    distance = float(_segment_distances(point[None], polygon)[0])
    return (-distance if inside else distance) - radius


def _far_apart(a: tuple[float, float], b: tuple[float, float], reach: float) -> bool:
    # Shapes within radii that add up to *reach* (with the gap) cannot come closer than the gap.
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 > reach**2


_BUCKET_RADIUS = float(np.hypot(BUCKET_SIZE[0], BUCKET_SIZE[1]) / 2.0)


@dataclass(frozen=True)
class TaskLayout:
    """Where one robot's layouts put the buckets and the objects (:func:`sample_layout`).

    The objects and buckets keep their size; only where they may stand changes per robot. The
    default is the Franka arms' layout, the module's constants; a smaller arm gets its areas and
    reach closer in (``scene_registry``'s ``profile_task_layouts``).
    """

    bucket_x: tuple[float, float] = BUCKET_X
    bucket_y: tuple[float, float] = BUCKET_Y
    object_x: tuple[float, float] = OBJECT_X
    object_y: tuple[float, float] = OBJECT_Y
    reach: tuple[float, float] = REACH_RADIUS
    """Distance from an arm's base axis at which it reaches every object centre."""
    bucket_reach: tuple[float, float] = REACH_RADIUS
    """Distance from an arm's base axis at which it reaches every point of :data:`BUCKET_REACH_POINTS`."""
    bases: tuple[tuple[float, float], ...] = ((0.0, 0.0),)
    """The arms' base axes on the table; an object or bucket point must be in reach of one of them."""
    hand_half_size: tuple[float, float] = HAND_HALF_SIZE
    """Half the hand's footprint about the grasp point (:data:`HAND_HALF_SIZE`)."""
    finger_half_size: tuple[float, float] = FINGER_HALF_SIZE
    face_bases: bool = False
    """Turn each bucket's open end toward its nearest base (plus the usual turn either way), not toward -x."""
    objects_reach_all: bool = False
    """Every object in reach of every base, not just one: with two arms, whichever arm reaches the
    named bucket also reaches the named object."""

    @property
    def hand_radius(self) -> float:
        """The radius of the circle around :attr:`hand_half_size`."""
        return float(np.hypot(*self.hand_half_size))

    @property
    def finger_radius(self) -> float:
        """The radius of the circle around :attr:`finger_half_size`."""
        return float(np.hypot(*self.finger_half_size))

    def nearest_base(self, x: float, y: float) -> tuple[float, float]:
        """Return the base nearest to ``(x, y)``.

        Returns:
            Its position.
        """
        return min(self.bases, key=lambda base: (x - base[0]) ** 2 + (y - base[1]) ** 2)


DEFAULT_LAYOUT = TaskLayout()
"""The Franka arms' layout: a base at the origin and the areas of the module's constants."""


def _current(layout: TaskLayout) -> TaskLayout:
    """Return *layout*, or for :data:`DEFAULT_LAYOUT` the module's constants as they are now.

    Returns:
        The layout to use.
    """
    if layout is not DEFAULT_LAYOUT:
        return layout
    return TaskLayout(
        bucket_x=BUCKET_X,
        bucket_y=BUCKET_Y,
        object_x=OBJECT_X,
        object_y=OBJECT_Y,
        reach=REACH_RADIUS,
        bucket_reach=REACH_RADIUS,
        hand_half_size=HAND_HALF_SIZE,
        finger_half_size=FINGER_HALF_SIZE,
    )


_HAND_RADIUS = DEFAULT_LAYOUT.hand_radius
_FINGER_RADIUS = DEFAULT_LAYOUT.finger_radius


def bucket_footprint(pose: Pose2D) -> np.ndarray:
    """Return a bucket's outer rectangle on the table at *pose*, corners counter-clockwise.

    Returns:
        ``(4, 2)`` corners in metres.
    """
    return _rectangle(pose, (BUCKET_SIZE[0] / 2.0, BUCKET_SIZE[1] / 2.0))


def _grasp_rectangle(pose: Pose2D, footprint: ObjectFootprint, half: tuple[float, float]) -> np.ndarray:
    # The rectangle *half* (along the closing axis, across) of a hand grasping an object at *pose*.
    return _rectangle(pose, half, pose.yaw + footprint.closing_yaw)


def _in_reach(
    x: float,
    y: float,
    radius: tuple[float, float] = REACH_RADIUS,
    bases: tuple[tuple[float, float], ...] = ((0.0, 0.0),),
) -> bool:
    for base_x, base_y in bases:
        dx, dy = x - base_x, y - base_y
        if radius[0] ** 2 <= dx * dx + dy * dy <= radius[1] ** 2:
            return True
    return False


def bucket_problems(
    pose: Pose2D, others: Iterable[Pose2D] = (), *, first: bool = False, layout: TaskLayout = DEFAULT_LAYOUT
) -> list[str]:
    """Return what keeps a bucket at *pose* out of a layout with the buckets *others*.

    Args:
        pose: The bucket's pose.
        others: The buckets placed before it.
        first: Stop at the first problem.
        layout: The robot's layout, for the reach.

    Returns:
        The problems; empty when the pose is fine.
    """
    problems: list[str] = []
    corners = bucket_footprint(pose)
    (x_lo, x_hi), (y_lo, y_hi) = TABLE_X, TABLE_Y
    if (
        corners[:, 0].min() < x_lo + TABLE_MARGIN
        or corners[:, 0].max() > x_hi - TABLE_MARGIN
        or corners[:, 1].min() < y_lo + TABLE_MARGIN
        or corners[:, 1].max() > y_hi - TABLE_MARGIN
    ):
        problems.append("table edge")
        if first:
            return problems
    layout = _current(layout)
    turn = _rotation(pose.yaw)
    points = [np.array([pose.x, pose.y]) + turn @ point for point in BUCKET_REACH_POINTS]
    if not all(_in_reach(float(x), float(y), layout.bucket_reach, layout.bases) for x, y in points):
        problems.append("reach")
        if first:
            return problems
    for other in others:
        if _far_apart((pose.x, pose.y), (other.x, other.y), 2.0 * _BUCKET_RADIUS + BUCKET_GAP):
            continue
        if polygon_gap(corners, bucket_footprint(other)) < BUCKET_GAP:
            problems.append("bucket gap")
            break
    return problems


def object_problems(
    name: str,
    pose: Pose2D,
    footprints: Mapping[str, ObjectFootprint],
    buckets: Iterable[Pose2D],
    others: Mapping[str, Pose2D] | None = None,
    *,
    first: bool = False,
    layout: TaskLayout = DEFAULT_LAYOUT,
) -> list[str]:
    """Return what keeps object *name* at *pose* out of a layout with *buckets* and the objects *others*.

    Args:
        name: The object.
        pose: Its pose.
        footprints: Every object's footprint (:func:`object_footprint`).
        buckets: The buckets' poses.
        others: The objects placed before it.
        first: Stop at the first problem.
        layout: The robot's layout, for the reach and the hand's footprint.

    Returns:
        The problems, sorted; empty when the pose is fine.
    """
    problems: set[str] = set()
    default = layout is DEFAULT_LAYOUT
    layout = _current(layout)
    shape = footprints[name]
    centre = (pose.x, pose.y)
    in_reach = (
        all(_in_reach(pose.x, pose.y, layout.reach, (base,)) for base in layout.bases)
        if layout.objects_reach_all
        else _in_reach(pose.x, pose.y, layout.reach, layout.bases)
    )
    if not in_reach:
        problems.add("reach")
        if first:
            return sorted(problems)
    hand_radius, finger_radius = (
        (_HAND_RADIUS, _FINGER_RADIUS) if default else (layout.hand_radius, layout.finger_radius)
    )
    hand = _grasp_rectangle(pose, shape, layout.hand_half_size)
    for bucket in buckets:
        if _far_apart(
            centre,
            (bucket.x, bucket.y),
            _BUCKET_RADIUS + max(shape.radius + OBJECT_BUCKET_GAP, hand_radius + HAND_BUCKET_GAP),
        ):
            continue
        corners = bucket_footprint(bucket)
        if circle_gap(centre, shape.radius, corners) < OBJECT_BUCKET_GAP:
            problems.add("object near bucket")
        if polygon_gap(hand, corners) < HAND_BUCKET_GAP:
            problems.add("hand near bucket")
        if first and problems:
            return sorted(problems)
    fingers = _grasp_rectangle(pose, shape, layout.finger_half_size)
    for other, other_pose in (others or {}).items():
        other_shape = footprints[other]
        other_centre = (other_pose.x, other_pose.y)
        if not _far_apart(centre, other_centre, shape.radius + other_shape.radius + OBJECT_GAP):
            problems.add("object gap")
        elif not _far_apart(
            centre, other_centre, finger_radius + max(shape.radius, other_shape.radius) + FINGER_OBJECT_GAP
        ) and (
            circle_gap(other_centre, other_shape.radius, fingers) < FINGER_OBJECT_GAP
            or circle_gap(centre, shape.radius, _grasp_rectangle(other_pose, other_shape, layout.finger_half_size))
            < FINGER_OBJECT_GAP
        ):
            problems.add("fingers near object")
        if first and problems:
            break
    return sorted(problems)


def sample_layout(
    rng: np.random.Generator, footprints: Mapping[str, ObjectFootprint], layout: TaskLayout = DEFAULT_LAYOUT
) -> Layout:
    """Draw a layout of the buckets and the objects that passes every check.

    The colours, then the objects, are placed in a random order, each drawn uniformly in its area
    (and heading range) until it passes :func:`bucket_problems` or :func:`object_problems` with
    those placed before it; a bucket or object that finds no place in :data:`LAYOUT_DRAWS` draws
    starts the layout over. The same generator state gives the same layout.

    Args:
        rng: The generator.
        footprints: Each object's :class:`ObjectFootprint` (:func:`object_footprint`).
        layout: Where the robot's buckets and objects may stand.

    Returns:
        The layout; :data:`FALLBACK_LAYOUT` when :data:`LAYOUT_RESTARTS` layouts all failed.
    """
    draws = 0
    default = layout is DEFAULT_LAYOUT
    layout = _current(layout)
    for _ in range(LAYOUT_RESTARTS):
        buckets: dict[str, Pose2D] = {}
        for index in rng.permutation(len(BUCKET_COLORS)):
            for _ in range(LAYOUT_DRAWS):
                draws += 1
                x, y = float(rng.uniform(*layout.bucket_x)), float(rng.uniform(*layout.bucket_y))
                yaw = BUCKET_YAW
                if layout.face_bases:
                    base_x, base_y = layout.nearest_base(x, y)
                    yaw += float(np.arctan2(y - base_y, x - base_x))
                pose = Pose2D(x, y, yaw + float(rng.uniform(-BUCKET_YAW_RANGE, BUCKET_YAW_RANGE)))
                if not bucket_problems(
                    pose, buckets.values(), first=True, layout=DEFAULT_LAYOUT if default else layout
                ):
                    buckets[BUCKET_COLORS[int(index)]] = pose
                    break
            else:
                break
        if len(buckets) < len(BUCKET_COLORS):
            continue
        objects: dict[str, Pose2D] = {}
        for index in rng.permutation(len(OBJECTS)):
            name = OBJECTS[int(index)]
            for _ in range(LAYOUT_DRAWS):
                draws += 1
                pose = Pose2D(
                    float(rng.uniform(*layout.object_x)),
                    float(rng.uniform(*layout.object_y)),
                    float(rng.uniform(0.0, 2.0 * np.pi)),
                )
                area = DEFAULT_LAYOUT if default else layout
                if not object_problems(name, pose, footprints, buckets.values(), objects, first=True, layout=area):
                    objects[name] = pose
                    break
            else:
                break
        if len(objects) == len(OBJECTS):
            return Layout(
                buckets={color: buckets[color] for color in BUCKET_COLORS},
                objects={name: objects[name] for name in OBJECTS},
                draws=draws,
            )
    logger.warning("Bucket sort: no layout found in {} draws; using the fallback layout", draws)
    return Layout(buckets=FALLBACK_LAYOUT.buckets, objects=FALLBACK_LAYOUT.objects, draws=draws, fallback=True)


# ----------------------------------------------------------------------
# Instructions
# ----------------------------------------------------------------------

VARY_INSTRUCTION_TEXT = True
"""Whether instructions are worded in many ways (:data:`PHRASES`) by default; ``False`` always
gives the canonical sentence. :class:`BucketSortTask` can override it."""

Weighted = tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class InstructionPhrases:
    """The words an instruction is made of; :data:`PHRASES` is the scene's set.

    A worded instruction is, with probability :attr:`canonical_fraction`, the canonical sentence
    (:attr:`Instruction.canonical_text`, "put the CPU in the red bucket"); otherwise a template
    filled with a name of the object, a container word and the bucket's colour, in sentence case,
    with an optional "please" and a final full stop. Every choice is weighted (the weights of a
    list need not sum to one).

    To extend it, add entries to the lists below. A name must name only its object (no word of
    another object's names, and no colour word), and a template must hold ``{object}`` once and
    either ``{bucket}`` (the colour and a container word: "red bin") or ``{containers}`` before
    ``{color} one`` ("of the three bins, ... the red one"); ``tests/test_bucket_sort.py`` checks
    that every sentence names the right object and colour and nothing else.
    """

    canonical_fraction: float
    """Probability of the canonical sentence, word for word."""
    templates: tuple[tuple[str, float, Literal["imperative", "question", "plain"]], ...]
    """``(template, weight, kind)``: an imperative can take "please" before or after it, a
    question takes it after "can you"/"could you" and ends in a question mark, a plain sentence takes none."""
    object_names: dict[str, Weighted]
    """Names of each object, used after "the"."""
    containers: Weighted
    """Words for a bucket."""
    plurals: dict[str, str]
    """The plural of each container word."""
    please: float
    """Probability of "please" in a template that can take it."""
    please_first: float
    """Of those, probability that "please" opens the sentence rather than ending it."""
    full_stop: float
    """Probability of a full stop after a sentence that is not a question."""


PHRASES = InstructionPhrases(
    canonical_fraction=0.3,
    templates=(
        ("put the {object} in the {bucket}", 2.0, "imperative"),
        ("put the {object} into the {bucket}", 1.0, "imperative"),
        ("place the {object} in the {bucket}", 1.0, "imperative"),
        ("place the {object} into the {bucket}", 1.0, "imperative"),
        ("drop the {object} in the {bucket}", 1.0, "imperative"),
        ("move the {object} to the {bucket}", 1.0, "imperative"),
        ("sort the {object} into the {bucket}", 1.0, "imperative"),
        ("take the {object} and put it in the {bucket}", 1.0, "imperative"),
        ("pick up the {object} and drop it in the {bucket}", 1.0, "imperative"),
        ("grab the {object} and place it in the {bucket}", 1.0, "imperative"),
        ("can you put the {object} in the {bucket}", 1.0, "question"),
        ("could you place the {object} into the {bucket}", 1.0, "question"),
        ("I want the {object} in the {bucket}", 1.0, "plain"),
        ("the {object} goes in the {bucket}", 1.0, "plain"),
        ("the {object} belongs in the {bucket}", 1.0, "plain"),
        ("of the three {containers}, put the {object} in the {color} one", 1.0, "plain"),
    ),
    object_names={
        "battery": (
            ("battery", 3.0),
            ("9 V battery", 2.0),
            ("9V battery", 1.0),
            ("9 volt battery", 1.0),
            ("nine volt battery", 1.0),
        ),
        "cpu": (("CPU", 3.0), ("processor", 1.5), ("chip", 1.5), ("microchip", 1.0), ("computer chip", 1.0)),
        "resistor": (("resistor", 1.0),),
    },
    containers=(("bucket", 3.0), ("bin", 2.0), ("box", 1.5), ("container", 1.5), ("tub", 1.0)),
    plurals={"bucket": "buckets", "bin": "bins", "box": "boxes", "container": "containers", "tub": "tubs"},
    please=0.25,
    please_first=0.6,
    full_stop=0.5,
)
"""The scene's instruction wording (:class:`InstructionPhrases`).

The resistor has no other name: it is the longest of the three objects, so "small resistor" would
point at the wrong one, and "the resistor" is what the templates already say."""


def _choose(rng: np.random.Generator, options: Weighted) -> str:
    return options[int(rng.choice(len(options), p=_normalized([weight for _, weight in options])))][0]


def canonical_text(obj: str, color: str) -> str:
    """Return the canonical instruction, e.g. ``put the CPU in the red bucket``.

    Returns:
        The sentence.
    """
    return f"put the {OBJECT_LABELS[obj]} in the {color} bucket"


def word_instruction(rng: np.random.Generator, obj: str, color: str, phrases: InstructionPhrases = PHRASES) -> str:
    """Word the instruction "put *obj* in the *color* bucket" at random (:class:`InstructionPhrases`).

    Args:
        rng: The generator.
        obj: One of :data:`OBJECTS`.
        color: One of :data:`BUCKET_COLORS`.
        phrases: The words to use.

    Returns:
        The sentence.
    """
    if rng.random() < phrases.canonical_fraction:
        return canonical_text(obj, color)
    index = int(rng.choice(len(phrases.templates), p=_normalized([weight for _, weight, _ in phrases.templates])))
    template, _, kind = phrases.templates[index]
    name = _choose(rng, phrases.object_names[obj])
    container = _choose(rng, phrases.containers)
    text = template.format(
        object=name, bucket=f"{color} {container}", containers=phrases.plurals[container], color=color
    )
    if kind != "plain" and rng.random() < phrases.please:
        if kind == "question":
            text = text.replace(" you ", " you please ", 1)
        elif rng.random() < phrases.please_first:
            text = f"please {text}"
        else:
            text = f"{text}, please"
    if kind == "question":
        text += "?"
    elif rng.random() < phrases.full_stop:
        text += "."
    return text[0].upper() + text[1:]


def _normalized(weights: list[float]) -> np.ndarray:
    values = np.asarray(weights, dtype=np.float64)
    return values / values.sum()


@dataclass(frozen=True)
class Instruction:
    """One episode's instruction: put ``object`` in the ``bucket`` bucket.

    Two instructions are equal when they name the same object and bucket, however they are worded.
    """

    object: str
    """One of :data:`OBJECTS`."""
    bucket: str
    """One of :data:`BUCKET_COLORS`."""
    wording: str | None = field(default=None, compare=False)
    """The sentence this episode uses; ``None`` for the canonical one."""

    @property
    def canonical_text(self) -> str:
        """The instruction in its fixed form, e.g. ``put the CPU in the red bucket``."""
        return canonical_text(self.object, self.bucket)

    @property
    def text(self) -> str:
        """The instruction as given to the robot: :attr:`wording`, else :attr:`canonical_text`."""
        return self.wording if self.wording is not None else self.canonical_text

    def as_dict(self) -> dict[str, str]:
        """Return the instruction for JSON: ``object``, ``bucket``, ``text`` and ``canonical_text``."""
        return {"object": self.object, "bucket": self.bucket, "text": self.text, "canonical_text": self.canonical_text}


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


def sample_instruction(
    rng: np.random.Generator, allowed_pairs: Iterable[Pair] | None = None, *, vary_text: bool | None = None
) -> Instruction:
    """Draw an instruction uniformly from the allowed (object, bucket colour) pairs, then word it.

    Args:
        rng: The generator.
        allowed_pairs: The pairs to draw from (:func:`check_pairs`); ``None`` draws from all nine.
        vary_text: Word it at random (:func:`word_instruction`) or use the canonical sentence;
            ``None`` follows :data:`VARY_INSTRUCTION_TEXT`.

    Returns:
        The instruction.
    """
    pairs = check_pairs(allowed_pairs)
    obj, color = pairs[int(rng.integers(len(pairs)))]
    vary = VARY_INSTRUCTION_TEXT if vary_text is None else vary_text
    return Instruction(object=obj, bucket=color, wording=word_instruction(rng, obj, color) if vary else None)


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
    buckets: dict[str, dict[str, list[float]]] = field(default_factory=dict)
    """Each bucket's pose in the world frame: ``position`` (metres, the centre of its footprint)
    and ``wxyz``."""

    def as_dict(self) -> dict[str, object]:
        """Return the state for JSON."""
        return {
            "in_bucket": dict(self.in_bucket),
            "resting": dict(self.resting),
            "success": self.success,
            "buckets": {color: dict(pose) for color, pose in self.buckets.items()},
        }


class BucketSortTask:
    """The scene's reset and success test, with the current layout and instruction.

    Call it as the scene's reset callback, ``task(model, data, rng)``: it puts the arm at home,
    lays the buckets and objects out and samples the instruction. The allowed pairs and the
    wording can be changed at any time and apply from the next reset.
    """

    def __init__(
        self,
        *,
        allowed_pairs: Iterable[Pair] | None = None,
        place_home: Callable[[object, object], None] | None = None,
        robot_roots: tuple[int, ...] = (),
        vary_text: bool | None = None,
        layout: TaskLayout = DEFAULT_LAYOUT,
    ) -> None:
        """Create the task; nothing is laid out until the first reset.

        Args:
            allowed_pairs: The (object, bucket colour) pairs instructions are drawn from
                (:func:`check_pairs`); ``None`` allows all nine.
            place_home: Puts the arm at its home pose at rest; ``None`` leaves the arm where it is.
            robot_roots: Root body of every robot: objects touching them are held, not resting.
            vary_text: Word instructions at random, or always use the canonical sentence;
                ``None`` follows :data:`VARY_INSTRUCTION_TEXT` at each reset.
            layout: Where the robot's buckets and objects may stand.
        """
        self._allowed = check_pairs(allowed_pairs)
        self._place_home = place_home
        self._robot_roots = tuple(robot_roots)
        self._vary_text = vary_text
        self.area = layout
        """Where the robot's buckets and objects may stand (:class:`TaskLayout`)."""
        self._footprints: tuple[object, dict[str, ObjectFootprint]] | None = None
        self.instruction: Instruction | None = None
        """The current instruction; ``None`` before the first reset."""
        self.layout: Layout | None = None
        """Where the last reset put the buckets and objects; ``None`` before the first reset."""
        self.slot_colors: tuple[str, ...] = BUCKET_COLORS
        """The bucket colours left to right (by decreasing y) as the last reset put them."""
        self.reset_count = 0
        """How many resets have run: a new value means a new layout and instruction."""

    @property
    def allowed_pairs(self) -> tuple[Pair, ...]:
        """The (object, bucket colour) pairs the next instructions are drawn from."""
        return self._allowed

    def set_allowed_pairs(self, pairs: Iterable[Pair] | None) -> None:
        """Draw later instructions from *pairs* (:func:`check_pairs`); ``None`` allows all nine.

        The current instruction stays until the next reset.
        """
        self._allowed = check_pairs(pairs)

    @property
    def vary_text(self) -> bool:
        """Whether the next resets word the instruction at random."""
        return VARY_INSTRUCTION_TEXT if self._vary_text is None else self._vary_text

    def set_vary_text(self, vary: bool | None) -> None:  # noqa: FBT001
        """Word later instructions at random (``True``) or canonically (``False``).

        ``None`` follows :data:`VARY_INSTRUCTION_TEXT`.
        """
        self._vary_text = vary

    @property
    def canonical_text(self) -> str | None:
        """The current instruction in its fixed form (:attr:`Instruction.canonical_text`); ``None`` before a reset."""
        return self.instruction.canonical_text if self.instruction is not None else None

    def footprints(self, model: object) -> dict[str, ObjectFootprint]:
        """Return each object's :class:`ObjectFootprint` on *model*, computed once per model."""
        if self._footprints is None or self._footprints[0] is not model:
            self._footprints = (model, {name: object_footprint(model, name) for name in OBJECTS})
        return self._footprints[1]

    def __call__(self, model: object, data: object, rng: np.random.Generator) -> None:
        """Reset: the arm at home, a new layout (:func:`sample_layout`) and a new instruction."""
        import mujoco  # noqa: PLC0415

        if self._place_home is not None:
            self._place_home(model, data)
        footprints = self.footprints(model)
        layout = sample_layout(rng, footprints, self.area)
        for color, pose in layout.buckets.items():
            _set_free_pose(model, data, f"bucket_{color}:joint", (pose.x, pose.y, 0.0), pose.yaw)
        for name, pose in layout.objects.items():
            _set_free_pose(
                model, data, f"{name}:joint", (pose.x, pose.y, footprints[name].half_height + DROP_HEIGHT), pose.yaw
            )
        self.layout = layout
        self.slot_colors = layout.colors_left_to_right
        self.instruction = sample_instruction(rng, self._allowed, vary_text=self.vary_text)
        self.reset_count += 1
        mujoco.mj_forward(model, data)
        logger.info(
            "Bucket sort: {!r} ({}); buckets left to right {}",
            self.instruction.text,
            self.instruction.canonical_text,
            ", ".join(self.slot_colors),
        )

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
        buckets = {
            color: {
                "position": [float(v) for v in data.xpos[body]],
                "wxyz": [float(v) for v in data.xquat[body]],
            }
            for color in BUCKET_COLORS
            for body in (_body(model, f"bucket_{color}"),)
        }
        return TaskState(in_bucket=in_bucket, resting=resting, success=success, buckets=buckets)

    def describe(self) -> dict[str, object]:
        """Return the instruction, the buckets left to right as laid out, and the allowed pairs, for JSON.

        ``slots`` lists the buckets by decreasing y at the last reset (``left``, ``centre``,
        ``right`` as seen from behind the robot) with that reset's centre and yaw; ``buckets`` in
        the episode status has their live poses.
        """
        layout = self.layout
        return {
            "instruction": self.instruction.as_dict() if self.instruction is not None else None,
            "slots": [
                {"slot": slot, "color": color, **(layout.buckets[color].as_dict() if layout is not None else {})}
                for slot, color in zip(SLOT_NAMES, self.slot_colors, strict=True)
            ],
            "allowed_pairs": [list(pair) for pair in self._allowed],
            "vary_text": self.vary_text,
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


def _set_free_pose(
    model: object, data: object, joint_name: str, position: tuple[float, float, float], yaw: float
) -> None:
    """Put a free joint at *position*, turned *yaw* about the vertical, at rest."""
    joint = _joint(model, joint_name)
    adr, dof = int(model.jnt_qposadr[joint]), int(model.jnt_dofadr[joint])
    data.qpos[adr : adr + 3] = position
    data.qpos[adr + 3 : adr + 7] = _yaw_quat(yaw)
    data.qvel[dof : dof + 6] = 0.0


def bucket_containing(model: object, data: object, point: np.ndarray) -> str | None:
    """Return the colour of the bucket whose interior (:data:`BUCKET_INTERIOR`) holds *point*.

    The interior moves with each bucket's current pose (``data.xpos`` and ``data.xmat``, as of the
    last ``mj_forward`` or step), so it follows a bucket that was pushed or dragged.

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


def object_footprint(model: object, name: str) -> ObjectFootprint:
    """Return what the layout needs of an object (:class:`ObjectFootprint`), from its collision geoms.

    Returns:
        The footprint.
    """
    import mujoco  # noqa: PLC0415

    radius, half_height = object_extent(model, name)
    body = _body(model, name)
    rotation = np.zeros(9)
    corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=np.float64)
    half = np.zeros(2)
    for geom in np.flatnonzero(np.asarray(model.geom_bodyid) == body):
        if int(model.geom_contype[geom]) == 0 and int(model.geom_conaffinity[geom]) == 0:
            continue
        mujoco.mju_quat2Mat(rotation, model.geom_quat[geom])
        points = (
            model.geom_pos[geom]
            + (model.geom_aabb[geom, :3] + corners * model.geom_aabb[geom, 3:]) @ rotation.reshape(3, 3).T
        )
        half = np.maximum(half, np.abs(points[:, :2]).max(axis=0))
    closing_yaw = 0.0 if half[0] <= half[1] else float(np.pi / 2)
    return ObjectFootprint(radius=radius, half_height=half_height, closing_yaw=closing_yaw)


Phase = Literal["idle", "success_hold"]


class BucketSortEpisode:
    """Report the task in ``/health`` and the viewer, and start a new episode after a success.

    It has the controls of the other episode helpers (``active``, the success dwell). Once the
    instruction has held for the dwell, it resets the scene as ``POST /reset`` does.
    """

    def __init__(
        self,
        task: BucketSortTask,
        reset_scene: Callable[[], None],
        *,
        success_dwell_s: float = 5.0,
        active: bool = True,
    ) -> None:
        """Watch *task*; *reset_scene* is the owner's scene reset, which the automatic resets share.

        Going through the owner keeps a fixed seed repeating the layout and drops viewer drags,
        as a manual reset does.
        """
        self.task = task
        self._reset_scene = reset_scene
        self._dwell_s = float(success_dwell_s)
        self._active = active
        self._phase: Phase = "idle"
        self._success_since: float | None = None
        self._resets_suspended = False
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
            "buckets": dict(state.buckets) if state is not None else {},
        }

    def suspend_resets(self, suspended: bool) -> None:  # noqa: FBT001
        """Keep evaluating the task but never reset the scene, while something else owns the episodes.

        Automatic Studio recording ends and resets each episode itself; :attr:`active` and the
        dwell stay as the user set them for afterwards.
        """
        if suspended and not self._resets_suspended:
            self.notify_manual_reset()  # drop a success countdown in progress
        self._resets_suspended = suspended

    def update(self, model: object, data: object) -> None:
        """Evaluate the task for one tick, and reset the scene once a success has held for the dwell."""
        state = self.task.evaluate(model, data)
        self._state = state
        if not self._active or self._resets_suspended:
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
        self._reset_scene()
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
    "BUCKET_GAP",
    "BUCKET_INTERIOR",
    "BUCKET_JOINTS",
    "BUCKET_SIZE",
    "BUCKET_X",
    "BUCKET_Y",
    "BUCKET_YAW",
    "BUCKET_YAW_RANGE",
    "DEFAULT_LAYOUT",
    "OBJECTS",
    "OBJECT_LABELS",
    "OBJECT_X",
    "OBJECT_Y",
    "PHRASES",
    "REACH_RADIUS",
    "VARY_INSTRUCTION_TEXT",
    "BucketSortEpisode",
    "BucketSortTask",
    "Instruction",
    "InstructionPhrases",
    "Layout",
    "ObjectFootprint",
    "Pose2D",
    "TaskLayout",
    "TaskState",
    "bucket_containing",
    "bucket_footprint",
    "bucket_problems",
    "canonical_text",
    "check_pairs",
    "object_extent",
    "object_footprint",
    "object_problems",
    "sample_instruction",
    "sample_layout",
    "word_instruction",
]
