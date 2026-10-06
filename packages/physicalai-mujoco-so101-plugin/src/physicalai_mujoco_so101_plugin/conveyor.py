# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Belt drive, item spawning and scoring for the ``conveyor_sort`` scene.

The belt is a slide-jointed slab driven by a velocity actuator. Items ride it by
friction, so a gripper pulling an item off the belt meets realistic resistance.
To run forever, the slab is wrapped back by exactly one texture tile
(`BELT_TILE`) whenever it has travelled that far, which is visually seamless.

MuJoCo cannot add bodies at run time, so items come from a fixed pool of free
bodies (every shape x color, plain and cracked). Unused items wait in a parking
lot outside the cell and are teleported to the belt entry, inside the hood.

Scoring follows one fixed rule: cracked items and colors without a bin go to
the reject bin, everything else goes to the bin of its color. An item that
comes to rest in a bin is scored and parked again; an item that rides off the
end of the belt, or ends up anywhere else, is a miss.
"""

# MuJoCo model/data attributes are supplied by the C extension at runtime.
# pyrefly: ignore-errors [missing-attribute]

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from collections.abc import Iterator

BELT_JOINT = "conveyor_belt"
BELT_ACTUATOR = "conveyor_belt"
BELT_TILE = 0.02
"""Belt texture period in metres; must match the belt material's texrepeat (50/m)."""

ITEM_PREFIX = "item_"
ITEM_SHAPES = ("cube", "cylinder", "hex")
ITEM_COLORS = ("red", "blue", "green", "purple")
BIN_COLORS = ("red", "blue", "green")
REJECT_BIN = "reject"

LIGHTS = ("green", "amber", "red")
"""Stack-light lamps; each lit lamp is a mocap body ``light_<name>_on`` in the scene."""
LIGHT_HIDDEN_POS = (0.344, -0.405, 0.05)
"""Where an unlit lamp waits: inside the belt's drive motor can (a closed 4 cm x 5 cm cylinder).

Not under the floor: the browser viewer draws the floor as a see-through grid.
"""

DEFAULT_BELT_SPEED = 0.03
MAX_BELT_SPEED = 0.10

Outcome = Literal["correct", "wrong", "missed"]


@dataclass
class ConveyorConfig:
    """Tunable parameters for belt motion, spawning and scoring."""

    belt_speed: float = DEFAULT_BELT_SPEED
    """Belt surface speed in m/s (items move toward -y)."""
    items_per_episode: int = 10
    spawn_spacing: float = 0.15
    """Belt travel between consecutive items, in metres (spacing is fixed in space, not time)."""
    spawn_spacing_jitter: float = 0.03
    cracked_probability: float = 0.3
    belt_x: float = 0.25
    belt_top: float = 0.06
    spawn_y: float = 0.37
    hood_exit_y: float = 0.30
    """Items become visible when they ride past this y (the entry hood's mouth)."""
    warn_s: float = 2.0
    """The amber light comes on this long before the next item leaves the hood."""
    lateral_jitter: float = 0.012
    belt_end_y: float = -0.40
    belt_half_width: float = 0.05
    """Items whose centre is within this x-distance of the belt centre line count as on the belt."""
    settle_speed: float = 0.02
    """Items slower than this (m/s) count as resting."""
    settle_s: float = 0.5
    """Seconds an item must rest before it is scored and parked."""
    rest_max_z: float = 0.045
    """Items resting higher than this (held, or on the belt) are not scored."""
    stuck_s: float = 10.0
    """Items off the belt and out of the gripper this long are scored wherever they are and however they
    move (balanced on a bin rim, wobbling in a bin), so an episode always ends."""
    bin_half: float = 0.046
    """Half the inner width of a bin, for the in-bin test."""


@dataclass
class ConveyorItem:
    """One pool item: its MuJoCo ids, what it looks like, and where the rule sends it."""

    name: str
    shape: str
    color: str
    cracked: bool
    body_id: int
    qpos_adr: int
    dof_adr: int
    half_height: float
    park_xyz: tuple[float, float, float]

    @property
    def target_bin(self) -> str:
        """Bin this item belongs in under the fixed rule."""
        if self.cracked or self.color not in BIN_COLORS:
            return REJECT_BIN
        return self.color


@dataclass
class _Score:
    correct: int = 0
    wrong: int = 0
    missed: int = 0

    def as_dict(self) -> dict[str, int]:
        return {"correct": self.correct, "wrong": self.wrong, "missed": self.missed}


@dataclass
class _Episode:
    spawned: int = 0
    score: _Score = field(default_factory=_Score)
    next_spawn_at: float = 0.0
    """Belt travel (m) at which the next item enters."""


class ConveyorSort:
    """Drive the belt, feed items and score sorting for one episode after another."""

    def __init__(
        self,
        model: object,
        items: list[ConveyorItem],
        bins: dict[str, int],
        belt_qpos_adr: int,
        belt_actuator: int,
        config: ConveyorConfig,
        rng: np.random.Generator,
    ) -> None:
        """Use `maybe_create` instead; it resolves names to MuJoCo ids."""
        self._model = model
        self._items = items
        self._bins = bins
        self._belt_qpos_adr = belt_qpos_adr
        self._belt_actuator = belt_actuator
        self._config = config
        self._rng = rng
        self._active = True
        self._feed_held = False
        self._travel = 0.0
        self._last_belt_q: float | None = None
        self._on_belt: dict[str, ConveyorItem] = {}
        self._rest_since: dict[str, float] = {}
        self._off_since: dict[str, float] = {}
        # Arm bodies (everything under the robot's root body) for the in-gripper test.
        geom_body = np.asarray(model.geom_bodyid, dtype=np.int64)
        root = np.asarray(model.body_rootid)
        base = _robot_root(model)
        self._geom_body = geom_body
        self._robot_body = (root == base) if base >= 0 else np.zeros(int(model.nbody), dtype=bool)
        self._episode = _Episode()
        self._episode_count = 0
        self._last_episode: dict[str, int] | None = None
        self._next_item_s: float | None = None
        # Lit lamp mocap ids and their "on" positions (the model default), for the stack light.
        self._lights: dict[str, tuple[int, np.ndarray]] = {}
        for light in LIGHTS:
            body = _body_id(model, f"light_{light}_on")
            if body >= 0 and int(model.body_mocapid[body]) >= 0:
                self._lights[light] = (int(model.body_mocapid[body]), np.array(model.body_pos[body], dtype=np.float64))

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def maybe_create(
        cls,
        model: object,
        *,
        rng: np.random.Generator,
        belt_speed: float = DEFAULT_BELT_SPEED,
        active: bool = True,
    ) -> ConveyorSort | None:
        """Return a controller when the model has a conveyor belt, else ``None``."""
        import mujoco  # noqa: PLC0415

        belt_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, BELT_JOINT)
        belt_actuator = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, BELT_ACTUATOR)
        if belt_joint < 0 or belt_actuator < 0:
            return None

        items = list(_discover_items(model))
        bins: dict[str, int] = {}
        for name in (*BIN_COLORS, REJECT_BIN):
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"bin_{name}")
            if body_id >= 0:
                bins[name] = int(body_id)
        if not items or REJECT_BIN not in bins:
            logger.warning("Conveyor belt found but no item pool or reject bin; conveyor disabled")
            return None

        config = ConveyorConfig(belt_speed=_clamp_speed(belt_speed))
        helper = cls(
            model,
            items,
            bins,
            belt_qpos_adr=int(model.jnt_qposadr[belt_joint]),
            belt_actuator=int(belt_actuator),
            config=config,
            rng=rng,
        )
        helper._active = active
        helper._episode.next_spawn_at = 0.0
        logger.info(
            "Conveyor enabled ({} pool items, bins {}, belt {:.1f} cm/s)",
            len(items),
            sorted(bins),
            100 * config.belt_speed,
        )
        return helper

    # ------------------------------------------------------------------
    # Controls (same surface as EpisodeAutoReset, plus belt speed)
    # ------------------------------------------------------------------

    @property
    def active(self) -> bool:
        """Whether the belt runs and items are fed."""
        return self._active

    def set_active(self, active: bool) -> None:  # noqa: FBT001
        """Pause or resume the belt and item feed; scoring continues while paused."""
        self._active = active

    @property
    def feed_held(self) -> bool:
        """Whether an automation holds the belt, apart from the user's pause (see `set_feed_hold`)."""
        return self._feed_held

    def set_feed_hold(self, held: bool) -> None:  # noqa: FBT001
        """Hold the belt and feed between recorded episodes without touching the user's pause."""
        self._feed_held = held

    @property
    def running(self) -> bool:
        """Whether the belt moves and feeds items: not paused by the user and not held."""
        return self._active and not self._feed_held

    @property
    def dwell_s(self) -> float:
        """Seconds an item must rest before it is scored."""
        return self._config.settle_s

    def set_dwell(self, dwell_s: float) -> None:  # noqa: PLR6301 - part of the episode-helper interface
        """Ignore the pick-place success dwell; the conveyor's settle time is fixed.

        The robot applies its remembered auto-reset dwell to whichever episode
        helper the scene has, and 5 s of settling would stall the belt feed.
        """
        _ = dwell_s

    @property
    def items_in_play(self) -> tuple[ConveyorItem, ...]:
        """Items fed onto the belt and not yet scored (on the belt, in the gripper, or falling)."""
        return tuple(self._on_belt.values())

    def is_in_play(self, item: ConveyorItem) -> bool:
        """Whether `item` has been fed and not yet scored.

        Returns:
            ``True`` while the item is on the belt, in the gripper, or falling.
        """
        return item.name in self._on_belt

    @property
    def bin_body_ids(self) -> dict[str, int]:
        """MuJoCo body id of each bin, keyed by bin name (``red``, ..., ``reject``)."""
        return dict(self._bins)

    @property
    def belt_speed(self) -> float:
        """Belt surface speed in m/s."""
        return self._config.belt_speed

    def set_belt_speed(self, speed: float) -> None:
        """Change the belt speed (m/s), clamped to ``[0, MAX_BELT_SPEED]``."""
        self._config.belt_speed = _clamp_speed(speed)
        logger.info("Conveyor belt speed set to {:.1f} cm/s", 100 * self._config.belt_speed)

    def notify_manual_reset(self) -> None:
        """Start a fresh episode after an explicit reset (the scene reset parks the items)."""
        self._on_belt.clear()
        self._rest_since.clear()
        self._off_since.clear()
        self._episode = _Episode(next_spawn_at=self._travel)
        self._last_belt_q = None

    def status(self) -> dict[str, object]:
        """Return a JSON-friendly snapshot for ``/health`` and the viewer."""
        return {
            "enabled": True,
            "kind": "conveyor",
            "active": self._active,
            "feed_held": self._feed_held,
            "phase": "held" if self._active and self._feed_held else ("running" if self._active else "paused"),
            "episode_count": self._episode_count,
            "belt_speed": self._config.belt_speed,
            "items_per_episode": self._config.items_per_episode,
            "spawned": self._episode.spawned,
            "on_belt": len(self._on_belt),
            "score": self._episode.score.as_dict(),
            "last_episode": self._last_episode,
            "next_item_s": self._next_item_s,
            "lights": self._light_states(),
            "rule": "cracked or purple -> reject; red/blue/green -> matching bin",
        }

    # ------------------------------------------------------------------
    # Per control tick
    # ------------------------------------------------------------------

    def update(self, model: object, data: object) -> None:
        """Advance belt, spawner, scoring and episode bookkeeping for one tick."""
        speed = self._config.belt_speed if self.running else 0.0
        data.ctrl[self._belt_actuator] = -speed
        self._wrap_belt(data)
        if self.running:
            self._maybe_spawn(model, data)
        self._score_items(data)
        self._maybe_finish_episode()
        self._next_item_s = self._time_to_next_item(data)
        self._update_lights(data)

    def reset_items(self, model: object, data: object) -> None:
        """Park every pool item, rewind the belt and start a fresh episode."""
        park_items(model, data)
        self.notify_manual_reset()

    def _wrap_belt(self, data: object) -> None:
        q = float(data.qpos[self._belt_qpos_adr])
        if self._last_belt_q is not None:
            self._travel += abs(q - self._last_belt_q)
        # Jump back by whole tiles; the belt texture repeats every tile, so this is invisible.
        if q <= -BELT_TILE:
            q += np.floor(-q / BELT_TILE) * BELT_TILE
        elif q > 0.0:
            q -= np.ceil(q / BELT_TILE) * BELT_TILE
        data.qpos[self._belt_qpos_adr] = q
        self._last_belt_q = float(data.qpos[self._belt_qpos_adr])

    def _maybe_spawn(self, model: object, data: object) -> None:
        episode = self._episode
        if episode.spawned >= self._config.items_per_episode or self._travel < episode.next_spawn_at:
            return
        item = self._pick_parked_item()
        if item is None:
            return
        self._place_on_belt(model, data, item)
        episode.spawned += 1
        jitter = float(self._rng.uniform(-1.0, 1.0)) * self._config.spawn_spacing_jitter
        episode.next_spawn_at = self._travel + self._config.spawn_spacing + jitter

    def _pick_parked_item(self) -> ConveyorItem | None:
        parked = [item for item in self._items if item.name not in self._on_belt]
        if not parked:
            return None
        cracked = bool(self._rng.random() < self._config.cracked_probability)
        preferred = [item for item in parked if item.cracked == cracked] or parked
        return preferred[int(self._rng.integers(len(preferred)))]

    def _place_on_belt(self, model: object, data: object, item: ConveyorItem) -> None:
        import mujoco  # noqa: PLC0415

        cfg = self._config
        x = cfg.belt_x + float(self._rng.uniform(-cfg.lateral_jitter, cfg.lateral_jitter))
        z = cfg.belt_top + item.half_height + 0.0005
        yaw = float(self._rng.uniform(0.0, 2.0 * np.pi))
        adr, dof = item.qpos_adr, item.dof_adr
        data.qpos[adr : adr + 3] = (x, cfg.spawn_y, z)
        data.qpos[adr + 3 : adr + 7] = (np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0))
        data.qvel[dof : dof + 6] = 0.0
        data.qvel[dof + 1] = -cfg.belt_speed
        self._on_belt[item.name] = item
        self._rest_since.pop(item.name, None)
        self._off_since.pop(item.name, None)
        mujoco.mj_forward(model, data)

    def _score_items(self, data: object) -> None:
        now = float(data.time)
        held = self._bodies_touching_robot(data)
        for name, item in list(self._on_belt.items()):
            pos = np.asarray(data.xpos[item.body_id])
            vel = float(np.linalg.norm(data.qvel[item.dof_adr : item.dof_adr + 3]))
            on_belt = (
                pos[2] > self._config.belt_top and abs(pos[0] - self._config.belt_x) < self._config.belt_half_width
            )
            if on_belt and pos[1] > self._config.belt_end_y:
                self._rest_since.pop(name, None)
                self._off_since.pop(name, None)
                continue
            if item.body_id in held:
                # In the gripper: neither resting nor stuck, however still it is.
                self._rest_since.pop(name, None)
                self._off_since[name] = now
                continue
            off_since = self._off_since.setdefault(name, now)
            if vel > self._config.settle_speed:
                self._rest_since.pop(name, None)
            rest_since = self._rest_since.setdefault(name, now) if vel <= self._config.settle_speed else now
            settled = pos[2] <= self._config.rest_max_z and now - rest_since >= self._config.settle_s
            # Stuck: off the belt and out of the gripper this long (balanced on a rim, wobbling...).
            stuck = now - off_since >= self._config.stuck_s
            if not (settled or stuck):
                continue
            landed = self._bin_containing(data, pos)
            outcome: Outcome
            outcome = "missed" if landed is None else "correct" if landed == item.target_bin else "wrong"
            setattr(self._episode.score, outcome, getattr(self._episode.score, outcome) + 1)
            logger.info(
                "Conveyor: {} -> {} ({}, expected {})", item.name, landed or "nowhere", outcome, item.target_bin
            )
            _park(data, item)
            del self._on_belt[name]
            self._rest_since.pop(name, None)
            self._off_since.pop(name, None)

    def _time_to_next_item(self, data: object) -> float | None:
        """Seconds until the next item rides out of the entry hood, or ``None`` if none is coming.

        Returns:
            The time in seconds, or ``None`` when the belt is stopped or the episode has no more items.
        """
        cfg = self._config
        speed = cfg.belt_speed if self.running else 0.0
        if speed <= 0.0:
            return None
        distances = [
            float(data.xpos[item.body_id][1]) - cfg.hood_exit_y
            for item in self._on_belt.values()
            if float(data.xpos[item.body_id][1]) > cfg.hood_exit_y
        ]
        if self._episode.spawned < cfg.items_per_episode:
            distances.append(max(0.0, self._episode.next_spawn_at - self._travel) + cfg.spawn_y - cfg.hood_exit_y)
        return min(distances) / speed if distances else None

    def _light_states(self) -> dict[str, bool]:
        running = self.running and self._config.belt_speed > 0.0
        soon = self._next_item_s is not None and self._next_item_s <= self._config.warn_s
        return {"green": running, "amber": running and soon, "red": not running}

    def _update_lights(self, data: object) -> None:
        for light, lit in self._light_states().items():
            if light not in self._lights:
                continue
            mocap, on_pos = self._lights[light]
            data.mocap_pos[mocap] = on_pos if lit else LIGHT_HIDDEN_POS

    def _bodies_touching_robot(self, data: object) -> set[int]:
        """Bodies in contact with any part of the arm (in practice: items in the gripper).

        Returns:
            Body ids touching the arm, excluding the arm's own bodies.
        """
        ncon = int(data.ncon)
        if ncon == 0:
            return set()
        b1 = self._geom_body[np.asarray(data.contact.geom1[:ncon])]
        b2 = self._geom_body[np.asarray(data.contact.geom2[:ncon])]
        r1, r2 = self._robot_body[b1], self._robot_body[b2]
        return {int(b) for b in np.concatenate([b2[r1 & ~r2], b1[r2 & ~r1]])}

    def _bin_containing(self, data: object, pos: np.ndarray) -> str | None:
        for name, body_id in self._bins.items():
            rot = np.asarray(data.xmat[body_id]).reshape(3, 3)
            local = rot.T @ (pos - np.asarray(data.xpos[body_id]))
            if abs(local[0]) < self._config.bin_half and abs(local[1]) < self._config.bin_half and local[2] < 0.06:  # noqa: PLR2004
                return name
        return None

    def _maybe_finish_episode(self) -> None:
        episode = self._episode
        if episode.spawned < self._config.items_per_episode or self._on_belt:
            return
        self._episode_count += 1
        self._last_episode = episode.score.as_dict()
        logger.info("Conveyor episode #{} done: {}", self._episode_count, self._last_episode)
        self._episode = _Episode(next_spawn_at=self._travel + self._config.spawn_spacing)


def pool_item_names() -> tuple[str, ...]:
    """Pool body names in a stable order: every shape x color, plain then cracked.

    The scene's ``conveyor_items.xml`` is generated from this list.

    Returns:
        Body names such as ``item_cube_red`` and ``item_hex_blue_cracked``.
    """
    return tuple(
        f"{ITEM_PREFIX}{shape}_{color}{'_cracked' if cracked else ''}"
        for cracked in (False, True)
        for shape in ITEM_SHAPES
        for color in ITEM_COLORS
    )


def park_items(model: object, data: object) -> None:
    """Park every pool item at its model default pose and rewind the belt."""
    import mujoco  # noqa: PLC0415

    for item in _discover_items(model):
        _park(data, item)
    belt_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, BELT_JOINT)
    if belt_joint >= 0:
        data.qpos[int(model.jnt_qposadr[belt_joint])] = 0.0
        data.qvel[int(model.jnt_dofadr[belt_joint])] = 0.0
    mujoco.mj_forward(model, data)


def _body_id(model: object, name: str) -> int:
    import mujoco  # noqa: PLC0415

    return int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name))


def _robot_root(model: object) -> int:
    """Id of the arm's root body.

    Returns:
        The ``base`` body id, or -1 when the model has no SO-101 arm.
    """
    import mujoco  # noqa: PLC0415

    return int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base"))


def _clamp_speed(speed: float) -> float:
    return float(min(max(speed, 0.0), MAX_BELT_SPEED))


def _discover_items(model: object) -> Iterator[ConveyorItem]:
    """Yield pool items named ``item_<shape>_<color>[_cracked]`` with free joints."""
    import mujoco  # noqa: PLC0415

    for body_id in range(int(model.nbody)):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        if not name.startswith(ITEM_PREFIX):
            continue
        parts = name[len(ITEM_PREFIX) :].split("_")
        if len(parts) not in {2, 3}:
            continue
        jnt = int(model.body_jntadr[body_id])
        if jnt < 0 or int(model.jnt_type[jnt]) != int(mujoco.mjtJoint.mjJNT_FREE):
            continue
        geom = int(model.body_geomadr[body_id])
        half_height = _geom_half_height(model, geom)
        qpos_adr = int(model.jnt_qposadr[jnt])
        park = tuple(float(v) for v in model.qpos0[qpos_adr : qpos_adr + 3])
        yield ConveyorItem(
            name=name,
            shape=parts[0],
            color=parts[1],
            cracked=len(parts) == 3 and parts[2] == "cracked",  # noqa: PLR2004
            body_id=body_id,
            qpos_adr=qpos_adr,
            dof_adr=int(model.jnt_dofadr[jnt]),
            half_height=half_height,
            park_xyz=(park[0], park[1], park[2]),
        )


def _geom_half_height(model: object, geom_id: int) -> float:
    import mujoco  # noqa: PLC0415

    gtype = int(model.geom_type[geom_id])
    size = model.geom_size[geom_id]
    if gtype == int(mujoco.mjtGeom.mjGEOM_BOX):
        return float(size[2])
    if gtype == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
        return float(size[1])
    # Meshes: use the bounding half-height MuJoCo stores in geom_rbound as an upper bound,
    # refined by the mesh's own vertex extent when available.
    mesh_id = int(model.geom_dataid[geom_id])
    if mesh_id >= 0:
        start = int(model.mesh_vertadr[mesh_id])
        count = int(model.mesh_vertnum[mesh_id])
        verts = model.mesh_vert[start : start + count]
        return float(np.max(np.abs(verts[:, 2])))
    return float(model.geom_rbound[geom_id])


def _park(data: object, item: ConveyorItem) -> None:
    adr, dof = item.qpos_adr, item.dof_adr
    data.qpos[adr : adr + 3] = item.park_xyz
    data.qpos[adr + 3 : adr + 7] = (1.0, 0.0, 0.0, 0.0)
    data.qvel[dof : dof + 6] = 0.0
