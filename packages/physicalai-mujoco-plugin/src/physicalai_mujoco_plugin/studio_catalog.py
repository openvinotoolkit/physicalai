# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""PhysicalAI Studio catalog registration for the simulated robots.

One follower entry is generated per supported profile (tiers ``twin``, ``dataset`` and
``experimental``) and arm layout (STU-1): one robot, and for twins two (``left_``, ``right_``),
which every tabletop scene runs. Dataset and experimental arms run two arms from the CLI only
(``physicalai-mujoco start --bimanual``). Its type is ``MuJoCo_<Profile>_Follower``, with ``Bimanual_``
before ``Follower`` for two robots, where ``<Profile>`` is the profile's display name without
spaces and punctuation; ``MuJoCo_SO101_Follower`` and ``MuJoCo_SO101_Bimanual_Follower`` keep
their types and payloads. Twins whose real robot has a Studio URDF reuse it (STU-2); the other
entries have no asset and rely on the owner's viewer (``GET /viewer``, STU-3). The SO-101 virtual
leader stays SO-101 specific (STU-6).

Importing this module neither imports MuJoCo nor downloads a model: the joint names of a profile
without channel overrides are derived from its model only when an entry is probed or built.
"""

from __future__ import annotations

import asyncio
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from loguru import logger
from physicalai_studio_plugin import (
    CatalogRobotFactory,
    PayloadContainer,
    PortScanner,
    RobotAdapterOptions,
    RobotAsset,
    RobotCatalogDefinition,
    RobotProbe,
    SerialPortInfo,
    robot_field_ui,
)
from pydantic import BaseModel, Field, create_model

from physicalai.config import export_config
from physicalai.robot.errors import RobotTransportError
from physicalai.robot.transport import SharedRobot
from physicalai_mujoco_plugin._urdf import get_urdf_path
from physicalai_mujoco_plugin.constants import (
    DEFAULT_BIMANUAL_MUJOCO_OWNER_NAME,
    DEFAULT_MUJOCO_OWNER_NAME,
    default_owner_name,
)
from physicalai_mujoco_plugin.profiles import PROFILES, SO101_PROFILE, RobotProfile
from physicalai_mujoco_plugin.profiles.trossen_wxai import TROSSEN_WXAI_PROFILE
from physicalai_mujoco_plugin.scene_registry import BIMANUAL_PREFIXES, supported_arm_counts
from physicalai_mujoco_plugin.virtual_leader import DEFAULT_HTTP_PORT, MuJoCoVirtualLeader

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from typing import Protocol

    import numpy as np

    from physicalai.robot.interface import Robot as PhysicalAIRobot
    from physicalai.robot.interface import RobotObservation

    class _RobotCatalogRegistry(Protocol):
        def register_robot(self, definition: RobotCatalogDefinition) -> None: ...


CATALOG_TIERS = ("twin", "dataset", "experimental")
"""Profile tiers that get Studio entries; ``unsupported`` models get none."""
BIMANUAL_TIERS = ("twin",)
"""Profile tiers that also get a two-arm entry: the robots with real bimanual leaders."""
DEFAULT_HTTP_URL = f"http://127.0.0.1:{DEFAULT_HTTP_PORT}"
"""The owner's camera and control server under ``physicalai-mujoco start``'s defaults."""
_LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
"""Hosts of a payload ``http_url`` that mean the simulation runs on Studio's machine."""
_NO_OWNER_ERROR = "no owner found for "
"""Start of the message an attach-only ``SharedRobot.connect`` raises when no owner has the name."""

_MUJOCO_SO101_TO_URDF: dict[str, list[str]] = {
    "shoulder_pan.pos": ["shoulder_pan"],
    "shoulder_lift.pos": ["shoulder_lift"],
    "elbow_flex.pos": ["elbow_flex"],
    "wrist_flex.pos": ["wrist_flex"],
    "wrist_roll.pos": ["wrist_roll"],
    "gripper.pos": ["gripper"],
}

_MUJOCO_SO101_BIMANUAL_TO_URDF: dict[str, list[str]] = {
    "left_shoulder_pan.pos": ["left_shoulder_pan"],
    "left_shoulder_lift.pos": ["left_shoulder_lift"],
    "left_elbow_flex.pos": ["left_elbow_flex"],
    "left_wrist_flex.pos": ["left_wrist_flex"],
    "left_wrist_roll.pos": ["left_wrist_roll"],
    "left_gripper.pos": ["left_gripper"],
    "right_shoulder_pan.pos": ["right_shoulder_pan"],
    "right_shoulder_lift.pos": ["right_shoulder_lift"],
    "right_elbow_flex.pos": ["right_elbow_flex"],
    "right_wrist_flex.pos": ["right_wrist_flex"],
    "right_wrist_roll.pos": ["right_wrist_roll"],
    "right_gripper.pos": ["right_gripper"],
}


def _get_mujoco_urdf_root() -> Path:
    return get_urdf_path()


_MUJOCO_SO101_ASSET = RobotAsset(
    urdf_relative_path=Path("so101/so101_new_calib.urdf"),
    packages={"so101": Path("so101")},
    joint_map=_MUJOCO_SO101_TO_URDF,
    root_resolver=_get_mujoco_urdf_root,
)

_MUJOCO_SO101_BIMANUAL_ASSET = RobotAsset(
    urdf_relative_path=Path("so101/so101_dual.urdf"),
    packages={"so101": Path("so101")},
    joint_map=_MUJOCO_SO101_BIMANUAL_TO_URDF,
    root_resolver=_get_mujoco_urdf_root,
)

# Studio's own WidowX AI URDF, the one its real WidowX AI entries use, names Trossen's joints; the
# gripper opening moves both carriages. No root resolver: the URDF ships with Studio.
_MUJOCO_WXAI_ASSET = RobotAsset(
    urdf_relative_path=Path("widowx/urdf/generated/wxai/wxai_follower.urdf"),
    packages={"trossen_arm_description": Path("widowx")},
    joint_map={
        "shoulder_pan.pos": ["joint_0"],
        "shoulder_lift.pos": ["joint_1"],
        "elbow_flex.pos": ["joint_2"],
        "wrist_flex.pos": ["joint_3"],
        "wrist_yaw.pos": ["joint_4"],
        "wrist_roll.pos": ["joint_5"],
        "gripper.pos": ["left_carriage_joint", "right_carriage_joint"],
    },
)

_ASSETS: dict[tuple[str, int], RobotAsset] = {
    (SO101_PROFILE.name, 1): _MUJOCO_SO101_ASSET,
    (SO101_PROFILE.name, 2): _MUJOCO_SO101_BIMANUAL_ASSET,
    (TROSSEN_WXAI_PROFILE.name, 1): _MUJOCO_WXAI_ASSET,
}
"""Studio URDFs of twin entries by ``(profile, number of robots)`` (STU-2); other entries have none."""


class MuJoCoRobotPayload(BaseModel):
    """Connection settings for a MuJoCo simulation owner."""

    name: str = Field(
        default=DEFAULT_MUJOCO_OWNER_NAME,
        description="Zenoh logical robot name of the running MuJoCo simulation",
    )
    allow_remote: bool = Field(  # type: ignore[call-overload]
        default=False,
        description="Allow connecting to a zenoh owner beyond localhost",
        json_schema_extra=robot_field_ui({"advanced_configuration": True}),
    )
    connect_timeout: float = Field(  # type: ignore[call-overload]
        default=10.0,
        description="Timeout in seconds for connecting to the zenoh owner",
        json_schema_extra=robot_field_ui({"advanced_configuration": True}),
    )
    http_url: str = Field(  # type: ignore[call-overload]
        default=DEFAULT_HTTP_URL,
        description="Address of the simulation's HTTP server (cameras and the 3D viewer at /viewer)",
        json_schema_extra=robot_field_ui({"advanced_configuration": True}),
    )


class MuJoCoSO101Payload(MuJoCoRobotPayload):
    """Connection settings for a MuJoCo SO-101 simulation owner."""


class MuJoCoSO101BimanualPayload(MuJoCoSO101Payload):
    """Connection settings for a bimanual MuJoCo SO-101 simulation owner.

    Identical to the single-arm payload apart from the default owner name, which
    matches ``physicalai-mujoco start --bimanual``. Sharing one default
    would make both catalog entries probe the same owner.
    """

    name: str = Field(
        default=DEFAULT_BIMANUAL_MUJOCO_OWNER_NAME,
        description="Zenoh logical robot name of the running bimanual MuJoCo simulation",
    )


@dataclass(frozen=True)
class CatalogEntry:
    """One generated follower entry: a profile attached at one or two scene anchors."""

    profile: RobotProfile
    prefixes: tuple[str, ...]
    """Robot prefixes in anchor order: ``("",)``, or :data:`BIMANUAL_PREFIXES`."""

    @property
    def bimanual(self) -> bool:
        """Whether the entry drives two robots."""
        return len(self.prefixes) == 2  # noqa: PLR2004

    @property
    def type(self) -> str:
        """Catalog type, e.g. ``MuJoCo_WidowXAI_Follower`` or ``MuJoCo_SO101_Bimanual_Follower``."""
        token = re.sub(r"[^0-9A-Za-z]", "", self.profile.display_name)
        return f"MuJoCo_{token}_{'Bimanual_' if self.bimanual else ''}Follower"

    @property
    def display_name(self) -> str:
        """Catalog display name, e.g. ``MuJoCo SO-101 Bimanual Follower``; experimental entries say so."""
        name = f"MuJoCo {self.profile.display_name} {'Bimanual ' if self.bimanual else ''}Follower"
        return f"{name} (experimental)" if self.profile.tier == "experimental" else name

    @property
    def owner_name(self) -> str:
        """Default owner name, as ``physicalai-mujoco start`` publishes it (CLI-3)."""
        return default_owner_name(self.profile.name, len(self.prefixes))

    def start_hint(self, owner_name: str, http_url: str = DEFAULT_HTTP_URL) -> str:
        """Return how to start the simulation that a follower of this entry attaches to (P2).

        Args:
            owner_name: The payload's owner name; ``--name`` is added when it is not the default.
            http_url: The payload's HTTP address: a local port other than the default adds
                ``--http-port``, and another host adds ``--http-host`` and ``--allow-remote`` and
                says where to run it. An address without a host is left out.

        Returns:
            ``Start it with: uv run physicalai-mujoco start --profile <profile> ...``.
        """
        command = f"uv run physicalai-mujoco start --profile {self.profile.name}"
        if self.bimanual:
            command += " --bimanual"
        if owner_name != self.owner_name:
            # One token, so a name that starts with "-" is not taken for an option.
            command += f" --name={shlex.quote(owner_name)}"
        try:
            url = urlsplit(http_url)
            host, port = url.hostname, url.port or (443 if url.scheme == "https" else 80)
        except ValueError:
            host, port = None, DEFAULT_HTTP_PORT
        if host is None or host in _LOCAL_HOSTS:
            return f"Start it with: {command}" + (f" --http-port {port}" if port != DEFAULT_HTTP_PORT else "")
        command += f" --http-host {shlex.quote(host)} --http-port {port} --allow-remote"
        return f"Start it on {host} with: {command}"

    def joint_names(self) -> tuple[str, ...]:
        """Return the public joint names an owner of this entry has.

        A profile with channel overrides names them; otherwise they are derived from the robot
        model, which is compiled (and, on a cold Menagerie cache, downloaded) on the first call.

        Returns:
            Every robot's names with its prefix, in anchor order.
        """
        names = [channel.name for channel in self.profile.channels]
        if not names:
            from physicalai_mujoco_plugin.compose import robot_layout  # noqa: PLC0415

            names = list(robot_layout(self.profile).joint_names)
        return tuple(f"{prefix}{name}" for prefix in self.prefixes for name in names)


def list_catalog_entries() -> tuple[CatalogEntry, ...]:
    """Return one entry per supported profile and arm layout, in registry order (STU-1).

    Every profile in :data:`CATALOG_TIERS` gets a single-robot entry; those in
    :data:`BIMANUAL_TIERS` that can run two arms also get a bimanual one.

    Returns:
        The entries, each profile's single-robot entry first.
    """
    entries = []
    for profile in PROFILES.values():
        if profile.tier not in CATALOG_TIERS:
            continue
        entries.append(CatalogEntry(profile, ("",)))
        if profile.tier in BIMANUAL_TIERS and len(BIMANUAL_PREFIXES) in supported_arm_counts(profile):
            entries.append(CatalogEntry(profile, BIMANUAL_PREFIXES))
    return tuple(entries)


_PAYLOAD_MODELS: dict[str, type[MuJoCoRobotPayload]] = {
    "MuJoCo_SO101_Follower": MuJoCoSO101Payload,
    "MuJoCo_SO101_Bimanual_Follower": MuJoCoSO101BimanualPayload,
}
"""Payload models by catalog type; the other entries' are generated once, on first use."""


def _payload_model(entry: CatalogEntry) -> type[MuJoCoRobotPayload]:
    """Return the entry's payload model: today's SO-101 classes, else one whose ``name`` defaults to the entry's owner.

    Returns:
        A payload model class, the same one on every call.
    """
    model = _PAYLOAD_MODELS.get(entry.type)
    if model is None:
        model = _PAYLOAD_MODELS[entry.type] = _generated_payload_model(entry)
    return model


def _generated_payload_model(entry: CatalogEntry) -> type[MuJoCoRobotPayload]:
    model_name = f"{entry.type.removesuffix('_Follower').replace('_', '')}Payload"
    return create_model(
        model_name,
        __base__=MuJoCoRobotPayload,
        __module__=__name__,
        __doc__=f"Connection settings for a MuJoCo {entry.profile.display_name} simulation owner.",
        name=(
            str,
            Field(
                default=entry.owner_name,
                description=f"Zenoh logical robot name of the running {entry.display_name} simulation",
            ),
        ),
    )


def _check_zenoh_robot_online(name: str, joint_order: tuple[str, ...]) -> bool:
    """Return whether an owner named `name` is reachable and drives exactly `joint_order`.

    A single-arm catalog entry pointed at a bimanual simulation (or the other
    way round) would otherwise attach and fail on its first action.
    """
    try:
        robot = SharedRobot.attach(name=name, connect_timeout=2.0)
        robot.connect()
        try:
            joint_names = tuple(robot.joint_names)
        finally:
            robot.disconnect()
    except (ConnectionError, TimeoutError, RuntimeError):
        return False
    if joint_names != joint_order:
        logger.warning(
            "MuJoCo owner {!r} drives joints {} but this robot type expects {}",
            name,
            list(joint_names),
            list(joint_order),
        )
        return False
    return True


def _check_entry_online(name: str, entry: CatalogEntry) -> bool:
    """Return whether an owner named `name` is reachable and drives the entry's joints.

    A profile without channel overrides derives its names from its model; when that fails (no
    cached model and no download), the entry reports offline.
    """
    try:
        joint_names = entry.joint_names()
    except (RuntimeError, ValueError) as exc:
        logger.warning("Cannot check the {} owner {!r}: {}", entry.profile.name, name, exc)
        return False
    return _check_zenoh_robot_online(name, joint_names)


class MuJoCoRobotProbe(RobotProbe[MuJoCoRobotPayload]):
    """Discover and query MuJoCo simulation owners of one catalog entry."""

    def __init__(self, entry: CatalogEntry) -> None:
        """Probe for owners that drive the joints of `entry`."""
        self.entry = entry

    async def discover(self, manager: PortScanner) -> list[SerialPortInfo]:
        """Return robots found by the port scanner."""
        _ = self
        await manager.find_robots()
        return manager.robots

    async def identify(
        self,
        payload: MuJoCoRobotPayload,
        manager: PortScanner | None = None,
        joint: str | None = None,
    ) -> None:
        """Perform no visual identification for the simulated robot."""
        _ = self, payload, manager, joint

    async def is_online(
        self,
        payload: MuJoCoRobotPayload,
        manager: PortScanner | None = None,
    ) -> bool:
        """Return whether the configured simulation owner is reachable and drives this entry's joints."""
        _ = manager
        return await asyncio.to_thread(_check_entry_online, payload.name, self.entry)


class MuJoCoVirtualLeaderPayload(BaseModel):
    """Connection settings for the simulation's virtual leader arm."""

    http_port: int = Field(  # type: ignore[call-overload]
        default=DEFAULT_HTTP_PORT,
        ge=1,
        le=65535,
        description="HTTP port of the running MuJoCo simulation on this machine (its --http-port)",
    )


def _check_virtual_leader_online(http_port: int) -> bool:
    leader = MuJoCoVirtualLeader(http_port=http_port, timeout_s=1.0)
    try:
        leader.connect()
    except ConnectionError:
        return False
    leader.disconnect()
    return True


class MuJoCoVirtualLeaderProbe(RobotProbe[MuJoCoVirtualLeaderPayload]):
    """Check that a simulation publishes a virtual leader pose."""

    async def discover(self, manager: PortScanner) -> list[SerialPortInfo]:
        """Return no serial devices: the leader is a simulation endpoint."""
        _ = self, manager
        await asyncio.sleep(0)
        return []

    async def identify(
        self,
        payload: MuJoCoVirtualLeaderPayload,
        manager: PortScanner | None = None,
        joint: str | None = None,
    ) -> None:
        """Perform no visual identification for the virtual leader."""
        _ = self, payload, manager, joint

    async def is_online(
        self,
        payload: MuJoCoVirtualLeaderPayload,
        manager: PortScanner | None = None,
    ) -> bool:
        """Return whether the simulation answers on its HTTP port."""
        _ = self, manager
        return await asyncio.to_thread(_check_virtual_leader_online, payload.http_port)


async def _build_virtual_leader(
    robot: PayloadContainer[MuJoCoVirtualLeaderPayload],
    factory: CatalogRobotFactory,
) -> PhysicalAIRobot:
    _ = factory
    await asyncio.sleep(0)
    raw = robot.payload
    validated = raw if isinstance(raw, MuJoCoVirtualLeaderPayload) else MuJoCoVirtualLeaderPayload.model_validate(raw)
    return MuJoCoVirtualLeader(http_port=validated.http_port)


@export_config(class_path="physicalai_mujoco_plugin.studio_catalog._SharedMuJoCoRobot")
class _SharedMuJoCoRobot:
    """A Studio robot attached to a running simulation owner, of any profile and arm layout.

    The joint names are the catalog entry's, so Studio knows them before connecting; observations
    pass through unchanged, including a floating base's longer ``state``.
    """

    def __init__(
        self, shared_robot: SharedRobot, joint_names: list[str] | tuple[str, ...], start_hint: str = ""
    ) -> None:
        """Wrap an attach-only shared robot.

        Args:
            shared_robot: Attaches to the simulation owner.
            joint_names: The catalog entry's joint names.
            start_hint: How to start the simulation (:meth:`CatalogEntry.start_hint`), given when no
                owner is running.
        """
        self._shared_robot = shared_robot
        self.joint_names = list(joint_names)
        self._start_hint = start_hint

    @property
    def device_ids(self) -> tuple[str, ...]:
        """The attached MuJoCo owner, not this wrapper, owns the simulation."""
        return ()

    def connect(self) -> None:
        """Attach to the running simulation owner.

        Raises:
            RobotTransportError: If no owner has the name: the message names it and says how to
                start it, so Studio can show it as is. Other transport errors pass through unchanged.
        """
        try:
            self._shared_robot.connect()
        except RobotTransportError as exc:
            if type(exc) is not RobotTransportError or not str(exc).startswith(_NO_OWNER_ERROR):
                raise
            msg = f"No MuJoCo simulation named {self._shared_robot.name!r} is running."
            if self._start_hint:
                msg += f" {self._start_hint}"
            raise RobotTransportError(msg) from exc

    def disconnect(self) -> None:
        self._shared_robot.disconnect()

    def get_observation(self) -> RobotObservation:
        return self._shared_robot.get_observation()

    def send_action(self, action: np.ndarray, *, goal_time: float = 0.1) -> None:
        self._shared_robot.send_action(action, goal_time=goal_time)

    def is_connected(self) -> bool:
        return self._shared_robot.is_connected()


def _mujoco_robot_builder(
    entry: CatalogEntry,
) -> Callable[[PayloadContainer[MuJoCoRobotPayload], CatalogRobotFactory], Awaitable[PhysicalAIRobot]]:
    """Build the catalog builder of one entry.

    Returns:
        An async robot builder that attaches to the payload's zenoh owner.
    """
    payload_model = _payload_model(entry)

    async def build(
        robot: PayloadContainer[MuJoCoRobotPayload],
        factory: CatalogRobotFactory,
    ) -> PhysicalAIRobot:
        _ = factory
        raw = robot.payload
        validated = raw if isinstance(raw, payload_model) else payload_model.model_validate(raw)
        joint_names = await asyncio.to_thread(entry.joint_names)

        shared = SharedRobot.attach(
            name=validated.name,
            allow_remote=validated.allow_remote,
            connect_timeout=validated.connect_timeout,
        )
        return _SharedMuJoCoRobot(shared, joint_names, entry.start_hint(validated.name, validated.http_url))

    return build


def _follower_definition(entry: CatalogEntry) -> RobotCatalogDefinition:
    return RobotCatalogDefinition(
        type=entry.type,
        display_name=entry.display_name,
        role="follower",
        robot_builder=_mujoco_robot_builder(entry),
        robot_payload=_payload_model(entry),
        asset=_ASSETS.get((entry.profile.name, len(entry.prefixes))),
        adapter_options=RobotAdapterOptions(
            include_velocities=False,
            external_effort_gain=None,
        ),
        probe=MuJoCoRobotProbe(entry),
        # A simulated joint's zero is the model's zero; there is no motor offset to store.
        zero_calibration=None,
    )


def _definitions() -> list[RobotCatalogDefinition]:
    """Return the generated follower entries, with the SO-101 virtual leader after the SO-101's.

    Returns:
        The SO-101 followers, the virtual leader, then every other profile's followers.
    """
    entries = list_catalog_entries()
    so101 = [_follower_definition(entry) for entry in entries if entry.profile is SO101_PROFILE]
    others = [_follower_definition(entry) for entry in entries if entry.profile is not SO101_PROFILE]
    leader = RobotCatalogDefinition(
        type="MuJoCo_SO101_Virtual_Leader",
        display_name="MuJoCo SO-101 Virtual Leader",
        role="leader",
        robot_builder=_build_virtual_leader,
        robot_payload=MuJoCoVirtualLeaderPayload,
        asset=_MUJOCO_SO101_ASSET,
        adapter_options=RobotAdapterOptions(
            include_velocities=False,
            external_effort_gain=None,
        ),
        probe=MuJoCoVirtualLeaderProbe(),
        zero_calibration=None,
    )
    return [*so101, leader, *others]


def _assert_payload_model_resolvable(model: type[BaseModel]) -> None:
    model.model_rebuild(_types_namespace=globals(), raise_errors=True)


def register_physicalai_studio_plugin(registry: _RobotCatalogRegistry) -> None:
    """Register the MuJoCo catalog definitions."""
    for definition in _definitions():
        payload_model = definition.robot_payload
        if isinstance(payload_model, type) and issubclass(payload_model, BaseModel):
            _assert_payload_model_resolvable(payload_model)
        registry.register_robot(definition)
