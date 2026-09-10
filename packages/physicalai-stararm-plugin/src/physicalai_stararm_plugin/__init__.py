# ruff: noqa: PLC0415

"""Star Arm 102 plugin for PhysicalAI.

Provides Star Arm 102-LD/102-HD leader drivers and a Star Arm 102-FL follower
driver compatible with the ``physicalai.robot.Robot`` protocol.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from physicalai_stararm_plugin._urdf import get_urdf_path as get_urdf_path

if TYPE_CHECKING:
    from physicalai_stararm_plugin.stararm102fl import StarArm102FLFollower as StarArm102FLFollower
    from physicalai_stararm_plugin.stararm102fl import (
        StarArm102FLFollowerObservation as StarArm102FLFollowerObservation,
    )
    from physicalai_stararm_plugin.stararm102hd import StarArm102HDLeader as StarArm102HDLeader
    from physicalai_stararm_plugin.stararm102hd import (
        StarArm102HDLeaderObservation as StarArm102HDLeaderObservation,
    )
    from physicalai_stararm_plugin.stararm102ld import StarArm102LDLeader as StarArm102LDLeader

__all__ = [
    "StarArm102FLFollower",
    "StarArm102FLFollowerObservation",
    "StarArm102HDLeader",
    "StarArm102HDLeaderObservation",
    "StarArm102LDLeader",
    "get_urdf_path",
]


def __getattr__(name: str) -> object:
    if name == "StarArm102HDLeader":
        from physicalai_stararm_plugin.stararm102hd import StarArm102HDLeader

        return StarArm102HDLeader
    if name == "StarArm102HDLeaderObservation":
        from physicalai_stararm_plugin.stararm102hd import StarArm102HDLeaderObservation

        return StarArm102HDLeaderObservation
    if name == "StarArm102LDLeader":
        from physicalai_stararm_plugin.stararm102ld import StarArm102LDLeader

        return StarArm102LDLeader
    if name == "StarArm102FLFollower":
        from physicalai_stararm_plugin.stararm102fl import StarArm102FLFollower

        return StarArm102FLFollower
    if name == "StarArm102FLFollowerObservation":
        from physicalai_stararm_plugin.stararm102fl import StarArm102FLFollowerObservation

        return StarArm102FLFollowerObservation
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
