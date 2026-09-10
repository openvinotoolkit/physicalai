# ruff: noqa: PLC0415

"""Third-party Seeed reBot B601 robot arm plugin for PhysicalAI.

Provides two robot drivers compatible with the ``physicalai.robot.Robot`` protocol:

* :class:`ReBotB601DM` -- Damiao motors over serial/CAN (POS_VEL / FORCE_POS modes).
* :class:`ReBotB601RS` -- RobStride motors over CAN (MIT mode).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from physicalai_rebot_b601_plugin._urdf import get_urdf_path as get_urdf_path

if TYPE_CHECKING:
    from physicalai_rebot_b601_plugin.dm import ReBotB601DM as ReBotB601DM
    from physicalai_rebot_b601_plugin.dm import ReBotB601DMObservation as ReBotB601DMObservation
    from physicalai_rebot_b601_plugin.rs import ReBotB601RS as ReBotB601RS
    from physicalai_rebot_b601_plugin.rs import ReBotB601RSObservation as ReBotB601RSObservation

__all__ = [
    "ReBotB601DM",
    "ReBotB601DMObservation",
    "ReBotB601RS",
    "ReBotB601RSObservation",
    "get_urdf_path",
]


def __getattr__(name: str) -> object:
    if name == "ReBotB601DM":
        from physicalai_rebot_b601_plugin.dm import ReBotB601DM

        return ReBotB601DM
    if name == "ReBotB601DMObservation":
        from physicalai_rebot_b601_plugin.dm import ReBotB601DMObservation

        return ReBotB601DMObservation
    if name == "ReBotB601RS":
        from physicalai_rebot_b601_plugin.rs import ReBotB601RS

        return ReBotB601RS
    if name == "ReBotB601RSObservation":
        from physicalai_rebot_b601_plugin.rs import ReBotB601RSObservation

        return ReBotB601RSObservation
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
