"""Star Arm 102-LD leader driver."""

from __future__ import annotations

from typing import ClassVar

from physicalai.config import export_config

from physicalai_stararm_plugin.stararm102hd import StarArm102HDLeader


@export_config
class StarArm102LDLeader(StarArm102HDLeader):
    """FashionStar UART leader arm driver for Star Arm 102-LD."""

    MODEL_NAME: ClassVar[str] = "Star Arm 102-LD"
    DEVICE_PREFIX: ClassVar[str] = "stararm102-ld"

    def __init__(
        self,
        port: str = "/dev/ttyUSB0",
        *,
        baudrate: int = 1_000_000,
        unlock_on_connect: bool = True,
        reset_multi_turn_on_connect: bool = True,
        zero_on_connect: bool = False,
    ) -> None:
        """Initialize the Star Arm 102-LD leader driver."""
        super().__init__(
            port=port,
            baudrate=baudrate,
            unlock_on_connect=unlock_on_connect,
            reset_multi_turn_on_connect=reset_multi_turn_on_connect,
            zero_on_connect=zero_on_connect,
            control_mode="passive",
            command_interval_ms=10,
        )
