# Robot protocol

Structural interface: `src/physicalai/robot/interface.py`.

## Robot

Required methods:

- `connect()` — idempotent; safe to call when already connected.
- `disconnect()` — release hardware resources.
- `is_connected() -> bool`
- `get_observation()` — returns an object satisfying `RobotObservation`.
- `send_action(action: np.ndarray, *, goal_time: float = ...)` — send command vector.

Class attribute:

- `joint_names: list[str]` — order matches `joint_positions` and training action layout.

Property:

- `device_ids: tuple[str, ...]` — canonical, scheme-qualified identities of every
  physical device this instance exclusively owns (e.g. `("serial:ttyUSB0",)`,
  `("tcp:192.168.1.2",)`). Must be derivable from constructor parameters alone,
  with no hardware I/O, and available before `connect()`. Composite robots
  return every constituent device's id; robots with no exclusively-owned
  hardware (e.g. a network subscriber) return `()`. Used by
  `physicalai.robot.transport` for host-local exclusivity locking — not a
  transport name or key.

## RobotObservation

Attributes:

- `joint_positions: np.ndarray` shape `(N,)`
- `timestamp: float` — `time.monotonic()` at capture
- `sensor_data: dict[str, np.ndarray] | None`
- `images: dict[str, Frame] | None` — embedded cameras only; external cameras use `physicalai.capture`

Property:

- `state` — defaults to `joint_positions`; override when inference expects concatenated state.

## Design notes

- Use `@runtime_checkable` protocol checks in tests: `isinstance(instance, Robot)`.
- External cameras are configured on `PolicyRuntime`, not inside the robot driver, unless the hardware truly provides embedded frames via `images`.
