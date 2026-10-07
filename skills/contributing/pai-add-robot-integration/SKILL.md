---
name: pai-add-robot-integration
description: Adds or modifies a first-party robot integration in the Physical AI Runtime source tree. Use when implementing the Robot protocol under src/physicalai/robot, registering built-in drivers or optional extras, or changing verify.py and robot tests. DO NOT USE FOR running a policy with an existing robot (use pai-run-policy) or creating a third-party plugin in its own repository.
license: Apache-2.0
---

# Adding a Robot Integration

Robots satisfy the structural `Robot` protocol in `src/physicalai/robot/interface.py` — no base class inheritance required. References: `src/physicalai/robot/so101/`, `src/physicalai/robot/trossen/` (WidowX). Connection helpers: `src/physicalai/robot/connect.py`; validation: `src/physicalai/robot/verify.py`.

## Workflow

1. **Read `references/robot-protocol.md`** and an existing integration (SO-101 for serial servos, WidowX for arms).
   - Done when: required methods and observation shape are listed.
2. **Implement connect lifecycle**: idempotent `connect()`, `disconnect()`, `is_connected()`.
3. **Observations**: return a type exposing `joint_positions`, `timestamp` (`time.monotonic()`), optional `sensor_data` / `images`; implement `state` property when inference expects more than positions.
4. **Actions**: `send_action(action, *, goal_time=...)` with `action` shape matching training data conventions; document joint order via `joint_names`.
5. **Optional extra** in `pyproject.toml` (`physicalai[so101]`, `physicalai[trossen]`); lazy-import vendor SDKs.
6. **Export** public class from `physicalai.robot` when user-facing.
7. **Tests** under `tests/unit/robot/` with mocked hardware.
   - Done when: `uv run pytest tests/unit/robot -k <name>` passes.
8. **Verification CLI/docs** — wire `verify.py` patterns if the robot supports automated checks.

## Required checks

- `isinstance(robot, Robot)` at runtime (`@runtime_checkable` protocol).
- No action sent when disconnected.
- Joint count and `joint_names` stay stable across connect/disconnect.
- Security: validate user-supplied port/path strings; no shell invocation with unsanitized device paths.

## References

- `references/robot-protocol.md`
- `docs/explanation/robots.md`
