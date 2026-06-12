---
name: runtime-adding-a-robot
description: Add or modify a Physical AI Runtime robot integration. Use when implementing robot hardware support, the Robot protocol, connect/disconnect behavior, get_observation, send_action, joint_names, verification, safety checks, SO-101, Trossen/WidowX, or other robot adapters.
license: Apache-2.0
---

# Adding a Runtime Robot

Use this skill for changes under `src/physicalai/robot` or tests/docs for robot integrations.

## Workflow

1. Inspect the `Robot` protocol and existing concrete robot packages before adding a new implementation.
2. Prefer structural typing. A robot integration must satisfy the protocol; inheritance is not required unless nearby code already uses it.
3. Implement connection lifecycle explicitly: `connect()`, `disconnect()`, and safe cleanup on errors.
4. Implement `get_observation()` with stable joint ordering and timestamp/metadata conventions that match existing robots.
5. Implement `send_action(...)` with clear units, limits, timing semantics, and safe failure behavior.
6. Add or update verification helpers and tests so users can validate hardware before running a policy.

## Safety Requirements

- Never skip joint limit, workspace, speed, or emergency-stop considerations in user-facing examples.
- Prefer reduced-speed first-run instructions in docs.
- Make hardware dependency errors actionable: mention missing packages, permissions, ports, udev rules, or drivers when relevant.

## References

- See `references/robot-protocol.md` for the protocol checklist.
