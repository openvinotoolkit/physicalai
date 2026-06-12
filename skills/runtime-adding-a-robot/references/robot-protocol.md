# Robot Protocol Checklist

Runtime robots are protocol-based. A concrete robot should provide the methods and properties used by `PolicyRuntime` and verification helpers.

## Required Behavior

- `connect()`: initialize hardware communication and fail with actionable errors.
- `disconnect()`: release hardware resources and be safe to call during cleanup.
- `get_observation()`: return joint state using stable joint ordering.
- `send_action(...)`: command the robot using documented units and timing semantics.
- `joint_names`: expose the exact joint order expected by observations and actions.

## Implementation Notes

- Keep hardware SDK imports lazy or guarded when possible so optional extras remain optional.
- Do not require inheritance unless needed by existing code. Structural typing is the intended extension mechanism.
- Ensure observation and action dimensions match `joint_names`.
- Keep verification code conservative. Users should validate joints individually before running policies.
