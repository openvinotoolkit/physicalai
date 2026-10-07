# Runtime contributor-skill scenarios

Run from a Runtime checkout. Reset agent context between prompts. Check for the correct contributor skill, a focused source-level change, and an executable test result.

## `pai-add-robot-integration`

1. "Add a built-in arm driver under `src/physicalai/robot`." Expect the structural Robot protocol, stable joint order, optional SDK extra, and fake-hardware tests.
2. "Reject a malicious SO-101 port string." Expect validation before any device operation, with no shell command from raw input.
3. "Export a new first-party driver." Expect package exports, `device_ids` before connection, and a protocol check.

## `pai-add-camera-backend`

1. "Add a new camera backend under `src/physicalai/capture`." Expect Camera lifecycle, monotonic frames, factory registration, and tests with fakes.
2. "Register a vendor camera with an optional extra." Expect lazy SDK imports and a clear dependency error.
3. "Explain SharedCamera for two processes." Expect transport-extra constraints, not a claim it works without dependencies.

## `pai-add-inference-component`

1. "Register a new built-in StatsNormalizer type." Expect component class, registry entry, a minimal manifest, and unit tests.
2. "Add a custom postprocessor to Runtime." Expect the correct processor boundary, artifact-path validation, and order tests.
3. "Reject a manifest with nested cyclic component specs." Expect the existing depth limit and a precise error.

## `pai-maintain-config`

1. "Change `Config.from_instance` serialization without breaking Studio." Expect Runtime-owned implementation, round-trip tests, and compatibility checks.
2. "Reject an untrusted class_path recipe." Expect the trust boundary and explicit `ConfigError` coverage.
3. "Add a documented Runtime config field." Expect canonical schema/how-to updates and focused `tests/unit/config/` checks.
