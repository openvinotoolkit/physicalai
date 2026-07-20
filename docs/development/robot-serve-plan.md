# Robot Serve design

## Status

Implemented production design. This document replaces the proof-of-concept plan
implemented in commit `2ed63a6ea2a53ea6548a149cc4a82cb1022ce64c`.

The PoC established that a robot can be exposed through the existing Zenoh
`SharedRobot` transport and consumed from another process or host. The
production implementation must now make foreground ownership, shutdown,
failure reporting, and CLI boundaries explicit.

## Decision

Add these operator commands:

```text
physicalai robot serve
physicalai robot discover
```

`physicalai robot serve` runs the hardware owner in the foreground process. It
does not launch a detached persistent child. Foreground serving and the
existing auto-spawn worker call one shared owner runtime that exclusively
constructs, connects, reads, commands, and disconnects the robot.

The auto-spawn path remains detached and retains its numeric idle timeout. The
explicit CLI path is persistent until interrupted and remains bound to the
lifetime of the foreground process.

## Goals

- Provide a convenient YAML- and argument-driven command for serving one robot.
- Reuse the proven `SharedRobot` Zenoh protocol and hardware ownership model.
- Keep exactly one process responsible for each live robot driver.
- Guarantee that normal CLI termination attempts a safe driver disconnect.
- Report startup, loop, forced-termination, and disconnect failures as nonzero.
- Preserve secure local-only networking unless remote access is explicit.
- Keep auto-spawn behavior compatible with existing `SharedRobot` callers.
- Expose deterministic human-readable and machine-readable discovery.

## Non-goals

- Authentication, authorization, or encryption in the robot transport.
- Cross-host robot-name arbitration or distributed ownership leases.
- Multiple-controller action arbitration.
- Daemonization, PID files, or a background-service manager in the CLI.
- Zenoh router provisioning.
- Driver-specific CLI arguments.
- Reworking the robot wire protocol.
- Detailed nested shell completion.

Operating-system service managers such as systemd, Docker, or Kubernetes own
background supervision. The CLI remains a foreground command suitable for
those supervisors.

## Why the PoC Architecture Must Change

The PoC reused `RobotOwner`, which always starts its worker with
`start_new_session=True`, and made that detached worker persistent by setting
`idle_timeout=None`. This combines two individually useful behaviors into an
unsafe lifecycle:

1. The CLI process starts a detached owner.
2. The owner has no idle self-termination.
3. If the CLI is killed or crashes before signaling the owner, the owner can
   continue indefinitely while retaining the hardware and ownership locks.

That violates the operator-facing contract that the robot is served for the
lifetime of `physicalai robot serve`.

The PoC also made the CLI depend on private transport validators and mixed
explicit-host networking changes into the serve feature. Those concerns should
be separated so each can be reviewed against one contract.

## Architecture

### Shared owner runtime

Extract the hardware-owning behavior from the subprocess entry point into one
internal, in-process runtime:

```python
class OwnerExitReason(enum.Enum):
    SHUTDOWN = "shutdown"
    IDLE_TIMEOUT = "idle_timeout"
    CONSECUTIVE_READ_FAILURES = "consecutive_read_failures"
    LOOP_FAILURE = "loop_failure"


@dataclass(frozen=True)
class OwnerResult:
    reason: OwnerExitReason
    exit_code: int


def run_owner(
    config: RobotOwnerConfig,
    shutdown: threading.Event,
    *,
    ready: Callable[[], None] | None = None,
) -> OwnerResult:
    ...
```

The exact names and callback shape are implementation details. The contract is
that this runtime owns the complete lifecycle:

```text
validate config
    -> import and construct driver
    -> acquire name and device locks
    -> connect driver
    -> read initial observation
    -> declare Zenoh endpoints
    -> report ready
    -> run control loop
    -> disconnect driver
    -> undeclare/close/release best-effort resources
    -> return structured result
```

No launch adapter duplicates hardware construction, loop, or cleanup logic.

### Launch adapters

Two adapters invoke the same runtime with different lifecycle policies:

| Concern              | Auto-spawn worker                | `robot serve`                |
| -------------------- | -------------------------------- | ---------------------------- |
| Process              | Dedicated child                  | Foreground CLI process       |
| Detachment           | New session                      | None                         |
| Idle timeout         | Numeric, default 10 seconds      | Disabled (`None`)            |
| Readiness            | `READY`/`ERROR` IPC              | Operator message after ready |
| Shutdown source      | SIGTERM or idle timeout          | SIGINT/SIGTERM               |
| Parent death         | Idle timeout eventually stops it | Process itself is gone       |
| Exit status consumer | `RobotOwner` parent              | Shell/service manager        |

The subprocess module remains a thin adapter around `run_owner` for the
existing auto-spawn handshake. The CLI adapter installs signal handlers, calls
`run_owner` directly, and maps `OwnerResult` to a process exit code.

### Responsibility boundaries

`physicalai.robot.transport` owns:

- Owner configuration and validation.
- Driver import and construction.
- Host-local name and device locks.
- Zenoh endpoints and payloads.
- The control loop and its termination reasons.
- Safe driver disconnect and cleanup ordering.
- Discovery of metadata records.

`physicalai.cli.robot` owns:

- Command-line and config-file parsing.
- Direct group and nested command help.
- Signal-to-shutdown-event adaptation for foreground serving.
- Operator-facing output and exit-code presentation.
- Sorting and formatting discovery results.

The CLI must not import private field validators or reproduce transport
validation. It constructs one supported owner configuration and catches the
documented validation and transport errors at the command boundary.

## Lifecycle Invariants

1. Only the owner runtime imports, constructs, connects, reads, commands, and
   disconnects a concrete robot driver.
2. Locks are acquired before `driver.connect()` performs hardware access.
3. Readiness is reported only after the driver is connected, an initial valid
   observation has been read, and all Zenoh endpoints are declared.
4. A foreground server never leaves a persistent detached worker by design.
5. SIGINT and SIGTERM request graceful shutdown; they do not bypass cleanup.
6. Driver disconnect is attempted after every post-connect exit path.
7. Driver-disconnect failure is safety-critical and forces a nonzero result.
8. Metadata undeclaration, Zenoh session close, and explicit lock release are
   best-effort. Their failures are logged and do not replace the primary result.
9. Repeated observation failures and unexpected loop failures are nonzero.
10. Numeric idle timeout and requested shutdown succeed only when safe driver
    disconnect also succeeds.
11. Subscriber disconnect never directly stops the robot; owner policy controls
    the safe-state transition.

`SIGKILL`, kernel failure, and power loss cannot run Python cleanup. Drivers and
hardware must fail safely where possible, but that is outside this command's
guarantees.

## Configuration

Use the existing flat owner shape rather than introducing another
`class_path`/`init_args` representation:

```yaml
name: follower-arm
robot_class: physicalai.robot.SO101
robot_kwargs:
  port: /dev/ttyACM0
  calibration: /path/to/calibration.json
  role: follower
allow_remote: false
rate_hz: 100.0
```

| Field          | Required | Meaning                                         |
| -------------- | -------- | ----------------------------------------------- |
| `name`         | Yes      | Logical name and Zenoh key segment              |
| `robot_class`  | Yes      | Trusted local dotted path to a driver class     |
| `robot_kwargs` | No       | JSON-serializable constructor keyword arguments |
| `allow_remote` | No       | Expose beyond localhost; default `false`        |
| `rate_hz`      | No       | Positive finite owner loop rate; default 100 Hz |

`idle_timeout` is not exposed by `robot serve`. Explicit serving is persistent
until interrupted. Auto-spawn continues to supply its existing numeric timeout.

Validate all fields before constructing hardware:

- `name` is one nonempty safe Zenoh key segment.
- `robot_class` is a nonempty dotted path. Import occurs only in the owner.
- `robot_kwargs` is JSON-serializable across the worker IPC boundary.
- `rate_hz` is finite and greater than zero; booleans are rejected.

The class path is trusted local configuration. Network metadata must never be
used as a class path or imported by discovery.

## CLI Design

### Serve

```bash
physicalai robot serve --config examples/so101/serve.yaml
```

Direct arguments remain available for scripting:

```bash
physicalai robot serve \
  --name follower-arm \
  --robot_class physicalai.robot.SO101 \
  --robot_kwargs.port /dev/ttyACM0 \
  --robot_kwargs.calibration /path/to/calibration.json \
  --allow_remote
```

`--allow_remote` is a bare `store_true` switch. Local-only operation is the
default. A readiness message goes to stderr so stdout remains available for
future structured output.

| Condition                                        | Exit    |
| ------------------------------------------------ | ------- |
| SIGINT/SIGTERM and successful disconnect         | 0       |
| Invalid config                                   | Nonzero |
| Driver construction/connect/initial-read failure | Nonzero |
| Name or device lock contention                   | Nonzero |
| Repeated read failure                            | Nonzero |
| Unexpected control-loop failure                  | Nonzero |
| Driver disconnect failure                        | Nonzero |

Errors identify the phase and an operator recovery action where known. Expected
operational errors do not print tracebacks by default. Detailed tracebacks
remain available through debug logging.

### Discover

```bash
physicalai robot discover
physicalai robot discover --allow_remote
physicalai robot discover --json
```

Discovery delegates to the public transport discovery function, sorts records
by `(name, host)`, and never imports advertised `robot_class` values.

Human output includes at least name, robot class, host, and joint count. JSON
mode writes exactly one array to stdout. Logs and warnings go to stderr. Empty
discovery is successful and produces `[]` in JSON mode.

Targeted `--host`/`--name` discovery is not required for the serve redesign. If
retained for unicast-only networks, implement and review it as a separate
transport/API change. User-provided endpoints must use a structured serializer
such as `json.dumps`, not interpolation into JSON5.

### Help routing

Register `robot` as one built-in command group. Direct group help can use the
lightweight path:

```text
physicalai robot --help
```

Nested help must reach the real nested parser:

```text
physicalai robot serve --help
physicalai robot discover --help
```

Prefer generic host routing based on direct help versus nested arguments rather
than a robot-specific nested-command registry. Existing plugin help behavior
must remain unchanged and covered by regression tests.

## Security

`allow_remote=false` disables remote scouting and binds the owner to loopback.
This is the default for both configuration and CLI arguments.

`allow_remote=true` exposes an unauthenticated action endpoint. Any reachable
peer can command the physical robot. Documentation and command help must state
that remote serving requires an isolated robot-cell VLAN/firewall or configured
Zenoh ACL/TLS.

Discovery metadata excludes constructor kwargs, calibration paths, device
paths, and other construction secrets. Physical device identifiers remain
host-local diagnostics and are not advertised remotely.

Robot names are operator-enforced as unique across the reachable Zenoh network.
Host-local locks do not provide cross-host arbitration.

## Public API

Do not add a public `RobotServer` class for this change. There is one concrete
consumer of foreground serving, and the CLI can call a narrow internal owner
runtime. Adding a public lifecycle abstraction now would freeze semantics that
have not demonstrated a second use case.

Keep `SharedRobot`, `SharedRobotClient`, and `discover_robots` as the user-facing
transport APIs. The owner runtime and owner configuration may remain internal.
Reconsider a public serving API when a second concrete embedding use case
requires it.

## Migration From the PoC

Reimplement from the parent of commit
`2ed63a6ea2a53ea6548a149cc4a82cb1022ce64c` rather than repairing that commit in
place. Carry behavior and tests selectively.

Retain:

- The `physicalai robot serve` and `physicalai robot discover` experience.
- Flat configuration fields.
- Persistent explicit serve and numeric auto-spawn timeout.
- Structured termination reasons and exit-code requirements.
- Sorted human/JSON discovery behavior.
- Local-only defaults and the remote-network warning.
- Tests that express these contracts.

Redesign:

- Run explicit serving in the foreground instead of supervising a detached
  persistent worker.
- Extract one owner runtime shared by foreground and subprocess adapters.
- Keep validation within the transport-owned configuration boundary.
- Route nested help generically where possible without plugin regressions.
- Convert operational failures to concise CLI errors.

Separate or defer:

- Explicit-host attachment and targeted discovery for unicast-only networks.
- Any unrelated transport refactor.

Discard:

- Robot-specific private validator imports in CLI code.
- Persistent detached-child supervision for explicit serve.
- Machine-specific paths and remote-enabled defaults in examples.

## Implementation Sequence

### Phase 1: Owner runtime

1. Restore the pre-PoC transport to a compiling, tested baseline.
2. Extract `run_owner` without changing auto-spawn behavior.
3. Add structured exit reasons and cleanup result mapping.
4. Run focused owner config, worker, handshake, lock, and `SharedRobot` tests.

### Phase 2: Foreground serve

1. Add the flat serve parser and config-file support.
2. Call `run_owner` directly in the CLI process.
3. Adapt SIGINT/SIGTERM to the runtime shutdown event.
4. Add concise startup, readiness, contention, and shutdown reporting.
5. Add process-level CLI tests with the fake robot.

### Phase 3: Discovery

1. Add the discover parser over existing discovery behavior.
2. Add deterministic human and clean JSON formatting.
3. Verify that network metadata never drives imports.

### Phase 4: Documentation

1. Add a portable SO101 example with placeholders and `allow_remote: false`.
2. Update the sharing how-to and CLI reference.
3. Restore the full Zenoh transport design and align its lifecycle invariants
   with this design.

Keep these phases as separate reviewable commits where practical.

## Test Strategy

### Unit tests

- Owner config accepts internal `idle_timeout=None` and round-trips it over IPC.
- Invalid names, rates, and kwargs fail before driver construction.
- Requested shutdown and numeric idle timeout return success after disconnect.
- Repeated reads, loop exceptions, and disconnect failures return nonzero.
- Cleanup continues after each best-effort cleanup failure.
- Auto-spawn remains detached with its 10-second default timeout.
- CLI serving passes `idle_timeout=None` and does not instantiate `RobotOwner`.
- Expected CLI failures do not leak tracebacks.
- Discovery is sorted and JSON stdout is one clean array.
- Group help stays lightweight and nested help reaches the selected parser.
- Plugin help behavior remains unchanged.

### Process-level tests

Start `physicalai robot serve` with a fake driver, wait for readiness, attach a
`SharedRobot`, observe state, send a stationary action, terminate the CLI, and
verify:

- The foreground process exits.
- The fake driver disconnect path ran.
- Name and device locks can be reacquired.
- A clean interruption exits zero.
- An injected disconnect failure exits nonzero.
- No owner process remains after the CLI exits.

### Manual hardware validation

On an isolated robot network:

1. Serve an SO101 on host A.
2. Discover and attach from host B.
3. Exchange one safe stationary action.
4. Disconnect host B and confirm host A remains available.
5. Stop host A and verify safe disconnect and lock release.
6. Restart under a service manager and verify signal-driven shutdown.

## Acceptance Criteria

- No tracked source contains the PoC syntax corruption or duplicate API
  definitions.
- Explicit serve owns hardware in the foreground and cannot orphan a detached
  persistent worker through its normal architecture.
- Auto-spawn remains compatible and exits after its existing idle timeout.
- Every lifecycle invariant in this document has an adjacent test.
- Remote serving remains explicit and documented as unauthenticated.
- The example contains no developer-specific paths and defaults to local-only.
- Design and user documentation agree on lifecycle and exit semantics.
- Focused tests, CLI smoke checks, type checking, and repository hooks pass:

```bash
uv run pytest tests/unit/robot/transport tests/unit/cli
uv run physicalai --help
uv run physicalai robot --help
uv run physicalai robot serve --help
uv run physicalai robot discover --help
pyrefly check
prek run --all-files
```
