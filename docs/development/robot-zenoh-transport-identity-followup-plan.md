# Plan: Implement Shared Robot Naming and Ownership Follow-up

**Status:** Ready for implementation

Sources of truth:

- [`robot-zenoh-transport-design.md`](robot-zenoh-transport-design.md)
- [`robot-zenoh-transport-identity-followup.md`](robot-zenoh-transport-identity-followup.md)

The original design remains authoritative except where the accepted follow-up
explicitly supersedes it.

## Goal

Replace constructor-kwarg identity guessing and hardcoded robot branches with:

- a required logical `name`;
- arbitrary importable `robot_class` construction;
- owner-reported `Robot.device_ids`;
- host-local name and device locks;
- `/metadata` with protocol compatibility;
- `allow_remote=False` as the secure default; and
- deterministic loopback rendezvous over ports 20000-59999.

Preserve the existing owner loop, msgpack state/action codec, latest-wins semantics,
QoS, idle shutdown, and `READY` / `ERROR:{json}` handshake unless this plan names a
change.

## Target API

Create or attach:

```python
robot = SharedRobot(
    name="left-arm",
    robot_class=SO101,
    robot_kwargs={
        "port": "/dev/ttyACM0",
        "calibration": "calibration.json",
    },
    allow_remote=False,
    rate_hz=100.0,
)
```

Attach only:

```python
robot = SharedRobot.attach("left-arm")
```

Configuration may supply a trusted importable string:

```yaml
class_path: physicalai.robot.SharedRobot
init_args:
  name: left-arm
  robot_class: physicalai.robot.SO101
  robot_kwargs:
    port: /dev/ttyACM0
    calibration: calibration.json
```

The old `robot_type`, `robot_id`, `from_owner()`, and `port`/`ip` identity heuristics
are removed rather than retained as silent compatibility paths. Robot constructor
arguments are accepted only through `robot_kwargs`; arbitrary flat extras are not
merged into them.

## Phase 1: Migrate the core Robot contract

### Implementation

1. Add `device_ids: tuple[str, ...]` to
   `src/physicalai/robot/interface.py` with the accepted pre-connect stability
   contract.
2. Implement first-party identities:
   - `SO101`: one canonical serial resource derived from its configured port;
   - `WidowXAI`: one canonical TCP resource derived from its configured IP;
   - `BimanualWidowXAI`: sorted, deduplicated union of both child robots' IDs.
3. Make `SharedRobot.device_ids` return `()` because the subscriber owns no physical
   hardware itself.
4. Update all protocol test doubles and transport fake robots. `FakeRobot` accepts
   explicit test `device_ids` so single-device, multi-device, and virtual cases are
   testable without constructor-key conventions.

Identity strings should be scheme-qualified (`serial:...`, `tcp:...`) to prevent a
serial path and network address from colliding as plain text. Identity derivation must
not connect to hardware.

### Tests

- Extend `tests/unit/robot/test_protocol.py` with a missing-`device_ids` negative case.
- Add first-party tests for stable single-device IDs and bimanual composition.
- Verify equivalent serial aliases normalize consistently where the OS can resolve
  them.
- Verify `SharedRobot` remains structurally conformant before connection.

### Focused validation

```bash
uv run pytest tests/unit/robot/test_protocol.py tests/unit/robot/so101 tests/unit/robot/trossen
```

## Phase 2: Replace identity and construction primitives

### Protocol constant

Add one internal module-level `ROBOT_TRANSPORT_PROTOCOL_VERSION = 1` constant used by
owner, subscriber, metadata, and tests. Its docstring or adjacent comment must state:

- it gates owner/subscriber wire compatibility before actions are published;
- bump for incompatible payload, required-field, or semantic changes; and
- do not bump for additive optional fields, refactors, driver changes, or package
  releases that preserve compatibility.

### Names and keys

Refactor `src/physicalai/robot/transport/_ids.py` to contain only transport naming:

- validate `name` as one non-empty Zenoh-safe segment;
- build `physicalai/robot/{name}`;
- build `/state`, `/action`, and `/metadata` keys;
- expose `physicalai/robot/*/metadata` discovery selector; and
- hash the full prefix into ports 20000-59999.

Delete `derive_device_id()` and connection-derived `derive_robot_id()`.

### Owner construction config

Replace transport `RobotSpec` with private `RobotOwnerConfig`, preferably in
`_owner_config.py`. It contains:

- `name`;
- normalized importable `robot_class` path;
- JSON-serializable `robot_kwargs`;
- `allow_remote`;
- validated positive finite `rate_hz` (default 100 Hz);
- idle/startup timeout fields needed by the worker.

Class normalization rules:

- preserve an explicit string path;
- derive `cls.__module__ + "." + cls.__qualname__` for a class object;
- reject local classes (`<locals>`), malformed paths, and non-class imports;
- resolve nested qualified names by importing the longest module prefix and traversing
  attributes; and
- instantiate only in the owner subprocess.

Extract the dotted-object import primitive into a dependency-free shared helper under
`src/physicalai/` rather than adding another local `import_module` + `getattr`
implementation. The helper supports nested qualified names by importing the longest
valid module prefix and traversing remaining attributes. Reuse it from:

- `RobotOwnerConfig` / the owner worker;
- `inference.component_factory._import_class`; and
- `inference.component_factory.ComponentRegistry.get_class`.

Keep policy and validation at each caller: the helper imports an object but does not
decide whether a path is trusted or whether the result must be a class. The adapter
registry's lazy module import and camera worker's `module:attribute` test hook have
different contracts and do not migrate in this change.

Treat class paths and constructor kwargs as trusted local application/config input.
Never import a path received from `/metadata` or any Zenoh payload. Document this at
the resolver to satisfy `docs/development/security.md` rules 4, 9, and 11.

### Tests

- Rewrite `test_ids.py` for name validation, new keys, and the 20000-59999 range.
- Replace `test_spec.py` with `test_owner_config.py` covering JSON round-trip, class
  objects, public string paths, arbitrary plugin-style classes, nested qualnames,
  invalid/local classes, non-JSON kwargs, and rate validation.
- Add `tests/unit/test_import_utils.py` for the shared importer and rerun
  `tests/unit/inference/test_manifest.py`, which covers `ComponentRegistry` and
  component instantiation.
- Test that no `port` or `ip` key receives special treatment.

### Focused validation

```bash
uv run pytest tests/unit/test_import_utils.py tests/unit/inference/test_manifest.py tests/unit/robot/transport/test_ids.py tests/unit/robot/transport/test_owner_config.py
```

## Phase 3: Generalize host-local locking and errors

### Locks

Refactor `_lock.py` from one raw device lock into a generic namespaced lock primitive:

- `name:{name}` for the service-name lock;
- `device:{device_id}` for each physical resource;
- SHA-256 digest for the filename under the existing user-scoped cache directory;
- `flock(LOCK_EX | LOCK_NB)` held by an open file descriptor; and
- diagnostic JSON contents containing lock kind, raw identity, owner name, and PID.

Add a small acquisition helper that:

1. acquires the name lock;
2. acquires sorted, deduplicated device locks;
3. releases partial acquisitions in reverse order on failure; and
4. supports an empty device list for virtual robots.

Do not delete lock files on release. The kernel lock, not file existence, determines
ownership and is crash-safe.

### Errors

Replace `RobotIdConflict` with:

```text
RobotError
├── RobotNotConnectedError
└── RobotTransportError
    ├── RobotNameConflict
    ├── RobotDeviceAlreadyOwned
    └── RobotProtocolMismatch
```

Define stable worker error codes for name lock contention, device lock contention,
construction failure, connection failure, endpoint collision, and unexpected startup
failure. Only the three accepted conflict/protocol conditions map to narrow public
classes; other codes remain `RobotTransportError`.

### Tests

- Rewrite `test_lock.py` for hashed paths, diagnostic contents, name/device namespaces,
  multi-lock ordering, partial rollback, empty IDs, and subprocess crash/release.
- Test that equal raw strings in name and device namespaces do not share a lock.
- Test structured error-code-to-exception mapping.

### Focused validation

```bash
uv run pytest tests/unit/robot/transport/test_lock.py tests/unit/robot/transport/test_errors.py
```

## Phase 4: Enforce transport scope and rendezvous

Refactor `_session.py` around `name`, endpoint role, and `allow_remote`.

### Local-only mode (`allow_remote=False`)

Both owner and subscriber sessions explicitly set:

```text
mode = peer
scouting/multicast/enabled = false
scouting/gossip/enabled = false
```

- Owner listens only on `tcp/127.0.0.1:{derived_port}`.
- Subscriber connects only to the same loopback endpoint.
- Do not add wildcard listeners or configured remote peers.

### Network mode (`allow_remote=True`)

- Retain peer mode and the D20 QoS decisions.
- Enable the configured remote discovery/listen behavior explicitly.
- Preserve the trusted robot-cell LAN warning; do not imply authentication.

The spawning caller fixes the owner's scope for its lifetime. A later attacher's
`allow_remote` value configures only that subscriber's session and cannot reconfigure
the running owner.

### Endpoint collisions

- Hash the full robot prefix into ports 20000-59999.
- Treat owner listen/bind failure as structured `endpoint_collision` startup failure.
- Include the endpoint and recovery guidance: choose another name or configure a local
  Zenoh router.
- Do not probe alternative ports because subscribers must derive the same endpoint.

### Tests

- Unit-test exact local-only and remote Zenoh configuration inserts.
- Verify local owner endpoint is loopback and network owner endpoint is explicit.
- Force a bind collision and verify actionable `RobotTransportError` propagation.
- Add a Linux integration test using a network namespace or second host fixture to
  prove a default local-only owner is unreachable off-host. Skip with an explicit
  reason when the environment cannot create the namespace; keep configuration/link
  assertions as the always-run check.
- Verify `SharedRobot.attach(name)` cannot discover a remote owner unless
  `allow_remote=True` is supplied.

### Focused validation

```bash
uv run pytest tests/unit/robot/transport/test_session.py tests/unit/robot/transport/test_ids.py
```

## Phase 5: Migrate the owner lifecycle

### Parent process

Refactor `_owner.py` to accept only `RobotOwnerConfig` and transport startup settings.
It no longer receives a parent-derived device ID. Serialize the config over stdin and
parse structured worker errors from stdout.

### Worker process

Refactor `_owner_worker.py` startup in this order:

1. deserialize and validate `RobotOwnerConfig`;
2. import and instantiate `robot_class` exactly once;
3. require the instance to satisfy `Robot` and read sorted/deduplicated `device_ids`;
4. acquire the name lock;
5. acquire all device locks;
6. call `driver.connect()`;
7. read the first observation;
8. open the scoped Zenoh session and declare endpoints;
9. publish `/metadata`; and
10. signal `READY`.

The worker returns resolved candidate `device_ids` in structured name-lock failures so
the parent can distinguish a same-device race from a different-device name conflict.
It returns lock diagnostics for device ownership failures without publishing constructor
kwargs.

Build metadata once after the first observation:

```python
{
    "protocol_version": ROBOT_TRANSPORT_PROTOCOL_VERSION,
    "name": name,
    "robot_class": normalized_class_path,
    "device_ids": sorted_device_ids,
    "host": socket.gethostname(),
    "joint_names": list(driver.joint_names),
    "num_joints": len(driver.joint_names),
    "state_dim": int(first_obs.state.shape[0]),
}
```

Keep constructor kwargs, calibration paths, credentials, and arbitrary config out of
metadata and errors.

Use 100 Hz when no override is supplied. Preserve write-first ordering, D20 publisher
QoS, matching-status idle shutdown, and safe `driver.disconnect()` cleanup. Release
device locks and then the name lock in `finally`; process death remains the ultimate
lock cleanup.

### Tests

- Rewrite owner handshake tests for arbitrary class paths and structured phases.
- Cover construction failure separately from connection failure.
- Cover name lock race with matching and differing `device_ids`.
- Cover device lock conflict under another name and multi-device conflict.
- Verify virtual robots skip device locks.
- Verify metadata has exactly the public fields and protocol version.
- Verify owner crash releases all locks.

### Focused validation

```bash
uv run pytest tests/unit/robot/transport/test_owner_handshake.py tests/unit/robot/transport/test_lock.py
```

## Phase 6: Migrate SharedRobot and discovery

Refactor `_shared_robot.py` to the target API.

### Construction

- Require `name` for normal construction.
- Require `robot_class` and `robot_kwargs` for create-or-attach.
- Add `SharedRobot.attach(name, *, allow_remote=False, ...)` for attach-only mode.
- Remove `from_owner()`, `robot_type`, `robot_id`, implicit ID derivation, and
  parent-side device identity.
- Expose `name`, `metadata`, and `device_ids == ()` properties.

### Connect behavior

1. Open a session with the requested scope.
2. Probe `physicalai/robot/{name}/metadata`.
3. If found, validate `protocol_version` and metadata consistency before declaring
   `/action`. Require `num_joints == len(joint_names)`, non-empty unique joint names,
   and positive `state_dim`.
4. When create-or-attach supplied `robot_class`, compare its normalized string with
   metadata and log a warning on mismatch. Do not fail and do not import the
   owner-advertised path.
5. If absent in attach-only mode, raise `RobotTransportError`.
6. If absent in create-or-attach mode, start the owner.
7. On name-lock startup failure, bounded re-probe the same metadata key:
   - matching candidate and winner `device_ids`: attach;
   - different IDs: raise `RobotNameConflict`.
8. Map device-lock contention to `RobotDeviceAlreadyOwned`.
9. Attach RingChannel(1) state subscriber and D20 action publisher, then await the
   first state as today.

Keep `robot_class` metadata diagnostic only. String mismatch is useful evidence of a
wrong name but is not proof of incompatibility because public re-exports, wrappers,
and subclasses can preserve the contract. An unsupported protocol version raises
`RobotProtocolMismatch` before the action publisher exists.

### Discovery

Update `discover_robots()` to query `physicalai/robot/*/metadata`, return `name`, and
honor `allow_remote=False` by default. Network discovery requires explicit
`allow_remote=True` or a caller-supplied network-enabled session.

### Tests

- Rewrite construction tests around name validation and attach-only behavior.
- Test create, existing-name attach, idempotent connect, and subscriber-only disconnect.
- Test same-name matching-device race attaches and differing-device race conflicts.
- Test existing-name behavior ignores construction kwargs, warns on class mismatch,
  and does not import the owner-advertised path.
- Test malformed metadata dimensions or joint names fail before action publication.
- Test protocol rejection occurs before action publisher declaration.
- Test local-only versus remote attach/discovery scope.
- Preserve observation, action, Ring(1), idle shutdown, and latency tests.

### Focused validation

```bash
uv run pytest tests/unit/robot/transport/test_shared_robot.py tests/unit/robot/transport/test_latency.py
```

## Phase 7: Update exports, examples, and documentation

1. Export new exception classes from `physicalai.robot` and remove
   `RobotIdConflict`.
2. Update `examples/runtime/teleop_runtime.yaml` to use `name`, `robot_class`, and
   `robot_kwargs`; keep `allow_remote` omitted to exercise the secure default.
3. Update `examples/so101/measure_transport_latency.py` to the new API.
4. Update `docs/how-to/runtime/share-a-robot.md`:
   - local-only default and explicit `allow_remote=True`;
   - the spawning caller fixes owner scope for its lifetime;
   - create-or-attach and attach-only examples;
   - trusted-LAN warning for remote mode;
   - host-local-only ownership guarantee; and
   - endpoint-collision recovery.
5. Update `docs/development/robot-zenoh-transport-plan.md` and
   `robot-zenoh-transport-handoff.md` to point to this completed migration and remove
   the superseded `device_id_from_kwargs`, `robot_type`, `/meta`, and derived-ID work.
6. Update API/reference docs and MkDocs links if they expose the old constructor or
   key names.
7. Update security documentation to state that local-only is the default and D18
   applies only after explicit remote opt-in.

### Focused validation

```bash
uv run pytest tests/unit/cli tests/unit/runtime tests/unit/robot/transport
```

Validate example configuration instantiation through the real jsonargparse parser,
not only YAML parsing.

## Phase 8: Full verification

Run in order:

```bash
uv run pytest tests/unit/robot
uv run pytest
pyrefly check
prek run --all-files
```

Manual checks:

1. Start a default local-only SO101 owner; attach from a second local process.
2. Confirm no non-loopback Zenoh listener/link exists.
3. Confirm a second local name targeting the same device raises
   `RobotDeviceAlreadyOwned` before hardware connect.
4. Confirm two concurrent starts for one name converge on one owner.
5. Confirm owner `SIGKILL` releases name and device locks.
6. Start with `allow_remote=True` on an isolated test LAN and attach from another host.
7. Measure p99 action latency at 100 Hz under best-effort/drop/express QoS.

## Completion criteria

- No transport code derives identity from `port`, `ip`, or arbitrary constructor keys.
- No hardcoded SO101/WidowXAI construction or rate branches remain.
- Every `Robot` implementation exposes pre-connect `device_ids`.
- Default sessions are unreachable off-host.
- Name and device ownership are enforced host-locally with crash-safe locks.
- `/metadata` contains the accepted schema and no constructor secrets.
- Subscribers reject incompatible protocol versions before publishing actions.
- All old API, examples, tests, and docs are migrated together.
- Full tests, type checking, and repository hooks pass.
