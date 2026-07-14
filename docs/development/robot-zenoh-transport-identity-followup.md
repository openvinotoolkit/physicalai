# Follow-up Design: Shared Robot Naming and Ownership

**Status:** Accepted follow-up to
[`robot-zenoh-transport-design.md`](robot-zenoh-transport-design.md).

Read this document together with the original design. Where they conflict, this
follow-up supersedes the original identity, construction, metadata, and locking
decisions listed below. All other original decisions remain in force.

## Scope and superseded decisions

The original implementation derives identity by inspecting constructor kwargs named
`port` or `ip`. That does not support arbitrary or composite robots and couples the
transport to specific constructor signatures.

This follow-up supersedes:

- **D1**, only where it says no `Robot` protocol change is required;
- **D3**, renaming `/meta` to `/metadata`;
- **D12**, replacing connection-derived `robot_id` with a required logical `name`;
- **D14**, deriving local lock identities from owner-constructed
  `driver.device_ids` and limiting the guarantee to one host;
- **D15**, replacing hardcoded `robot_type` construction with an importable arbitrary
  robot class plus serializable kwargs;
- **D17**, renaming `/meta` probes to `/metadata` while retaining the handshake and
  bounded race re-probe; and
- **D18**, making network exposure an explicit opt-in rather than the default while
  retaining its trusted-LAN requirement for network mode; and
- **D19**, replacing ID-conflict terminology with name/device conflict terminology.

## Public API

Creating a `SharedRobot` means ensure that the named service exists, then attach:

```python
robot = SharedRobot(
    name="left-arm",
    robot_class=SO101,
    allow_remote=False,
    robot_kwargs={
        "port": "/dev/ttyACM0",
        "calibration": "calibration.json",
    },
)
```

Attaching without a construction recipe is explicit:

```python
robot = SharedRobot.attach("left-arm")
```

`name` and `robot_kwargs` remain separate, so a robot constructor may itself accept a
`name` kwarg without ambiguity. Grouped `robot_kwargs` is the canonical configuration
and subprocess representation. `SharedRobot` does not merge arbitrary flat keyword
arguments into the robot constructor.

If the named service already exists, the caller attaches to it. Construction inputs
are a fallback recipe used only when the name does not exist. Raw constructor kwargs
are neither published nor compared.

`allow_remote=False` is the default. Cross-host access requires an explicit
`allow_remote=True` opt-in because any peer that can reach `/action` can move the robot.

## Arbitrary robot construction

The transport accepts an importable robot class rather than a hardcoded robot type.
Manual callers may pass the class object. Configuration may pass its stable public
path, for example `physicalai.robot.SO101`. Before subprocess startup, the transport
normalizes the value to an importable dotted path. An explicit string path is preserved;
a class object uses `cls.__module__ + "." + cls.__qualname__` because Python does not
retain the re-export path through which a class was imported. The owner imports that
path and calls the class with `robot_kwargs`.

The private subprocess payload is `RobotOwnerConfig`. It is internal IPC, not a
user-facing configuration object. The name avoids collision with
`physicalai.inference.manifest.RobotSpec`.

Local classes, lambdas, live instances, and non-serializable constructor arguments
cannot be auto-spawned. The class must be importable and all `robot_kwargs` must be
JSON-serializable.

## Robot interface identity

`device_ids` becomes part of the core `Robot` protocol:

```python
@property
def device_ids(self) -> tuple[str, ...]:
    """Canonical identities of all physical devices exclusively owned.

    Available after construction and before connect(). Stable across processes,
    restarts, and equivalent connection aliases.
    """
    ...
```

Single-device robots return one ID. Composite robots return every constituent device
ID. This describes physical ownership, not the Zenoh transport name.

## Zenoh keys and names

`name` is a globally unique logical name within the reachable Zenoh deployment and is
validated as one safe key segment. The robot class is deliberately absent from keys:

```text
physicalai/robot/{name}/state
physicalai/robot/{name}/action
physicalai/robot/{name}/metadata
```

For example:

```text
physicalai/robot/left-arm/state
physicalai/robot/left-arm/action
physicalai/robot/left-arm/metadata
```

This keeps `SharedRobot.attach("left-arm")` sufficient and keeps the logical name
stable if an implementation class changes. `metadata` is preferred over `meta` because
clarity in a public protocol outweighs saving four characters.

## Transport scope and secure default

`SharedRobot` ships with two transport scopes:

| `allow_remote`    | Reachability               | Security contract                                           |
| ----------------- | -------------------------- | ----------------------------------------------------------- |
| `False` (default) | Processes on the same host | No Zenoh transport is exposed to the LAN                    |
| `True`            | Reachable Zenoh peers      | Trusted robot-cell LAN; deployer provides network isolation |

Local-only mode must enforce locality in the Zenoh session configuration; binding the
owner to loopback without disabling discovery is insufficient. Both owner and
subscriber sessions set:

```text
mode = peer
scouting/multicast/enabled = false
scouting/gossip/enabled = false
```

The owner listens only on the deterministic
`tcp/127.0.0.1:{derived_port}` endpoint. Subscribers connect only to that loopback
endpoint. They do not listen on wildcard addresses or connect to configured remote
peers. Tests inspect active links or use a second host/network namespace to verify that
the local-only owner is unreachable off-host.

Network mode is explicit (`allow_remote=True`). It may enable multicast scouting and a
non-loopback listen endpoint according to deployment configuration. D18's warning
still applies in this mode: there is no application-level authorization on `/action`,
so VLAN/firewall isolation or Zenoh ACL/TLS is the deployer's responsibility.

Transport scope is part of attachment semantics. `SharedRobot.attach("left-arm")`
defaults to local-only and cannot discover a network owner. Attaching to a network
owner requires `SharedRobot.attach("left-arm", allow_remote=True)`.

The caller that spawns an owner fixes its transport scope for that owner's lifetime.
Later attachers cannot widen or narrow the running owner's reachability; their
`allow_remote` value only controls how their own session searches for and reaches it.

`allow_remote` describes the security capability being granted. It is preferred over
`networked` because local mode still uses Zenoh over loopback TCP.

## Deterministic local rendezvous

Local-only operation must work when multicast is unavailable, so owner and subscriber
derive the same loopback TCP port from the full robot name. The hash maps into the
unprivileged range **20000-59999** (40,000 ports), replacing the original 1,000-port
range.

Hash collisions remain possible. The owner must treat a listen bind failure as an
endpoint collision and return a structured `RobotTransportError` that identifies the
derived endpoint and instructs the user to choose another `name` or configure a local
Zenoh router. It must not probe alternate ports because an independent subscriber
could not derive the selected port.

This is sufficient for the expected small number of robot owners per host. A fixed
local `zenohd` router is the documented scalable alternative if a deployment needs to
eliminate per-name endpoint collisions.

## Metadata and protocol compatibility

The `/metadata` record is:

```python
{
    "protocol_version": 1,
    "name": "left-arm",
    "robot_class": "physicalai.robot.SO101",
    "device_ids": ["serial:/dev/serial/by-id/usb-FTDI_1234"],
    "host": "robot-cell-01",
    "joint_names": [...],
    "num_joints": 6,
    "state_dim": 6,
}
```

`robot_class` uses the normalized importable path from `RobotOwnerConfig`. Callers that
need a stable public path supply it explicitly as a string; class objects use their
defining module and qualified name. It identifies arbitrary implementations without a
transport registry and is diagnostic; exact class equality does not by itself define
wire compatibility.

`device_ids` are sorted and deduplicated. Publishing them is an intentional diagnostic
choice under the trusted robot-cell network assumption. They may expose serial
numbers, device paths, or IP addresses to every reachable Zenoh peer. Metadata must
not include full constructor kwargs, calibration paths or contents, credentials,
tokens, or arbitrary configuration.

`protocol_version` versions the robot transport wire contract, not the robot class or
package release. Physical AI Runtime owns it. Maintainers bump one shared
`ROBOT_TRANSPORT_PROTOCOL_VERSION` constant in the same change that introduces an
incompatible wire change.

The constant must have a docstring or immediately adjacent comment stating that:

1. it validates owner/subscriber compatibility before actions are published;
2. it is bumped for backward-incompatible payload, required-field, or semantic
   changes; and
3. it is not bumped for additive optional fields, internal refactors, robot-driver
   changes, or package releases that preserve wire compatibility.

Subscribers reject unsupported versions before declaring the action publisher.

## Create-or-attach behavior

| Existing name              | Candidate devices                 | Result                                |
| -------------------------- | --------------------------------- | ------------------------------------- |
| No                         | Unlocked                          | Spawn owner, then attach              |
| Yes                        | Existing owner is compatible      | Attach; construction recipe is unused |
| No                         | Locked locally under another name | Raise device-owned error              |
| Concurrent same-name spawn | Winner has same devices           | Losing caller re-probes and attaches  |
| Concurrent same-name spawn | Winner has different devices      | Raise name-conflict error             |

When an owner existed before the call, the parent cannot compare its `device_ids` with
the candidate without constructing a throwaway robot. The name is therefore
authoritative. Attach validates protocol compatibility and metadata consistency. When
the caller supplied a construction recipe, it also compares the already-normalized
caller and owner `robot_class` strings and logs a warning on mismatch. It does not
fail: public re-export paths, wrappers, subclasses, or an implementation change may
preserve the service contract. The subscriber must not import the owner-advertised
path because network metadata is untrusted.

Metadata consistency checks require `num_joints == len(joint_names)`, non-empty unique
joint names, and positive `state_dim`. Comparing joint names or dimensions against the
caller's intended robot is not possible without adding a separate expected-contract
input or constructing the robot, so those values are not identity checks.

## Host-local lock lifecycle

The owner constructs the driver once and reads `driver.device_ids`. It then acquires a
host-local name lock, followed by all local device locks, and only then calls
`driver.connect()`:

```text
construct driver -> read device_ids -> acquire name lock -> acquire device locks
-> connect -> publish
```

The name lock is necessary because a Zenoh query followed by declaration is not
atomic. Two same-host owners targeting different devices can both observe no metadata,
acquire different device locks, and declare publishers/queryables on the same keys.
Zenoh permits multiple entities on one key; a TCP bind failure is incidental and must
not serve as arbitration. The losing parent performs the existing bounded metadata
re-probe. The worker includes its already-resolved `device_ids` in the structured lock
error, allowing the parent to attach when they match the winner or raise a name
conflict when they differ. Construction may validate configuration or read files, but
it must not connect to hardware; hardware access starts only after all locks are held.

IDs are sorted and deduplicated before locking. Locks are acquired in sorted order and
released in reverse order. If an acquisition fails, all locks acquired by that attempt
are released. Lock ordering is always name lock first, then sorted device locks.

An empty `device_ids` tuple is valid for a virtual robot with no exclusively owned
physical resources. It holds only the name lock. A robot controlling an external
exclusive resource must identify that resource and must not use an empty tuple merely
to bypass locking.

Locks use POSIX `flock`. The lock belongs to an open file descriptor, so the OS releases
it when the owner process exits, including exceptions, `SIGKILL`, or a crash. A stale
file may remain, but it is not a stale held lock. `driver.disconnect()` still runs in
`finally`, but lock release does not depend on successful disconnect.

Lock filenames are hashes of canonical device IDs. File contents may record the raw
device ID, owner name, and PID for local diagnostics.

## Cross-host ownership is deferred

`flock` protects only processes on one host. Two hosts can both connect to a network
robot such as WidowXAI at the same IP. Neither `/metadata` nor Zenoh liveliness tokens
provide atomic mutual exclusion: concurrent candidates can both observe no owner,
multiple sessions may declare the same token, and partitions can hide an owner.

Phase 1 policy is therefore:

1. guarantee exclusive owner processes only within one host using `flock`;
2. require deployments to designate one owner host per physical robot, enforced by
   topology, firewall policy, or operator configuration; and
3. make no claim of network-wide hardware exclusivity.

Distributed leases are explicitly deferred. If cross-host automatic ownership becomes
a firm requirement, design a separate coordinator or hardware-fencing component. A
coordinator would need serialized grants, lease expiry and renewal, and ideally a
fencing epoch accepted by the hardware or an exclusive gateway. Keep local `flock` as
defense in depth if such a coordinator is later added.

## Startup errors

The existing `READY` / `ERROR:{json}` stdout handshake carries owner failures back to
`SharedRobot.connect()`.

- `construction_failed`: importing or instantiating `robot_class` failed, including
  invalid kwargs or calibration loading.
- `connection_failed`: construction and locking succeeded, but `driver.connect()`
  failed.

These are error phases and may both map to `RobotTransportError` while preserving the
worker details. Conflicts that support different recovery remain typed, notably name
conflict and device already owned.

The public hierarchy is deliberately small:

```text
RobotError
├── RobotNotConnectedError
└── RobotTransportError
    ├── RobotNameConflict
    ├── RobotDeviceAlreadyOwned
    └── RobotProtocolMismatch
```

- `RobotNameConflict` means a concurrent owner claimed the same name for different
  `device_ids`.
- `RobotDeviceAlreadyOwned` means another local owner name holds at least one requested
  device lock.
- `RobotProtocolMismatch` means an existing owner uses an unsupported transport
  protocol version.

Import, construction, connection, handshake, timeout, codec, and endpoint failures
remain `RobotTransportError`. The structured worker payload carries a stable error
code and phase for diagnostics; they do not need separate public exception classes
until callers have distinct recovery behavior.

## Owner loop rate

The transport default owner loop rate is **100 Hz** for every robot class. It bounds
action pickup to approximately 10 ms when the driver keeps pace and is the existing
fallback for unknown robot types. A slower blocking driver naturally runs below the
target without accumulating scheduling debt. Callers override it with `rate_hz` when
hardware measurements justify another value; `rate_hz` must be finite and greater
than zero.

This replaces the hardcoded SO101/WidowXAI rate table. Per-class defaults are deferred
unless measurements show that repeated overrides are a real problem.

## Decisions established by this follow-up

- Public logical identity is `name`; internal key prefixes may still be called robot
  IDs.
- `SharedRobot(...)` is create-or-attach; `SharedRobot.attach(name)` is attach-only.
- Arbitrary robots use an importable `robot_class` plus serializable `robot_kwargs`.
- The private subprocess descriptor is `RobotOwnerConfig`.
- `device_ids` is required by `Robot` and is available before `connect()`.
- Keys use `physicalai/robot/{name}` and `/metadata`.
- `allow_remote=False` is the secure default; network reachability requires explicit
  opt-in and retains the trusted-LAN requirement.
- Metadata includes `protocol_version`, `robot_class`, and diagnostic `device_ids`, but
  not constructor kwargs.
- Device locking uses crash-safe, host-local `flock` over all constituent devices.
- A host-local name lock serializes same-name owner creation before device locking.
- Empty `device_ids` is valid only for a robot with no exclusive physical resource.
- The default owner loop rate is 100 Hz, overridable with `rate_hz`.
- A supplied `robot_class` mismatch logs a warning but does not block attachment;
  metadata shape fields are validated for internal consistency.
- Public conflict exceptions are `RobotNameConflict`, `RobotDeviceAlreadyOwned`, and
  `RobotProtocolMismatch`, all derived from `RobotTransportError`.
- Local rendezvous hashes the name into ports 20000-59999; bind collisions fail
  explicitly and require another name or a local router.
- Cross-host ownership is deferred; deployments designate one owner host per robot.
