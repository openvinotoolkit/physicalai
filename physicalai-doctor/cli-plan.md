# Physical AI Doctor: CLI and implementation proposal

Status: proposed; the commands and architecture below are not implemented.
Updated: 2026-10-28.

Related documents: [hardware snapshot](hardware-summary.md) and
[implementation progress](progress.md).

## Goals

- Explain camera and robot connection problems without requiring users to
  interpret `lsusb`, sysfs, kernel logs, or vendor SDK errors themselves.
- Support Linux first, with a portable core and dedicated macOS and Windows
  strategies. Missing platform capabilities must be visible, not reported as
  successful checks.
- Discover hardware independently of whether it is busy, accessible, supported
  by an installed SDK, or ready to stream.
- Reuse camera discovery and installed robot plugins' `RobotProbe` contracts.
- Preserve reliable device or physical-port bindings through configuration export
  and Runtime device opening.
- Offer interactive setup that writes YAML or JSON to a user-selected location.

The initial implementation must not change kernel settings, install drivers,
create udev rules, open robot serial ports, or start camera streams automatically.
This folder is a design/handoff area, not a new Python distribution. Runtime
continues to own the `physicalai` command and its `pai` alias.

## Proposed commands

All examples use `pai`; `physicalai` exposes the same commands.

| Command | Purpose |
| --- | --- |
| `pai doctor` | Safe inventory, health checks, and actionable findings. |
| `pai doctor --json` | One versioned JSON diagnostic report. |
| `pai doctor --watch` | Initial assessment followed by hardware changes and new findings. |
| `pai doctor --watch --duration 60 --json` | Bounded watch with newline-delimited JSON events. |
| `pai doctor --config ./runtime.yaml` | Inspect configured selectors and modes without constructing or connecting hardware. |
| `pai doctor --stream --camera SELECTOR --duration 10` | Explicit, bounded camera streaming test. Repeat `--camera` to test concurrency. |
| `pai devices` | Flat inventory of physical devices and their endpoints. |
| `pai devices --tree` | Host controllers, root ports, hubs, links, and endpoints. |
| `pai devices --json` | Versioned machine-readable inventory. |
| `pai devices inspect SELECTOR` | Identity evidence, endpoint roles, formats, paths, permissions, and backend status. |
| `pai discover --output ./runtime.yaml` | Interactive robot/camera selection and complete Runtime configuration export. |
| `pai discover --output ./runtime.json --format json` | The same workflow with explicit JSON output. |
| `pai discover --components-only --output ./hardware.yaml` | Export named component recipes without requiring a policy or teleoperation source. |

`SELECTOR` accepts an inventory selection handle, an explicit device path, or a
typed identity selector. An inventory handle is session-local unless explicitly
documented otherwise; it must not become a persisted identity by accident.

### Output and exit behavior

- Human output groups findings as `PASS`, `INFO`, `WARN`, `FAIL`, and `SKIP`.
  Every finding includes a stable check ID, affected device/endpoint, evidence,
  recommended action, and a way to verify the result.
- Distinguish unsupported, missing dependency, permission denied, timed out, and
  failed checks. An irrelevant, uninstalled vendor SDK is not itself a fault.
- JSON includes schema version, collection time, platform, capabilities,
  inventory, findings, and collection errors. Do not mix log messages into stdout.
- Watch JSON is explicitly NDJSON, not a repeatedly rewritten JSON document.
- Proposed exit codes: `0` for completed checks without failures, `1` for failed
  health checks, `2` for usage/execution errors, and `3` for an inconclusive report
  because essential checks were unavailable. Warnings alone do not fail by
  default; `--strict` makes warnings fail. A failure takes precedence over a
  partial report. Optional skipped checks do not imply exit code `3`.
- On a bounded watch, evaluate the final assessment and report observed/resolved
  problems separately. Ctrl+C cleans up watchers and exits `130`.
- Default operations do not request elevation. Suggest a narrowly scoped manual
  collection command when required; never run the whole CLI as root automatically.

## Architecture: portable models and platform strategies

Keep collection, device discovery, identity resolution, checks, rendering, and
configuration export separate. Use protocols/strategies rather than scattering
OS conditionals through checks and CLI handlers.

Proposed Runtime-owned layout:

```text
src/physicalai/diagnostics/
  models.py             # Inventory, endpoints, links, findings, capabilities, events
  inventory.py          # Merge OS metadata with camera/plugin discovery
  identity.py           # Device-bound and port-bound selectors, ambiguity checks
  checks/               # Pure checks over collected evidence
  platforms/
    base.py             # Platform collector and event-source protocols
    linux.py
    macos.py
    windows.py
  watch.py              # Baselines, event correlation, inventory diffs
  discovery.py          # Selection/session logic, separate from terminal prompts
  export.py             # Passive recipes and Runtime configuration generation
src/physicalai/cli/
  doctor.py
  devices.py
  discover.py
```

Names are provisional. Register built-ins through the existing `SubcommandSpec`
contract. Keep top-level help free of SDK loading and hardware discovery.

### Strategy contract

A platform strategy reports its capabilities and collects:

- Physical devices, host controllers, hubs, upstream links, and child endpoints.
- Native identifiers, stable path candidates, reported serials, VID/PID, drivers,
  negotiated link speeds, and available power-management information.
- Endpoint access and available ownership evidence.
- Relevant system settings and bounded, device-related log excerpts.
- Hardware change events, or a documented polling fallback.

Return explicit collection errors and provenance instead of suppressing failures.
Inject the strategy and OS data sources in tests. Instantiate only the current
platform's strategy and lazy-load optional platform libraries.

| Platform | Initial data sources and intended behavior |
| --- | --- |
| Linux | sysfs, udev properties/links, V4L2 capability/format queries, serial enumeration; journal access with `dmesg` as an alternative. Prefer native device events, with bounded polling fallback. `usbfs_memory_mb` is Linux-specific. |
| macOS | Existing camera backend and serial enumeration, supplemented by IOKit/IORegistry USB location and identity information. Assess `system_profiler` as a fallback, not a frequent polling source. Add notification-based changes where available. |
| Windows | Existing camera backend and serial enumeration, supplemented by SetupAPI/Configuration Manager device instance IDs, container IDs, and location paths. Add device notifications where available. Do not treat COM numbers or camera indices as permanent identities. |

The shared core must work on all three platforms. Linux gets the first complete
topology/log/resource checks; macOS and Windows get basic discovery/export and
explicitly reported enrichment gaps before advanced native collectors land.
Represent unknown speed/topology as unknown. Do not equate macOS USB location IDs,
Windows device instance IDs, or serial library `location` values with guaranteed
cross-reboot or port-independent identity without validation.

### Shared data model

Separate the physical device from its endpoints. One Orbbec camera can own depth,
image, metadata, media, and USB-access endpoints; one robot can use multiple serial
or network connections. Merge backend reports by OS physical-device evidence,
not by display name or reported serial alone.

Each identifier records its source, binding scope, observed uniqueness, and
limitations. A reported serial and a udev-generated `ID_SERIAL` are not equivalent.
Keep endpoint interface/node identity in addition to physical-device identity.
Do not silently remove duplicate-serial devices or hide busy/inaccessible ones.

## Camera and robot discovery integration

### Cameras

- Reuse `DeviceInfo` and backend discovery without making an installed vendor SDK
  a prerequisite for OS-level inventory.
- The current `discover_all()` suppresses backend exceptions and defaults to
  filtering usable devices. Add structured discovery outcomes for diagnostics,
  preserving existing callers' behavior through a compatible adapter/API.
- Reuse Linux V4L2 enrichment, but inventory inaccessible nodes as well as
  successfully queried ones. Distinguish capture from metadata capabilities.
- Preserve stream choices even if the ordinary camera listing groups entries
  into one physical camera. The lowest node can be depth rather than RGB.
- Verify stable-selector opening through both V4L2 and the default OmniCamera
  path. Merely printing a `by-path` link does not make it usable by every backend.

### Robot plugins and `RobotProbe`

Use the existing `physicalai.studio.catalog_plugins` registrations and
`RobotCatalogDefinition.probe` through a lightweight integration layer. Do not
require the Studio application/server or introduce a second incompatible probe
protocol. Inventory still works when the plugin-contract package is absent.

Implement a portable `PortScanner` adapter over the collected serial candidates.
Cache each scan in an inventory refresh so multiple probes do not rescan ports
independently. Associate returned `SerialPortInfo` records with enriched endpoint
records rather than discarding location and identity evidence.

The existing probe contract does not establish safety or exact robot type:

- Several probes simply return the scanner's serial ports. A CH340 adapter is a
  candidate, not proof of StarArm versus SO-101 versus another robot model.
- `SerialPortInfo` currently has only `connection_string` and `serial_number`.
  Add compatible enrichment or an associated richer record; do not make platform
  location fields mandatory for existing plugins.
- Audit/classify each probe before running its `discover()` or `is_online()` in
  passive mode. Unknown third-party behavior requires explicit opt-in. Timeouts
  contain hangs, but do not make active hardware operations safe.
- Never call `identify()` automatically. Some implementations physically move a
  gripper; others are no-ops. Offer identification only as a separate, explicit
  action with the selected device, described behavior, and confirmation.
- Report plugin load failures and SDK availability independently. Lazy-load
  installed, operator-trusted plugins; hardware metadata never selects Python
  import paths. No implicit network probing or remote-owner attachment.

Keep `pai robot discover` as discovery of shared robot services. Physical
discovery belongs to `pai devices` and the interactive `pai discover` workflow.

## Identity and stable connection rules

Offer an explicit choice between:

1. **Device-bound:** follow validated serial/vendor identity, qualified by model
   and endpoint/stream identity. Refuse duplicates; never silently use the first.
2. **Port-bound:** follow the host controller and physical hub/port chain, with
   expected device and endpoint characteristics. Explain that this binds a
   location, not the physical unit plugged into it.

On Linux, prefer verified unique `by-id` aliases for appropriate device-bound
selectors and `by-path` aliases for port-bound selectors. Alias existence is not
proof of uniqueness: multiple endpoints can claim the same name while only one
target wins. Preserve intended bindings during reconnect and check alias claims
as well as their current resolved targets.

On macOS/Windows, use native selectors only after proving how the Runtime backend
resolves them to the live endpoint. Do not export Linux device paths there.
When no reliable binding is available, show the limitation and require explicit
acceptance of an ephemeral selector or refuse a stable export.

Persist selectors, not their current `/dev/videoN`, `/dev/ttyUSBN`, COM number, or
camera-index resolutions. Never fall back from a missing device identity to a
different same-model device automatically. Same-slot replacement by an identical
serial-less robot remains unverifiable and requires an operator warning.

## `doctor --watch`

1. Collect a baseline inventory and current assessment before subscribing to
   changes. Buffer/reconcile subscription events so startup cannot miss a replug.
2. Observe device arrivals/removals/changes and new relevant kernel/system events.
3. Debounce composite-device bursts and refresh only the affected inventory/checks
   when possible. Keep endpoint events linked to their physical device and hub.
4. Report connect, disconnect, node-renumber, speed-change, ambiguity, and
   finding-opened/resolved events. Include event sequence and timestamps.
5. Match reconnects by trustworthy evidence. For anonymous port-bound devices,
   report a replacement at a location, not a proven return of the same unit.

Example: a camera disconnects at `/dev/video40`, returns at `/dev/video44`, and
retains its selected physical path. Watch should explain that the binding remains
valid. If it returns on USB 2 instead of USB 3, report the speed change and affected
stream constraints. If only ZED HID appears, report partial enumeration.

Mark log history as historical; correlate live events using boot/session identity
and appropriate clocks/cursors. Do not treat all errors from a long-running boot
as caused by the latest plug-in. Fall back to polling when native notifications
are unavailable and explain loss of log/event visibility. Bound queues, subprocess
timeouts, and retained history; release all resources on termination.

Watch mode never starts streams, opens serial connections, changes settings, or
automatically runs visual identification. An interactive discovery session can
reuse the same watcher to guide one-device-at-a-time unplug/replug labeling.

## Interactive discovery and Runtime export

The wizard should:

1. Show all discovered cameras, serial candidates, and installed robot types,
   including inaccessible devices and backend failures.
2. Let users label devices by role, choose the actual robot model, and select
   device-bound versus port-bound connections. Offer passive unplug/replug
   guidance when identical devices cannot otherwise be distinguished.
3. Select camera endpoints, backend, wire format, resolution, and FPS from
   supported capabilities. Explain USB 2 limits, unsupported formats, and missing
   video interfaces. An optional preview/stream test requires separate consent.
4. Ask for driver-specific required inputs, such as calibration paths, baud rate,
   leader/follower role, or controller options. Use trusted plugin payload schemas
   and an audited passive recipe-export mapping; do not duplicate every driver's
   defaults in the CLI or call `robot_builder` blindly to construct a recipe.
5. In complete-runtime mode, choose one Runtime robot (or a supported composite
   robot), named cameras, control-loop rate, and an explicit action source. For
   policy execution, collect an existing exported policy and execution mode. For
   teleoperation, collect a supported leader and necessary mapping/calibration.
   Discovery alone cannot infer these choices. Do not invent a three-robot
   `RobotRuntime` schema for the three independently attached StarArms.
6. Show a final configuration, bindings, unresolved limitations, dependencies,
   and connection/startup effects. Writing configuration must not run it.
7. Validate passive structure and supported constructor arguments, then write
   YAML/JSON at the selected path. Re-scan before committing the selections to
   catch disconnects, changed aliases, or duplicate identities.

### Two distinct export products

**Complete runtime:** use the existing `pai run --config` shape, not a new doctor
execution format. The document contains `runtime.robot`, `runtime.action_source`,
`runtime.cameras`, and `runtime.fps`, with nested `class_path`/`init_args` recipes.
`PolicySource` owns its model, execution strategy, and action queue. Only write a
complete runtime when required fields are supplied; never label placeholders as
ready to run. Print the eventual `pai run --config <path>` command without
executing it or implying that hardware/policy compatibility has been proven.

**Components only:** write named robot and camera `physicalai.config.Config`
recipes in a clearly documented component bundle for Python integration or
assembly into a runtime. This is not directly accepted by `pai run` and must be
labeled accordingly. Its schema and any loader need explicit implementation and
tests; do not imply an existing bundle loader. A components-only export allows
all three independently attached robots to be represented without inventing a
multi-robot runtime.

Reuse Runtime's `physicalai.config` serialization rather than creating a parallel
object-construction system. Add passive plugin recipe-export support where
current builders cannot safely export without constructing hardware objects.
Selectors must resolve in the receiving backend before export is considered
supported. Advanced identity guards require Runtime integration, not metadata
that `pai run` silently ignores. Existing paths can be used where supported;
broader typed selector support is a separate implementation task.

### Files, trust, and validation

- Infer format from `.yaml`, `.yml`, or `.json`, or require `--format` when
  ambiguous. Reject conflicting format/extension choices explicitly.
- Ask for an output location if omitted in an interactive terminal. Fail clearly
  without a terminal rather than blocking on invisible prompts. Noninteractive
  discovery can use `pai devices --json`; scripted selection/export is later work.
- Confirm overwrites, handle races with exclusive/no-clobber semantics, and use
  an atomic replacement strategy for explicitly approved replacements. Do not
  leave a truncated final document on cancellation or failure.
- Validate destinations and parent-directory access. Ask before creating missing
  parents; do not write outside the operator-selected location implicitly.
- Define calibration/policy path semantics explicitly: normalize against the
  output location or export absolute paths initially. Loading from another
  working directory must not silently select different files.
- Keep report metadata and raw hardware evidence outside runnable configuration
  fields; an optional inventory sidecar is separately requested. Redacted support
  reports must not be presented as deployable device configurations.
- USB strings and network metadata are untrusted data. Only reviewed local
  registries/plugins select `class_path`; export never executes device-supplied
  Python paths. Respect existing config depth and dynamic-construction rules.
- Default validation parses and checks recipes without instantiating drivers,
  loading policy weights, connecting devices, or sending actions. Test actual
  config construction using fake devices and require explicit consent for any
  real-hardware validation.

## Diagnostic checks and active testing

Initial checks cover missing/duplicate serials, colliding aliases, ephemeral
selectors, endpoint roles, missing drivers/SDKs, access, busy-device evidence,
negotiated speeds and USB 2 bottlenecks, missing interfaces, and relevant logs.

Collect Linux `usbfs_memory_mb` as a limit, not a preallocated buffer or measured
usage. A default value of 16 MiB is not evidence of exhaustion. Account for the
capture backend: usbfs/libusb transfer allocation and kernel V4L2 buffers are
different. Suggested changes must include scope, verification, rollback, and
explicit operator approval; no automatic tuning in the first release.

Topology identifies shared upstream links, not actual utilization. Estimate
uncompressed wire traffic from negotiated format/stride/FPS, aggregate at shared
links, and label assumptions. MJPEG traffic is variable. Advertised power values
are not current measurements, and runtime suspension of an idle camera is not
automatically a fault. Errors such as `ENOSPC` need context rather than a blanket
diagnosis of disk space, bandwidth, or memory exhaustion.

Later, explicit streaming diagnostics test selected cameras individually and
together with the same requested modes. Measure achieved FPS, frame gaps,
timeouts, negotiated mode, and correlated log events. Bound duration, clean up
devices, respect busy endpoints, and separate transport, decode, and application
limits. Do not include robot handshake/motion in camera streaming tests.

## Delivery and acceptance

Deliver a safe Linux snapshot CLI first, then selectors and passive plugin
integration, then watch mode, interactive export, and active camera tests. Develop
portable models and cross-platform basic enumeration from the start; advanced
macOS/Windows topology and event enrichment can follow Linux functionality.

Use the [progress checklist](progress.md) for milestones and acceptance criteria.
Synthetic fixtures cover the [observed hardware cases](hardware-summary.md);
hardware-dependent tests are opt-in. Runtime and Studio should consume the same
inventory/identity service. Run Python tests with `uv run pytest`, then relevant
repo hooks and type checks when implementation begins.
