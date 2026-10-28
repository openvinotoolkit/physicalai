# Physical AI Doctor: implementation progress

Updated: 2026-10-28.

**Current state: documentation only.** No doctor, device inventory, watch, or
interactive export command has been implemented as part of this work. Existing
camera discovery, robot probes, and Runtime configuration are foundations, not
completed milestones for this feature.

Design: [CLI and architecture proposal](cli-plan.md).
Evidence: [connected hardware snapshot](hardware-summary.md).

Use `[x]` only for completed work with evidence. Add test commands, relevant
changes, limitations, and follow-up decisions under the corresponding milestone.
Do not mark hardware acceptance complete based only on fixture tests.

## 0. Proposal and scope

- [x] Record the proposed CLI commands and read-only safety boundaries.
- [x] Record the inspected Linux hardware, topology, aliases, and limitations.
- [x] Include Linux-first macOS/Windows strategy-based architecture.
- [x] Include `RobotProbe` reuse, watch mode, and interactive YAML/JSON export.
- [ ] Review/approve command names, default checks, exit codes, and export products.
- [ ] Confirm shared contract ownership and compatibility with Studio and plugins.
- [ ] Confirm the first-release platform support matrix and optional dependencies.

## 1. Portable inventory and evidence model

- [ ] Define typed devices, endpoints, host controllers, links, identifiers, findings,
  capabilities, collection errors, and event records.
- [ ] Define schema versions, identifier scope/provenance, and observed-uniqueness rules.
- [ ] Define injectable platform collector and event-source protocols and factory.
- [ ] Keep CLI help and generic imports free of device scans and heavy SDK imports.
- [ ] Add baseline portable serial/camera enumeration with explicit backend outcomes.
- [ ] Preserve inaccessible/busy devices and distinguish missing SDKs from no devices.
- [ ] Add fixture tests for physical-device grouping and multiple endpoints.

Acceptance: basic inventory runs without optional SDKs and never collapses
distinct same-model/duplicate-serial devices or hides collection failures.

## 2. Linux collection, checks, and snapshot CLI

- [ ] Collect sysfs/udev topology, controller and port paths, drivers, and link speeds.
- [ ] Collect current symlink targets and all alias claims to detect collisions.
- [ ] Query V4L2 capture/metadata capabilities and formats without starting streams.
- [ ] Collect endpoint permissions and available process-ownership evidence.
- [ ] Collect `usbfs_memory_mb`, power-management state, and bounded relevant logs.
- [ ] Support readable journal access when `dmesg` is restricted; never auto-elevate.
- [ ] Implement pure identity, endpoint, access, link, and resource checks.
- [ ] Register `pai devices`, inspect/tree/JSON output, and snapshot `pai doctor`.
- [ ] Add config-selector inspection without instantiating configured objects.
- [ ] Test human/JSON output, schema stability, partial reports, and exit codes.

Acceptance: synthetic versions of all applicable snapshot cases produce evidence
and useful remedies, without treating 16 MiB, autosuspend, or old errors as faults.

## 3. Stable selectors and robot plugin integration

- [ ] Define device-bound versus port-bound selectors and endpoint qualification.
- [ ] Refuse missing/duplicate identities instead of silently choosing a device.
- [ ] Preserve selectors in saved config; resolve them at connection time.
- [ ] Support verified Linux stable paths through V4L2 and default OmniCamera opening.
- [ ] Define native macOS/Windows selector resolution and explicit unstable fallbacks.
- [ ] Load existing robot catalog registrations without requiring the Studio app.
- [ ] Audit first-party `RobotProbe.discover()`/`is_online()` safety and add an opt-in
  policy for unaudited third-party probes.
- [ ] Supply a cached portable `PortScanner` adapter and enrich probe candidates.
- [ ] Distinguish a serial candidate from a verified/user-selected robot model.
- [ ] Keep visual `identify()` and any handshake/motion explicitly opt-in.
- [ ] Plan compatible contract enrichment and a shared Studio consumer path.
- [ ] Test alias collisions, duplicate identities, replacements, and re-enumeration.

Acceptance: three serial-less CH340 adapters remain distinct, Orbbec streams
remain selectable, and no passive scan opens a robot port or moves hardware.

## 4. Watch mode

- [ ] Add race-safe baseline/subscription reconciliation and event buffering.
- [ ] Implement Linux device notifications and journal/log cursors where available.
- [ ] Add bounded polling fallback and capability/visibility notices.
- [ ] Debounce composite-device events; bound retained history and work queues.
- [ ] Correlate controller/hub/device/endpoint events and distinguish historical logs.
- [ ] Emit arrivals, removals, node/speed changes, and opened/resolved findings.
- [ ] Avoid claiming identity continuity for anonymous same-port replacements.
- [ ] Implement duration, Ctrl+C cleanup, human output, and versioned NDJSON output.
- [ ] Test replug bursts, permission loss, removed sysfs entries, startup races,
  clock/session changes, and lost notification sources.

Acceptance: a simulated node renumber preserves a selected port binding, a
speed downgrade becomes a finding, and partial ZED enumeration is explained.

## 5. Interactive robot/camera discovery and export

- [ ] Separate testable discovery/session logic from terminal prompts.
- [ ] Select and name robots/cameras; guide passive unplug/replug labeling.
- [ ] Select models, roles, bindings, camera streams, backends, and supported modes.
- [ ] Collect required calibration/controller settings through trusted plugin schemas.
- [ ] Add audited passive plugin recipe export without constructing hardware drivers.
- [ ] Offer complete Runtime config versus explicitly labeled component bundles.
- [ ] Define/test the component bundle schema and documented Python integration.
- [ ] Require an explicit action source and other required inputs for complete runtime
  export; support one robot or an existing supported composite robot recipe.
- [ ] Serialize YAML/JSON using Runtime-owned `physicalai.config` conventions.
- [ ] Validate recipes and backend selector support without connecting/instantiating.
- [ ] Handle user-selected output paths, formats, parent creation, overwrite races,
  atomic approved replacements, cancellation, and noninteractive terminals.
- [ ] Define calibration/policy path semantics and revalidate inventory before saving.
- [ ] Keep evidence/sidecars separate from executable config fields.
- [ ] Print a reviewed `pai run --config` next step without executing it.
- [ ] Test YAML/JSON round trips, fake Runtime construction, and path handling from
  another working directory; ensure no missing-field placeholders are called runnable.

Acceptance: a user can save a complete, structurally valid runtime configuration
to a chosen path, or explicitly save reusable hardware recipes without a policy.
No stream, serial connection, model execution, or motion occurs during export.

## 6. macOS and Windows validation/enrichment

- [ ] Verify basic discovery/export and import safety on native CI for both systems.
- [ ] Add/test macOS IOKit/IORegistry identity/topology enrichment and event support.
- [ ] Add/test Windows SetupAPI/Configuration Manager identity/topology and events.
- [ ] Validate native selector resolution across reconnect and reboot where possible.
- [ ] Report absent log/speed/topology features as unsupported or unavailable.
- [ ] Document vendor SDK/driver limitations separately from doctor platform support.
- [ ] Run opt-in native-hardware smoke tests and record remaining platform gaps.

Acceptance: unsupported Linux-specific checks do not crash or falsely pass on
macOS/Windows, and exported selectors work through the actual capture/robot backend.

## 7. Explicit active camera diagnostics

- [ ] Require selected cameras and explicit streaming consent, with bounded duration.
- [ ] Test individually and concurrently using identical requested modes.
- [ ] Measure negotiated mode, achieved FPS, frame gaps, timeouts, and errors.
- [ ] Correlate failures with links and fresh system logs without claiming causation.
- [ ] Separate variable compressed traffic and decode/CPU limits from USB constraints.
- [ ] Respect occupied devices and guarantee cleanup on errors/cancellation.
- [ ] Keep robot handshake and motion outside camera streaming diagnostics.
- [ ] Record real-hardware results only after operator-approved experiments.

## 8. Supportability and release readiness

- [ ] Add correlation-preserving redaction for reports, not deployable configs.
- [ ] Bound subprocess output/timeouts and audit plugin/config trust boundaries.
- [ ] Add fixture-based unit tests, CLI tests, and opt-in hardware integration tests.
- [ ] Run targeted/full `uv run pytest`, relevant `prek` hooks, and `pyrefly check`.
- [ ] Document permissions, missing tools, partial reports, watch, and export workflows.
- [ ] Document port-binding limitations, multi-camera troubleshooting, and platform matrix.
- [ ] Re-run the snapshot's acceptance cases on real hardware with explicit approval.
- [ ] Update user documentation, release notes, and minimal-install smoke coverage.

## Decisions and evidence log

| Date | Entry | Evidence / next action |
| --- | --- | --- |
| 2026-10-28 | Initial hardware inspection and proposal documented. | See the linked snapshot; no streaming or robot-protocol test performed. |
| 2026-10-28 | Added cross-platform strategies, existing probe integration, watch, and interactive export requirements. | Documentation only; review interfaces and first-release scope before implementation. |
