# Physical AI Doctor: implementation progress

Updated: 2026-10-28.

**Current state: initial Runtime integration implemented.** The CLI, shared
inventory/assessment API, Linux collection and notifications, interactive export,
passive probe contract, and bounded camera tests are present. Native macOS/Windows
enrichment, broader robot exporters, and advanced Runtime identity/ownership guards
remain follow-up work. Studio and the SDK migration are intentionally separate.

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
- [x] Review/approve command names, default checks, exit codes, and export products.
- [x] Confirm shared contract ownership and compatibility with Studio and plugins.
- [x] Confirm the Linux-first platform matrix and optional portable serial dependency.

## 1. Portable inventory and evidence model

- [x] Define typed devices, endpoints, host controllers, links, identifiers, findings,
      capabilities, collection errors, and event records.
- [x] Define schema versions, identifier scope/provenance, and observed-uniqueness rules.
- [x] Define injectable platform collector and event-source protocols and factory.
- [x] Keep CLI help and generic imports free of device scans and heavy SDK imports.
- [x] Add baseline portable serial/camera enumeration with explicit backend outcomes.
- [x] Preserve inaccessible/busy devices and distinguish missing SDKs from no devices.
- [x] Add fixture tests for physical-device grouping and multiple endpoints.

Acceptance: basic inventory runs without optional SDKs and never collapses
distinct same-model/duplicate-serial devices or hides collection failures.

## 2. Linux collection, checks, and snapshot CLI

- [x] Collect sysfs/udev topology, controller and port paths, drivers, and link speeds.
- [x] Collect current symlink targets and all alias claims to detect collisions.
- [x] Query V4L2 capture/metadata capabilities and formats without starting streams.
- [x] Collect endpoint permissions and available process-ownership evidence.
- [x] Collect `usbfs_memory_mb`, power-management state, and bounded relevant logs.
- [x] Support readable journal access when `dmesg` is restricted; never auto-elevate.
- [x] Implement pure identity, endpoint, access, link, and resource checks.
- [x] Register `pai devices`, inspect/tree/JSON output, and snapshot `pai doctor`.
- [x] Add config-selector inspection without instantiating configured objects.
- [x] Test human/JSON output, schema stability, partial reports, and exit codes.

Acceptance: synthetic versions of all applicable snapshot cases produce evidence
and useful remedies, without treating 16 MiB, autosuspend, or old errors as faults.

## 3. Stable selectors and robot plugin integration

- [x] Define device-bound versus port-bound selectors and endpoint qualification.
- [x] Refuse missing/duplicate inventory/export identities instead of choosing a device.
- [x] Preserve supported selectors in saved config; resolve paths at connection time.
- [x] Support verified Linux stable paths through V4L2 and default OmniCamera opening.
- [ ] Define native macOS/Windows selector resolution and explicit unstable fallbacks.
- [x] Load existing robot catalog registrations without requiring the Studio app.
- [x] Add keyword-only `passive=False` to probe discovery/status; pass `True`
      explicitly, skip legacy signatures, and avoid active scanner refreshes.
- [x] Supply a cached portable `PortScanner` adapter and enrich probe candidates.
- [x] Distinguish a serial candidate from a verified/user-selected robot model.
- [x] Never invoke visual `identify()`, robot handshakes, or motion in this tooling.
- [x] Extend current SDK contracts compatibly; leave migration/Studio changes separate.
- [x] Test alias collisions, duplicate identities, replacements, and re-enumeration.
- [ ] Add persistent expected-VID/PID/serial guards and alias-independent shared-owner
      identities across Runtime drivers; exported paths do not provide authentication
      or rewrite existing drivers' `device_ids`/locking behavior.

Acceptance: three serial-less CH340 adapters remain distinct, Orbbec streams
remain selectable, and no passive scan opens a robot port or moves hardware.

## 4. Watch mode

- [x] Subscribe before baseline collection and reconcile queued Linux notifications.
- [x] Implement Linux device notifications and journal/log cursors where available.
- [x] Add bounded polling fallback and capability/visibility notices.
- [x] Debounce composite-device events; bound retained history and work queues.
- [x] Link controller/hub/device/endpoint evidence and distinguish historical logs.
- [x] Emit arrivals, removals, node/speed/address changes, and finding lifecycles.
- [x] Avoid claiming identity continuity for anonymous same-port replacements.
- [x] Implement duration, Ctrl+C cleanup, human output, and versioned NDJSON output.
- [ ] Test replug bursts, permission loss, removed sysfs entries, startup races,
      clock/session changes, and lost notification sources.

Acceptance: a simulated node renumber preserves a selected port binding, a
speed downgrade becomes a finding, and partial ZED enumeration is explained.

## 5. Interactive robot/camera discovery and export

- [x] Separate testable discovery/session logic from terminal prompts.
- [x] Select and name robots/cameras; guide passive unplug/replug labeling.
- [x] Select models, roles, bindings, UVC streams, backends, and supported modes.
- [x] Collect supported driver/controller settings through trusted plugin schemas.
- [x] Add passive recipe export for all three StarArm variants without constructing drivers.
- [x] Offer complete Runtime config versus explicitly labeled component bundles.
- [x] Define/test the component bundle schema and documented Python integration.
- [x] Require an explicit action source and required inputs for complete runtime export.
- [x] Serialize YAML/JSON using Runtime-owned `physicalai.config` conventions.
- [x] Validate supported recipe shapes/selectors without connecting hardware.
- [x] Handle user-selected output paths, formats, parent creation, overwrite races,
      atomic approved replacements, cancellation, and noninteractive terminals.
- [x] Use absolute policy paths and revalidate inventory before saving.
- [x] Keep evidence/sidecars separate from executable config fields.
- [x] Print a reviewed `pai run --config` next step without executing it.
- [x] Test YAML/JSON round trips, fake Runtime construction, and path handling from
      another working directory; ensure no missing-field placeholders are called runnable.
- [ ] Add exporters and calibration-file path semantics for other first-party robots.
- [ ] Add nested/composite/network robot selection and vendor depth-camera export adapters.

Acceptance: a user can save a complete, structurally valid runtime configuration
to a chosen path, or explicitly save reusable hardware recipes without a policy.
No stream, serial connection, model execution, or motion occurs during export.

## 6. macOS and Windows validation/enrichment

- [ ] Verify basic discovery/export and import safety on native CI for both systems.
- [ ] Add/test macOS IOKit/IORegistry identity/topology enrichment and event support.
- [ ] Add/test Windows SetupAPI/Configuration Manager identity/topology and events.
- [ ] Validate native selector resolution across reconnect and reboot where possible.
- [x] Report absent log/speed/topology features as unsupported or unavailable.
- [x] Document vendor SDK/driver limitations separately from doctor platform support.
- [ ] Run opt-in native-hardware smoke tests and record remaining platform gaps.

Acceptance: unsupported Linux-specific checks do not crash or falsely pass on
macOS/Windows, and exported selectors work through the actual capture/robot backend.

## 7. Explicit active camera diagnostics

- [x] Require selected cameras and explicit streaming consent, with bounded duration.
- [x] Test individually and concurrently using identical requested modes.
- [x] Measure negotiated mode, received/timestamp FPS, gaps, timeouts, and errors.
- [x] Collect fresh post-test system evidence without claiming causation.
- [x] Separate variable compressed traffic and decode/CPU limits from USB constraints.
- [x] Refuse visibly occupied devices; stop/reap workers and report restoration failures.
- [x] Save, restore, and verify previous camera wire formats after explicit tests.
- [x] Keep robot handshake and motion outside camera streaming diagnostics.
- [x] Record real-hardware results only after operator-approved experiments.
- [ ] Add measured/estimated payload attribution and CPU profiling beyond rate comparisons.

## 8. Supportability and release readiness

- [x] Add correlation-preserving redaction for reports, not deployable configs.
- [ ] Bound subprocess output/timeouts and audit plugin/config trust boundaries.
- [ ] Add fixture-based unit tests, CLI tests, and opt-in hardware integration tests.
- [x] Run targeted/full unit tests, relevant `prek` hooks, and scoped/full type checks
      (full-project typing remains blocked by unrelated existing errors, recorded below).
- [x] Document permissions, missing tools, partial reports, watch, and export workflows.
- [x] Document port-binding limitations, multi-camera troubleshooting, and platform matrix.
- [ ] Re-run the snapshot's acceptance cases on real hardware with explicit approval.
- [ ] Update user documentation, release notes, and minimal-install smoke coverage.

## Decisions and evidence log

### Validation commands and current results

- Root unit suite, with optional test dependencies in a temporary uv environment
  and local SO-101/SDK source paths: **1,832 passed, 21 skipped**.
  Download/integration/smoke markers were excluded. Initial failures without
  SO-101 were missing-test-dependency errors, not feature regressions.
- SDK and StarArm package suites: **123 passed**; no robot connections or motion.
- Scoped diagnostics/CLI type check: **0 errors**. The full-project checker still
  reports existing inference/fixture/SDK schema typing issues; **0 error locations
  in the new diagnostics/CLI feature**. Do not describe the full-project type
  check as passing.
- Ruff on changed Runtime diagnostics/capture/CLI and plugin source: passing.
- Markdownlint, Prettier, lockfile consistency, and `git diff --check`: passing.
- Approved real-camera comparison and restoration evidence are recorded in the
  hardware summary. The user removed the ZED; it is excluded from further live work.

Commands used for the broader unit check:

```bash
PYTHONPATH=src:packages/physicalai-studio-plugin/src:packages/physicalai-so101-plugin/src \
  uv run --no-sync --with eclipse-zenoh==1.9.0 --with msgpack==1.2.1 \
  --with feetech-servo-sdk==1.0.0 python -m pytest tests/unit -q \
  -m 'not requires_download and not integration and not ov_smoke and not mujoco_smoke'
```

No Studio files, kernel/udev settings, robot ports, or existing user Runtime recipe
files were changed. No commit was created; existing staged/unstaged work is preserved.

| Date       | Entry                                                                                                    | Evidence / next action                                                                  |
| ---------- | -------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| 2026-10-28 | Initial hardware inspection and proposal documented.                                                     | See the linked snapshot; no streaming or robot-protocol test performed.                 |
| 2026-10-28 | Added cross-platform strategies, existing probe integration, watch, and interactive export requirements. | Documentation only; review interfaces and first-release scope before implementation.    |
| 2026-10-28 | User selected existing SDK contract extension and Runtime-first implementation.                          | No SDK namespace migration or Studio repository changes.                                |
| 2026-10-28 | User requested optional operation-level `passive` arguments instead of catalog safety metadata.          | Implemented on `discover()`/`is_online()`; no `passive_discovery` field.                |
| 2026-10-28 | User approved bounded Hiwonder/Orbbec streaming tests.                                                   | Both streamed; original wire modes restored; no robot ports or kernel settings changed. |
