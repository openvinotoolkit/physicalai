# Diagnostics API

`physicalai.diagnostics` provides passive Linux USB diagnostics for CLI and application
consumers, including Physical AI Studio. It has no Studio, robot plugin, or camera SDK
dependency. Importing it does not enumerate hardware. Collection never opens serial
ports; video nodes may be queried read-only to distinguish image and metadata roles.
No camera settings, streams, robot state, or system configuration are changed.

## Collect and assess

```python
from physicalai.diagnostics import diagnose

report = diagnose()                       # collect and assess all USB camera/serial devices
selected = diagnose("/dev/video0")         # findings for one device; retain topology context
document = report.to_dict()               # detached, JSON-serializable dictionary

for finding in report.findings:
    print(finding.code, finding.severity, finding.devices, finding.hub_id)
```

`diagnose(selector=None) -> Report` is synchronous and performs blocking filesystem
and read-only ioctl work. Use a worker thread in an async application:

```python
import asyncio
from physicalai.diagnostics import diagnose

async def hardware_diagnostics() -> dict[str, object]:
    report = await asyncio.to_thread(diagnose)
    return report.to_dict()
```

The API does not print, colorize output, invoke a CLI, or depend on an event loop.
UI consumers should use typed fields and finding codes, not parse human messages.
Non-Linux hosts raise `NotImplementedError`; invalid, ambiguous, or stale selectors
raise `ValueError`. Collection failures are retained in the returned inventory.

For separate collection and pure assessment:

```python
from physicalai.diagnostics import assess, collect_inventory

inventory = collect_inventory()
findings = assess(inventory)               # no I/O, does not mutate inventory
device, endpoint = inventory.resolve("/dev/video0")
if endpoint is not None:
    aliases = inventory.unique_aliases(device, endpoint)
route = inventory.ancestors(device)        # immediate parent to controller
```

## Report data

The public models are dataclasses exported from `physicalai.diagnostics`:

| Model       | Main fields                                                                                                                                                            |
| ----------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Report`    | `schema_version` (1), `platform` (OS identifier; currently `linux`), `collected_at` (UTC), `inventory`, `findings`, `selected_device_id`, `selected_endpoint`          |
| `Inventory` | `devices`, `errors` (collection failures), `notes` (limitations), `usbfs_memory_mb` (system-wide transfer-memory ceiling)                                              |
| `Device`    | `id`, `name`, `kind` (`controller`, `hub`, `device`), `parent_id`, `speed_mbps`, USB vendor/product IDs, reported `serial`, `root_hub`, `video_interface`, `endpoints` |
| `Endpoint`  | `path`, `kind` (`video`, `serial`), `role` (`capture`, `metadata`, `unknown`), `readable`, `writable`, `aliases`, `query_error`                                        |
| `Alias`     | `path`, current `target`, `claimed` (whether udev supplied the claim)                                                                                                  |
| `Finding`   | `code`, `severity` (`error`, `warning`, `info`), affected `devices` (IDs), `message`, `hint`, optional `hub_id`                                                        |

`Report.has_errors` is true for access errors or incomplete collection. Warnings and
informational limitations alone do not make it true. An empty inventory does not
establish that hardware expected by a user is absent. Collection is not an atomic
snapshot: refresh after connection changes.

`Inventory.usbfs_memory_mb` is an optional integer read from
`/sys/module/usbcore/parameters/usbfs_memory_mb`: positive values are the system-wide
usbfs transfer-memory ceiling in MB, `0` means unlimited, and `None` (JSON `null`)
means unavailable. Missing, unreadable, or invalid values add an inventory note,
not an error. This additive schema-version-1 field defaults to `None` when omitted
and remains in selected-device reports because it applies to the host, not one
device. This field is Linux-specific; other platforms leave it unset and do not
display a usbfs setting. Consumers should tolerate additional report fields.
The terminal renderer only shows the usbfs section for unknown values or values
below 128 MB, including unlimited (`0`). This display heuristic does not affect
collection, assessment, or JSON/API data and does not establish a healthy limit.

The value is not current memory use, a bandwidth measurement, or evidence that
capture will fail. It applies to usbfs/libusb transfers, not native kernel V4L2
video buffers. Assessment does not infer a problem from the value or identify the
camera SDK's streaming backend. See
[USB transfer memory](../how-to/cli/doctor.md#usb-transfer-memory) for conditional
troubleshooting steps; the API never changes the setting.

Finding codes are `access`, `video_endpoint_missing`, `video_role_unknown`,
`ambiguous_identity`, and `shared_usb_link`. For shared links, `hub_id` references
the hub in `inventory.devices`; each camera appears once in `devices`, regardless
of its number of image/metadata nodes. Messages are explanatory text, not identifiers.

USB location IDs and endpoint numbers are snapshot handles, not persistent device
identities. `unique_aliases()` prefers an unambiguous device-bound alias when udev
claims and a unique serial support it, otherwise a port-bound path. A symlink winner
alone does not establish `by-id` uniqueness. No path or serial authenticates a device.
`resolve()` rejects a `by-id` selector for a known duplicate vendor/product/serial
identity even when only the winning symlink was collected. Direct endpoints, device
IDs, and unambiguous `by-path` selectors remain available.

The report includes all USB topology nodes as context, but camera/serial checks only;
non-USB cameras, network robots, camera modes, streaming, process ownership, and system
logs are outside scope. A 480 Mb/s camera is not automatically faulty. See
[Diagnose USB Connections](../how-to/cli/doctor.md) for interpretation.

`to_dict()` preserves raw device text and serials without terminal styling or
redaction. Treat device names and paths as untrusted display text (escape them in the
UI), and review reports before sharing. The terminal renderer escapes control characters
and defaults to plain text; the CLI opts into ANSI styling only for interactive terminals.

## Platform boundary

Linux is the only implemented collector. Windows and macOS remain unsupported;
accepting their OS identifiers in a report does not enable collection on those hosts.

OS-specific enumeration and queries belong in a collector module such as `linux.py`.
`collect_inventory()` selects and lazily imports the collector, whose `collect()`
method returns an `Inventory` with no live handles. A future platform adds its own
collector and dispatch branch without changing the assessment or CLI flow. Populate
the shared device, endpoint, and topology fields with available evidence and record
collection limitations in `notes` or `errors`; leave Linux-only usbfs data unset.
The current alias verification rules describe Linux udev `by-id`/`by-path` aliases;
another collector should omit aliases unless their uniqueness can be established.

`diagnose()` records the host OS identifier in `Report.platform`. The renderer uses
that field rather than the machine displaying the report, and only shows usbfs
settings for Linux reports. Pure checks and report models do not import OS APIs.
The command remains focused on hardware diagnostics; no general software or
environment-check framework is introduced.
