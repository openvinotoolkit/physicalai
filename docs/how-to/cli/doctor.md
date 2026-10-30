# Diagnose Hardware Connections

Use `pai doctor` to inspect camera and robot connections without starting a policy
or moving hardware. `physicalai` and `pai` are equivalent commands.

```bash
pai doctor
pai devices --tree
pai devices inspect /dev/ttyUSB0
pai devices inspect /dev/video40
```

## Read throughput and shared links first

Human output starts with a compact throughput table and a USB branch tree.
`devices` uses the tree by default; `--flat` selects an ungrouped device table.
`doctor --verbose` expands routine checks, query errors, and raw endpoint details
that the default view summarizes.

```text
USB topology
└── USB host controller 0000:00:14.0
    └── USB root hub usb3 — USB 2.0 / 480 Mb/s root capability; root ports independent
        └── Port 4: USB2.0 Hub [usb:3-4] — USB 2.0 / 480 Mb/s link; 5 devices share uplink
            └── Port 1: USB2.0 HUB [usb:3-4.1] — USB 2.0 / 480 Mb/s link; 4 devices share uplink
                ├── Port 1: USB Serial — /dev/ttyUSB2; 12 Mb/s link
                ├── Port 2: Orbbec Gemini 305 — /dev/video32, /dev/video34 +3 image nodes
                ├── Port 3: Mouse — 12 Mb/s link
                └── Port 4: USB Serial — /dev/ttyUSB0; 12 Mb/s link
```

Inspecting a node shows its selected path and shared branches, then explicitly
lists the physical peers sharing **each upstream external hub**, including nested
devices. A camera's metadata and image nodes do not count as separate devices.
Other root-port branches are distinguished from the shared external uplink.

The throughput table separates:

- **Needs (payload):** estimated bytes/second for the configured uncompressed
  wire format, including reported padding. For example, 640×480 YUYV at 30 FPS
  needs about **18.43 MB/s**, not the larger decoded RGB array size. Configured
  settings can be idle; rows are not proof that every endpoint runs concurrently.
- **Getting (payload):** encoded frame bytes actually dequeued during an explicit
  camera test, alongside received FPS. Passive commands say **not measured** rather
  than opening streams or pretending the link rate is delivered throughput.
- **USB ceiling:** signaling speed divided by eight, expressed in the same MB/s
  units for comparison. A 480 Mb/s link has a 60 MB/s raw ceiling, **not 60 MB/s
  of available payload**: USB overhead, scheduling, and other devices also matter.

MJPEG is variable-size. Passive inspection does not estimate it from buffer
capacity (`sizeimage`) or decoded RGB bytes. Explicit tests estimate target demand
from the sampled encoded frame size at requested FPS, clearly marked as an
estimate. Reported payload excludes USB headers/retries and traffic lost before
the application dequeues frames. Shared-hub test summaries sum only streams from
the same concurrent trial, never individual trials taken at different times;
unmeasured peer traffic remains unknown.

Robot/UART throughput depends on baud, protocol, and workload. It remains unknown
without protocol measurements; the USB adapter's 12 Mb/s link is not UART baud.

### Color and emphasis

Interactive terminals use bold headings/rates and emphasize USB 2.0 in yellow and
USB 3.x in cyan. These identify link generation, not a proven fault or a bandwidth
reservation. Full-/low-speed device rates do not get mislabeled as USB 2.0 or 3.x.

```bash
pai devices --color always
pai doctor --color never
NO_COLOR=1 pai doctor
```

`--color auto` is the default: it requires a TTY, honors `NO_COLOR`, and disables
ANSI for `TERM=dumb`. Explicit `always` overrides auto detection/`NO_COLOR` for
color-aware pagers. JSON/NDJSON never includes presentation styling, even with
`--color always`; redirected human output stays plain in auto mode.

Passive commands enumerate OS evidence. On Linux they can open video nodes
read-only to query capabilities and formats, but they do not start streams, open
robot serial connections, ping robot endpoints, change USB settings, or authorize
devices. A visible USB serial adapter is a **robot candidate**, not proof of its
robot model or calibration.

## Read the assessment

Findings include evidence, a next action, and a verification step:

- **PASS:** the named check succeeded, such as current-user endpoint access. This
  does not establish that a camera can stream or a robot can execute actions.
- **INFO:** context or a limitation that is not itself a fault.
- **WARN:** an ambiguity or risk, such as multiple devices claiming one alias.
- **FAIL:** an observed problem, such as inaccessible required endpoints or a ZED
  Mini exposing HID without a visible stereo-video interface.
- **SKIP:** evidence was unavailable, unsupported, or could not be collected.

The USB tree separates host controllers, USB root hubs, external hubs, and device
endpoints. Root-hub speed figures are not one bandwidth budget shared by every
root port. Shared external-hub uplinks indicate risk, not measured utilization.

`usbfs_memory_mb=16` is informational. It limits usbfs buffer allocation, not USB
bandwidth or every V4L2 buffer. Increase it only with relevant workload/error
evidence and a reviewed, reversible system change. Doctor never tunes it.

Process ownership is limited by OS visibility. A visible opener is useful
evidence; no visible opener does not prove that a device is idle system-wide.

## Watch a replug or intermittent failure

```bash
pai doctor --watch
pai doctor --watch --duration 60 --interval 1
```

Linux subscribes to kernel device notifications before taking the baseline, then
reconciles OS state periodically. If notifications are unavailable, it falls back
to polling and discloses the limitation. A full unplug/replug between samples can
still be missed: notification bursts are coalesced, not treated as an authoritative
device inventory.

Watch reports arrivals, removals, node renumbering, speed changes, new system
messages, and findings that open or cease to be observed. A finding disappearing
because a device was removed does **not** prove that the underlying fault was fixed.
Anonymous devices returning at one location are not authenticated as the same unit.

Baseline kernel excerpts are marked historical. Subsequent journal records use
cursors rather than repeatedly treating old boot errors as new failures. The
journal collector is bounded to 100 records per read; very large bursts can exceed
that window. Missing journal permissions are disclosed; no automatic sudo is used.

## Inspect a configuration without executing it

```bash
pai doctor --config ./runtime.yaml
```

This checks explicit endpoint selectors against current evidence. It does not
import configured `class_path` targets, instantiate drivers, load policy weights,
or establish joint-order/calibration/policy compatibility. YAML/JSON inspection
rejects cycles and oversized/deep documents.

Prefer a verified device-bound identifier when a serial and stream identity are
unique. Otherwise use a physical-port binding, for example:

```text
/dev/serial/by-path/pci-0000:00:14.0-usb-0:4.1.4:1.0-port0
/dev/v4l/by-path/pci-0000:00:14.0-usb-0:4.1.2:1.6-video-index0
```

These are examples, not paths to copy to another machine. Use your inventory's
paths. A port binding follows a controller/hub/slot chain, not a physical unit;
replacing a serial-less robot with an identical one is not detectable from USB
metadata. Hub moves, firmware changes, or interface-layout changes can change paths.

Alias existence alone is not uniqueness: multiple endpoints can claim one `by-id`
name while only one wins the symlink. Doctor checks udev claims as well as current
targets. Without claim metadata, device-bound export based solely on a symlink
winner is refused. Recheck saved configs after topology/identity changes. This is
not device authentication or a replacement for driver-specific ownership locks.

## Discover hardware and export a Runtime configuration

```bash
pai discover --output ./runtime.yaml
pai discover --output ./runtime.json --format json
pai discover --components-only --output ./hardware.yaml
```

The interactive wizard lets you select and label devices, locate a connection by
safely unplugging/replugging one idle device, choose bindings and supported camera
modes, and review driver settings. It validates selections again before saving.
It asks before creating directories or replacing existing files. New files are
published atomically without overwriting a file that appeared during setup.

For a complete Runtime config, select a follower, cameras, and an explicit action
source. Policy setup accepts a local export containing `manifest.json`, optional
task text, and synchronous/asynchronous/RTC execution. RTC needs explicit policy
compatibility confirmation. Teleoperation requires a leader and confirmation of
matching joint order, units, and calibration; the exported mapping is identity.
Policy paths are absolute so a different working directory does not select another
export. Structural validation does not test inference or hardware compatibility.

The wizard prints a reviewed `pai run --config ...` next step but never runs it.
Running that command later can move hardware and change torque/startup state.
Review every driver option and the robot work area first.

Installed plugins are trusted local Python code, not sandboxed extensions.
`passive=True` is an operation contract: plugin imports, schemas, validators, and
export hooks must also avoid hardware side effects. Default `doctor`/`devices`
inventory does not load robot catalogs. Legacy probes without the named passive
argument are skipped during interactive discovery rather than invoked actively.

Robot recipe export requires an installed plugin's passive `config_exporter`
hook. The initial integration includes all three StarArm configurations. Other
plugins remain discoverable but need an exporter before the wizard can generate
their Runtime recipe. Nested/composite and network payloads need specialized
selection adapters; the current robot wizard selects single serial connections.
Camera export currently uses supported UVC image modes, not vendor-specific depth
pipelines. All other detected endpoints remain visible in the inventory.

**Components-only output is not a `pai run` document.** Its schema has
`schema_version: 1`, `kind: physicalai.components`, and named `robots` and `cameras`
mappings. Each value is an ordinary Runtime `Config` recipe. After reviewing the
trusted local file, a Python application can load a selected recipe:

```python
from pathlib import Path
import yaml
from physicalai.config import Config

document = yaml.safe_load(Path("hardware.yaml").read_text())
assert document["schema_version"] == 1
assert document["kind"] == "physicalai.components"
camera_recipe = Config.from_dict(document["cameras"]["wrist"])
# Instantiate only after reviewing the recipe; this can execute local Python code.
camera = camera_recipe.instantiate()
```

## Explicit camera streaming tests

Only run these when it is safe to open the selected cameras:

```bash
pai doctor --stream --camera /dev/video42 --camera /dev/video38 --duration 5
```

Repeat `--camera` for concurrency tests; a stable alias is preferable to a changing
video number. Linux V4L2 tests select advertised MJPEG/YUYV integer-FPS modes near
640×480 at 30 FPS, report the actual negotiated mode, and use the same requested
modes for individual and concurrent trials. The duration applies **per trial**,
plus startup/cleanup time. Large sets of cameras take longer overall.

Tests refuse visibly occupied endpoints, run SDK work in isolated processes, and
stop/reap stalled workers. They close streams and attempt to restore the previous
format/interval, reporting restoration failures. Killing a wedged worker cannot
guarantee restoration; inspect the device before another application uses it.
No robot driver, serial handshake, or visual identification runs in these tests.

Received FPS, frame timestamps, and frame gaps are host-observed measurements, not
USB utilization. A 20% comparison threshold highlights substantial concurrency
degradation. Low FPS already present alone is reported separately. Exposure,
firmware, variable compressed traffic, buffering, and CPU/decode work also matter.

## Reports, redaction, and exit codes

```bash
pai doctor --json --redact > doctor-report.json
pai devices --json --redact > inventory.json
pai doctor --watch --duration 60 --json --redact > events.ndjson
```

Snapshot JSON is one versioned document. Watch JSON is NDJSON: one versioned record
per line, beginning with a baseline and ending with a final assessment when the
duration completes. Worker/SDK logs do not mix with JSON stdout.

Redaction replaces known serials, the local hostname, and home path with consistent
pseudonyms within one report/watch. It is not a general secret scrubber: review
free-text logs/device names before publication. Redacted paths are not deployable
selectors, and redacted inventories cannot be used for configuration export.

| Code  | Meaning                                                                       |
| ----- | ----------------------------------------------------------------------------- |
| `0`   | No failed checks; warnings alone are allowed.                                 |
| `1`   | A failed check, or a warning with `--strict`.                                 |
| `2`   | Invalid arguments/configuration or command execution error.                   |
| `3`   | Essential inventory collection was incomplete and no failure took precedence. |
| `130` | Interrupted by the user.                                                      |

The overall stream command also includes passive findings: a missing unrelated
camera interface can make the report exit `1` even if selected streams succeed.

## Platform coverage

Linux provides USB topology, alias claims, V4L2 modes/access, visible process users,
kernel journal evidence, notifications, and explicit streaming tests.

macOS/Windows currently use portable serial enumeration and the existing camera
backend, with polling watch. Install the optional serial dependency in a uv project
with `uv add 'physicalai[doctor]'` (contributors: `uv sync --extra doctor`). Native
USB topology, persistent native selectors, log enrichment, and streaming tests on
those platforms remain follow-up work. Missing capabilities are reported rather
than silently treated as healthy. Native-platform hardware/CI validation is still
required; fake-platform tests do not establish native SDK compatibility.

Robot plugin availability has its own Python/vendor SDK requirements. The CLI's
portable inventory does not make an unsupported robot driver available on an OS.
