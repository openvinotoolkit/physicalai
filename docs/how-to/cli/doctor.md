# Diagnose USB Connections

Use `pai doctor` to check USB cameras and serial adapters before starting Runtime.
It does not connect robots, load plugins, change camera settings, or start streams.
`physicalai` and `pai` are equivalent commands. USB diagnostics currently support Linux.

```bash
pai doctor
pai doctor --tree
pai doctor /dev/video0
pai doctor /dev/ttyACM0
pai doctor --json
```

## Start with the negotiated link

The overview lists physical USB cameras and serial adapters, their useful endpoints,
and negotiated signaling speeds. Metadata nodes are not listed as extra cameras.
`--tree` shows their controller, root hub, and external hub ancestry, omitting
unrelated peripheral branches and non-USB internal camera pipelines.

- **480 Mb/s (USB 2)** can limit camera modes and concurrent capture. It is not
  automatically a fault: USB 2-only cameras cannot become USB 3 devices by changing
  ports. If your camera requires USB 3, check the entire port, hub, and cable path,
  then reconnect and compare its negotiated speed.
- **5000 Mb/s or higher (USB 3.x)** indicates a SuperSpeed link, not proof that
  streaming will work at every advertised mode.
- **12 Mb/s** is normal for many USB serial adapters. It is not UART baud or robot
  traffic, and the adapter does not establish a robot model or calibration state.

Multiple cameras sharing an external hub may compete for its uplink. Doctor reports
that relationship once for nested hubs carrying the same cameras, choosing the slowest
reported uplink. Different root ports are separate links, not one root-hub bandwidth
budget. A shared connection is not proof of congestion. If simultaneous capture fails,
try separate upstream connections and repeat the same workload.

Doctor reports signaling rates only. It does not estimate available bandwidth or
measure frame rate, payload throughput, or USB utilization.

## Inspect an endpoint

```bash
pai doctor /dev/video0
pai doctor /dev/video0 --tree
```

Inspection shows the USB route, peers on shared upstream hubs, read/write access,
and a unique connection alias when available. It also distinguishes image capture
from metadata-only video nodes; choose an image endpoint for a camera stream.
Unknown roles and failed queries are disclosed, not treated as verified capture.

Identical cameras sometimes report the same serial. Their `by-id` names can collide,
even though a symlink points to just one winner. Doctor reads udev claims to detect
this and refuses ambiguous alias selectors. Use distinct `by-path` selectors instead.
Port-bound paths follow a slot, not a physical unit; moving a hub or device can change
them. Backend support for these selectors varies. Inspect again after reconnecting.

If access fails, review the current user's device ACLs, groups, and vendor rules.
Do not run Runtime as root as a workaround. A successful access check does not prove
the device is idle or operational; doctor does not inspect process ownership.

## Reports and application use

`--json` writes the same versioned report returned by the
[Python diagnostics API](../../reference/diagnostics-api.md). It contains device
IDs, parent relationships, endpoints, findings, and collection limitations. A selected
device scopes findings but retains the full USB inventory as topology context.
`--tree` affects human formatting only.

Reports are **not redacted**: review serials, device names, and paths before sharing.
There is no setup wizard, configuration export, streaming benchmark, or watch mode.
Re-run doctor after changing connections; no system settings are tuned automatically.

Exit codes are `0` for a report without observed errors (warnings are allowed), `1`
for access or collection errors, and `2` for an invalid selector/argument or unsupported
OS. No devices found is not itself an error or proof that expected hardware is absent.
