# Diagnostics API

`physicalai.diagnostics` runs passive Linux USB diagnostics for cameras and serial
devices. It only reads system state; it never opens serial ports or changes settings.

## Usage

```python
from physicalai.diagnostics import diagnose

report = diagnose()                    # all USB camera/serial devices
report = diagnose("/dev/video0")       # one device, with topology context

for finding in report.findings:
    print(finding.code, finding.severity, finding.devices, finding.message)

data = report.to_dict()                # JSON-serializable
```

`diagnose()` blocks on filesystem work. In async code:

```python
report = await asyncio.to_thread(diagnose)
```

Non-Linux hosts raise `NotImplementedError`. Invalid, ambiguous, or stale selectors raise
`ValueError`.

To collect and assess separately:

```python
from physicalai.diagnostics import assess, collect_inventory

inventory = collect_inventory()
findings = assess(inventory)           # pure, no I/O

device, endpoint = inventory.resolve("/dev/video0")
aliases = inventory.unique_aliases(device, endpoint)
route = inventory.ancestors(device)    # parent hub(s) up to the controller
```

## Report

| Model       | Main fields                                                                                          |
| ----------- | ---------------------------------------------------------------------------------------------------- |
| `Report`    | `platform`, `collected_at`, `inventory`, `findings`, `selected_device_id`, `selected_endpoint`       |
| `Inventory` | `devices`, `errors`, `notes`, `usbfs_memory_mb`                                                      |
| `Device`    | `id`, `name`, `kind` (`controller`/`hub`/`device`), `parent_id`, `speed_mbps`, `serial`, `endpoints` |
| `Endpoint`  | `path`, `kind` (`video`/`serial`), `role` (`capture`/`metadata`/`unknown`), `aliases`, `query_error` |
| `Finding`   | `code`, `severity` (`error`/`warning`/`info`), `devices`, `message`, `hint`, `hub_id`                |

- `report.has_errors` is true for access errors or incomplete collection.
- Use `finding.code`, not `message`, in code. Codes: `access`, `video_endpoint_missing`,
  `video_role_unknown`, `ambiguous_identity`, `shared_usb_link`.
- For `shared_usb_link`, `hub_id` identifies the shared hub in `inventory.devices`.
- `usbfs_memory_mb` is the system usbfs transfer-memory limit in MB (`0` is unlimited,
  `None` is unavailable). See
  [USB transfer memory](../how-to/cli/doctor.md#usb-transfer-memory).

## Notes

- Device IDs and endpoint numbers are valid only for one snapshot; refresh after
  reconnecting hardware.
- An empty inventory doesn't prove a device is absent.
- `to_dict()` includes raw device names and serials. Escape them in UIs and review
  before sharing.
- Only USB cameras and serial devices are checked. See
  [Diagnose USB Connections](../how-to/cli/doctor.md) for interpreting results.
