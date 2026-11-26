# Diagnose USB Connections

Use `pai doctor` to check USB cameras and serial adapters before starting Runtime.
It does not connect robots, load plugins, change camera settings, or start streams.
`physicalai` and `pai` are equivalent commands. USB diagnostics currently support Linux.
The current scope is hardware connection diagnostics, not general software or environment checks.
Windows and macOS collection are not implemented.

```bash
pai doctor
pai doctor --tree
pai doctor /dev/video0
pai doctor /dev/ttyACM0
pai doctor --json
pai doctor --no-ansi
```

Interactive terminals use bold headings, yellow USB 2 links, cyan USB 3.x links,
and colored finding labels. Link colors identify USB generations, not proven faults.
Use `--no-ansi` to disable both color and bold text. Formatting is also disabled for
redirected output, `TERM=dumb`, or a nonempty `NO_COLOR` environment variable.
JSON and the Python API report never contain presentation styling.

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

## USB transfer memory

Doctor also shows `usbcore.usbfs_memory_mb`, Linux's **system-wide memory ceiling
for outstanding usbfs transfers**, read from
`/sys/module/usbcore/parameters/usbfs_memory_mb`. This is a host setting shared by
usbfs users, not a per-camera allowance, memory-usage measurement, or USB bandwidth
limit. Increasing it does not reserve that amount of memory immediately. A value
of `0` means unlimited; `unavailable` means doctor could not read a valid value.

This setting matters for camera SDKs that stream through **usbfs/libusb**. It does
not govern video buffers managed by a native kernel V4L2 driver. For example,
RealSense's RSUSB backend uses userspace USB transfers; its native Linux V4L2
video backend uses kernel-managed transfers. Using `pyrealsense2` alone does not
identify which backend the installed librealsense build uses. Doctor does not
detect the streaming backend, measure outstanding transfers, or determine whether
the ceiling is sufficient, so it does not label a value such as 16 MB unhealthy.

### Try a temporary increase

Consider this experiment when the camera SDK reports transfer-buffer allocation
or submission failures consistent with exhausted usbfs memory, especially with
multiple cameras or large queued transfers. A generic frame timeout alone does
not establish this cause. First check the negotiated link, cable, power, and
shared upstream connections; increasing transfer memory cannot fix bandwidth
congestion, a USB 2 fallback, or repeated disconnects.

On the affected Linux host, record the current setting and, **if it is below
256 MB and not already unlimited**, try a 256 MB ceiling:

```bash
# Save the current value in this shell before changing it.
original_usbfs_memory_mb=$(cat /sys/module/usbcore/parameters/usbfs_memory_mb)
echo "$original_usbfs_memory_mb"

# Temporary change; requires administrator permission on the host.
echo 256 | sudo tee /sys/module/usbcore/parameters/usbfs_memory_mb
```

Stop and restart the camera stream in Studio or Runtime, then repeat the same
resolution, FPS, streams, and camera count. Re-run `pai doctor --json` to confirm the
setting. Only the manual setting change requires administrator permission;
do not run Studio or Runtime as root.

If the change does not help, restore the original value in the same shell:

```bash
echo "$original_usbfs_memory_mb" | sudo tee /sys/module/usbcore/parameters/usbfs_memory_mb
```

The temporary setting resets at reboot. Avoid an unlimited ceiling as a general
workaround: a finite limit bounds memory available to outstanding usbfs transfers.

### Persist only a confirmed improvement

On a **GRUB-based Ubuntu installation**, edit `/etc/default/grub` with administrator
permission. Preserve existing options and append `usbcore.usbfs_memory_mb=256`
inside the existing `GRUB_CMDLINE_LINUX_DEFAULT` quotes. If that parameter is
already present, update it rather than adding a duplicate. Then run:

```bash
sudo update-grub
sudo reboot
```

After reboot, verify the value with `pai doctor` or `cat` as above. To undo the
persistent change, restore the previous parameter (or remove the one you added),
run `sudo update-grub`, and reboot again. Other bootloaders and distributions
require their own kernel-command-line configuration; do not assume GRUB is in use.
Doctor never writes this setting or edits boot configuration.

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
IDs, parent relationships, endpoints, findings, the host usbfs memory ceiling, and
collection limitations. A selected device scopes findings but retains the full USB
inventory as topology context and the system-wide setting.
`--tree` affects human formatting only.

Reports are **not redacted**: review serials, device names, and paths before sharing.
There is no setup wizard, configuration export, streaming benchmark, or watch mode.
Re-run doctor after changing connections; no system settings are tuned automatically.

Exit codes are `0` for a report without observed errors (warnings are allowed), `1`
for access or collection errors, and `2` for an invalid selector/argument or unsupported
OS. No devices found is not itself an error or proof that expected hardware is absent.
