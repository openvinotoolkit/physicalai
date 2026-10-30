# Connected hardware: diagnostic snapshot

Observed: 2026-10-28, approximately 13:50 CET, on Linux `7.2.8-9-omarchy`
(`x86_64`). This records the earlier inspection; it is not a live inventory and
was not refreshed when these documents were written.

**Current scope update (2026-10-28):** the user subsequently unplugged the ZED
because a colleague needed it. Its rows below are historical evidence, not a
current missing-device problem. Further live validation excludes it.

Related documents: [CLI proposal](cli-plan.md) and [implementation progress](progress.md).

## Scope and limitations

The inspection used `lsusb`, sysfs, udev metadata/links, V4L2 capability and format
queries, process-ownership checks, and kernel journal excerpts. It did not start
camera streams, open robot serial connections, query robot firmware, change
settings, or establish concurrent streaming reliability.

Device numbers and `/dev/videoN`, `/dev/ttyUSBN`, and `/dev/hidrawN` assignments are
snapshot values, not persistent identities. The user identified the three serial
connections as StarArm robots and the cameras as Hiwonder, Orbbec, and Stereolabs.
USB metadata alone does not confirm the robots' models or the Hiwonder branding.
Serial numbers below are local hardware identifiers; redact them before sharing
this document publicly if needed.

## Main connected devices

| User/device description                                          | USB VID:PID | Reported USB serial | USB topology path | Negotiated speed | Current endpoints                                              |
| ---------------------------------------------------------------- | ----------- | ------------------- | ----------------- | ---------------- | -------------------------------------------------------------- |
| StarArm at `ttyUSB0`; CH340 adapter                              | `1a86:7523` | None                | `3-4.1.4`         | 12 Mb/s          | `/dev/ttyUSB0`                                                 |
| StarArm at `ttyUSB1`; CH340 adapter                              | `1a86:7523` | None                | `3-1.2`           | 12 Mb/s          | `/dev/ttyUSB1`                                                 |
| StarArm at `ttyUSB2`; CH340 adapter                              | `1a86:7523` | None                | `3-4.1.1`         | 12 Mb/s          | `/dev/ttyUSB2`                                                 |
| Likely Hiwonder camera; enumerates as icSpring `icspring camera` | `32e6:9221` | `202404160005`      | `3-1.4`           | 480 Mb/s         | Capture `/dev/video42`; metadata `/dev/video43`; `/dev/media4` |
| Orbbec Gemini 305                                                | `2bc5:0840` | `CV2T661000T1`      | `3-4.1.2`         | 480 Mb/s         | `/dev/video32`–`41`; `/dev/media1`–`3`                         |
| Stereolabs ZED-M HID interface                                   | `2b03:f681` | `10209071`          | `3-1.3`           | 12 Mb/s          | `/dev/hidraw8`; no video interface visible                     |

The serial adapters use `ch341`; both enumerated USB cameras use `uvcvideo`.
The ZED HID interface uses `usbhid`.

## Host controllers and physical topology

```text
0000:00:14.0 — Intel standalone xHCI
  Bus 003: USB 2 root hub
    root port 1 → 3-1 USB 2 hub (480 Mb/s)
      port 2 → 3-1.2   CH340 / ttyUSB1
      port 3 → 3-1.3   ZED-M HID only
      port 4 → 3-1.4   icSpring / likely Hiwonder camera
    root port 3 → 3-3 Synaptics SVP7500
    root port 4 → 3-4 VIA USB 2 hub (480 Mb/s)
      port 1 → 3-4.1 USB 2 hub (480 Mb/s)
        port 1 → 3-4.1.1 CH340 / ttyUSB2
        port 2 → 3-4.1.2 Orbbec Gemini 305
        port 3 → 3-4.1.3 Logitech G Pro mouse
        port 4 → 3-4.1.4 CH340 / ttyUSB0
      port 2 → 3-4.2 Dell DA20 / Billboard interface
  Bus 004: SuperSpeed root hub, no downstream devices visible

0000:00:0d.0 — Intel Type-C subsystem xHCI
  Bus 001: USB 2 root hub, no downstream devices visible
  Bus 002: SuperSpeed root hub
    root port 3 → 2-3 VIA USB 3 hub (5 Gb/s)
      no downstream devices visible
```

For `3-4.1.2`, `3` is the current USB bus number and `4.1.2` is the downstream
port chain. The two camera-bearing USB 2 hubs use different root ports, although
they share a host controller. Do not infer one global 480 Mb/s budget merely from
the shared bus number. Each hub's upstream link is shared by its descendants.

The PCI-qualified `by-path` aliases below avoid relying on the current bus number.
They describe topology, not a chassis label such as “left rear connector.” Map
chassis labels using user-assisted unplug/replug observation. They remain
port-stable only while the host topology and upstream hub connections remain
equivalent; firmware, hub, port, or interface-layout changes can change them.

## Robot selectors: use physical paths for these adapters

Existing port-bound paths:

```text
/dev/ttyUSB0
  /dev/serial/by-path/pci-0000:00:14.0-usb-0:4.1.4:1.0-port0

/dev/ttyUSB1
  /dev/serial/by-path/pci-0000:00:14.0-usb-0:1.2:1.0-port0

/dev/ttyUSB2
  /dev/serial/by-path/pci-0000:00:14.0-usb-0:4.1.1:1.0-port0
```

Equivalent `usbv2` aliases also exist. These three devices do not expose USB
serial numbers. Their udev `ID_SERIAL=1a86_USB_Serial` is generated from generic
device metadata; it is not a unique manufacturer serial.

All three claim the same alias:

```text
/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0
```

It resolved to `/dev/ttyUSB2` during inspection. Do not use it to identify one of
the three robots. Another adapter can win the alias after re-enumeration.
Port-bound selectors distinguish slots but cannot prove which identical,
serial-less physical robot occupies a slot.

## Hiwonder / icSpring camera selectors and formats

Existing capture aliases, both resolving to `/dev/video42`:

```text
/dev/v4l/by-id/usb-icSpring_icspring_camera_202404160005-video-index0
/dev/v4l/by-path/pci-0000:00:14.0-usb-0:1.4:1.0-video-index0
```

The corresponding `video-index1` links resolve to the metadata node,
`/dev/video43`, not a second capture stream. `usbv2` path aliases also exist.
The reported serial is present and not duplicated by another observed camera;
uniqueness across other units of this model was not established.

Advertised modes include MJPEG at 1920×1080 and 1280×720 at 30 FPS, plus YUYV at
640×480 at 30 FPS. The queried configuration was 1280×720 MJPEG at 30 FPS. This
does not establish that a stream was active or functioning at that rate.

## Orbbec: select an endpoint, not just the camera serial

The Gemini 305 exposes five capture nodes, each with a paired metadata node.
Prepend the following to every suffix in the table:

```text
/dev/v4l/by-path/pci-0000:00:14.0-usb-0:
```

| Capture node   | Metadata node  | Advertised format(s) | Port-bound capture suffix |
| -------------- | -------------- | -------------------- | ------------------------- |
| `/dev/video32` | `/dev/video33` | `Z16` depth          | `4.1.2:1.0-video-index0`  |
| `/dev/video34` | `/dev/video35` | `GREY` grayscale     | `4.1.2:1.0-video-index2`  |
| `/dev/video36` | `/dev/video37` | `BA81` Bayer         | `4.1.2:1.0-video-index4`  |
| `/dev/video38` | `/dev/video39` | `YUYV`, `MJPG`       | `4.1.2:1.4-video-index0`  |
| `/dev/video40` | `/dev/video41` | `YUYV`               | `4.1.2:1.6-video-index0`  |

For example, the slot/interface-bound replacement for `/dev/video40` is:

```text
/dev/v4l/by-path/pci-0000:00:14.0-usb-0:4.1.2:1.6-video-index0
```

All queried capture nodes reported 640×480 at 30 FPS as their current configured
mode, without streaming. On `/dev/video38`, advertised 1280×720 modes were limited
to 10 or 5 FPS in the observed USB 2 configuration. The camera supports USB 2;
the vendor recommends USB 3. The current two-hub USB 2 path limits its options.

The genuine camera serial does not make every `by-id` alias stream-unique:

```text
/dev/v4l/by-id/usb-Orbbec_R__Orbbec_Gemini_305_CV2T661000T1-video-index0
```

`/dev/video32`, `/dev/video38`, and `/dev/video40` all claim that name. It resolved
to `/dev/video38` during inspection. The paired metadata endpoints similarly
collide on `video-index1`. The aliases for indices 2–5 pointed to `/dev/video34`–`37`.
The existing `/dev/Gemini_305` alias resolved to `/dev/bus/usb/003/049`, not a V4L2
capture node. Do not assume it selects an image stream.

These are observed paths, not a claim that every Runtime camera backend currently
accepts them. The proposal includes end-to-end stable-selector support; default
OmniCamera path handling currently needs work for arbitrary `by-path` strings.

## Stereolabs: partial enumeration

Only the ZED-M HID/IMU interface appeared. There was no Stereolabs video device in
`lsusb` and no ZED V4L2 capture node.

Existing HID aliases, both resolving to `/dev/hidraw8`:

```text
/dev/input/by-id/usb-STEREOLABS_ZED-M_Hid_Device_10209071-hidraw
/dev/input/by-path/pci-0000:00:14.0-usb-0:1.3:1.0-hidraw
```

Stereolabs documents that the HID/IMU interface works over USB 2, while stereo
video requires SuperSpeed. This device is behind a USB 2-only hub, consistent
with the missing video interface. First try a direct USB 3 connection using a
verified SuperSpeed data cable, bypassing that hub. Stereolabs also recommends
trying the camera-side connector orientation when diagnosing this symptom.
No replug experiment was performed during this inspection.

The current user could not read or write `/dev/hidraw8` (`0600`, root-owned), nor
write the device's raw USB node. Access requirements depend on the SDK's chosen
interface; this is a separate observation, not proof of the missing-video cause.

## Other enumerated USB devices

| Device                        | VID:PID     | Path           | Reported serial / note                                                |
| ----------------------------- | ----------- | -------------- | --------------------------------------------------------------------- |
| Huasheng USB 2 hubs           | `214b:7260` | `3-1`, `3-4.1` | No serial exposed                                                     |
| VIA USB 2 hub                 | `2109:2211` | `3-4`          | No serial exposed                                                     |
| VIA USB 3 hub                 | `2109:0211` | `2-3`          | 5 Gb/s; no downstream devices visible                                 |
| Logitech G Pro mouse          | `046d:c088` | `3-4.1.3`      | `3A37794068477D38`; 12 Mb/s                                           |
| Dell DA20 adapter / Billboard | `413c:b080` | `3-4.2`        | `00000000000000000`; not a useful unique serial                       |
| Synaptics SVP7500             | `06cb:0701` | `3-3`          | `01.00.00.00`; internal `usbio-bridge` device, not a discovered robot |

There is also an internal ISP/media device (`ipu7`, `/dev/media0`) and a virtual
“Hardware ISP Camera” at `/dev/video50`. Many internal ISP nodes exist; node count
must not be presented as the number of physical USB cameras. No RealSense camera
was observed.

## Resource settings, access, and logs

- `/sys/module/usbcore/parameters/usbfs_memory_mb` was `16` (16 MiB). This is an
  allocation limit for usbfs buffers, not reserved memory or bus bandwidth. It is
  likely the setting from the colleague's multi-RealSense workaround, but there
  is no evidence here proving exhaustion or the benefit of changing it. It does
  not govern every kernel/V4L2 buffer allocation.
- `/proc/sys/kernel/dmesg_restrict` was `1`; unprivileged `dmesg` failed. Kernel
  messages were readable using `journalctl -k` without elevation.
- The current user had read/write access to all three serial ports (group `uucp`)
  and the external USB V4L2 nodes. Raw USB access differs from serial/V4L2 access.
- No process holders were visible for the inspected serial/video nodes within
  the current user's process visibility. This is not proof of system-wide idleness.
- Earlier messages in this boot included USB protocol/address errors (`-71`),
  a descriptor timeout (`-110`), and “Maybe the USB cable is bad?” messages. They
  involved other paths or earlier configurations and do not establish a fault
  with the present device layout. No current bandwidth-exhaustion diagnosis was
  established by the inspected logs.
- Some idle cameras reported runtime suspension with autosuspend enabled. That
  alone is normal power management, not evidence that streaming will fail.

Useful read-only starting commands for a subsequent inspection:

```bash
lsusb
lsusb -t
v4l2-ctl --list-devices
udevadm info --query=property --name=/dev/ttyUSB0
udevadm info --query=property --name=/dev/video40
journalctl -k -b --since '10 minutes ago' --no-pager
cat /sys/module/usbcore/parameters/usbfs_memory_mb
```

## Acceptance cases derived from this snapshot

1. Preserve all three CH340 devices even though their generated identities match.
2. Flag their `by-id` collision and offer three distinct port-bound selectors.
3. Group Orbbec endpoints under one physical camera without hiding stream choice.
4. Detect Orbbec alias collisions within one genuine-serial device.
5. Report ZED HID-only presence rather than either “camera healthy” or “absent.”
6. Explain Orbbec USB 2 constraints without claiming unsupported USB 2 operation.
7. Do not classify a 16 MiB buffer limit, idle autosuspend, or historical error
   as proof of a current failure.
8. Use journal fallback when `dmesg` is restricted, without automatic sudo.
9. After a later replug test, preserve a port binding across node renumbering and
   report speed changes. This test remains unperformed.

## Operator-approved camera streaming validation

On 2026-10-28 the user approved camera-only testing of Hiwonder/icSpring and the
Orbbec image stream. Neither camera had a visible existing V4L2 opener before the
test. No robot port was opened and no USB/kernel settings were changed.

Each camera was tested individually, then both were tested concurrently, for
approximately five seconds per trial at requested/driver-reported 640×480 MJPEG
at 30 FPS. Initial host-received rates were:

| Camera                                   | Individual FPS | Concurrent FPS | Previous wire mode restored |
| ---------------------------------------- | -------------- | -------------- | --------------------------- |
| Hiwonder / icSpring                      | 10.4           | 10.2           | Yes: 1280×720 MJPEG, 30 FPS |
| Orbbec image stream (`1.4-video-index0`) | 30.2           | 31.0           | Yes: 640×480 YUYV, 30 FPS   |

These are short host-observed measurements, not USB utilization or a guaranteed
sensor frame rate; initial queued frames can affect short-window rates. The
diagnostic now flushes queued startup frames and also reports timestamp-derived
FPS. No substantial extra concurrency penalty appeared in this workload. The
Hiwonder's lower rate was already present when it ran alone, so adding the other
selected camera does not explain that result.

A read-only control inspection found aperture-priority auto exposure enabled on
Hiwonder (`auto_exposure=3`). The inactive absolute-exposure field reported its
default value, not a confirmed live exposure measurement. Exposure, firmware,
format negotiation, and decode/CPU behavior remain possible investigation areas;
no exposure setting was changed and no cause was established.

The saved formats were restored and queried again after capture. The overall
initial doctor exit code also included the then-present, incomplete ZED interface;
that historical finding is not an issue to investigate after the user's removal
of the ZED.

A final validation of only Hiwonder and Orbbec completed at
**2026-10-28 15:50 UTC**. It used the same requested 640×480 MJPEG/30 FPS modes,
flushed startup frames, and recorded frame-timestamp-derived rates:

| Camera              | Individual received FPS | Concurrent received FPS | Timestamp-derived FPS (individual / concurrent) |
| ------------------- | ----------------------- | ----------------------- | ----------------------------------------------- |
| Hiwonder / icSpring | 19.93                   | 20.11                   | 19.88 / 19.95                                   |
| Orbbec              | 30.13                   | 30.11                   | 30.06 / 30.06                                   |

Previous formats were again restored and verified for both cameras. The collector
observed no new USB-related kernel records during that test window. The overall
doctor exit code was `0` (warnings are non-fatal without `--strict`).

The Hiwonder's baseline rate varied between the two runs; these observations do
not isolate why. Within each run, adding the other selected stream did not produce
a substantial extra frame-rate penalty. Do not attribute the across-run change to
one device, setting, or cause without a controlled experiment.

## Throughput-view validation after a topology change

During the subsequent output/visualization work, passive enumeration found that
the Hiwonder had moved to `3-4.1.4`, beside Orbbec at `3-4.1.2`. Both then shared
the nested USB 2 hub `3-4.1` and its upstream hub `3-4`. The serial adapter
previously at that slot was no longer present there. The obsolete Hiwonder
port selector failed safely before any camera trial started.

After matching the two cameras by their previously observed VID/PID/serials,
the approved bounded camera-only test was repeated at the current port aliases.
Encoded V4L2 payload counters (not decoded RGB sizes) measured approximately:

| Camera   | Individual payload | Concurrent payload | Sampled target estimate at 30 FPS | Received FPS |
| -------- | ------------------ | ------------------ | --------------------------------- | ------------ |
| Hiwonder | 0.205 MB/s         | 0.205 MB/s         | 0.308 MB/s                        | 20.0         |
| Orbbec   | 1.360 MB/s         | 1.359 MB/s         | 1.355 MB/s                        | 30.1         |

The two concurrent streams together delivered about **1.56 MB/s**, compared with
about **1.66 MB/s estimated target demand** for this sampled MJPEG content. These
are selected application payloads, not complete USB traffic; compression varies,
and headers/retries/unmeasured peers are excluded. The shared link's 480 Mb/s
signaling rate corresponds to a **60 MB/s raw ceiling**, not available payload.
Both cameras' previous wire modes were restored and verified. No robot port or
kernel setting was changed; the ZED remained excluded.

References:

- [Orbbec Gemini 305 specifications](https://www.orbbec.com/gemini-305/).
- [Stereolabs explanation of ZED Mini HID-only enumeration](https://community.stereolabs.com/t/i-cant-get-my-zed-mini-camera-to-work/11761/2).
- [Linux kernel parameter documentation](https://docs.kernel.org/admin-guide/kernel-parameters.html),
  `usbcore.usbfs_memory_mb`.
