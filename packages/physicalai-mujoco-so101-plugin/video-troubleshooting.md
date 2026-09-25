# Video Camera Troubleshooting

> Since the REST/MJPEG camera server was introduced, the MuJoCo cameras no
> longer need v4l2loopback by default. The default `start` serves `overview`
> (plus `wrist` for single-arm, or `left_wrist`/`right_wrist` for bimanual)
> over HTTP; v4l2loopback output is opt-in via `--v4l2`. This document covers
> both paths.

## HTTP camera server (default)

The `start` CLI runs a FastAPI server on `--http-port` (default `8080`) that
serves each camera as an MJPEG stream and exposes control endpoints:

- `GET /cameras/{name}/mjpeg` — browser/VLC-viewable stream
- `GET /cameras/{name}/frame.jpg` — single JPEG snapshot
- `POST /scenes/{scene_id}` — switch scene
- `POST /reset` — reset/randomize the current scene
- `POST /shutdown` — stop the simulation owner

Because the owner process is detached from the CLI, `Ctrl+C` on the CLI does
not stop it; use `POST /shutdown` or `--idle-timeout`.

## v4l2loopback (opt-in, `--v4l2`)

The legacy v4l2loopback path below applies only when running with `--v4l2`.

## Device layout

This XPS 14 uses an OV08x40 MIPI CSI camera sensor connected to the Intel IPU7 ISP.

- `/dev/video16` through `/dev/video31` are IPU7 media-pipeline endpoints.
- These nodes expose raw Bayer/capture paths, not browser-ready webcam video.
- The IPU7 udev rule deliberately hides these nodes from normal applications.
- `v4l2-relayd@ipu7.service` runs Intel's `icamerasrc` stack, processes the IPU7 camera stream, and writes standard webcam frames to a `v4l2loopback` device labelled `Hardware ISP Camera`.
- `/dev/video32` is the external Innomaker USB webcam. `/dev/video33` is its metadata node.

## Incident: internal camera disappeared

MuJoCo loopback cameras were loaded with:

```bash
sudo modprobe v4l2loopback devices=2 video_nr=60,61 \
  card_label="MuJoCo Wrist,MuJoCo Overview" \
  exclusive_caps=1 max_buffers=2
```

`v4l2loopback` parameters apply to the whole loaded kernel module. Reloading it with only the two MuJoCo devices removed the configuration supplied by `intel-ipu7-camera` for `Hardware ISP Camera`. The IPU7 relay then wrote to an unintended loopback node, which conflicted with MuJoCo. Reloading while the relay still had mapped buffers caused a `v4l2loopback` kernel fault and left `v4l2-relayd` as a zombie process.

Symptoms:

- Video-call applications could not use the internal camera.
- `v4l2-relayd@ipu7.service` appeared active but its main process was a zombie.
- PipeWire exposed `Hardware ISP Camera`, but it had no frame producer.

## Recovery procedure

<!-- markdownlint-disable MD029 -->

1. Stop MuJoCo publishers and confirm no process has loopback devices open:

```bash
fuser -v /dev/video60 /dev/video61
sudo lsof /dev/video60 /dev/video61
```

2. Stop the IPU7 relay:

```bash
sudo systemctl stop v4l2-relayd@ipu7.service
```

3. Unload the old loopback module only after the devices are unused:

```bash
sudo modprobe -r v4l2loopback
```

4. Confirm loopback devices disappeared, then load the IPU and MuJoCo loopbacks together:

```bash
v4l2-ctl --list-devices

sudo modprobe v4l2loopback \
  devices=3 \
  video_nr=50,60,61 \
  card_label="Hardware ISP Camera,MuJoCo Wrist,MuJoCo Overview" \
  exclusive_caps=1,1,1 \
  max_buffers=16
```

`max_buffers` is module-wide, so it accepts one value rather than one value per device.

5. Start the relay and refresh PipeWire/WirePlumber discovery:

```bash
sudo systemctl start v4l2-relayd@ipu7.service
systemctl --user restart wireplumber.service
```

6. Verify:

```bash
systemctl status v4l2-relayd@ipu7.service
v4l2-ctl --list-devices
wpctl status
```

<!-- markdownlint-enable MD029 -->

The relay must have a real running PID, and PipeWire should list `Hardware ISP Camera` as a video source.

## Current known-good state

After recovery, the card labels and PipeWire sources were correct:

- `Hardware ISP Camera` is produced by the running IPU7 relay.
- `MuJoCo Wrist` and `MuJoCo Overview` are separate loopback devices.
- The Innomaker USB camera remains available.

The requested minor numbers were already occupied, so Linux assigned the next available numbers:

```text
Hardware ISP Camera: /dev/video51
MuJoCo Wrist:        /dev/video60
MuJoCo Overview:     /dev/video61
```

The IPU relay finds its output by the `Hardware ISP Camera` label, so `/dev/video51` is valid. Configure MuJoCo for `/dev/video60` and `/dev/video61` in this session.

## Safe operating rules

- Do not run a separate `modprobe v4l2loopback` command for MuJoCo while the module is already loaded; it will not add devices or change module parameters.
- Do not unload or reload `v4l2loopback` while MuJoCo publishers, the IPU relay, OBS, or another client has a loopback device open.
- When loopback configuration must change, first stop all producers and consumers, stop `v4l2-relayd@ipu7.service`, then unload and load all required devices together.
- Prefer selecting cameras by their descriptive card label in applications. Device minors can change when another driver reserves a requested number.
