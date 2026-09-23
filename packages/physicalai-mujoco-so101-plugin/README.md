# PhysicalAI MuJoCo SO-101 Plugin

Run single-arm or bimanual SO-101 simulations through [PhysicalAI Runtime](https://github.com/openvinotoolkit/physicalai). The plugin provides a browser viewer, camera streams, task scenes, and follower catalog entries for [Physical AI Studio](https://github.com/open-edge-platform/physical-ai-studio).

## Features

- Run a virtual SO-101 as a PhysicalAI transport owner
- Viser browser viewer with Reset Scene and confirmed Shutdown controls
- Camera streams over HTTP (MJPEG) and optional v4l2loopback
- REST control server (scenes, reset, shutdown)
- Built-in task scenes
- Automatic cube respawn after a five-second success dwell in `single_pick_place`

## What this is for

- Teleoperating a virtual SO-101 from PhysicalAI Studio.
- Testing robot setup, task logic, and control flows without physical hardware.
- Playing back policy/inference outputs in a simulated scene while monitoring robot + camera streams.
- Developing workflows where Studio, transport, and robot APIs stay identical between sim and real deployments.

## Quick start

Requires Python 3.12 or newer. From the Runtime repository root:

```bash
git lfs pull
uv sync --package physicalai-mujoco-so101-plugin --extra tests
uv run --package physicalai-mujoco-so101-plugin physicalai-mujoco-so101 start
```

In an existing Studio environment, install the plugin with `uv pip install physicalai-mujoco-so101-plugin` and restart Studio to discover its catalog entries.

By default this starts:

- A MuJoCo SO-101 owner named `mujoco-so101-follow`
- Browser viewer at `http://127.0.0.1:9090`
- Camera streams served over HTTP (MJPEG + REST control server on port `8080`)
- Control rate `50 Hz`
- Substeps `10`

Then open PhysicalAI Studio and connect to the robot type `MuJoCo SO-101 Follower` with name `mujoco-so101-follow`.

Add HTTP camera records in Studio for `http://127.0.0.1:8080/cameras/wrist/mjpeg` and `http://127.0.0.1:8080/cameras/overview/mjpeg`. The plugin registers follower robots; choose an available leader separately when configuring teleoperation.

The viewer's **Reset Scene** button respawns the scene's objects while keeping the target fixed. In `single_pick_place`, the cube respawns automatically after remaining still over the target for five seconds of simulation time. The **Shutdown** button asks for confirmation before stopping the owner.

### Bimanual simulation

Start the garment scene alongside a single-arm owner using separate server ports:

```bash
uv run --package physicalai-mujoco-so101-plugin physicalai-mujoco-so101 start --bimanual --http-port 8081 --viser-port 9091
```

Connect Studio's `MuJoCo SO-101 Bimanual Follower` to `mujoco-so101-bimanual`. Its camera names are `left_wrist`, `right_wrist`, and `overview`, served on the selected HTTP port.

### Stop an owner

```bash
uv run --package physicalai-mujoco-so101-plugin physicalai-mujoco-so101 stop --name mujoco-so101-follow
uv run --package physicalai-mujoco-so101-plugin physicalai-mujoco-so101 stop --name mujoco-so101-bimanual --http-port 8081
```

The stop command selects a local owner by name. Its HTTP fallback checks that the server reports the requested name.

### CLI teleoperation (self-relay)

With the owner running, you can also relay the simulation back to itself using
the [PhysicalAI CLI](https://github.com/openvinotoolkit/physicalai):

```bash
uv run --package physicalai-mujoco-so101-plugin physicalai run --config packages/physicalai-mujoco-so101-plugin/examples/runtime/teleop.yaml
```

Press `Ctrl+C` to stop.

View the camera streams in a browser or with `curl`:

```bash
curl http://127.0.0.1:8080/health
```

MJPEG stream URLs are `http://127.0.0.1:8080/cameras/<name>/mjpeg` (e.g. open
`http://127.0.0.1:8080/cameras/overview/mjpeg` in a browser, or play it in VLC).

## CLI options

```bash
uv run --no-sync physicalai-mujoco-so101 start --help
```

Common options:

- `--name <robot-name>`: transport name (must match Studio payload)
- `--bimanual`: run two arms with the `garment_fold` scene by default
- `--model <path>`: custom XML/URDF path (bypasses scene resolution)
- `--scene <name>`: scene name (`single_pick_place`, `pick_lift`, `pick_place`, or `yahtzee`, default `single_pick_place`)
- `--no-gui`: disable all viewers
- `--viser-port <port>`: browser viewer port (default `9090`)
- `--no-cameras`: disable camera rendering entirely (HTTP streams and v4l2loopback)
- `--http-host <host>`: host for the camera/control HTTP server (default `127.0.0.1`)
- `--http-port <port>`: port for the camera/control HTTP server (default `8080`)
- `--no-http`: disable the camera/control HTTP server
- `--v4l2`: also publish cameras to v4l2loopback devices (requires `modprobe v4l2loopback`)
- `--rate-hz <float>`: owner loop frequency
- `--substeps <int>`: MuJoCo steps per control cycle
- `--idle-timeout <seconds>`: seconds with zero subscribers before self-exit
  (default `10` without HTTP, disabled when HTTP is enabled so stream viewers keep the sim alive)
- `--allow-remote`: allow non-loopback zenoh connections

## Cameras over HTTP (default)

The plugin renders two camera feeds and serves them over HTTP:

- `wrist` -> `http://127.0.0.1:8080/cameras/wrist/mjpeg`
- `overview` -> `http://127.0.0.1:8080/cameras/overview/mjpeg`

Each camera is also available as a single JPEG snapshot at
`http://127.0.0.1:8080/cameras/<name>/frame.jpg`.

### REST control API

The HTTP server exposes control endpoints for resetting and switching scenes:

```bash
# List available scenes and the current one
curl http://127.0.0.1:8080/scenes

# Switch to another scene
curl -X POST http://127.0.0.1:8080/scenes/pick_place

# Reset/randomize the current scene
curl -X POST http://127.0.0.1:8080/reset

# Stop the simulation owner
curl -X POST http://127.0.0.1:8080/shutdown
```

| Endpoint                    | Method | Description                                   |
| --------------------------- | ------ | --------------------------------------------- |
| `/`                         | GET    | Service info, endpoint index                  |
| `/health`                   | GET    | Sim status: connected, current scene, cameras |
| `/cameras`                  | GET    | Camera list with stream/snapshot URLs         |
| `/cameras/{name}/mjpeg`     | GET    | MJPEG stream (`multipart/x-mixed-replace`)    |
| `/cameras/{name}/frame.jpg` | GET    | Latest frame as a JPEG snapshot               |
| `/scenes`                   | GET    | Current scene and available scene IDs         |
| `/scenes/{scene_id}`        | POST   | Switch to another registered scene            |
| `/reset`                    | POST   | Reset/randomize the current scene             |
| `/shutdown`                 | POST   | Gracefully stop the simulation owner          |

With HTTP enabled, `Ctrl+C` in the start command requests owner shutdown through HTTP and disconnects the CLI subscriber. With HTTP disabled, the detached owner exits after its configured idle timeout once all subscribers leave. Use the named `stop` command to stop an owner directly.

The `start` command also returns when its local owner exits through `stop`, HTTP, or the viewer's Shutdown control. Launcher cleanup checks the endpoint's owner name and the original owner's PID before forwarding an operator-requested shutdown. An invocation that attaches without a local owner record detaches instead of waiting indefinitely.

HTTP control has no authentication and binds to loopback by default. Use explicit non-loopback bindings only on a trusted robot-cell network. The viewer also exposes controls: restrict its port to trusted clients when accessing it remotely.

## Cameras and v4l2loopback (opt-in)

For workflows that need a webcam-visible device, pass `--v4l2` to publish the
same camera feeds to v4l2loopback devices:

- `wrist` -> `/dev/video<wrist-video-id>` (default `/dev/video60`)
- `overview` -> `/dev/video<overview-video-id>` (default `/dev/video62`)

Example with custom IDs:

```bash
uv run --no-sync physicalai-mujoco-so101 start --v4l2 --wrist-video-id 70 --overview-video-id 71
```

### One-time setup (Linux)

Install and load v4l2loopback:

```bash
sudo modprobe v4l2loopback exclusive_caps=1 video_nr=60,62
```

If you use custom camera IDs, use matching `video_nr` values. Example:

```bash
sudo modprobe v4l2loopback exclusive_caps=1 video_nr=70,71
```

If the module is already loaded with different params, unload and reload:

```bash
sudo rmmod v4l2loopback
sudo modprobe v4l2loopback exclusive_caps=1 video_nr=60,62
```

### Verify devices

```bash
ls /dev/video60 /dev/video62
```

For custom IDs, verify those device nodes instead.

Optional sanity checks:

```bash
v4l2-ctl --all -d /dev/video60
v4l2-ctl --all -d /dev/video62
```

## Troubleshooting

### HTTP server unavailable or port already in use

- The camera/control server binds to `--http-host`/`--http-port` (default `127.0.0.1:8080`).
- If the port is taken the simulation continues without HTTP and logs a warning — pass a different `--http-port`.
- When running multiple simulations, give each a distinct `--http-port`.

### `Camera '<name>' unavailable` or invalid argument on `/dev/video*` (with `--v4l2`)

- Ensure v4l2loopback is loaded with `exclusive_caps=1`
- Ensure the configured video devices exist (default `/dev/video60`, `/dev/video62`)
- Check permissions on device nodes

### Viewer opens but has Wayland warnings (`libdecor`, window position)

These warnings are typically non-fatal on Wayland and can be ignored if simulation continues.

### Camera/control server is not started

- Confirm the sim is running and the port is free: `curl http://127.0.0.1:8080/health`
- `--no-http` or `--http-port 0` disables the server; `--no-cameras` disables all camera rendering

## Using with Studio teleop and inference playback

Typical workflow:

1. Start simulation owner: `physicalai-mujoco-so101 start`
2. Connect from PhysicalAI Studio (`MuJoCo SO-101 Follower`)
3. Teleoperate in Studio and observe state/cameras
4. Run policy inference and play action outputs into the same simulated robot

Because this uses PhysicalAI transport + Studio catalog integration, you can iterate on control and inference loops in simulation before moving to hardware.

## Scenes

The plugin ships with built-in scenes that provide different environments for the robot:

| Scene ID                      | Description                             | Free objects    | Target      |
| ----------------------------- | --------------------------------------- | --------------- | ----------- |
| `single_pick_place` (default) | One block and a target disc             | 1 cube          | target disc |
| `pick_lift`                   | Three colored cubes and a target disc   | 3 cubes         | target disc |
| `pick_place`                  | A cube, a cylinder, and a target zone   | cube + cylinder | target zone |
| `yahtzee`                     | Six dice and a cup                      | 6 dice          | cup         |
| `garment_fold`                | Bimanual SO-101 with a flexible garment | garment         | none        |

Start with a specific scene:

```bash
uv run --no-sync physicalai-mujoco-so101 start --scene pick_place
```

### Switching scenes at runtime

`POST /scenes/{scene_id}` switches a running simulation to another registered scene. The switch happens on the next control cycle:

- Loads the new scene XML
- Updates the existing viewer with the new environment
- Respawns the scene's free-object joints clear of their target, using the new spawn parameters (target bodies stay fixed)

A scene that does not declare the joints the running robot drives is rejected and the current scene is kept, so a single-arm simulation will not accept `garment_fold` and a bimanual one will not accept the single-arm scenes.

The native MuJoCo viewer also binds **`n`** (next scene) to cycle through the compatible scenes. That viewer is only used on Linux/Windows when the browser viewer fails to start; with the default viser viewer (and always on macOS) use the HTTP endpoint instead.

`--model` bypasses scene resolution entirely; only the exported scene XML path is loaded, and scene switching is unavailable.

## Development

Fetch LFS assets before running the real-model tests. They load MuJoCo models without opening a display:

```bash
uv sync --all-packages --extra tests --extra transport
uv run pytest --import-mode=importlib packages/physicalai-mujoco-so101-plugin/tests
uv build --package physicalai-mujoco-so101-plugin
```

The package was imported from [physicalai-plugins PR #36](https://github.com/MarkRedeman/physicalai-plugins/pull/36) at `53bf606ef0f340a2a1821fc42a1ab6f44c33edf7`. See `NOTICE` for provenance and `CHANGELOG.md` for earlier releases. Release publishing uses this repository's shared workflows; the existing PyPI project needs a trusted publisher for `openvinotoolkit/physicalai` before its first release here.

## Current status and future improvements

Planned/desired improvements:

- More configurable camera presets (pose/FOV/fps via CLI or payload)
- Scene randomization presets (object layouts, target marker variants, textures)
- RTSP streaming sink (the frame-buffer plumbing is sink-agnostic; an RTSP
  server such as `aiortsp` or MediaMTX can consume the same buffers later)
- Optional richer sensor streams (depth/segmentation style outputs)
- Additional sample tasks and policy playback recipes in this package
