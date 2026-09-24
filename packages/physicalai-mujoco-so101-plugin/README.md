# PhysicalAI MuJoCo SO-101 Plugin

Run single-arm or bimanual SO-101 simulations through [PhysicalAI Runtime](https://github.com/openvinotoolkit/physicalai). The plugin provides a browser viewer, camera streams, task scenes, and follower catalog entries for [Physical AI Studio](https://github.com/open-edge-platform/physical-ai-studio).

## Features

- Run a virtual SO-101 as a PhysicalAI transport owner
- Viser browser viewer with a Simulation tab (scene switching, reset, arm homing, seeded resets, episode auto-reset, object dragging, camera previews, confirmed shutdown) and a Camera tab to follow an object, the target, or a gripper
- Camera streams over HTTP (MJPEG) and optional v4l2loopback
- REST control server with the same controls as the viewer
- Built-in task scenes
- Automatic cube respawn after a configurable success dwell (five seconds by default) in `single_pick_place`

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

By default this starts:

- A MuJoCo SO-101 owner named `mujoco-so101-follow`
- Browser viewer at `http://127.0.0.1:9090`
- Camera streams served over HTTP (MJPEG + REST control server on port `8080`)
- Control rate `50 Hz`
- Substeps `10`

To drive the simulation from Physical AI Studio, see [Use with Physical AI Studio](#use-with-physical-ai-studio).

### Browser viewer controls

The viewer opens on its **Simulation** tab, which controls the simulation. The **Camera** tab controls whether the view follows a body. The **Visualization** and **Groups** tabs come from mjviser and control what is drawn.

- **Scene**: pick another scene that fits the running robot (single-arm or bimanual). **Reset Scene** respawns the scene's objects while keeping the target fixed. **Home Arm** puts the arm joints and their position targets at the scene's home pose. An active policy or teleop session will drive the arm away again on its next action.
- **Randomization**: tick **Fixed seed** to reseed before every reset and scene switch, so object layouts repeat. Untick it to go back to random layouts.
- **Episode** (`single_pick_place` only): the cube respawns after it rests on the target for the success dwell. Turn **Auto-reset** off or change the dwell here. The panel shows the countdown and the number of completed episodes.
- **Objects**: tick **Drag objects** to show a handle on each free object. While you drag, the object stays at the handle's pose. When you let go, it falls from there with zero velocity. The target stays fixed.
- **Cameras**: tick **Show previews** to see low-rate thumbnails of the rendered cameras. Previews start off because every open viewer receives them.
- **Shutdown** asks for confirmation before stopping the owner.

On the **Camera** tab, **Follow** chooses a body for the view to stay on: a free object, the target, or a gripper. Orbiting and zooming keep the view on it. Panning away stops following, and **Follow** goes back to **None**. **None**, the default, is a free camera that nothing moves. While you drag the followed object, the view holds still, then glides back onto it when you let go. **FOV** and **Reset View** apply to every open viewer. The follow choice is shared too. Following moves the viewer cameras; the rendered scene itself never shifts.

Switching scenes rebuilds the whole viewer for the new model. The follow, drag and preview settings carry over to the new scene; following an object the new scene lacks falls back to **None**. Changes made over HTTP show up in the panel within about half a second. Camera settings are viewer-only and have no HTTP endpoint.

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

## Use with Physical AI Studio

Studio sees the simulation as an ordinary follower robot plus IP cameras, so teleoperation, dataset recording and policy inference work the same way they do with a real SO-101. The simulation runs as its own process; Studio attaches to it by name over the local Physical AI transport.

### 1. Install the plugin into Studio

The plugin is not on Studio's curated **Plugins** page, so install it into Studio's backend environment. From the Studio repository:

```bash
cd application/backend
uv add physicalai-mujoco-so101-plugin
uv sync
```

The PyPI release may lag behind this repository. To use the version in this checkout, including the viewer controls described above, install it as an editable path instead:

```bash
cd application/backend
uv add --editable /path/to/physicalai/packages/physicalai-mujoco-so101-plugin
uv sync
```

Restart Studio. The robot picker then offers **MuJoCo SO-101 Follower** and **MuJoCo SO-101 Bimanual Follower**. See Studio's robot plugin guide for Docker and other install options.

### 2. Start the simulation

Start the simulation before you add or open the robot in Studio. Studio checks that the robot is online by connecting to it:

```bash
uv run --package physicalai-mujoco-so101-plugin physicalai-mujoco-so101 start
```

Keep this terminal open. Open the viewer at `http://127.0.0.1:9090` to watch the scene and use the **Simulation** tab.

Studio and the simulation must run on the same machine. The transport only accepts local connections unless both sides allow remote ones (`--allow-remote` here, and the remote-connection option in the robot's advanced settings in Studio). A Studio instance running in Docker does not share the host's localhost.

### 3. Add the robot and cameras

In your Studio project:

1. Open **Robots** and select **Add robot**. Choose **MuJoCo SO-101 Follower** and keep the name `mujoco-so101-follow`. The name must match `--name` if you changed it.
2. Add a leader for teleoperation. The simulation provides only the follower. Use a real **SO101 Leader** arm connected over USB.
3. Add an **IP Camera** for each simulated camera, with these stream URLs:

   - `http://127.0.0.1:8080/cameras/wrist/mjpeg`
   - `http://127.0.0.1:8080/cameras/overview/mjpeg`

   The cameras render 640×480 at 30 fps. Set the camera's frame rate to 30; Studio's default is 25.

4. Open **Environments** and create an environment with the MuJoCo follower, the leader, and both cameras.

For the bimanual simulation, start it with `--bimanual`, choose **MuJoCo SO-101 Bimanual Follower** with the name `mujoco-so101-bimanual`, and add the `left_wrist`, `right_wrist` and `overview` cameras from its HTTP port. Its leader is a bimanual SO-101 leader, for example from the Bimanual SO-101 plugin.

### 4. Teleoperate and record datasets

Record from the dataset page as with a real robot (**Add episode**, **Start episode**, then **Accept** or **Discard**). Use the viewer in place of resetting a physical scene:

- **Reset Scene** before each episode respawns the objects. The target stays fixed.
- **Home Arm** puts the simulated arm at its home pose. The next teleop action moves it again, so use it while the leader is still or before starting teleoperation.
- In `single_pick_place`, auto-reset respawns the cube five seconds after it rests on the target, which can happen in the middle of a recording. Turn **Auto-reset** off in the **Episode** section if you prefer to reset by hand.
- Tick **Fixed seed** to get the same object layouts in the same order, for example to compare runs.

The same controls are available over HTTP (see [REST control API](#rest-control-api)) if you want to script resets between episodes.

### 5. Run a trained policy

On the **Models** page, select **Run model** for a policy trained on the simulation dataset. Studio loads the environment the dataset was recorded with, so keep the simulation running with the same owner name and camera ports. Use **Reset Scene** or **Fixed seed** in the viewer between runs.

To run the policy outside Studio, download its runtime configuration and run it with the Physical AI CLI. The plugin must be installed in that environment too; from this repository:

```bash
uv run --package physicalai-mujoco-so101-plugin physicalai run --config runtime.yaml
```

### 6. Stop

Press `Ctrl+C` in the `start` terminal, use the viewer's **Shutdown** button, or run the [`stop` command](#stop-an-owner). Stop any teleoperation or inference session in Studio first; otherwise Studio reports the robot as disconnected.

## CLI options

```bash
uv run --no-sync physicalai-mujoco-so101 start --help
```

Common options:

- `--name <robot-name>`: transport name (must match Studio payload)
- `--bimanual`: run two arms with the `garment_fold` scene by default
- `--model <path>`: custom XML/URDF path to load initially (bypasses default scene resolution)
- `--scene <name>`: scene name (`single_pick_place`, `pick_lift`, `pick_place`, or `yahtzee`, default `single_pick_place`)
- `--no-gui`: disable all viewers
- `--viser-port <port>`: browser viewer port (default `9090`)
- `--viser-host <host>`: browser viewer bind host (default `127.0.0.1`; use `0.0.0.0` to expose it remotely)
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

The HTTP server exposes the same controls as the viewer's Simulation tab:

```bash
# List available scenes, the ones this robot can switch to, and the current one
curl http://127.0.0.1:8080/scenes

# Switch to another scene
curl -X POST http://127.0.0.1:8080/scenes/pick_place

# Reset/randomize the current scene, or move the arm to its home pose
curl -X POST http://127.0.0.1:8080/reset
curl -X POST http://127.0.0.1:8080/home

# Make resets repeatable, then random again
curl -X POST http://127.0.0.1:8080/seed -H 'content-type: application/json' -d '{"seed": 42}'
curl -X POST http://127.0.0.1:8080/seed -H 'content-type: application/json' -d '{"seed": null}'

# Pause episode auto-reset and change its success dwell
curl -X POST http://127.0.0.1:8080/episode/auto-reset -H 'content-type: application/json' \
  -d '{"enabled": false, "dwell_s": 3.0}'

# Read and move free objects (world frame, metres; wxyz is optional)
curl http://127.0.0.1:8080/objects
curl -X POST 'http://127.0.0.1:8080/objects/block1:joint/pose' -H 'content-type: application/json' \
  -d '{"position": [0.25, 0.0, 0.05], "wxyz": [1, 0, 0, 0]}'

# Stop the simulation owner
curl -X POST http://127.0.0.1:8080/shutdown
```

| Endpoint                    | Method | Description                                                                      |
| --------------------------- | ------ | -------------------------------------------------------------------------------- |
| `/`                         | GET    | Service info, endpoint index                                                     |
| `/health`                   | GET    | Sim status: connected, scene, compatible scenes, seed, episode, objects, cameras |
| `/cameras`                  | GET    | Camera list with stream/snapshot URLs                                            |
| `/cameras/{name}/mjpeg`     | GET    | MJPEG stream (`multipart/x-mixed-replace`)                                       |
| `/cameras/{name}/frame.jpg` | GET    | Latest frame as a JPEG snapshot                                                  |
| `/scenes`                   | GET    | Current scene, available scene IDs, and IDs compatible with this robot           |
| `/scenes/{scene_id}`        | POST   | Switch to a compatible scene (`409` for another arm count)                       |
| `/reset`                    | POST   | Reset/randomize the current scene                                                |
| `/home`                     | POST   | Move the arm joints and targets to the scene's home pose                         |
| `/seed`                     | POST   | `{"seed": <0..4294967295> or null}`: fix or clear the reset seed                 |
| `/episode/auto-reset`       | POST   | `{"enabled": bool, "dwell_s": 0.5..120}` (either field); `409` if unsupported    |
| `/objects`                  | GET    | Free-object joint names and world poses                                          |
| `/objects/{joint}/pose`     | POST   | `{"position": [x, y, z], "wxyz": [w, x, y, z]}`: teleport a free object          |
| `/shutdown`                 | POST   | Gracefully stop the simulation owner                                             |

Control requests are queued and applied on the next control cycle. Invalid bodies return `422`. Object coordinates must be finite and within ±2 m.

With HTTP enabled, `Ctrl+C` in the start command requests owner shutdown through HTTP. If the HTTP endpoint is unavailable, the launcher sends SIGTERM to the original owner only after confirming its registered name and PID still match. It then disconnects the CLI subscriber. With HTTP disabled, the same verified signal path stops a local owner; detached owners also exit after their configured idle timeout once all subscribers leave. Use the named `stop` command to stop an owner directly.

The `start` command also returns when its local owner exits through `stop`, HTTP, or the viewer's Shutdown control. Launcher cleanup checks the endpoint's owner name and the original owner's PID before forwarding an operator-requested shutdown. An invocation that attaches without a local owner record detaches instead of waiting indefinitely.

HTTP control has no authentication and binds to loopback by default. Use explicit non-loopback bindings only on a trusted robot-cell network. The Viser viewer exposes the same controls, including moving objects and Shutdown, to anyone who can open it; it binds to loopback by default. Use `--viser-host 0.0.0.0` only when remote access is needed on a trusted network.

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

### Studio shows the MuJoCo robot as offline

- Start the simulation first, then reload the robot in Studio.
- Check that the robot name in Studio matches the owner name printed by `start` (default `mujoco-so101-follow`).
- Run Studio and the simulation on the same machine, or enable remote connections on both sides.

### Studio shows no camera image

- Open the stream URL in a browser; if it does not load, the HTTP server is not running or uses another port.
- Use an **IP Camera** with the full `/cameras/<name>/mjpeg` URL, not the server root.

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

Pick a scene in the viewer's **Scene** dropdown or send `POST /scenes/{scene_id}`. The switch happens on the next control cycle:

- Loads the new scene XML
- Rebuilds the viewer for the new environment
- Respawns the scene's free-object joints clear of their target, using the new spawn parameters (target bodies stay fixed)

Only scenes with the running robot's arm count are offered: a single-arm simulation cannot switch to `garment_fold`, and a bimanual one cannot switch to the single-arm scenes. Over HTTP such a request returns `409` and the current scene is kept. A scene whose model lacks the joints the robot drives is also rejected.

The native MuJoCo viewer also binds **`n`** (next scene) to cycle through the compatible scenes. That viewer is only used on Linux/Windows when the browser viewer fails to start.

`--model` selects the initial model instead of resolving a registered scene. The viewer's Scene dropdown shows it as **Custom model**, and **Home Arm** uses the model's default joint positions. Runtime scene-switch requests can still explicitly replace it with a compatible registered scene; switching back to the custom model is not available unless it is registered as a scene.

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
