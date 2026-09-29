# PhysicalAI MuJoCo SO-101 Plugin

Run single-arm or bimanual SO-101 simulations through [PhysicalAI Runtime](https://github.com/openvinotoolkit/physicalai). The plugin provides a browser viewer, camera streams, task scenes, and follower catalog entries for [Physical AI Studio](https://github.com/open-edge-platform/physical-ai-studio).

## Features

- Run a virtual SO-101 as a PhysicalAI transport owner
- Viser browser viewer with a Simulation tab (scene switching, reset, arm homing, seeded resets, episode auto-reset, conveyor belt controls, object dragging, camera previews, confirmed shutdown) and a Camera tab to follow an object, the target, or a gripper
- Camera streams over HTTP (MJPEG), used in Studio as IP cameras
- REST control server with the same controls as the viewer
- Built-in task scenes
- Automatic cube respawn after a configurable success dwell (five seconds by default) in `single_pick_place`
- A conveyor-belt sorting cell (`conveyor_sort`) with an adjustable belt speed, a continuous item feed, and per-episode scoring

## What this is for

- Teleoperating a virtual SO-101 from PhysicalAI Studio.
- Testing robot setup, task logic, and control flows without physical hardware.
- Playing back policy/inference outputs in a simulated scene while monitoring robot + camera streams.
- Developing workflows where Studio, transport, and robot APIs stay identical between sim and real deployments.

## Quick start

Requires Python 3.12 or newer. From the Runtime repository root:

```bash
git lfs pull
uv sync --package physicalai-mujoco-so101-plugin
uv run --package physicalai-mujoco-so101-plugin physicalai-mujoco-so101 start
```

On a headless or remote Linux machine (no desktop session, for example over SSH), MuJoCo cannot create an OpenGL context for the cameras and `start` fails with `an OpenGL platform library has not been loaded into this process`. Render with EGL instead:

```bash
MUJOCO_GL=egl uv run --package physicalai-mujoco-so101-plugin physicalai-mujoco-so101 start
```

Use `MUJOCO_GL=osmesa` on a machine without a GPU driver (slower, needs OSMesa installed). The same applies to the `start` command in the Studio setup below.

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
- **Performance**: the simulation speed as a multiple of real time, the control loop rate, and each camera's frame rate, over the last two seconds. Below 0.95x real time, the belt, arm and physics all run slower than the wall clock, while Studio records and policies run at wall-clock rates. Keep it at 1.00x when recording or evaluating.
- **Randomization**: tick **Fixed seed** to reseed before every reset and scene switch, so object layouts repeat. Untick it to go back to random layouts.
- **Episode** (`single_pick_place` only): the cube respawns after it rests on the target for the success dwell. Turn **Auto-reset** off or change the dwell here. The panel shows the countdown and the number of completed episodes.
- **Conveyor** (`conveyor_sort` only, in place of **Episode**): **Belt running** pauses or resumes the belt and the item feed. **Belt speed** sets the belt surface speed from 0 to 10 cm/s. The panel shows the items fed so far, the current and last episode's score, and the sorting rule.
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

Connect Studio's `MuJoCo SO-101 Bimanual Follower` to `mujoco-so101-bimanual-follow`. Its camera names are `left_wrist`, `right_wrist`, and `overview`, served on the selected HTTP port.

### Stop an owner

```bash
uv run --package physicalai-mujoco-so101-plugin physicalai-mujoco-so101 stop --name mujoco-so101-follow
uv run --package physicalai-mujoco-so101-plugin physicalai-mujoco-so101 stop --name mujoco-so101-bimanual-follow --http-port 8081
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

The PyPI release may lag behind this repository. To use the version in this checkout, including the viewer controls described above, run `git lfs pull` in this repository (the robot meshes are stored in Git LFS), then install it as an editable path instead:

```bash
cd application/backend
uv add --editable /path/to/physicalai/packages/physicalai-mujoco-so101-plugin
uv sync
```

Restart Studio. The robot picker then offers **MuJoCo SO-101 Follower** and **MuJoCo SO-101 Bimanual Follower**. See Studio's robot plugin guide for Docker and other install options.

### 2. Start the simulation

Start the simulation before you add or open the robot in Studio. Studio checks that the robot is online by connecting to it. Run it from Studio's backend environment, so the simulation and Studio use the same `physicalai` version:

```bash
cd application/backend
uv run physicalai-mujoco-so101 start
```

The simulation registers under the name `mujoco-so101-follow` (`mujoco-so101-bimanual-follow` with `--bimanual`). These are also the default robot names that Studio fills in for the MuJoCo followers, so you can keep both unchanged. To run under another name, pass `--name` and enter the same name when you add the robot in Studio:

```bash
uv run physicalai-mujoco-so101 start --name my-sim
```

Keep this terminal open. Open the viewer at `http://127.0.0.1:9090` to watch the scene and use the **Simulation** tab.

Studio and the simulation must run on the same machine. The transport only accepts local connections unless both sides allow remote ones (`--allow-remote` here, and the remote-connection option in the robot's advanced settings in Studio). A Studio instance running in Docker does not share the host's localhost.

### 3. Add the robot and cameras

Studio hides the **IP Camera** type behind a feature flag that is off by default. Enable it once in the browser: open Studio, run the following in the developer console, then reload the page.

```js
setFeatureFlag("ipcam", true);
```

Then, in your Studio project:

1. Open **Robots** and select **Configure new robot**. Choose **MuJoCo SO-101 Follower** and keep the name `mujoco-so101-follow`. The name must match `--name` if you changed it.
2. Add a leader for teleoperation. The simulation provides only the follower. Use a real **SO101 Leader** arm connected over USB.
3. Open **Cameras**, select **Configure new camera**, and add an **IP Camera** for each simulated camera, with these stream URLs:

   - `http://127.0.0.1:8080/cameras/wrist/mjpeg`
   - `http://127.0.0.1:8080/cameras/overview/mjpeg`

   The cameras render 640×480 at 30 fps, which matches the IP Camera defaults.

4. Open **Environments**, select **Configure new environment**, and add the MuJoCo follower, the leader, and both cameras.

For the bimanual simulation, start it with `--bimanual`, choose **MuJoCo SO-101 Bimanual Follower** with the name `mujoco-so101-bimanual-follow`, and add the `left_wrist`, `right_wrist` and `overview` cameras from its HTTP port. Its leader is a bimanual SO-101 leader, for example from the Bimanual SO-101 plugin.

### 4. Teleoperate and record datasets

Record from the dataset page as with a real robot (**Add episode**, **Start episode**, then **Accept** or **Discard**). Use the viewer in place of resetting a physical scene:

- **Reset Scene** before each episode respawns the objects. The target stays fixed.
- **Home Arm** puts the simulated arm at its home pose. The next teleop action moves it again, so use it while the leader is still or before starting teleoperation.
- In `single_pick_place`, auto-reset respawns the cube five seconds after it rests on the target, which can happen in the middle of a recording. Turn **Auto-reset** off in the **Episode** section if you prefer to reset by hand.
- Tick **Fixed seed** to get the same object layouts in the same order, for example to compare runs.

The same controls are available over HTTP (see [REST control API](#rest-control-api)) if you want to script resets between episodes.

#### Record the conveyor autopilot, with automatic episodes

In `conveyor_sort`, the scripted demonstrator can act as the leader arm, and the simulation can save each conveyor episode to the open dataset. No one has to teleoperate, and no one has to press **Accept** or **Start episode**.

1. In Studio, add a robot of type **MuJoCo SO-101 Virtual Leader**. Its port is the simulation's `--http-port` (default `8080`). Use it as the leader of the MuJoCo environment.
2. In the viewer's **Autopilot** folder, set **Autopilot** to **Virtual leader (Studio teleop)**. The demonstrator now only publishes its joint targets (`GET /leader`); the arm moves once Studio sends them back as actions.
3. In Studio, open the dataset's recording view and start teleoperation. The demonstrator's targets become the recorded actions.
4. In the viewer, set **Task** and **Keep**, then tick **Record in Studio**. The simulation attaches to Studio's running session. For each episode it clears the belt, starts a recording with the task, and holds the belt when the tenth item is scored. Then it saves the episode, or discards it if **Keep** is **Perfect episodes only** and the episode had a wrong or missed item.
5. Untick **Record in Studio** to stop, or set **Episodes** to stop after that many saved episodes. Pressing stop in Studio also stops the automation.

**Record in Studio** also works with a physical leader arm. You teleoperate, and the simulation saves an episode each time the belt's ten items are scored.

The simulation joins Studio's session the same way a second browser tab does. It finds the session through `GET /api/runtime/sessions` and never stops that session. Use `--studio-url` if Studio does not run at `http://127.0.0.1:7860`.

**Autopilot** has a third mode, **Drive the arm**. It moves the arm directly and ignores incoming actions, which is useful for demos, but don't record in this mode.

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
- `--scene <name>`: scene name (`single_pick_place`, `yahtzee` or `conveyor_sort`; with `--bimanual`, `garment_fold`). Default `single_pick_place`
- `--no-gui`: disable all viewers
- `--viser-port <port>`: browser viewer port (default `9090`)
- `--viser-host <host>`: browser viewer bind host (default `127.0.0.1`; use `0.0.0.0` to expose it remotely)
- `--no-cameras`: disable camera rendering (HTTP streams and viewer previews)
- `--http-host <host>`: host for the camera/control HTTP server (default `127.0.0.1`)
- `--http-port <port>`: port for the camera/control HTTP server (default `8080`)
- `--no-http`: disable the camera/control HTTP server
- `--rate-hz <float>`: owner loop frequency
- `--substeps <int>`: MuJoCo steps per control cycle
- `--unit <normalized|degrees>`: joint units for observations and actions (default `normalized`, see [Joint units](#joint-units))
- `--idle-timeout <seconds>`: seconds with zero subscribers before self-exit
  (default `10` without HTTP, disabled when HTTP is enabled so stream viewers keep the sim alive)
- `--allow-remote`: allow non-loopback zenoh connections
- `--studio-url <url>`: Physical AI Studio backend for automatic episode recording (default `http://127.0.0.1:7860`)

## Joint units

The simulation reports and accepts joint values in the same units as the real SO-101 (`physicalai.robot.SO101`):

- Body joints go from `-100` to `100`, and the gripper from `0` to `100`.
- On a real robot, calibration sets these ranges: `-100` and `100` are the two ends you moved each joint to while calibrating, and `0` is halfway between them.
- The simulation has no calibration, so it uses the joint limits from the robot's URDF (`urdf/so101/so101_new_calib.urdf`) instead: `-100` and `100` are each joint's limits, and `0` is halfway between them. The MuJoCo models use the same limits.

This lets a real leader arm drive the simulation, and makes simulated datasets use the same values as real ones.

The match is not exact. If a joint was moved further one way than the other during calibration, its `0` is not at the joint's straight position, so the same value points the real joint and the simulated joint in slightly different directions. A real arm's calibrated range also differs from the URDF limits by a few degrees because of assembly tolerances, which adds to the difference towards the ends of a joint's range.

Use `--unit degrees` (or `unit="degrees"` on `MuJoCoSO101`) to get joint angles in degrees instead.

## Cameras over HTTP

The plugin renders two camera feeds on their own thread and serves them over HTTP. The control loop only hands the camera thread a pose snapshot each tick, so rendering never delays physics:

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
curl -X POST http://127.0.0.1:8080/scenes/yahtzee

# Reset/randomize the current scene, or move the arm to its home pose
curl -X POST http://127.0.0.1:8080/reset
curl -X POST http://127.0.0.1:8080/home

# Make resets repeatable, then random again
curl -X POST http://127.0.0.1:8080/seed -H 'content-type: application/json' -d '{"seed": 42}'
curl -X POST http://127.0.0.1:8080/seed -H 'content-type: application/json' -d '{"seed": null}'

# Pause episode auto-reset and change its success dwell
curl -X POST http://127.0.0.1:8080/episode/auto-reset -H 'content-type: application/json' \
  -d '{"enabled": false, "dwell_s": 3.0}'

# conveyor_sort only: set the belt speed in m/s (pause it with /episode/auto-reset enabled=false)
curl -X POST http://127.0.0.1:8080/conveyor/belt-speed -H 'content-type: application/json' -d '{"speed": 0.05}'

# Read and move free objects (world frame, metres; wxyz is optional)
curl http://127.0.0.1:8080/objects
curl -X POST 'http://127.0.0.1:8080/objects/block1:joint/pose' -H 'content-type: application/json' \
  -d '{"position": [0.25, 0.0, 0.05], "wxyz": [1, 0, 0, 0]}'

# Stop the simulation owner
curl -X POST http://127.0.0.1:8080/shutdown
```

| Endpoint                    | Method | Description                                                                                    |
| --------------------------- | ------ | ---------------------------------------------------------------------------------------------- |
| `/`                         | GET    | Service info, endpoint index                                                                   |
| `/health`                   | GET    | Sim status: connected, scene, compatible scenes, seed, episode, timing, objects, cameras       |
| `/cameras`                  | GET    | Camera list with stream/snapshot URLs                                                          |
| `/cameras/{name}/mjpeg`     | GET    | MJPEG stream (`multipart/x-mixed-replace`)                                                     |
| `/cameras/{name}/frame.jpg` | GET    | Latest frame as a JPEG snapshot                                                                |
| `/scenes`                   | GET    | Current scene, available scene IDs, and IDs compatible with this robot                         |
| `/scenes/{scene_id}`        | POST   | Switch to a compatible scene (`409` for another arm count)                                     |
| `/reset`                    | POST   | Reset/randomize the current scene                                                              |
| `/home`                     | POST   | Move the arm joints and targets to the scene's home pose                                       |
| `/seed`                     | POST   | `{"seed": <0..4294967295> or null}`: fix or clear the reset seed                               |
| `/episode/auto-reset`       | POST   | `{"enabled": bool, "dwell_s": 0.5..120}` (either field); `409` if unsupported                  |
| `/conveyor/belt-speed`      | POST   | `{"speed": 0..0.1}` belt speed in m/s; `409` if the scene has no conveyor belt                 |
| `/autopilot`                | POST   | `{"mode": "off" \| "drive" \| "leader"}`; `409` if the scene has no autopilot                  |
| `/leader`                   | GET    | Virtual leader pose: `seq`, `mode`, `unit`, `joint_names`, `joint_positions`                   |
| `/studio/recording`         | POST   | `{"enabled": bool, "task": str (1-200), "keep": "perfect" \| "all", "max_episodes": 0..10000}` |
| `/objects`                  | GET    | Free-object joint names and world poses                                                        |
| `/objects/{joint}/pose`     | POST   | `{"position": [x, y, z], "wxyz": [w, x, y, z]}`: teleport a free object                        |
| `/shutdown`                 | POST   | Gracefully stop the simulation owner                                                           |

Control requests are queued and applied on the next control cycle. Invalid bodies return `422`. Object coordinates must be finite and within ±2 m.

With HTTP enabled, `Ctrl+C` in the start command requests owner shutdown through HTTP. If the HTTP endpoint is unavailable, the launcher sends SIGTERM to the original owner only after confirming its registered name and PID still match. It then disconnects the CLI subscriber. With HTTP disabled, the same verified signal path stops a local owner; detached owners also exit after their configured idle timeout once all subscribers leave. Use the named `stop` command to stop an owner directly.

The `start` command also returns when its local owner exits through `stop`, HTTP, or the viewer's Shutdown control. Launcher cleanup checks the endpoint's owner name and the original owner's PID before forwarding an operator-requested shutdown. An invocation that attaches without a local owner record detaches instead of waiting indefinitely.

HTTP control has no authentication and binds to loopback by default. Use explicit non-loopback bindings only on a trusted robot-cell network. The Viser viewer exposes the same controls, including moving objects and Shutdown, to anyone who can open it; it binds to loopback by default. Use `--viser-host 0.0.0.0` only when remote access is needed on a trusted network.

## Troubleshooting

### `start` fails with `an OpenGL platform library has not been loaded into this process`

MuJoCo could not create an OpenGL context, usually because there is no desktop session (headless machine, SSH, container). Set `MUJOCO_GL=egl` before `start`, or `MUJOCO_GL=osmesa` without a GPU driver. See [Quick start](#quick-start).

### HTTP server unavailable or port already in use

- The camera/control server binds to `--http-host`/`--http-port` (default `127.0.0.1:8080`).
- If the port is taken the simulation continues without HTTP and logs a warning — pass a different `--http-port`.
- When running multiple simulations, give each a distinct `--http-port`.

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
- If **IP Camera** is missing from the camera types, enable its feature flag (see [Add the robot and cameras](#3-add-the-robot-and-cameras)).

## Scenes

The plugin ships with built-in scenes that provide different environments for the robot:

| Scene ID                      | Description                             | Free objects | Target      |
| ----------------------------- | --------------------------------------- | ------------ | ----------- |
| `single_pick_place` (default) | One block and a target disc             | 1 cube       | target disc |
| `yahtzee`                     | Six dice and a cup                      | 6 dice       | cup         |
| `conveyor_sort`               | Sort items off a moving belt into bins  | 24-item pool | 4 bins      |
| `garment_fold`                | Bimanual SO-101 with a flexible garment | garment      | none        |

Start with a specific scene:

```bash
uv run --no-sync physicalai-mujoco-so101 start --scene yahtzee
```

### Conveyor sort

`conveyor_sort` is a factory cell: items ride a belt past the arm and have to be sorted into bins before they reach the end.

- **Items**: cubes, upright cylinders and hexagonal prisms, 3 cm across, in red, blue, green or purple. About 30% are cracked (a bold dark crack texture). Unused items wait in the lidded supply crate behind the robot, and the feed moves them onto the belt inside the entry hood.
- **Bins**: red, blue and green bins plus a striped reject bin, each labelled on its floor (RED, BLUE, GREEN, REJECT) so the labels read upright in the overview camera.
- **Rule**: cracked items and purple items (which have no bin of their own) go to the striped reject bin; the others go to the bin of their color. An item that rides off the end of the belt, or comes to rest anywhere else, is a miss.
- **Episodes**: an episode feeds 10 items, 15 cm apart along the belt (so the spacing does not depend on the belt speed). Once every item is scored, the next episode starts. **Reset Scene** starts a fresh episode.
- **Stack light**: a three-lamp light on the entry hood shows green while the belt runs, amber about 2 s before the next item leaves the hood, and red while the belt is paused. `/health` reports the same lamps and the seconds until the next item (`episode.lights`, `episode.next_item_s`).
- **Belt speed**: 3 cm/s by default, adjustable from 0 to 10 cm/s in the viewer or with `POST /conveyor/belt-speed`. **Home Arm** holds the gripper above the pick zone.

At 224 px the overview camera shows item colors clearly, but not cracks: an item covers about 5 px. The wrist camera and the full-resolution overview show them.

The scene's textures and item pool (`conveyor_items.xml`) come from `scripts/generate_conveyor_assets.py`. The visual meshes for the conveyor frame, hood and bins come from `scripts/build_conveyor_meshes.py`, run with `blender --background --factory-startup --python scripts/build_conveyor_meshes.py`. Collisions use primitives in `scene.xml`, so keep the dimensions in both files in sync. Give every textured surface UV coordinates and a 2D texture: the browser viewer draws primitives in a flat color and reduces cube maps to one color per face, while the camera streams render either.

#### Scripted demonstrator

`physicalai_mujoco_so101_plugin.conveyor_demo.ConveyorDemonstrator` sorts items using privileged simulation state (item poses, the belt speed, the sorting rule). It is meant for generating consistent demonstrations, not as a policy. Each cycle, it waits over the belt, tracks the most downstream item in the pick window, grasps it with the jaw closing along the belt, and drops it in the bin the rule assigns. It writes the arm's joint targets directly and runs headless:

```bash
cd packages/physicalai-mujoco-so101-plugin
uv run python scripts/run_conveyor_demo.py --speed 0.03 --episodes 3             # score
uv run python scripts/run_conveyor_demo.py --sweep 0.01 0.03 0.05 0.07 --seeds 3  # success vs belt speed
uv run python scripts/run_conveyor_demo.py --speed 0.03 --video /tmp/demo.mp4     # overview, wrist, orbit video
```

Measured over 3 seeds of 3 episodes (90 items per speed): 99–100% of items sorted correctly at 1–5 cm/s, 81% at 7 cm/s and 48% at 10 cm/s. Above about 5 cm/s, items arrive faster than one pick-and-place cycle (about 2.3 s).

### Switching scenes at runtime

Pick a scene in the viewer's **Scene** dropdown or send `POST /scenes/{scene_id}`. The switch happens on the next control cycle:

- Loads the new scene XML and attaches the arm
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

### Changing the simulation and viewer

- **Only the simulation thread touches MuJoCo state.** Viewer and HTTP callbacks only enqueue a `SimCommand` (`http_server.py`), which `MuJoCoSO101._handle_command` applies on the next control cycle. HTTP and viewer readouts go through snapshots (`_http_status`, `_panel_state`) taken under `_state_lock`. To add a control, add a command, handle it in `_handle_command`, then add its HTTP route and its input in `viser_controls.py`.
- **Viewer inputs ignore server-side events.** Setting a Viser input's `value` from the server fires its `on_update` callbacks with `client=None`. Panel callbacks skip those events, so syncing the panel from sim state never sends a command back.
- **mjviser is used partly through its internals.** The plugin builds the viewer tabs itself, and does not use mjviser's `create_scene_gui` or `create_visualization_gui`. mjviser's camera tracking follows an arbitrary body by shifting the world, and each call registers another client-connect hook. The plugin also sets `camera_tracking_enabled`, installs a refresh handler, and moves the fixed-body handles itself. mjviser is pinned to `<0.1`; after upgrading it, check a scene switch and camera follow in a browser, because the unit tests mock viser.
- **Scene switches rebuild the viewer.** `_build_viser_gui` clears every GUI element and scene node, then builds them again for the new model. Viewer preferences that should survive a switch live on `SimControlPanel`, not on the per-build handles.
- **Scenes hold no robot.** The arm is MuJoCo Menagerie's SO-101, vendored unchanged in `urdf/robots/so101/`. Each scene marks an arm's base with a `<frame name="{prefix}robot_mount"/>`, and `robot_profile.load_scene_model` attaches the arm there with that name prefix (`left_`/`right_` in the bimanual scene). `SO101_PROFILE` pins the joint ranges, force limits and wrist camera the plugin has always used; change the arm or its wrist camera there, not in the vendored XML. Load scenes with `SceneConfig.load_model()`: `mujoco.MjModel.from_xml_path` on a scene file gives a model without the arm. `urdf/so101/*.urdf` and their meshes remain for Studio's 3D view.
- **Camera images are streamed as MuJoCo renders them.** `mujoco.Renderer` already returns upright images, so set a camera's orientation in the scene XML rather than flipping frames in code. In a MuJoCo camera frame, `-z` is the viewing direction and `+y` is the top of the image. `mirror_horizontal` exists for setups that need a mirrored feed; the `start` command does not use it.

### Compatibility with Studio

Studio pins `physicalai` to a release (`v0.2.0` at the time of writing), and the plugin runs against that version when it is installed into Studio. Run the tests against it before changing how the plugin uses Runtime APIs:

```bash
git fetch origin tag v0.2.0
git worktree add /tmp/physicalai-v0.2.0 v0.2.0
uv venv -p 3.12 /tmp/physicalai-v0.2.0-venv
VIRTUAL_ENV=/tmp/physicalai-v0.2.0-venv uv pip install --no-sources "/tmp/physicalai-v0.2.0[transport]" \
  -e packages/physicalai-mujoco-so101-plugin pytest pytest-asyncio httpx
/tmp/physicalai-v0.2.0-venv/bin/python -m pytest --import-mode=importlib packages/physicalai-mujoco-so101-plugin/tests
git worktree remove /tmp/physicalai-v0.2.0
```

`--no-sources` keeps `uv` from replacing the pinned release with this checkout.

### Checks before a pull request

- Run the repository hooks on the files you changed (`prek run --files <files>`). `prek run --all-files` also reformats unrelated files elsewhere in the repository.
- The ruff hook runs with `--unsafe-fixes` and can rewrite code, for example by collapsing a lambda into a bound method. Run the tests again after the hooks.
- Tests that step a real simulation for seconds are marked `@pytest.mark.slow`. Run the quick set with `-m "not slow"` while iterating, and the full suite before pushing.
- Run the security scans used by CI:

  ```bash
  uvx bandit -c .github/bandit_config.yml -r packages/physicalai-mujoco-so101-plugin/src
  SEMGREP_RULES="p/default p/cwe-top-25 p/trailofbits p/owasp-top-ten" uvx semgrep scan --metrics=off <changed files>
  ```

  Semgrep reports two existing `urlopen` findings in `tests/test_http_server.py`; CI only fails on new findings.

The package was imported from [physicalai-plugins PR #36](https://github.com/MarkRedeman/physicalai-plugins/pull/36) at `53bf606ef0f340a2a1821fc42a1ab6f44c33edf7`. See `NOTICE` for provenance and `CHANGELOG.md` for earlier releases. Release publishing uses this repository's shared workflows; the existing PyPI project needs a trusted publisher for `openvinotoolkit/physicalai` before its first release here.

## Current status and future improvements

Planned/desired improvements:

- More configurable camera presets (pose/FOV/fps via CLI or payload)
- Scene randomization presets (object layouts, target marker variants, textures)
- RTSP streaming sink (the frame-buffer plumbing is sink-agnostic; an RTSP
  server such as `aiortsp` or MediaMTX can consume the same buffers later)
- Optional richer sensor streams (depth/segmentation style outputs)
- Additional sample tasks and policy playback recipes in this package
