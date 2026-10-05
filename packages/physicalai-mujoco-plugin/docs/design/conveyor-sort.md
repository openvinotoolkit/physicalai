# Design: `conveyor_sort` scene

Status: implemented. The scene, the [scripted demonstrator](#scripted-demonstrator) and [automatic Studio episodes](#autopilot-virtual-leader-and-automatic-studio-episodes) exist, and an ACT policy trained on about 100 recorded episodes runs from Studio. Still open: pi0.5 rule templates, recovery from drops, and how cracks become visible to the policy.

## Goal

Add a harder single-arm task than `single_pick_place`: items move along a conveyor belt and the SO-101 sorts them into bins. It should:

- Show a moving world, where Sync, Async and RTC execution behave differently. This makes it a benchmark for Runtime, not only for the policy.
- Give ACT and a VLA (pi0.5) each a task they are good at, in the same scene, with nothing special-cased.
- Look like a factory cell, so it works as a demo.

## Task lineup (agreed)

| Tier              | Scene                        | Model(s)                             | What it shows                                       |
| ----------------- | ---------------------------- | ------------------------------------ | --------------------------------------------------- |
| Simple            | `single_pick_place` (exists) | ACT; pi0.5 as a baseline             | Something working right away                        |
| Harder            | `conveyor_sort` (this note)  | ACT sorts by color; pi0.5 by prompt  | Reacting to motion; language rules; execution modes |
| Optional, for fun | `tic_tac_toe` (later)        | VLM plans moves, pi0.5 places pieces | Interactive play against a human                    |

Other scenes kept: `yahtzee` (extra), `garment_fold` (bimanual). Removed in #296: `pick_lift`, `pick_place`, `garment_fold_mesh`.

## Scene contents

- **Belt:** runs past the arm within comfortable reach (roughly 20–25 cm from the base). Must be inside the overview camera view and reachable by the wrist camera.
- **Bins:** 2–3 color bins plus a **reject bin**, all within reach. Items that reach the end of the belt fall into an end-of-belt catch, which counts as a miss.
- **Items:**
  - **Colors:** distinct hues far apart (red, blue, yellow, maybe purple). Avoid green, which the existing targets use. Check the colors under the scene lighting and shadows. A color with no bin of its own (it goes to reject) makes rules more interesting.
  - **Shapes:** cube, cylinder standing on its flat end, hexagonal or triangular prism. **No spheres**: they roll off the belt and out of the gripper. Cylinders on their side roll too.
  - **Size:** about 2–4 cm, so they fit the gripper.
  - **Cracks:** a random subset is "cracked", done with a **texture** (thick, high-contrast dark lines), not geometry. Before collecting data, check they are visible in a **224 px downscaled** overview frame.

## Belt speed and difficulty

- **Belt speed setting**, exposed in the viewer, over HTTP and on the CLI. Suggested test points: 1, 2, 3 and 5 cm/s.
- **The headline result is success rate against belt speed**, for each model and execution mode. There is no separate stop-and-go mode for VLAs; everything runs on the same continuous belt.
- **Keep the speed the same in training and evaluation.** Neither ACT nor pi0.5 looks at more than one frame, so neither can measure how fast the belt is moving.
- **Optional debug setting:** freeze physics while inference runs. This separates "the policy is bad" from "the policy is too slow". Use it for diagnosis, not for headline results.
- **Optional:** put the belt speed in the robot state, or feed two frames, so the model gets velocity information.

## Model expectations

- **ACT:** a fixed rule, for example cracked items to reject and everything else by color. Inference is fast (about 10–30 ms), so re-plan often: `SyncExecution` or `AsyncExecution` with a high `request_threshold` (about 0.9, which re-plans every ~10 of 100 actions) plus `LerpSmoother`. Runtime has no ACT-style temporal ensembling, and ensembling that weights older predictions more would react late to moving items anyway.
- **pi0.5:** rules given by prompt and changed live through `PolicySource.set_task()`: "sort by shape", "reject cylinders", "red goes to bin A now". **Every rule template must appear in the training data.** Hold out some combinations of rules for testing. Use `RTCExecution` and tune `execution_horizon` (default 15 at 30 fps); a smaller value re-plans more often. Things that lower latency: fewer denoising steps, one camera or lower resolution, INT8, an Intel GPU.
- **Rough numbers:** at about 250 ms of inference, a 2–3 cm/s belt moves about 1–1.5 cm per re-plan, which is manageable. Sync mode pauses the arm during inference while the belt keeps moving, so the timing the policy learned from demos is wrong.

## Demos (training data)

- Human demos of catching moving items are the bottleneck; inconsistent demos produce averaged behavior. Prefer a **scripted demonstrator** that reads the sim state, with **template instructions**, for consistent, large datasets.
- For demos, fix a pick order (for example, the item closest to the end of the belt first). ACT averages demos that solve the same situation in different ways.
- **Episodes:** an episode is N items, then a reset. A continuous stream doesn't fit episode-based recording.

## Implementation notes

- **Belt physics.** Start with a contact-driven velocity: each step, set the velocity of free bodies touching the belt surface. It's simple and gives an exact speed. Switch to motor-driven rollers (cylinders with velocity actuators) if the physics looks wrong.
- **Item pool.** MuJoCo can't add bodies at run time. Use a fixed pool of free bodies, park unused ones off-screen, and teleport them to the start of the belt at set or random intervals. The existing object-pose code (`/objects/{joint}/pose`, `spawn.place_freejoint`) already does this kind of teleport.
- **Randomization on respawn:** color, shape (pre-built bodies for each shape, or several pools), cracked or not, spacing, and a small lateral offset.
- **Scoring from the sim state:** count correct bins, wrong bins and misses per episode. Report them in `/health` and in the viewer's Episode panel, like the `single_pick_place` auto-reset status.
- **Camera:** use one standard overview camera pose. Scenes used to differ, so data from one scene didn't transfer to another.
- **Follow the plugin conventions** (see the README's development section):
  - only the sim thread touches MuJoCo state
  - viewer and HTTP inputs enqueue a `SimCommand`
  - new controls need a command, a handler in `_handle_command`, an HTTP route and a viewer input
  - the robot is defined in three XML files, so arm and camera changes go into all three
- **Registry:** add a `SceneConfig` and a reset function in `scene_registry.py`, and extend the scene tests, which load real models without a display.

## Proof of concept (2026-09-25)

What exists:

- **Scene:** `urdf/scenes/conveyor_sort/scene.xml`, registered as `conveyor_sort`.
- **Controller:** `src/.../conveyor.py` (`ConveyorSort`). It plugs in where `EpisodeAutoReset` does and has the same interface (`update`, `status`, `set_active`, `notify_manual_reset`), plus `set_belt_speed`.
- **Controls:** a Conveyor folder in the viewer (belt running, belt speed 0–10 cm/s, live score) and `POST /conveyor/belt-speed`. Pausing reuses `POST /episode/auto-reset`.
- **Assets:** textures and the item pool come from `scripts/generate_conveyor_assets.py`. The visual meshes come from `scripts/build_conveyor_meshes.py`, run with Blender headless or live through BlenderMCP. Collisions are primitives in `scene.xml`.
- **Tests:** `tests/test_conveyor.py` covers belt speed, wrap, misses, the scoring rule for every bin, pausing, held items, and settings surviving scene switches. There are also HTTP tests.

Decisions made while building, including changes from the plan above:

- **Belt:** a slide-jointed slab driven by a velocity actuator, not contact-velocity overrides. Items ride by friction, so a gripper pulling an item off the belt feels real resistance. The slab wraps back by one 2 cm texture tile, which is seamless. Items match the belt speed to within 0.3 mm/s at 1–5 cm/s and don't tip over.
- **Belt height 6 cm, not 3.5 cm.** With a low belt, items that fell off the end lay under the belt's end overhang, got pinned, and stalled the belt. At 6 cm the belt underside clears items lying in the catch tray.
- **Layout:** belt along y at x = 0.25 m, flowing from +y (entry hood) to −y (drive end and catch tray). Four axis-aligned bins sit 18–21 cm from the base: red (0.13, 0.13), blue (0.03, 0.21), green (0.13, −0.13) and reject (0.03, −0.21). They're axis-aligned because angled bins at ±100° fell partly outside the standard overview frame.
- **Colors:** red, blue, **green** and purple, with purple as the color that has no bin. The note above said to avoid green, but there's no green target disc in this scene, and **yellow would be confused with the yellow arm**.
- **Cracks aren't visible at 224 px from the standard overview pose.** A 3 cm item covers about 5 px there (16 px at 640 px). Cracks are bold in the wrist camera and visible in the full-resolution overview. Options: rely on the wrist camera for inspection, which is realistic; add a third "inspection" camera over the belt, which needs the plugin to stream more than wrist and overview; or use bigger items.
- **Item pool:** 24 items (3 shapes × 4 colors × plain or cracked). Unused items wait in a lidded supply crate behind the robot, which is outside the ±110° pan range and outside the overview view. The feed teleports them into the entry hood, so their appearance is hidden.
- **Feed:** items are spaced 15 ± 3 cm apart _along the belt_, measured as belt travel. The spacing in space is then the same at every speed, and only the time between items changes.
- **Scoring:** an item is scored once it rests below 4.5 cm for 0.5 s. Items that are moving, held, or still on the belt aren't scored. The pick-place "success dwell" setting is deliberately ignored by the conveyor.
- **Textures are UV-mapped meshes.** mjviser 0.0.14 draws primitives in a flat color and reduces cube maps to one color per face, so the belt, the cracked items and the reject bin first rendered white in the browser viewer, even though the camera streams were fine. Cracked items now carry a UV-mapped visual mesh over an invisible collision primitive, so their physics is unchanged. The belt has a UV-mapped visual mesh on the moving slab, and the bin mesh has box-projected UVs. `test_conveyor.py` checks that no textured primitive or cube map comes back.
- **Home pose:** shoulder_lift −1.4, elbow_flex 0.7, wrist_flex 1.6. The gripper hovers about 8 cm above the near rail. Reset also applies it.

### Teleoperation aids (2026-09-28)

- **Item timing:** items are 15 ± 3 cm apart along the belt: 5 s at 3 cm/s, 7.5 s at 2 cm/s. Teleoperating works but isn't easy at 3 cm/s.
- **Stack light on the hood:** green = belt running, amber = the next item leaves the hood within 2 s, red = paused. The lit lamps are mocap bodies moved over dim lenses, because the browser viewer syncs poses but not colors. Unlit lamps wait inside the drive motor's can, not under the floor, which the browser viewer draws as a see-through grid. The overview camera sees the light too, and training and evaluation both have it.
- **Wrist camera:** kept on the side mount, as on the physical SO-101. At neutral wrist roll the belt appears vertical, and the view is level with the wrist rolled about 90°, which is how the demonstrator grasps. A mount behind the fixed jaw would give a level view at neutral roll, but the jaw then hides the item being grasped.
- **Cameras render on their own thread.** MuJoCo's render call releases the GIL: a render thread running flat out made 130 frames/s while physics kept 99% of its speed. The control loop now publishes a pose snapshot each tick, and a camera thread renders from its own `MjData`. Both scenes went from 0.55–0.88x to **1.00x real time** with two 30 fps cameras, with no mesh changes. The viewer's Performance folder and `/health` `timing` show the real-time factor, control rate and camera frame rates.
- **Rendering:** the floor reflection is off (it cost ~2.8 ms of ~10.9 ms per frame). MuJoCo's classic renderer runs on macOS through OpenGL 2.1 translated to Metal. It is limited by geometry passes, not pixels or GPU power. MuJoCo 3.14's experimental Filament renderer (Metal-native) is worth trying.

### Scripted demonstrator

`conveyor_demo.ConveyorDemonstrator` uses privileged state and writes the arm's joint targets directly. `scripts/run_conveyor_demo.py` scores it, sweeps belt speeds, or records a video. Its episodes reach Studio through a virtual leader (see [below](#autopilot-virtual-leader-and-automatic-studio-episodes)); a headless dataset writer is only needed later, for DAgger.

| Belt speed                  | 1 cm/s | 2 cm/s | 3 cm/s | 5 cm/s | 7 cm/s | 10 cm/s |
| --------------------------- | ------ | ------ | ------ | ------ | ------ | ------- |
| Sorted correctly (90 items) | 100%   | 98.9%  | 100%   | 100%   | 81.1%  | 47.8%   |

What it took to get there. Each of these is also a constraint for learned policies and for human teleoperation:

- **IK:** position is the primary task, with "fingers down, jaw at yaw" in the null space. Joints at a limit drop out (an active set), and each iteration's step is capped. The SO-101 has five arm joints and cannot always point straight down: the gripper tilts up to ~16° at the ends of the belt.
- **Pick window y ∈ [−0.15, 0.10]:** further upstream the gripper tilts 4–5°, and cubes get squeezed out sideways into the far rail.
- **Close along the belt:** the moving jaw pushes the item about 1 cm toward the fixed jaw. Across the belt, that push shoved items into a rail.
- **Two jaw variants (180° apart):** the demonstrator prefers the one that is upright enough and within the wrist's range, then the smaller wrist turn. A tilted gripper dips the open moving jaw's tip into the belt.
- **Stiff belt drive (kv 500, ±500 N):** closing presses the item into the belt, and with the first drive (kv 20, ±20 N) that friction pushed the belt _backwards_ at up to 0.28 m/s, spilling new items out of the back of the hood.
- **Item damping (0.0002):** without it, cylinders wobbled in the bins forever like spinning coins.
- **Scoring:** an item that is off the belt and out of the gripper for 10 s is scored wherever it is, for example balanced on a bin rim. An episode always ends.

Next steps: prompt templates for pi0.5, recovery from drops (disturbances while recording), and a decision on how cracks become visible to the policy.

## Open questions

1. ~~Where the belt and bins go~~: settled in the proof of concept.
2. ~~How the belt is driven~~: a slide-jointed slab (see the proof of concept).
3. ~~How items enter~~: spacing along the belt, one item at a time.
4. The exact rule templates for pi0.5, and which combinations to hold out.
5. ~~What the scripted demonstrator needs~~: see [Scripted demonstrator](#scripted-demonstrator).
6. ~~How cracks are textured~~: one cube-mapped texture per color. What's still open is how cracks become visible at 224 px.

## Autopilot, virtual leader and automatic Studio episodes

The scripted demonstrator runs inside the simulation loop (`autopilot.py`) in one of two modes:

- **drive**: it writes the actuators and the simulation ignores client actions.
- **leader**: it only publishes its targets on `GET /leader`, in the follower's units.

The plugin's Studio catalog adds a **MuJoCo SO-101 Virtual Leader** (`role="leader"`, `virtual_leader.py`). This is a read-only robot that polls `/leader` over one keep-alive loopback connection. Studio's teleoperation sends what it reads straight to the follower as the action, so the recorded actions are the demonstrator's commands, exactly as with a human on a leader arm. The leader imports no MuJoCo, because Studio's environment carries its own version. Its timestamp advances only when the simulation ticked, so a frozen simulation reads as a stalled leader.

The demonstrator already integrates its IK from its own previous target rather than from measured joints. So the round trip through Studio adds lag, not oscillation. With targets sampled at 30 Hz and applied 60 ms late (`run_conveyor_demo.py --action-hz 30 --latency-ms 60`), 60/60 items were sorted with no failed grasps. At 120 ms, 60/60 were still sorted, with 22 grasp retries.

Studio has no episode REST API, so `studio_recorder.py` joins the runtime websocket that the recording view uses:

1. `GET /api/runtime/sessions` finds the live session for this simulation's follower (matched on `payload.name` = owner name). The session's `leader_name` gives the leader.
2. The handshake repeats the follower and leader (Studio's identity digest covers them) with `camera_ids: []`. An empty camera list never restarts the session, and an empty claim neither pins nor releases the UI's cameras.
3. The link never sends `disconnect`, which would stop Studio's session.

`AutoRecorder` is pure logic, called each tick; it holds the feed through `ConveyorSort.set_feed_hold`, which is separate from the user's pause:

- **waiting**: until the dataset is loaded, teleoperation is on and no recording is in progress.
- **starting**: park the items, send `start_recording`, and hold the belt until Studio reports `is_recording`.
- **recording**: when the conveyor episode ends, hold the belt and send `save_episode`, or `discard_episode` for mistakes when keeping only perfect episodes. Then go back to waiting.

If Studio stops the recording itself, or the link fails, the automation stops and releases the belt. Studio's URL is a launch option (`--studio-url`), never an HTTP request field.

Tested against a fake Studio on a real websocket, including a full episode on the real conveyor. Live discovery against a local Studio reports the missing session correctly. An end-to-end run with Studio's UI still needs this plugin reinstalled into Studio's environment.
