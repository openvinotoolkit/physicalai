# ROS bag converter

Convert recorded ROS 2 demonstrations to LeRobot v3 for Physical AI Studio, without
ROS, cameras, or a robot. Conversion is offline; it does not connect to hardware.

## Install and run

From a Runtime source checkout, use an isolated Python 3.12 environment:

```bash
uv venv --python 3.12 .venv-rosbag
uv pip install --python .venv-rosbag/bin/python -e . -e 'packages/physicalai-rosbag-plugin[tests]' --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
uv run --no-project --python .venv-rosbag/bin/python python packages/physicalai-rosbag-plugin/examples/generate_demo.py /tmp/rosbag-demo
.venv-rosbag/bin/physicalai rosbag convert --input /tmp/rosbag-demo/bags --mapping /tmp/rosbag-demo/mapping.yaml --output /tmp/rosbag-demo/converted --repo-id local/demonstrations
```

`pai rosbag convert` is equivalent. Outputs must not already exist. The plugin
requires a usable CPU H.264 encoder in PyAV, plus FFmpeg for LeRobot's video-file
operations. On Fedora, provide an FFmpeg build with H.264 support. Publication
uses Linux's atomic no-replace rename; Windows uses its non-replacing rename.
Other operating systems currently fail explicitly instead of risking replacement.

The plugin is excluded from Runtime's default UV workspace to isolate LeRobot and
Torch dependencies. Do not use root `uv sync` to install this plugin.

## Mapping

```yaml
schema_version: 1
fps: 30
task: Move the object
robot_type: so101
max_age_ms: 100
state:
  - name: shoulder_pan.pos
    topic: /joint_states
    message_type: sensor_msgs/msg/JointState
    field: position
    joint: shoulder_pan
    conversion: rad_to_deg
action:
  - name: shoulder_pan.pos
    topic: /commands
    message_type: std_msgs/msg/Float64MultiArray
    field: data
    index: 0
    conversion: rad_to_deg
cameras:
  - name: front
    topic: /camera/image_raw
    message_type: sensor_msgs/msg/Image
```

State/action lists define feature order and `names` metadata exactly. Use names
such as `shoulder_pan.pos` for joint positions; Studio strips the `.pos` suffix
when inferring the robot's joints. Use consistent joint names across state/action.

**Studio 0.3 / LeRobot 0.6.0 limitation:** one-component actions are stored as scalars
by LeRobot and Studio's episode preview expects action vectors. Such datasets remain
readable by LeRobot, but Studio episode preview is not supported for them. Use the
actual multi-dimensional robot mapping; never add fabricated components to bypass
this limitation. The worked example records separate shoulder and gripper commands.
Select JointState fields (`position`, `velocity`, `effort`) by joint name, never
by implicit message ordering. Empty selected arrays are errors. Numeric commands
support Float32/Float64 scalars and Float32MultiArray/Float64MultiArray flat `data`
indices. No custom message definitions or executable mapping expressions are accepted.

Conversions are `identity` (default), `rad_to_deg`, or `deg_to_rad`. Alternatively
use `scale` and `offset` for `output = input * scale + offset`; non-identity named
conversion cannot be combined with nondefault scale/offset. Configure each joint
separately: prismatic joints and grippers need not use angular units.

Each component/camera can override `max_age_ms`. Repeated references to one topic
must agree on type and maximum age. Unsupported and misspelled fields fail validation.

Python callers can use `MappingConfig.from_yaml(Path(...))` and
`convert_rosbags(input_dir, mapping, output, repo_id)`. `ConversionResult` contains
the final output path, episode count, and frame count. Calls raise on failure;
the CLI reports the failure with a nonzero exit code.

## Recording and timing contract

Input is one recording directory with `metadata.yaml`, or a parent of such
directories. Recordings are sorted by name; each becomes one episode. Split
storage files stay in that episode. Only uncompressed SQLite3/MCAP containers
are accepted. Bag file/message compression and compressed MCAP chunks are rejected
**before** opening the reader. JPEG/PNG camera payloads remain supported.
Storage files must sit beside `metadata.yaml`; nested metadata paths are rejected
because the pinned reader resolves them by basename. Both an empty compression
mode and the explicit `none` marker identify uncompressed recordings.

All topics use bag-record timestamps in integer nanoseconds. Set `t0` to the latest
first required sample and `t_end` to the earliest last required sample. For integer
FPS from 1 through 1,000, sample at `t0 + (n * 1_000_000_000) // fps` while the result
is at most `t_end`. Select the latest sample at or before each tick. Sample age
equal to the limit is accepted; greater age fails. No interpolation, future
samples, automatic units, zero-filling, dropped internal frames, or skipped episodes.

Identical payloads at duplicate topic/timestamps collapse; conflicting duplicates
and timestamp regressions fail. LeRobot writes its own timestamps `frame_index/fps`,
starting at zero per episode. `conversion.json` records the mapping and source intervals.

## Images, resources, and failures

Raw images support rgb8, bgr8, rgba8, bgra8, and mono8, including padded rows.
CompressedImage supports 8-bit grayscale/RGB JPEG and PNG. Depth, Bayer/YUV, and
other encodings fail explicitly. Dimensions remain fixed per camera across all
episodes and must be even for H.264; conversion does not silently resize images.

Optional `limits` mapping defaults:

| Key              |   Default |
| ---------------- | --------: |
| `image_pixels`   |  16000000 |
| `message_bytes`  |  67108864 |
| `chunk_bytes`    | 268435456 |
| `episode_frames` |    100000 |
| `total_frames`   |   1000000 |

Metadata and MCAP metadata records are limited to 16 MiB; the mapping file is
limited to 1 MiB. Container lengths and storage path containment are checked before
reading. Image dimensions are checked before decompression. Treat source recordings
as reviewed local inputs, keep them unchanged during conversion, and use OS process
isolation for hostile inputs; these limits are not a native-code sandbox.

The converter keeps latest topic samples and bounded video queues, not all decoded
images. Slow encoders block conversion instead of dropping frames. One-frame
episodes retain camera statistics. LeRobot buffers numeric data per episode, so memory grows with episode
length and feature count. Video encoders and MCAP indexes consume additional memory.
Two passes validate messages and then write frames; allow disk space for the whole
dataset plus staging. Reduce episode length or limits for constrained machines.

Output is staged in a sibling `.NAME.rosbag-*` directory, finalized, checked,
published without replacement, and reopened from its final path. Failures clean up
only this invocation's files. CLI SIGINT/SIGTERM follow cleanup; SIGKILL, power loss,
or an OS crash can leave an identifiable staging directory, or a published directory
whose final validation did not finish. Inspect ownership and contents before removing
abandoned artifacts manually; directory existence alone is not proof of success.

## Studio and verification

ZIP the converted directory and use Studio's existing dataset upload/import UI.
The backend discovers cameras and joints, pauses for review, and registers the
dataset after environment/task finalization. For large local datasets use Studio's
`datasets import-dir` CLI in copy mode. A successful file conversion alone does not
prove the complete Studio workflow.

```bash
uv run --no-project --python .venv-rosbag/bin/python python -m pytest --import-mode=importlib -q packages/physicalai-rosbag-plugin/tests
bash packages/physicalai-rosbag-plugin/examples/studio_smoke.sh /tmp/rosbag-studio-smoke
```

The opt-in smoke script uses a pinned Studio snapshot, actual backend HTTP endpoints,
SQLite database and worker processes, preview endpoints, directory import, and tiny
CPU ACT training/export/Runtime inference. It downloads substantial dependencies;
it never requires a robot or GPU. See the script's generated report and logs for
executed results. A public real-hardware recording has not been validated; synthetic
success does not establish compatibility with arbitrary robot recordings.
