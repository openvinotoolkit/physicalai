# ROS 2 bag import: source research

Research date: 2026-10-29. This note records source findings and a validation
recipe; it is not a claim that integration tests have passed. Studio references
are pinned to `894ea9fb9c502cfa996b6cce4195a9993301f3c9`. No Git commands were used
to collect these findings. Context7 supplied documentation leads, which were
checked against the owning sources.

## Dependency and dataset contracts

- `rosbags==0.11.9` requires Python >=3.10 and reads ROS 2 without an installed ROS
  stack. SQLite3 and MCAP are supported storage formats. The plugin can generate
  genuine synthetic ROS 2 containers using `rosbags.rosbag2.Writer`, a typestore,
  and CDR serialization. This establishes CPU-only testing feasibility, without
  claiming compatibility with every ROS message type. [Package metadata][rosbags]
  [Storage support][rosbags-formats] [Writer API][rosbags-api]
- `AnyReader` merges paths into a single chronological stream. Read each logical
  bag separately to preserve episode boundaries; split storage files belonging
  to a directory remain one recording. Embedded message definitions are loaded
  automatically; older bags require a suitable default typestore. Timestamp
  values yielded by the reader are integer nanoseconds. [Reader documentation][reader]
- Studio's backend pins `lerobot[dataset,feetech]==0.6.0`, `torchcodec==0.11.1`,
  and Python >=3.12,<3.14. Its CPU extra selects torch 2.11 and torchvision 0.26
  CPU wheels. A full backend installation also includes several large policy
  extras, so an installation failure or reduced test environment must be reported
  explicitly rather than treated as complete Studio validation. [Pinned dependencies][studio-deps]
- In LeRobot **v0.6.0**, call `LeRobotDataset.create`, add each frame, call
  `save_episode` once per recording, and finally call `finalize`. A frame requires
  `task`; callers must not supply `timestamp` or `frame_index`. The writer derives
  time from `frame_index / fps`. Resample source streams before writing and keep
  ROS source timestamps in provenance rather than replacing native timestamps.
  Finalization flushes images/video and closes Parquet writers. [Dataset API][lr-dataset]
  [Frame writer and finalization][lr-writer]
- V0.6.0 declares dataset format `v3.0`; metadata includes `tasks.parquet` and
  episode Parquet shards. Some unversioned tutorials still show older filenames
  or prerelease installation commands. Use the pinned writer rather than
  manually reproducing a remembered layout. [Metadata implementation][lr-meta]

## Decoding and alignment constraints

- ROS Image `step` is the row length in bytes; image acquisition time is in its
  header. A contiguous reshape is incorrect when rows have padding. [Image schema][image-schema]
- Inspected `rosbags-image v0.10.3` ignores `step` when constructing its NumPy
  array. Its compressed path uses OpenCV `IMREAD_ANYCOLOR` and assumes BGR8 during
  color conversion. Consequently that helper alone is insufficient to guarantee
  correct padded-row and malformed-image behavior. A narrow explicit RGB decoder
  should validate row stride, buffer length, dimensions, channels and decoded
  dtype, then normalize supported raw/JPEG/PNG inputs to RGB uint8. Unsupported
  depth/Bayer/YUV inputs need explicit errors. [Pinned decoder source][rosbags-image]
- JointState identifies joints by name. Positions are radians for revolute joints
  and metres for prismatic joints, so reorder by configured names and convert
  selected components explicitly. An empty optional array is valid ROS but cannot
  satisfy a required mapped field. [JointState schema][joint-schema]
- Rosetta provides useful prior art for topic mapping, selection, unit transforms,
  receive/header timelines and age-bounded hold. Its porter also makes choices
  unsuitable to copy silently, including zero-filling absent action topics and
  continuing after failed episodes. The importer should document its own strict
  missing/stale-data behavior. Rosetta's current dataset adapter imports ROS, so
  it is a reference rather than a required dependency. [Timing contract][rosetta-contract]
  [Porter behavior][rosetta-port] [Architecture][rosetta-design]

## Resource limits must precede reader allocation

- In rosbags 0.11.9, directory `compression_mode=file` expands files into a
  temporary directory during `open` with no output-byte budget. Message mode
  calls unrestricted Zstd decompression before yielding a message. Reject those
  modes during bounded metadata preflight, before opening the library reader.
  [Pinned directory reader][container-reader]
- MCAP chunk decompression uses unrestricted Zstd/LZ4 calls and checks decoded
  length only afterward. `read_sized` accepts a 64-bit record length without a
  practical allocation cap. Unindexed `open` can decompress during metadata
  scanning. Indexed iteration retains and sorts selected messages within chunks;
  overlapping chunks can remain alive together. V1 therefore needs a bounded
  container scan before `AnyReader`, rejection of compressed chunks, validation
  of record/chunk lengths against file boundaries, and limits on counts and
  aggregate storage bytes. A per-frame limit alone does not bound the reader.
  [Pinned MCAP implementation][mcap-reader]
- SQLite iteration materializes each selected BLOB through APSW. Check storage
  size and message lengths before normal iteration. This is separate from image
  decoder limits; compressed images can expand independently of their containers.
  Hard memory guarantees require a constrained worker as well as format checks.
  [Pinned SQLite implementation][sqlite-reader]

## Actual Studio backend lifecycle

Use an isolated storage root and settings file, set before any backend imports.
`STORAGE_DIR` controls datasets, models, logs and `data/physicalai.db`;
`SETTINGS_FILE` prevents reading the user's usual settings. The public command
`physicalai-studio serve --host 127.0.0.1 --port <free-port>` configures absolute
Alembic paths, runs database migrations, establishes multiprocessing spawn, and
starts `main:app`. FastAPI lifespan starts the scheduler and event processor;
calling an adapter in isolation does not exercise these processes.
[Settings][studio-settings] [Serve entry point][studio-serve] [Lifespan][studio-lifecycle]

The real ZIP workflow is:

1. Create a project through `POST /api/projects` and a named environment through
   `POST /api/projects/{project_id}/environments`. Their schemas accept explicit
   UUIDs (required); an environment can contain empty robot/camera lists without opening
   hardware. [Project API][studio-project] [Environment schema][studio-environment]
2. `POST /api/projects/{project_id}/imports/datasets:prepare`, form fields
   `format_hint=auto` and nonempty `dataset_name`.
3. `PUT /api/projects/{project_id}/imports/datasets/{job_id}:upload`, multipart
   field `archive` with the completed dataset ZIP.
4. Poll `GET /api/jobs/{job_id}` until `payload.step=awaiting_user_review` and
   verify the validation report and inferred schema.
5. `POST /api/projects/{project_id}/imports/datasets/{job_id}:finalize`, JSON
   `environment_id` and `default_task`; poll until completed, capture
   `payload.result_dataset_id`, and fail on terminal failure or timeout.
   [Import API][studio-import-api] [Job schema][studio-import-schema]
6. Verify the persisted dataset through dataset/episode/video APIs, and read its
   actual imported path for downstream training. The worker commit phase
   extracts to Studio-managed storage and persists through `DatasetService`.
   Draft detection alone is not a completed import. [Worker][studio-import-worker]
   [Dataset API][studio-dataset-api]

Minimal creation bodies are `{"id": "<project-uuid>", "name": "Rosbag smoke"}`
and `{"id": "<environment-uuid>", "name": "Offline import", "robots": [],
"cameras": []}`. Dataset routes use the **singular** `/api/dataset` prefix:
`GET /api/dataset/{id}`, `GET /api/dataset/{id}/episodes`,
`GET /api/dataset/{id}/episodes/{episode_index}`, and
`GET /api/dataset/{id}/episodes/{episode_index}/thumbnail?width=64&height=64`.
The optional thumbnail `camera` query selects a camera feature. Video retrieval
uses `GET /api/dataset/{id}/video/{relative_video_path}`. Check the returned
PNG/video bytes, not only an HTTP success status. [Dataset API][studio-dataset-api]

The existing local CLI also provides `physicalai-studio datasets import-dir`
with `--source-dir`, `--project-id`, `--environment-id`, `--dataset-name`, and
optional `--default-task`; copy is the default. This is useful additional
coverage but does not replace the scheduled ZIP lifecycle above.
[Directory import CLI][studio-import-cli]

## Tiny CPU training, export and Runtime smoke recipe

The following is a proposed low-cost source-compatible configuration, not a
recorded test result. Use the **imported** dataset path, not the converter's
original directory. Disable pretrained downloads explicitly.

```python
from physicalai.data import LeRobotDataModule
from physicalai.policies import ACT
from physicalai.train import Trainer

data = LeRobotDataModule(
    root=imported_dataset_path,
    train_batch_size=2,
    num_workers=0,
    persistent_workers=False,
    pin_memory=False,
    val_split=0.0,
)
policy = ACT(
    chunk_size=2,
    n_action_steps=2,
    pretrained_backbone_weights=None,
    image_size=(64, 64),
    dim_model=32,
    n_heads=4,
    dim_feedforward=64,
    n_encoder_layers=1,
    n_decoder_layers=1,
    use_vae=False,
    compile_model=False,
)
trainer = Trainer(
    accelerator="cpu",
    devices=1,
    max_steps=2,
    logger=False,
    enable_checkpointing=False,
    enable_progress_bar=False,
    default_root_dir=isolated_training_dir,
)
trainer.fit(policy, datamodule=data)
trainer.save_checkpoint(checkpoint_path)
policy = ACT.load_from_checkpoint(checkpoint_path, map_location="cpu").eval()
policy.export(export_path, backend="openvino")
```

The datamodule accepts a local root and defaults to PhysicalAI data formatting;
ACT `setup` initializes its features/statistics from it and adjusts the dataset
for the policy. Trainer injects policy/dataset integration behavior. Confirm at
least one finite loss and actual optimizer progress, reload the saved checkpoint,
and export that trained model. [Datamodule][studio-data] [ACT constructor/setup][studio-act]
[Trainer][studio-trainer] [Export API][studio-export]

Load the exported directory through Runtime `InferenceModel.from_pretrained` and
run a CPU prediction using the exported input schema and a real imported frame.
Assert finite action values and expected shape. Save logs, input/output shapes,
versions, checkpoint/export paths and exact commands. This establishes format
interoperability, not useful learned behavior or physical robot safety.
[Runtime inference implementation](../../../src/physicalai/inference/model.py)

Studio's own integration test constructs the compatible inference input with
`FormatConverter` (below); the smoke must apply it to the imported datamodule
and compare against the trained/reloaded policy, rather than creating a fresh
policy for export. [Existing integration test][studio-e2e]

```python
import numpy as np
from physicalai.data.lerobot import FormatConverter
from physicalai.inference import InferenceModel

batch = next(iter(data.train_dataloader()))
observation = FormatConverter.to_observation(batch)[0:1].to("cpu")
inference_input = observation.to_numpy().to_dict(flatten=False)
runtime = InferenceModel.from_pretrained(export_path)
action = runtime.select_action(inference_input)
assert np.isfinite(action).all()
assert action.shape[-1] == expected_action_width
```

## Public recordings: evidence and outstanding gap

No small, suitably licensed physical-hardware ROS 2 recording containing all
supported camera, joint and command types was verified during this research.
Do not claim complete real-world E2E coverage from a camera-only bag or simulation.

- ROS Physical AI's published SO-ARM101 bags are explicitly recorded in Gazebo.
  The linked converted dataset has two cameras, six joint positions and six
  commands. The raw bags are currently linked through Google Drive. This is useful
  independent simulation evidence, not physical-hardware evidence. [Demo provenance][demo]
- CORESENSE's real TIAGo RoboCup data includes Image, JointState and `/cmd_vel`
  Twist, but Twist is outside the narrow numeric-array v1. The Carry My Luggage
  artifact is `carry_my_luggage.zip`, 17.2 GB, MD5
  `41335d6a5c3b772e59e198a923bfa53a`. The Receptionist artifact
  `rosbag2_2024_07_18-13_56_14_0.db3` is 9.2 GB, MD5
  `1d119791456e676c1d4a242a860d7f5b`. The retrieved pages did not expose a dataset
  license value; do not infer redistribution rights from the paper license.
  Neither artifact was downloaded. [Luggage record][luggage] [Receptionist record][receptionist]

## Finding from executed integration: singleton actions

The pinned LeRobot writer stores features of shape `(1,)` as scalar Arrow values.
The pinned Studio episode API requires `actions: list[list[float]]`, and its
adapter passes those scalar rows through unchanged. A real imported one-action
dataset therefore fails episode preview validation. Conversion and LeRobot reload
remain valid; the converter warns and does not fabricate extra dimensions. The
worked acceptance demonstration uses two independently recorded command channels.
This upstream interoperability limitation needs a separate Studio fix.
[LeRobot feature conversion](https://github.com/huggingface/lerobot/blob/v0.6.0/src/lerobot/datasets/feature_utils.py)
[Studio episode schema](https://github.com/open-edge-platform/physical-ai-studio/blob/894ea9fb9c502cfa996b6cce4195a9993301f3c9/application/backend/src/schemas/dataset.py)

## Findings from the final converter review

- Rosbags 0.11.9 opens each metadata storage path using its basename. Preflight
  must check the same file: v1 rejects nested storage paths and accepts the
  documented uncompressed `none` marker. [Directory reader][container-reader]
- LeRobot 0.6.0's streaming writer is designed for live capture and drops frames
  when a bounded queue remains full for 100 ms. Offline conversion must wait for
  the consumer instead. The plugin isolates that adaptation in `encoding.py`,
  retains bounded queues, and tests slow and failed encoder threads.
- The same streaming encoder returns no image statistics for one-frame episodes.
  The adapter retains at most one frame per camera to compute those statistics
  with LeRobot's own statistics implementation. [Streaming encoder][lr-video]
- LeRobot can download from the Hub if local metadata/video files are missing.
  Final validation now checks episode metadata and every referenced local file
  before the full dataset constructor. Regression tests prohibit Hub fallback
  while removing published files. [Dataset API][lr-dataset]

[lr-video]: https://github.com/huggingface/lerobot/blob/v0.6.0/src/lerobot/datasets/video_utils.py

## Sources

[rosbags]: https://pypi.org/project/rosbags/0.11.9/
[rosbags-formats]: https://ternaris.gitlab.io/rosbags/topics/rosbag2.html
[rosbags-api]: https://ternaris.gitlab.io/rosbags/api/rosbags.rosbag2.html
[reader]: https://ternaris.gitlab.io/rosbags/topics/highlevel.html
[lr-dataset]: https://github.com/huggingface/lerobot/blob/v0.6.0/src/lerobot/datasets/lerobot_dataset.py
[lr-writer]: https://github.com/huggingface/lerobot/blob/v0.6.0/src/lerobot/datasets/dataset_writer.py
[lr-meta]: https://github.com/huggingface/lerobot/blob/v0.6.0/src/lerobot/datasets/dataset_metadata.py
[image-schema]: https://github.com/ros2/common_interfaces/blob/rolling/sensor_msgs/msg/Image.msg
[joint-schema]: https://github.com/ros2/common_interfaces/blob/rolling/sensor_msgs/msg/JointState.msg
[rosbags-image]: https://gitlab.com/ternaris/rosbags-image/-/raw/v0.10.3/src/rosbags/image/image.py
[rosetta-contract]: https://github.com/iblnkn/rosetta/blob/main/docs/reference/contract.md
[rosetta-port]: https://github.com/iblnkn/rosetta/blob/main/docs/reference/rosetta-port.md
[rosetta-design]: https://github.com/iblnkn/rosetta/blob/main/docs/explanation/design.md
[container-reader]: https://gitlab.com/ternaris/rosbags/-/raw/v0.11.9/src/rosbags/rosbag2/reader.py
[mcap-reader]: https://gitlab.com/ternaris/rosbags/-/raw/v0.11.9/src/rosbags/rosbag2/storage_mcap.py
[sqlite-reader]: https://gitlab.com/ternaris/rosbags/-/raw/v0.11.9/src/rosbags/rosbag2/storage_sqlite3.py
[studio-deps]: https://github.com/open-edge-platform/physical-ai-studio/blob/894ea9fb9c502cfa996b6cce4195a9993301f3c9/application/backend/pyproject.toml
[studio-settings]: https://github.com/open-edge-platform/physical-ai-studio/blob/894ea9fb9c502cfa996b6cce4195a9993301f3c9/application/backend/src/settings.py
[studio-serve]: https://github.com/open-edge-platform/physical-ai-studio/blob/894ea9fb9c502cfa996b6cce4195a9993301f3c9/application/backend/src/cli/serve.py
[studio-lifecycle]: https://github.com/open-edge-platform/physical-ai-studio/blob/894ea9fb9c502cfa996b6cce4195a9993301f3c9/application/backend/src/core/lifecycle.py
[studio-project]: https://github.com/open-edge-platform/physical-ai-studio/blob/894ea9fb9c502cfa996b6cce4195a9993301f3c9/application/backend/src/api/project.py
[studio-environment]: https://github.com/open-edge-platform/physical-ai-studio/blob/894ea9fb9c502cfa996b6cce4195a9993301f3c9/application/backend/src/schemas/environment.py
[studio-import-api]: https://github.com/open-edge-platform/physical-ai-studio/blob/894ea9fb9c502cfa996b6cce4195a9993301f3c9/application/backend/src/api/dataset_import.py
[studio-import-schema]: https://github.com/open-edge-platform/physical-ai-studio/blob/894ea9fb9c502cfa996b6cce4195a9993301f3c9/application/backend/src/schemas/dataset_import_job.py
[studio-import-worker]: https://github.com/open-edge-platform/physical-ai-studio/blob/894ea9fb9c502cfa996b6cce4195a9993301f3c9/application/backend/src/workers/dataset_import_worker.py
[studio-dataset-api]: https://github.com/open-edge-platform/physical-ai-studio/blob/894ea9fb9c502cfa996b6cce4195a9993301f3c9/application/backend/src/api/dataset.py
[studio-import-cli]: https://github.com/open-edge-platform/physical-ai-studio/blob/894ea9fb9c502cfa996b6cce4195a9993301f3c9/application/backend/src/cli/datasets.py
[studio-data]: https://github.com/open-edge-platform/physical-ai-studio/blob/894ea9fb9c502cfa996b6cce4195a9993301f3c9/library/src/physicalai/data/lerobot/datamodule.py
[studio-act]: https://github.com/open-edge-platform/physical-ai-studio/blob/894ea9fb9c502cfa996b6cce4195a9993301f3c9/library/src/physicalai/policies/act/policy.py
[studio-trainer]: https://github.com/open-edge-platform/physical-ai-studio/blob/894ea9fb9c502cfa996b6cce4195a9993301f3c9/library/src/physicalai/train/trainer.py
[studio-export]: https://github.com/open-edge-platform/physical-ai-studio/blob/894ea9fb9c502cfa996b6cce4195a9993301f3c9/library/src/physicalai/export/mixin_policy.py
[studio-e2e]: https://github.com/open-edge-platform/physical-ai-studio/blob/894ea9fb9c502cfa996b6cce4195a9993301f3c9/library/tests/integration/test_first_party_e2e.py
[demo]: https://github.com/ros-physical-ai/demos/blob/ea931a9fc58de19d9c9bea0fc7dc99da1f30274b/docs/demos/pretrained-demo.md
[luggage]: https://zenodo.org/records/13748065
[receptionist]: https://zenodo.org/records/13902011
