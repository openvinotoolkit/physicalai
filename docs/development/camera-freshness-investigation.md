# Camera freshness and renderer consistency

## Status

The Runtime capture fix prevents consumer/publication pacing from stalling IP stream
acquisition. Real JPEG/FFmpeg tests reproduce growing source-image age before the
fix and bounded age afterward. This is **capture validation, not deployment or
sorting qualification**. No renderer correction is included.

The original evidence bundle is `act-gpu-20261023/dynamic-trace-20261026`:
`INVESTIGATION-UPDATE.md`, `evidence-review.md`, `timing-path-findings.md`,
`remote-investigation-summary.json`, and `renderer-control/candidate-comparison.json`.
Its diagnostic interventions are not production baselines and its different
experiment families must not be pooled into a controlled factorial comparison.

## Root cause and owning source

- `src/physicalai/capture/cameras/ip/_camera.py` previously performed one OpenCV
  read per consumer call. `CAP_PROP_BUFFERSIZE=1` is best-effort and does not
  guarantee that FFmpeg/network buffers contain only the newest image.
- `src/physicalai/capture/transport/_publisher_worker.py` added a fixed
  `0.85 / fps` delay **after** acquisition and publication. At 30 FPS this added
  28.33 ms without subtracting acquisition/processing cost. A paced MJPEG source
  therefore accumulated upstream backlog when the combined work exceeded its
  frame period. Consumer stalls also stopped acquisition.
- `src/physicalai/capture/transport/_shared_camera.py` discarded old samples only
  **after** acquisition. Its latest shared-memory frame could therefore contain
  an old source image with a recent local decode timestamp.
- `src/physicalai/capture/frame.py` and `capture/camera.py` described timestamps and
  sequential reads too generally. Local decode time cannot establish source age.

The fix continuously drains IP streams into one overwriteable slot, without an
unbounded frame queue. `read()` waits for a newer locally decoded frame;
`read_latest()` performs no network I/O and can return an unchanged cached frame.
Both can skip intermediate frames. Cached reads preserve timestamps/sequences.
This intentionally clarifies the IP live-stream contract, not hardware-camera
recording semantics. Shared-camera reads inspect queued data before waiting,
ignore stale notifications, bound queue draining to the subscriber capacity, and
retain the backing sample when returning a cached zero-copy frame.

Publisher pacing subtracts elapsed acquisition/publication work from the frame
period. Unpaced sources still receive a publication rate limit; hardware camera
implementations, controls and configuration are unchanged. There is no camera-type
special case that deletes all rate limiting.

FFmpeg read timeouts are open-only. A real OpenCV 5.0.0 MJPEG check accepted
`CAP_PROP_READ_TIMEOUT_MSEC` in constructor parameters and rejected later
`cap.set()` attempts. `BUFFERSIZE` is not a supported FFmpeg constructor parameter.
Consumer waits now have independent deadlines. Native reads have a one-second
maximum timeout; disconnect signals consumers, joins acquisition, and releases the
handle in its owning reader. Startup, read/conversion and release failures have
focused tests; native exception text is not forwarded when it can contain an
authenticated URL. Explicit disconnect/reconnect resets the slot and sequence;
there is no new automatic reconnection policy.

### Compatibility limits

- IP timestamps remain local decode/conversion-completion times, not exposure,
  render-start or simulator-pose timestamps. Sequences count local decoded frames,
  not source exposures. Repeated source images cannot be detected from them.
- Source/network buffering and streams faster than the host can decode remain
  outside the latest-slot guarantee. No timestamp is relabeled as source time.
- IP and shared-camera reads are latest-oriented, not lossless recording APIs.
- Connect failure cleanup can additionally wait for native I/O to end. A backend
  that violates its native timeout produces a shutdown error; reconnect must not
  replace a still-running reader. RTSP/hardware-device testing remains necessary.
- RGB/BGR/gray conversion, independently requested width/height, RTSP TCP option
  restoration, camera controls and the transport wire format are preserved.
- No Studio code, installed packages, model weights, policy FPS, action queues or
  conveyor speed changed. No existing simulator was taken over.

## Freshness validation

`tests/unit/capture/test_freshness.py` serves real 640x480 MJPEG on an ephemeral
loopback port. Source frame IDs and source-relative times are encoded in image
pixels before JPEG encoding and recovered after actual FFmpeg decoding. The
measurement does not trust receiver-generated `Frame.timestamp`.

One local paired measurement used the pre-change source and working-tree source,
30 FPS input, 10 ms conversion cost, real transport encoding plus 14 ms publication
cost, 60 sampled frames and two 200 ms consumer stalls. Each version used its own
publisher pacing. These are synthetic load costs, not measured ACT processing
costs. The age endpoint is **after simulated publication work**, not inference
invocation or actuation.

| Metric                                              |               Before |            After |
| --------------------------------------------------- | -------------------: | ---------------: |
| Pixel-derived image-age median                      |          1,069.04 ms |         41.33 ms |
| Maximum image age                                   |          2,107.69 ms |         64.24 ms |
| Early / late median age                             | 302.90 / 1,843.35 ms | 31.50 / 50.82 ms |
| Maximum source-frame lag                            |                   63 |                1 |
| Frame-ID jumps after the two stalls                 |                1 / 1 |            7 / 7 |
| Actual source rate                                  |            30.05 FPS |        30.27 FPS |
| Delivered rate including stalls                     |             14.54 Hz |         23.29 Hz |
| Delivered rate excluding the explicit 400 ms stalls |             16.10 Hz |         27.56 Hz |

These camera delivery rates are **not action rates**. The measurement ran no
policy or controller. The test uses generous image-age limits plus source-ID lag
and recovery checks, rather than a tight wall-clock throughput assertion.
Separate deterministic checks cover elapsed-work pacing for paced, unpaced,
overbudget and unspecified-rate sources; slot deadlines; stale notifications;
continually refilled queues; and copy/zero-copy cache ownership. Native subprocess
SharedCamera tests use a unique service name and real MJPEG, stall the consumer,
and stop only the publisher they created.

### Supplied ACT evidence, not new production-patch results

The supplied isolated remote diagnostic changed matched **source-pose publication
age at inference invocation**, for uniquely matched dynamic images after six
seconds, from 737.94 to 48.57 ms overview and 1,334.88 to 47.09 ms wrist. It removed
only the publisher sleep in diagnostic subprocesses; it did not test this source
patch. Remote action rate was 29.24 to 29.30 Hz, cameras about 30 FPS, sorting 1/10
to 2/10. The corresponding isolated Mac diagnostic changed action rate 25.34 to
25.12 Hz and sorting 0/10 to 7/10. Instrumentation preserved values but was not
timing-neutral.

The remote combined 9/10 and 10/10 results additionally used a thresholded
reference-image adjustment. Those 19/20 outcomes are diagnostic only. The
appearance-only family used different instrumentation/environment settings and
must remain separate. No new ACT action-rate, numerical-parity or sorting results
are claimed for this production source patch.

## Renderer assessment: unresolved; no setting change

The owning simulator source is
`packages/physicalai-mujoco-plugin/src/physicalai_mujoco_plugin/`:

- `cameras.py` creates ordinary `mujoco.Renderer` contexts and publishes contiguous
  RGB. Its availability probe checks that rendering works, not image equivalence.
- `camera_thread.py` snapshots all qpos, mocap positions and mocap quaternions into
  private rendering data, then calls `mj_fwdPosition`. This covers the rigid-pose
  fields used in the supplied replay, not arbitrary time-dependent visualization
  or a complete physics checkpoint.
- `robot.py` also creates renderers for snapshots. A future principled renderer
  policy must account for both creation paths.
- `urdf/scenes/scene_common.xml` requests 8 samples and an 8192 shadow map. Matching
  that XML does not establish matching effective GL settings.

Saved context probes show:

| Backend    | GL identity                         | Effective MSAA | GL maximum samples |
| ---------- | ----------------------------------- | -------------: | -----------------: |
| Mac        | Apple M3 Pro, OpenGL 2.1 Metal      |              4 |                  4 |
| NVIDIA EGL | A100, OpenGL 4.6, driver 580.159.03 |              8 |                 64 |

The saved Mac/NVIDIA shadow probes match 8192-square storage, 24-bit depth,
internal format, nearest filtering and depth comparison settings. This does not
prove identical rasterization or identify a driver bug. The original base render
records do not attest those settings per frame; separate probes provide them.

A new isolated local replay loaded the exact shared binary and complete pose
archive, without starting a simulator or advancing physics:

- MuJoCo 3.16.0.
- Model SHA256: `67c93e37f07d2baac21d2f674833a31f7370bc793fc509047ad1c4b6722a6b12`.
- Pose SHA256: `f258855349d9c44b6c47df2c04fd836dd2867af34b39d9e97595a582a6ca83a0`.
- Current context: Apple M3 Pro, OpenGL `2.1 Metal - 91.7`, maximum 4 samples.
- Requested/effective MSAA: 8/4. Effective shadow size: 8192; output: 640x480.
- All ten default raw RGB images (five poses, both cameras) were byte-identical
  to the retained Mac images. No policy predictions were recomputed in this replay.

The five poses are identical across platforms but reconstructed diagnostics,
not synchronized snapshots of the historical live trials. Saved comparisons show
Mac/NVIDIA overview raw MAE about 2.97–3.02/255 and action-chunk MAE up to 13.23
(maximum action difference 79.41). Forcing NVIDIA to effective 4x MSAA reduces that
pose's action MAE only to 10.95. Reducing shadows to 1024 does not resolve it;
disabling shadows changes intended appearance and is not a qualified correction.
Software rendering is much closer offline but its observed 1.5–1.7 FPS, or roughly
2–3 FPS with quality reductions, cannot support the required camera rate.

No portable correction is demonstrated. Accordingly this change includes no
pixel-reference adjustment, overlays, camera-specific calibration, software
renderer selection, or speculative global MSAA/shadow change. A future candidate
needs a demonstrated rendering mechanism, actual per-camera GL attestation,
identical-model/full-pose image and fixed-backend action comparisons, sustained
camera throughput, and ordinary matched-seed closed-loop qualification.

## Commands and remaining validation

Run from the repository root, without synchronizing or modifying Studio's venv:

```bash
uv run --no-sync pytest tests/unit/capture -q
uv run --no-sync --with iceoryx2==0.9.0 pytest tests/unit/capture -q
uv run --no-sync --with iceoryx2==0.9.0 --with msgpack==1.2.1 \
  pytest tests/unit/capture tests/unit/runtime -q
uv run --no-sync pyrefly check \
  src/physicalai/capture/cameras/ip/_camera.py \
  src/physicalai/capture/transport/_publisher_worker.py \
  src/physicalai/capture/transport/_shared_camera.py
uvx ruff@0.15.12 check \
  src/physicalai/capture/camera.py src/physicalai/capture/frame.py \
  src/physicalai/capture/cameras/ip/_camera.py \
  src/physicalai/capture/transport/_publisher_worker.py \
  src/physicalai/capture/transport/_shared_camera.py
# The same source paths were checked with ruff format --check.
git diff --check
```

The capture-only run passed **293 tests**. The final combined capture/Runtime
run passed **569 tests**, with one RealSense factory test skipped because its
hardware SDK is unavailable and two jsonargparse deprecation warnings. An initial
Runtime run lacked `msgpack` (three telemetry test import failures); rerunning
with that declared transport dependency in the ephemeral environment passed.
Focused source and new-test type checking reported zero errors; Ruff 0.15.12
source lint/format checks passed. Unscoped type checks in the development/uvx
environments reported errors outside the changed files (122 with pyrefly 1.4.0;
118 with 1.2.0). Those runs are not a clean repository-wide type-check result;
the isolated, configured commit-hook type check and all applicable changed-file
hooks passed after fixing two test-helper annotations.

The `--with` dependency lives in an ephemeral uv environment; it does not install
into the live Studio environment. The real OpenCV constructor/timeout check, paired
pixel-age measurement and offline replay ran as local Python heredocs. Their JSON
outputs are local investigation artifacts, not public regression fixtures. The
paired measurement reuses `measure_stream` in `test_freshness.py`, loading the
pre-change IP module from Git into a temporary directory and using its original
post-work pacing; the working-tree measurement uses `_pace_capture`.

The offline replay used:
`uv run --no-sync --with mujoco==3.16.0 --with PyOpenGL python`, restored every
qpos/mocap array from the verified archive, queried actual GL identity/limits and
renderer context settings, and compared both cameras to retained raw RGB arrays.
No live process or installed plugin was patched.

Not run: the full repository suite, repository-wide `--all-files` hooks, physical cameras/RTSP,
new NVIDIA EGL rendering, or ordinary ACT closed-loop sorting at 4 cm/s. The
verified local export exists, but a source-patched ordinary simulator/Runtime
trial and matched-seed qualification remain follow-up work; the live Studio
environment was deliberately left unchanged. Passing capture tests and local
render replay does not establish recovery of the original deployment gap.
