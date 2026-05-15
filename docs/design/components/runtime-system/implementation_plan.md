# Runtime System — Implementation Plan

This document is the implementation plan for `physicalai.runtime`. It refines the original [policy_runtime_design.md](./policy_runtime_design.md) based on codebase exploration, bug analysis, and architecture review.

Read [policy_runtime_design.md](./policy_runtime_design.md) first for API shape and ownership rules. This document covers what to build, in what order, and why.

## Reference Implementation

The golden reference is `physicalai/examples/runtime/inference_async.py` — a working async prototype with QueueMixer, InferenceThread, velocity clamping, and camera discovery. Every runtime component must match or exceed its behavior.

---

## Phase 1: Critical Bug Fixes (half day) ✅

Fix bugs on the code path Phase 2 depends on. Phase 2 defines a public runtime contract — workarounds would calcify into permanent API shape. `predict_action_chunk()` currently raises `RuntimeError` without these fixes because Bug 2's inverted guard blocks the runtime's call to `model.predict_action_chunk(obs)`. This is not stylistic — it is a hard blocker.

### Bug 1: `use_action_queue` checks manifest, ignores runtime runner

**File**: `physicalai/src/physicalai/inference/model.py` — `use_action_queue` property

**Problem**: Reads `self.manifest.model.runner` class_path. Ignores `self.runner` passed at construction or set at runtime.

**Fix**: Check `isinstance(self.runner, ActionChunking)` instead:

```python
@property
def use_action_queue(self) -> bool:
    from physicalai.inference.runners.action_chunking import ActionChunking
    return isinstance(self.runner, ActionChunking)
```

### Bug 2: `select_action()` / `predict_action_chunk()` guards inverted

**File**: `physicalai/src/physicalai/inference/model.py`

**Problem**: `select_action()` raises when `not use_action_queue`. `predict_action_chunk()` raises when `use_action_queue`. Both are backwards — `select_action()` should be the generic one-action API, `predict_action_chunk()` should return raw chunks.

**Fix**: Remove guards entirely. Both methods should work for any runner (shape-stable contract per design doc §3):

| Runner          | `select_action()`  | `predict_action_chunk()` |
| --------------- | ------------------ | ------------------------ |
| single-pass     | runner output      | wrap as `(1, D)` chunk   |
| chunk-producing | pop one via cursor | runner output            |

### Bug 3: ACT export manifest declares SinglePass instead of ActionChunking

**File**: `library/src/physicalai/export/mixin_policy.py` — `_build_manifest()`

**Problem**: Checks `metadata.get("use_action_queue", False)` but ACT never sets this metadata flag. Manifest always gets `SinglePass` runner.

**Fix**: ACT export must pass `use_action_queue=True` and `chunk_size=<config.chunk_size>` in its metadata kwargs. Same fix needed for Pi0.5 (Bug 5).

### Deferred Bugs (document as issues, do not block Phase 2)

| Bug   | Summary                                                               | Why deferred                                            |
| ----- | --------------------------------------------------------------------- | ------------------------------------------------------- |
| Bug 4 | Manifest missing `hardware` section (`RobotSpec`, `CameraSpec`)       | Not on inference code path                              |
| Bug 5 | Pi0.5 export also declares SinglePass                                 | Same root cause as Bug 3, fix together                  |
| Bug 7 | Pi0.5 normalization not baked into graph (external pre/postprocessor) | By design — manifest `preprocessors_specs` handles this |
| Bug 8 | Pi0.5 denoising loop not exportable cleanly (11x graph size)          | Export concern, not runtime concern                     |
| Bug 9 | `OVTokenizer` may need `import openvino_tokenizers` for custom ops    | Needs verification — may already work via adapter       |

---

## Phase 2: Runtime System (2–3 days) ✅

New package: `physicalai/src/physicalai/runtime/`

```text
physicalai/src/physicalai/runtime/
├── __init__.py              # exports: PolicyRuntime, SyncExecution, AsyncExecution,
│                            #          ActionQueue, LerpSmoother, ReplaceSmoother, RunStats
├── smoothers.py             # ChunkSmoother ABC, ReplaceSmoother, LerpSmoother
├── _action_queue.py         # ActionQueue (public via __init__, internal module)
├── execution.py             # Execution ABC, SyncExecution, AsyncExecution, WorkerDiedError
└── runtime.py               # PolicyRuntime, RunStats, default_observation_to_input
```

Dependency order: `smoothers.py` → `_action_queue.py` → `execution.py` → `runtime.py` → `__init__.py`

### Architectural Decisions

**ActionQueue is owned by PolicyRuntime, not hidden inside Execution.**

The original design doc keeps Execution (scheduling) and ActionQueue (buffering) as separate concerns. This is correct — when `AsyncExecution(transport="process")` or `RemoteExecution` arrive, they should push chunks into the same ActionQueue without duplicating buffer logic.

Users get a clean default API:

```python
runtime = PolicyRuntime(
    robot=robot,
    model=model,
    execution=AsyncExecution(threshold=0.5),
    fps=30,
)
runtime.run(duration_s=60)
```

Power users can override buffering:

```python
runtime = PolicyRuntime(
    robot=robot,
    model=model,
    execution=AsyncExecution(threshold=0.5),
    action_queue=ActionQueue(smoother=LerpSmoother(duration_frames=10)),
    fps=30,
)
```

**Execution is a scheduler, not a buffer.** It decides when/where inference runs and pushes results into ActionQueue. It does not own pop, remaining, or chunk_size.

**InferenceModel must NOT import ActionQueue.** Per design doc §4: if both layers need pop-from-chunk mechanics, they share `ActionChunkCursor`, not `ActionQueue`.

### 2.1 `smoothers.py`

Extracted from `QueueMixer.add()` in inference_async.py.

```python
class ChunkSmoother(ABC):
    """Merges a new action chunk into remaining actions from the previous chunk."""

    @abstractmethod
    def merge(
        self,
        remaining: np.ndarray,    # (R, action_dim) — unconsumed actions from previous chunk
        incoming: np.ndarray,     # (H, action_dim) — new chunk from inference
        offset: int,              # skip first N actions of incoming (latency compensation)
    ) -> np.ndarray:
        """Return merged actions array. Called by ActionQueue.push_chunk()."""
        ...


class ReplaceSmoother(ChunkSmoother):
    """Drop remaining actions, use incoming[offset:]."""

    def merge(self, remaining, incoming, offset):
        return incoming[offset:]


class LerpSmoother(ChunkSmoother):
    """Lerp-blend overlapping region, then append non-overlapping tail.

    Stateless merge — no hidden mutation. duration_frames is the fallback
    blending window used when offset is 0. When offset > 0, the blending
    window is computed from offset directly: lerp_dur = max(offset, 1).

    Matches QueueMixer.add() from inference_async.py:
    - Weights: w_i = max(1.0 - i / lerp_dur, 0.0) for old actions
    - Overlap region: blended = w * remaining + (1 - w) * incoming
    """

    def __init__(self, duration_frames: int = 5) -> None:
        self.duration_frames = duration_frames

    def merge(self, remaining, incoming, offset):
        lerp_dur = max(offset, 1) if offset > 0 else self.duration_frames

        incoming = incoming[offset:]
        n_remain = len(remaining)
        lerp_dur = min(n_remain, lerp_dur)

        weights = np.maximum(1.0 - np.arange(n_remain) / max(lerp_dur, 1), 0.0)
        weights = weights[:, np.newaxis]

        n_blend = min(n_remain, len(incoming))
        blended = weights[:n_blend] * remaining[:n_blend] + (1.0 - weights[:n_blend]) * incoming[:n_blend]

        return np.concatenate([blended, incoming[n_blend:]], axis=0).astype(np.float32)
```

Key: `offset` is not just "skip N actions." It is latency compensation — `offset = int(inference_latency * fps)`. The smoother must handle this, not the caller.

### 2.2 `_action_queue.py` (public API, internal module)

Thread-safe action buffer with smoother integration and hold telemetry.

```python
class ActionQueue:
    """Thread-safe action buffer with chunk smoothing and starvation telemetry.

    Public API — exported from physicalai.runtime. Power users can override
    the default ActionQueue on PolicyRuntime to customize smoothing behavior.
    Execution pushes chunks into it; PolicyRuntime pops actions from it.
    """

    def __init__(self, smoother: ChunkSmoother | None = None) -> None:
        self._smoother = smoother or ReplaceSmoother()
        self._deque: deque[np.ndarray] = deque()
        self._lock = threading.Lock()
        self._consecutive_holds: int = 0
        self._total_holds: int = 0
        self._total_pops: int = 0

    def push_chunk(self, chunk: np.ndarray, offset: int = 0) -> None:
        """Merge a new chunk into the queue. Thread-safe."""
        with self._lock:
            remaining = (np.stack(list(self._deque))
                         if self._deque
                         else np.empty((0, chunk.shape[1]), dtype=chunk.dtype))
            merged = self._smoother.merge(remaining, chunk, offset)
            self._deque.clear()
            self._deque.extend(merged)

    def pop(self) -> np.ndarray | None:
        """Pop next action, or None if empty. Thread-safe."""
        with self._lock:
            if not self._deque:
                self._consecutive_holds += 1
                self._total_holds += 1
                return None
            self._consecutive_holds = 0
            self._total_pops += 1
            return self._deque.popleft()

    @property
    def remaining(self) -> int:
        with self._lock:
            return len(self._deque)

    def below_threshold(self, threshold: int) -> bool:
        return self.remaining < threshold

    def clear(self) -> None:
        with self._lock:
            self._queue = None
            self._index = 0

    # --- Telemetry ---
    @property
    def consecutive_holds(self) -> int:
        return self._consecutive_holds

    @property
    def total_holds(self) -> int:
        return self._total_holds

    @property
    def total_pops(self) -> int:
        return self._total_pops
```

### 2.3 `execution.py`

**Execution ABC** — scheduler only. Pushes chunks into ActionQueue, does not own pop/remaining.

```python
class Execution(ABC):
    """Decides when and where inference runs. Pushes results into ActionQueue."""

    @abstractmethod
    def start(self, model: InferenceModel, action_queue: ActionQueue) -> None:
        """Bind to model and queue. Called once before the loop."""
        ...

    @abstractmethod
    def maybe_request(self, observation: dict[str, np.ndarray]) -> None:
        """Check if new inference is needed. If so, run or schedule it."""
        ...

    @abstractmethod
    def warmup(self, sample_observation: dict[str, np.ndarray]) -> None:
        """Run one inference to discover chunk_size and seed the queue.

        After warmup():
        - action_queue has one chunk ready (robot starts moving immediately)
        - self.chunk_size is set
        - self.action_dim is set
        """
        ...

    @abstractmethod
    def stop(self) -> None:
        """Stop scheduling. For async: signal thread, join with timeout."""
        ...

    @property
    @abstractmethod
    def chunk_size(self) -> int:
        """Discovered after warmup(). Used to compute threshold."""
        ...
```

**SyncExecution** — blocks on inference when queue runs low.

```python
class SyncExecution(Execution):
    """Synchronous inference in the control thread."""

    def __init__(self) -> None:
        self._model: InferenceModel | None = None
        self._queue: ActionQueue | None = None
        self._chunk_size: int = 0

    def start(self, model, action_queue):
        self._model = model
        self._queue = action_queue

    def warmup(self, sample_observation):
        actions = self._model.predict_action_chunk(sample_observation)
        self._chunk_size = actions.shape[0]
        self._queue.push_chunk(actions, offset=0)

    def maybe_request(self, observation):
        if self._queue.below_threshold(1):  # refill when empty
            actions = self._model.predict_action_chunk(observation)
            self._queue.push_chunk(actions, offset=0)

    def stop(self):
        pass

    @property
    def chunk_size(self):
        return self._chunk_size
```

**AsyncExecution** — background thread with health monitoring. Maps to `InferenceThread` from inference_async.py.

```python
class WorkerDiedError(RuntimeError):
    """Raised when the inference worker thread dies unexpectedly."""
    pass


class AsyncExecution(Execution):
    """Async inference in a background thread.

    Thread architecture (matches inference_async.py):

        Control thread (main):              Inference thread (background):
        ─────────────────────               ────────────────────────────
        loop at fps:                        loop:
          obs = robot.get_observation()       wait for obs_slot
          execution.maybe_request(obs)        chunk = model.predict_action_chunk(obs)
          action = queue.pop()                offset = int(latency * fps)
          robot.send_action(action)           queue.push_chunk(chunk, offset)
    """

    def __init__(
        self,
        threshold: float = 0.5,
        fps: int = 30,
        watchdog_timeout_s: float = 30.0,
        max_consecutive_holds: int | None = None,   # default: 3 * fps (3 seconds)
    ) -> None:
        self._threshold_frac = threshold
        self._fps = fps
        self._watchdog_timeout_s = watchdog_timeout_s
        self._max_consecutive_holds = max_consecutive_holds or 3 * fps

        # Set during start()
        self._model: InferenceModel | None = None
        self._queue: ActionQueue | None = None
        self._chunk_size: int = 0
        self._threshold_count: int = 0

        # Thread state
        self._lock = threading.Lock()
        self._obs_slot: dict | None = None
        self._obs_ready = threading.Event()
        self._running_inference = False
        self._request_time: float = 0.0
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._death_cause: BaseException | None = None
        self._inference_count: int = 0

    def start(self, model, action_queue):
        self._model = model
        self._queue = action_queue
        self._thread = threading.Thread(target=self._run, name="InferenceThread", daemon=True)
        self._thread.start()

    def warmup(self, sample_observation):
        actions = self._model.predict_action_chunk(sample_observation)
        self._chunk_size = actions.shape[0]
        self._threshold_count = int(self._chunk_size * self._threshold_frac)
        self._queue.push_chunk(actions, offset=0)

    def maybe_request(self, observation):
        # Check for worker death — raise, don't silently continue
        if self._thread is not None and not self._thread.is_alive():
            if self._death_cause is not None:
                raise WorkerDiedError(
                    f"Inference thread died: {self._death_cause}"
                ) from self._death_cause

        # Check for stuck inference
        if self._busy_duration > self._watchdog_timeout_s:
            logger.warning(
                "Inference stuck for %.0fs — force resetting", self._busy_duration,
            )
            self._force_reset()

        # Submit if queue is low and worker is idle
        if self._queue.below_threshold(self._threshold_count):
            if not self._busy:
                # Defensive copy — observation may be reused by caller
                snapshot = {
                    k: v.copy() if isinstance(v, np.ndarray) else v
                    for k, v in observation.items()
                }
                with self._lock:
                    self._obs_slot = snapshot
                    self._request_time = time.perf_counter()
                self._obs_ready.set()

    def stop(self):
        if self._thread is not None:
            self._stop_event.set()
            self._obs_ready.set()   # unblock wait
            self._thread.join(timeout=10.0)

    @property
    def chunk_size(self):
        return self._chunk_size

    # --- Health properties ---

    @property
    def alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def _busy(self) -> bool:
        with self._lock:
            return self._obs_slot is not None or self._running_inference

    @property
    def _busy_duration(self) -> float:
        with self._lock:
            if not (self._obs_slot is not None or self._running_inference):
                return 0.0
            return time.perf_counter() - self._request_time

    @property
    def inference_count(self) -> int:
        return self._inference_count

    # --- Internal ---

    def _force_reset(self) -> None:
        with self._lock:
            self._obs_slot = None
            self._running_inference = False
        logger.warning("Force reset — cleared stuck inference state")

    def _run(self) -> None:
        """Inference thread main loop."""
        try:
            while not self._stop_event.is_set():
                self._obs_ready.wait()
                self._obs_ready.clear()

                if self._stop_event.is_set():
                    return

                with self._lock:
                    obs = self._obs_slot
                    self._obs_slot = None
                    if obs is None:
                        continue
                    self._running_inference = True

                t0 = time.perf_counter()
                actions = self._model.predict_action_chunk(obs)
                latency = time.perf_counter() - t0

                offset = int(latency * self._fps)

                self._queue.push_chunk(actions, offset=offset)
                self._inference_count += 1

                with self._lock:
                    self._running_inference = False

        except Exception as e:
            self._death_cause = e
            logger.error("Inference thread died: %s", e, exc_info=True)
```

### 2.4 `runtime.py`

```python
from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

import numpy as np

from physicalai.capture.camera import Camera, Frame
from physicalai.inference import InferenceModel
from physicalai.robot.interface import Robot, RobotObservation
from physicalai.runtime._action_queue import ActionQueue
from physicalai.runtime.execution import Execution, WorkerDiedError
from physicalai.runtime.smoothers import LerpSmoother

logger = logging.getLogger(__name__)

def default_observation_to_input(
    robot_obs: RobotObservation,
    camera_frames: dict[str, Frame],
) -> dict[str, Any]:
    """Default observation-to-model-input conversion.

    Maps:
    - Joint positions → "state" array
    - Camera frames → "images.{name}" arrays

    For Pi0.5 or other models needing custom keys (e.g. "task"),
    pass a custom obs_to_input callable to PolicyRuntime.
    """
    model_input: dict[str, Any] = {}

    # Collect joint positions into "state" vector
    if robot_obs.joint_positions:
        model_input["state"] = np.array([robot_obs.joint_positions], dtype=np.float32)

    # Map camera frames to "images.{name}"
    for name, frame in camera_frames.items():
        model_input[f"images.{name}"] = frame.data

    return model_input


class RuntimeCallback(Protocol):
    """Optional hook points in the PolicyRuntime control loop."""

    def before_send_action(self, *, action: np.ndarray, step: int) -> np.ndarray | None:
        """Called before sending action. Return modified action or None to use original."""
        ...

    def on_action_sent(self, *, action: np.ndarray, step: int) -> None:
        """Called after action is sent to robot."""
        ...

    def on_hold(self, *, step: int, holds: int) -> None:
        """Called when action queue is empty and robot holds last position."""
        ...


class PolicyRuntime:
    """Runs a policy on robot hardware.

    Loop shape (matches inference_async.py):
        obs = robot.get_observation()
        model_input = obs_to_input(obs, cameras)
        execution.maybe_request(model_input)
        action = action_queue.pop()
        if action is None: hold position
        robot.send_action(action)
        sleep_until_next_tick()
    """

    def __init__(
        self,
        robot: Robot,
        model: InferenceModel,
        execution: Execution,
        fps: float,
        cameras: Mapping[str, Camera] | None = None,
        action_queue: ActionQueue | None = None,
        obs_to_input: Callable[[RobotObservation, dict[str, Frame]], dict[str, Any]] | None = None,
        callbacks: Sequence[RuntimeCallback] = (),
    ) -> None:
        self._robot = robot
        self._model = model
        self._execution = execution
        self._fps = fps
        self._cameras = cameras or {}
        self._action_queue = action_queue or ActionQueue(smoother=LerpSmoother(duration_frames=5))
        self._obs_to_input = obs_to_input or default_observation_to_input
        self._callbacks = list(callbacks)

    def run(self, *, duration_s: float | None = None) -> RunStats:
        """Run the control loop.

        1. Warm up — run one inference, seed queue, discover chunk_size
        2. Loop — observe, maybe_request, pop, send, sleep
        3. Shutdown — stop execution, drain
        """
        # --- Init ---
        self._execution.start(self._model, self._action_queue)

        sample_obs = self._build_model_input()
        self._execution.warmup(sample_obs)

        goal_time = 1.0 / self._fps
        step = 0
        last_action: np.ndarray | None = None

        try:
            while True:
                if duration_s is not None and step * goal_time >= duration_s:
                    break

                loop_start = time.perf_counter()

                # 1. Observe
                obs = self._build_model_input()

                # 2. Maybe request inference
                self._execution.maybe_request(obs)

                # 3. Pop action
                action = self._action_queue.pop()
                if action is not None:
                    last_action = action
                else:
                    action = last_action
                    holds = self._action_queue.consecutive_holds
                    if holds == 1:
                        logger.warning("Queue empty — holding position")
                    elif holds % self._fps == 0:
                        logger.warning(
                            "Queue starvation: %d consecutive holds (%.1fs)",
                            holds, holds / self._fps,
                        )
                    self._invoke_callback("on_hold", step=step, holds=holds)

                if action is None:
                    # No warmup result and no previous action — skip
                    logger.error("No action available (warmup may have failed)")
                    continue

                # 4. Callbacks
                action = self._invoke_callback("before_send_action", action=action, step=step) or action

                # 5. Send
                self._robot.send_action(action)
                self._invoke_callback("on_action_sent", action=action, step=step)

                # 6. Timing
                elapsed = time.perf_counter() - loop_start
                sleep_time = goal_time - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

                step += 1

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except WorkerDiedError as e:
            logger.error("Worker died during runtime: %s", e)
            raise
        finally:
            self._shutdown(step)

        return RunStats(
            steps=step,
            total_pops=self._action_queue.total_pops,
            total_holds=self._action_queue.total_holds,
            inference_count=getattr(self._execution, "inference_count", 0),
        )

    def _build_model_input(self) -> dict[str, Any]:
        robot_obs = self._robot.get_observation()
        camera_frames = {name: cam.read_latest() for name, cam in self._cameras.items()}
        return self._obs_to_input(robot_obs, camera_frames)

    def _shutdown(self, step: int) -> None:
        """Robot and cameras must be connected before run(). Caller owns lifecycle."""
        # 1. Stop inference scheduling
        self._execution.stop()

        # 2. Drain remaining actions (up to 1s) for smooth stop
        remaining = self._action_queue.remaining
        drain_limit = min(remaining, int(self._fps))
        for _ in range(drain_limit):
            action = self._action_queue.pop()
            if action is not None:
                self._robot.send_action(action)
                time.sleep(1.0 / self._fps)

        logger.info(
            "Shutdown complete — %d steps, %d pops, %d holds",
            step, self._action_queue.total_pops, self._action_queue.total_holds,
        )

    def _invoke_callback(self, method: str, **kwargs):
        result = None
        for cb in self._callbacks:
            fn = getattr(cb, method, None)
            if fn is not None:
                result = fn(**kwargs)
        return result
```

### 2.5 Tests

```text
physicalai/tests/unit/runtime/
├── test_smoothers.py        # ReplaceSmoother, LerpSmoother: offset handling, lerp weights,
│                            #   dynamic duration, edge cases (empty remaining, offset > chunk)
├── test_action_queue.py     # push/pop, smoother integration, thread safety (concurrent
│                            #   push+pop from 2 threads), hold counters, clear()
├── test_execution.py        # SyncExecution: warmup seeds queue, maybe_request refills
│                            # AsyncExecution: mock model, health monitoring (alive/busy/
│                            #   busy_duration), WorkerDiedError propagation, force_reset
└── test_runtime.py          # PolicyRuntime with mock robot + mock model:
                             #   full loop, hold fallback, shutdown drain, callbacks,
                             #   WorkerDiedError propagation, duration_s limit
```

All tests use mock `InferenceModel` and mock `Robot` — no hardware, no exported models.

---

## Phase 3: CLI and Integration (1–2 days)

1. `physicalai run --config so101_pi05.yaml` CLI command
2. YAML config loader (`PolicyRuntime.from_config()`)
3. Observation builder (bridges `Robot` protocol + `Camera` → model input dict)
4. Migrate `inference_async.py` to use `PolicyRuntime` (becomes ~20 lines)

### Velocity clamping and camera discovery stay outside core

Velocity clamping (`max_speed`, `ramp_steps`, `commanded_pos` tracking) is SO-101-specific. Camera discovery (interactive selection, name mapping, blank cameras, flip) is app-specific.

These belong in:

- Example scripts (`examples/runtime/`)
- CLI helpers (`physicalai.cli.run`)
- User callbacks

Not in `PolicyRuntime` or `Execution`. If a reusable pattern emerges across 2+ robots, promote to a formal action-transform layer later.

---

## Phase 3.5: Runtime Telemetry (1–2 days)

Streaming observability for the runtime control loop. The runtime process must not be bottlenecked by telemetry — all emission is fire-and-forget over zenoh pub-sub. A separate observer process handles visualization, aggregation, and persistence.

### Constraint

Two-week release deadline. Scope: emitter in runtime, observer CLI, zenoh-only transport. No OpenTelemetry, no Studio UI integration, no remote telemetry. Use OTel-compatible metric naming so a future OTLP bridge is trivial.

### Architecture

```text
┌──────────────────────────────────────────┐
│  Runtime Process (real-time budget)      │
│                                          │
│  PolicyRuntime.run()                     │
│    └─ TelemetryEmitter                   │
│         zenoh.put() per tick/inference   │
│         no-op if zenoh not installed     │
└──────────┬───────────────────────────────┘
           │ zenoh pub-sub (local SHM)
           ▼
┌──────────────────────────────────────────┐
│  Observer Process (separate, optional)   │
│                                          │
│  TelemetrySubscriber                     │
│    ├─ Live console dashboard             │
│    ├─ JSONL file sink (--record)         │
│    └─ Future: WebSocket → Studio UI      │
└──────────────────────────────────────────┘
```

### Topic Schema

All topics prefixed with `physicalai/rt/{session_id}/`. Session ID is a short hex string generated at `run()` start.

| Topic       | Frequency                            | Payload (msgpack)                                                                                                            | Purpose                                   |
| ----------- | ------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------- |
| `tick`      | Every control loop tick (30 Hz)      | `step`, `timestamp`, `joint_positions: (D,)`, `action_sent: (D,)`, `queue_remaining: int`, `loop_duration_s`, `sleep_time_s` | Core loop health + executed trajectory    |
| `inference` | Per inference completion (~2–3 Hz)   | `latency_s`, `offset`, `chunk: (H, D)`                                                                                       | Latency monitoring + predicted trajectory |
| `lifecycle` | On start / warmup / shutdown / error | `event: str`, `metadata: dict`                                                                                               | Session boundaries                        |

**Serialization**: msgpack with numpy arrays encoded as `{"__np__": true, "dtype": str, "shape": list, "data": bytes}`. Encode cost is ~50 μs per tick event, ~100 μs per inference event. Negligible in a 33 ms tick budget.

**Naming convention**: Scalar fields use OTel-compatible names (`physicalai.runtime.loop_duration_s`, `physicalai.runtime.inference_latency_s`) so a future OTLP exporter in the observer can re-export without renaming.

### New Files

```text
physicalai/src/physicalai/runtime/
├── _telemetry.py                    # TelemetryEmitter (zenoh publisher, no-op fallback)
└── observer/
    ├── __init__.py
    ├── __main__.py                  # python -m physicalai.runtime.observer
    ├── _subscriber.py               # TelemetrySubscriber (zenoh sub + dispatch)
    ├── _console.py                  # Live console dashboard
    └── _recorder.py                 # JSONL file sink for offline replay
```

### Dependency

```toml
# pyproject.toml
[project.optional-dependencies]
telemetry = ["zenoh>=1.0", "msgpack>=1.0"]
```

zenoh is optional. `TelemetryEmitter` gracefully degrades to no-op when zenoh is not installed. Observer process requires `physicalai[telemetry]`.

### ActionQueue Change

Add `peek_remaining()` — returns a copy of queued actions as `(R, D)` array without consuming them. Called once per inference event (not per tick) to snapshot the post-smooth future trajectory.

```python
def peek_remaining(self) -> np.ndarray | None:
    """Return copy of remaining actions without consuming them. Thread-safe."""
    with self._lock:
        if not self._deque:
            return None
        return np.stack(list(self._deque))
```

### Integration Points

**PolicyRuntime** (4 touch points in `runtime.py`):

1. Accept optional `telemetry: TelemetryEmitter | None` in `__init__`
2. `emit_lifecycle("start")` at top of `run()`
3. `emit_tick(...)` at end of each tick
4. `emit_lifecycle("shutdown")` in `_shutdown()`

**AsyncExecution** (1 touch point in `execution.py`):

1. Accept optional `telemetry: TelemetryEmitter | None` in `__init__`
2. `emit_inference(...)` after `push_chunk()` in the `_run()` thread

No changes to `smoothers.py` or `__init__.py` exports. `TelemetryEmitter` is internal wiring, not user-facing API.

### Out of Scope

- Remote telemetry (zenoh is network-transparent; enabling it is config, not code)
- Camera frame streaming (stays on iceoryx2)
- OpenTelemetry / OTLP exporter (future observer plugin)
- Studio UI WebSocket bridge
- Model input snapshot recording (expensive, needs opt-in design)
- Smoothing delta visualization (derivable in observer from `inference.chunk` vs `tick.action_sent`)

---

## Phase 3.6: Fault Tolerance (1 day)

Robot connections over USB/serial and camera feeds are fragile. USB hubs lose power, serial timeouts occur, cameras drop frames. The runtime must not crash on transient hardware errors — it must retry and recover, or degrade gracefully.

### Current Problem

`PolicyRuntime._build_model_input()` calls `robot.get_observation()` and `cam.read_latest()` with no error handling. `robot.send_action()` is also unprotected. Any `ConnectionError`, `OSError`, or `serial.SerialException` from the SO-101 serial bus kills the control loop.

### Design Principles

1. **Never crash the loop on a transient error.** Retry the operation. If retries are exhausted, hold position and log — do not raise.
2. **Distinguish transient vs fatal.** USB disconnect that resolves after replug = transient. `ValueError` from wrong action shape = fatal (programmer error). `KeyboardInterrupt` = always propagate.
3. **No silent degradation.** Every recovery must log a warning and emit a telemetry lifecycle event. The operator must know the robot hiccupped.
4. **Bound retry duration.** Retries must not exceed the tick budget. If `get_observation()` fails 3 times within the tick, use the last known observation and move on.
5. **Camera failure ≠ robot failure.** A dropped camera frame should use the last known frame (stale but safe). A robot communication failure is more serious but still retryable.

### Implementation: `_resilient_observe()` and `_resilient_send()`

Replace the raw calls in the control loop with retry-wrapped variants:

```python
_MAX_OBS_RETRIES = 3
_MAX_SEND_RETRIES = 2
_RETRY_BACKOFF_S = 0.001  # 1ms between retries — bounded, won't bust 33ms budget

def _resilient_observe(self) -> dict[str, Any]:
    """Build model input with retry on transient hardware errors."""
    # Robot observation
    robot_obs = None
    for attempt in range(_MAX_OBS_RETRIES):
        try:
            robot_obs = self._robot.get_observation()
            self._consecutive_error_ticks = 0
            break
        except (ConnectionError, OSError) as e:
            logger.warning("Robot observation failed (attempt %d/%d): %s",
                           attempt + 1, _MAX_OBS_RETRIES, e)
            if self._telemetry:
                self._telemetry.emit_lifecycle("obs_error", error=str(e),
                                               attempt=attempt + 1)
            if attempt < _MAX_OBS_RETRIES - 1:
                time.sleep(_RETRY_BACKOFF_S)

    if robot_obs is None:
        self._consecutive_error_ticks += 1
        if self._consecutive_error_ticks >= self._max_consecutive_error_ticks:
            msg = (f"Robot unreachable for {self._consecutive_error_ticks} consecutive ticks "
                   f"({self._consecutive_error_ticks / self._fps:.1f}s)")
            if self._telemetry:
                self._telemetry.emit_lifecycle("connection_lost", error=msg)
            raise ConnectionError(msg)
        if self._last_robot_obs is not None:
            logger.warning("Using stale robot observation (error tick %d/%d)",
                           self._consecutive_error_ticks,
                           self._max_consecutive_error_ticks)
            robot_obs = self._last_robot_obs
            self._stale_obs_ticks += 1
        else:
            raise ConnectionError("Robot observation failed and no fallback available")
    self._last_robot_obs = robot_obs

    # Camera frames — independent per camera, each can fail independently
    camera_frames: dict[str, Frame] = {}
    for name, cam in self._cameras.items():
        try:
            camera_frames[name] = cam.read_latest()
            self._last_camera_frames[name] = camera_frames[name]
        except (ConnectionError, OSError) as e:
            logger.warning("Camera '%s' read failed: %s — using last frame", name, e)
            if name in self._last_camera_frames:
                camera_frames[name] = self._last_camera_frames[name]
                self._stale_obs_ticks += 1

    return self._obs_to_input(robot_obs, camera_frames)


def _resilient_send(self, action: np.ndarray) -> None:
    """Send action with retry on transient errors."""
    for attempt in range(_MAX_SEND_RETRIES):
        try:
            self._robot.send_action(action)
            self._consecutive_error_ticks = 0
            return
        except (ConnectionError, OSError) as e:
            logger.warning("send_action failed (attempt %d/%d): %s",
                           attempt + 1, _MAX_SEND_RETRIES, e)
            if self._telemetry:
                self._telemetry.emit_lifecycle("send_error", error=str(e),
                                               attempt=attempt + 1)
            if attempt < _MAX_SEND_RETRIES - 1:
                time.sleep(_RETRY_BACKOFF_S)
    self._consecutive_error_ticks += 1
    self._transient_errors += 1
    logger.error("send_action failed after %d attempts — skipping tick",
                 _MAX_SEND_RETRIES)
```

### Error Classification

| Exception type               | Source                                                    | Treatment                         |
| ---------------------------- | --------------------------------------------------------- | --------------------------------- |
| `ConnectionError`            | SO-101 serial port closed, servo not responding           | Retry, then stale fallback        |
| `OSError`                    | USB disconnect, file descriptor error, camera device lost | Retry, then stale fallback        |
| `TimeoutError`               | Serial read timeout (subclass of `OSError`)               | Retry, then stale fallback        |
| `ValueError`, `RuntimeError` | Wrong action shape, uncalibrated mode, programming error  | **Fatal** — propagate immediately |
| `WorkerDiedError`            | Inference thread crash                                    | **Fatal** — propagate immediately |
| `KeyboardInterrupt`          | User Ctrl+C                                               | **Always propagate**              |

### State Tracking

`PolicyRuntime` gains fields for stale-observation fallback and error escalation:

```python
self._last_robot_obs: RobotObservation | None = None
self._last_camera_frames: dict[str, Frame] = {}
self._consecutive_error_ticks: int = 0
self._max_consecutive_error_ticks: int = int(3 * fps)  # ~3 seconds
self._stale_obs_ticks: int = 0
self._transient_errors: int = 0
```

`_consecutive_error_ticks` is reset to 0 on any successful `get_observation()` or `send_action()`. It increments when all retries within a tick fail. When it reaches `max_consecutive_error_ticks`, the runtime raises `ConnectionError`.

`RunStats` gains fault tolerance metrics:

```python
@dataclass(frozen=True)
class RunStats:
    steps: int
    total_pops: int
    total_holds: int
    inference_count: int
    transient_errors: int    # total retried hardware errors
    stale_obs_ticks: int     # ticks where stale observation was used
```

The telemetry `tick` event gains a `stale_obs: bool` field so the observer can flag degraded ticks.

### Warmup Resilience

Warmup is the most fragile moment — USB may not be fully enumerated, servos may still be initializing. The first call to `_build_model_input()` has no stale fallback.

Warmup gets its own retry loop with longer timeout:

```python
_WARMUP_RETRIES = 5
_WARMUP_BACKOFF_S = 1.0  # 1 second between warmup retries

def _warmup_with_retry(self) -> None:
    for attempt in range(_WARMUP_RETRIES):
        try:
            sample_obs = self._build_model_input()
            self._execution.warmup(sample_obs)
            return
        except (ConnectionError, OSError) as e:
            logger.warning("Warmup failed (attempt %d/%d): %s",
                           attempt + 1, _WARMUP_RETRIES, e)
            if attempt < _WARMUP_RETRIES - 1:
                time.sleep(_WARMUP_BACKOFF_S)
    msg = f"Warmup failed after {_WARMUP_RETRIES} attempts — robot or cameras unreachable"
    raise ConnectionError(msg)
```

Total warmup retry budget: 5 seconds. Long enough for USB enumeration, short enough to not confuse the user.

### Reconnection

Retry within a tick handles brief glitches (serial timeout, dropped USB packet). For sustained disconnects (USB cable pulled), the robot's `is_connected()` returns `False` and consecutive retries will keep failing.

Full reconnection (calling `robot.disconnect()` then `robot.connect()`) is **not** done automatically in the runtime. Reconnecting a serial bus mid-loop is dangerous — it resets servo state, torque settings, and calibration. This must be an explicit user decision.

Instead, the runtime:

1. Logs an error with reconnection instructions
2. Continues holding position (stale obs + last action)
3. Emits `lifecycle("connection_lost")` telemetry event
4. After `max_consecutive_errors` (default: `3 * fps` = 3 seconds of failures), raises `ConnectionError` with a clear message

This gives the user time to replug USB without the loop crashing, but doesn't silently run for minutes with a dead robot.

### Camera Recovery

Cameras are more resilient than robots — `SharedCamera` (iceoryx2) handles publisher death and re-spawn. Direct camera backends (`RealsenseCamera`, `UVCCamera`) may need explicit reconnection.

The runtime treats camera failure as soft: use last frame, log warning, continue. If a camera has never produced a frame (failure on first read), omit it from the model input dict. The model's behavior with missing camera keys is model-specific — not the runtime's concern.

### Tests

```text
physicalai/tests/unit/runtime/
└── test_fault_tolerance.py   # Mock robot that raises on get_observation/send_action:
                              #   - transient error → retry succeeds
                              #   - sustained error → stale fallback
                              #   - fatal error → propagates
                              #   - camera failure → stale frame used
                              #   - max_consecutive_errors → ConnectionError raised
```

---

## Phase 4: Advanced (later)

1. `AsyncExecution(transport="process")` for PyTorch CPU (GIL contention)
2. RTC guidance correction (requires split export from `library/`)
3. `RemoteExecution` + gRPC `PolicyServer` (see [policy_server_design.md](./policy_server_design.md))

---

## Component Mapping: inference_async.py → Library

| Script component                            | Library target                                             | Notes                                                   |
| ------------------------------------------- | ---------------------------------------------------------- | ------------------------------------------------------- |
| `QueueMixer`                                | `ActionQueue` + `LerpSmoother`                             | Offset-aware blending extracted into smoother           |
| `QueueMixer.lerp_duration = max(offset, 1)` | `LerpSmoother.merge()` computes from offset                | Stateless — `duration_frames` is fallback when offset=0 |
| `InferenceThread`                           | `AsyncExecution`                                           | Same thread architecture: obs_slot + result push        |
| `InferenceThread.force_reset()`             | `AsyncExecution._force_reset()`                            | Clears stuck state                                      |
| `InferenceThread.busy_duration`             | `AsyncExecution._busy_duration`                            | Watchdog timeout trigger                                |
| `InferenceThread.alive`                     | `AsyncExecution.alive`                                     | Dead thread detection                                   |
| `get_full_observation()`                    | `default_observation_to_input()` + `obs_to_input` callable | Separates robot obs format from model input format      |
| `action_to_robot_dict()`                    | `Robot.send_action(ndarray)`                               | Robot protocol handles conversion                       |
| `main()` while-loop                         | `PolicyRuntime.run()`                                      | Same 5-step structure                                   |
| Velocity clamping + ramp                    | User callback or example code                              | Too robot-specific for core runtime                     |
| Camera discovery + `SharedCamera`           | User code / CLI (Phase 3)                                  | App-specific                                            |
| Warm-up inference + queue seeding           | `Execution.warmup()`                                       | Seeds queue so loop starts with actions                 |
| `inference_thread.alive` check + restart    | `AsyncExecution.maybe_request()` raises `WorkerDiedError`  | Raise instead of silent restart                         |
| `force_reset()` for stuck thread            | `AsyncExecution._force_reset()` via watchdog               | Auto-triggered when `busy_duration > timeout`           |
| Hold position + `hold_count`                | `ActionQueue.consecutive_holds` + `PolicyRuntime` logging  | Telemetry via queue counters                            |
| Two-backend support (torch/OV)              | `InferenceModel` abstraction                               | Runtime only calls `predict_action_chunk()`             |
| (none — ad-hoc print/log)                   | `TelemetryEmitter` + zenoh pub-sub                         | Fire-and-forget, no-op when zenoh absent (Phase 3.5)    |
| (none — loop crashes on USB error)          | `_resilient_observe()` / `_resilient_send()`               | Retry + stale fallback for transient errors (Phase 3.6) |

---

## Design Gaps Addressed

Gaps identified during architecture review, with resolutions.

### Gap 1: Observation ownership in async

**Problem**: Main thread passes observation dict to `maybe_request()`. If main thread reuses camera buffers, inference thread reads corrupted data.

**Resolution**: `AsyncExecution.maybe_request()` performs defensive copy before submitting:

```python
snapshot = {
    k: v.copy() if isinstance(v, np.ndarray) else v
    for k, v in observation.items()
}
```

Cost: one dict of numpy copies per inference request (not per tick — only when threshold triggers). At 30fps with threshold=0.5 and chunk_size=50, that's roughly once every ~0.8s.

### Gap 2: Empty-queue telemetry

**Problem**: "Hold last action" silently masks queue starvation.

**Resolution**: `ActionQueue` tracks `consecutive_holds` and `total_holds`. `PolicyRuntime` logs warnings on first hold and every `fps` consecutive holds (1 per second). `on_hold` callback exposes starvation events to user code.

### Gap 3: Graceful shutdown

**Problem**: Hard stop can jerk the robot.

**Resolution**: `PolicyRuntime._shutdown()` drains up to 1 second of remaining actions at loop FPS. Beyond 1s, hard stop — the user pressed Ctrl+C for a reason. Robot and cameras stay connected here; caller owns connect/disconnect lifecycle.

### Gap 4: Error propagation

**Problem**: Inference thread exceptions silently swallowed — `pop_action()` returns None forever.

**Resolution**: `AsyncExecution` stores `_death_cause` on thread exception. `maybe_request()` checks `alive` and raises `WorkerDiedError` with original traceback preserved via `raise ... from`. PolicyRuntime catches and re-raises.

### Gap 5: Two-backend support

**Problem**: inference_async.py handles PyTorch and OpenVINO with different preprocessing.

**Resolution**: Not a runtime concern. `InferenceModel` abstracts backend differences. Runtime only calls `model.predict_action_chunk(obs)`. The torch bypass in inference_async.py exists because the script bypasses `InferenceModel` for direct Pi05 policy access — the library runtime won't need this.

---

## Resolved Questions

1. **`PolicyRuntime.run()` returns `RunStats`.** `@dataclass` with 4 core fields (steps, total_pops, total_holds, inference_count) plus 2 fault tolerance fields (transient_errors, stale_obs_ticks) added in Phase 3.6. Useful for testing and logging.

2. **Warm-up happens inside `run()`.** User doesn't forget, no "did I call warmup?" failure mode.

3. **Dead worker raises `WorkerDiedError`.** Let caller decide recovery strategy. Auto-restart can mask systematic failures.

---

## Relationship to Existing Design Docs

This plan refines [policy_runtime_design.md](./policy_runtime_design.md). Key differences:

| Topic                           | Original design doc                                              | This plan                                                                                                  |
| ------------------------------- | ---------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| ActionQueue visibility          | Public parameter on PolicyRuntime                                | Public parameter with sensible default. Internal `_action_queue.py` module, exported via `__init__`.       |
| Execution ABC                   | `start(action_queue, model)`, `maybe_request(obs, action_queue)` | `start(model, action_queue)`, `maybe_request(obs)` — queue stored internally on start, not passed per call |
| Warmup                          | `warmup(sample_observation, n=2)`                                | `warmup(sample_observation)` — one call, seeds queue                                                       |
| Health monitoring               | Not specified                                                    | First-class: alive, busy, busy_duration, WorkerDiedError, watchdog                                         |
| Smoothing                       | `LerpChunkSmoother`, `ReplaceMerger`                             | `LerpSmoother`, `ReplaceSmoother` — stateless merge, offset-aware                                          |
| Observation bridge              | Not specified                                                    | `obs_to_input` callable with `default_observation_to_input` fallback                                       |
| `predict_action_chunk()` return | `Mapping[str, Any]` with `"actions"` key                         | `np.ndarray` directly (matches actual implementation)                                                      |
| Bug fixes                       | Not applicable                                                   | Phase 1 prerequisite — bugs 1-3 on critical path                                                           |
| Telemetry                       | Not specified                                                    | Phase 3.5: zenoh pub-sub emitter (fire-and-forget), separate observer process, optional dep                |
| Fault tolerance                 | Not specified                                                    | Phase 3.6: retry + stale fallback for transient HW errors, error classification, bounded retry             |

All ownership rules, boundary constraints, and deferred-until-needed decisions from the original design doc remain in effect.
