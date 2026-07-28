# Runtime System Design

This is the design of `physicalai.runtime`: the package that runs a robot
control loop — policy rollout, teleop, or any other action source — at a
fixed rate, with resilient IO and a uniform telemetry stream.

## TL;DR

- **One `RobotRuntime` class.** There is no policy-only runtime subclass. An
  action source is a required constructor argument — `PolicySource` is simply
  the implementation you pass for a policy rollout, `TeleopSource` for teleop.
- **`ActionSource` is 3 methods:** `connect()`, `update()`, `disconnect()`, all
  returning `None` bar `update()`. No capability protocols, no `isinstance`
  anywhere in the runtime. **No shutdown drain** — `disconnect()` tears down
  and returns nothing; any queued actions are discarded and the robot holds
  its last commanded position.
- **No `Tick` wrapper.** The runtime reads robot state + camera frames once
  per tick and passes the same plain values to the action source and to
  telemetry — eager, not lazy.
- **No copy needed for camera frames** at the runtime's read step — verified
  against real hardware, not just reasoned about (see
  [Empirical Validation](#empirical-validation-zero-copy-camera-safety)). The
  only copies that exist are local to `AsyncCallback` and
  `AsyncExecution`/`RTCExecution`.
- **No "hold" concept in the runtime.** `update()` always returns a sendable
  action; what to do when there's nothing new (repeat last, safe pose,
  whatever) is the action source's own decision, made internally.
- **No stats mechanism in the runtime.** `run()` returns just `steps: int`.
  Anything else (inference counts, error totals) is read directly off objects
  you already built, or via a tiny opt-in callback.
- **One config schema.** `action_source:` is always explicit — no flat
  shorthand.
- **Callback hooks:** `on_action_ready` (can transform the action) and
  `on_action_sent` (notification only), plus `on_tick`, `on_inference`, and
  `on_lifecycle`.

See the [Decision Summary](#decision-summary) at the end for the compact
reference-card version of all of this. Read on for the reasoning behind each
call.

## First-Principles Requirements

### Why a runtime at all

- Avoid rewriting the control loop for every new use case.
- Config-driven execution — decouples _what to run_ (a config file) from
  _how the loop works_ (code), so a run can be launched without writing
  Python.
- IO resilience is safety-critical and easy to get subtly wrong per call site
  (retry-with-backoff on transient errors, stale-observation fallback,
  consecutive-error circuit breaker) — centralizing it means it's implemented
  and tested once, not reinvented per script.
- Timing correctness (drift-free fixed-rate loop) is easy to get wrong with a
  naive `sleep(1/fps)`, and matters for recorded-data quality.
- One audit trail — every run emits the same event stream regardless of who's
  driving the arm, so one console view / recorder / visualizer works for
  every use case instead of one-off print statements per script.
- Lowers the floor for non-experts — "run this policy" becomes a config file,
  not a systems-programming exercise.
- Non-goal: one-off diagnostic scripts (e.g. `examples/so101/move_joints.py`)
  don't need any of this. The line is: anything that runs repeatedly, is
  timed, is recorded, or is safety-relevant needs the runtime; a hardware
  smoke test doesn't.

### Why a generic (pluggable action source) runtime, not policy-only

- **Teleop is data collection, not a side feature.** If policy rollout is the
  "real" loop and teleop is a separate hand-rolled script, the
  timing/capture semantics used to _collect_ demonstrations and the ones
  used to _replay_ a trained policy are two different code paths — any skew
  between them (exactly when the observation is captured relative to the
  action) is a train/deploy mismatch waiting to happen.
- **HIL and DAgger need to switch or mix action sources.** That has to be a
  seam from day one, or they become a second loop implementation instead of
  a plugin.
- **A future safety/supervisory layer wants one place to intercept the
  action, regardless of who's driving** — including a human in teleop.
- **Review surface** — a new action source is one small, reviewable class,
  not a fork of the loop.
- Caveat, held throughout this design: "generic" means **exactly one right
  seam** (the action source), not maximal configurability everywhere. Adding
  speculative capability protocols that only one consumer needs is the
  mistake to avoid here.

### What the runtime owns vs. delegates

Two genuinely distinct seams, kept structurally separate:

- **Callbacks** react to what already happened (telemetry) or transform the
  chosen action. They cannot originate an action or veto sending one.
- **Action source** decides what the action _is_, this tick.

The runtime owns: fixed-rate timing, resilient IO (read robot/cameras, send
action, retry/fallback/circuit-breaker), the connect/disconnect resource
lifecycle (with rollback on partial failure), the telemetry emission point,
and the fail-stop safety net — deciding _when to give up_ is uniform across
every action source and is not each action source's discretion.

The runtime delegates entirely: what the action is, and what to do when there
is no new action to produce (see [ActionSource](#actionsource)).

## Design Principles

- One way to do things, not two — there is exactly one runtime class and one
  required action-source protocol, no shortcut class and no dual config
  schema.
- The runtime loop body is readable top-to-bottom with no `isinstance` calls.
- `ActionSource`: the minimum a developer must implement to plug in a new
  action source — 3 required methods, nothing optional.
- Internals (queue state, hold/repeat tracking, per-source stats) are the
  action source's private concern — the runtime does not introspect them.

## Target Architecture

### Overview

```text
RobotRuntime     the fixed-rate loop: read → decide → send → sleep → emit. One class.
ActionSource     decides the action, every tick. Required, not optional.
PolicySource     adapts model + execution + queue into ActionSource
TeleopSource     reads a leader device into ActionSource
Callbacks        side-effects and action transforms — a distinct seam from ActionSource
```

### Component Ownership

| Component         | Owns                                                                                                     | Does not own                                    |
| ----------------- | -------------------------------------------------------------------------------------------------------- | ----------------------------------------------- |
| `InferenceModel`  | load model, preprocess input, run backend, return actions (see `docs/design/components/inferencekit.md`) | robot timing, action queue, callbacks, shutdown |
| `Execution`       | when and where inference runs: sync, thread, RTC                                                         | queueing policy, robot IO                       |
| `ActionQueue`     | store chunks, merge chunks, smooth boundaries, pop one action per tick                                   | model inference, robot IO                       |
| `ActionSource`    | decide the action for this tick, own connect/disconnect for its own resources                            | timing, robot/camera IO, telemetry emission     |
| `RobotRuntime`    | fixed-rate timing, resilient IO, connect/disconnect lifecycle, telemetry emission, fail-stop safety net  | policy math, action-source internals            |
| `RuntimeCallback` | telemetry side-effects, action transforms (`on_action_ready`)                                            | deciding or vetoing the action                  |

### ActionSource

```python
class ActionSource(Protocol):
    def connect(self, *, bus: _CallbackBus, session_id: str) -> None: ...
    def update(self, robot_state: RobotObservation,
               camera_frames: Mapping[str, Frame], step: int) -> np.ndarray: ...
    def disconnect(self) -> None: ...
```

Three required methods. No capability protocols, no `isinstance` anywhere in
the runtime. `Protocol`, not ABC — matches the `Robot` protocol precedent
(`src/physicalai/robot/interface.py`); jsonargparse instantiates the concrete
class named in `class_path`, never the shared protocol type.

- **`connect(bus, session_id)`** — resource setup (spawn threads, connect a
  leader device). Folds into the _same_ connect/disconnect lifecycle stage the
  runtime already runs for robot + cameras (one stage, one
  rollback-on-failure path). `bus`/`session_id` are handed here because
  `connect()` is called fresh every `run()`, which is exactly when the
  runtime generates a new `session_id` — construction-time injection would
  miss that.
- **`update(robot_state, camera_frames, step) -> np.ndarray`** — the per-tick
  decision. **Always returns a sendable action — no `None` sentinel.** What to
  do when there's nothing new to decide (repeat the last action, go to a safe
  pose, whatever) is entirely the action source's own call, made internally.
  If it truly cannot produce anything (e.g. warmup never succeeded), it
  raises — handled by the existing fail-stop semantics for `update()` errors.
  No runtime-level "hold" concept exists at all.
- **No dedicated warmup step.** The runtime calls `update()` on tick 0 like
  any other tick. An action source that needs to seed internal state before
  its first real decision (e.g. `PolicySource` discovering `chunk_size`)
  special-cases its own first `update()` call internally (a private
  not-yet-seeded flag). Accepted consequence: tick 0 may take longer —
  visible as a `loop_duration_s` spike on `TickEvent`, not hidden in a
  pre-loop phase. An action source that wants dedicated warmup retry
  resilience implements that itself.
- **`disconnect() -> None`** — teardown only (stop threads, release a leader
  device). **No drain:** queued-but-unsent actions are discarded, not
  flushed. On shutdown the robot simply stops receiving new actions and
  holds its last commanded position — safer than replaying a stale queue
  after the operator hit stop. The action source never receives a robot
  reference.

### Reads: robot state + camera frames (no `Observation` type)

The runtime reads robot state and camera frames **once per tick** and passes
them as **two plain values**, not a wrapper:

- `robot_state: RobotObservation` — the robot's own reading, reused as-is.
- `camera_frames: Mapping[str, Frame]` — the standalone `cameras=` sources.

No `Observation` wrapper type. `TickEvent` carries these as two separate
fields; the codebase already has two distinct "observation" senses
(`RobotObservation`; the inference model-input dict), so a third would only
add confusion. `RobotObservation` stays the robot's product; standalone
cameras stay a separate arg because they are read outside the robot and
`RobotObservation.images` is contractually built-in cameras only. If a third
sensor kind ever appears, `update()`/`TickEvent` grow a param — a
non-breaking retrofit, and YAGNI today.

Eager, plain values — no lazy pull, no memoization contract. Ordinary Python
object-reference passing, not a special mechanism.

**Hard invariant, and the reason no copy is needed:** exactly one
`read_latest()`/`get_observation()` call per device, per tick — nothing may
re-read a device until the next tick. Enforced structurally, not by a
runtime check: `ActionSource` and `RuntimeCallback` never receive
`Camera`/`Robot` object references, only already-read `Frame`/
`RobotObservation` values — so nothing downstream even has a handle to call
`read_latest()` again.

Given that invariant, **no copy is needed at the runtime's read step.**
Synchronous consumers (the action source, any non-deferred callback) finish
using those values before the next tick's read can invalidate anything. The
two places that _do_ need a copy — because they hand a value to a different
thread for later processing — already have one: `AsyncCallback` (copies
non-owned frame buffers before enqueueing to its background worker) and
`AsyncExecution`/`RTCExecution` (copy observation arrays before publishing to
their background inference thread). See
[Empirical Validation](#empirical-validation-zero-copy-camera-safety) for how
this was confirmed against real hardware, not just reasoned about.

### RobotRuntime

```python
class RobotRuntime:
    def __init__(
        self,
        robot: Robot,
        action_source: ActionSource,      # required — no optional/default
        fps: float,
        cameras: Mapping[str, Camera] | None = None,
        callbacks: Sequence[RuntimeCallback] = (),
    ) -> None: ...

    def connect(self) -> None: ...
    def disconnect(self) -> None: ...
    def __enter__ / __exit__: ...

    def run(self, *, duration_s: float | None = None) -> int: ...   # returns steps

    @classmethod
    def from_config(cls, config: str | Path) -> Self: ...

    @property
    def action_source(self) -> ActionSource: ...   # public — see Observability
```

The loop body:

```python
def run(self, *, duration_s=None):
    self._connect_if_needed()
    self._action_source.connect(bus=self._bus, session_id=self._session_id)

    step = 0
    try:
        while not self._done(step, duration_s):
            loop_start = time.perf_counter()

            robot_state, camera_frames = self._read_observation()   # the ONE read for this tick
            action = self._action_source.update(robot_state, camera_frames, step)

            action = self._bus.invoke_on_action_ready(action=action, step=step)
            self._resilient_send(action)
            self._bus.invoke_on_action_sent(action=action, step=step)

            elapsed, sleep_time = self._tick_sleep(loop_start, self._goal_time)
            self._bus.emit_tick(TickEvent(
                session_id=self._session_id, step=step, timestamp=time.time(),
                robot_state=robot_state, camera_frames=camera_frames,
                action_sent=action, loop_duration_s=elapsed, sleep_time_s=sleep_time,
            ))
            step += 1
    except KeyboardInterrupt:
        pass
    finally:
        self._shutdown(step)   # action_source.disconnect(), emit shutdown, close bus

    return step
```

There is **one runtime class**, action source always required —
`PolicySource` is simply the implementation you pass for the policy case,
`TeleopSource` for teleop. There is exactly one way to build a runtime.

Key properties of the loop:

- No `Tick` wrapper, no `isinstance` anywhere — the action source is opaque.
- No hold branch — `update()` always returns something sendable.
- `run()` returns `steps: int` — see
  [Observability](#observability-no-stats-mechanism).

### PolicySource

```python
class PolicySource:
    def __init__(self, model, execution, action_queue=None, task=None): ...

    def connect(self, *, bus, session_id):
        self._execution.set_bus(bus, session_id)
        self._execution.start(self._model, self._action_queue)

    def update(self, robot_state, camera_frames, step):
        model_input = self._to_model_input(robot_state, camera_frames)
        if not self._warmed_up:
            self._execution.warmup(model_input)
            self._warmed_up = True

        self._execution.maybe_request(model_input)
        action = self._action_queue.pop()
        if action is None:
            if self._last is None:
                msg = "No action available and none produced yet (warmup may have failed)"
                raise RuntimeError(msg)
            return self._last                 # this action source's own hold decision
        self._last = action
        return action

    def disconnect(self):
        self._execution.stop()          # queued actions discarded, not flushed
```

### TeleopSource

```python
class TeleopSource:
    def __init__(self, leader: Robot, *, to_action=None):
        self._leader = leader
        self._to_action = to_action or (lambda obs: obs.joint_positions)
        self._leader_owned = False

    def connect(self, *, bus, session_id):
        if not self._leader.is_connected():
            self._leader.connect()
            self._leader_owned = True

    def update(self, robot_state, camera_frames, step):
        return self._to_action(self._leader.get_observation())   # ignores both reads

    def disconnect(self):
        if self._leader_owned:
            with contextlib.suppress(Exception):
                self._leader.disconnect()
```

Note `update()` never touches `robot_state`/`camera_frames` — the follower's
state/cameras aren't inputs to a teleop action. They're still read once per
tick by the runtime (for telemetry/recording), just not consumed by this
action source.

### Execution

`Camera.read_latest()` is a cheap buffered fetch (returns the latest
already-captured frame), not a blocking capture, so reading every tick adds
negligible IO — not worth a lazy-provider indirection threaded through every
execution. `maybe_request` takes a materialized observation directly, matching
the eager-read model used everywhere else in the loop:

```python
def maybe_request(self, observation: dict[str, np.ndarray]) -> None: ...
```

`SyncExecution`/`AsyncExecution` gate on `below_threshold` before using the
observation. `AsyncExecution`/`RTCExecution` copy the observation before
publishing to their background inference thread — that copy is these
executions protecting their own thread boundary, unrelated to camera
zero-copy semantics.

### RuntimeCallback

```python
class RuntimeCallback(Protocol):
    def on_action_ready(self, *, action: np.ndarray, step: int) -> np.ndarray: ...
    def on_action_sent(self, *, action: np.ndarray, step: int) -> None: ...
```

Plus the fire-and-forget hooks: `on_tick(TickEvent)`,
`on_inference(InferenceEvent)`, `on_lifecycle(LifecycleEvent)`.

`on_action_ready` is the one hook whose return value matters (a chain: each
callback sees the previous one's output, can transform it — e.g.
`LowPassFilterCallback` smoothing). Always returns a valid action — no `None`
sentinel; a callback that doesn't want to change anything returns its input
unchanged. Every other hook, including `on_action_sent`, is pure
notification — return value ignored.

There is no `on_hold` hook: the action source already tracks and returns its
own last action when it has nothing new (see [ActionSource](#actionsource)),
so the runtime never signals "hold" to callbacks. An action source that wants
to warn about repeats/starvation does so with its own internal counters and
`logging` calls — no callback bus involvement required.

### TickEvent

```python
@dataclass(frozen=True, slots=True)
class TickEvent:
    session_id: str
    step: int
    timestamp: float
    robot_state: RobotObservation
    camera_frames: Mapping[str, Frame]
    action_sent: np.ndarray
    loop_duration_s: float
    sleep_time_s: float
    stale_obs: bool
```

Carries the same `robot_state` / `camera_frames` values read for that tick
directly — no `Observation` wrapper, no `Tick` reference. Callbacks that read
per-tick observation data (`JsonlCallback`, `RerunCallback`, `AsyncCallback`)
use `event.robot_state` / `event.camera_frames` directly.

### Observability: no stats mechanism

`run()` returns `steps: int` — no `RunStats` type. The loop's own
`_consecutive_error_ticks` stays as **live, in-loop** state (the circuit
breaker needs it operationally) but final aggregate totals
(`transient_errors`, `stale_obs_ticks`) are **not** tracked or returned —
every individual occurrence already fires a `LifecycleEvent` (`obs_error`,
`send_error`) or is visible on `TickEvent.stale_obs` in real time.
Aggregating a final count in the runtime would duplicate the existing event
stream. Anyone wanting a total attaches a small callback instead:

```python
class ErrorCounter:                     # not runtime code — an optional plugin
    def __init__(self):
        self.transient_errors = 0
        self.stale_obs_ticks = 0

    def on_lifecycle(self, event):
        if event.event in ("obs_error", "send_error"):
            self.transient_errors += 1

    def on_tick(self, event):
        if event.stale_obs:
            self.stale_obs_ticks += 1
```

Same logic for action-source-specific numbers (`total_pops`, `total_holds`,
`inference_count`): no `stats()` method, no capability protocol. Whoever
builds the `ActionSource` already holds a reference to it, so they read its
properties directly:

```python
policy_source = PolicySource(model=model, execution=execution)
runtime = RobotRuntime(robot=robot, action_source=policy_source, fps=30)
with runtime:
    steps = runtime.run(duration_s=60)

print(policy_source.action_queue.total_pops)
print(execution.inference_count)
```

For the config-driven path, `RobotRuntime.action_source` is a public property
for exactly this reason — `runtime.action_source.action_queue.total_pops`
works the same way after a config-built run. `cli/run.py`'s summary log is
generic (`steps` only), optionally with a purely cosmetic
`isinstance(runtime.action_source, PolicySource)` check for a richer
one-line message — CLI polish, not a runtime mechanism.

**Live, per-tick numbers (e.g. a Rerun panel plotting queue depth) are a
different case** from end-of-run inspection — direct property access only
works _after_ `run()` returns. For this, an `ActionSource` emits its own
`MetricsEvent` through the bus it already receives at `connect(bus,
session_id)` — the exact same mechanism `Execution` already uses for
`InferenceEvent`. Not part of `RuntimeCallback`'s required 2 hooks; a 4th,
fully optional fire-and-forget event alongside `TickEvent`/`InferenceEvent`/
`LifecycleEvent`:

```python
@dataclass(frozen=True, slots=True)
class MetricsEvent:
    session_id: str
    step: int
    timestamp: float
    values: Mapping[str, float]
```

```python
# PolicySource.update(), after popping from the queue:
if self._bus is not None:
    self._bus.emit_metrics(MetricsEvent(
        session_id=self._session_id, step=step, timestamp=time.time(),
        values={"queue_remaining": self._action_queue.remaining},
    ))
```

`MetricsEvent` carries **only** source-owned, per-tick, live values with no
other home — currently just `queue_remaining`. It is **not** a general stats
channel: inference `latency_s`/`chunk` stay on `InferenceEvent` (fires on
completion, its own cadence), and end-of-run totals (`total_pops`,
`inference_count`) stay direct-property reads. No per-tick
`inference_requested` flag — `InferenceEvent` already signals that an
inference happened.

`RerunCallback` implements `on_metrics` and logs whatever keys it recognizes;
`TeleopSource` never emits `MetricsEvent` at all, so the panel is simply
empty for a teleop session — no capability protocol, no `isinstance` in the
runtime, same "only pay for what you use" shape as everything else here.
This was chosen over a callback-constructor callable (e.g.
`queue_remaining_fn`) specifically because a callable closing over another
object can't be expressed in a YAML config — this mechanism works
identically whether the runtime was built by hand or from a config file.

### Config: one schema, no shorthand

```yaml
runtime:
  robot:
    { class_path: physicalai.robot.SO101, init_args: { port: /dev/ttyACM0 } }
  action_source:
    class_path: physicalai.runtime.PolicySource
    init_args:
      model:
        {
          class_path: physicalai.inference.InferenceModel,
          init_args: { export_dir: ./exports/act },
        }
      execution: { class_path: physicalai.runtime.SyncExecution }
  fps: 30.0
  cameras:
    wrist:
      {
        class_path: physicalai.capture.UVCCamera,
        init_args: { device: /dev/video0 },
      }
  callbacks:
    - { class_path: physicalai.runtime.ConsoleCallback }
```

No flat/legacy shorthand, no dual schema, no YAML pre-parse peek.
`action_source` is always required and explicit. `cli/run.py` uses exactly
one parser-building path: `add_class_arguments(RobotRuntime, "runtime")` +
`add_method_arguments(RobotRuntime, "run", "run")`. Choosing which concrete
`ActionSource` class to build is ordinary `class_path`/`init_args`
polymorphism, the same mechanism `robot:`/`cameras:` already use — nothing
special.

## Implementing a New ActionSource

```python
class MyActionSource:
    def connect(self, *, bus, session_id) -> None: ...

    def update(self, robot_state: RobotObservation,
               camera_frames: Mapping[str, Frame], step: int) -> np.ndarray: ...

    def disconnect(self) -> None: ...
```

Three required methods, full stop. Queue state, hold/repeat tracking, and
per-source stats are your own private concern — the runtime never
introspects them. If you need a session-scoped telemetry channel for live
metrics, emit a `MetricsEvent` through `bus` inside `update()` (see
[Observability](#observability-no-stats-mechanism)); everything else is a
plain property your caller reads after `run()` returns.

## Empirical Validation: Zero-Copy Camera Safety

The "no copy needed" claim above isn't just reasoned about — it was tested
against real hardware, because the underlying transport (`SharedCamera`,
iceoryx2 shared memory, `zero_copy=True`) is real and shipped
(`src/physicalai/capture/transport/_shared_camera.py`).

**Mechanism, confirmed by reading source:** `_held_sample` is a single-slot
attribute on the `SharedCamera` instance, unconditionally cleared and
reassigned on every `read_latest()` call. The returned zero-copy array is
built via raw `ctypes.from_address()` with no Python-refcounting backreference
to the sample — so a previously returned `Frame`'s `.data` goes stale the
moment the _next_ `read_latest()` call runs, regardless of whether the old
`Frame` object is still referenced in Python. GC is irrelevant; only "has
`read_latest()` been called again" matters. `Frame.sequence`/`.timestamp` are
plain values copied from an already-copied header struct — stable — while
`Frame.data` can silently show completely different pixel content later, with
no exception anywhere.

**`scripts/shared_camera_race_repro.py`** (built to test this) exercises the
real `SharedCamera.read_latest()`/`_decode_sample()` code; only the iceoryx2
subscriber is faked for the default mode. It also supports `--real-camera`
(auto-discovers a real camera via `physicalai.capture.discovery.discover_all()`,
real iceoryx2 + `CameraPublisher` subprocess) and `--single-reader` (isolates
the single-reader cadence our design actually uses).

Run against real hardware (UVC camera) with an adversarial second reader
(~11M `read_latest()` calls/sec via GIL handoff during the main thread's
sleeps): **136/150 ticks (91%) showed a held frame's content change between
the start and end of a simulated tick-work window.** Confirms the hazard is
real and frequent on real hardware, not theoretical.

**What this does and doesn't prove about our design:** that run used an
adversarial _second reader_ on one `SharedCamera` instance — explicitly
outside its documented single-reader contract, and not something our design
ever does (one reader, always). But the same underlying mechanism applies to
a _single_ reader whenever a **deferred** consumer (`AsyncCallback`'s worker
thread, `AsyncExecution`/`RTCExecution`'s inference thread) processes a frame
after the reader's own next tick has already happened — which is exactly why
those two places already carry a local copy, and why no additional/blanket
copy is needed anywhere else. The 91% figure is an upper-bound existence proof
for the mechanism, not a measurement of our design's own risk level.

**Caveat found mid-investigation, worth keeping in mind if this script is used
again:** byte-diff-percentage is not a reliable tear-vs-clean-swap signal for
organic (real) image content — moving the camera mid-run alone shifted
measured diff% enough to flip an arbitrary classification threshold, proving
the percentage tracks scene motion/similarity, not whether a read was torn.
The script reports `UNCHANGED` vs. `CHANGED` for real-camera mode; it does not
classify _how_ it changed.

## Decision Summary

```text
entry point                one RobotRuntime class; action_source always required
action source protocol     ActionSource: connect(bus, session_id), update(robot_state, camera_frames, step), disconnect()
concrete sources           PolicySource, TeleopSource
observation type           none — two params: robot_state (RobotObservation) + camera_frames; no Observation class
capability protocols       none — no isinstance anywhere in the runtime
shutdown                   disconnect() returns None; no queue drain — queued actions discarded,
                           robot holds last commanded position
per-tick reads              eager; plain robot_state + camera_frames, shared by reference; no Tick wrapper
maybe_request               eager: takes a materialized observation directly (read_latest is a cheap
                           buffered fetch, negligible per-tick IO)
warmup                      no dedicated step; folded into each action source's own first update()
hold handling                deleted; update() always returns a sendable action; fallback is the
                           action source's own internal decision
copy safety                  no copy at the runtime read step; single-read-per-tick invariant enforced
                           structurally (ActionSource/RuntimeCallback never see Camera/Robot refs);
                           AsyncCallback + Async/RTCExecution keep their own local copies
callbacks                    RuntimeCallback: on_action_ready (return matters), on_action_sent
                           (notification) + on_tick/on_inference/on_lifecycle
live metrics                source-owned per-tick values via optional MetricsEvent on the bus
                           (queue_remaining only); latency stays InferenceEvent; totals direct-access
stats                        no RunStats type; run() returns steps: int; everything else via direct
                           property access on runtime.action_source, or an opt-in callback for aggregates
config                        one schema only — action_source: always explicit, no flat shorthand
```
