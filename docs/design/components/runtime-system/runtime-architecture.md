# Runtime Redesign

## TL;DR

- **One `RobotRuntime` class.** No `PolicyRuntime`. An action source is a
  required constructor argument, not an optional subclass path —
  `PolicySource` is just the implementation you pass for a policy rollout.
- **`ActionSource` is 3 methods:** `connect()`, `update()`, `disconnect()`, all
  returning `None` bar `update()`. No capability protocols, no `isinstance`
  anywhere in the runtime. **No shutdown drain** — `disconnect()` tears down and
  returns nothing; any queued actions are discarded and the robot holds its last
  commanded position.
- **No `Tick` class.** The runtime reads robot state + camera frames once per
  tick and passes the same values to the action source and to telemetry —
  eager, not lazy.
- **No copy needed for camera frames** at the runtime's read step — verified
  against real hardware, not just reasoned about (see
  [Empirical Validation](#empirical-validation-zero-copy-camera-safety)). The
  only copies that exist are local to `AsyncCallback` and
  `Async`/`RTCExecution`, unchanged.
- **No "hold" concept.** `update()` always returns a sendable action; what to
  do when there's nothing new (repeat last, safe pose, whatever) is the action
  source's own decision, made internally.
- **No stats mechanism in the runtime.** `run()` returns just `steps: int`.
  Anything else (inference counts, error totals) is read directly off objects
  you already built, or via a tiny opt-in callback.
- **One config schema.** `action_source:` is always explicit — no flat/legacy
  shorthand. Existing example configs need a small migration.
- **Callback hooks:** `on_action_ready` (can transform the action) and
  `on_action_sent` (notification only), plus the existing
  `on_tick`/`on_inference`/`on_lifecycle`. `before_send_action` and `on_hold`
  are gone.

See the [Decision Summary](#decision-summary) at the end for the compact
reference-card version of all of this. Read on for the reasoning behind each
call.

## Why a rebuild, not a patch

The original version of this document proposed simplifying the shipped
`Controller`/`Tick`/4-capability-protocol design in place. Reviewing that
proposal against the actual codebase surfaced concrete, verifiable problems,
not just style preferences:

- Turning `PolicyRuntime` into a bare factory function (as proposed) breaks
  `cli/run.py`'s jsonargparse wiring, which requires a class exposing `run()`
  — the sibling doc had already reasoned through and rejected this exact move.
- The proposed slimmed `TickEvent` drops `queue_remaining`, which 3 of 4
  shipped callbacks read — contradicting the same document's claim that
  callbacks stay "unchanged."
- The proposed slimmed `RunStats` drops fields `cli/run.py`'s own summary log
  reads directly.
- The proposed `PolicyController.stop()` needs to send queued actions to the
  robot but is never given a robot reference.
- Moving bus/session-id injection to construction time loses the fresh
  `session_id` the runtime generates every `run()` call, breaking multi-session
  telemetry.
- The "what gets deleted" savings estimate was measured against a stale
  (pre-implementation) mental model — the actual `PolicyRuntime` class today
  is 44 lines, not the ~110 the estimate assumed.

Rather than patch around each of these individually, the design was rebuilt
from first principles: why does a runtime need to exist at all, why does it
need to be generic rather than policy-only, and what should it actually own.
That process (and a real-hardware investigation into a specific safety
question it raised — see [Empirical validation](#empirical-validation-zero-copy-camera-safety))
produced the design below.

## First-Principles Requirements

### Why a runtime at all

- Avoid rewriting the control loop for every new use case.
- Config-driven execution — decouples _what to run_ (a config file) from
  _how the loop works_ (code), so a run can be launched without writing Python.
- IO resilience is safety-critical and easy to get subtly wrong per call site
  (retry-with-backoff on transient errors, stale-observation fallback,
  consecutive-error circuit breaker) — centralizing it means it's implemented
  and tested once, not reinvented per script.
- Timing correctness (drift-free fixed-rate loop) is easy to get wrong with a
  naive `sleep(1/fps)`, and matters for recorded-data quality.
- One audit trail — every run emits the same event stream regardless of who's
  driving the arm, so one console view / recorder / visualizer works for every
  use case instead of one-off print statements per script.
- Lowers the floor for non-experts — "run this policy" becomes a config file,
  not a systems-programming exercise.
- Non-goal: one-off diagnostic scripts (e.g. `examples/so101/move_joints.py`)
  don't need any of this. The line is: anything that runs repeatedly, is
  timed, is recorded, or is safety-relevant needs the runtime; a hardware
  smoke test doesn't.

### Why a generic (pluggable action source) runtime, not policy-only

- **Teleop is data collection, not a side feature.** If policy rollout is the
  "real" loop and teleop is a separate hand-rolled script, the timing/capture
  semantics used to _collect_ demonstrations and the ones used to _replay_ a
  trained policy are two different code paths — any skew between them
  (exactly when the observation is captured relative to the action) is a
  train/deploy mismatch waiting to happen.
- **HIL and DAgger need to switch or mix action sources.** That has to be a
  seam from day one, or they become a second loop implementation instead of a
  plugin.
- **A future safety/supervisory layer wants one place to intercept the
  action, regardless of who's driving** — including a human in teleop.
- **Review surface** — a new action source should be one small, reviewable
  class, not a fork of the loop.
- Caveat, held throughout this redesign: "generic" means **exactly one right
  seam** (the action source), not maximal configurability everywhere. The
  original implementation's mistake wasn't that pluggability was wrong — it
  was 4 speculative capability protocols nobody but one consumer used. Do not
  re-derive that mistake.

### What the runtime owns vs. delegates

Two genuinely distinct seams — kept structurally separate throughout:

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

## Problem Statement (original diagnosis — still valid)

The pre-redesign implementation had ~25 named types in the runtime package. A
developer had to learn the `Controller` protocol, 4 capability protocols
(`SupportsBus`, `SupportsStats`, `SupportsDrain`, `SupportsHoldInfo`), the
`Tick` class with its memoization contract, the `PolicyRuntime` subclass, and
the runtime's `isinstance` dispatch logic before writing or modifying
anything. Specific issues that motivated the rebuild:

1. **Capability-protocol proliferation.** `PolicyController` implements all 4
   optional protocols; `TeleopController` implements essentially none. The
   protocols describe exactly one consumer — they are premature abstractions.
2. **`isinstance` dispatch in the loop.** The runtime checks capability
   protocols to decide how to interact with the controller — tight coupling
   expressed through fragmented interfaces rather than a direct call.
3. **Dual entry points.** `PolicyRuntime` (subclass) and `RobotRuntime` +
   `PolicyController` do the same thing via different paths, confusing newcomers.

## Design Principles

- One way to do things, not two — there is exactly one runtime class and one
  required action-source protocol, no shortcut class and no dual config schema.
- The runtime loop body is readable top-to-bottom with no `isinstance` calls.
- `ActionSource`: the minimum a developer must implement to plug in a new
  action source — 3 required methods, nothing optional.
- Internals (queue state, hold/repeat tracking, per-source stats) are the
  action source's private concern — the runtime does not introspect them.
- Config/CLI compatibility with the pre-redesign flat schema is **not**
  preserved — see [Config](#config-one-schema-no-shorthand) for why that's a
  deliberate, accepted trade rather than an oversight.

## Target Architecture

### Overview

```text
RobotRuntime     the fixed-rate loop: read → decide → send → sleep → emit. One class.
ActionSource     decides the action, every tick. Required, not optional.
PolicySource     adapts model + execution + queue into ActionSource
TeleopSource     reads a leader device into ActionSource
Callbacks        side-effects and action transforms — a distinct seam from ActionSource
```

### ActionSource

```python
class ActionSource(Protocol):
    def connect(self, *, bus: _CallbackBus, session_id: str) -> None: ...
    def update(self, robot_state: RobotObservation,
               camera_frames: Mapping[str, Frame], step: int) -> np.ndarray: ...
    def disconnect(self) -> None: ...
```

Three required methods. No capability protocols, no `isinstance` anywhere in
the runtime. `Protocol`, not ABC — matches the existing `Robot` protocol
precedent (`src/physicalai/robot/interface.py`); jsonargparse doesn't care
either way, it instantiates the concrete class named in `class_path`, never the
shared protocol type.

- **`connect(bus, session_id)`** — resource setup (spawn threads, connect a
  leader device). Folds into the _same_ connect/disconnect lifecycle stage the
  runtime already runs for robot + cameras (one stage, one
  rollback-on-failure path — not a separate stage). `bus`/`session_id` are
  handed here because `connect()` is called fresh every `run()`, which is
  exactly when the runtime generates a new `session_id` — construction-time
  injection would miss that.
- **`update(robot_state, camera_frames, step) -> np.ndarray`** — the per-tick decision.
  **Always returns a sendable action — no `None` sentinel.** What to do when
  there's nothing new to decide (repeat the last action, go to a safe pose,
  whatever) is entirely the action source's own call, made internally. If it
  truly cannot produce anything (e.g. warmup never succeeded), it raises —
  handled by the existing fail-stop semantics for `update()` errors. No
  runtime-level "hold" concept exists at all.
- **No dedicated warmup step.** The runtime calls `update()` on tick 0 like any
  other tick. An action source that needs to seed internal state before its
  first real decision (e.g. `PolicySource` discovering `chunk_size`)
  special-cases its own first `update()` call internally (a private
  not-yet-seeded flag). Accepted consequence: tick 0 may take longer — visible
  as a `loop_duration_s` spike on `TickEvent`, not hidden in a pre-loop phase.
  Today's dedicated 5-attempt/1s-backoff warmup retry is not reproduced by the
  runtime; an action source that wants that resilience implements it itself.
- **`disconnect() -> None`** — teardown only (stop threads, release a leader
  device). **No drain:** queued-but-unsent actions are discarded, not flushed.
  On shutdown the robot simply stops receiving new actions and holds its last
  commanded position — safer than replaying a stale queue after the operator
  hit stop, and it removes the whole `SupportsDrain`/runtime-flush/pacing path.
  The action source never receives a robot reference.

### Reads: robot state + camera frames (no `Observation` type)

The runtime reads robot state and camera frames **once per tick** and passes
them as **two plain values**, not a wrapper:

- `robot_state: RobotObservation` — the robot's own reading, reused as-is.
- `camera_frames: Mapping[str, Frame]` — the standalone `cameras=` sources.

No new `Observation` type. `TickEvent` already carries these as two separate
fields, so a wrapper would only be a transient carrier into `update()` — and
the codebase already has two distinct "observation" senses (`RobotObservation`;
the inference model-input dict), so a third would only add confusion.
`RobotObservation` stays the robot's product; standalone cameras stay a
separate arg because they are read outside the robot and
`RobotObservation.images` is contractually built-in cameras only. If a third
sensor kind ever appears, `update()`/`TickEvent` grow a param — a non-breaking
retrofit, and YAGNI today.

Eager, plain values — no `Tick`, no lazy pull, no memoization contract.
Ordinary Python object-reference passing, not a special mechanism.

**Hard invariant, and the reason no copy is needed:** exactly one
`read_latest()`/`get_observation()` call per device, per tick — nothing may
re-read a device until the next tick. Enforced structurally, not by a runtime
check: `ActionSource` and `RuntimeCallback` never receive `Camera`/`Robot`
object references, only already-read `Frame`/`RobotObservation` values — so
nothing downstream even has a handle to call `read_latest()` again.

Given that invariant, **no copy is needed at the runtime's read step.**
Synchronous consumers (the action source, any non-deferred callback) finish
using those values before the next tick's read can invalidate anything. The
two places that _do_ need a copy — because they hand a value to a different
thread for later processing — already have one, unaffected by this redesign:
`AsyncCallback` (copies non-owned frame buffers before enqueueing to its
background worker) and `AsyncExecution`/`RTCExecution` (copy observation
arrays before publishing to their background inference thread). See
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

There is **no separate `PolicyRuntime` class.** One runtime, action source
always required — `PolicySource` is simply the implementation you pass for
the policy case. This fully resolves the old "dual entry points" problem
(shipped `PolicyRuntime` vs. `RobotRuntime`+`PolicyController`) — there's exactly one
way to build a runtime now.

Key differences from the pre-redesign loop:

- No `Tick`, no `isinstance` anywhere — the action source is opaque.
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

### Execution (unchanged in spirit, one signature revert)

The shipped code passes `maybe_request` a **provider** (`observe_fn`) so an
async/idle tick reads no camera at all (`execution.py`, plus a main-thread
`below_threshold` pre-check in `rtc_execution.py`). That pull machinery is
**deleted**: `Camera.read_latest()` is a cheap buffered fetch (returns the
latest already-captured frame), not a blocking capture, so reading every tick
adds negligible IO — not worth a lazy-provider indirection threaded through all
three executions. `maybe_request` takes a materialized observation directly
again, matching the eager-read model:

```python
def maybe_request(self, observation: dict[str, np.ndarray]) -> None: ...
```

`SyncExecution`/`AsyncExecution` already gate on `below_threshold` before using
the observation — unchanged. `AsyncExecution`/`RTCExecution` still copy the
observation before publishing to their background inference thread
(`execution.py:233`, `rtc_execution.py:269`) — that copy was never about
camera zero-copy semantics, it's these executions protecting their own thread
boundary, and it stays exactly as-is.

### RuntimeCallback

```python
class RuntimeCallback(Protocol):
    def on_action_ready(self, *, action: np.ndarray, step: int) -> np.ndarray: ...
    def on_action_sent(self, *, action: np.ndarray, step: int) -> None: ...
```

Plus the existing fire-and-forget hooks, unchanged: `on_tick(TickEvent)`,
`on_inference(InferenceEvent)`, `on_lifecycle(LifecycleEvent)`.

`on_action_ready` is the one hook whose return value matters (a chain: each
callback sees the previous one's output, can transform it — e.g.
`LowPassFilterCallback` smoothing). Always returns a valid action — no `None`
sentinel; a callback that doesn't want to change anything returns its input
unchanged. Every other hook, including `on_action_sent`, is pure notification —
return value ignored. `on_hold` is **deleted**.

**Why `on_hold` is gone:** the pre-redesign `PolicyController.update()` already
tracked and returned its own last action when its queue was empty —
`on_hold`/`SupportsHoldInfo` were purely an additional reporting side-channel
on top of a decision the controller had already made. None of the four
shipped callbacks (`Console`/`Jsonl`/`Async`/`Rerun`) did anything with it —
`AsyncCallback` explicitly refused to forward it — so it was vestigial in
practice. Now that `update()` always returns a sendable action and never
signals "hold" to the runtime, there's nothing for the runtime to report. An
action source that wants to warn about repeats/starvation does so with its own
internal counters and `logging` calls — no callback bus involvement required.

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
directly — no `Observation` wrapper, no `Tick` reference. Breaking change for
the 3 callbacks that read `event.tick.*` today
(`JsonlCallback`, `RerunCallback`, `AsyncCallback`) — they switch to
`event.robot_state` / `event.camera_frames` directly. `queue_remaining` is
dropped (it was action-source-specific — see Observability below for where
that kind of number lives now).

### Observability: no stats mechanism

`run()` returns `steps: int` — no `RunStats` type. The loop's own
`_consecutive_error_ticks` stays as **live, in-loop** state (the circuit
breaker needs it operationally) but final aggregate totals
(`transient_errors`, `stale_obs_ticks`) are **not** tracked or returned — every
individual occurrence already fires a `LifecycleEvent` (`obs_error`,
`send_error`) or is visible on `TickEvent.stale_obs` in real time. Aggregating
a final count in the runtime duplicates the existing event stream. Anyone
wanting a total attaches a small callback instead:

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
`inference_count`): no `stats()` method, no capability protocol. Whoever builds
the `ActionSource` already holds a reference to it, so they read its
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
works the same way after a config-built run. `cli/run.py`'s summary log
becomes generic (`steps` only), optionally with a purely cosmetic
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
`inference_count`) stay direct-property reads. No per-tick `inference_requested`
flag — `InferenceEvent` already signals that an inference happened.

`RerunCallback` implements `on_metrics` and logs whatever keys it recognizes;
`TeleopSource` never emits `MetricsEvent` at all, so the panel is simply
empty for a teleop session — no capability protocol, no `isinstance` in the
runtime, same "only pay for what you use" shape as everything else here. This
was chosen over a callback-constructor callable (e.g. `queue_remaining_fn`)
specifically because a callable closing over another object can't be
expressed in a YAML config — this mechanism works identically whether the
runtime was built by hand or from a config file.

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
`action_source` is always required and explicit. `cli/run.py` collapses to
exactly one parser-building path: `add_class_arguments(RobotRuntime, "runtime")`

- `add_method_arguments(RobotRuntime, "run", "run")` — what today's
  `_build_general_parser` already does, and the only thing needed. Choosing
  which concrete `ActionSource` class to build is ordinary `class_path`/
  `init_args` polymorphism, the same mechanism `robot:`/`cameras:` already use —
  nothing special.

This drops backward compatibility with the flat schema entirely. Accepted
cost: existing example configs (`examples/runtime/*.yaml`) and direct
constructor call sites (e.g. `examples/runtime/demo_loop.py`'s
`PolicyRuntime(robot=..., model=..., execution=..., ...)` calls) need a small,
mechanical migration to wrap `model`/`execution` under an `action_source:`
block. Consistent with the round-1 argument for building this generically now:
migration cost is low today (internal users only) and only grows later.

## Empirical Validation: Zero-Copy Camera Safety

The "no copy needed" claim above isn't just reasoned about — it was tested
against real hardware, because the underlying transport (`SharedCamera`,
iceoryx2 shared memory, `zero_copy=True`) is real and already shipped
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

## What Gets Deleted

| Item                                                                | Reason                                                                                         |
| ------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| `tick.py` (`Tick` class)                                            | Eager reads shared by reference; no lazy pull needed                                           |
| `SupportsBus`, `SupportsStats`, `SupportsDrain`, `SupportsHoldInfo` | Folded into 3 required `ActionSource` methods; no isinstance dispatch                          |
| `isinstance` dispatch in `run()`                                    | Gone — action source is opaque                                                                 |
| `PolicyRuntime` class                                               | One `RobotRuntime`, action source always required                                              |
| `RunStats` type                                                     | `run()` returns `steps: int`; everything else via direct property access or an opt-in callback |
| `on_hold` callback hook                                             | Vestigial — no shipped callback used it; action sources own their own fallback/logging         |
| Warmup retry loop in the runtime                                    | Folds into each action source's own first `update()`                                           |
| `maybe_request(observe_fn)` provider / lazy pull                    | Eager per-tick read; `read_latest()` is a cheap buffered fetch, negligible IO                  |
| Shutdown queue drain (`SupportsDrain`, runtime flush + pacing)      | `disconnect()` returns `None`; queued actions discarded, robot holds last position             |
| Flat/legacy config schema                                           | One schema only — `action_source:` always explicit                                             |

## What Stays Unchanged

| Component                                          | Why                                                 |
| -------------------------------------------------- | --------------------------------------------------- |
| `Execution` ABC + Sync/Async/RTC                   | Core inference scheduling — irreducible complexity  |
| `ActionQueue` protocol + implementations           | Queue mechanics unchanged                           |
| `_CallbackBus` dispatch                            | Unchanged (2 fewer hooks to dispatch)               |
| `InferenceEvent`, `LifecycleEvent`                 | Unchanged                                           |
| Resilient read/send/circuit-breaker logic          | Same logic, same owner (the runtime)                |
| Observer module (telemetry subscriber)             | Unchanged                                           |
| Smoothers (Lerp/Replace)                           | Unchanged                                           |
| `AsyncCallback`'s frame-copy safety check          | Unchanged — still the right, and only, place for it |
| `AsyncExecution`/`RTCExecution`'s observation copy | Unchanged — still the right, and only, place for it |

## Migration

### For callback authors

```python
# Before
def on_tick(self, event: TickEvent):
    state = event.tick.robot_state()
    frames = event.tick.camera_frames()

# After
def on_tick(self, event: TickEvent):
    state = event.robot_state
    frames = event.camera_frames
```

```python
# Before
def before_send_action(self, *, action, step) -> np.ndarray | None: ...

# After (renamed, return is now required)
def on_action_ready(self, *, action, step) -> np.ndarray: ...
```

`on_hold` has no replacement — delete it from any callback that implements it.

### For action source implementors

```python
# Before: 3 required + up to 4 optional capability protocols
class MyController:
    def start(self) -> None: ...
    def warmup(self, tick: Tick) -> None: ...
    def update(self, tick: Tick) -> np.ndarray | None: ...
    def stop(self) -> None: ...
    def reset(self) -> None: ...
    def set_bus(self, bus, session_id) -> None: ...     # if SupportsBus
    def stats(self) -> dict: ...                        # if SupportsStats
    def drain(self, limit) -> Iterable: ...             # if SupportsDrain
    # + last_was_hold, holds properties                 # if SupportsHoldInfo

# After: 3 required methods, full stop
class MyActionSource:
    def connect(self, *, bus, session_id) -> None: ...
    def update(self, robot_state: RobotObservation,
               camera_frames: Mapping[str, Frame], step: int) -> np.ndarray: ...
    def disconnect(self) -> None: ...
```

### For config authors

Wrap `model`/`execution` under `action_source:`:

```yaml
# Before
runtime:
  robot: {...}
  model: {...}
  execution: {...}
  fps: 30.0

# After
runtime:
  robot: {...}
  action_source:
    class_path: physicalai.runtime.PolicySource
    init_args: { model: {...}, execution: {...} }
  fps: 30.0
```

## Phasing

1. **Core loop.** `ActionSource` protocol (`update` takes `robot_state` +
   `camera_frames`, no `Observation` type), `RobotRuntime` loop rewrite (no
   `Tick`, no isinstance, no hold branch), `RuntimeCallback` renamed/shrunk to
   2 hooks, `TickEvent` carries plain values. Delete `tick.py`, the 4
   capability protocols, `PolicyRuntime`, `RunStats`.
2. **Config/CLI.** Collapse `cli/run.py` to one parser path. Migrate
   `examples/runtime/*.yaml` and direct-construction call sites (`demo_loop.py`
   and others) to the single schema.
3. **Verify.** `TeleopSource` re-sketched under the new shape (above) — a
   mechanically small change. Existing tests updated for the renamed callback
   hooks and `TickEvent` shape; fault-tolerance tests should be largely
   unaffected (same resilient-IO logic, different call sites).

## Decision Summary

```text
entry point                one RobotRuntime class; action_source always required
action source protocol     ActionSource: connect(bus, session_id), update(robot_state, camera_frames, step), disconnect()
concrete sources           PolicySource, TeleopSource (renamed from shipped PolicyController/TeleopController)
observation type           none — two params: robot_state (RobotObservation) + camera_frames; no new Observation class
capability protocols       deleted (SupportsBus, Stats, Drain, HoldInfo) — no isinstance anywhere
shutdown                   disconnect() returns None; no queue drain — queued actions discarded,
                           robot holds last commanded position
Tick class                 deleted; eager reads, plain robot_state + camera_frames, shared by reference
maybe_request              eager: materialized observation again; observe_fn pull deleted
                           (read_latest is a cheap buffered fetch, negligible per-tick IO)
warmup                     no dedicated step; folded into each action source's own first update()
hold handling               deleted; update() always returns a sendable action; fallback is the
                           action source's own internal decision
copy safety                 no copy at the runtime read step; single-read-per-tick invariant enforced
                           structurally (ActionSource/RuntimeCallback never see Camera/Robot refs);
                           AsyncCallback + Async/RTCExecution keep their existing local copies
callbacks                   RuntimeCallback: on_action_ready (return matters), on_action_sent
                           (notification) + existing on_tick/on_inference/on_lifecycle
live metrics               source-owned per-tick values via optional MetricsEvent on the bus
                           (queue_remaining only); latency stays InferenceEvent; totals direct-access
stats/RunStats              deleted; run() returns steps: int; everything else via direct property
                           access on runtime.action_source, or an opt-in callback for aggregates
config                       one schema only — action_source: always explicit, no flat shorthand
```
