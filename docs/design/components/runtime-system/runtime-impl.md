# Runtime Redesign — Implementation Plan

Status: **ready** — execution checklist for
[runtime-simplification.md](./runtime-simplification.md) ("Runtime Redesign",
status: decided). That document is the source of truth for **what** and **why**;
this one is the disposable **how** — file-by-file changes, ordering, deletions,
and the test plan. When the two disagree, the design doc wins; fix this one.

This is a **breaking** refactor with no backward-compatibility layer
(`PolicyRuntime` and the flat config schema are removed on purpose). Do it on a
branch. Expect the test suite to be red mid-flight; the ordering below front-loads
the core types so downstream edits compile against them, and Phase 8 brings tests
back to green in one focused pass.

## Naming map (applies everywhere)

| Old (shipped)             | New                                             |
| ------------------------- | ----------------------------------------------- |
| `PolicyController`        | `PolicySource`                                  |
| `TeleopController`        | `TeleopSource`                                  |
| `Controller` protocol     | `ActionSource` protocol                         |
| `PolicyRuntime` (class)   | _deleted_ — use `RobotRuntime` + `PolicySource` |
| `RunStats`                | _deleted_ — `run() -> int` (steps)              |
| `before_send_action` hook | `on_action_ready` (return required)             |
| `on_hold` hook            | _deleted_                                       |
| `Tick`                    | _deleted_ — two plain values per tick           |

## Ordering strategy

Bottom-up, so each phase compiles against already-landed lower layers:

1. Events + bus (leaf types).
2. Action sources (`controller.py`).
3. Runtime loop (`runtime.py`).
4. Callbacks.
5. CLI / config.
6. Telemetry / observer.
7. Examples + config migration.
8. Tests green + docs.

Run `pytest tests/unit/runtime` at the end of each phase to see the shrinking
break surface; don't expect green before Phase 8.

---

## Phase 1 — Events + callback bus

**`src/physicalai/runtime/events.py`**

- `TickEvent`: replace `tick: Tick` with `robot_state: RobotObservation` and
  `camera_frames: Mapping[str, Frame]`. **Remove** `queue_remaining`. Keep
  `stale_obs`, `action_sent`, `loop_duration_s`, `sleep_time_s`.
- Add `MetricsEvent(session_id, step, timestamp, values: Mapping[str, float])`.
- `InferenceEvent`, `LifecycleEvent`: unchanged.

**`src/physicalai/runtime/_callback_bus.py`**

- Rename `invoke_before_send_action` → `invoke_on_action_ready` (dispatch
  `on_action_ready`; return is required, so drop the `if modified is not None`
  guard — a callback must return an action).
- **Delete** `invoke_on_hold`.
- Add `emit_metrics(event: MetricsEvent)` — fire-and-forget dispatch to
  `on_metrics`, exceptions isolated (mirror `emit_lifecycle`). Called on the
  control thread from `PolicySource.update()`, so no queue needed (unlike
  `emit_inference`).
- `emit_tick` / `emit_inference` / `emit_lifecycle` / `close`: unchanged.

## Phase 2 — Action sources (`src/physicalai/runtime/controller.py`)

Consider renaming the module to `action_source.py` (optional; update imports if so).

- **Delete** the `Controller` protocol and all four capability protocols
  (`SupportsBus`, `SupportsStats`, `SupportsDrain`, `SupportsHoldInfo`).
- Add `ActionSource(Protocol)`: `connect(*, bus, session_id) -> None`,
  `update(robot_state, camera_frames, step) -> np.ndarray`, `disconnect() -> None`.
- `PolicyController` → **`PolicySource`**:
  - `update(self, robot_state, camera_frames, step)` — build model input once via
    `_to_model_input(robot_state, camera_frames)`; on first call do
    `execution.warmup(...)` behind a `self._warmed_up` flag (no runtime warmup);
    then `maybe_request(model_input)`, `pop()`, and return `self._last` on empty
    (its own hold decision, raise if never seeded).
  - `connect(*, bus, session_id)`: `execution.set_bus(bus, session_id)` +
    `execution.start(...)`.
  - `disconnect()`: `execution.stop()` only — **no drain**, no queue flush.
  - Keep `action_queue` property (public, for end-of-run stats access).
  - **Delete** `drain()`, `stats()`, `last_was_hold`, `holds`, `remaining`
    (capability-protocol surface).
  - Emit `MetricsEvent({"queue_remaining": self._action_queue.remaining})` via the
    bus at the end of `update()` (guard `if self._bus is not None`). Store the bus
    from `connect()`.
- `TeleopController` → **`TeleopSource`**: `update(self, robot_state, camera_frames,
step)` ignores both reads; `disconnect()` drops the trailing `return ()`.

## Phase 3 — Runtime loop (`src/physicalai/runtime/runtime.py`)

- **Delete** `PolicyRuntime`, `RunStats`, `LowPassFilterCallback.on_hold`,
  `RuntimeCallback.before_send_action`/`on_hold` (the protocol stub), and the
  module-level `_config_has_class_path` peek (`runtime.py:43-56`). This is a
  **twin** of the near-identically-named helpers in `cli/run.py` (Phase 5) —
  two functions in two files; delete both or `from_config` keeps a dead branch.
- `from_config` (`runtime.py:~305`): drop the `_config_has_class_path` branch and
  collapse to a single path — `add_class_arguments(RobotRuntime, "runtime")` +
  `add_method_arguments(RobotRuntime, "run", "run")`.
- `RuntimeCallback` protocol: `on_action_ready(*, action, step) -> np.ndarray`,
  `on_action_sent(*, action, step) -> None`. The fire-and-forget hooks
  (`on_tick`/`on_inference`/`on_lifecycle`/`on_metrics`) stay **duck-typed via
  `getattr` in the bus** — do not add them to the protocol and do not "fix" the
  getattr dispatch.
- `RobotRuntime.__init__`: `controller: Controller` → `action_source: ActionSource`
  (required, no default). Drop `SupportsBus/Drain/Stats/HoldInfo` imports.
  Tightening `callbacks: Sequence[Any]` (`runtime.py:200`) → `Sequence[RuntimeCallback]`
  is **cosmetic only** (hooks are duck-typed) — optional, and must not change dispatch.
- `run() -> int`:
  - `self._action_source.connect(bus=self._bus, session_id=self._session_id)` after
    `_reset_session()` (no `isinstance(SupportsBus)` check).
  - **Delete** `_warmup_with_retry` and its call — warmup is the action source's job.
  - Loop body: `robot_state, camera_frames = self._read_observation()` (a tuple);
    `action = self._action_source.update(robot_state, camera_frames, step)`;
    `action = self._bus.invoke_on_action_ready(...)`; `_resilient_send`;
    `invoke_on_action_sent`; `emit_tick(TickEvent(..., robot_state=..., camera_frames=...))`.
    **No** hold branch, **no** `None` action branch, **no** `isinstance`.
  - Return `step`.
- Add `_read_observation() -> tuple[RobotObservation, dict[str, Frame]]` (fold
  today's `_read_robot_resilient` + `_read_cameras_resilient`; carry `stale_obs`
  out for `TickEvent`). Keep `_resilient_send`, retry/circuit-breaker, connect/
  disconnect, `from_config`.
- `_shutdown(step)`: `action_source.disconnect()` → emit `shutdown` lifecycle →
  `bus.close()`. **Delete** the `SupportsDrain` drain-send-pacing block and the
  `SupportsStats` summary read.
- Keep the `action_source` public property (config-path stats access).
- Keep `_consecutive_error_ticks` live (circuit breaker); drop the final
  `transient_errors`/`stale_obs_ticks` aggregates from the return value (they stay
  as live counters only if the loop still needs them; otherwise delete).

## Phase 4 — Callbacks (`src/physicalai/runtime/callbacks.py`)

- All `event.tick.robot_state()` / `event.tick.camera_frames()` →
  `event.robot_state` / `event.camera_frames` (Console, Jsonl, Rerun).
- Remove every `event.queue_remaining` read (Console line ~56, Jsonl ~83, Rerun
  ~268). Rerun's live queue-depth plot moves to a new `on_metrics(event)` that
  reads `event.values["queue_remaining"]`.
- Rename `before_send_action` → `on_action_ready` where implemented
  (`LowPassFilterCallback`), return required.
- Delete `on_hold` implementations. Update `AsyncCallback._ACTION_HOOKS`
  (drop `on_hold`, rename `before_send_action`).
- `AsyncCallback` frame-copy: it copied borrowed frames via `tick._set_camera_frames`;
  now it copies `event.camera_frames` directly before enqueuing (the `Tick` helper
  is gone). Keep the copy — still required for the deferred-thread boundary.

## Phase 5 — CLI / config (`src/physicalai/cli/run.py`)

- Delete `_peek_config_uses_general_schema`, `_yaml_has_runtime_class_path`,
  `_build_legacy_parser`, `_build_general_parser`, and the flat/general branching.
  (Twin peek `_config_has_class_path` lives in `runtime.py` — deleted in Phase 3;
  don't miss either.)
- `build_parser`: one path — `add_class_arguments(RobotRuntime, "runtime")` +
  `add_method_arguments(RobotRuntime, "run", "run")`.
- `run(...)`: `runtime.run()` returns `int`; summary log becomes
  `"Run complete: %d steps"`. Optional cosmetic
  `isinstance(runtime.action_source, PolicySource)` for a richer line — CLI-only,
  not required.
- Update `_HELP_TEMPLATE` (drop the two-schema description; document the single
  `action_source:` schema).
- **Verify:** `add_class_arguments(RobotRuntime)` exposes `action_source.class_path`
  for the `Protocol`-typed param. Precedent is strong (`robot:` is `Robot`-Protocol-
  typed and works today) — confirm with a `--print_config` smoke test.

## Phase 6 — Telemetry / observer (minimal — no live wire exists)

**Reality check:** `TelemetryEmitter.emit_tick` (`_telemetry.py`) is **never
called** anywhere in `src/` — the only live `emit_tick` is `_CallbackBus.emit_tick`
(`runtime.py:385`). `_telemetry.py` is an **unwired** zenoh emitter; only its
`_decode_numpy` is imported (by `observer/_subscriber.py:9`). So there is **no
existing `TickEvent` → telemetry data flow to re-source**, and nothing currently
publishes the `queue_remaining` that `observer/_console.py:26` reads.

Correct (small) scope — do **not** build an `on_metrics` publish path:

- **`_telemetry.py:73`**: drop the now-unused `queue_remaining` param from
  `TelemetryEmitter.emit_tick`'s signature (it referenced the removed `TickEvent`
  field). No re-sourcing.
- **`observer/_console.py:26`**: leave as-is — its
  `payload.get("queue_remaining", "?")` fallback already handles the field being
  absent.
- Wiring telemetry to `MetricsEvent` at all is a **separate, optional** decision,
  out of scope here. Don't imply a flow that isn't connected.

## Phase 7 — Examples + config migration

Committed artifacts only. (Local-only, uncommitted files — e.g. extra
`examples/runtime/*.yaml` and `demo_loop.py` — get the same mechanical treatment
if present, but are not part of this plan.)

- **YAML — `examples/runtime/runtime.yaml`** (the only committed config): it is the
  **flat schema** today (top-level `model:`/`execution:` under `runtime:`,
  `runtime.yaml:31-38`). Migrate: wrap `model` + `execution` (+ `action_queue`/`task`
  if present) under an `action_source:` block,
  `class_path: physicalai.runtime.PolicySource`. There is **no** committed
  `teleop_runtime.yaml`, so no teleop config to migrate.
- **Python — committed examples using `PolicyRuntime`:**

  - `sync_inference.py` (`:38` import, `:126` construct, `:130`
    `action_queue=ChunkedActionQueue()`, `:134` `task=args.task`)
  - `async_inference.py` (`:50` import + `action_queue=` kwarg)
  - `rtc_inference.py` (`:44`)
  - `run_from_config.py` (`:28`, `:58`)

  Replace `PolicyRuntime(robot=..., model=..., execution=..., action_queue=..., task=...)`
  with `RobotRuntime(robot=..., action_source=PolicySource(model=..., execution=...,
action_queue=..., task=...), fps=..., cameras=..., callbacks=...)`. **Move the
  `action_queue` and `task` kwargs into `PolicySource(...)`** — they are real
  call-site kwargs and `PolicySource.__init__(model, execution, action_queue=None,
task=None)` takes them. `run()` returns `int` — drop any `stats.total_pops` /
  `RunStats` usage at the call site. (`demo_loop.py` is **not** in the committed
  tree — ignore earlier mentions of it.)

- **Notebook** `examples/tutorials/collect_train_deploy.ipynb`: same substitution in
  prose + code cells.

## Phase 8 — Tests + docs

**Tests — `tests/unit/runtime/`**

- `test_runtime.py`: construct `RobotRuntime(action_source=...)`; assert `run()` is
  `int`; drop `RunStats` assertions; update fakes to the 3-method `ActionSource`
  (two-param `update`).
- `test_fault_tolerance.py`: same resilient-IO logic, new call sites
  (`_read_observation`, no warmup retry). Warmup-retry tests are deleted (behavior
  moved into the action source).
- `conftest.py`: fake controller → fake `ActionSource` (two-param `update`, no
  capability protocols).
- **`queue_remaining` / `.tick` are guaranteed constructor breaks** (dropping the
  `TickEvent` field + the `Tick` type). Update every site, not "if touched":
  - `test_telemetry.py` — `:67, :95, :132, :234, :261`
  - `test_rerun_callback.py` — `:72, :219` (+ `event.robot_state` /
    `event.camera_frames`; queue depth now via `on_metrics`)
  - `test_observer.py` — `:27`
- **New tests:** two-param `update` wiring; no-drain shutdown (queue discarded, no
  extra sends); `MetricsEvent` emission + `on_metrics` dispatch; `run()` returns
  step count.

**Tests — `tests/unit/cli/test_cli.py` (guards the jsonargparse wiring you change)**

- `:26` `from physicalai.runtime import PolicyRuntime, RunStats` → import
  `RobotRuntime` (and `PolicySource`); `RunStats` is gone.
- Flat-schema assertions move to the nested `action_source` schema:
  `cfg.runtime.robot.class_path` (`:191, :216`) stays top-level; `execution` moves to
  `cfg.runtime.action_source.init_args.execution.class_path` (`:192`). The inline YAML
  fixtures (`:200-208, :224-232`) and the `--runtime.execution=` CLI override (`:84`)
  rewrap under `action_source`.
- `_fake_runtime` / `MagicMock(spec=PolicyRuntime)` (`:243-244`) → `spec=RobotRuntime`;
  `run()` returns `int`, so the `RunStats(...)` fixtures (`:254, :271`) become plain
  step counts and the summary-log assertions change to `"%d steps"`.

**Docs**

- `docs/reference/runtime-api.md`, `docs/reference/config-schema.md`,
  `docs/reference/cli.md`: `PolicyRuntime` → `RobotRuntime` + `PolicySource`; new
  config schema; `run() -> int`.
- `docs/explanation/runtime.md`, `cli.md`, `configuration.md`, `inference.md`,
  `architecture.md`; `docs/getting-started/quickstart.md`, `run-a-policy.md`;
  `docs/index.md`; `README.md`: same rename + schema updates.
- Mark `docs/design/components/runtime-system/runtime-architecture.md` fully
  superseded (already flagged historical).

## Deletions (grep to confirm zero remaining references)

`Tick` / `tick.py`; `Controller`, `SupportsBus`, `SupportsStats`, `SupportsDrain`,
`SupportsHoldInfo`; `PolicyRuntime`; `RunStats`; `on_hold` / `invoke_on_hold`;
`before_send_action`; `queue_remaining` on `TickEvent`; `drain` / warmup-retry in
the runtime; **both** config-schema peeks — `_config_has_class_path` (`runtime.py`)
and `_peek_config_uses_general_schema` / `_yaml_has_runtime_class_path` /
`_build_legacy_parser` / `_build_general_parser` (`cli/run.py`).

## Open / verify items

- **jsonargparse + `Protocol` arg** (Phase 5) — confirm `action_source.class_path`
  resolves. Strong precedent (`robot:`).
- **Optional `connect(bus)` shrink** — moving `session_id` onto the bus (stamped per
  `run()`) removes it from `connect()` and every execution. Not required; do only if
  you want the smaller surface. Out of scope unless requested.
- **`__init__.py` exports** — drop `Controller`, `PolicyRuntime`, `RunStats`, `Tick`;
  add `ActionSource`, `PolicySource`, `TeleopSource`, `MetricsEvent`; rename in
  `__all__` and the module docstring examples.
