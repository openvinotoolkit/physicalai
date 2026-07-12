# Plan: Zenoh Transport Layer for Shared Robots

Source of truth for decisions: `docs/development/robot-zenoh-transport-design.md` (D1–D20).
Implements `physicalai.robot.transport` — one owner process holds the hardware, N subscribers
read state (pull) / send actions (fire-and-forget) over Zenoh. Mirrors the capture-transport
structure (`_spec`/`_publisher`/`_publisher_worker`/`_shared_camera`), different transport.

## Mirror map (capture → robot)

- `capture/transport/_spec.py` (CameraSpec) → `robot/transport/_spec.py` (RobotSpec: robot_type + serializable kwargs)
- `capture/transport/_publisher.py` (CameraPublisher.start, READY/ERROR parse) → `robot/transport/_owner.py` (parent-side spawn + handshake)
- `capture/transport/_publisher_worker.py` (signal_ready/suppress_stdout, worker loop) → `robot/transport/_owner_worker.py` (lock file, driver construct, write-first loop, meta queryable, idle shutdown)
- `capture/transport/_shared_camera.py` (probe→spawn→race-retry→subscribe) → `robot/transport/_shared_robot.py` (SharedRobot)
- `runtime/_telemetry.py` `_encode_numpy`/`_decode_numpy` → promote to shared codec
- `capture/errors.py` (CaptureError hierarchy) → new `robot/errors.py`

## Phases / Steps

### Phase 0 — Scaffolding (parallelizable)

1. `robot/errors.py`: `RobotError(RuntimeError)` base, `RobotIdConflict(RobotError)`, `RobotNotConnectedError`, `RobotTransportError`. Mirror `capture/errors.py`. (D19)
2. Shared msgpack+numpy codec: promote `_encode_numpy`/`_decode_numpy` + `_pack` default hook into a shared module (proposed `src/physicalai/_serialization.py`); refactor `runtime/_telemetry.py` to import it (no behavior change). (D9)
3. Add `msgpack` to `transport` extra in `pyproject.toml` (`transport = ["iceoryx2==0.9.0", "eclipse-zenoh==1.9.0", "msgpack"]`).
4. `robot/transport/__init__.py` exports (currently empty file exists).

### Phase 1 — Wire format + robot_id (depends on P0.2)

5. `robot/transport/_codec.py` (or in shared module): `encode_state`, `decode_state`, `encode_action`, `decode_action`, `encode_meta`, `decode_meta` using the record schemas in design §6. State ships `joint_positions`+`state`+`timestamp`+`sensor_data`, images excluded. (D10, D11)
6. `robot/transport/_ids.py`: `derive_robot_id(robot_type, kwargs)` → `physicalai/robot/{robot_type}/{host}/{device_id}`; symlink-resolve serial (`Path(...).resolve().name`), Trossen uses IP; role excluded; explicit override passthrough. Key builders for `/state|/action|/meta`. (D12)

### Phase 2 — Owner (depends on P1)

7. `robot/transport/_spec.py`: `RobotSpec` (robot_type + serializable kwargs); `build_driver(spec)` constructs SO101/WidowXAI (calibration as path, role as str). (D15)
8. `robot/transport/_lock.py`: user-scoped lock file at `~/.cache/physicalai/robot-locks/{device_id}.lock`; acquire/release; both backends. (D14)
9. `robot/transport/_owner_worker.py`: subprocess entrypoint — acquire lock → build+connect driver → declare `/state` pub, `/action` sub (RingChannel(1)), `/meta` queryable, **QoS: reliability=BEST_EFFORT, congestion_control=DROP, express=True; session peer mode** (D20) → `signal_ready()` (READY) or `ERROR:{json}` on failure → **write-first loop** (try_recv+apply action → read → publish state) at per-robot configurable rate → idle shutdown via `matching_status()`+`idle_timeout` → `driver.disconnect()` on exit. (D5, D6, D7, D8, D13, D17, D20)
   - NOTE: `robot/transport/__init__.py` already exists (empty) — populate, don't create the dir. QoS (D20) MUST be applied at every `declare_publisher`/`declare_subscriber`/session open in Phase 2/3.
10. `robot/transport/_owner.py`: parent-side spawn (`subprocess.Popen`, stdout PIPE), `_read_stdout_line` with generous timeout, parse READY/ERROR, raise `RobotTransportError` on failure. Mirror `CameraPublisher.start()`.

### Phase 3 — SharedRobot subscriber (depends on P2)

11. `robot/transport/_shared_robot.py`: `SharedRobot` implementing `Robot` protocol.
    - `connect()`: query `/meta` → if owner exists, validate meta vs own kwargs (mismatch → `RobotIdConflict`) and attach; else spawn owner (Phase 2) → on lost race, re-probe `/meta` with retry and attach.
    - `get_observation()`: `try_recv()` on RingChannel(1) `/state`, decode, cache `_latest`; return shipped `state`/`joint_positions`/`sensor_data`. (D4, D10)
    - `send_action(action, goal_time)`: `pub.put(encode_action(...))` fire-and-forget. (D7)
    - `disconnect()`: close own session only. (D16)
    - `is_connected()`, `joint_names` (from `/meta`).
12. `/meta` query answering lives in owner (Phase 2); subscriber discovery via `physicalai/robot/*/meta` wildcard helper (`discover_robots()`).

### Phase 4 — Integration

13. Export `SharedRobot` from `robot/__init__.py`; optional `from_owner`-style classmethod.
14. Confirm `robot/connect.py` context manager works unchanged with `SharedRobot` (structural Robot).

### Phase 5 — Tests (depends on P3; some parallel)

15. `tests/unit/robot/transport/test_codec.py`: round-trip dtype/shape exactness (float32 stays float32); action/meta records.
16. `test_ids.py`: derivation determinism, serial symlink resolution, override, role-excluded.
17. `test_lock.py`: single acquire wins; second blocks/fails; user-scoped path.
18. `test_shared_robot.py`: with a **fake in-process driver** (mirror capture fake-device tests) — spawn-or-attach, get_observation pull, send_action, RobotIdConflict on mismatched kwargs, disconnect detach.
19. `test_owner_handshake.py`: READY path, ERROR path, timeout fallback to re-probe.
20. Optional integration test: real zenoh session, two SharedRobot attach to one fake owner, latest-wins + Ring(1) behavior.
21. `test_latency.py`: measure **p99 action-latency jitter** at target loop rate under D20 QoS (best-effort/drop/express) — batching regressions are invisible to functional tests. (D20)

### Phase 6 — Docs

22. Link design doc + add how-to under `docs/how-to/runtime/` (shared robot over zenoh); note trusted-LAN assumption (D18). Add security.md rule on network transport trust boundary.

## Relevant files

- `src/physicalai/robot/interface.py` — `Robot`/`RobotObservation` protocol to satisfy.
- `src/physicalai/robot/so101/so101.py`, `robot/trossen/widowxai.py` — drivers; `.state` computation; `connect()` (~2s homing on WidowXAI L155).
- `src/physicalai/capture/transport/_publisher.py`, `_publisher_worker.py`, `_shared_camera.py`, `_spec.py` — patterns to mirror (READY/ERROR, suppress_stdout, probe/spawn/race-retry).
- `src/physicalai/runtime/_telemetry.py` — `_encode_numpy`/`_decode_numpy` to promote.
- `src/physicalai/capture/errors.py` — error-hierarchy pattern.
- `src/physicalai/robot/connect.py` — context manager (should work unchanged).
- `pyproject.toml` — `transport` extra (add msgpack).
- `.venv/.../zenoh/__init__.pyi`, `handlers.pyi` — RingChannel, try_recv, matching_status, Liveliness API reference.

## Verification

1. `uv run pytest tests/unit/robot` — all new + existing robot tests pass.
2. Manual: `examples/` script — owner auto-spawns on first `SharedRobot("so101", port=...).connect()`, second process attaches; kill subscriber → owner idle-exits after timeout and homes (verify `driver.disconnect()` called).
3. Round-trip: publish float32 state, subscriber `get_observation().state.dtype == float32` and shape matches (WidowXAI 14, SO101 6).
4. Conflict: two `SharedRobot` same override id + different port → `RobotIdConflict` raised.
5. GIL: subscriber runs a long C call, then `get_observation()` returns freshest buffered state (Ring(1)), not a backlog.
6. `prek run --all-files` + `pyrefly check` clean.
7. **p99 action-latency jitter** measured at target rate under D20 QoS is within budget (batching not silently degrading latency). (D20)

## Decisions (all locked in design doc)

- Structural Robot protocol satisfaction; Zenoh transport; 3 keys; RingChannel(1) pull; single write-first owner loop; per-robot fixed rate; latest-wins actions; hold-on-no-action; msgpack+numpy records; ship computed `.state`; images excluded; robot_id derived-default+override; matching_status shutdown + driver.disconnect; lock-file arbiter; READY/ERROR handshake; documented trusted-LAN boundary; robot/errors.py; transport QoS best-effort/drop/express + peer mode (D20).

## Further considerations

1. Shared codec location — `src/physicalai/_serialization.py` (new top-level) vs `robot/transport/_codec.py` with telemetry importing across packages. Recommend new top-level shared module. (A/B)
2. Owner subprocess entrypoint — module `python -m physicalai.robot.transport._owner_worker` vs a hidden CLI subcommand. Recommend `-m` module (mirrors capture).
3. `host` component — hostname (readable) vs `/etc/machine-id` (guaranteed unique). Recommend hostname default, overridable. (design open item)
4. Start scope — SO-101 only first (simplest params), WidowXAI second. Recommend SO-101 reference impl, defer Trossen wiring to a follow-up.
