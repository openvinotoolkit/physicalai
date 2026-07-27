# Captured component configuration (design note)

Status: **accepted — implement per rollout**. Follow-up to Studio SharedRobot
integration
([physical-ai-studio#818](https://github.com/open-edge-platform/physical-ai-studio/pull/818));
out of scope for that PR, now ready for Runtime implementation.

## Decision

Add one opt-in, transport-agnostic decorator for components whose constructor
configuration must be exportable so a fresh instance can be created later.

```text
Robot / Camera / ActionSource = runtime behavior (unchanged)
@export_config / @export_config(class_path=..., scalar_var_kwargs=...) = opt into ComponentConfig export
ComponentConfig               = plain class_path + init_args data
```

Use the same contract for robots, cameras, action sources, runtimes,
callbacks, and nested components. Keep process names, rates, timeouts, and
other transport settings in their existing transport envelopes.

Mental model:

```text
@export_config / @export_config(class_path="...", scalar_var_kwargs=...)  → whole component → {class_path, init_args}
to_config_value()                                  → domain arg     → plain JSON inside init_args
(nested @export_config)                            → nested component → nested {class_path, init_args}
```

Studio needs only `is_config_exportable(obj)` and `to_config(obj)`, then
domain helpers such as `SharedRobot.from_config(...)`. Studio does not
implement or require domain/component escape hatches.

The contract closes both directions:

```text
live component -> to_config(value) -> JSON/YAML -> instantiate(config) -> new component
```

It uses the same `class_path` + `init_args` vocabulary already consumed by
jsonargparse runtime configs.

## Problem

jsonargparse can instantiate a component tree from typed constructor
signatures, but it cannot recover constructor inputs from an arbitrary live
object. `SharedRobot` has the same gap: owner spawn needs an importable class
and JSON-safe constructor arguments, not a live driver with open hardware
resources.

Studio must not special-case SO101, WidowXAI, camera, runtime, or action-source
serializers. Plugins must not import `SharedRobot` or `SharedCamera`. Adding
serialization methods to `Robot`, `Camera`, or `ActionSource` would mix
construction with runtime behavior.

## Goals

1. Export an opted-in live component as JSON-safe `class_path` +
   `init_args` data.
2. Instantiate that data as a new component without invoking lifecycle methods
   beyond its constructor.
3. Recursively support nested components and collections of components.
4. Emit nested `class_path` + `init_args` fragments that jsonargparse accepts
   without translation. A root config may sit under a CLI wrapper key such as
   `runtime:`, or be a bare `RobotRuntime` export that `physicalai run --config`
   unwraps into `runtime:`.
5. Keep hardware and action-source protocols unchanged.
6. Let third-party plugins opt in without depending on transports.
7. Fail at serialization or validation time when replay is not possible.

## Non-goals

- Snapshot live state, open handles, queues, threads, or connections.
- Reflect over arbitrary objects and guess how to reconstruct them.
- Serialize lambdas, closures, or arbitrary callables.
- Make untrusted `class_path` imports safe.
- Replace transport configuration such as robot names or camera service names.
- Guarantee that mutation after construction appears in the config.

## Public API

Place this API in a neutral module such as `physicalai.config`,
not under robot, capture, runtime, or inference.

```python
from typing import Protocol, TypedDict

JsonScalar = None | bool | int | float | str
JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


ComponentConfig = TypedDict("ComponentConfig", {"class_path": str, "init_args": dict[str, JsonValue]})


class ConfigValue(Protocol):
    def to_config_value(self) -> JsonValue: ...


def export_config(
    cls: type | None = None,
    *,
    class_path: str | None = None,
    scalar_var_kwargs: bool = False,
): ...


def to_config(value: object) -> ComponentConfig: ...


def instantiate(config: ComponentConfig) -> object: ...


def is_config_exportable(value: object) -> bool: ...
```

`scalar_var_kwargs` defaults to `False`. Pass
`@export_config(..., scalar_var_kwargs=True)` when flattened `**kwargs` must
export as JSON scalars only (`None` / `bool` / `int` / `float` / `str`);
non-scalar var-keyword values then fail at `to_config`. Path-rooted
`InferenceModel` uses this option.

Normal use stays small:

```python
@export_config
class MyCamera:
    def __init__(self, device: str): ...


@export_config(class_path="physicalai.robot.SO101")
class SO101:
    def __init__(self, port: str, calibration: dict | str): ...


@export_config(
    class_path="physicalai.inference.InferenceModel",
    scalar_var_kwargs=True,
)
class InferenceModel:
    def __init__(self, export_dir, *, backend=None, device=None, **kwargs): ...


robot = SO101("/dev/ttyACM0", "./calibration.json")
config = to_config(robot)
restored = instantiate(config)
```

The decorator can also inject `robot.to_config()` as a runtime convenience.
Library code and documentation always use module-level `to_config(robot)`
because static type checkers cannot express that a class decorator adds an
instance method without changing the decorated class's declared type. This
avoids introducing a public protocol only to make the convenience method
type-check. Structural `Protocol` checks ignore the extra method, so the
injection does not break robot/camera/action-source conformance.

`ComponentConfig` is a typed description of the existing jsonargparse wire
shape, not a stateful model with its own methods. Callers can pass ordinary
dictionaries loaded from JSON or YAML to `instantiate()`; it validates them
before importing code. `instantiate()` is the one shared, explicit trusted-code
boundary: it imports `class_path`, recursively instantiates nested configs,
then calls the class with decoded keyword arguments.

The implementation rejects local classes (`<locals>`), non-class import
targets, non-string mapping keys, malformed nested configs, and values outside
the supported JSON model.

Do not add a public `Constructible` or `Replayable` protocol. Runtime and
Studio code that needs to test exportability uses `is_config_exportable(value)`.
A value is exportable if and only if it carries the private `@export_config`
decorator marker. `to_config(value)` uses the same predicate. This keeps the
plugin-facing contract to one decorator; Studio's share path keys off that
single exportability check.

### Naming

Use `export_config` because it names the public capability: the class can export
a `ComponentConfig` via `to_config`. That aligns with `is_config_exportable`
and does not claim to serialize live object state. The decorator works by
remembering supplied `__init__` arguments; that mechanism is an implementation
detail documented here, not part of the public name.

Rejected names:

| Name                              | Problem                                                               |
| --------------------------------- | --------------------------------------------------------------------- |
| `capture_init`                    | Collides with the `physicalai.capture` camera package                 |
| `record_init`                     | Will collide with planned runtime recording callbacks                 |
| `serializable`                    | Implies complete object-state serialization                           |
| `configurable`                    | Usually means an object accepts configuration, not that it exports it |
| `replayable`                      | Does not say what is replayed and can suggest runtime/action replay   |
| `reconstructible`                 | Accurate but cumbersome and still hides the config-export contract    |
| `remember_init` / `snapshot_init` | Mechanism-only; weaker fit next to `to_config`                        |
| `config_exportable`               | Accurate but clumsy as a class decorator                              |
| `export_init`                     | Precise about ctor args, but rhymes less with `is_config_exportable`  |
| `has_captured_init`               | Mechanism-only; use `is_config_exportable`                            |

Use `to_config(value)` for export because the result is directly consumable
jsonargparse component configuration. Use `instantiate()` for the trusted
inverse because it creates a fresh instance; avoid `deserialize()`, which
suggests restoring prior object state.

### Relationship to inference `ComponentSpec`

Inference factory unification is explicitly out of scope for v1.
`ComponentSpec` is a manifest compatibility model with registry aliases, flat
parameters, artifact handling, and `extra="allow"` semantics. Tightening its
class-path mode or routing registry mode through `instantiate()` changes the
manifest schema and inference critical path; the robot/camera construction use
case does not require either change.

For v1:

- `ComponentConfig` owns strict `class_path` + `init_args` validation and
  bounded recursive instantiation in `physicalai.config`.
- `ComponentSpec`, `ComponentSpec.from_class()`, `_MAX_COMPONENT_DEPTH`, and
  `instantiate_component()` retain their current behavior.
- Rollout step 1 consolidates the duplicated inference and robot dotted-path
  import helpers into `physicalai.config` in a behavior-preserving refactor.
  Transports call `instantiate()` and add protocol checks; they do not add
  another importer.

A follow-up design can audit exported manifest fixtures, then decide whether
`ComponentSpec` class-path mode should delegate to `ComponentConfig`. That
work must address two known differences: `ComponentSpec.from_class()` applies
defaults while `@export_config` omits unsupplied defaults, and class-path extra
fields are currently accepted and ignored. Registry mode remains an inference
adapter even if class-path construction is later shared.

Use `_MAX_CONFIG_DEPTH = 10` for the new config engine. Count traversal through
lists and mappings as well as nested component configs, and track container
identities to reject cycles before JSON serialization. Do not apply this
stricter counting rule to existing inference manifests in v1.

## Exporting constructor configuration

`@export_config` is the opt-in contract. Constructor signature changes flow
into exported configuration automatically:

```python
@export_config
class PolicySource(ActionSource):
    def __init__(self, model, execution=None, action_queue=None, *, task=None):
        self._execution = execution or SyncExecution()
        self._action_queue = action_queue or ChunkedActionQueue(...)
```

The wrapper uses `inspect.signature()` and `Signature.bind()` before invoking
the constructor. It converts positional arguments to their parameter names,
removes `self`, and flattens a bound `**kwargs` mapping into `init_args`.
When `scalar_var_kwargs=True`, those flattened var-keyword values are sealed
to JSON scalars: non-scalars fail later at `to_config` instead of normalizing
as nested JSON (requires a `**kwargs` parameter on the constructor).
Positional-only parameters and `*args` are not replayable through
`class_path` + keyword `init_args`; reject decorated constructors that declare
them.

Do not call `BoundArguments.apply_defaults()`. Capture only arguments the
caller supplied. Omitted arguments remain omitted so reconstruction invokes
the current constructor defaults. An explicitly supplied `None` remains in
the spec. This produces compact configs and avoids freezing default component
objects selected internally by the current release.

The wrapper must preserve `__wrapped__` and the original signature through
`functools.wraps()`. jsonargparse and documentation tools must continue to see
the real constructor signature.

Call the constructor before committing the captured arguments. A failed
constructor must not leave a usable construction record. Do not normalize or
reject captured values during `__init__`; unsupported values fail later when
`to_config(value)` is requested, so opting in does not change ordinary
construction behavior.

Snapshot built-in mutable containers recursively when capturing them. Keep
nested decorated components and explicit domain values by reference so their
own conversion remains authoritative. This prevents later mutation of a
caller-owned list or dictionary from silently changing the recipe.

### Inheritance rule

Decorate every concrete class that should export constructor configuration. Do
not decorate a shared base solely to make all subclasses exportable: a base
signature cannot capture arguments owned by a subclass.

The decorator tracks construction depth on the instance. If decorated
constructors call other decorated constructors through `super()`, only the
outermost successful call commits its bound arguments. This prevents a camera
base constructor from overwriting the complete concrete-camera recipe with
only `color_mode`.

A concrete subclass that overrides `__init__` must apply `@export_config`
again. Tests and contributor documentation must state this explicitly.

There is no merge or last-write behavior. The outermost decorated constructor
owns the complete `init_args`. `to_config(value)` verifies that the
most-derived class's effective `__init__` carries the decorator marker; an
overriding, undecorated `__init__` fails loudly instead of emitting a partial
recipe. A subclass that inherits a decorated constructor unchanged remains valid
because its effective constructor has the marker and the inherited signature
is complete.

Select `class_path` from the most-derived `type(self)`, never from the class
that defined an inner decorated constructor. Pass
`@export_config(class_path="physicalai.robot.SO101")` when the public import
path differs from the defining module; otherwise export uses
`type(self).__module__ + "." + type(self).__qualname__`. The `class_path=`
override applies only to the concrete class that was decorated (owns the
wrapped `__init__` in its class dict). A subclass that inherits a decorated
constructor unchanged does **not** inherit that override — it exports its own
`__module__.__qualname__` unless it re-decorates with its own `class_path=`.
Pass `scalar_var_kwargs=True` on the same decorator when flattened `**kwargs`
must be JSON scalars at export (default `False`; used by path-rooted
`InferenceModel`). Before emitting the spec, resolve the selected path and
verify that it identifies exactly `type(self)`.

First-party public robot and camera classes must pass their stable public
re-export path (for example, `physicalai.robot.SO101`) instead of emitting an
internal defining-module path. Plugins must do the same when they promise a
stable public import surface.

`to_config(value)` describes the object **as constructed**, not its current
mutable state.

## Serialization and decoding rules

Normalization is recursive and produces JSON-safe values.

| Constructor value                            | Stored value                  | Value passed on replay       |
| -------------------------------------------- | ----------------------------- | ---------------------------- |
| `str`, `int`, finite `float`, `bool`, `None` | as-is                         | as-is                        |
| non-finite `float` (`NaN`, `±Inf`)           | error                         | not applicable               |
| `Path`                                       | string as given (`str(path)`) | string                       |
| `Enum`                                       | JSON-safe `.value`            | enum value representation    |
| decorated (`@export_config`) component       | `{class_path, init_args}`     | instantiated component       |
| mapping with string keys                     | normalized mapping            | decoded mapping              |
| list or tuple                                | normalized list               | list                         |
| domain value with `to_config_value()`        | re-normalized codec output    | constructor-compatible value |
| any other object                             | error                         | not applicable               |

`Path` values serialize as `str(path)` **without** `resolve()`. Relative paths
stay relative; absolute paths stay absolute. Prefer relative paths for
project- or folder-local configs so an exported directory of config +
artifacts remains portable. Resolution happens at `instantiate` /
constructor open against the process cwd (see Transport integration).

Reject non-finite floats during normalization. Python's default `json` encoder
permits `NaN` / `Infinity`, which are not portable across strict JSON
consumers.

For v1, domain codecs remain minimal. Non-component domain values implement
`to_config_value()` (`ConfigValue` Protocol). The method must return a **new**
JSON-compatible value that is then re-normalized (non-finite floats, reserved
`class_path` maps, depth and cycle checks, nested codecs). Absence of the
method means _no codec_ for that value. Returning `None` from the method is a
real JSON `null` payload. Mutually recursive domain codecs fail with
`ComponentConfigError` (not `RecursionError`). `SO101Calibration.to_config_value()`
returns `to_dict()` because the SO101 constructor already accepts a dictionary.
Do not auto-call arbitrary `to_dict()`. Do not add a global arbitrary-object
codec registry until a second concrete domain type requires one.

Constructors participating in this contract must accept the normalized JSON
representation of their arguments. For example, constructors typed with an
enum must also accept its string value or normalize it internally. This keeps
direct `instantiate()` and jsonargparse reconstruction
equivalent.

This requirement is enforced at the opt-in boundary. Every first-party class
gains a test that serializes through `json.dumps()` / `json.loads()`, invokes
its constructor through `instantiate()`, and verifies normalized
re-serialization. A class does not ship with `@export_config` until that test
passes. Provide small constructor-side coercion helpers for repeated types such
as `StrEnum`; do not build a general annotation-driven coercion engine. Plugins
have the same mandatory JSON-boundary round-trip test.

Any dictionary containing `class_path` is reserved as a nested component
config. Its only allowed keys are `class_path` and `init_args`; omitted
`init_args` means an empty mapping. Extra keys or invalid values are malformed
configs, not ordinary dictionaries. A caller that needs `class_path` as a normal
data key must encode that mapping through `to_config_value()` (or another
domain wrapper). Error messages and the plugin contract name this escape.
This rule removes ambiguity and fails closed during generic recursive
deserialization. Do not add a sentinel marker unless a second concrete domain
collision demonstrates the need.

Errors identify the full argument path:

```text
physicalai.runtime.TeleopSource.init_args.to_action:
cannot encode function '<lambda>'; omit it or use a supported component value
```

## Round-trip contract

For every opted-in component `value`, tests establish:

```python
config = to_config(value)
wire = json.loads(json.dumps(config))
restored = instantiate(wire)
```

The guarantee is construction equivalence, not object equality:

- `type(restored)` matches the represented class.
- `instantiate()` invokes constructors but does not call lifecycle methods such
  as `connect()`, `start()`, or `run()`.
- Constructors may still read files, allocate memory, initialize SDKs, or load
  models according to their existing contract.
- Calling `to_config(restored)` produces the same normalized config.
- Connecting it addresses the same configured hardware or service.
- Runtime-selected defaults may change between package versions when the
  original constructor argument was omitted or `None`.

Round-trip tests use the public import path emitted in `class_path`, not only
the defining module path. Public paths keep serialized configs stable across
internal module moves.

### External references and portability

A component config is a replay recipe, not necessarily a self-contained
artifact. Classify emitted values in two practical profiles:

- **Local replay** permits paths and other external references. Relative
  paths are preferred for project- or folder-local configs. Owner/publisher
  subprocess IPC inherits the parent cwd at `Popen` (see Transport
  integration); that is the v1 resolution root.
- **Self-contained replay** requires domain values to be inline and rejects
  unresolved external dependencies. For example, SO101 calibration must be an
  inline dictionary rather than a path.

The full runtime example below is a local-replay config because both calibration
and `export_dir` reference the filesystem. Relative paths stay meaningful when
the working directory (or a future exported folder layout) is the project root.
A future config-bundling API can copy referenced artifacts and keep or rewrite
relative paths; that is separate from construction replay.

`to_config(value)` emits local replay by default. A later
`to_config(value, profile="self_contained")` API should be added only when a
concrete portable-export workflow owns artifact bundling and validation.

`ComponentConfig` has no version field in v1. It is a nested jsonargparse shape,
not an owned persistent document format. IPC envelopes version their own wire
formats as described below. Before advertising generated runtime YAML as a
stable persisted artifact, define a versioned root document that contains the
runtime component config; do not add a version field to every nested component.

## Full runtime example

A `RobotRuntime` config recursively contains all participating components:

```yaml
class_path: physicalai.runtime.RobotRuntime
init_args:
  robot:
    class_path: physicalai.robot.SO101
    init_args:
      port: /dev/ttyACM0
      calibration: ./calibration.json
  action_source:
    class_path: physicalai.runtime.PolicySource
    init_args:
      model:
        class_path: physicalai.inference.InferenceModel
        init_args:
          export_dir: ./exports/act_policy
  fps: 30
  cameras:
    wrist:
      class_path: physicalai.capture.UVCCamera
      init_args:
        device: /dev/video0
        width: 640
        height: 480
        fps: 30
```

This example is a bare `RobotRuntime` export. `physicalai run --config` accepts
it directly — it unwraps the top-level `RobotRuntime` `class_path` into the
parser's `runtime:` section — as well as a `runtime:`-rooted CLI document.
Trusted local code may pass the bare fragment to `instantiate()` or
`RobotRuntime.from_config()`. Nested `class_path` + `init_args` fragments are
native jsonargparse and need no translation. So `to_config(runtime)` →
`save_yaml(...)` → `physicalai run --config` round-trips with one canonical
shape; no separate export API (`save_config` / `to_run_document`) is needed.

## Component-specific decisions

### Robots

SO101 and WidowXAI opt in with `@export_config(class_path="physicalai.robot.…")`
so export emits the public re-export path. SO101 calibration is stored as a
path when constructed from a path and as a normalized dictionary when
constructed from an object (`SO101Calibration.to_config_value()` →
`to_dict()` — a domain value, not a nested component).

Private arguments are not excluded by naming convention. If replay needs an
argument, capture it. `SO101.uncalibrated()` calls the constructor with
`allow_uncalibrated=True` — a **public** keyword-only parameter (default
`False`) that `@export_config` captures. `calibration` defaults to `None`, so
the uncalibrated recipe (`calibration: null`, `allow_uncalibrated: true`) loads
through jsonargparse and `physicalai run --config` (a required, unset
`calibration` would reject `null`). The constructor still raises unless
`allow_uncalibrated=True` accompanies a `None` calibration, so uncalibrated
bringup stays explicit and cannot be reached by accident.

Bimanual robots need no special serialization once each arm uses
`@export_config`: `left` and `right` are nested `@export_config` components
under `BimanualWidowXAI` (contrast with domain `to_config_value` for
calibration). v1 `SharedRobot` spawn treats the composite as **one owner**.
Per-arm `SharedRobot` wrapping of nested arms is out of scope.

### Cameras

Camera implementations opt in using public class paths. This supports
third-party cameras without extending the `create_camera()` registry for
shared spawn.

The private publisher envelope contains `camera: ComponentConfig` plus
transport fields including `service_name`; `build()` delegates to
`instantiate()` and verifies the result satisfies `Camera`. Writers and
readers speak only this shape. Flat `camera_type` + `camera_kwargs` are
rejected before import. Built-in names are public class paths in the new
shape.

The publisher's runtime control channel is a separate, untrusted peer
boundary. A `RECONFIGURE` request carries only allowlisted scalar capture
settings (`width`, `height`, `fps`). The publisher patches those values onto
the trusted startup recipe; a subscriber can never replace `class_path`,
device identity, backend, or other constructor arguments.

#### SharedCamera service naming

Transport owns naming; `ComponentConfig` never embeds `service_name`.

Spawn has no `camera_type` on the construction config. Rules:

- **Shareable built-in spawn:** a private map from camera `class_path` to the
  legacy `CameraType` token for backends with a real driver under `cameras/`
  (`uvc`, `realsense`, `basler` only). That map lives in capture transport
  (`builtin.py`), not in `physicalai.config`. Derive `service_name` with the
  existing scheme using that token plus device id from `init_args`
  (`serial_number` else `device`, including `/dev/` symlink resolve). This
  preserves existing attacher discovery.
- **The map is a static table and must not import a driver module.** Since
  string `class_path` values are stored as written, the table lists every
  accepted spelling per backend — the public re-export (canonical, index 0),
  the sub-package re-export, and the defining module — so one physical device
  derives one service name however the config names its driver. A test
  asserts the table still matches each driver's `@export_config(class_path=)`
  whenever the optional camera extra is installed.
- **Stub / non-shareable registry types and third-party spawn:** `ip` and
  `genicam` are intentionally absent from the map (same rule as third-party).
  Require an explicit `service_name` on `SharedCamera` / the publisher
  envelope; fail before publisher start if missing. Do not hash `class_path`
  or `init_args` into a name (unstable across arg ordering and omitted
  defaults).
- **Attach-only:** unchanged — `service_name` required; no construction config
  needed.
- The publisher payload carries `service_name` **alongside**
  `camera: ComponentConfig`, never inside `init_args`.

### Action sources

v1 first-party graph that must round-trip under `PolicySource` (anything else
fails at `to_config`):

| Class                | Notes                                                                                                                      |
| -------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| `PolicySource`       | Captures `model`, optional `execution`, `action_queue`, `task`                                                             |
| `InferenceModel`     | Path-rooted (see below)                                                                                                    |
| `SyncExecution`      |                                                                                                                            |
| `AsyncExecution`     |                                                                                                                            |
| `RTCExecution`       | JSON-safe / scalar constructor args only; `latency_tracker` and live `postprocessors` must be omitted or `to_config` fails |
| `ChunkedActionQueue` | Includes nested `smoother` when supplied                                                                                   |
| `RTCActionQueue`     |                                                                                                                            |
| `LerpSmoother`       | Required when an explicit `ChunkedActionQueue(smoother=...)` is captured                                                   |
| `ReplaceSmoother`    | Same                                                                                                                       |

Omitted `execution` / `action_queue` stay omitted so reconstruction uses
current constructor defaults. `PolicySource` itself defaults to
`SyncExecution()` and `ChunkedActionQueue(smoother=LerpSmoother(...))`.
Bare `ChunkedActionQueue()` (smoother omitted) restores `ReplaceSmoother()` —
that is the queue's own default, not the PolicySource default. Explicit nested
values must themselves be exportable.

`TeleopSource.leader` can be a nested robot config. Its optional `to_action`
callable is replayable only when omitted in v1. Module-level callable support
can be added later with a distinct callable-reference type; do not treat
arbitrary dotted paths as both classes and functions in `ComponentConfig`.

### Inference model

`InferenceModel.__init__` eagerly reads the manifest, creates an adapter, and
loads model artifacts. In v1 it is a path-rooted component: capture
`export_dir`, `policy_name`, `backend`, `device`, and JSON-safe scalar adapter
kwargs. Omitted runner, processor, and callback overrides remain omitted and
are reconstructed from the exported package.

Non-scalar or live override arguments among captured kwargs (including
non-scalar `adapter_kwargs`, and live `runner` / `preprocessors` /
`postprocessors` / `callbacks`) make `to_config` fail. Do not silently drop
them. Live overrides are in scope only when every nested value independently
exports via `@export_config` (or encodes via `to_config_value()` for domain
args). A full-component escape hatch without the decorator is deferred and not
part of the v1 contract. Instantiating an inference config can allocate
substantial resources; it is not analogous to constructing a disconnected
robot driver.

### Runtime

`RobotRuntime` captures its robot, action source, FPS, camera mapping, and
callbacks. A runtime config is therefore the root of a complete jsonargparse
workflow config. It never captures connection state, session IDs, callback
bus state, observations, or action queues created during a run.

v1 first-party callbacks that must round-trip when supplied (anything else
fails at `to_config`):

| Class                   | Notes                                                                                                                            |
| ----------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `ConsoleCallback`       | scalar args                                                                                                                      |
| `LowPassFilterCallback` | scalar args                                                                                                                      |
| `JsonlCallback`         | path arg; `Path` → `str(path)` as given; ctor opens the file immediately (eager I/O like `InferenceModel`); tests use temp paths |
| `RerunCallback`         | JSON-safe / scalar args only (`save_path`, `connect_addr`, mode, …)                                                              |
| `AsyncCallback`         | nested `inner` must itself be exportable; reject inners with action hooks per the existing `AsyncCallback` guard                 |

Omitted or empty `callbacks` stay omitted or empty. Observer helpers under
`runtime/observer/` that take `session_id` after construction are out of scope
for `RobotRuntime` capture in v1.

## Transport integration

Keep construction and transport ownership separate:

```python
SharedRobot(name="follower", robot={"class_path": ..., "init_args": {...}})
SharedRobot.from_config(to_config(robot), name="follower")   # trusted recipe (Studio API)
```

The `SharedRobot` constructor (`robot=`) and `SharedRobot.from_config()` are the
two entry points; there is no `from_robot()` sugar. `from_config()` is the
named, Studio-facing API for a trusted `ComponentConfig`. Both take a
`ComponentConfig` mapping only. A caller holding a live driver converts it
explicitly with `to_config(driver)`.

**Declared config args.** `robot` is declared via
`@export_config(config_args=("robot",))`. `instantiate()` treats a declared
config arg as **data**: for a nested `robot={class_path, init_args}` it passes
the mapping straight through instead of building the inner driver and handing
it to `SharedRobot`. This removes a build-then-unbuild round-trip, keeps the
subscriber process from importing the driver package at all, and removes the
need for a connected-state guard on the constructor. `SharedCamera` declares
`camera` the same way.

### Private startup envelopes (hard cutover)

**Status:** Accepted.

Private robot-owner and camera-publisher startup stdin is a same-package
ephemeral `Popen` handshake, not a persisted peer protocol. Changing
`RobotOwnerConfig` from `robot_class` + `robot_kwargs` to
`robot: ComponentConfig` also changes that private startup JSON. Hard-cutover
both envelopes in the same PR as the Shared\* spawn path: rewrite writers,
readers, and fixtures; do **not** add a `config_format` field, dual-read, or
shape-detection fallback.

```text
# Robot owner stdin — before
{name, robot_class, robot_kwargs, allow_remote, rate_hz, idle_timeout, …}

# Robot owner stdin — after
{name, robot: {class_path, init_args}, allow_remote, rate_hz, idle_timeout, …}

# Camera publisher stdin — before
{camera_type, camera_kwargs, service_name, idle_timeout, …}

# Camera publisher stdin — after
{camera: {class_path, init_args}, service_name, idle_timeout, …}
```

Writers and readers speak only the new shape. Validate each envelope
schema-positively (known transport keys only; required `robot` / `camera`
ComponentConfig). Unknown keys — including legacy flat `robot_class` /
`robot_kwargs` or `camera_type` / `camera_kwargs` — are rejected before
import or hardware access; no silent translation or denylist-only drops.

Do not reuse `capture.transport.PROTOCOL_VERSION` or
`ROBOT_TRANSPORT_PROTOCOL_VERSION` for these changes. Those version frame and
robot network payloads respectively, not the one-shot stdin construction
envelopes. Existing detached owners/publishers are discovered through their
current transport protocols and do not receive a new startup config.

This hard cutover applies only to trusted parent→child startup stdin. Camera
reconfigure messages arrive from subscribers and therefore never carry a
ComponentConfig or enter `instantiate()`.

### Paths and cwd for local replay / IPC

Relative paths are preferred for project- and folder-local configs (for
example an exported directory that contains `runtime.yaml`, calibration, and
model artifacts together). Do **not** rewrite paths to absolute in `to_config`
or in IPC writers — that fights portable folder export and is easy to get
wrong for non-path strings (URLs, roles, `host:port`).

v1 rules:

1. `to_config`: `Path` → `str(path)` as given; `str` paths unchanged. Relative
   stays relative; absolute stays absolute.
2. Resolution is against the **process cwd** at `instantiate` / constructor
   open (for example `Path(calibration).read_text()`).
3. Owner/publisher children are started with `Popen` and **no `cwd=`
   override**, so they inherit the parent cwd at spawn. That **is** the v1
   IPC contract for relative paths — the same behavior today’s transport
   already uses.
4. **Unsupported:** calling `os.chdir` (or otherwise changing the process cwd)
   between capturing/exporting a relative-path config and owner/publisher
   spawn when those configs still contain relatives.
5. Future: an explicit bundle/export-folder root may pin resolution without
   absolutizing paths; that is out of scope for v1.

`port` remains absolute (`/dev/…`); relative serial ports are unsupported.
Camera URL/stream fields are not filesystem paths. The shareable built-in
camera `class_path` → `CameraType` map (`uvc` / `realsense` / `basler`) lives
in **capture transport** (`builtin.py`), not in `physicalai.config`. Stub
types (`ip`, `genicam`) are not in that map.

### Public SharedRobot, CLI, and metadata

Public construction matches private stdin: `robot` / `--robot` only.
Flat `robot_class` / `robot_kwargs` on the public API and CLI are
removed; owner stdin uses an allowlist envelope (`name`, `robot`,
`allow_remote`, `rate_hz`, `idle_timeout`) so legacy flat keys fail as
unknown keys before import.

`SharedRobot` is `@export_config(class_path="physicalai.robot.SharedRobot")`
— a **construction recipe** only (`name`, nested `robot`
ComponentConfig, `allow_remote` / `rate_hz` / `idle_timeout` /
`connect_timeout`). `to_config` of a `RobotRuntime` (or other
container) that holds a `SharedRobot` is supported. Never capture Zenoh
session / owner / publisher / connected flags; a supplied private
`_session` live handle must fail at `to_config`, not export as an
object.

- **`SharedRobot` constructor:** accept `robot: ComponentConfig` to spawn,
  or `robot=None` for attach-only (prefer :meth:`SharedRobot.attach`).
  Prefer `SharedRobot.from_config(...)` when building from a trusted config.
  `robot` is declared via `@export_config(config_args=("robot",))`, so nested
  `instantiate()` passes the recipe through instead of constructing a driver.
- **Public path normalization:** a **class object** given for
  `robot.class_path` normalizes through the same public-path resolution used
  by `to_config` (decorator `class_path=` override, else
  `__module__.__qualname__`). A **string** path is trusted and stored exactly
  as written — it is deliberately not imported. Envelope construction happens
  in the subscriber process, which must not load the driver package and often
  cannot (the vendor SDK is only installed where the hardware lives). Import
  errors surface in the process that calls `instantiate()`.
- **Network metadata:** do not rename the advertised key. Keep `robot_class`
  as the metadata field so `ROBOT_TRANSPORT_PROTOCOL_VERSION` stays
  unchanged. Populate it from `robot["class_path"]`. Conflict checks compare
  that string unchanged; because spellings are no longer canonicalized, a
  defining-module path and a public re-export of the same driver compare as a
  mismatch. That stays **diagnostic, not fatal** — the warning names both
  strings and attach proceeds.
- **CLI `physicalai robot serve`:** require `--robot` (ComponentConfig
  JSON/YAML with `class_path` + `init_args`). Write only the new stdin
  shape.

### Public SharedCamera

Mirror the SharedRobot public story so step-5 implementers do not invent
three APIs:

`SharedCamera` is `@export_config(class_path="physicalai.capture.SharedCamera")`
— a **construction recipe** only (nested `camera` ComponentConfig,
`service_name`, `color_mode`, zero-copy / validation / idle knobs).
`to_config` of a runtime that holds `SharedCamera` is supported. Never
capture publisher / iceoryx2 node / frame / connected state.

- **`SharedCamera.from_config(camera, *, service_name=None, …)`** — primary
  API. Transport kwargs (zero-copy, validation, idle timeout, …) stay on the
  SharedCamera / publisher side, never inside `ComponentConfig`.
- **Constructor:** accept `camera: ComponentConfig` to spawn, or
  `camera=None` + `service_name` for attach-only (prefer
  :meth:`SharedCamera.from_publisher`). Flat `camera_type` /
  `camera_kwargs` are unsupported on the public API and publisher stdin
  (rejected before import) — not a dual adapter API. `camera` is a declared
  `config_args` argument, so nested `instantiate()` passes the recipe through
  without constructing a camera. There is no `from_camera()` sugar.
- **Who derives `service_name`:** `from_config` and the constructor.
  If `service_name` is omitted and `class_path` is a shareable built-in
  (`uvc` / `realsense` / `basler`), derive via the transport map + device id
  from `init_args`. If stub (`ip` / `genicam`) or third-party and
  `service_name` is omitted, fail before spawn. The publisher envelope always
  carries a concrete `service_name`; the child never re-derives it.
- **Attach-only / `from_publisher(service_name=…)`:** unchanged.

`ComponentConfig` does not import transport code. Instead, transport envelopes
consume the plain config:

```python
@dataclass(frozen=True)
class RobotOwnerConfig:
    name: str
    robot: ComponentConfig
    allow_remote: bool = False
    rate_hz: float = 100.0
    idle_timeout: float | None = 10.0
```

The same separation applies to camera service names, validation settings,
zero-copy mode, publisher rates, and idle timeouts. These settings describe
how a component is shared, not how the underlying component was constructed.

Config export capability does not imply process sharing. Studio owns the
product decision of whether a built driver should run in-process or be wrapped
in `SharedRobot`. `is_config_exportable` only answers whether that sharing path
can obtain a component config. Once wrapped, `SharedRobot` / `SharedCamera`
themselves are `@export_config` construction recipes, so Studio can
`to_config` a full runtime that already holds Shared\* without interim
serializers (Studio drop of those serializers remains rollout step 8).
Sketch:

```python
driver = await builder(robot, self)
if should_share(robot):  # Studio policy — not a Runtime field
    if not is_config_exportable(driver):
        raise ...
    if driver.is_connected():
        raise ...
    driver = SharedRobot.from_config(to_config(driver), name=robot.name)
```

`should_share` is Studio's concern (UI toggle, deployment setting, etc.). It is
not part of the generic component-config API and is not a Runtime attribute on
robots.

Catalog and plugin builders return ordinary runtime objects. Plugins never
import Zenoh, iceoryx2, `SharedRobot`, or `SharedCamera`.

## jsonargparse relationship

jsonargparse remains responsible for:

- generating CLI schemas from typed constructor signatures;
- merging command-line, environment, and YAML values;
- validating CLI input;
- instantiating runtime configs through the existing CLI.

The component-config layer is responsible for:

- recovering a replay recipe from an opted-in live object;
- validating the JSON-safe recipe;
- reconstructing it outside the CLI;
- feeding emitted **nested** `class_path` + `init_args` fragments back into
  jsonargparse without translation. A document root may use a CLI wrapper key
  such as `runtime:`, or be a bare `RobotRuntime` export that
  `physicalai run --config` unwraps into `runtime:`.

Do not depend on jsonargparse internals to remember live-object construction.
The explicit construction contract is also needed in subprocesses and Studio,
where no parser namespace exists.

## Security boundary

`class_path` is executable local configuration. `instantiate()` only accepts
trusted application or user-authored config. Never instantiate a config
received from robot metadata, camera metadata, Zenoh, shared memory, or
another untrusted peer.

Validation makes malformed input predictable; it does not make arbitrary
imports safe. Transport wire protocols may carry a config only from a trusted
parent process to the child it spawned. They must not accept component
configs from network subscribers. In particular, camera control requests may
patch only allowlisted scalar settings onto the publisher's trusted startup
recipe; they cannot select a class, device, or backend.

An allowlist resolver can be added for contexts that need a narrower trust
policy, but it does not replace the trusted-input rule for v1.

## Plugin contract

1. Implement the relevant runtime protocol (`Robot`, `Camera`,
   `ActionSource`, callback, or another typed component).
2. Opt into config export with `@export_config`. When the public import path
   differs from the defining module, pass
   `@export_config(class_path="physicalai.robot.SO101")`. Pass
   `scalar_var_kwargs=True` when flattened `**kwargs` must export as JSON
   scalars only (non-scalars fail at `to_config`).
3. Ensure every captured value is JSON-normalizable and constructor-compatible
   (JSON-safe scalars/collections, nested `@export_config` components, or
   domain values with `to_config_value()`). Do not auto-call arbitrary
   `to_dict()`. Dictionaries containing reserved `class_path` as ordinary data
   must go through `to_config_value()` or another wrapper.
4. Path-shaped constructor args may be relative. They resolve against the
   process cwd at open/`instantiate`. Owner/publisher children inherit the
   parent cwd at spawn — keep that cwd stable between export and spawn when
   using relatives. Absolute paths and `Path` are fine when you need them.
5. Export the class from a stable public import path (and set `class_path=`
   on the decorator when that path differs from the defining module).
6. Add construction round-trip tests.

Studio consumes only `is_config_exportable` and `to_config`, then
`SharedRobot.from_config` / camera equivalents. A full-component escape hatch
(without `@export_config`) is not part of the v1 contract.

Opt-in is transitive: a container component can export config only when all
captured nested component values can also export config.

## Failure semantics

- `to_config(value)` raises `ComponentConfigError` when a captured value cannot
  be normalized.
- `instantiate()` raises `ComponentConfigError` for malformed data before
  importing anything.
- `instantiate()` raises `ComponentImportError` when `class_path` cannot
  resolve to an importable class.
- Constructor validation and hardware configuration errors propagate from the
  constructor with the component path added as context.

Callers generally recover from these failures by correcting configuration, so
the first implementation can use one public `ComponentConfigError` base with
specific subclasses only where tests or callers distinguish phases.

## Suggested rollout

Status: steps 1–7 are implemented in Runtime, including the
`SharedRobot` / `SharedCamera` `@export_config` construction recipes that
enable step 8. Steps 8–10 (Studio client cutover, user-facing docs, inference
follow-up) remain open.

1. Add `ComponentConfig`, bounded normalization/instantiation, cycle checks,
   and one shared dotted-path resolver to `physicalai.config`. Consolidate the
   duplicated inference and robot importers behind that resolver
   (behavior-preserving). Do not change inference factory behavior.
2. Add `@export_config`, `to_config()`, and `is_config_exportable()` with
   signature, inheritance-depth, `class_path=`, domain `to_config_value`, and
   mutable-container tests.
3. Wire SO101, WidowXAI, and `BimanualWidowXAI` (nested `left`/`right`
   composite round-trips); add JSON and construction round-trip tests.
4. Add `SharedRobot.from_config()`; hard-cutover
   `RobotOwnerConfig` stdin to `robot: ComponentConfig` (rewrite writers,
   readers, fixtures; no `config_format`, no dual-read). Public ctor and
   CLI accept only `robot=` / `--robot` (plus `from_config`);
   flat `robot_class` / `robot_kwargs` are unsupported on public API, CLI,
   and stdin. Normalize class paths to public re-exports before
   store/advertise/compare; advertise metadata `robot_class` from that public
   `class_path`. Relative path args rely on parent cwd inheritance at `Popen`
   (no path-absolutizing helper).
5. Wire camera implementations, then hard-cutover the camera publisher stdin
   to `camera: ComponentConfig` with `service_name` beside it (same PR:
   rewrite fixtures; no dual-read). Add `SharedCamera.from_config()`;
   public ctor accepts only `camera=` / `from_config`
   (flat `camera_type` / `camera_kwargs` unsupported on public
   API and stdin — same post-step-4 SharedRobot simplification). Transport-
   owned shareable class-path → type-token map (`uvc` / `realsense` /
   `basler`) for derived names; stub (`ip` / `genicam`) and third-party
   require explicit `service_name`. Do not claim third-party shared camera
   support before this lands.
6. Wire the exact PolicySource graph listed above, `TeleopSource`, path-rooted
   `InferenceModel` configuration, and the v1 callback set.
7. Wire `RobotRuntime` and verify emitted nested configs load through both
   `instantiate()` and jsonargparse (under `runtime:` where the CLI requires
   it).
8. Studio drops interim serializers and applies explicit sharing policy to
   exportable plugin results. Runtime already exports
   `SharedRobot` / `SharedCamera` construction recipes (so a Studio
   step-8 path can `to_config` a runtime that holds Shared\*); full
   Studio client cutover remains this step.
9. Document component config next to `class_path` / `init_args` and state that
   persisted workflow versioning remains preview work.
10. In a separate inference design/change, audit manifest fixtures before
    considering `ComponentSpec` delegation, default semantics, depth counting,
    or class-path extra-field tightening.

The public jsonargparse `class_path` + `init_args` shape remains unchanged.
Robot-owner and camera-publisher startup payloads are private wire hard
cutovers; do not describe them as schema-preserving.

## Required tests

- Primitive, path, enum, mapping, list, and nested-component normalization.
- Non-finite floats are rejected during normalization.
- `Path` values emit `str(path)` as given (relative stays relative; absolute
  stays absolute); plain relative `str` paths are unchanged.
- Relative calibration (and similar path args) survive owner/publisher spawn
  when the parent cwd is unchanged between export and `Popen`; children
  inherit that cwd. Changing cwd in between with relative paths in the config
  is unsupported.
- The shared depth limit applies through configs, mappings, and lists; cyclic
  Python containers fail during normalization.
- JSON dump/load followed by reconstruction.
- Re-export produces the same normalized config.
- Supplied defaults remain represented; omitted defaults remain omitted.
- A config emitted with omitted optional nested components parses through
  jsonargparse with the same constructor defaults; examples do not emit nulls
  for arguments the caller omitted.
- Omitted `PolicySource.action_queue` reconstructs with `LerpSmoother`; bare
  `ChunkedActionQueue()` without smoother reconstructs with `ReplaceSmoother`.
- Positional calls bind to names and `**kwargs` flatten into `init_args`.
- Decorated `super()` calls do not overwrite the outer constructor capture.
- An undecorated overriding constructor fails instead of emitting base-only
  arguments, and the selected public path resolves to the most-derived type.
- `is_config_exportable` is true for decorator-marked classes; `to_config`
  works for them; Studio share keys off that predicate.
- jsonargparse still sees the original decorated constructor signature.
- Injected instance `to_config()` does not break structural Protocol checks;
  docs prefer module `to_config`.
- Malformed and ambiguous nested configs fail before import.
- Local classes and non-class import targets fail.
- Unsupported objects and lambdas report the complete argument path.
- `instantiate()` does not invoke lifecycle methods; eager constructor side
  effects such as `InferenceModel` artifact loading and `JsonlCallback` file
  open remain visible in tests.
- `InferenceModel` non-scalar / live override args fail at `to_config` (no
  silent drop).
- Robot owner and camera publisher subprocess handshakes remain JSON-only,
  accept only the new `robot:` / `camera: ComponentConfig` shape, and
  schema-positively reject unknown envelope keys (including legacy flat
  `robot_class` / `camera_type` forms) before import.
- Camera reconfigure control requests accept only positive integer
  `width` / `height` / `fps` settings, reject ComponentConfig and unknown-key
  payloads before side effects, and preserve the trusted startup class,
  device, backend, and remaining constructor arguments.
- Shareable SharedCamera spawn (`uvc` / `realsense` / `basler`) derives the
  legacy `service_name` via the transport class-path → type-token map inside
  `from_config` / the constructor; stub (`ip` / `genicam`) and third-party
  spawn without explicit `service_name` fails before publisher start; the
  publisher envelope carries a concrete `service_name` not inside
  `init_args`.
- The `SharedCamera` constructor takes a `ComponentConfig` mapping only;
  `camera` is declared via `config_args`, so a nested `instantiate()` does not
  construct the camera in the subscriber process (no `from_camera()` sugar);
  public ctor and publisher stdin reject flat `camera_type` /
  `camera_kwargs` as unsupported (not a dual API).
- Third-party camera class paths survive the publisher envelope and bypass the
  built-in camera registry.
- Public `SharedRobot` ctor and `physicalai robot serve` accept only
  `robot=` / `--robot` ComponentConfig (plus `from_config`);
  owner stdin allowlists envelope keys so flat `robot_class` /
  `robot_kwargs` fail as unknown keys (not a public dual API);
  a string `class_path` is stored exactly as written and metadata
  `robot_class` equals that string.
- Building a transport envelope imports nothing: a subscriber constructs a
  `SharedRobot` / `SharedCamera` and derives its service name on a machine
  where the driver package (and its vendor SDK) is not installed. Only the
  process that calls `instantiate()` imports the `class_path`.
- The `SharedRobot` constructor takes a `ComponentConfig` mapping only;
  `robot` is declared via `config_args`, so a nested `instantiate()` does not
  construct the driver in the subscriber process (no `from_robot()` sugar).
- Composite bimanual robots spawn as one owner; nested arm configs round-trip
  under the composite.
- The exact PolicySource nested graph (including smoothers) round-trips;
  unsupported nested values fail at `to_config`.
- The exact v1 callback set round-trips; other callbacks fail at `to_config`;
  omitted/empty callbacks stay omitted/empty.
- Referenced local configs and inline self-contained values have distinct tests.
- One complete `RobotRuntime` tree round-trips through both the generic loader
  and the jsonargparse CLI parser (nested fragments; root may use `runtime:`).
- Existing inference manifest fixtures and `ComponentSpec.from_class()` retain
  their current behavior in v1.
- Untrusted transport metadata is never passed to `instantiate()`.

## Alternatives considered

| Option                                                               | Why not                                                                                                                                     |
| -------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `to_dict()` on runtime protocols                                     | Mixes runtime behavior with construction                                                                                                    |
| Studio-only component specs                                          | Creates a parallel config model and does not scale to plugins                                                                               |
| Transport-specific robot/camera serializers                          | Duplicates the same replay problem by component type                                                                                        |
| jsonargparse namespace as source of truth                            | Live objects and subprocesses often have no parser namespace                                                                                |
| Automatic reflection of object attributes                            | Cannot distinguish constructor input, derived state, mutation, or resources                                                                 |
| Arbitrary codec registry in v1                                       | Adds global extension and security complexity before a second use case exists                                                               |
| Unify with inference `ComponentSpec` / `instantiate_component` in v1 | Different defaults, depth counting, extras, and registry mode; changes the inference critical path without helping robot/camera spawn       |
| Embed `service_name` inside `ComponentConfig`                        | Mixes transport naming with construction                                                                                                    |
| Hash `class_path` into third-party camera service names              | Unstable across arg ordering and omitted defaults; collisions                                                                               |
| Rename metadata field to `class_path` in v1                          | Would bump `ROBOT_TRANSPORT_PROTOCOL_VERSION`; keep `robot_class` populated from `class_path`                                               |
| Changing cwd between relative-path export and owner/publisher spawn  | Unsupported in v1; relatives resolve against process cwd at open                                                                            |
| Public `absolutize_component_paths`                                  | Easy to forget; fights folder-local relative export; redundant with Popen cwd inheritance                                                   |
| Universal string heuristic for IPC path absolutization               | Corrupts URLs / non-path tokens; wrong tool once relatives are first-class                                                                  |
| `__component_path_keys__` in v1                                      | Not needed without an absolutizer; defer until a second concrete need                                                                       |
| `config_format` / dual-read on owner/publisher stdin                 | Same-package ephemeral `Popen` handshake; hard cutover is enough — see [Private startup envelopes](#private-startup-envelopes-hard-cutover) |
| Shape dual-read (`robot` vs `robot_class`) without a version field   | Soft landing only; fixture churn is in-repo, so rewrite once                                                                                |
| Attach-only Studio                                                   | Clean separation, but changes operations by requiring serve-first                                                                           |

## References

- Runtime config shape: [config schema](../reference/config-schema.md)
- Runtime YAML guide: [write runtime config](../how-to/config/write-runtime-config.md)
- Security rules: [runtime security](security.md)
- Robot owner envelope: `physicalai.robot.transport._owner_config.RobotOwnerConfig`
- Camera publisher envelope: `physicalai.capture.transport._spec.CameraPublisherConfig`
  — named for symmetry with `RobotOwnerConfig`, which is itself named that way
  to avoid colliding with the unrelated `physicalai.inference.manifest.RobotSpec`
  manifest model. `CameraSpec` would collide with
  `physicalai.inference.manifest.CameraSpec` the same way, and the type no
  longer describes a `camera_type` + `camera_kwargs` spec anyway — it wraps one
  `camera: ComponentConfig`.
- Review context: Mark's comments on Studio PR #818 (factory wrap; plugins
  should not care about `SharedRobot`)
