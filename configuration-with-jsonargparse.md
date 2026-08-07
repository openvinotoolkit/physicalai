# Configuration and component construction with jsonargparse

Status: proposed

## Decision

PR #212 keeps the public `Config` API in Runtime. This migration makes
jsonargparse the implementation for typed configuration and known-type
component construction. It does not create a second configuration engine.

The final rule is:

```text
known target type + mapping/file/CLI
    -> jsonargparse parser
    -> validation
    -> parser.instantiate()
```

Only behavior that jsonargparse does not provide remains in Runtime-owned
compatibility or policy code: live constructor capture, portable recipe safety,
transport envelopes, and explicit manifest alias/artifact handling.

## Current Runtime API

PR #212 moves the `Config` API from Studio into Runtime. The current API has two
related roles.

### Typed configuration classes

Runtime currently supports typed configuration classes based on dataclasses:

```python
@dataclass
class ACTConfig(Config):
    hidden_size: int = 256
    learning_rate: float = 1e-3


config = ACTConfig.from_dict({"hidden_size": 512})
config.save("act.yaml")
config = ACTConfig.load("act.yaml")
```

`Config` and its serialization/loading utilities inspect constructor or dataclass
fields, convert values, apply defaults, and reconstruct objects.

### Component configuration utilities

Runtime also supports a generic component recipe:

```python
recipe = {
    "class_path": "physicalai.capture.UVCCamera",
    "init_args": {"device": 0},
}

camera = instantiate(recipe)
```

The surrounding utility functions validate the recipe shape, normalize values,
import the class, recursively instantiate nested recipes, and provide
compatibility helpers such as `instantiate_obj` and `FromConfig`.

### CLI configuration

The CLI already uses jsonargparse as its parser and construction engine:

```python
parser = ArgumentParser()
parser.add_class_arguments(RobotRuntime, "runtime")
parser.add_method_arguments(RobotRuntime, "run", "run")

parsed = parser.parse_args()
initialized = parser.instantiate(parsed)
runtime = initialized.runtime
```

Therefore Runtime currently has two legitimate construction lanes:

```text
typed Python API / CLI
    -> jsonargparse
    -> parser.instantiate()

portable class_path recipe
    -> Runtime safety/policy validation
    -> jsonargparse when an expected type is available
    -> strict compatibility constructor otherwise
```

## Overlap With jsonargparse

The two paths implement substantially overlapping behavior:

| Runtime utility behavior         | Existing jsonargparse capability             |
| -------------------------------- | -------------------------------------------- |
| Read YAML or dictionaries        | `parse_path`, `parse_object`                 |
| Inspect constructor signatures   | `add_class_arguments`                        |
| Apply typed defaults             | Typed argument parsing and `defaults` policy |
| Validate constructor arguments   | Signature and type-hint validation           |
| Reject unknown arguments         | Schema-aware parsing                         |
| Reconstruct dataclasses          | Dataclass argument support and instantiation |
| Construct nested typed objects   | Nested class and subclass arguments          |
| Select component subclasses      | `add_subclass_arguments`                     |
| Serialize parsed configuration   | `dump`, `save`                               |
| Use one schema for CLI and files | The same parser accepts both                 |

The migration removes duplicate signature parsing and typed reconstruction. It
does not remove policy code merely because it also calls a constructor.

## Target Design

Use jsonargparse as the construction engine for both Python APIs and the CLI.
The compatibility `Config` methods are thin entry points into this same engine:

```text
                 file / dictionary / CLI
                              |
                              v
                  package-owned parser builder
                              |
                              v
                       jsonargparse
                    parse / validate / dump
                              |
                              v
                     parsed Namespace
                         |          |
                         |          +--> save / inspect / policy checks
                         v
                   parser.instantiate()
                              |
                              v
                       live object graph
```

The generic API is a parser factory parameterized by the target type. It can be
used with any inspectable class, dataclass, Pydantic model, or compatible
constructor:

```python
T = TypeVar("T")


def build_class_parser(
    target: type[T],
    *,
    root: str = "config",
) -> ArgumentParser:
    parser = ArgumentParser(exit_on_error=False)
    parser.add_class_arguments(target, root)
    return parser


parser = build_class_parser(Client, root="client")
parsed = parser.parse_object(
    {"client": {"host": "api.local", "port": 8080}},
    defaults=False,
)
client = parser.instantiate(parsed).client
```

The same `build_class_parser(Client)` can be used by a library API, a CLI
command, tests, or a service. This is the generic setup. The target type is
still required because it supplies the schema that jsonargparse validates.

For a complete application with multiple objects and method arguments, a
package-owned builder composes the same generic operations:

```python
def build_runtime_parser() -> ArgumentParser:
    parser = ArgumentParser(exit_on_error=False)
    parser.add_class_arguments(RobotRuntime, "runtime")
    parser.add_method_arguments(RobotRuntime, "run", "run")
    return parser
```

This builder is workflow-specific, but it does not implement configuration
parsing or construction. It only declares the application schema. The generic
factory handles one known target; package-owned builders describe larger
object graphs.

`Config`, `FromConfig`, and `instantiate_obj` remain public compatibility APIs.
Runtime code should use jsonargparse-backed `Config` methods and package parser
builders directly. The compatibility APIs delegate to those paths whenever an
expected type is available.

This gives the project one internal path without forcing downstream packages to
migrate immediately:

```text
Runtime internal code:
    parser builder -> jsonargparse -> parser.instantiate()

Downstream compatibility code:
    FromConfig / instantiate_obj -> thin adapter -> same parser path
```

## Remaining Runtime Adapters

jsonargparse does not replace every configuration responsibility. Runtime may
still need small, package-specific adapters for:

- canonical `class_path` and `init_args` recipe validation;
- explicit component aliases and plugin registries;
- recursion and cycle limits before dynamic imports;
- import allowlists for manifests and lower-trust inputs;
- artifact path resolution;
- transport envelope validation;
- temporary object-to-config export compatibility.

These adapters must not implement constructor signature parsing. Once a recipe
has been normalized and authorized, jsonargparse should validate it against the
expected component base type and instantiate it. The strict schema-free recipe
path remains only for compatibility and process-boundary cases where no root
type is available.

The canonical portable recipe remains:

```yaml
class_path: package.module.ClassName
init_args:
  option: value
```

This does not require one Python class to represent typed application
configuration, manifest aliases, and live-object export.

## Scope

This design covers reusable configuration infrastructure for:

- command-line applications;
- Python library APIs;
- training configuration;
- inference manifests;
- plugin components;
- runtime object graphs;
- open-source and closed-source packages.

This design does not define a PhysicalAI-specific workflow schema. A package
still owns its root configuration model, plugin interfaces, artifact rules, and
trust policy.

## Goals

1. Use one implementation for type-aware parsing and object construction.
2. Keep configuration files and CLI arguments on the same schema.
3. Reject misspelled or incorrectly typed constructor arguments before calling
   constructors.
4. Support nested components using `class_path` and `init_args`.
5. Support dataclasses, Pydantic models, protocols, enums, paths, collections,
   and subclass selection.
6. Keep package-specific policy outside the generic engine.
7. Preserve a stable JSON-compatible recipe for manifests and process startup.
8. Make the configuration layer usable without importing PhysicalAI-specific
   modules.
9. Define explicit behavior for trusted and untrusted configuration sources.
10. Make future extraction of `physicalai.inference` possible without copying a
    second instantiation engine.

## Non-goals

1. Make arbitrary dynamic imports safe. `class_path` executes installed Python
   code regardless of which parser imports it.
2. Automatically install packages named by a configuration file.
3. Use one Python model for every kind of configuration.
4. Store live hardware state, open handles, sessions, or mutable runtime state.
5. Replace manifest metadata validation with constructor signature parsing.
6. Use reflection-based subclass names as a stable plugin registry.

## Terminology

### Typed configuration

Data whose schema is known from a class, dataclass, Pydantic model, function, or
method signature.

```python
@dataclass
class TrainConfig:
    epochs: int = 10
    learning_rate: float = 1e-3
```

### Class recipe

A JSON-compatible description of one constructor call.

```python
{
    "class_path": "package.Normalize",
    "init_args": {"mean": 0.5},
}
```

### Component reference

A manifest-facing reference that can use either a class recipe or a registered
alias.

```yaml
type: normalize
params:
  artifact: stats.safetensors
```

### Workflow document

An application-level document containing multiple constructor groups and method
arguments.

```yaml
runtime:
  robot: ...
  action_source: ...
run:
  duration_s: 60
```

### Parsed configuration

A jsonargparse `Namespace` produced by `parse_args`, `parse_object`,
`parse_path`, or another parser method.

## Current PhysicalAI implementation

The current `physicalai.config` package contains several overlapping systems.

| Current area                     | Responsibility                                             |
| -------------------------------- | ---------------------------------------------------------- |
| `base.py`                        | Direct class recipe and typed dataclass base class         |
| `_normalize.py`                  | JSON normalization and strict recipe validation            |
| `_instantiate.py`                | Generic recursive class recipe instantiation               |
| `_export.py`                     | Capture constructor arguments from live objects            |
| `loading.py`                     | Dispatch across dict, file, Pydantic, and dataclass inputs |
| `mixin.py`                       | `FromConfig` methods and decorator                         |
| `serializable.py`                | Legacy dataclass-to-plain-value conversion only            |
| `_yaml.py`                       | YAML loading and saving                                    |
| `_envelope.py`                   | Transport envelope validation                              |
| `inference/component_factory.py` | Manifest alias/artifact policy plus thin parser call       |
| `cli/run.py`                     | jsonargparse parsing and instantiation                     |

Before this migration, the overlap created multiple paths with different
behavior:

```text
Config typed methods             -> custom typed utilities
instantiate_obj()               -> custom source dispatch -> strict instantiator
RobotRuntime.from_config()      -> jsonargparse parser -> parser.instantiate()
physicalai run                  -> jsonargparse parser -> parser.instantiate()
instantiate_component()         -> inference-specific recursive instantiator
```

The target is one schema-aware path for normal construction:

```text
all typed inputs -> jsonargparse parser -> parser.instantiate()
```

A small strict recipe adapter remains only for cases where no root schema is
available.

## jsonargparse capability assessment

The assessment below refers to jsonargparse 4.50.0.

| Required behavior                    | jsonargparse support     | Notes                                                                     |
| ------------------------------------ | ------------------------ | ------------------------------------------------------------------------- |
| YAML configuration files             | Yes                      | `parse_path`, `action="config"`                                           |
| Dict configuration                   | Yes                      | `parse_object`                                                            |
| CLI overrides                        | Yes                      | Config files and CLI use one parser                                       |
| Environment overrides                | Yes                      | Parser configuration controls precedence                                  |
| Typed constructor validation         | Yes                      | Signature and type-hint based                                             |
| Unknown-key rejection                | Yes                      | When the expected class type is known                                     |
| Nested class construction            | Yes                      | Class annotations use `class_path` and `init_args`                        |
| Subclass validation                  | Yes                      | Checks import, subclass relation, and init arguments                      |
| Protocol-based components            | Yes                      | Protocol signatures are checked                                           |
| Dataclasses                          | Yes                      | Parsing, nesting, and construction                                        |
| Pydantic models                      | Yes                      | Parsing, nesting, and construction                                        |
| Enums and paths                      | Yes                      | Type-aware parsing and serialization                                      |
| Callable factories                   | Yes                      | `Callable`, protocols, and functions returning classes                    |
| Config serialization                 | Yes                      | `dump`, `save`, and `--print_config`                                      |
| Omitted/default distinction          | Yes                      | `defaults=False`, `skip_default`, and 4.50 `Unset` support                |
| Config fragments                     | Yes                      | `sub_configs=True`                                                        |
| Argument links                       | Yes                      | Parse-time and instantiate-time links                                     |
| Custom constructor hook              | Yes                      | `add_instantiator`                                                        |
| `from_config` mixin                  | Yes                      | Public `FromConfigMixin` since 4.48                                       |
| Stable alias registry                | No                       | Short names are based on imported subclasses, not explicit aliases        |
| Strict arbitrary recipe API          | Partial                  | Strict when expected type is known; unsafe fallback with `Any`            |
| Recursion depth limit                | No                       | Documentation states type nesting has no limit                            |
| General Python-container cycle check | No documented API        | Config include loops are checked, not arbitrary object cycles             |
| Import allowlist callback            | No public API identified | Required for lower-trust manifests                                        |
| Scoped custom instantiators          | Limited                  | Public 4.50 API registers global instantiators                            |
| Live object to constructor recipe    | No general solution      | Dumping a live object does not recover caller-supplied constructor intent |
| Transport envelope schema            | No                       | Application-specific concern                                              |
| Artifact path resolution             | No                       | Manifest-specific concern                                                 |

## Important jsonargparse behavior

### Strict parsing requires an expected type

This is strict because the parser knows `Parent`:

```python
parser = ArgumentParser(exit_on_error=False)
parser.add_argument("--component", type=Parent)

cfg = parser.parse_object({
    "component": {
        "class_path": "package.Parent",
        "init_args": {
            "child": {
                "class_path": "package.Child",
                "init_args": {"value": 7},
            },
        },
    },
})

objects = parser.instantiate(cfg)
parent = objects.component
```

The parser validates:

- `package.Parent` is importable;
- it is compatible with `Parent`;
- `child` is a valid constructor argument;
- the nested class is compatible with the annotation of `child`;
- `value` has the expected type.

Misspelled arguments fail during parsing:

```python
cfg = {
    "component": {
        "class_path": "package.Parent",
        "init_args": {"chid": {}},
    },
}

# parse_object raises before Parent.__init__ is called.
parser.parse_object(cfg)
```

### `Any` is not a strict arbitrary-class boundary

jsonargparse supports class specs nested under `Any`, but its implementation
intentionally catches adaptation errors and returns the original mapping. The
following can remain uninstantiated instead of raising:

```python
parser = ArgumentParser(exit_on_error=False)
parser.add_argument("--component", type=Any)

cfg = parser.parse_object({
    "component": {
        "class_path": "does.not.Exist",
        "init_args": {},
    },
})

result = parser.instantiate(cfg)

# result.component can still be the original mapping.
```

Therefore, do not implement a strict generic API as:

```python
# Do not use this as a strict recipe loader.
parser.add_argument("--component", type=Any)
```

Use a known base class or protocol whenever possible.

### Defaults require an explicit policy

Most configuration workflows want constructor defaults to apply at construction
time without materializing every default in persisted configuration.

Use:

```python
cfg = parser.parse_object(data, defaults=False)
obj = parser.instantiate(cfg)
```

Persist with a consistent omission policy:

```python
text = parser.dump(
    cfg,
    skip_default=True,
    skip_unset=True,
)
```

Exact options should be verified against the supported jsonargparse version.
The project should not mix documents that materialize defaults with documents
that preserve omission without an explicit reason.

### Parsing and instantiation are separate phases

Keep these operations separate:

```python
parsed = parser.parse_path("config.yaml")

# Inspect, validate package policy, resolve artifacts, or display the result.
validate_policy(parsed)

objects = parser.instantiate(parsed)
```

Do not hide parsing and instantiation inside a loader when callers need to
inspect the resolved configuration first.

## Proposed architecture

### Layers

```text
+--------------------------------------------------------------+
| Package-specific schemas                                     |
| RuntimeConfig, TrainConfig, Manifest, VisionConfig            |
+--------------------------------------------------------------+
| Package-specific adapters                                    |
| alias registry, artifact paths, trust policy, workflow shape |
+--------------------------------------------------------------+
| Shared configuration conventions                             |
| canonical ClassSpec, depth check, parser builders             |
+--------------------------------------------------------------+
| jsonargparse                                                 |
| parse, merge, validate, dump, save, instantiate               |
+--------------------------------------------------------------+
| Python constructors                                           |
+--------------------------------------------------------------+
```

### Dependency direction

```text
application package
    |
    +-- owns root schema and parser builder
    |
    +-- uses shared configuration conventions
             |
             +-- uses jsonargparse public API
```

The shared layer must not import robotics, vision, training, or application
classes.

### Canonical class recipe

Use one wire shape across packages:

```python
class ClassSpec(TypedDict):
    class_path: str
    init_args: NotRequired[dict[str, JsonValue]]
```

JSON value definition:

```python
JsonScalar = None | bool | int | float | str
JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
```

Rules for portable documents:

1. `class_path` is a non-empty fully qualified import path.
2. `init_args` is optional and defaults to an empty mapping.
3. Mapping keys are strings.
4. Floats are finite.
5. Values are JSON-compatible.
6. A configured maximum depth applies before imports.
7. Cyclic Python mappings and lists are rejected.
8. Package trust policy authorizes each dynamic class path.

jsonargparse can use richer internal values while parsing CLI input. Manifests,
transport startup recipes, and persisted portable recipes use the stricter JSON
subset.

### Parser ownership

Each package should expose parser builders rather than a process-global parser.

```python
def build_runtime_parser() -> ArgumentParser:
    parser = ArgumentParser(exit_on_error=False)
    parser.add_class_arguments(RobotRuntime, "runtime")
    parser.add_method_arguments(RobotRuntime, "run", "run")
    return parser
```

```python
def build_training_parser() -> ArgumentParser:
    parser = ArgumentParser(exit_on_error=False)
    parser.add_class_arguments(Trainer, "trainer")
    parser.add_class_arguments(DataModule, "data")
    parser.add_subclass_arguments(Policy, "policy", required=True)
    return parser
```

```python
def build_vision_parser() -> ArgumentParser:
    parser = ArgumentParser(exit_on_error=False)
    parser.add_subclass_arguments(VisionModel, "model", required=True)
    return parser
```

Parser builders are normal library functions. They can be used by CLI code,
Python APIs, tests, and services.

## Recommended APIs

### Preserve the typed `Config` model

```python
T = TypeVar("T")

@dataclass
class ClientConfig(Config):
    host: str = "localhost"
    port: int = 80


client_config = ClientConfig.load(
    {"host": "api.local", "port": 8080},
)
client_config = ClientConfig.load("client.yaml")
```

This preserves the existing model and inheritance relationship. `Config` remains
the base class for typed configuration objects, and `load` returns an instance
of the subclass for either a mapping or a file path. jsonargparse replaces the
implementation behind this method; it does not change its return type.

The implementation may retain parser state privately on instances created
through `load` so `save()` can serialize the same parsed
configuration. This is implementation state, not a second configuration model.

### Parse a dictionary for a known class

```python
def parse_class_config(
    target: type[T],
    data: Mapping[str, object],
) -> T:
    parser = ArgumentParser(exit_on_error=False)
    parser.add_class_arguments(target, "object")

    namespace = parser.parse_object(
        {"object": dict(data)},
        defaults=False,
    )

    return cast(T, parser.instantiate(namespace).object)
```

Usage:

```python
client = parse_class_config(
    Client,
    {"host": "api.local", "port": 8080},
)

```

### Generic helper boundary

The helpers above are intentionally generic. They do not depend on PhysicalAI
types and can be used for any class whose constructor can be inspected by
jsonargparse:

- ordinary classes with typed constructor arguments;
- dataclasses and Pydantic models;
- protocols with compatible implementations;
- subclass-selected components;
- nested combinations of the above.

They are not a replacement for a schema. A strict helper must receive either a
known target type or a known component base type. An arbitrary dictionary with
no expected type cannot be validated against constructor arguments by
jsonargparse.

The intended generic API is small and consists of the helpers already shown:

```python
parse_class_config(target, data)       # flat config for one known target
parse_component(base, class_spec)     # class_path recipe for a known base
```

The helper returns the requested target type. Parser construction and
instantiation remain centralized in jsonargparse:

```python
client = parse_class_config(Client, {"host": "api.local"})
```

The equivalent convenience function is deliberately thin:

```python
def instantiate_class_config(
    target: type[T],
    data: Mapping[str, object],
) -> T:
    return parse_class_config(target, data)


def instantiate_component_config(
    base: type[T],
    spec: ClassSpec,
) -> T:
    return parse_component(base, spec)
```

These functions must not contain a second recursive constructor, signature
parser, serializer, or source-dispatch system. Package-specific code remains
responsible for building the root schema, resolving aliases, resolving artifact
paths, and applying trust policy before calling the generic helpers.

### Pseudocode design

The complete flow is intentionally straightforward:

```text
known typed class/config
        |
        v
parse_class_config(target, mapping)
        |
        +--> ArgumentParser()
        +--> add_class_arguments(target, "object")
        +--> parse_object({"object": mapping}, defaults=False)
        |
        v
parsed configuration
        |
        +--> inspect / policy validation / dump
        |
        +--> parser.instantiate(namespace)
        |
        v
live target object
```

For a component recipe:

```text
class_path/init_args recipe
        |
        v
portable preflight
  - shape and JSON checks
  - cycle and depth checks
  - import policy check
        |
        v
parse_component(base, recipe)
        |
        +--> ArgumentParser()
        +--> add_subclass_arguments(base, "component")
        +--> parse_object({"component": recipe}, defaults=False)
        |
        v
parsed configuration
        |
        +--> parser.instantiate(namespace)
        |
        v
live component object
```

The artificial root keys (`object` and `component`) are implementation details
of the helper. They provide jsonargparse with the expected type and must not
become part of the persisted class recipe.

### Config construction helpers

`Config` retains the existing typed-model API while delegating parsing and
construction to jsonargparse. `load` accepts either a mapping or a file path:

```python
class Config:
    @classmethod
    def load(cls, source: Mapping[str, object] | str | Path) -> Self:
        if isinstance(source, Mapping):
            return parse_class_config(cls, source)
        return parse_class_config(cls, load_mapping(source))
```

If PR #212 exposes `from_dict`, it may remain as a deprecated compatibility
alias for `load(mapping)`, but it is not part of the target API. `Config` should
not implement another parser. New code should
prefer a parser builder and a configuration-first flow:

```python
parser = build_application_parser()
namespace = parser.parse_path(path, defaults=False)
validate_application_policy(namespace)
application = parser.instantiate(namespace)
```

This preserves one schema for dictionaries, files, and CLI arguments while
keeping the existing typed `Config` API stable. `save_legacy_config` represents
the fallback for directly constructed instances; it is not part of the new
jsonargparse construction engine.

### Parse a subclass recipe

```python
def parse_component(
    base: type[T],
    spec: ClassSpec,
) -> T:
    validate_class_recipe(spec)
    check_import_policy(spec)

    parser = ArgumentParser(exit_on_error=False)
    parser.add_subclass_arguments(
        base,
        "component",
        required=True,
    )

    namespace = parser.parse_object(
        {"component": spec},
        defaults=False,
    )

    return cast(T, parser.instantiate(namespace).component)
```

Usage:

```python
processor = parse_component(
    Preprocessor,
    {
        "class_path": "my_package.Resize",
        "init_args": {"width": 640, "height": 480},
    },
)

```

### Parse a protocol implementation

jsonargparse supports protocols. A domain can avoid a shared implementation
base class:

```python
class Preprocessor(Protocol):
    def __call__(
        self,
        inputs: dict[str, Array],
    ) -> dict[str, Array]: ...
```

```python
parser.add_argument(
    "--preprocessor",
    type=Preprocessor,
)
```

Protocol compatibility requirements should be tested with real plugin classes.
Exact protocol signature matching can be stricter than normal structural typing.

### Direct class API

For classes that need a convenience constructor, use jsonargparse's public
mixin where its behavior matches the required API:

```python
from jsonargparse import FromConfigMixin

class Client(FromConfigMixin):
    def __init__(
        self,
        host: str = "localhost",
        port: int = 80,
    ) -> None:
        self.host = host
        self.port = port
```

```python
client = Client.from_config({
    "host": "api.local",
    "port": 8080,
})
```

```python
client = Client.from_config("client.yaml")
```

Do not build a second signature parser inside a package-specific `FromConfig`
mixin.

If the existing `FromConfig` API requires methods such as `from_pydantic` or
`from_dataclass`, convert those values to a dictionary and delegate:

```python
class FromConfig(FromConfigMixin):
    @classmethod
    def from_pydantic(cls, value: BaseModel):
        return cls.from_config(value.model_dump())

    @classmethod
    def from_dataclass(cls, value):
        return cls.from_config(dataclasses.asdict(value))
```

Keep these adapters small. They should not recursively instantiate classes on
their own.

### Typed configuration without a custom base class

Prefer plain dataclasses or Pydantic models:

```python
@dataclass
class PolicyConfig:
    hidden_size: int = 256
    activation: Activation = Activation.GELU
    checkpoint: Path | None = None
```

Parse through a parser:

```python
parser = ArgumentParser(exit_on_error=False)
parser.add_class_arguments(PolicyConfig, "policy")

cfg = parser.parse_path("policy.yaml")
policy_config = parser.instantiate(cfg).policy
```

Save through the same parser:

```python
parser.save(cfg, "policy.yaml")
```

This removes the need for a custom dataclass reconstruction engine for normal
configuration parsing.

A compatibility `Config` base can delegate to this path during migration.

## ComponentSpec convergence

### Current issue

The inference manifest `ComponentSpec` supports two forms:

```yaml
class_path: package.Normalize
init_args:
  mean: 0.5
```

and:

```yaml
type: normalize
artifact: stats.safetensors
```

The second form combines alias selection and flat parameters in one permissive
Pydantic model. The component factory then implements its own recursive
instantiation.

### Target model

Represent the forms separately:

```python
class ClassComponentSpec(BaseModel):
    class_path: str
    init_args: dict[str, JsonValue] = Field(default_factory=dict)


class RegisteredComponentSpec(BaseModel):
    type: str
    params: dict[str, JsonValue] = Field(default_factory=dict)
```

Accept the existing flat manifest shape at the input boundary:

```python
def parse_legacy_registered_spec(data: Mapping[str, object]):
    type_name = require_string(data, "type")
    params = {
        key: value
        for key, value in data.items()
        if key != "type"
    }
    return RegisteredComponentSpec(
        type=type_name,
        params=params,
    )
```

Normalize aliases before invoking jsonargparse:

```python
def resolve_component_spec(
    spec: ComponentSpec,
    registry: ComponentRegistry,
) -> ClassComponentSpec:
    if isinstance(spec, ClassComponentSpec):
        return spec

    class_path = registry.resolve(spec.type)
    return ClassComponentSpec(
        class_path=class_path,
        init_args=resolve_nested_specs(spec.params, registry),
    )
```

Then use the expected component interface:

```python
def instantiate_component(
    spec: ComponentSpec,
    *,
    base: type[T],
    registry: ComponentRegistry,
) -> T:
    canonical = resolve_component_spec(spec, registry)
    parsed = parse_component(base, canonical.model_dump())
    return parsed.instantiate()
```

The registry remains package-specific. jsonargparse's short subclass names are
not a replacement because they depend on imported subclasses and class names.
Explicit aliases are stable and can be backed by entry points.

### Nested components

Let type annotations drive nested construction:

```python
class Pipeline:
    def __init__(
        self,
        preprocessors: list[Preprocessor],
        runner: Runner,
        postprocessors: list[Postprocessor],
    ) -> None:
        ...
```

A parser built from `Pipeline` knows the expected type at every nested field:

```python
parser.add_class_arguments(Pipeline, "pipeline")
parsed = parser.parse_object(document)
pipeline = parser.instantiate(parsed).pipeline
```

Avoid generic recursion that treats every dictionary containing `class_path` as
a component regardless of its typed location.

## Configuration-first export

### Problem with object-to-config reconstruction

The current `@export_config` decorator captures caller-supplied constructor
arguments. This supports:

```python
camera = UVCCamera(device=0)
spec = Config.from_instance(camera)
```

jsonargparse does not provide a general equivalent. Serializing an arbitrary
live instance cannot reliably recover:

- which constructor arguments were supplied;
- which values came from defaults;
- values transformed by the constructor;
- mutable arguments before later mutation;
- a stable public import path;
- arguments intentionally excluded from persisted configuration.

### Preferred model

Keep configuration as the source of truth:

```python
parsed = parser.parse_object(document, defaults=False)
camera = parser.instantiate(parsed).camera

save_for_replay(parsed)
```

For imperative construction, construct the recipe first:

```python
camera_spec = {
    "class_path": "physicalai.capture.UVCCamera",
    "init_args": {"device": 0},
}

camera = instantiate_component(
    camera_spec,
    base=Camera,
    registry=registry,
)
```

For APIs that need replay or persistence, keep the parsed `Namespace` in the
calling application or service. Do not attach it to the live object and do not
introduce a generic object-plus-configuration wrapper:

```python
parsed = parse_component(Camera, camera_spec)
camera = parsed.instantiate()

# The application decides whether and where to persist this configuration.
save_for_replay(parsed.namespace)
```

### Compatibility period

Keep `@export_config` temporarily for existing APIs that construct objects first
and export later. Do not extend it into the new generic foundation.

Migration direction:

```text
object first -> inspect/capture -> recipe
```

becomes:

```text
recipe first -> parse -> object
                    |
                    +-> caller may retain parsed Namespace for replay
```

## Process-boundary recipes

Some constructors consume a component recipe as data and pass it to another
process. The current `@export_config(config_args=...)` marks these arguments so
the custom instantiator does not construct them in the current process.

Represent this intent in the constructor type instead:

```python
class ClassSpec(TypedDict):
    class_path: str
    init_args: NotRequired[dict[str, JsonValue]]
```

```python
class SharedCamera:
    def __init__(
        self,
        camera: ClassSpec,
        service_name: str,
    ) -> None:
        self.camera_spec = camera
```

Because `camera` is data, not `Camera`, jsonargparse should validate it as a
mapping and not instantiate it.

The worker uses a typed parser at the construction boundary:

```python
def worker_main(spec: ClassSpec) -> None:
    camera = parse_component(
        Camera,
        spec,
    ).instantiate()
    camera.connect()
```

This makes process ownership visible in type annotations instead of decorator
metadata.

## Manifests and metadata

Keep Pydantic or an equivalent schema library for the complete manifest:

```python
class Manifest(BaseModel):
    format: Literal["policy_package"]
    version: str
    artifacts: dict[str, str]
    runner: ComponentSpec
    preprocessors: list[ComponentSpec]
    postprocessors: list[ComponentSpec]
```

Responsibilities remain separate:

```text
Pydantic manifest model
    -> validates package metadata and component-reference shapes
    -> resolves aliases and artifact paths
    -> produces canonical class recipes

jsonargparse
    -> validates constructor signatures and nested component types
    -> instantiates the object graph
```

Do not ask jsonargparse to validate the complete manifest unless the manifest is
itself directly modeled as constructor arguments. Metadata schemas and
constructor schemas change for different reasons.

## Security model

### Trust levels

Define trust at each entry point.

| Source                           | Dynamic class policy                                           |
| -------------------------------- | -------------------------------------------------------------- |
| Local developer CLI config       | Full installed `class_path`, with warning in docs              |
| Parent-to-child startup config   | Class paths from already validated parent config               |
| First-party signed model package | Allowlisted namespaces or registered plugins                   |
| Downloaded third-party manifest  | Registered aliases by default; explicit opt-in for class paths |
| Network peer metadata            | No dynamic class paths                                         |
| Runtime reconfiguration request  | Allowlisted scalar fields only                                 |

### Preflight before jsonargparse imports

jsonargparse imports classes while parsing class specs. Apply inexpensive
structural and policy checks first for portable or lower-trust documents:

```python
def preflight(
    value: object,
    *,
    max_depth: int,
    class_policy: ClassPolicy,
) -> None:
    walk_json_tree(
        value,
        max_depth=max_depth,
        reject_cycles=True,
        reject_non_finite_floats=True,
        require_string_keys=True,
    )

    for spec in find_class_specs(value):
        class_policy.check(spec["class_path"])
```

Example policy:

```python
class ClassPolicy(Protocol):
    def check(self, class_path: str) -> None: ...
```

```python
class PrefixAllowlist:
    def __init__(self, prefixes: tuple[str, ...]):
        self.prefixes = prefixes

    def check(self, class_path: str) -> None:
        if not class_path.startswith(self.prefixes):
            raise ConfigSecurityError(
                f"class_path not allowed: {class_path}"
            )
```

Prefix checks alone are not sufficient for hostile environments, but they are a
simple policy for first-party package boundaries. Entry-point identity checks or
explicit class-path sets are stronger.

### Depth limit

jsonargparse documents arbitrary type nesting without a configured limit. Keep a
preflight limit for external documents even if jsonargparse later adds one.

```python
DEFAULT_MAX_CONFIG_DEPTH = 20
```

Select the final value from realistic component graphs and test it. Do not
silently retry with a larger value after a limit failure.

### No remote package installation

A manifest can name a required package, but loading must not automatically run
an installer.

```python
if plugin_not_installed:
    raise MissingPluginError(
        "Install package-name to load component alias 'name'"
    )
```

### Parse before instantiate

Never combine lower-trust parsing and construction in one opaque call:

```python
parsed = parse_manifest(path)
resolved = resolve_components(parsed)
preflight(resolved)
constructor_cfg = parse_constructor_config(resolved)
objects = constructor_cfg.instantiate()
```

## Package placement

### Option A: use jsonargparse directly in each package

```text
vision-package ------> jsonargparse
training-package ----> jsonargparse
inference-package ---> jsonargparse
physicalai ----------> jsonargparse
```

Advantages:

- no extra shared package;
- public upstream APIs remain visible;
- fewer wrapper compatibility problems.

Disadvantages:

- alias, trust, and canonical-recipe policy can diverge;
- repeated parser-builder utilities;
- repeated conformance tests.

### Option B: small shared configuration package

```text
vision-package ------+
training-package ----+--> component-config --> jsonargparse
inference-package ---+
physicalai ----------+
```

The shared package contains only:

```text
component_config/
  types.py        # JsonValue, ClassSpec
  preflight.py    # JSON/depth/cycle/import-policy checks
    config.py       # optional shared policy/config helpers
  builders.py     # known-class/subclass parser helpers
  registry.py     # generic explicit alias registry interface
  errors.py       # package-neutral boundary errors
```

It must not contain:

- robotics types;
- inference artifacts;
- training defaults;
- workflow-specific envelopes;
- copied jsonargparse parsing logic;
- another recursive instantiator.

Advantages:

- one portable recipe contract;
- one security preflight implementation;
- one conformance suite;
- independent use by open and closed packages.

Disadvantages:

- another package and release boundary;
- risk of becoming a broad wrapper around jsonargparse;
- version coordination.

### Option C: inference package owns shared configuration

```text
vision-package ------> inferencekit.components
physicalai ----------> inferencekit.components
training-package ----> inferencekit.components
```

This is reasonable only when all consumers already depend on the inference
package. It is a poor fit for packages that need configuration but not
inference.

### Recommendation

Start with Option A while developing the API against jsonargparse 4.50. Extract
Option B only after two independent packages use the same preflight and parser
helpers.

Do not place general training or application configuration under the inference
package. The inference package can own manifest component normalization while
using the shared class recipe contract.

## Upstream jsonargparse requests

Before implementing a permanent strict adapter, contact the jsonargparse
maintainer with concrete examples and ask whether existing public APIs cover the
cases below.

### 1. Strict parser-independent class recipe parsing

Desired API shape:

```python
parsed = parse_class_spec(
    value,
    expected_type=Preprocessor,
    defaults=False,
    max_depth=20,
)
```

For an intentionally unconstrained root:

```python
parsed = parse_class_spec(
    value,
    expected_type=None,
    strict=True,
)
```

Required behavior:

- reject malformed recipe keys;
- reject an unimportable class path;
- reject unknown constructor arguments;
- recursively validate typed nested classes;
- never silently return an invalid recipe unchanged;
- separate validation from instantiation.

Question:

> Is there a public supported way to parse and instantiate a standalone strict
> `class_path` / `init_args` recipe without creating an artificial root parser
> argument and without using `type=Any` fallback behavior?

### 2. Import policy hook

Desired API shape:

```python
parser = ArgumentParser(
    class_path_policy=policy,
)
```

or:

```python
with class_path_policy(policy):
    parsed = parser.parse_object(data)
```

The callback runs before importing each class path.

Question:

> Can applications intercept or authorize every `class_path` before import
> using a public API?

### 3. Configurable recursion limit

Desired API shape:

```python
parser = ArgumentParser(max_config_depth=20)
```

The limit should cover:

- nested mappings and sequences;
- nested class specs;
- sub-config files;
- parser instantiation recursion.

Question:

> Is there a supported recursion limit for nested values and class specs, beyond
> loop detection for recursively included config files?

### 4. Scoped custom instantiators

jsonargparse 4.50 exposes global `add_instantiator`. Multiple libraries in one
process can register handlers for related base classes.

Desired API shape:

```python
parser = ArgumentParser()
parser.set_instantiator(
    Component,
    instantiate_component,
    subclasses=True,
)
```

or a context manager:

```python
with instantiators({Component: instantiate_component}):
    objects = parser.instantiate(cfg)
```

Question:

> What is the recommended public API for library-scoped custom instantiators
> after `ArgumentParser.add_instantiator` deprecation?

### 5. Strict `Any` behavior

Desired opt-in behavior:

```python
parser.add_argument(
    "--component",
    type=Any,
    strict_class_specs=True,
)
```

Question:

> Can the current `adapt_classes_any` fallback be made strict through a public
> option so malformed class specs raise instead of remaining plain mappings?

### 6. Retaining source configuration with instances

The preferred solution is configuration-first. Still ask whether jsonargparse
has a supported pattern to return instantiated values together with the parsed
subtree that created each value.

Desired shape:

```python
initialized = parser.instantiate(
    cfg,
    retain_config=True,
)

initialized.component.value
initialized.component.config
```

This is more reliable than reconstructing config from live objects.

## Migration plan

### Phase 0: pin behavior and add conformance tests

Raise the minimum jsonargparse version to the selected baseline:

```toml
jsonargparse[signatures,shtab] >= 4.50, < 5
```

A `<5` upper bound is appropriate while using behavior affected by announced
v5 changes. Remove the upper bound after testing and adapting to v5.

Add package-neutral tests for:

```text
parse dict
parse YAML
CLI override precedence
unknown key rejection
nested subclass construction
protocol construction
list and mapping of components
omitted defaults
explicit null
enum and path round-trip
constructor failure
unimportable class path
forbidden class path
maximum depth
cyclic Python mapping
alias resolution
artifact containment
process-boundary recipe pass-through
```

### Phase 1: replace generic loading

Replace all Runtime call sites of `physicalai.config.loading.instantiate_obj`
with parser-based known-type construction. Keep `instantiate_obj` exported for
downstream users, but do not add new internal call sites.

Before:

```python
instantiate_obj(config, target_cls=Model)
```

After:

```python
parse_class_config(Model, config)
```

For a config containing `class_path`, require an expected base where possible:

```python
parse_component(ModelBase, spec)
```

Keep the old function as a compatibility facade for downstream users:

```python
def instantiate_obj(config, *, target_cls=None, key=None):
    warn_deprecated()
    selected = select_key(config, key)

    if target_cls is not None:
        return parse_class_config(target_cls, selected)

    return strict_recipe_adapter(selected).instantiate()
```

### Phase 2: replace custom FromConfig parsing

Replace all Runtime call sites and new class declarations using the custom
`FromConfig` implementation. Keep the existing public `FromConfig` object and
its name for downstream compatibility. Runtime code should no longer use it
internally. New classes should use `jsonargparse.FromConfigMixin` directly, or
use a package parser builder when the class is part of a larger workflow.

During the compatibility period, the existing `FromConfig` can delegate to
jsonargparse:

```python
class FromConfig(FromConfigMixin):
    @classmethod
    def from_pydantic(cls, config):
        return cls.from_config(config.model_dump())
```

The existing `FromConfig` may preserve compatibility methods such as
`from_pydantic` and `from_dataclass`, but delegates construction to
jsonargparse. Delete custom recursive instantiation from it. Once downstream
users no longer require the compatibility methods, the project can replace the
implementation with `jsonargparse.FromConfigMixin` or remove the Runtime
facade entirely where no public compatibility is needed.

### Phase 3: remove duplicate typed utilities

Keep typed configuration classes compatible with PR #212:

```python
@dataclass
class ACTConfig(Config):
    hidden_size: int = 256
```

`Config.load`, `Config.from_dict`, and `Config.save` now delegate typed parsing,
validation, construction, and serialization to jsonargparse. Remove the old
typed reconstruction and serialization implementations only after their direct
imports are no longer required by downstream users.

### Phase 4: converge inference ComponentSpec

1. Split alias and class-path forms internally.
2. Resolve aliases and artifact paths.
3. Produce canonical class recipes.
4. Parse recipes against known component protocols or base classes.
5. Use `parser.instantiate`.
6. Keep `instantiate_component(base, spec)` as a thin jsonargparse convenience
   function; delete the inference-specific recursive constructor.

### Phase 5: move to configuration-first object graphs

New APIs parse and instantiate at the application boundary. The caller retains
the parsed `Namespace` when replay, inspection, or persistence is required:

```python
parser = build_runtime_parser()
parsed = parser.parse_object(document, defaults=False)
runtime = parser.instantiate(parsed).runtime
save(parsed)
```

Keep `@export_config` for existing imperative APIs during a deprecation period.
Do not require it for new components.

### Phase 6: remove duplicate internals

Candidate removals after compatibility ends:

```text
physicalai/config/loading.py custom construction branches
physicalai/config/mixin.py custom parsing branches
physicalai/config/serializable.py dict_to_dataclass and typed reconstruction
physicalai/inference/_importing.py
physicalai/inference/component_factory.py recursive constructor logic
physicalai/robot/transport/_importing.py
```

Retained package-specific code:

```text
manifest models
component alias registries
artifact resolution
security preflight
transport envelope schemas
compatibility export capture, if still required
```

## Compatibility strategy

### Target ownership

The target Runtime configuration package keeps only code with a distinct
responsibility:

```text
physicalai/config/
  __init__.py       public API
  base.py           Config API and jsonargparse typed bridge
  _export.py        @export_config/from_instance compatibility
  _instantiate.py   strict schema-free recipe compatibility
  _normalize.py    portable recipe and safety validation
  _envelope.py     transport envelope policy
  _errors.py        public error taxonomy
  _types.py         public JSON/recipe types
  importing.py      public dotted-path compatibility
```

`loading.py`, `mixin.py`, `serializable.py`, and `_yaml.py` may remain as
deprecated import-compatible facades, but new Runtime code must not depend on
their custom parsing or reconstruction logic. Their implementations should
delegate to `Config` and jsonargparse, then be removed after the compatibility
period.

`_instantiate.py` is not a second typed configuration engine. It exists only
for schema-free trusted recipes, transport pass-through arguments, and existing
`Config(class_path, init_args).instantiate()` callers. New typed or component
construction must supply an expected class/base to jsonargparse.

The public API can remain stable during migration:

```python
from physicalai.config import Config
from physicalai.config import FromConfig
from physicalai.config import instantiate
from physicalai.config import instantiate_obj
```

Each public function delegates to the new engine. Internal module imports are
not preserved unless a downstream package requires them.

Do not preserve multiple semantics under the same name. Document whether each
entry point expects:

- a flat known-class config;
- a subclass recipe;
- a complete workflow document;
- a trusted arbitrary recipe.

## Testing strategy

### Unit tests

Test parser builders with minimal local classes:

```python
class Child:
    def __init__(self, value: int):
        self.value = value


class Parent:
    def __init__(self, child: Child):
        self.child = child
```

```python
def test_nested_component():
    parser = build_parser(Parent)
    parsed = parser.parse_object({
        "root": {
            "child": {
                "class_path": path(Child),
                "init_args": {"value": 3},
            },
        },
    })

    parent = parser.instantiate(parsed).root
    assert parent.child.value == 3
```

### Conformance tests

Run one shared suite against each package's parser builder:

```python
@pytest.mark.parametrize(
    "build_parser, fixture",
    PACKAGE_CONFIG_CASES,
)
def test_config_conformance(build_parser, fixture):
    parser = build_parser()
    parsed = parser.parse_object(fixture.valid)
    parser.instantiate(parsed)
```

### Cross-package tests

Verify a recipe produced by one package is consumed by another without importing
producer-only code:

```text
training export fixture
    -> JSON manifest
    -> clean runtime environment
    -> manifest parse
    -> component parse
    -> object construction
```

### Version tests

Run the conformance suite against:

```text
minimum supported jsonargparse
latest supported jsonargparse
next major prerelease when available
```

## Decision points

The following decisions should be made before implementation:

1. Is a separate shared configuration distribution acceptable after a second
   package adopts the helpers?
2. Which inputs permit arbitrary installed `class_path` values?
3. Which component interfaces are concrete base classes and which are
   protocols?
4. What maximum depth covers real workflows?
5. Must direct object-to-config export remain a permanent public feature?
6. Should persisted configs materialize constructor defaults or preserve
   omission?
7. Are explicit registry aliases required in every external manifest?
8. What is the minimum jsonargparse version across all consumers?

## Recommended decision

Adopt jsonargparse as the primary parse, validation, serialization, and
instantiation engine for schema-aware configuration.

Keep a small strict boundary for portable class recipes until jsonargparse has a
public strict standalone recipe API, import policy hook, and recursion limit.

Use explicit package registries for aliases. Use Pydantic for complete manifest
metadata. Move toward configuration-first construction and retain parsed
configuration instead of reconstructing recipes from live objects.

Do not create another broad configuration framework around jsonargparse. Any
shared wrapper should contain policy and safety adapters only, and should be
extracted after at least two packages require the same implementation.

## References

- jsonargparse repository: <https://github.com/mauvilsa/jsonargparse>
- jsonargparse 4.50 configuration files:
  <https://jsonargparse.readthedocs.io/en/v4.50.0/#configuration-files>
- jsonargparse class and subclass configuration:
  <https://jsonargparse.readthedocs.io/en/v4.50.0/#class-type-and-subclasses>
- jsonargparse `FromConfigMixin`:
  <https://jsonargparse.readthedocs.io/en/v4.50.0/#from-config-mixin>
- PhysicalAI configuration explanation: `docs/explanation/configuration.md`
- PhysicalAI configuration schema: `docs/reference/config-schema.md`
- PhysicalAI inference manifest explanation: `docs/explanation/manifests.md`
- PhysicalAI runtime security rules: `docs/development/security.md`
