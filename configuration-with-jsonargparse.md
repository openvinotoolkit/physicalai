# Implementing configuration with jsonargparse

Status: implementation specification

## Decision

Use jsonargparse as the construction engine whenever the expected Python type is
known.

This applies to:

- typed `Config` subclasses retained for compatibility;
- normal classes constructed from mappings or files;
- CLI and workflow configuration;
- components with a known base class or protocol;
- nested typed objects described by constructor annotations.

Runtime retains custom code only for behavior that jsonargparse does not
provide:

- portable `class_path` / `init_args` validation before imports;
- cycle, depth, finite-float, and JSON-value checks;
- live-object constructor capture through `@export_config`;
- transport envelopes that must be validated without importing drivers;
- explicit aliases, artifact paths, and import policy;
- temporary schema-free recipe instantiation for backward compatibility.

Do not add another general parsing, signature inspection, serialization, or
recursive construction engine around jsonargparse.

## Construction lanes

### Known type

```text
mapping / file / CLI
        |
        v
package-owned jsonargparse parser
        |
        v
parse and validate
        |
        v
parser.instantiate()
        |
        v
typed object graph
```

An expected type is mandatory for normal construction.

```python
parser = ArgumentParser(exit_on_error=False)
parser.add_class_arguments(Client, "client")

parsed = parser.parse_object(
    {"client": {"host": "api.local", "port": 8080}},
    defaults=False,
)
client = parser.instantiate(parsed).client
```

### Portable recipe

```text
class_path / init_args
        |
        v
portable preflight without imports
        |
        +--> expected type available --> jsonargparse
        |
        +--> no expected type --------> legacy compatibility path
```

The canonical recipe remains:

```yaml
class_path: package.module.ClassName
init_args:
  option: value
```

New Runtime code must provide an expected type. Schema-free construction exists
only for shipped public compatibility.

## Public API contract

### `Config`

PR #212 establishes Runtime ownership of `Config`. During migration it has two
roles:

1. direct portable recipe;
2. base class for typed dataclass configs used by downstream Studio releases.

Do not expand this dual role. The steady-state `Config` is recipe-only.

#### Portable recipe API

```python
Config(
    class_path: str,
    init_args: Mapping[str, JsonValue] | None = None,
)

Config.from_dict(data) -> Config
Config.from_instance(instance) -> Config
Config.load(path) -> Config

config.to_dict() -> ClassSpec
config.save(path) -> None
config.instantiate(expected_type: type[T] | None = None) -> T | object
```

Required behavior:

- validate the entire recipe before any import;
- reject unknown recipe keys;
- require a non-empty dotted `class_path`;
- require string mapping keys and finite JSON values;
- reject cycles and excessive depth;
- defensively copy external mappings;
- revalidate before save and instantiate;
- preserve constructor exceptions and add recipe-path context;
- use jsonargparse when `expected_type` is supplied;
- isolate schema-free construction as deprecated compatibility.

#### Typed compatibility API

Existing downstream code may continue temporarily:

```python
@dataclass
class ACTConfig(Config):
    hidden_size: int = 256


config = ACTConfig.load("config.yaml")
config = ACTConfig.load({"hidden_size": 512})
```

The implementation must delegate parsing, validation, construction, dumping,
and saving to jsonargparse. It must not retain a separate dataclass
reconstruction engine.

Typed compatibility requirements:

- preserve PR #212's supported file and mapping shapes;
- reject unknown fields before constructor execution;
- preserve enum/path/tuple behavior or perform an explicit documented wire
  migration;
- validate an envelope's `class_path` against the requested typed class;
- keep omitted defaults and explicit null behavior tested;
- coordinate removal with a Studio release;
- deprecate `strict=False` rather than pretending jsonargparse is permissive.

### Expected-type component construction

jsonargparse already provides the required engine:

```python
def instantiate_component(base: type[T], spec: ClassSpec) -> T:
    validate_portable_recipe(spec)

    parser = ArgumentParser(exit_on_error=False)
    parser.add_subclass_arguments(base, "component", required=True)
    parsed = parser.parse_object(
        {"component": dict(spec)},
        defaults=False,
    )
    return cast(T, parser.instantiate(parsed).component)
```

This helper is optional convenience, not a framework. It must not implement
imports, signature parsing, recursive constructor calls, alias lookup, artifact
resolution, or registry discovery.

Package code performs policy before calling it:

```python
canonical = resolve_alias(spec, registry)
canonical = resolve_artifacts(canonical, export_dir)
processor = instantiate_component(Preprocessor, canonical)
```

The expected base is required in new code:

```python
instantiate_component(Camera, recipe)
instantiate_component(Robot, recipe)
instantiate_component(InferenceRunner, recipe)
instantiate_component(Preprocessor, recipe)
instantiate_component(Postprocessor, recipe)
```

## Compatibility policy

Backward compatibility does not make an API canonical.

The following APIs and modules may remain temporarily for downstream users:

- `instantiate_obj` and `instantiate_obj_from_*`;
- `FromConfig` and `@from_config`;
- `load_yaml`, `save_yaml`, and `to_yaml`;
- typed `Config` subclasses;
- `import_class` and older module import paths;
- schema-free `instantiate`.

Runtime production code must not use these compatibility APIs after migration.
Replace internal usage with:

- package-owned parser builders;
- direct jsonargparse parsing for known classes;
- expected-type component construction;
- `Config.from_dict()` for portable recipe validation;
- domain-owned transport or manifest policy.

Compatibility facades must delegate to the canonical engine. They must not:

- inspect constructor signatures themselves;
- recursively instantiate mappings themselves;
- directly call `cls(**values)` or `target_cls(*args, **kwargs)`;
- define a second default or unknown-key policy;
- silently ignore `class_path`;
- accept inputs that canonical APIs reject without explicit deprecation tests.

Use `DeprecationWarning` with a correct stack level for APIs scheduled for
removal.

## Target file ownership

### Retained core

```text
src/physicalai/config/
  __init__.py       documented public surface
  base.py           Config recipe and temporary typed compatibility
  _export.py        @export_config and Config.from_instance
  _instantiate.py   expected-type path plus schema-free compatibility
  _normalize.py     portable JSON recipe normalization and preflight
  _envelope.py      package-neutral no-import envelope validation
  _errors.py        public error taxonomy
  _types.py         ClassSpec, JsonValue, ConfigValue
  importing.py      temporary public import compatibility
```

### Removed utility facades

The implementation removes these modules rather than preserving low-value
utility compatibility:

```text
loading.py
mixin.py
serializable.py
_yaml.py
_envelope.py
```

Expected disposition:

| Current code                                | Action                                                                |
| ------------------------------------------- | --------------------------------------------------------------------- |
| `instantiate_obj`                           | Removed; use jsonargparse or `Config`                                 |
| `instantiate_obj_from_*`                    | Removed                                                               |
| `_instantiate_recursive`                    | Delete                                                                |
| positional `args` convention                | Deprecate and remove                                                  |
| flattened arbitrary `**kwargs` construction | Deprecate or explicitly translate; do not inspect signatures manually |
| `FromConfig`                                | Removed; inherit `jsonargparse.FromConfigMixin` directly              |
| `@from_config`                              | Removed                                                               |
| `dict_to_dataclass`                         | Delete immediately                                                    |
| `dataclass_to_dict`                         | Retain only for legacy wire/checkpoint compatibility                  |
| `_yaml.py`                                  | Removed; use `Config.load/save`                                       |
| `import_class`                              | Deprecated wrapper only                                               |
| inference/robot import shims                | Delete                                                                |

### Domain ownership

These concerns do not belong in generic config helpers:

- inference aliases and artifacts: `physicalai.inference`;
- runtime workflow parser: `physicalai.runtime`;
- camera publisher envelope: `physicalai.capture.transport`;
- robot owner envelope: `physicalai.robot.transport`;
- trust policy: the package accepting the external input.

## Required migrations

### Runtime parser

Create one package-owned runtime parser builder and use it from both:

- `physicalai run`;
- `RobotRuntime.from_config()`.

It owns:

- `RobotRuntime` class arguments;
- `run()` method arguments;
- config-file and CLI merge behavior;
- bare exported Runtime recipe reshaping;
- defaults and dump/save policy.

### Inference

`component_factory.py` retains only:

- explicit alias registry;
- artifact containment/resolution;
- canonical recipe production;
- thin expected-type jsonargparse call.

Remove its recursive constructor and require expected bases internally.
Nested aliases must be resolved before jsonargparse.

### Robot and camera transport

Parent/subscriber processes validate recipes as data without importing drivers.
Only the hardware-owning child instantiates:

```python
robot = recipe.instantiate(expected_type=Robot)
camera = recipe.instantiate(expected_type=Camera)
```

Keep protocol checks after construction as defense in depth.

Transport control or peer metadata must never introduce or modify class paths.

### Internal compatibility removal

After migration, no production module under `src/physicalai/` may import:

```text
physicalai.config.loading
physicalai.config.mixin
physicalai.config.serializable
```

Imports within compatibility modules themselves are allowed.

## Portable recipe safety

Portable validation runs before jsonargparse because jsonargparse imports class
paths while parsing.

Required checks:

1. root and nested recipe shape;
2. non-empty dotted class path;
3. optional `init_args`, normalized to an empty mapping;
4. string mapping keys;
5. JSON-compatible values;
6. finite floats;
7. cycle rejection;
8. maximum depth;
9. package class-path policy before imports.

Validation itself must not import configured classes.

Do not use `type=Any` as a strict recipe boundary. jsonargparse can leave an
invalid class mapping unchanged under `Any`.

## Serialization policy

The parser that owns a typed schema also owns typed dump/save behavior.

Do not implement a separate generic serializer for:

- dataclass reconstruction;
- enum conversion;
- path conversion;
- tuple conversion;
- nested typed models.

Portable recipe serialization remains stricter because it is a stable JSON
wire format.

During compatibility, preserve existing documents or provide an explicit
migration. In particular, test:

- enum names versus enum values;
- bare field mappings versus `class_path` envelopes;
- omitted defaults;
- explicit null;
- NumPy/checkpoint values;
- YAML and JSON extension behavior.

Never silently reinterpret or drop fields.

## Error policy

Public compatibility APIs preserve documented exception classes where
practical. Internal canonical APIs expose jsonargparse parse failures with clear
field context.

Requirements:

- parse errors occur before constructor execution;
- wrong subclass fails before construction;
- constructor exceptions keep their original exception type;
- portable recipe errors use `ConfigError` / `ConfigImportError`;
- compatibility adapters do not swallow or rewrite useful jsonargparse context.

## Implementation phases

### Phase 0: conformance tests

Before deleting code, add tests for:

- dictionary and file parsing;
- unknown keys;
- invalid types;
- nested exact classes;
- nested subclass recipes;
- protocol construction;
- list and mapping components;
- dataclasses and Pydantic models;
- enums, paths, tuples, and explicit null;
- omitted defaults;
- parser dump/save round trip;
- wrong or unimportable class paths;
- depth, cycles, and no-import preflight;
- old persisted Config documents;
- Studio typed Config compatibility.

### Phase 1: expected-type recipe construction

1. Add `expected_type` to `Config.instantiate()` and package `instantiate()`.
2. Implement this path with jsonargparse only.
3. Migrate robot, camera, runner, processor, and feature callers.
4. Keep schema-free construction isolated and marked compatibility-only.

### Phase 2: parser ownership

1. Add the shared runtime parser builder.
2. Reuse it in CLI and `RobotRuntime.from_config()`.
3. Establish one defaults and serialization policy.
4. Remove duplicate config reshaping and source dispatch.

### Phase 3: compatibility removal

1. Remove `FromConfig` and direct users use `FromConfigMixin`.
2. Remove `instantiate_obj`; known types use jsonargparse directly.
3. Delete direct constructor and recursive helper branches.
4. Remove utility-only documentation and tests.

### Phase 4: downstream migration

1. Migrate Studio typed configs to plain dataclasses or Pydantic models.
2. Use Studio-owned parsers for training/application workflows.
3. Release compatible Runtime and Studio versions in order.

### Phase 5: remove facades

In the next breaking release, remove:

- `loading.py`;
- `mixin.py`;
- `serializable.py`;
- `_yaml.py`;
- typed `Config` subclass behavior;
- mapping emulation on `Config`;
- schema-free generic loading APIs;
- obsolete import shims.

## Acceptance criteria

### Architecture

- all known-type construction uses jsonargparse;
- no production compatibility facade directly invokes constructors;
- no custom signature parser remains;
- CLI and Python workflow APIs share package parser builders;
- `_instantiate.py` has no new internal schema-free callers;
- compatibility modules are unused by production code.

### Correctness

- unknown fields fail before constructor execution;
- `class_path` is validated against the expected type;
- nested construction follows annotations;
- existing portable recipe YAML remains readable;
- old typed Config documents remain readable during migration;
- save/load preserves enum, path, tuple, null, and default behavior;
- malformed recipes trigger no configured imports;
- depth and cycles fail deterministically;
- transport recipes cross process boundaries unchanged.

### Quality gates

```text
uv run pytest
prek run --all-files
```

Also run config conformance tests against the minimum and latest supported
jsonargparse 4.x versions and a supported Studio release.

## Definition of done

This migration is complete when:

1. jsonargparse is the only typed parsing and construction engine;
2. Runtime production code no longer uses `instantiate_obj`, custom
   `FromConfig`, or typed reconstruction helpers;
3. strict portable recipe/export/transport code is clearly separated from
   generic typed configuration;
4. compatibility facades delegate to canonical APIs and are covered by
   deprecation tests;
5. the documented target file ownership matches the source tree;
6. all tests and quality hooks pass.
