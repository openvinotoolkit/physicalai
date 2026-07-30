# Use FromConfig and @from_config

`FromConfig` and `@from_config` add class-level helpers on top of
`instantiate_obj`. Physical AI Studio policies and other Lightning-facing types
use them so YAML, dict, Pydantic, and dataclass inputs share one construction
path.

## Mixin and decorator

```python
from physicalai.config import FromConfig, from_config

class MyModel(FromConfig):
    def __init__(self, hidden_size: int, num_layers: int = 3) -> None:
        ...

@from_config
class LegacyModel:
    def __init__(self, hidden_size: int) -> None:
        ...
```

Both expose:

- `from_yaml(path, *, key=None)`
- `from_dict(config, *, key=None)`
- `from_pydantic(config, *, key=None, recursive=False)`
- `from_dataclass(config, *, key=None, recursive=False)`
- `from_config(config, *, key=None, recursive=False)` — dispatches by input type

Use the decorator when you cannot change the class inheritance.

## Direct kwargs vs `class_path`

### Direct kwargs

```python
model = MyModel.from_dict({"hidden_size": 256, "num_layers": 4})
```

### `class_path` / `init_args`

```python
model = MyModel.from_dict({
    "class_path": "mypkg.models.BigModel",
    "init_args": {"hidden_size": 512},
})
```

This matches the shape used by `jsonargparse` and Lightning-style CLIs.

## Nested instantiation

Nested `class_path` blocks are instantiated before the constructor runs:

```python
policy = Policy.from_dict({
    "model": {
        "class_path": "mypkg.models.Backbone",
        "init_args": {"hidden_size": 256},
    },
})
```

## Dataclass and Pydantic sources

```python
model = MyModel.from_dataclass(ModelConfig())
model = MyModel.from_pydantic(ModelConfig())
```

### `recursive` flag

With `recursive=False` (default), nested dataclass or Pydantic instances are
passed through to the constructor unchanged. With `recursive=True`, nested
objects are flattened to plain dicts first — use that when `__init__` expects
mappings, not nested config objects.

## Typed policy configs

Studio policy families define `<Name>Config(Config)` dataclasses in
`config.py` and construct policies via `from_config` or Lightning CLI
`class_path` wiring. The shared `Config` base lives in Runtime; see
[`Instantiate Components`](instantiate-components.md) for export and strict
recipes.

## Validation

Validation belongs in Pydantic models, dataclass `__post_init__`, or
constructors — not in the instantiator. Import and constructor errors surface
as `ConfigError`, `ConfigImportError`, or exceptions from the target class.

## Related

- [`instantiate_obj`](instantiate-objects.md) — backend without mixins
- [`Configuration`](../../explanation/configuration.md) — recipes, export, trust
- Implementation: `src/physicalai/config/mixin.py`
