# Instantiate Objects from Config

`physicalai.config.instantiate_obj()` is the generic entry point for turning
config-like inputs into Python objects. Training CLIs, Lightning integration,
and infrastructure code use it when the concrete class may come from YAML or
when you do not want a `FromConfig` mixin on the target type.

## Main API

```python
from physicalai.config import instantiate_obj

obj = instantiate_obj(config, *, key=None, target_cls=None)
```

`config` may be:

- `dict`
- YAML or JSON file path (`str` or `pathlib.Path`)
- dataclass instance
- Pydantic `BaseModel`

All paths converge on `instantiate_obj_from_dict`, which:

1. Applies `key` to select a nested sub-config when provided.
2. Instantiates via `class_path` when present (imports and calls the class).
3. Otherwise calls `target_cls(**config)` when `target_cls` is provided.
4. Recursively instantiates nested `class_path` blocks inside dicts, lists, and tuples.

If both `class_path` and `target_cls` are present, `class_path` wins.

## Examples

### Import from `class_path`

```python
optimizer = instantiate_obj({
    "class_path": "torch.optim.Adam",
    "init_args": {"lr": 1e-3},
})
```

### Direct kwargs via `target_cls`

```python
model = instantiate_obj(
    {"hidden_size": 256, "num_layers": 3},
    target_cls=MyModel,
)
```

### YAML or JSON file

```python
policy = instantiate_obj("config.yaml")
model = instantiate_obj("train.json", key="model")
```

### Dataclass or Pydantic source

```python
model = instantiate_obj(dataclass_cfg, target_cls=MyModel)
model = instantiate_obj(pydantic_cfg, target_cls=MyModel)
```

### Nested components

```python
policy = instantiate_obj({
    "class_path": "mypkg.Policy",
    "init_args": {
        "model": {
            "class_path": "mypkg.Model",
            "init_args": {"hidden_size": 256},
        },
    },
})
```

## When to use it

Prefer `instantiate_obj` when:

- the config decides the concrete class;
- the target class should not inherit `FromConfig`; or
- you are writing generic loaders (trainer wiring, plugins, tests).

For strict captured construction recipes (`Config.from_instance`, transport
export), use [`Instantiate Components`](instantiate-components.md) and
[`Configuration`](../../explanation/configuration.md) instead.

## Trust

`class_path` imports and executes local Python code. Pass only trusted
application or user-authored configuration. See
[`Configuration`](../../explanation/configuration.md#trust).

## Related

- Implementation: `src/physicalai/config/instantiate.py`
- Tests: `tests/unit/config/`
