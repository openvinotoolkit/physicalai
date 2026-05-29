# CLI Reference

The runtime CLI commands use the same schemas as the Python APIs.

## `physicalai run`

```bash
physicalai run --config runtime.yaml [--run.duration_s=60]
```

Arguments:

| Argument           | Required | Description                              |
| ------------------ | -------- | ---------------------------------------- |
| `--config`         | yes      | Runtime config YAML                      |
| `--run.duration_s` | no       | Stop after the given duration in seconds |

The equivalent Python call is shown below.

```python
PolicyRuntime.from_config("runtime.yaml").run(duration_s=60)
```

## Plugin Commands

Training packages can add commands through entry points.

```toml
[project.entry-points."physicalai.cli.subcommands"]
fit = "physicalai.cli.fit:register"
validate = "physicalai.cli.validate:register"
test = "physicalai.cli.test:register"
predict = "physicalai.cli.predict:register"
benchmark = "physicalai.cli.benchmark:register"
export = "physicalai.cli.export:register"
```
