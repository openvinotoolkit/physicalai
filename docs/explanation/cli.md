# CLI

The CLI is a thin wrapper over the same config APIs used by Python.

```bash
physicalai run --config runtime.yaml --run.duration_s=60
```

Equivalent Python:

```python
PolicyRuntime.from_config("runtime.yaml").run(duration_s=60)
```

## Runtime Commands

| Command          | Purpose                          |
| ---------------- | -------------------------------- |
| `physicalai run` | Runs a policy on robot hardware. |

## Training Commands

Training commands should come from training packages or entry-point plugins.

```toml
[project.entry-points."physicalai.cli.subcommands"]
fit = "physicalai.cli.fit:register"
benchmark = "physicalai.cli.benchmark:register"
```

Importing `physicalai` should not pull in training dependencies.
