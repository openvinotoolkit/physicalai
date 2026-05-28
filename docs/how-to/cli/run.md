# CLI: Run

Use this command to run a policy runtime from a YAML file.

```bash
physicalai run --config runtime.yaml
```

You can limit the run duration when needed.

```bash
physicalai run --config runtime.yaml --run.duration_s=60
```

The equivalent Python call is shown below.

```python
from physicalai.runtime import PolicyRuntime

PolicyRuntime.from_config("runtime.yaml").run(duration_s=60)
```

The CLI uses the same config schema as the Python APIs.

Runtime commands live in the `physicalai` package. Training commands should be provided by training packages or plugin entry points so the runtime package stays lightweight.
