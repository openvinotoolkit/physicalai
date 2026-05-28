# CLI: Infer

Offline inference does not have a dedicated CLI subcommand yet.

Use the Python API when you need to load an exported policy package and run
inference outside a robot control loop.

```python
from physicalai.inference import InferenceModel

model = InferenceModel.load("./exports/act_policy")
action = model.select_action(observation)
```

Use `physicalai run` when you need the runtime-managed robot control loop.
