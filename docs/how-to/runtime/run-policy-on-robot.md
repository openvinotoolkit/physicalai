# Run a Policy on a Robot

## Python API

```python
from physicalai.runtime import RobotRuntime, PolicySource, SyncExecution
from physicalai.inference import InferenceModel
from physicalai.robot import SO101
from physicalai.capture import UVCCamera

runtime = RobotRuntime(
    fps=30,
    robot=SO101(port="/dev/ttyACM0"),
    action_source=PolicySource(
        model=InferenceModel("./exports/act_policy"),
    ),
    cameras={
        "wrist": UVCCamera(device="/dev/video0", width=640, height=480),
    },
)

with runtime:
    runtime.run(duration_s=60)
```

## From Config

Write a runtime configuration file.

```yaml
# runtime.yaml
runtime:
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
      execution:
        class_path: physicalai.runtime.SyncExecution
  cameras:
    wrist:
      class_path: physicalai.capture.UVCCamera
      init_args:
        device: /dev/video0
        width: 640
        height: 480
  fps: 30
```

Run it from the CLI.

```bash
physicalai run --config runtime.yaml --run.duration_s=60
```

## Component Responsibilities

| Object           | Owns                                                  |
| ---------------- | ----------------------------------------------------- |
| `InferenceModel` | policy inference                                      |
| `PolicySource`   | action source wiring model + execution + action queue |
| `RobotRuntime`   | robot loop and timing                                 |
| `Execution`      | where inference runs                                  |
| `Robot`          | hardware IO                                           |
| `Camera`         | image capture                                         |
