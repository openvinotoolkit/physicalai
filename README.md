<p align="center">
  <img src="docs/assets/physicalai.png" alt="Physical AI" width="100%">
</p>

<div align="center">

**Runtime package for deploying robot policies trained with [Physical AI Studio](https://github.com/open-edge-platform/physical-ai-studio)**

[Installation](#installation) •
[Camera API](#camera-api) •
[Robot API](#robot-api) •
[Inference](#inference) •
[Docs](#documentation)

</div>

---

Physical AI Runtime provides the deployment-side components for running trained policies on real hardware. It handles camera capture, robot control, and policy inference with a unified API that works across different hardware vendors.

**Key Features:**

- **Unified Camera API** — Same interface for UVC, RealSense, Basler, and IP cameras
- **Robot Protocol** — Structural typing for any robot; no inheritance required
- **Inference Engine** — Load exported policies from Studio with auto-detected backends
- **Policy Runtime** — Control loop with observation building and action dispatch

---

<p align="center">
  <img src="docs/assets/inference_rerun.gif" alt="Inference demo" width="100%">
</p>

## Installation

```bash
pip install physicalai
```

With hardware-specific extras:

```bash
pip install physicalai[realsense]   # Intel RealSense cameras
pip install physicalai[basler]      # Basler industrial cameras
pip install physicalai[so101]       # SO-101 robot arm
pip install physicalai[trossen]     # Trossen WidowX robots
```

---

## Camera API

All cameras share a unified interface: `connect()`, `read()`, `read_latest()`, and context manager support. Switch hardware without changing application code.

```python
from physicalai.capture import UVCCamera

with UVCCamera(device="/dev/video0", width=640, height=480, fps=30) as camera:
    frame = camera.read_latest()
    print(frame.data.shape)  # (480, 640, 3)
    print(frame.timestamp)   # monotonic timestamp
```

<details>
<summary><strong>Intel RealSense (RGB + Depth)</strong></summary>

```python
from physicalai.capture import RealSenseCamera

with RealSenseCamera(serial_number="123456789", width=640, height=480, fps=30) as camera:
    rgb, depth = camera.read_rgbd()
    print(rgb.data.shape)    # (480, 640, 3) RGB
    print(depth.data.shape)  # (480, 640) depth in mm
```

</details>

<details>
<summary><strong>Basler Industrial Camera</strong></summary>

```python
from physicalai.capture import BaslerCamera

with BaslerCamera(serial_number="12345678", width=1920, height=1080, fps=60) as camera:
    frame = camera.read_latest()
    print(frame.data.shape)  # (1080, 1920, 3)
```

</details>

<details>
<summary><strong>Multi-Camera Sync</strong></summary>

```python
from physicalai.capture import UVCCamera, RealSenseCamera, read_cameras

cameras = {
    "wrist": UVCCamera(device="/dev/video0"),
    "overhead": RealSenseCamera(serial_number="123456789"),
}

# Connect all
for cam in cameras.values():
    cam.connect()

# Read from all cameras concurrently
synced = read_cameras(cameras)
print(synced.frames["wrist"].data.shape)
print(synced.frames["overhead"].data.shape)

# Cleanup
for cam in cameras.values():
    cam.disconnect()
```

</details>

<details>
<summary><strong>Camera Discovery</strong></summary>

```python
from physicalai.capture import discover_all, UVCCamera

# Discover all connected cameras (returns dict of camera_type -> list of devices)
all_devices = discover_all()
for camera_type, devices in all_devices.items():
    for dev in devices:
        print(f"{camera_type}: {dev.device_id} - {dev.name}")

# Discover specific type
uvc_devices = UVCCamera.discover()
```

</details>

---

## Robot API

Robots implement a Protocol-based interface. Any class with `connect()`, `disconnect()`, `get_observation()`, `send_action()`, and `joint_names` works — no inheritance required.

```python
from physicalai.robot import SO101

robot = SO101(port="/dev/ttyUSB0")
robot.connect()

obs = robot.get_observation()
print(obs.joint_positions)  # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
print(robot.joint_names)    # ['shoulder_pan', 'shoulder_lift', ...]

robot.send_action(target_positions, goal_time=0.1)
robot.disconnect()
```

<details>
<summary><strong>Trossen WidowX-AI</strong></summary>

```python
from physicalai.robot import WidowXAI

robot = WidowXAI()
robot.connect()

obs = robot.get_observation()
print(obs.joint_positions)

robot.send_action(target_positions)
robot.disconnect()
```

</details>

<details>
<summary><strong>Bimanual WidowX-AI</strong></summary>

```python
from physicalai.robot import BimanualWidowXAI

robot = BimanualWidowXAI()
robot.connect()

obs = robot.get_observation()
# Joint positions for both arms concatenated
print(obs.joint_positions.shape)

robot.send_action(bimanual_targets)
robot.disconnect()
```

</details>

<details>
<summary><strong>Robot Verification</strong></summary>

```python
from physicalai.robot import SO101, verify_robot

robot = SO101(port="/dev/ttyUSB0")
verify_robot(robot)  # Interactive joint-by-joint check
```

</details>

---

## Inference

Load exported policies from [Physical AI Studio](https://github.com/open-edge-platform/physical-ai-studio). The `InferenceModel` class auto-detects the backend (OpenVINO or ONNX in this package; companion distributions may contribute additional adapters such as ExecuTorch) and handles action chunking automatically.

```python
from physicalai.inference import InferenceModel

# Load exported policy
model = InferenceModel.load("./exports/act_policy")

# Reset state for new episode
model.reset()

# Run inference
action = model.select_action(observation)
```

<details>
<summary><strong>With Explicit Backend</strong></summary>

```python
from physicalai.inference import InferenceModel

# Force specific backend
model = InferenceModel.load(
    "./exports/act_policy",
    backend="openvino",
    device="GPU",
)
```

</details>

---

## Policy Runtime

The `PolicyRuntime` orchestrates the full control loop: connecting hardware, reading cameras, building observations, running inference, and dispatching actions to the robot.

```python
from physicalai.runtime import PolicyRuntime, SyncExecution
from physicalai.inference import InferenceModel
from physicalai.capture import UVCCamera, RealSenseCamera
from physicalai.robot import SO101

runtime = PolicyRuntime(
    fps=30,
    robot=SO101(port="/dev/ttyACM0"),
    model=InferenceModel.load("./exports/act_policy"),
    cameras={
        "wrist": UVCCamera(device="/dev/video0", width=640, height=480),
        "overhead": RealSenseCamera(serial_number="123456789"),
    },
    execution=SyncExecution(),
)

with runtime:
    runtime.run(duration_s=60)
```

<details>
<summary><strong>From YAML Config</strong></summary>

> **Preview:** This API is not yet implemented.

```python
runtime = PolicyRuntime.from_config("runtime.yaml")
runtime.run(duration_s=60)
```

```yaml
# runtime.yaml
runtime:
  class_path: physicalai.runtime.PolicyRuntime
  init_args:
    fps: 30
    robot:
      class_path: physicalai.robot.so101.SO101
      init_args:
        port: /dev/ttyACM0
    model:
      class_path: physicalai.inference.InferenceModel
      init_args:
        export_dir: ./exports/act_policy
    cameras:
      wrist:
        class_path: physicalai.capture.UVCCamera
        init_args:
          device: /dev/video0
          width: 640
          height: 480
    execution:
      class_path: physicalai.runtime.SyncExecution
      init_args:
        mode: chunk
```

</details>

<details>
<summary><strong>CLI</strong></summary>

```bash
physicalai run --config runtime.yaml --run.duration_s=60
```

The runtime package owns the shared `physicalai` executable. Training packages
can add subcommands such as `fit` and `benchmark` through the
`physicalai.cli.subcommands` entry-point group.

</details>

<details>
<summary><strong>Async Execution</strong></summary>

Async execution runs inference in a background thread while the main loop handles camera reads and robot commands at a fixed frequency. Useful when inference is slower than the control rate.

```python
from physicalai.runtime import PolicyRuntime, AsyncExecution

runtime = PolicyRuntime(
    fps=30,
    robot=robot,
    model=model,
    cameras=cameras,
    execution=AsyncExecution(fps=30),
)

with runtime:
    runtime.run(duration_s=60)
```

</details>

<details>
<summary><strong>Remote Execution</strong></summary>

Remote execution sends observations to an inference server and receives actions over the network. Useful for running large models on a separate GPU machine.

> **Preview:** This API is not yet implemented.

```python
from physicalai.runtime import PolicyRuntime, RemoteExecution

runtime = PolicyRuntime(
    fps=30,
    robot=robot,
    cameras=cameras,
    execution=RemoteExecution(endpoint="http://gpu-server:8080/infer"),
)

runtime.run(duration_s=60)
```

</details>

---

> **Full walkthrough:** See [`examples/collect_train_deploy.ipynb`](./examples/collect_train_deploy.ipynb) for a complete collect → train → deploy guide.

---

## Documentation

[Home](./docs/index.md) • [Getting Started](./docs/getting-started/) • [How-To Guides](./docs/how-to/) • [Concepts](./docs/explanation/) • [API Reference](./docs/reference/)

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md).
