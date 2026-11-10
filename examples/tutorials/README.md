# Tutorials

End-to-end walkthroughs that combine data collection, training, and deployment, on real hardware or in simulation.

## Collect → Train → Deploy

**Notebook:** [collect_train_deploy.ipynb](collect_train_deploy.ipynb)

Take a physical robot from zero to a working learned policy. The tutorial walks through the full pipeline:

1. **Collect** teleoperation demonstrations of a manipulation task.
2. **Train** a π0.5 visuomotor diffusion policy on the collected data.
3. **Export** the trained policy to OpenVINO IR.
4. **Deploy** the policy on the robot using the `physicalai` runtime.

### What you'll have at the end

A π0.5 policy running on an SO-101 arm at 30 fps, executing a natural-language task (e.g. _"pick up the can"_) from live camera input.

### Prerequisites

**Hardware**

- SO-101 follower arm (a leader arm is needed for teleoperated collection)
- 1–2 cameras — UVC webcam or Intel RealSense
- A GPU with **≥ 40 GB VRAM** for training

**Software**

```bash
pip install "physicalai[so101,capture]" physicalai-train
```

Optional but **recommended**: [Physical AI Studio](https://github.com/open-edge-platform/physical-ai-studio) — a web UI that wraps data collection, training, and export in a guided workflow. The tutorial works either way; Studio just removes the need to run individual scripts.

**Data**

50–100 teleoperated demonstrations of the target task, with varied object positions. The tutorial covers two collection paths:

- [Physical AI Studio](https://github.com/open-edge-platform/physical-ai-studio) (web UI, recommended)
- [LeRobot](https://github.com/huggingface/lerobot) (CLI)

### Related documentation

- [Run a Policy on a Robot](../../docs/how-to/runtime/run-policy-on-robot.md)
- [Load an Exported Policy](../../docs/how-to/inference/load-exported-policy.md)
- [Robot API reference](../../docs/reference/robot-api.md)

## Run a Policy in Simulation

**Notebook:** [run_policy_in_simulation.ipynb](run_policy_in_simulation.ipynb)

Run a trained policy without any hardware. A simulated SO-101 arm sorts items from a conveyor belt into bins by colour, driven by an ACT policy from the Hugging Face Hub. The notebook takes the same steps as on a real robot:

1. **Start** the simulation with `MuJoCoRobot`, which serves its cameras and a 3D viewer.
2. **Load** the policy with `InferenceModel.from_pretrained`.
3. **Connect** the robot, the cameras and the policy in a `RobotRuntime`.
4. **Run** it from a few buttons and watch it in the viewer, next to the cameras the policy sees.

### What you'll have at the end

An ACT policy ([iMaxxgx/act-so101-conveyor-sort](https://huggingface.co/iMaxxgx/act-so101-conveyor-sort), OpenVINO) sorting items in the MuJoCo `conveyor_sort` scene at 30 Hz, with the score shown in the viewer.

### Prerequisites

No robot, camera or GPU: it runs on a CPU, with Python 3.12 or newer. The notebook installs its packages from PyPI in its first code cell. From a checkout of this repository, skip that cell and use:

```bash
uv sync --all-packages --extra notebooks
uv run jupyter lab examples/tutorials
```

### Related documentation

- [MuJoCo plugin README](../../packages/physicalai-mujoco-plugin/README.md): robots, scenes and the viewer
- [Run a Policy on a Robot](../../docs/how-to/runtime/run-policy-on-robot.md)
- [Load an Exported Policy](../../docs/how-to/inference/load-exported-policy.md)
