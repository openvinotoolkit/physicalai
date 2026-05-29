# Tutorials

End-to-end walkthroughs that combine data collection, training, and deployment on real hardware.

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
