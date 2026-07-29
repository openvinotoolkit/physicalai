# Tutorials

Collection of notebooks that walk users through various end-to-end workflows. The notebooks here provide workflows for developers to get started with data collection, physical AI model fine tuning and testing the OpenVINO™ Physical AI APIs for deploying the trained policies.


## Getting Started

Install the shared environment before starting any notebook. On Ubuntu 24.04,
install the native build tools required by LeRobot and the LIBERO benchmark:

```bash
sudo apt update
sudo apt install -y build-essential libegl1 libgl1 python3-venv
```

Create the tutorial environment and install the shared Python requirements:

```bash
git clone https://github.com/openvinotoolkit/physicalai.git
cd physicalai/examples/tutorials
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --force-reinstall "cmake==3.31.10" ninja
python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
```

Verify the environment manually before launching JupyterLab:

```bash
python -c "import torch, lerobot, openvino; print(torch.__version__, openvino.__version__)"
python -c "import cmake; print(cmake.__version__)"
which cmake
cmake --version
c++ --version
export MUJOCO_GL=egl
jupyter lab
```

Individual notebooks may install a small number of workflow-specific packages
in their first code cell. The CMake pin above is required by the LeRobot version
used by the Physical AI Studio LIBERO benchmark. CMake 4 removes the legacy
policy compatibility required by `egl_probe==1.0.2`; CMake 3.31 still supports
it. The simulation notebook builds that probe without pip build isolation so it
can use this CMake installation.

On Linux, `MUJOCO_GL=egl` selects MuJoCo's hardware-accelerated headless
renderer. Set it before starting JupyterLab because the rendering backend is
chosen when MuJoCo is first imported.

### List of notebooks:

| **Notebook** | **Description** |
|:-------------|:----------------|
| [001_Introduction](001_Introduction.ipynb) | Introduction to different pathways to test OpenVINO Physical AI |
| [002_Using_Physical_AI_Studio](002_Using_Physical_AI_Studio.ipynb) | Utilizing Physical AI Studio for full workflow with built-in OpenVINO Physical AI API, with a physical robot |
| [003_OpenVINO_Optimization](003_OpenVINO_Optimization.ipynb) | Bring model from Physical AI Studio or Lerobot, optimize with OpenVINO, and deploy using OpenVINO Physical AI API |
| [004_Test_Deployment_Without_Robot](004_Test_Deployment_Without_Robot.ipynb) | Test deployment on a subset of dataset, without a physical robot |
| [005_Collect_Train_Deploy_SO101](005_collect_train_deploy.ipynb) | Workflow for data collection, model fine tuning, and deployment with SO101 arms and π0.5 visuomotor diffusion policy |


### Related documentation

- [Run a Policy on a Robot](../../docs/how-to/runtime/run-policy-on-robot.md)
- [Load an Exported Policy](../../docs/how-to/inference/load-exported-policy.md)
- [Robot API reference](../../docs/reference/robot-api.md)
