# Tutorials

Collection of notebooks that walk users through various end-to-end workflows. The notebooks here provide workflows for developers to get started with data collection, physical AI model fine tuning and testing the OpenVINO™ Physical AI APIs for deploying the trained policies.


## Getting Started

Install the shared environment before starting any notebook. On Ubuntu 24.04,
install the native build tools required by LeRobot and the LIBERO benchmark:

```bash
sudo apt update
sudo apt install -y build-essential ffmpeg libegl1 libgl1 python3-venv
```

Create the repository environment with `uv` from the repository root:

```bash
git clone https://github.com/openvinotoolkit/physicalai.git
cd physicalai
uv sync --frozen --extra notebooks
```

Register the uv-managed project environment as a dedicated Jupyter kernel:

```bash
uv run python -m ipykernel install --user \
    --env VIRTUAL_ENV "$(pwd)/.venv" \
    --name physicalai-tutorials \
    --display-name "PhysicalAI Tutorials (uv)"
```

Verify the project environment, then launch JupyterLab from the tutorials
directory and select the **PhysicalAI Tutorials (uv)** kernel:

```bash
uv run python -c "import sys, openvino, physicalai; print(sys.executable, openvino.__version__)"
c++ --version
cd examples/tutorials
export MUJOCO_GL=egl
uv run --project ../.. --with jupyter jupyter lab
```

Do not use the generic Python kernel created inside `~/.cache/uv/builds-v0` by
the temporary Jupyter environment. The notebook requests the dedicated kernel
above, whose Python executable is `<repository>/.venv/bin/python`.

Individual notebooks may install a small number of workflow-specific packages
in their first code cell. They use `uv pip --python <kernel-python>` so packages
are installed into the selected project kernel instead of a separate Jupyter
environment. The simulation notebook pins CMake 3.31.10 and builds
`hf-egl-probe==1.0.2` without build isolation because CMake 4 removes the legacy
policy compatibility required by that package. It also selects CPU-only
PyTorch wheels for the simulation dependencies; this does not limit the
OpenVINO device selected later in the notebook.

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
