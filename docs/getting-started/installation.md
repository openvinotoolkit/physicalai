# Installation

Install the runtime package first.

```bash
pip install physicalai
```

Install hardware extras only when you need a specific backend or robot integration.

```bash
pip install "physicalai[realsense]"
pip install "physicalai[so101]"
pip install "physicalai[robots]"
pip install "physicalai[capture]"  # all camera backends + IPC transport
```

For local development, install the repository environment and run the test suite.

```bash
uv sync
uv run pytest
```

## Package Boundary

`physicalai` is the runtime package. It should remain usable on deployment hosts that do not have training dependencies such as Torch or Lightning installed.

Training commands may be provided by a separate training distribution or by plugin entry points.

When installed together with the coordinated `physicalai-train` release, the
shared `physicalai` executable also exposes training and benchmark subcommands.

## OpenVINO GPU (optional)

The bundled OpenVINO runtime can execute models on the CPU out of the box. To
target an Intel GPU (`device: GPU` or `AUTO` in your inference config) the host
needs the OpenCL runtime and Intel compute driver, which are system packages
that `pip` cannot install:

```bash
# Debian / Ubuntu
sudo apt install ocl-icd-libopencl1 intel-opencl-icd

# allow your user to access the GPU device (log out / back in after this)
sudo usermod -aG render $USER

# verify a GPU device is visible to OpenCL
clinfo -l
```

Without these, loading the GPU plugin fails with either
`libOpenCL.so.1: cannot open shared object file` (loader missing) or
`[GPU] Can't get PERFORMANCE_HINT property as no supported devices found`
(driver missing or device not accessible). CPU inference is unaffected.

## Check the Install

```python
import physicalai
from physicalai.inference import InferenceModel
```

If hardware extras are not installed, importing the base runtime should still work.
