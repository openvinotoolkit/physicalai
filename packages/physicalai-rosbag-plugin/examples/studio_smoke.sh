#!/usr/bin/env bash
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
work="${1:?Usage: studio_smoke.sh NEW_WORK_DIRECTORY}"
mkdir "$work"
work="$(cd "$work" && pwd)"
revision=894ea9fb9c502cfa996b6cce4195a9993301f3c9
# Use an archive and disable UV source overrides: the smoke runs against this
# checkout's Runtime without fetching Studio's Git/LFS dependency override.
python3 - "$work" "$revision" <<'PY'
import sys, tarfile, urllib.request
from pathlib import Path
work, revision = Path(sys.argv[1]), sys.argv[2]
archive = work / "studio.tar.gz"
urllib.request.urlretrieve(f"https://codeload.github.com/open-edge-platform/physical-ai-studio/tar.gz/{revision}", archive)
with tarfile.open(archive) as bundle:
    bundle.extractall(work, filter="data")
PY
studio="$work/physical-ai-studio-$revision"
uv venv --python 3.12 "$work/venv"
# The local Runtime is the 0.3 API; avoid SCM probes in archive/editable builds.
SETUPTOOLS_SCM_PRETEND_VERSION=0.3.0 uv pip install --no-sources --python "$work/venv/bin/python" \
  -e "$root" -e "$studio/library" -e "$studio/application/backend" \
  -e "$root/packages/physicalai-rosbag-plugin" \
  'torch>=2.11,<2.12' 'torchvision>=0.26,<0.27' \
  --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
uv pip check --python "$work/venv/bin/python"
uv run --no-project --python "$work/venv/bin/python" python "$root/packages/physicalai-rosbag-plugin/examples/generate_demo.py" "$work/demo"
"$work/venv/bin/physicalai" rosbag convert --input "$work/demo/bags" --mapping "$work/demo/mapping.yaml" --output "$work/demo/converted" --repo-id local/rosbag-smoke
uv run --no-project --python "$work/venv/bin/python" python "$root/packages/physicalai-rosbag-plugin/examples/studio_smoke.py" \
  --dataset "$work/demo/converted" --work-dir "$work/results"
