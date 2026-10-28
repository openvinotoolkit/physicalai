# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Generate offline assets from a reviewed, pinned UR description checkout.

Run with uv run --no-project --with xacro==2.1.1 python scripts/generate_assets.py SOURCE.
Only development uses Xacro; the installed plugin needs neither ROS nor Xacro.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from xml.etree import ElementTree

import xacro

REVISION = "81f7c9fe003c2f0e02eec5a7253f1b262ad38b61"
MODELS = ("ur3e", "ur5e", "ur7e", "ur10e", "ur12e", "ur16e")
JOINTS = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint")


def generate(source: Path, destination: Path) -> None:
    """Generate URDFs and copy only referenced BSD-licensed meshes."""
    source = source.resolve(strict=True)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=source, text=True)
    if revision != REVISION or dirty:
        msg = f"Expected clean upstream checkout at {REVISION}; got {revision}."
        raise ValueError(msg)
    destination.mkdir(parents=True, exist_ok=True)
    limits = {}
    # Resolve ROS package substitutions in a temporary developer-only copy.
    with tempfile.TemporaryDirectory(prefix="ur-xacro-") as directory:
        root = Path(directory)
        for path in (source / "urdf").rglob("*.xacro"):
            target = root / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            text = path.read_text().replace("$(find ur_description)/urdf", str(root / "urdf"))
            target.write_text(text.replace("$(find ur_description)", str(source)))
        for model in MODELS:
            document = xacro.process_file(str(root / "urdf/ur.urdf.xacro"), mappings={"ur_type": model, "name": model})
            text = document.toprettyxml(indent="  ").replace(str(root), "ur_description")
            robot = ElementTree.fromstring(text)
            for mesh in robot.findall(".//mesh"):
                filename = mesh.attrib["filename"]
                relative = Path(filename.removeprefix("package://ur_description/"))
                if not filename.startswith("package://ur_description/meshes/") or relative.parts[1] not in {
                    "ur3e",
                    "ur5e",
                    "ur10e",
                    "ur16e",
                }:
                    msg = f"Unexpected or non-BSD asset: {filename}"
                    raise ValueError(msg)
                original = (source / relative).resolve(strict=True)
                if not original.is_relative_to(source):
                    msg = f"Asset escapes checkout: {filename}"
                    raise ValueError(msg)
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(original, target)
            joints = {joint.attrib["name"]: joint.find("limit") for joint in robot.findall("joint")}
            limits[model] = {
                key: [
                    float(joints[name].attrib[attribute]) if attribute in joints[name].attrib else None
                    for name in JOINTS
                ]
                for key, attribute in (("lower", "lower"), ("upper", "upper"), ("velocity", "velocity"))
            }
            (destination / f"{model}.urdf").write_text(text, encoding="utf-8")
    shutil.copyfile(source / "LICENSE", destination / "LICENSE-BSD-3-Clause")
    shutil.copyfile(Path(__file__).resolve().parents[3] / "LICENSE", destination / "LICENSE-Apache-2.0")
    (destination / "UPSTREAM-README.txt").write_text((source / "README.md").read_text(), encoding="utf-8")
    (destination / "limits.json").write_text(json.dumps(limits, indent=2) + "\n", encoding="utf-8")
    (destination / "SOURCE").write_text(
        f"https://github.com/UniversalRobots/Universal_Robots_ROS2_Description\n{REVISION}\nxacro==2.1.1\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    generate(args.source, Path(__file__).resolve().parents[1] / "src/physicalai_ur5e_plugin/assets")
