from __future__ import annotations

from pathlib import Path

import pytest
import yaml

_EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "runtime"


@pytest.mark.parametrize("path", sorted(_EXAMPLES.glob("teleop-*.yaml")), ids=lambda p: p.name)
def test_star_arm_leaders_emit_the_b601_frame(path: Path) -> None:
    # The B601 followers take LeRobot-style joints; a Star Arm leader only emits them with the b601 profile.
    runtime = yaml.safe_load(path.read_text())["runtime"]
    leader = runtime["action_source"]["init_args"]["leader"]

    assert leader["class_path"].startswith("physicalai_stararm_plugin.")
    assert leader["init_args"]["follower_profile"] == "b601"
