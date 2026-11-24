# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""OpenArm v2 twin: Enactic's model from the pinned ``openarm-mujoco`` archive, under LeRobot's names and units."""

# MuJoCo model/data attributes are supplied by the C extension at runtime.
# pyrefly: ignore-errors [missing-attribute]

from __future__ import annotations

import functools
import hashlib
import http.server
import importlib.util
import os
import sys
import tarfile
import threading
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

import mujoco
import numpy as np
import pytest

from physicalai_mujoco_plugin.compose import compose_scene, fetch_profile
from physicalai_mujoco_plugin.external_models import EXTERNAL_MODELS, OPENARM_V2, external_available, fetch_external
from physicalai_mujoco_plugin.profiles import get_profile
from physicalai_mujoco_plugin.profiles.openarm import (
    OPENARM_FINGER_OPEN_DEG,
    OPENARM_GRIPPER_OPEN_DEG,
    OPENARM_HOME_DEG,
    OPENARM_PROFILE,
)
from physicalai_mujoco_plugin.robot import MuJoCoRobot
from physicalai_mujoco_plugin.scene_registry import get_scene, list_scenes_naming, supported_arm_counts
from physicalai_mujoco_plugin.studio_catalog import list_catalog_entries
from physicalai_mujoco_plugin.studio_urdf import models_available

if TYPE_CHECKING:
    from collections.abc import Iterator
    from types import ModuleType

LEROBOT_NAMES = tuple(
    sorted(
        f"{side}_{joint}" for side in ("left", "right") for joint in (*(f"joint_{i}" for i in range(1, 8)), "gripper")
    )
)
"""``BiOpenArmFollower``'s ``*.pos`` keys without the suffix, sorted as ``physicalai-lerobot-plugin`` sorts them."""
CELL_GEOMS = ("floor", "cell_")


# --- The archive -----------------------------------------------------------------------------------


class _CountingHandler(http.server.SimpleHTTPRequestHandler):
    requests: list[str]

    def do_GET(self) -> None:
        self.requests.append(self.path)
        super().do_GET()

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - the parent's parameter name
        pass


@pytest.fixture
def archive(tmp_path: Path) -> Iterator[tuple[object, list[str]]]:
    """Serve an archive laid out like ``openarm-mujoco``'s over local HTTP; yield its model and the requests."""
    tree = tmp_path / "src" / "openarm_mujoco-9.9.9"
    (tree / "v2").mkdir(parents=True)
    (tree / "v2" / "openarm_bimanual.xml").write_text("<mujoco/>")
    served = tmp_path / "served"
    served.mkdir()
    path = served / "openarm_mujoco-9.9.9.tar.gz"
    with tarfile.open(path, "w:gz") as tar:
        tar.add(tree, arcname=tree.name)
    data = path.read_bytes()
    requests: list[str] = []
    handler = functools.partial(type("Handler", (_CountingHandler,), {"requests": requests}), directory=str(served))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    model = replace(
        OPENARM_V2,
        archive_url=f"http://127.0.0.1:{server.server_address[1]}/{path.name}",
        sha256=hashlib.sha256(data).hexdigest(),
        size=len(data),
    )
    env = {"MENAGERIE_CACHE_DIR": str(tmp_path / "cache"), "MENAGERIE_ROOT": str(tmp_path / "menagerie")}
    with patch.dict(os.environ, env):
        os.environ.pop(OPENARM_V2.root_env, None)
        yield model, requests
    server.shutdown()
    server.server_close()


def test_the_archive_downloads_once_into_the_menagerie_cache(archive: tuple[object, list[str]], tmp_path: Path) -> None:
    model, requests = archive
    progress: list[tuple[int, int]] = []

    path = fetch_external(model, progress=lambda done, total: progress.append((done, total)))  # type: ignore[arg-type]

    assert path.read_text() == "<mujoco/>"
    assert path.is_relative_to(tmp_path / "cache")  # MENAGERIE_ROOT names a Menagerie checkout, not this archive
    assert progress[-1] == (model.size, model.size)  # type: ignore[attr-defined]
    assert fetch_external(model) == path  # type: ignore[arg-type]
    assert len(requests) == 1


def test_an_archive_that_does_not_match_its_digest_is_refused(archive: tuple[object, list[str]]) -> None:
    model, _ = archive
    with pytest.raises(RuntimeError, match="does not match the digest"):
        fetch_external(replace(model, sha256="0" * 64))  # type: ignore[type-var]


def test_a_local_copy_replaces_the_download(archive: tuple[object, list[str]], tmp_path: Path) -> None:
    model, requests = archive
    copy = tmp_path / "checkout"
    (copy / "v2").mkdir(parents=True)
    (copy / "v2" / "openarm_bimanual.xml").write_text("<mujoco/>")
    entry = copy / "v2" / "openarm_bimanual.xml"
    with patch.dict(os.environ, {OPENARM_V2.root_env: str(copy)}):
        assert fetch_external(model) == entry  # type: ignore[arg-type]
        entry.unlink()
        with pytest.raises(RuntimeError, match="has no v2/openarm_bimanual.xml"):
            fetch_external(model)  # type: ignore[arg-type]
    assert requests == []


def test_studio_finds_the_model_once_it_is_downloaded_or_copied(archive: tuple[object, list[str]], tmp_path: Path) -> None:
    """Studio generates the 3D model only from a model already there (``models_available``), never downloading it."""
    model, _ = archive
    profile = get_profile("openarm")
    with patch.dict(EXTERNAL_MODELS, {profile.menagerie_model: model}):  # type: ignore[dict-item]
        assert not external_available(model)  # type: ignore[arg-type]
        assert not models_available(profile)  # MENAGERIE_ROOT is set, but it is a Menagerie checkout
        fetch_external(model)  # type: ignore[arg-type]
        assert external_available(model)  # type: ignore[arg-type]
        assert models_available(profile)
        copy = tmp_path / "empty"
        copy.mkdir()
        with patch.dict(os.environ, {OPENARM_V2.root_env: str(copy)}):
            assert not models_available(profile)  # the local copy replaces the cache, and it lacks the file


def test_an_unknown_entry_is_refused() -> None:
    with pytest.raises(RuntimeError, match="no entry 'cell'"):
        OPENARM_V2.entry_file("cell")


def test_the_profile_names_the_pinned_archive() -> None:
    assert EXTERNAL_MODELS[OPENARM_PROFILE.menagerie_model] is OPENARM_V2
    assert OPENARM_V2.archive_name == "openarm_mujoco-2.3.0.tar.gz"
    assert OPENARM_V2.archive_url.startswith("https://files.pythonhosted.org/")


# --- The twin ----------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def robot() -> Iterator[MuJoCoRobot]:
    robot = MuJoCoRobot("openarm", cameras=[])
    robot.connect()
    yield robot
    robot.disconnect()


def _settle(robot: MuJoCoRobot, action: np.ndarray, ticks: int = 150) -> np.ndarray:
    robot.send_action(action.astype(np.float32))
    observation = robot.get_observation()
    for _ in range(ticks - 1):
        observation = robot.get_observation()
    return np.asarray(observation.joint_positions)


@pytest.mark.requires_download
class TestTwin:
    def test_channels_follow_lerobots_bi_openarm_follower(self, robot: MuJoCoRobot) -> None:
        assert tuple(robot.joint_names) == LEROBOT_NAMES
        assert tuple(channel.name for channel in OPENARM_PROFILE.channels) == LEROBOT_NAMES

    def test_it_starts_at_home_with_the_grippers_open(self, robot: MuJoCoRobot) -> None:
        robot.reset()
        observed = dict(zip(robot.joint_names, robot.get_observation().joint_positions, strict=True))
        for side in ("left", "right"):
            for index, degrees in enumerate(OPENARM_HOME_DEG, start=1):
                assert observed[f"{side}_joint_{index}"] == pytest.approx(degrees, abs=0.5)
            assert observed[f"{side}_gripper"] == pytest.approx(-OPENARM_GRIPPER_OPEN_DEG, abs=0.5)

    def test_both_grippers_close_at_zero_and_open_at_minus_65(self, robot: MuJoCoRobot) -> None:
        robot.reset()
        home = robot.get_observation().joint_positions.copy()
        gripper = [robot.joint_names.index(f"{side}_gripper") for side in ("left", "right")]
        data = robot._sim.data  # noqa: SLF001
        finger_open = np.radians(OPENARM_FINGER_OPEN_DEG)
        for public, fingers in ((0.0, (0.0, 0.0)), (-OPENARM_GRIPPER_OPEN_DEG, (finger_open, -finger_open))):
            action = home.copy()
            action[gripper] = public
            observed = _settle(robot, action)
            np.testing.assert_allclose(observed[gripper], public, atol=1.0)
            for side, finger in zip(("left", "right"), fingers, strict=True):
                for joint in (1, 2):  # the second finger follows the first
                    qpos = data.joint(f"openarm_{side}_finger_joint{joint}").qpos[0]
                    assert qpos == pytest.approx(finger, abs=0.02), (side, joint, public)

    def test_arm_joints_are_degrees_of_the_model_joints(self, robot: MuJoCoRobot) -> None:
        robot.reset()
        action = robot.get_observation().joint_positions.copy()
        index = robot.joint_names.index("right_joint_2")
        action[index] = 30.0
        observed = _settle(robot, action)
        assert observed[index] == pytest.approx(30.0, abs=1.0)
        assert np.degrees(robot._sim.data.joint("openarm_right_joint2").qpos[0]) == pytest.approx(30.0, abs=1.0)  # noqa: SLF001

    def test_it_streams_its_gripper_cameras_and_the_overhead_camera(self) -> None:
        model = get_scene("openarm_cell").load_model(OPENARM_PROFILE)
        assert {model.camera(i).name for i in range(model.ncam)} == {"overview", "left_wrist", "right_wrist"}
        for side in ("left", "right"):
            camera = model.camera(f"{side}_wrist")
            assert model.body(camera.bodyid[0]).name == f"openarm_{side}_ee_base_link"
            assert camera.fovy[0] == pytest.approx(66.0)


@pytest.mark.requires_download
class TestCell:
    def test_the_cell_runs_the_openarm_alone(self) -> None:
        assert list_scenes_naming("openarm") == {"openarm_cell": get_scene("openarm_cell")}
        assert supported_arm_counts(OPENARM_PROFILE) == (1,)
        assert get_profile("openarm") is OPENARM_PROFILE
        (entry,) = (entry for entry in list_catalog_entries() if entry.profile is OPENARM_PROFILE)
        assert entry.launch_labels() == ("2 arms", "twin", "provisional")

    def test_at_home_only_the_block_touches_the_cell(self) -> None:
        scene = get_scene("openarm_cell")
        composed = compose_scene(scene.scene_xml_path, OPENARM_PROFILE, scene_layout=scene.layout_for(OPENARM_PROFILE, 1))
        model = composed.model
        data = mujoco.MjData(model)
        (binding,) = composed.robots
        for joint, values in binding.layout.home_qpos.items():
            address = model.joint(joint).qposadr[0]
            data.qpos[address : address + len(values)] = values
        mujoco.mj_forward(model, data)
        data.ctrl[:] = [data.qpos[model.jnt_qposadr[model.actuator_trnid[i, 0]]] for i in range(model.nu)]
        for _ in range(500):
            mujoco.mj_step(model, data)
        touching = {
            frozenset((model.body(model.geom_bodyid[c.geom1]).name, model.body(model.geom_bodyid[c.geom2]).name))
            for c in data.contact[: data.ncon]
        }
        assert touching == {frozenset(("world", "block1"))}

    def test_the_overhead_camera_looks_straight_down_at_the_sheet(self) -> None:
        model = get_scene("openarm_cell").load_model(OPENARM_PROFILE)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        camera = model.camera("overview").id
        view = -data.cam_xmat[camera].reshape(3, 3)[:, 2]
        np.testing.assert_allclose(view, (0.0, 0.0, -1.0), atol=1e-9)
        sheet = model.geom("cell_sheet").id
        assert np.linalg.norm(data.cam_xpos[camera][:2] - data.geom_xpos[sheet][:2]) < model.geom_size[sheet][0]


def _load_reach_test() -> ModuleType:
    """Load ``test_reach.py`` by path: CI runs pytest with ``--import-mode=importlib``, so it is not importable by name."""
    spec = importlib.util.spec_from_file_location("_openarm_reach", Path(__file__).with_name("test_reach.py"))
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # its dataclasses look their module up by name
    spec.loader.exec_module(module)
    return module


@pytest.mark.mujoco_smoke
@pytest.mark.slow
@pytest.mark.requires_download
def test_both_arms_reach_the_spawn_area_and_the_right_arm_the_target_from_above() -> None:
    """The reach test's IK (``test_reach.py``) on the cell's spawn disc and target."""
    test_reach = _load_reach_test()

    scene = get_scene("openarm_cell")
    arms = test_reach.build_arms(scene, OPENARM_PROFILE, np.random.default_rng(0))
    points = test_reach.spawn_points(scene.for_profile(OPENARM_PROFILE))
    for label, point in points.items():
        for arm in arms:
            error, tilt = test_reach.solve(arm, point, good_tilt_deg=test_reach.GOOD_TILT_DEG)
            assert error <= test_reach.POSITION_TOLERANCE, (label, arm.name, error)
            assert tilt <= test_reach.TILT_TOLERANCE_DEG, (label, arm.name, tilt)
    target = np.array([*scene.load_model(OPENARM_PROFILE).body("target").pos[:2], test_reach.FREEJOINT_SPAWN_Z])
    right = next(arm for arm in arms if arm.name.startswith("openarm_right"))
    error, tilt = test_reach.solve(right, target, good_tilt_deg=test_reach.GOOD_TILT_DEG)
    assert error <= test_reach.POSITION_TOLERANCE
    assert tilt <= test_reach.TILT_TOLERANCE_DEG


def test_fetching_the_profile_uses_the_external_model() -> None:
    with patch("physicalai_mujoco_plugin.compose.fetch_external", return_value="sentinel") as fetch:
        assert fetch_profile(OPENARM_PROFILE) == "sentinel"
    fetch.assert_called_once_with(OPENARM_V2, "openarm_bimanual", None)
