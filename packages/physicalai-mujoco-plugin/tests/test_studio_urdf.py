# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Studio's 3D models of the simulated robots: URDFs generated from the MuJoCo models the simulations run."""

# MuJoCo model/data attributes are supplied by the C extension at runtime.
# pyrefly: ignore-errors [missing-attribute]

from __future__ import annotations

import math
import struct
import threading
from typing import TYPE_CHECKING

import defusedxml.ElementTree as ET
import mujoco
import numpy as np
import pytest

from physicalai_mujoco_plugin import studio_urdf
from physicalai_mujoco_plugin.channels import ArmChannels
from physicalai_mujoco_plugin.studio_catalog import _definitions, list_catalog_entries
from physicalai_mujoco_plugin.studio_urdf import (
    CACHE_DIR_ENV,
    URDF_FILE,
    bind_channels,
    cache_root,
    compose_robot,
    ensure_robot_urdf,
    entry_key,
    joint_drives,
    public_names,
    write_robot_urdf,
)

if TYPE_CHECKING:
    from pathlib import Path
    from xml.etree.ElementTree import Element

    from physicalai_mujoco_plugin.studio_catalog import CatalogEntry

ENTRIES = {entry.type: entry for entry in list_catalog_entries()}
CHECKED = (
    "MuJoCo_FrankaResearch3_Follower",
    "MuJoCo_FrankaEmikaPanda_Follower",
    "MuJoCo_FrankaDuo_Follower",
    "MuJoCo_reBotB601_Follower",
    "MuJoCo_reBotB601_Bimanual_Follower",
    "MuJoCo_WidowXAI_Follower",
    "MuJoCo_WidowXAI_Bimanual_Follower",
)
"""The entries every test here generates and checks; ``test_every_entry_generates`` covers the rest."""
FK_TOLERANCE_M = 1e-3
ROTATION_TOLERANCE = 1e-6


@pytest.fixture(scope="module")
def generated(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Each checked entry's URDF directory, generated once."""
    root = tmp_path_factory.mktemp("urdf")
    directories = {}
    for name in CHECKED:
        entry = ENTRIES[name]
        directories[name] = root / name
        directories[name].mkdir()
        write_robot_urdf(entry.profile, entry.prefixes, directories[name])
    return directories


def _root(directory: Path) -> Element:
    return ET.parse(directory / URDF_FILE).getroot()


def _floats(text: str | None, default: tuple[float, ...]) -> np.ndarray:
    return np.array([float(v) for v in text.split()] if text else default)


def _rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = (f(a) for a in (roll, pitch, yaw) for f in (math.cos, math.sin))
    return (
        np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        @ np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        @ np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    )


def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * k @ k


def _joint_values(root: Element, channels: dict[str, float] | None, model_joints: dict[str, float] | None) -> dict:
    """Return every URDF joint's value: from its ``mimic`` and the channel values, or given directly."""
    values = {}
    for joint in root.findall("joint"):
        name = joint.get("name")
        mimic = joint.find("mimic")
        if model_joints is not None:
            values[name] = model_joints.get(name, 0.0)
        elif mimic is not None:
            multiplier, offset = float(mimic.get("multiplier", "1")), float(mimic.get("offset", "0"))
            values[name] = multiplier * (channels or {}).get(mimic.get("joint"), 0.0) + offset
        else:
            values[name] = (channels or {}).get(name, 0.0)
    return values


def urdf_forward_kinematics(root: Element, values: dict[str, float]) -> dict[str, np.ndarray]:
    """Return every link's world pose (4x4) for the joint *values*, from the URDF's origins and axes alone."""
    by_child = {joint.find("child").get("link"): joint for joint in root.findall("joint")}
    poses: dict[str, np.ndarray] = {}

    def pose(link: str) -> np.ndarray:
        if link in poses:
            return poses[link]
        joint = by_child.get(link)
        if joint is None:
            poses[link] = np.eye(4)
            return poses[link]
        origin = joint.find("origin")
        local = np.eye(4)
        local[:3, :3] = _rpy(*_floats(origin.get("rpy"), (0.0, 0.0, 0.0)))
        local[:3, 3] = _floats(origin.get("xyz"), (0.0, 0.0, 0.0))
        motion = np.eye(4)
        axis_element = joint.find("axis")
        axis = _floats(axis_element.get("xyz") if axis_element is not None else None, (1.0, 0.0, 0.0))
        axis /= np.linalg.norm(axis)
        value = values[joint.get("name")]
        if joint.get("type") in {"revolute", "continuous"}:
            motion[:3, :3] = _axis_angle(axis, value)
        elif joint.get("type") == "prismatic":
            motion[:3, 3] = axis * value
        poses[link] = pose(joint.find("parent").get("link")) @ local @ motion
        return poses[link]

    for link in root.findall("link"):
        pose(link.get("name"))
    return poses


def _bodies_error(model: mujoco.MjModel, data: mujoco.MjData, poses: dict[str, np.ndarray]) -> tuple[float, float]:
    """Return the largest position (m) and rotation-matrix error between MuJoCo's bodies and their links."""
    position = rotation = 0.0
    for body in range(1, model.nbody):
        link = poses[model.body(body).name]
        position = max(position, float(np.linalg.norm(link[:3, 3] - data.xpos[body])))
        rotation = max(rotation, float(np.abs(link[:3, :3] - data.xmat[body].reshape(3, 3)).max()))
    return position, rotation


def _end_effector_bodies(entry: CatalogEntry, model: mujoco.MjModel) -> list[int]:
    bodies = []
    for prefix in entry.prefixes:
        for effector in entry.profile.end_effectors:
            if effector.body is not None:
                bodies.append(model.body(f"{prefix}{effector.body}").id)
            else:
                bodies.append(int(model.site_bodyid[model.site(f"{prefix}{effector.site}").id]))
    return bodies


@pytest.mark.parametrize("name", CHECKED)
def test_the_urdf_has_a_link_per_body_and_a_joint_per_model_joint(name: str, generated: dict[str, Path]) -> None:
    entry = ENTRIES[name]
    model, _ = compose_robot(entry.profile, entry.prefixes)
    root = _root(generated[name])
    links = {link.get("name") for link in root.findall("link")}
    joints = {joint.get("name"): joint for joint in root.findall("joint")}
    assert root.tag == "robot"
    assert {model.body(body).name for body in range(1, model.nbody)} <= links
    for joint in range(model.njnt):
        if model.jnt_type[joint] not in {mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE}:
            continue
        element = joints[model.joint(joint).name]
        hinge = model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_HINGE
        limited = bool(model.jnt_limited[joint])
        assert element.get("type") == ("prismatic" if not hinge else "revolute" if limited else "continuous")
        np.testing.assert_allclose(_floats(element.find("axis").get("xyz"), ()), model.jnt_axis[joint], atol=1e-9)
        if limited:
            limit = element.find("limit")
            qpos0 = model.qpos0[model.jnt_qposadr[joint]]
            lower, upper = float(limit.get("lower")), float(limit.get("upper"))
            np.testing.assert_allclose((lower, upper), model.jnt_range[joint] - qpos0, atol=1e-9)
    # One virtual joint per public channel, named after the observation Studio records.
    for channel in entry.joint_names():
        virtual = joints[f"{channel}_channel"]
        assert virtual.get("type") == "prismatic"
        assert virtual.find("parent").get("link") == "world"


def test_the_expected_robots_are_there(generated: dict[str, Path]) -> None:
    joints = {name: {joint.get("name") for joint in _root(path).findall("joint")} for name, path in generated.items()}
    fr3 = {f"fr3_joint{index}" for index in range(1, 8)} | {"finger_joint1", "finger_joint2"}
    assert fr3 <= joints["MuJoCo_FrankaResearch3_Follower"]
    assert {f"{side}_{joint}" for side in ("left", "right") for joint in fr3} <= joints["MuJoCo_FrankaDuo_Follower"]
    assert {f"joint{index}" for index in range(1, 8)} <= joints["MuJoCo_FrankaEmikaPanda_Follower"]
    widowx = {f"joint_{index}" for index in range(6)} | {"left_carriage_joint", "right_carriage_joint"}
    assert widowx <= joints["MuJoCo_WidowXAI_Follower"]
    assert {f"{side}_{joint}" for side in ("left", "right") for joint in widowx} <= joints[
        "MuJoCo_WidowXAI_Bimanual_Follower"
    ]
    rebot = {f"joint{index}" for index in range(1, 7)} | {"joint_left", "joint_right"}
    assert rebot <= joints["MuJoCo_reBotB601_Follower"]
    assert {f"{side}_{joint}" for side in ("left", "right") for joint in rebot} <= joints[
        "MuJoCo_reBotB601_Bimanual_Follower"
    ]


@pytest.mark.parametrize("name", CHECKED)
def test_forward_kinematics_match_mujoco(name: str, generated: dict[str, Path]) -> None:
    """Random joint positions within their ranges put every link where ``mj_kinematics`` puts its body."""
    entry = ENTRIES[name]
    model, _ = compose_robot(entry.profile, entry.prefixes)
    data = mujoco.MjData(model)
    root = _root(generated[name])
    rng = np.random.default_rng(0)
    worst = (0.0, 0.0)
    for _ in range(5):
        data.qpos[:] = model.qpos0
        joints = {}
        for joint in range(model.njnt):
            if model.jnt_type[joint] not in {mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE}:
                continue
            low, high = model.jnt_range[joint] if model.jnt_limited[joint] else (-math.pi, math.pi)
            address = model.jnt_qposadr[joint]
            data.qpos[address] = rng.uniform(low, high)
            joints[model.joint(joint).name] = data.qpos[address] - model.qpos0[address]
        mujoco.mj_kinematics(model, data)
        poses = urdf_forward_kinematics(root, _joint_values(root, None, joints))
        error = _bodies_error(model, data, poses)
        worst = (max(worst[0], error[0]), max(worst[1], error[1]))
        for body in _end_effector_bodies(entry, model):
            assert np.linalg.norm(poses[model.body(body).name][:3, 3] - data.xpos[body]) < FK_TOLERANCE_M
    assert worst[0] < FK_TOLERANCE_M
    assert worst[1] < ROTATION_TOLERANCE


@pytest.mark.parametrize("name", CHECKED)
def test_observations_move_the_joints_as_the_simulation_does(name: str, generated: dict[str, Path]) -> None:
    """Studio sets each virtual joint to its observation; the mimics give each model joint the simulation's position.

    A joint member takes the channel's model value; the Franka Hand's fingers each take half its
    width (the tendon the gripper drives), and a joint equality's follower follows its leader.
    """
    entry = ENTRIES[name]
    model, robots = compose_robot(entry.profile, entry.prefixes)
    channels = bind_channels(model, robots, entry.profile)
    root = _root(generated[name])
    rng = np.random.default_rng(1)
    for _ in range(5):
        model_values = {}
        for channel in channels:
            low, high = channel.range or channel.first.range or (0.0, 0.08)
            model_values[channel.name] = rng.uniform(low, high)
        public = {f"{c.name}_channel": ArmChannels.to_public(c, model_values[c.name]) for c in channels}
        values = _joint_values(root, public, None)
        qpos = {
            model.joint(j).name: values[model.joint(j).name] + model.qpos0[model.jnt_qposadr[j]]
            for j in range(model.njnt)
            if model.joint(j).name in values
        }
        for channel in channels:
            expected = model_values[channel.name]
            for position, member in enumerate(channel.members):
                scale = 1.0 if position == 0 or not channel.member_scales else channel.member_scales[position - 1]
                if member.joint is not None:
                    assert qpos[member.joint] == pytest.approx(expected * scale, abs=1e-9)
                else:  # a fixed tendon: its joints add up to the channel's length
                    tendon = int(model.actuator_trnid[member.actuator_id, 0])
                    start, count = model.tendon_adr[tendon], model.tendon_num[tendon]
                    length = sum(
                        model.wrap_prm[w] * qpos[model.joint(model.wrap_objid[w]).name]
                        for w in range(start, start + count)
                    )
                    assert length == pytest.approx(expected * scale, abs=1e-9)
        for equality in range(model.neq):
            if model.eq_type[equality] != mujoco.mjtEq.mjEQ_JOINT or not model.eq_active0[equality]:
                continue
            follower, leader = (
                model.joint(int(model.eq_obj1id[equality])),
                model.joint(int(model.eq_obj2id[equality])),
            )
            c0, c1 = model.eq_data[equality, :2]
            assert values[follower.name] == pytest.approx(c0 + c1 * values[leader.name], abs=1e-9)


def test_the_franka_hand_and_widowx_carriages_follow_their_gripper(generated: dict[str, Path]) -> None:
    """A width of 0.06 m opens each Franka finger 0.03 m; the WidowX carriages both take its 0.02 m."""
    fr3 = _joint_values(_root(generated["MuJoCo_FrankaResearch3_Follower"]), {"gripper_channel": 0.06}, None)
    assert (fr3["finger_joint1"], fr3["finger_joint2"]) == (pytest.approx(0.03), pytest.approx(0.03))
    widowx = _joint_values(_root(generated["MuJoCo_WidowXAI_Follower"]), {"gripper_channel": 0.02}, None)
    assert (widowx["left_carriage_joint"], widowx["right_carriage_joint"]) == (pytest.approx(0.02), pytest.approx(0.02))
    degrees = _joint_values(_root(generated["MuJoCo_WidowXAI_Follower"]), {"shoulder_pan_channel": 90.0}, None)
    assert degrees["joint_0"] == pytest.approx(math.pi / 2)


@pytest.mark.parametrize("name", CHECKED)
def test_meshes_are_binary_stl_files_in_the_package(name: str, generated: dict[str, Path]) -> None:
    directory = generated[name]
    entry = ENTRIES[name]
    package = f"package://mujoco_{entry.profile.name}_{len(entry.prefixes)}/"
    filenames = [mesh.get("filename") for mesh in _root(directory).iter("mesh")]
    assert filenames
    for filename in filenames:
        assert filename.startswith(package)
        path = directory / filename.removeprefix(package)
        data = path.read_bytes()
        (count,) = struct.unpack_from("<I", data, 80)
        assert count > 0
        assert len(data) == 84 + 50 * count
        record = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")])
        rows = np.frombuffer(data, dtype=record, count=count, offset=84)
        assert np.isfinite(rows["vertices"]).all()
        assert np.allclose(np.linalg.norm(rows["normal"], axis=1)[np.any(rows["normal"] != 0, axis=1)], 1.0, atol=1e-5)
    materials = {material.get("name") for material in _root(directory).findall("material")}
    assert {visual.find("material").get("name") for visual in _root(directory).iter("visual")} <= materials


@pytest.mark.parametrize("entry", list_catalog_entries(), ids=lambda entry: entry.type)
def test_every_entry_generates(entry: CatalogEntry, tmp_path: Path) -> None:
    """Every catalog entry's robot exports, with a virtual joint for each of its channels."""
    generated = write_robot_urdf(entry.profile, entry.prefixes, tmp_path)
    names = {joint.get("name") for joint in _root(tmp_path).findall("joint")}
    assert set(generated.joint_map) == {f"{name}.pos" for name in entry.joint_names()}
    assert {joint for joints in generated.joint_map.values() for joint in joints} <= names
    assert {drive.channel for drive in generated.drives.values()} <= {j for js in generated.joint_map.values() for j in js}


@pytest.mark.parametrize("entry", list_catalog_entries(), ids=lambda entry: entry.type)
def test_public_names_need_no_compile(entry: CatalogEntry) -> None:
    """The names read from the model file are the ones the compiled model derives."""
    assert public_names(entry.profile, entry.prefixes) == entry.joint_names()


class TestCatalogAssets:
    """Every follower entry has a 3D model; the SO-101's are hand-made, the others generated on first use."""

    def test_every_follower_has_an_asset_whose_joint_map_names_its_joints(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv(CACHE_DIR_ENV, str(tmp_path))
        definitions = {d.type: d for d in _definitions() if d.role == "follower"}
        assert set(definitions) == set(ENTRIES)
        for name in (*CHECKED, "MuJoCo_SO101_Follower"):
            asset = definitions[name].asset
            assert asset is not None
            assert asset.root_resolver is not None
            urdf = asset.root_resolver() / asset.urdf_relative_path
            assert urdf.is_file()
            joints = {joint.get("name") for joint in ET.parse(urdf).getroot().findall("joint")}
            assert set(asset.joint_map) == {f"{n}.pos" for n in ENTRIES[name].joint_names()}
            assert {joint for joints_ in asset.joint_map.values() for joint in joints_} <= joints
        for name in set(ENTRIES) - {"MuJoCo_SO101_Follower", "MuJoCo_SO101_Bimanual_Follower"}:
            asset = definitions[name].asset
            assert asset is not None
            assert (
                str(asset.urdf_relative_path)
                == f"{entry_key(ENTRIES[name].profile, len(ENTRIES[name].prefixes))}/robot.urdf"
            )
            assert list(asset.packages.values()) == [asset.urdf_relative_path.parent]

    def test_mesh_urls_resolve_under_the_package_root(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Studio serves ``package://<package>/<path>`` from the package's directory under the asset root."""
        monkeypatch.setenv(CACHE_DIR_ENV, str(tmp_path))
        asset = {d.type: d for d in _definitions()}["MuJoCo_FrankaResearch3_Follower"].asset
        assert asset is not None
        root = asset.root_resolver()
        ((package, directory),) = asset.packages.items()
        for mesh in ET.parse(root / asset.urdf_relative_path).getroot().iter("mesh"):
            relative = mesh.get("filename").removeprefix(f"package://{package}/")
            assert (root / directory / relative).resolve().is_relative_to(root)
            assert (root / directory / relative).is_file()


class TestCache:
    def test_generates_once_and_reuses_the_cache(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv(CACHE_DIR_ENV, str(tmp_path))
        monkeypatch.setattr(studio_urdf, "_READY", set())
        entry = ENTRIES["MuJoCo_WidowXAI_Follower"]
        root = ensure_robot_urdf(entry.profile, entry.prefixes)
        assert root == cache_root().resolve()
        assert root.is_relative_to(tmp_path.resolve())
        urdf = root / entry_key(entry.profile, 1) / URDF_FILE
        assert urdf.is_file()
        stamp = urdf.stat().st_mtime_ns
        monkeypatch.setattr(studio_urdf, "_READY", set())
        monkeypatch.setattr(studio_urdf, "write_robot_urdf", lambda *_: pytest.fail("generated twice"))
        assert ensure_robot_urdf(entry.profile, entry.prefixes) == root
        assert urdf.stat().st_mtime_ns == stamp

    def test_concurrent_requests_write_one_entry(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv(CACHE_DIR_ENV, str(tmp_path))
        monkeypatch.setattr(studio_urdf, "_READY", set())
        entry = ENTRIES["MuJoCo_reBotB601_Follower"]
        roots: list[Path] = []
        threads = [
            threading.Thread(target=lambda: roots.append(ensure_robot_urdf(entry.profile, entry.prefixes)))
            for _ in range(4)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert len(set(roots)) == 1
        assert [path.name for path in roots[0].iterdir()] == [entry_key(entry.profile, 1)]

    def test_the_cache_is_versioned(self) -> None:
        version = studio_urdf.cache_version()
        assert version.startswith(f"v{studio_urdf.GENERATOR_VERSION}-")
        assert "menagerie" in version
        assert "mujoco" in version

    def test_a_model_that_needs_a_download_is_left_out(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Studio answers 404 rather than downloading a model inside a request."""
        monkeypatch.setenv(CACHE_DIR_ENV, str(tmp_path))
        monkeypatch.setattr(studio_urdf, "_READY", set())
        monkeypatch.setattr(studio_urdf, "models_available", lambda _profile: False)
        entry = ENTRIES["MuJoCo_UnitreeGo2_Follower"]
        root = ensure_robot_urdf(entry.profile, entry.prefixes)
        assert not (root / entry_key(entry.profile, 1)).exists()

    def test_a_failed_generation_leaves_nothing_behind(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv(CACHE_DIR_ENV, str(tmp_path))
        monkeypatch.setattr(studio_urdf, "_READY", set())

        def fail(*_: object) -> None:
            msg = "broken"
            raise RuntimeError(msg)

        monkeypatch.setattr(studio_urdf, "write_robot_urdf", fail)
        entry = ENTRIES["MuJoCo_KinovaGen3_Follower"]
        root = ensure_robot_urdf(entry.profile, entry.prefixes)
        assert list(root.iterdir()) == []


def test_joint_drives_cover_every_channel() -> None:
    entry = ENTRIES["MuJoCo_FrankaDuo_Follower"]
    model, robots = compose_robot(entry.profile, entry.prefixes)
    drives = joint_drives(model, bind_channels(model, robots, entry.profile))
    assert {drive.channel for drive in drives.values()} == {f"{name}_channel" for name in entry.joint_names()}
