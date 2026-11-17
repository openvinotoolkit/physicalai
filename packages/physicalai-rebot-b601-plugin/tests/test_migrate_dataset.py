# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the temporary 0.8 -> 0.9 dataset migration helper."""

from __future__ import annotations

import json
import re
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pytest

from physicalai_rebot_b601_plugin import migrate_dataset as md

pa = pytest.importorskip("pyarrow")
pytestmark = pytest.mark.filterwarnings("ignore:physicalai_rebot_b601_plugin.migrate_dataset:DeprecationWarning")
pq = pytest.importorskip("pyarrow.parquet")

NAMES = [f"{joint}.pos" for joint in md.JOINTS]
OLD_TO_NEW = np.array([-1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -6.0])
# Two frames in the old RS frame: lift and gripper non-negative.
OLD_STATE = np.array([[10.0, 40.0, -30.0, 5.0, 1.0, 2.0, 20.0], [-20.0, 80.0, -60.0, -5.0, 0.0, -3.0, 40.0]])


def _stats(values: np.ndarray) -> dict:
    return {
        "min": values.min(0).tolist(),
        "max": values.max(0).tolist(),
        "mean": values.mean(0).tolist(),
        "std": values.std(0).tolist(),
        "count": [len(values)],
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q50": np.quantile(values, 0.5, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
    }


def _write_dataset(root: Path, robot_type: str = "ReBot_B601_RS_Follower", state: np.ndarray = OLD_STATE) -> Path:
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    feature = {"dtype": "float32", "shape": [7], "names": NAMES}
    info = {
        "codebase_version": "v3.0",
        "robot_type": robot_type,
        "fps": 30,
        "features": {
            "action": feature,
            "observation.state": feature,
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        },
    }
    (root / "meta" / "info.json").write_text(json.dumps(info))
    action = state + 1.0
    table = pa.table({
        "observation.state": pa.array(state.astype(np.float32).tolist(), type=pa.list_(pa.float32())),
        "action": pa.array(action.astype(np.float32).tolist(), type=pa.list_(pa.float32())),
        "timestamp": pa.array([0.0, 1 / 30], type=pa.float32()),
        "episode_index": pa.array([0, 0], type=pa.int64()),
    })
    pq.write_table(table, root / "data" / "chunk-000" / "file-000.parquet")
    stats = {"action": _stats(action), "observation.state": _stats(state)}
    (root / "meta" / "stats.json").write_text(json.dumps(stats))
    episodes = {"episode_index": pa.array([0], type=pa.int64())}
    for key, values in (("action", action), ("observation.state", state)):
        for stat, value in _stats(values).items():
            episodes[f"stats/{key}/{stat}"] = pa.array([value], type=pa.list_(pa.float64()))
    pq.write_table(pa.table(episodes), root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    return root


def _column(root: Path, key: str) -> np.ndarray:
    table = pq.read_table(root / "data" / "chunk-000" / "file-000.parquet")
    return np.array(table[key].to_pylist())


def test_rs_dataset_is_converted_and_the_original_kept(tmp_path: Path) -> None:
    old = _write_dataset(tmp_path / "old")
    new = tmp_path / "new"

    summary = md.migrate(old, new)

    assert summary["variant"] == "rs"
    np.testing.assert_allclose(_column(new, "observation.state"), OLD_STATE * OLD_TO_NEW, rtol=1e-6)
    np.testing.assert_allclose(_column(new, "action"), (OLD_STATE + 1.0) * OLD_TO_NEW, rtol=1e-6)
    np.testing.assert_allclose(_column(old, "observation.state"), OLD_STATE, rtol=1e-6)  # untouched
    assert pq.read_table(new / "data" / "chunk-000" / "file-000.parquet").schema.field("action").type == pa.list_(
        pa.float32()
    )
    assert (new / "meta" / "physicalai_rebot_b601_migration.json").exists()


def test_dm_observations_stay_and_actions_are_converted(tmp_path: Path) -> None:
    old = _write_dataset(tmp_path / "old", robot_type="ReBot_B601_DM_Follower")
    new = tmp_path / "new"

    md.migrate(old, new)

    # DM 0.8 reported raw motor degrees, already the 0.9 frame.
    np.testing.assert_allclose(_column(new, "observation.state"), OLD_STATE, rtol=1e-6)
    np.testing.assert_allclose(_column(new, "action"), (OLD_STATE + 1.0) * OLD_TO_NEW, rtol=1e-6)


def test_stats_follow_the_data(tmp_path: Path) -> None:
    old = _write_dataset(tmp_path / "old")
    new = tmp_path / "new"

    md.migrate(old, new)

    expected = _stats(OLD_STATE * OLD_TO_NEW)
    stats = json.loads((new / "meta" / "stats.json").read_text())["observation.state"]
    for key in ("min", "max", "mean", "std", "q01", "q50", "q99"):
        np.testing.assert_allclose(stats[key], expected[key], rtol=1e-6, atol=1e-9, err_msg=key)
    episodes = pq.read_table(new / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    np.testing.assert_allclose(episodes["stats/observation.state/min"][0].as_py(), expected["min"], rtol=1e-6)
    np.testing.assert_allclose(episodes["stats/observation.state/max"][0].as_py(), expected["max"], rtol=1e-6)


def test_converted_or_new_frame_datasets_are_refused(tmp_path: Path) -> None:
    old = _write_dataset(tmp_path / "old")
    new = tmp_path / "new"
    md.migrate(old, new)

    with pytest.raises(md.MigrationError, match="already converted"):
        md.migrate(new, tmp_path / "again")
    new_frame = _write_dataset(tmp_path / "new_frame", state=OLD_STATE * OLD_TO_NEW - 2.0)
    with pytest.raises(md.MigrationError, match="looks converted"):
        md.migrate(new_frame, tmp_path / "out")
    md.migrate(new_frame, tmp_path / "forced", force=True)


def test_unknown_robot_type_needs_a_variant(tmp_path: Path) -> None:
    old = _write_dataset(tmp_path / "old", robot_type="so101_follower")

    with pytest.raises(md.MigrationError, match="cannot tell the reBot variant"):
        md.migrate(old, tmp_path / "out")
    md.migrate(old, tmp_path / "out", variant="rs")


def test_seeed_lerobot_datasets_are_refused(tmp_path: Path) -> None:
    old = _write_dataset(tmp_path / "old", robot_type="seeed_b601_rs_follower")

    with pytest.raises(md.MigrationError, match="Seeed"):
        md.migrate(old, tmp_path / "out")


def test_existing_destination_is_refused(tmp_path: Path) -> None:
    old = _write_dataset(tmp_path / "old")
    (tmp_path / "exists").mkdir()

    with pytest.raises(md.MigrationError, match="exists"):
        md.migrate(old, tmp_path / "exists")


def test_migrate_warns_that_it_is_temporary(tmp_path: Path) -> None:
    old = _write_dataset(tmp_path / "old")

    with pytest.warns(DeprecationWarning, match=md.REMOVE_IN):
        md.migrate(old, tmp_path / "new")


def test_prefixed_joint_names_are_matched() -> None:
    info = {
        "features": {
            "action": {"dtype": "float32", "names": [f"left_{n}" for n in NAMES] + ["right_gripper.pos"]},
        },
    }

    factors = md.feature_factors(info, "rs")["action"]

    np.testing.assert_allclose(factors, [*OLD_TO_NEW, -6.0])


def test_cli_reports_errors_without_a_traceback(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert md.main([str(tmp_path / "missing"), str(tmp_path / "out")]) == 1
    assert "not a LeRobot dataset" in capsys.readouterr().err


def test_migration_helper_is_removed_by_its_deadline() -> None:
    """This helper is temporary. Delete migrate_dataset.py and this test when this fails."""
    try:
        installed = version("physicalai-rebot-b601-plugin")
    except PackageNotFoundError:  # pragma: no cover - always installed in the workspace
        pytest.skip("plugin not installed")

    def release(text: str) -> tuple[int, ...]:
        return tuple(int(part) for part in re.findall(r"\d+", text.split("+")[0].split(".dev")[0])[:3])

    assert release(installed) < release(md.REMOVE_IN), (
        f"migrate_dataset was due for removal in {md.REMOVE_IN}; installed plugin is {installed}"
    )
