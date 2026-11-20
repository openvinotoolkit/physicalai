# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Convert reBot B601 LeRobot datasets recorded with plugin 0.8 or older to the 0.9 joint frame.

Plugin 0.9.0 moved the reBot B601 to LeRobot's public joint frame. Datasets recorded with older
versions (Physical AI Studio or ``physicalai run``) keep their joint values in the old frame, so a
policy trained on them would drive the arm the wrong way. The conversion is exact: each joint is
scaled by a constant per variant and per feature.

    python -m physicalai_rebot_b601_plugin.migrate_dataset OLD_DATASET_DIR NEW_DATASET_DIR [--variant rs|dm]

The old dataset is not modified; the converted copy is written to ``NEW_DATASET_DIR``. Import it
into Physical AI Studio as a new dataset. Models trained on old datasets need retraining on the
converted copy.

This helper is temporary: it will be removed in the plugin version named by ``REMOVE_IN``.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REMOVE_IN = "0.11.0"
"""Plugin version that removes this helper; ``tests/test_migrate_dataset.py`` fails from then on."""

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_yaw", "wrist_roll", "gripper")

# value_0.9 = factor * value_0.8. Both versions compute joint values from the motor angle with a
# per-joint direction (0.8 RS: 1, 1, -1, -1, -1, 1, 6 for public -> motor; 0.9 RS: all -1;
# 0.8 DM: -1, -1, 1, 1, 1, -1, -6 for actions only; 0.9 DM: all 1).
_OLD_TO_NEW = (-1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -6.0)
_UNCHANGED = (1.0,) * len(JOINTS)
FACTORS: dict[str, dict[str, tuple[float, ...]]] = {
    # RS 0.8 reported observations and took actions in the same (Seeed) frame.
    "rs": {"observation": _OLD_TO_NEW, "action": _OLD_TO_NEW},
    # DM 0.8 reported observations in raw motor degrees, which is already the 0.9 frame, and took
    # actions in the Seeed frame (issue #299).
    "dm": {"observation": _UNCHANGED, "action": _OLD_TO_NEW},
}

_NAME = re.compile(rf"^(?P<prefix>.*?)(?P<joint>{'|'.join(JOINTS)})\.(?P<kind>pos|vel)$")
_QUANTILE = re.compile(r"^q(?P<p>\d{2})$")
_MARKER = "meta/physicalai_rebot_b601_migration.json"
# In the old frame the shoulder lift and the gripper act on non-negative values; in the new one on
# non-positive values. A dataset whose actions never go above this (degrees) is already converted.
_OLD_FRAME_MIN_MAX_DEG = 1.0


class MigrationError(RuntimeError):
    """The dataset cannot be converted safely; the message says why."""


def _parquet() -> Any:  # noqa: ANN401 - optional module
    try:
        import pyarrow as pa  # noqa: PLC0415
        import pyarrow.parquet as pq  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - environment dependent
        msg = "pyarrow is required to convert datasets: pip install pyarrow (it comes with LeRobot)."
        raise MigrationError(msg) from exc
    return pa, pq


def feature_factors(info: dict[str, Any], variant: str) -> dict[str, np.ndarray]:
    """Per-element factors for every feature with reBot joint names; other features are left out.

    Returns:
        Feature key -> factor per element (1 for elements that are not reBot joint positions or
        velocities).
    """
    factors: dict[str, np.ndarray] = {}
    for key, feature in info.get("features", {}).items():
        names = feature.get("names")
        if not isinstance(names, list) or not str(feature.get("dtype", "")).startswith("float"):
            continue
        group = "action" if key == "action" else "observation" if key.startswith("observation.") else None
        if group is None:
            continue
        per_element = np.ones(len(names))
        matched = False
        for i, name in enumerate(names):
            match = _NAME.match(str(name))
            if match:
                per_element[i] = FACTORS[variant][group][JOINTS.index(match["joint"])]
                matched = True
        if matched and not np.allclose(per_element, 1.0):
            factors[key] = per_element
    return factors


def _detect_variant(info: dict[str, Any]) -> str:
    robot_type = str(info.get("robot_type", ""))
    if robot_type.lower().startswith("seeed"):
        # Seeed's LeRobot plugin records observations in raw motor degrees and actions in its own
        # frame; that is a different conversion.
        msg = f"datasets recorded with Seeed's LeRobot plugin ({robot_type!r}) are not supported"
        raise MigrationError(msg)
    match = re.search(r"rebot_b601_(rs|dm)", robot_type.lower())
    if match is None:
        msg = f"cannot tell the reBot variant from robot_type {robot_type!r}; pass --variant rs or --variant dm"
        raise MigrationError(msg)
    return match[1]


def _check_old_frame(root: Path, info: dict[str, Any]) -> None:
    """Refuse datasets that were already converted or that look like the new frame.

    Raises:
        MigrationError: If the dataset carries the migration marker or its actions look converted.
    """
    if (root / _MARKER).exists():
        msg = f"{root} was already converted ({_MARKER} exists)"
        raise MigrationError(msg)
    stats_path = root / "meta" / "stats.json"
    names = info["features"].get("action", {}).get("names", [])
    if not stats_path.exists() or not names:
        return  # nothing to judge by; the caller decided
    action_max = json.loads(stats_path.read_text())["action"]["max"]
    old = [
        action_max[i]
        for i, name in enumerate(names)
        if (m := _NAME.match(str(name))) and m["joint"] in {"shoulder_lift", "gripper"} and m["kind"] == "pos"
    ]
    if old and max(old) <= _OLD_FRAME_MIN_MAX_DEG:
        msg = (
            "the actions' shoulder lift and gripper never go above "
            f"{_OLD_FRAME_MIN_MAX_DEG} deg, as in the 0.9 frame: this dataset looks converted or recorded "
            "with 0.9 already. Pass --force to convert it anyway."
        )
        raise MigrationError(msg)


def scale_stats(stats: dict[str, Any], factor: np.ndarray) -> dict[str, Any]:
    """Scale one feature's statistics by a per-element factor.

    Negative factors swap min and max and mirror quantiles (q01 <-> q99).

    Returns:
        The converted statistics; unknown entries are copied unchanged.
    """
    out = dict(stats)
    neg = factor < 0

    def arr(key: str) -> np.ndarray:
        return np.asarray(stats[key], dtype=float)

    if "min" in stats and "max" in stats:
        lo, hi = arr("min") * factor, arr("max") * factor
        out["min"] = np.where(neg, hi, lo).tolist()
        out["max"] = np.where(neg, lo, hi).tolist()
    if "mean" in stats:
        out["mean"] = (arr("mean") * factor).tolist()
    if "std" in stats:
        out["std"] = (arr("std") * np.abs(factor)).tolist()
    quantiles = {int(m["p"]): key for key in stats if (m := _QUANTILE.match(key))}
    for p, key in quantiles.items():
        mirror = quantiles.get(100 - p, key)
        out[key] = np.where(neg, arr(mirror) * factor, arr(key) * factor).tolist()
    return out


def _scale_list_column(pa: Any, column: Any, factor: np.ndarray) -> Any:  # noqa: ANN401
    """Multiply a list<float> column element-wise by ``factor``.

    Returns:
        The scaled column, with the same Arrow type.
    """
    column = column.combine_chunks()
    values = column.flatten().to_numpy(zero_copy_only=False).reshape(len(column), len(factor))
    scaled = pa.array((values * factor).astype(values.dtype).ravel())
    if pa.types.is_fixed_size_list(column.type):
        return pa.FixedSizeListArray.from_arrays(scaled, len(factor), type=column.type)
    return pa.ListArray.from_arrays(column.offsets, scaled, type=column.type)


def _convert_data(pa: Any, pq: Any, path: Path, factors: dict[str, np.ndarray]) -> None:  # noqa: ANN401
    table = pq.read_table(path)
    for key, factor in factors.items():
        if key in table.column_names:
            index = table.column_names.index(key)
            table = table.set_column(index, table.schema.field(index), _scale_list_column(pa, table[key], factor))
    pq.write_table(table, path)


def _convert_episodes(pa: Any, pq: Any, path: Path, factors: dict[str, np.ndarray]) -> None:  # noqa: ANN401
    table = pq.read_table(path)
    for key, factor in factors.items():
        prefix = f"stats/{key}/"
        columns = {name[len(prefix) :]: name for name in table.column_names if name.startswith(prefix)}
        if not columns:
            continue
        rows = [{stat: table[col][i].as_py() for stat, col in columns.items()} for i in range(table.num_rows)]
        converted = [scale_stats(row, factor) for row in rows]
        for stat, col in columns.items():
            index = table.column_names.index(col)
            field = table.schema.field(index)
            table = table.set_column(index, field, pa.array([row[stat] for row in converted], type=field.type))
    pq.write_table(table, path)


def migrate(source: Path, destination: Path, *, variant: str | None = None, force: bool = False) -> dict[str, Any]:
    """Write a copy of ``source`` with reBot joint values in the plugin 0.9 frame.

    Returns:
        A summary (variant, factors per feature, files converted), also written to the copy.

    Raises:
        MigrationError: If the dataset is not a LeRobot v3 reBot dataset in the old frame, or the
            destination exists.
    """
    warnings.warn(
        f"physicalai_rebot_b601_plugin.migrate_dataset will be removed in plugin version {REMOVE_IN}.",
        DeprecationWarning,
        stacklevel=2,
    )
    pa, pq = _parquet()
    info_path = source / "meta" / "info.json"
    if not info_path.is_file():
        msg = f"{source} is not a LeRobot dataset (no meta/info.json)"
        raise MigrationError(msg)
    info = json.loads(info_path.read_text())
    if not str(info.get("codebase_version", "")).startswith("v3"):
        msg = f"only LeRobot v3 datasets are supported, got {info.get('codebase_version')!r}"
        raise MigrationError(msg)
    if destination.exists():
        msg = f"{destination} exists; choose a new directory"
        raise MigrationError(msg)
    variant = variant or _detect_variant(info)
    factors = feature_factors(info, variant)
    if not factors:
        msg = "no reBot joint positions or velocities found in the dataset's features"
        raise MigrationError(msg)
    if not force:
        _check_old_frame(source, info)

    shutil.copytree(source, destination)
    data_files = sorted((destination / "data").rglob("*.parquet"))
    for path in data_files:
        _convert_data(pa, pq, path, factors)
    episode_files = sorted((destination / "meta" / "episodes").rglob("*.parquet"))
    for path in episode_files:
        _convert_episodes(pa, pq, path, factors)
    stats_path = destination / "meta" / "stats.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text())
        for key, factor in factors.items():
            if key in stats:
                stats[key] = scale_stats(stats[key], factor)
        stats_path.write_text(json.dumps(stats, indent=4))

    summary = {
        "converted_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "source": str(source),
        "variant": variant,
        "from_frame": "physicalai-rebot-b601-plugin <= 0.8",
        "to_frame": "physicalai-rebot-b601-plugin >= 0.9 (LeRobot)",
        "factors": {key: factor.tolist() for key, factor in factors.items()},
        "data_files": len(data_files),
        "episode_files": len(episode_files),
    }
    (destination / _MARKER).write_text(json.dumps(summary, indent=2))
    return summary


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(
        prog="python -m physicalai_rebot_b601_plugin.migrate_dataset",
        description=f"Convert a reBot B601 dataset recorded with plugin 0.8 or older to the 0.9 joint frame. "
        f"Temporary: removed in plugin version {REMOVE_IN}.",
    )
    parser.add_argument("source", type=Path, help="dataset recorded with plugin 0.8 or older (left unchanged)")
    parser.add_argument("destination", type=Path, help="new directory for the converted copy")
    parser.add_argument("--variant", choices=sorted(FACTORS), help="reBot variant, if robot_type does not tell")
    parser.add_argument("--force", action="store_true", help="convert even if the data looks converted")
    args = parser.parse_args(argv)
    sys.stderr.write(f"Note: this helper is temporary and will be removed in plugin version {REMOVE_IN}.\n")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            summary = migrate(args.source, args.destination, variant=args.variant, force=args.force)
    except MigrationError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 1
    sys.stdout.write(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
