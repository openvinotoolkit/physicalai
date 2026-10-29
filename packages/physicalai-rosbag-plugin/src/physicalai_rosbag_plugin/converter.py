# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Transactional conversion into a finalized, locally readable dataset."""

from __future__ import annotations

import ctypes
import json
import os
import shutil
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from physicalai_rosbag_plugin.config import MappingConfig
    from physicalai_rosbag_plugin.reader import EpisodePlan


@dataclass(frozen=True)
class ConversionResult:
    """Published dataset location and conversion totals."""

    output: Path
    episodes: int
    frames: int


def _publish(source: Path, destination: Path) -> None:
    """Atomically publish without replacing an existing destination.

    Raises:
        OSError: Atomic publication failed, including destination collisions.
    """
    if sys.platform == "linux":
        libc = ctypes.CDLL(None, use_errno=True)
        rename = libc.renameat2
        rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        if rename(-100, os.fsencode(source), -100, os.fsencode(destination), 1) != 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error), str(destination))
    elif sys.platform == "win32":
        source.rename(destination)
    else:
        msg = "Safe no-replace publication currently supports Linux and Windows"
        raise NotImplementedError(msg)


def _encoder_probe(root: Path) -> None:
    import av  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    probe = root / "encoder-probe.mp4"
    try:
        with av.open(str(probe), "w") as container:
            stream = container.add_stream("libx264", rate=30)
            stream.width = stream.height = 32
            stream.pix_fmt = "yuv420p"
            frame = av.VideoFrame.from_ndarray(np.zeros((32, 32, 3), dtype=np.uint8), format="rgb24")
            for packet in (*stream.encode(frame), *stream.encode()):
                container.mux(packet)
    finally:
        probe.unlink(missing_ok=True)


def _features(config: MappingConfig, shapes: dict[str, tuple[int, ...]]) -> dict[str, Any]:
    features: dict[str, Any] = {
        key: {"dtype": "float32", "shape": (len(specs),), "names": [spec.name for spec in specs]}
        for key, specs in (("observation.state", config.state), ("action", config.action))
    }
    for name, shape in shapes.items():
        if shape[0] % 2 or shape[1] % 2:
            msg = f"Camera {name}: H.264 requires even width and height; resizing is not implicit"
            raise ValueError(msg)
        features[f"observation.images.{name}"] = {
            "dtype": "video",
            "shape": shape,
            "names": ["height", "width", "channels"],
        }
    return features


_TIMESTAMP_TOLERANCE = 1e-4


def _validate_numeric(
    path: Path,
    plans: list[EpisodePlan],
    config: MappingConfig,
    expected: dict[str, Any],
) -> None:
    import numpy as np  # noqa: PLC0415
    import pyarrow.parquet as pq  # noqa: PLC0415

    # Scan numerical data in bounded batches, including every timestamp and episode boundary.
    episode = 0
    offset = 0
    for file in sorted((path / "data").rglob("file-*.parquet")):
        for batch in pq.ParquetFile(file).iter_batches(batch_size=4096):
            rows = batch.to_pydict()
            for index, episode_index in enumerate(rows["episode_index"]):
                if episode >= len(plans) or episode_index != episode or rows["frame_index"][index] != offset:
                    msg = "Invalid episode/frame index in finalized dataset"
                    raise ValueError(msg)
                expected_time = np.float32(offset / config.fps)
                timestamp = rows["timestamp"][index]
                if not np.isfinite(timestamp) or abs(timestamp - expected_time) > _TIMESTAMP_TOLERANCE:
                    msg = "Invalid finalized timestamp"
                    raise ValueError(msg)
                for key in ("observation.state", "action"):
                    if (
                        not np.isfinite(rows[key][index]).all()
                        or np.atleast_1d(rows[key][index]).size != expected[key]["shape"][0]
                    ):
                        msg = f"Invalid finalized numeric values: {key}"
                        raise ValueError(msg)
                offset += 1
                if offset == plans[episode].frames:
                    episode += 1
                    offset = 0
    if episode != len(plans) or offset:
        msg = "Incomplete finalized numeric data"
        raise ValueError(msg)


def _validate_dataset(path: Path, repo_id: str, plans: list[EpisodePlan], config: MappingConfig) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: PLC0415

    info = json.loads((path / "meta/info.json").read_text())
    if info["total_episodes"] != len(plans) or info["total_frames"] != sum(plan.frames for plan in plans):
        msg = "Finalized dataset counts differ from the conversion plan"
        raise ValueError(msg)
    expected = _features(config, plans[0].shapes)
    for key, feature in expected.items():
        actual = info["features"][key]
        if (
            actual["dtype"] != feature["dtype"]
            or actual["names"] != feature["names"]
            or tuple(actual["shape"]) != feature["shape"]
        ):
            msg = f"Finalized feature metadata differs: {key}"
            raise ValueError(msg)
    for marker in ("meta/stats.json", "meta/tasks.parquet"):
        if not (path / marker).is_file():
            msg = f"Missing finalized metadata: {marker}"
            raise ValueError(msg)
    _validate_numeric(path, plans, config, expected)
    _validate_episode_files(path, repo_id, plans, config)
    dataset = LeRobotDataset(repo_id=repo_id, root=path, video_backend="pyav")
    start = 0
    for plan in plans:
        for index in {start, start + plan.frames - 1}:
            frame = dataset[index]
            for camera in config.cameras:
                value = frame[f"observation.images.{camera.name}"]
                height, width, _ = plan.shapes[camera.name]
                if tuple(value.shape) != (3, height, width):
                    msg = f"Invalid decoded camera dimensions: {camera.name}"
                    raise ValueError(msg)
        start += plan.frames


def _validate_episode_files(path: Path, repo_id: str, plans: list[EpisodePlan], config: MappingConfig) -> None:
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata  # noqa: PLC0415

    if not any((path / "meta/episodes").rglob("file-*.parquet")):
        msg = "Missing finalized episode metadata"
        raise ValueError(msg)
    metadata = LeRobotDatasetMetadata(repo_id=repo_id, root=path)
    if metadata.episodes is None or len(metadata.episodes) != len(plans):
        msg = "Finalized episode metadata count differs"
        raise ValueError(msg)
    for index, plan in enumerate(plans):
        episode = metadata.episodes[index]
        if episode["episode_index"] != index or episode["length"] != plan.frames:
            msg = "Finalized episode metadata boundaries differ"
            raise ValueError(msg)
        files = [metadata.get_data_file_path(index)]
        files.extend(
            metadata.get_video_file_path(index, f"observation.images.{camera.name}") for camera in config.cameras
        )
        for relative in files:
            file = (path / relative).resolve()
            if not file.is_relative_to(path.resolve()) or not file.is_file():
                msg = f"Missing finalized local file: {relative}"
                raise ValueError(msg)


def _check_unchanged(identities: tuple[tuple[Path, int, int], ...]) -> None:
    for file, size, modified in identities:
        stat = file.stat()
        if stat.st_size != size or stat.st_mtime_ns != modified:
            msg = f"Input changed during conversion: {file}"
            raise ValueError(msg)


def _write_dataset(
    dataset_path: Path,
    repo_id: str,
    plans: list[EpisodePlan],
    mapping: MappingConfig,
    features: dict[str, Any],
    identities: dict[Path, tuple[tuple[Path, int, int], ...]],
) -> None:
    from lerobot.configs import RGBEncoderConfig  # noqa: PLC0415

    from physicalai_rosbag_plugin.encoding import OfflineLeRobotDataset  # noqa: PLC0415
    from physicalai_rosbag_plugin.reader import sample_episode  # noqa: PLC0415

    writer = OfflineLeRobotDataset.create(
        repo_id=repo_id,
        root=dataset_path,
        fps=mapping.fps,
        features=features,
        robot_type=mapping.robot_type,
        use_videos=True,
        rgb_encoder=RGBEncoderConfig(vcodec="h264", pix_fmt="yuv420p", crf=18),
        streaming_encoding=True,
        encoder_queue_maxsize=2,
        encoder_threads=1,
    )
    try:
        for plan in plans:
            _check_unchanged(identities[plan.path])
            logger.info("Converting {} ({} frames)", plan.path.name, plan.frames)
            for frame in sample_episode(plan, mapping):
                writer.add_frame(frame)
            writer.save_episode()
        writer.finalize()
    except BaseException:
        try:
            writer.finalize()
        except Exception:  # noqa: BLE001 - cleanup must preserve the original error.
            logger.exception("Could not close dataset writer")
        raise


def convert_rosbags(
    input_dir: Path,
    mapping: MappingConfig,
    output: Path,
    repo_id: str,
) -> ConversionResult:
    """Convert required topics and return only after reopening the published dataset.

    Inputs are read-only. Existing output paths are never replaced. Errors propagate
    after removal of this invocation's temporary output; no partial episode is published.

    Returns:
        The published path and conversion counts.

    Raises:
        FileExistsError: Output already exists.
        ValueError: Configuration, source data, or finalized output is invalid.
    """
    from physicalai_rosbag_plugin.containers import preflight, recordings  # noqa: PLC0415
    from physicalai_rosbag_plugin.reader import inspect_episode  # noqa: PLC0415

    # Resolve the parent only: resolving the final component would follow an existing symlink.
    output = output.parent.resolve() / output.name
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    if not repo_id.strip():
        msg = "repo_id must be nonempty"
        raise ValueError(msg)
    paths = recordings(input_dir)
    if any(output.is_relative_to(path) for path in [input_dir.resolve(), *paths]):
        msg = "Output must be outside every resolved source recording directory"
        raise ValueError(msg)
    identities = {path: preflight(path, mapping.limits) for path in paths}
    plans = [inspect_episode(path, mapping) for path in paths]
    total = sum(plan.frames for plan in plans)
    if total > mapping.limits.total_frames:
        msg = f"total_frames limit exceeded ({total})"
        raise ValueError(msg)
    if any(plan.shapes != plans[0].shapes for plan in plans):
        msg = "Camera shapes differ across episodes"
        raise ValueError(msg)
    if len(mapping.action) == 1:
        logger.warning("Studio 0.3 episode preview cannot display singleton actions produced by LeRobot 0.6.0")
    features = _features(mapping, plans[0].shapes)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.rosbag-", dir=output.parent))
    dataset_path = staging / "dataset"
    published_identity: tuple[int, int] | None = None
    try:
        _encoder_probe(staging)
        _write_dataset(dataset_path, repo_id, plans, mapping, features, identities)
        report = {
            "mapping": mapping.model_dump(mode="json"),
            "episodes": [{**asdict(plan), "path": str(plan.path)} for plan in plans],
        }
        (dataset_path / "conversion.json").write_text(json.dumps(report, indent=2) + "\n")
        _validate_dataset(dataset_path, repo_id, plans, mapping)
        stat = dataset_path.stat()
        own_identity = (stat.st_dev, stat.st_ino)
        published_identity = own_identity
        _publish(dataset_path, output)
        _validate_dataset(output, repo_id, plans, mapping)
        return ConversionResult(output, len(plans), total)
    except BaseException:
        if published_identity is not None and output.exists() and not output.is_symlink():
            stat = output.stat()
            if (stat.st_dev, stat.st_ino) == published_identity:
                shutil.rmtree(output)
        raise
    finally:
        shutil.rmtree(staging)
