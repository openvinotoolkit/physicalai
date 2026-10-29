# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import math
import struct
from pathlib import Path

import numpy as np
import pytest
import yaml

from physicalai_rosbag_plugin import MappingConfig, convert_rosbags
from physicalai_rosbag_plugin.containers import preflight
from physicalai_rosbag_plugin.converter import _publish
from physicalai_rosbag_plugin.reader import inspect_episode, sample_episode


@pytest.mark.parametrize("storage", ["sqlite3", "mcap"])
def test_round_trip(tmp_path, demo, storage):
    bags = tmp_path / "bags"
    bags.mkdir()
    demo["write_demo"](bags / "a", storage)
    demo["write_demo"](bags / "b", storage)
    config = MappingConfig.model_validate(demo["demo_mapping"]())
    result = convert_rosbags(bags, config, tmp_path / "dataset", "local/test")
    assert (result.episodes, result.frames) == (2, 16)
    info = json.loads((result.output / "meta/info.json").read_text())
    assert info["features"]["action"]["names"] == ["shoulder_pan.pos", "gripper.pos"]
    assert info["features"]["observation.state"]["names"] == ["shoulder_pan.pos", "gripper.pos"]
    assert info["features"]["observation.images.front"]["dtype"] == "video"
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset("local/test", root=result.output, video_backend="pyav")
    np.testing.assert_allclose(dataset[3]["observation.state"], [0.03 * 180 / math.pi, 0.25], rtol=1e-6)
    np.testing.assert_allclose(dataset[3]["action"], [0.06 * 180 / math.pi, 0.265], rtol=1e-6)
    assert dataset[8]["frame_index"].item() == 0
    assert dataset[8]["timestamp"].item() == 0
    assert not list(tmp_path.glob(".dataset.rosbag-*"))


@pytest.mark.parametrize("duplicate", ["identical", "conflict"])
def test_duplicate_timestamp(tmp_path, demo, duplicate):
    bag = tmp_path / "bag"
    demo["write_demo"](bag, duplicate=duplicate)
    config = MappingConfig.model_validate(demo["demo_mapping"]())
    if duplicate == "conflict":
        with pytest.raises(ValueError, match="conflicting duplicate"):
            inspect_episode(bag, config)
    else:
        assert inspect_episode(bag, config).frames == 8


def test_stale_samples_cleanup(tmp_path, demo):
    bag = tmp_path / "bag"
    demo["write_demo"](bag, times=[0, 500_000_000])
    config = MappingConfig.model_validate(demo["demo_mapping"]())
    plan = inspect_episode(bag, config)
    frames = sample_episode(plan, config)
    next(frames)
    next(frames)  # Exactly max_age_ms is permitted.
    with pytest.raises(ValueError, match="stale"):
        next(frames)
    with pytest.raises(ValueError, match="stale"):
        convert_rosbags(bag, config, tmp_path / "out", "local/test")
    assert not (tmp_path / "out").exists()
    assert not list(tmp_path.glob(".out.rosbag-*"))


def test_missing_topic_and_frame_limit(tmp_path, demo):
    bag = tmp_path / "bag"
    demo["write_demo"](bag, missing="/command")
    config = MappingConfig.model_validate(demo["demo_mapping"]())
    with pytest.raises(ValueError, match="missing required"):
        inspect_episode(bag, config)
    other = tmp_path / "other"
    demo["write_demo"](other)
    config = config.model_copy(update={"limits": config.limits.model_copy(update={"episode_frames": 2})})
    with pytest.raises(ValueError, match="episode_frames"):
        inspect_episode(other, config)


@pytest.mark.parametrize("kind", ["directory", "symlink"])
def test_collision_does_not_replace(tmp_path, kind):
    source = tmp_path / "source"
    source.mkdir()
    (source / "data").write_text("new")
    destination = tmp_path / "destination"
    if kind == "directory":
        destination.mkdir()
    else:
        destination.symlink_to(tmp_path / "missing")
    with pytest.raises(FileExistsError):
        _publish(source, destination)
    assert (source / "data").read_text() == "new"
    assert destination.is_symlink() if kind == "symlink" else destination.is_dir()


def test_reject_compression_and_path_escape(tmp_path, demo):
    bag = tmp_path / "bag"
    demo["write_demo"](bag)
    config = MappingConfig.model_validate(demo["demo_mapping"]())
    path = bag / "metadata.yaml"
    metadata = yaml.safe_load(path.read_text())
    metadata["rosbag2_bagfile_information"]["compression_mode"] = "file"
    path.write_text(yaml.safe_dump(metadata))
    with pytest.raises(ValueError, match="compression"):
        preflight(bag, config.limits)
    metadata["rosbag2_bagfile_information"]["compression_mode"] = ""
    metadata["rosbag2_bagfile_information"]["relative_file_paths"] = ["../escape.db3"]
    path.write_text(yaml.safe_dump(metadata))
    with pytest.raises(ValueError, match="escapes"):
        preflight(bag, config.limits)


def test_final_path_validation_failure_removes_only_own_output(tmp_path, demo, monkeypatch):
    import physicalai_rosbag_plugin.converter as converter

    bag = tmp_path / "bag"
    demo["write_demo"](bag)
    config = MappingConfig.model_validate(demo["demo_mapping"]())
    output = tmp_path / "out"
    original = converter._validate_dataset
    calls = []

    def validate(path, *args):
        calls.append(path)
        if path == output:
            raise ValueError("injected final-path read failure")
        return original(path, *args)

    monkeypatch.setattr(converter, "_validate_dataset", validate)
    with pytest.raises(ValueError, match="final-path"):
        convert_rosbags(bag, config, output, "local/test")
    assert len(calls) == 2
    assert not output.exists()


def test_mcap_compression_checked_before_reader_open(tmp_path, demo):
    from physicalai_rosbag_plugin.containers import _mcap

    config = MappingConfig.model_validate(demo["demo_mapping"]())
    chunk = struct.pack("<QQQI", 0, 1, 100, 0) + struct.pack("<I", 4) + b"zstd" + struct.pack("<Q", 0)
    file = tmp_path / "bad.mcap"
    magic = b"\x89MCAP0\r\n"
    file.write_bytes(magic + struct.pack("<BQ", 6, len(chunk)) + chunk + magic)
    with pytest.raises(ValueError, match="Compressed MCAP"):
        _mcap(file, config.limits)


@pytest.mark.parametrize("storage", ["sqlite3", "mcap"])
def test_storage_timestamp_regression(tmp_path, demo, storage):
    bag = tmp_path / "bag"
    demo["write_demo"](bag, storage, times=[100_000_000, 0])
    config = MappingConfig.model_validate(demo["demo_mapping"]())
    with pytest.raises(ValueError, match="regression"):
        preflight(bag, config.limits)


def test_output_cannot_mutate_source(tmp_path, demo):
    bag = tmp_path / "bag"
    demo["write_demo"](bag)
    config = MappingConfig.model_validate(demo["demo_mapping"]())
    with pytest.raises(ValueError, match="outside"):
        convert_rosbags(bag, config, bag / "converted", "local/test")


def test_metadata_symlink_escape(tmp_path, demo):
    bag = tmp_path / "bag"
    demo["write_demo"](bag)
    metadata = bag / "metadata.yaml"
    external = tmp_path / "metadata.yaml"
    metadata.rename(external)
    metadata.symlink_to(external)
    config = MappingConfig.model_validate(demo["demo_mapping"]())
    with pytest.raises(ValueError, match="metadata"):
        preflight(bag, config.limits)


@pytest.mark.parametrize("storage", ["sqlite3", "mcap"])
def test_split_recording_stays_one_episode(tmp_path, demo, storage):
    import shutil

    first, second = tmp_path / "first", tmp_path / "second"
    demo["write_demo"](first, storage, times=[0, 100_000_000])
    demo["write_demo"](second, storage, times=[200_000_000, 300_000_000])
    metadata = yaml.safe_load((first / "metadata.yaml").read_text())
    info = metadata["rosbag2_bagfile_information"]
    other = yaml.safe_load((second / "metadata.yaml").read_text())["rosbag2_bagfile_information"]
    second_name = f"second.{'db3' if storage == 'sqlite3' else 'mcap'}"
    shutil.copyfile(second / other["relative_file_paths"][0], first / second_name)
    info["relative_file_paths"].append(second_name)
    info["message_count"] += other["message_count"]
    info["duration"]["nanoseconds"] = 300_000_000
    for entry in info["topics_with_message_count"]:
        entry["message_count"] *= 2
    if "files" in info:
        entry = other["files"][0]
        entry["path"] = second_name
        info["files"].append(entry)
    (first / "metadata.yaml").write_text(yaml.safe_dump(metadata))
    config = MappingConfig.model_validate(demo["demo_mapping"]())
    result = convert_rosbags(first, config, tmp_path / "converted", "local/split")
    assert (result.episodes, result.frames) == (1, 4)
    info["relative_file_paths"].reverse()
    (first / "metadata.yaml").write_text(yaml.safe_dump(metadata))
    with pytest.raises(ValueError, match="regression"):
        preflight(first, config.limits)


@pytest.mark.parametrize("failure", [RuntimeError("encoder failure"), KeyboardInterrupt()])
def test_interruption_or_encoder_failure_cleanup(tmp_path, demo, monkeypatch, failure):
    import physicalai_rosbag_plugin.converter as converter

    bag = tmp_path / "bag"
    demo["write_demo"](bag)
    config = MappingConfig.model_validate(demo["demo_mapping"]())

    def fail(root):
        (root / "partial").write_text("only owned staging")
        raise failure

    monkeypatch.setattr(converter, "_encoder_probe", fail)
    with pytest.raises(type(failure)):
        convert_rosbags(bag, config, tmp_path / "out", "local/test")
    assert not (tmp_path / "out").exists()
    assert not list(tmp_path.glob(".out.rosbag-*"))


def test_total_frame_limit(tmp_path, demo):
    bag = tmp_path / "bag"
    demo["write_demo"](bag)
    config = MappingConfig.model_validate({**demo["demo_mapping"](), "limits": {"total_frames": 2}})
    with pytest.raises(ValueError, match="total_frames"):
        convert_rosbags(bag, config, tmp_path / "out", "local/test")
    assert not list(tmp_path.glob(".out.rosbag-*"))


def test_storage_payload_limit(tmp_path, demo):
    bag = tmp_path / "bag"
    demo["write_demo"](bag)
    config = MappingConfig.model_validate({**demo["demo_mapping"](), "limits": {"message_bytes": 16}})
    with pytest.raises(ValueError, match="message_bytes"):
        preflight(bag, config.limits)


def test_causal_sampling_and_disjoint_intervals(tmp_path, demo):
    import sqlite3
    from contextlib import closing

    bag = tmp_path / "bag"
    demo["write_demo"](bag, times=[0, 150_000_000, 300_000_000])
    config = MappingConfig.model_validate({**demo["demo_mapping"](), "max_age_ms": 150})
    plan = inspect_episode(bag, config)
    frames = list(sample_episode(plan, config))
    # Tick 100ms must keep the command at 0, never take the future sample at 150ms.
    np.testing.assert_allclose(
        [frame["action"][0] for frame in frames], np.array([0, 0, 0.02, 0.04]) * 180 / math.pi, rtol=1e-6
    )
    with closing(sqlite3.connect(next(bag.glob("*.db3")))) as connection:
        connection.execute(
            "UPDATE messages SET timestamp=timestamp+1000000000 WHERE topic_id=(SELECT id FROM topics WHERE name='/command')"
        )
        connection.commit()
    with pytest.raises(ValueError, match="no common interval"):
        inspect_episode(bag, config)


def test_changing_joint_dimensions(tmp_path, demo):
    import sqlite3
    from contextlib import closing
    from rosbags.typesys import Stores, get_typestore

    bag = tmp_path / "bag"
    demo["write_demo"](bag)
    store = get_typestore(Stores.ROS2_HUMBLE)
    kind = "sensor_msgs/msg/JointState"
    with closing(sqlite3.connect(next(bag.glob("*.db3")))) as connection:
        row_id, raw = connection.execute(
            "SELECT id,data FROM messages WHERE topic_id=(SELECT id FROM topics WHERE name='/state') ORDER BY id DESC LIMIT 1"
        ).fetchone()
        message = store.deserialize_cdr(raw, kind)
        message.name.append("extra")
        message.position = np.append(message.position, 0.0)
        connection.execute("UPDATE messages SET data=? WHERE id=?", (bytes(store.serialize_cdr(message, kind)), row_id))
        connection.commit()
    config = MappingConfig.model_validate(demo["demo_mapping"]())
    with pytest.raises(ValueError, match="changing numeric dimensions"):
        inspect_episode(bag, config)


@pytest.mark.parametrize("payload", [b"bad", b"\x89MCAP0\r\n" + struct.pack("<BQ", 5, 999) + b"\x89MCAP0\r\n"])
def test_malformed_mcap_lengths(tmp_path, demo, payload):
    from physicalai_rosbag_plugin.containers import _mcap

    file = tmp_path / "malformed.mcap"
    file.write_bytes(payload)
    config = MappingConfig.model_validate(demo["demo_mapping"]())
    with pytest.raises(ValueError, match="magic|length"):
        _mcap(file, config.limits)


def test_concurrent_destination_survives_cleanup(tmp_path, demo, monkeypatch):
    import physicalai_rosbag_plugin.converter as converter

    bag = tmp_path / "bag"
    demo["write_demo"](bag)
    output = tmp_path / "out"
    original = converter._publish

    def publish(source, destination):
        destination.mkdir()
        (destination / "other-owner").write_text("keep")
        original(source, destination)

    monkeypatch.setattr(converter, "_publish", publish)
    config = MappingConfig.model_validate(demo["demo_mapping"]())
    with pytest.raises(FileExistsError):
        convert_rosbags(bag, config, output, "local/test")
    assert (output / "other-owner").read_text() == "keep"
    assert not list(tmp_path.glob(".out.rosbag-*"))


def test_finalization_failure_does_not_publish(tmp_path, demo, monkeypatch):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    bag = tmp_path / "bag"
    demo["write_demo"](bag)
    original = LeRobotDataset.finalize

    def fail(self):
        original(self)
        raise RuntimeError("injected finalization failure")

    monkeypatch.setattr(LeRobotDataset, "finalize", fail)
    config = MappingConfig.model_validate(demo["demo_mapping"]())
    with pytest.raises(RuntimeError, match="finalization"):
        convert_rosbags(bag, config, tmp_path / "out", "local/test")
    assert not (tmp_path / "out").exists()
    assert not list(tmp_path.glob(".out.rosbag-*"))


@pytest.mark.parametrize("storage", ["sqlite3", "mcap"])
def test_compressed_camera_messages_in_real_bag(tmp_path, demo, storage):
    bag = tmp_path / "bag"
    demo["write_demo"](bag, storage=storage, compressed=True)
    mapping = demo["demo_mapping"]()
    for camera in mapping["cameras"]:
        camera["message_type"] = "sensor_msgs/msg/CompressedImage"
    config = MappingConfig.model_validate(mapping)
    result = convert_rosbags(bag, config, tmp_path / "out", "local/compressed-images")
    assert result.frames == 8
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset("local/compressed-images", root=result.output, video_backend="pyav")
    frame = dataset[0]
    for camera in ("front", "wrist"):
        image = frame[f"observation.images.{camera}"]
        assert tuple(image.shape) == (3, 64, 64)
        assert float(image[0].mean()) > float(image[2].mean())


@pytest.mark.parametrize("storage", ["sqlite3", "mcap"])
def test_explicit_uncompressed_mode(tmp_path, demo, storage):
    bag = tmp_path / "bag"
    demo["write_demo"](bag, storage=storage)
    path = bag / "metadata.yaml"
    metadata = yaml.safe_load(path.read_text())
    metadata["rosbag2_bagfile_information"]["compression_mode"] = "none"
    path.write_text(yaml.safe_dump(metadata))
    config = MappingConfig.model_validate(demo["demo_mapping"]())
    result = convert_rosbags(bag, config, tmp_path / "out", "local/uncompressed")
    assert result.frames == 8


@pytest.mark.parametrize("storage", ["sqlite3", "mcap"])
def test_reject_storage_path_that_reader_would_reinterpret(tmp_path, demo, storage):
    import shutil

    bag = tmp_path / "bag"
    demo["write_demo"](bag, storage=storage)
    path = bag / "metadata.yaml"
    metadata = yaml.safe_load(path.read_text())
    name = metadata["rosbag2_bagfile_information"]["relative_file_paths"][0]
    (bag / "nested").mkdir()
    shutil.copyfile(bag / name, bag / "nested" / name)
    metadata["rosbag2_bagfile_information"]["relative_file_paths"] = [f"nested/{name}"]
    path.write_text(yaml.safe_dump(metadata))
    # rosbags 0.11.9 opens bag/name, not bag/nested/name. Never validate one and read the other.
    config = MappingConfig.model_validate(demo["demo_mapping"]())
    with pytest.raises(ValueError, match="Nested storage"):
        preflight(bag, config.limits)


def test_output_outside_resolved_recordings(tmp_path, demo):
    bag = tmp_path / "original"
    demo["write_demo"](bag)
    aliases = tmp_path / "aliases"
    aliases.mkdir()
    (aliases / "episode").symlink_to(bag, target_is_directory=True)
    config = MappingConfig.model_validate(demo["demo_mapping"]())
    with pytest.raises(ValueError, match="outside"):
        convert_rosbags(aliases, config, bag / "out", "local/test")
    assert not (bag / "out").exists()


def test_numeric_validation_rejects_nonfinite_timestamps(tmp_path, demo):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from physicalai_rosbag_plugin.converter import _features, _validate_numeric
    from physicalai_rosbag_plugin.reader import EpisodePlan

    data = tmp_path / "data" / "chunk-000"
    data.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "episode_index": [0],
                "frame_index": [0],
                "timestamp": [float("nan")],
                "observation.state": [[0.0, 0.0]],
                "action": [[0.0, 0.0]],
            }
        ),
        data / "file-000.parquet",
    )
    config = MappingConfig.model_validate(demo["demo_mapping"]())
    plan = EpisodePlan(tmp_path, 0, 0, 1, {"front": (64, 64, 3), "wrist": (64, 64, 3)})
    with pytest.raises(ValueError, match="timestamp"):
        _validate_numeric(tmp_path, [plan], config, _features(config, plan.shapes))


@pytest.mark.parametrize("missing", ["video", "episodes"])
def test_missing_published_files_fail_without_hub_fallback(tmp_path, demo, monkeypatch, missing):
    import physicalai_rosbag_plugin.converter as converter
    from lerobot.datasets import dataset_metadata, lerobot_dataset

    bag = tmp_path / "bag"
    demo["write_demo"](bag)
    config = MappingConfig.model_validate(demo["demo_mapping"]())
    output = tmp_path / "out"
    original = converter._validate_dataset

    def no_hub(*args, **kwargs):
        raise AssertionError("Local validation attempted Hub fallback")

    def validate(path, *args):
        if path == output:
            directory, pattern = ("videos", "*.mp4") if missing == "video" else ("meta/episodes", "*.parquet")
            next((path / directory).rglob(pattern)).unlink()
        return original(path, *args)

    monkeypatch.setattr(lerobot_dataset, "get_safe_version", no_hub)
    monkeypatch.setattr(dataset_metadata, "get_safe_version", no_hub)
    monkeypatch.setattr(converter, "_validate_dataset", validate)
    with pytest.raises(ValueError, match="Missing finalized"):
        convert_rosbags(bag, config, output, "local/test")
    assert not output.exists()


def test_slow_encoder_preserves_all_frames(tmp_path, demo, monkeypatch):
    import time
    import av
    from lerobot.datasets.video_utils import _CameraEncoderThread

    original = _CameraEncoderThread.run

    def slow_start(self):
        time.sleep(0.4)  # Exceeds the live recorder's 100 ms queue timeout.
        original(self)

    monkeypatch.setattr(_CameraEncoderThread, "run", slow_start)
    bag = tmp_path / "bag"
    demo["write_demo"](bag)
    config = MappingConfig.model_validate(demo["demo_mapping"]())
    result = convert_rosbags(bag, config, tmp_path / "out", "local/slow-encoder")
    assert result.frames == 8
    for video in (result.output / "videos").rglob("*.mp4"):
        with av.open(str(video)) as container:
            assert len(list(container.decode(video=0))) == 8


def test_single_frame_episode_keeps_camera_statistics(tmp_path, demo):
    bag = tmp_path / "bag"
    demo["write_demo"](bag, frames=1)
    config = MappingConfig.model_validate(demo["demo_mapping"]())
    result = convert_rosbags(bag, config, tmp_path / "out", "local/single-frame")
    stats = json.loads((result.output / "meta/stats.json").read_text())
    assert result.frames == 1
    for camera in ("front", "wrist"):
        assert f"observation.images.{camera}" in stats
        mean = np.array(stats[f"observation.images.{camera}"]["mean"]).reshape(-1)
        np.testing.assert_allclose(mean, [30 / 255, 31.5 / 255, 0], atol=1e-5)


def test_encoder_thread_crash_cleans_output(tmp_path, demo, monkeypatch):
    from lerobot.datasets.video_utils import _CameraEncoderThread

    def fail(self):
        self.result_queue.put(("error", "injected encoder thread failure"))

    monkeypatch.setattr(_CameraEncoderThread, "run", fail)
    bag = tmp_path / "bag"
    demo["write_demo"](bag)
    config = MappingConfig.model_validate(demo["demo_mapping"]())
    with pytest.raises(RuntimeError, match="encoder"):
        convert_rosbags(bag, config, tmp_path / "out", "local/test")
    assert not (tmp_path / "out").exists()
    assert not list(tmp_path.glob(".out.rosbag-*"))
