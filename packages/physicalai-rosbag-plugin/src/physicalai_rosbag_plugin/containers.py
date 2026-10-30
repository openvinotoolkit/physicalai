# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Preflight storage before rosbags can decompress or allocate message payloads."""

from __future__ import annotations

import sqlite3
import struct
from contextlib import closing
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO

import yaml

if TYPE_CHECKING:
    from physicalai_rosbag_plugin.config import ResourceLimits

_MAGIC = b"\x89MCAP0\r\n"
_MAX_METADATA = 16 * 1024 * 1024
_CHUNK = 6
_MESSAGE = 5
_CHANNEL = 4
_MESSAGE_HEADER_BYTES = 22


def recordings(source: Path) -> list[Path]:
    """Discover one recording or immediate child recordings in name order.

    Returns:
        Ordered absolute recording paths.

    Raises:
        ValueError: No recordings exist or discovery is ambiguous.
    """
    if not source.is_dir():
        msg = f"Expected a ROS 2 recording directory: {source}"
        raise ValueError(msg)
    if (source / "metadata.yaml").is_file():
        return [source.resolve()]
    paths = [p.resolve() for p in sorted(source.iterdir()) if p.is_dir() and (p / "metadata.yaml").is_file()]
    if not paths or len(set(paths)) != len(paths):
        msg = "No recordings found, or duplicate recording symlinks"
        raise ValueError(msg)
    return paths


def _read(stream: BinaryIO, size: int, end: int) -> bytes:
    if size < 0 or size > end - stream.tell():
        msg = "Malformed MCAP length exceeds containing record"
        raise ValueError(msg)
    data = stream.read(size)
    if len(data) != size:
        msg = "Truncated MCAP record"
        raise ValueError(msg)
    return data


def _string(stream: BinaryIO, end: int) -> bytes:
    size = struct.unpack("<I", _read(stream, 4, end))[0]
    if size > _MAX_METADATA:
        msg = "MCAP string exceeds metadata limit"
        raise ValueError(msg)
    return _read(stream, size, end)


def _scan_records(
    stream: BinaryIO,
    end: int,
    limits: ResourceLimits,
    previous: dict[str, int],
    channels: dict[int, str],
    *,
    nested: bool = False,
) -> None:
    while stream.tell() < end:
        opcode, length = struct.unpack("<BQ", _read(stream, 9, end))
        stop = stream.tell() + length
        if stop > end:
            msg = "Malformed MCAP record length"
            raise ValueError(msg)
        if opcode not in {_CHUNK, _MESSAGE} and length > _MAX_METADATA:
            msg = "MCAP metadata record exceeds limit"
            raise ValueError(msg)
        if opcode == _CHUNK:
            if nested or length > limits.chunk_bytes:
                msg = "Nested MCAP chunk or chunk_bytes exceeded"
                raise ValueError(msg)
            _read(stream, 16, stop)
            size = struct.unpack("<Q", _read(stream, 8, stop))[0]
            _read(stream, 4, stop)
            if _string(stream, stop):
                msg = "Compressed MCAP chunks are unsupported; provide an uncompressed recording"
                raise ValueError(msg)
            data_size = struct.unpack("<Q", _read(stream, 8, stop))[0]
            if size != data_size or size > limits.chunk_bytes or stream.tell() + data_size != stop:
                msg = "Malformed MCAP chunk size"
                raise ValueError(msg)
            _scan_records(stream, stop, limits, previous, channels, nested=True)
        elif opcode == _CHANNEL:
            channel, _ = struct.unpack("<HH", _read(stream, 4, stop))
            channels[channel] = _string(stream, stop).decode("utf-8")
        elif opcode == _MESSAGE:
            if length < _MESSAGE_HEADER_BYTES or length - _MESSAGE_HEADER_BYTES > limits.message_bytes:
                msg = "MCAP message_bytes exceeded or malformed message"
                raise ValueError(msg)
            channel, _, timestamp, _ = struct.unpack("<HIQQ", _read(stream, _MESSAGE_HEADER_BYTES, stop))
            topic = channels.get(channel)
            if topic is None:
                msg = f"MCAP message references unknown channel {channel}"
                raise ValueError(msg)
            if timestamp < previous.get(topic, timestamp):
                msg = f"Timestamp regression in MCAP topic {topic}"
                raise ValueError(msg)
            previous[topic] = timestamp
        stream.seek(stop)


def _mcap(path: Path, limits: ResourceLimits, previous: dict[str, int] | None = None) -> None:
    with path.open("rb") as stream:
        size = path.stat().st_size
        if size < 2 * len(_MAGIC) or _read(stream, 8, size) != _MAGIC:
            msg = "Invalid MCAP magic"
            raise ValueError(msg)
        stream.seek(size - 8)
        if stream.read(8) != _MAGIC:
            msg = "Missing MCAP trailing magic"
            raise ValueError(msg)
        stream.seek(8)
        _scan_records(stream, size - 8, limits, previous if previous is not None else {}, {})


def _sqlite(path: Path, limits: ResourceLimits, previous: dict[str, int]) -> None:
    with closing(sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)) as connection:
        connection.execute("PRAGMA query_only=ON")
        max_size = connection.execute("SELECT max(length(data)) FROM messages").fetchone()[0]
        if max_size is not None and max_size > limits.message_bytes:
            msg = "SQLite message_bytes exceeded"
            raise ValueError(msg)
        for topic, timestamp in connection.execute(
            "SELECT topics.name,messages.timestamp FROM messages JOIN topics ON topics.id=messages.topic_id "
            "ORDER BY messages.id"
        ):
            if timestamp < previous.get(topic, timestamp):
                msg = f"Timestamp regression in SQLite topic id {topic}"
                raise ValueError(msg)
            previous[topic] = timestamp


def preflight(path: Path, limits: ResourceLimits) -> tuple[tuple[Path, int, int], ...]:
    """Validate storage before opening the message reader.

    Returns:
        File paths, lengths, and modification times for change detection.

    Raises:
        ValueError: Storage is unsupported, malformed, oversized, or escapes its root.
        TypeError: Metadata does not contain the required mapping structure.
    """
    metadata = path / "metadata.yaml"
    if not metadata.resolve().is_relative_to(path.resolve()):
        msg = "Bag metadata escapes its recording directory"
        raise ValueError(msg)
    with metadata.open("rb") as stream:
        content = stream.read(_MAX_METADATA + 1)
    if len(content) > _MAX_METADATA:
        msg = "Bag metadata exceeds limit"
        raise ValueError(msg)
    document = yaml.safe_load(content)
    if not isinstance(document, dict) or not isinstance(document.get("rosbag2_bagfile_information"), dict):
        msg = "Bag metadata must contain rosbag2_bagfile_information"
        raise TypeError(msg)
    info = document["rosbag2_bagfile_information"]
    mode = info.get("compression_mode", "")
    if not isinstance(mode, str) or mode.lower() not in {"", "none"} or info.get("compression_format", ""):
        msg = "Bag file/message compression is unsupported"
        raise ValueError(msg)
    storage = info.get("storage_identifier")
    if storage not in {"sqlite3", "mcap"}:
        msg = f"Unsupported storage: {storage}"
        raise ValueError(msg)
    names = info.get("relative_file_paths")
    if not isinstance(names, list) or not names or not all(isinstance(name, str) for name in names):
        msg = "Bag metadata must contain relative_file_paths"
        raise ValueError(msg)
    files = []
    previous: dict[str, int] = {}
    for name in names:
        file = (path / name).resolve()
        if Path(name).is_absolute() or not file.is_relative_to(path.resolve()) or not file.is_file():
            msg = f"Storage path escapes recording or is missing: {name}"
            raise ValueError(msg)
        # rosbags 0.11.9 strips directory components when opening metadata paths.
        # Never preflight a different file from the one the reader will open.
        if Path(name).parent != Path():
            msg = f"Nested storage paths are unsupported; place storage files beside metadata.yaml: {name}"
            raise ValueError(msg)
        if file in files:
            msg = "Duplicate storage path"
            raise ValueError(msg)
        files.append(file)
        if storage == "sqlite3":
            _sqlite(file, limits, previous)
        else:
            _mcap(file, limits, previous)
    return tuple((file, file.stat().st_size, file.stat().st_mtime_ns) for file in [metadata, *files])
