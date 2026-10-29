# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Two-pass, causal sampling of one logical recording at a time."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

from physicalai_rosbag_plugin.images import decode_image

if TYPE_CHECKING:
    from collections.abc import Generator, Iterator
    from pathlib import Path

    from numpy.typing import NDArray

    from physicalai_rosbag_plugin.config import Component, MappingConfig


@dataclass(frozen=True)
class EpisodePlan:
    """Validated source interval and image dimensions for an episode."""

    path: Path
    start_ns: int
    end_ns: int
    frames: int
    shapes: dict[str, tuple[int, ...]]


def frame_times(start_ns: int, end_ns: int, fps: int) -> Iterator[int]:
    """Yield the inclusive frame grid without floating-point epoch calculations."""
    index = 0
    while (timestamp := start_ns + index * 1_000_000_000 // fps) <= end_ns:
        yield timestamp
        index += 1


def _messages(path: Path, config: MappingConfig) -> Generator[tuple[str, int, Any]]:
    topics = config.topics()
    previous: dict[str, tuple[int, bytes]] = {}
    with AnyReader([path], default_typestore=get_typestore(Stores.ROS2_HUMBLE)) as reader:
        connections = [connection for connection in reader.connections if connection.topic in topics]
        if {connection.topic for connection in connections} != set(topics):
            msg = f"{path.name}: missing required topics {set(topics) - {c.topic for c in connections}}"
            raise ValueError(msg)
        for connection in connections:
            if connection.msgtype != topics[connection.topic][0]:
                msg = f"{path.name}: unexpected type on {connection.topic}: {connection.msgtype}"
                raise ValueError(msg)
        for connection, timestamp, raw in reader.messages(connections=connections):
            topic = connection.topic
            if len(raw) > config.limits.message_bytes:
                msg = f"{topic}: serialized message exceeds limit"
                raise ValueError(msg)
            payload = bytes(raw)
            if topic in previous:
                last_time, last_payload = previous[topic]
                if timestamp < last_time:
                    msg = f"{topic}: timestamp regression at {timestamp}"
                    raise ValueError(msg)
                if timestamp == last_time:
                    if payload != last_payload:
                        msg = f"{topic}: conflicting duplicate timestamp {timestamp}"
                        raise ValueError(msg)
                    continue
            previous[topic] = (timestamp, payload)
            yield topic, timestamp, reader.deserialize(raw, connection.msgtype)


def _component(msg: Any, spec: Component) -> float:  # noqa: ANN401
    value = getattr(msg, spec.field)
    if spec.joint is not None:
        if len(set(msg.name)) != len(msg.name) or len(value) != len(msg.name):
            msg_0 = f"{spec.topic}: duplicate joints or inconsistent {spec.field} length"
            raise ValueError(msg_0)
        try:
            value = value[list(msg.name).index(spec.joint)]
        except ValueError as error:
            msg_0 = f"{spec.topic}: missing joint {spec.joint}"
            raise ValueError(msg_0) from error
    elif spec.index is not None:
        if spec.index >= len(value):
            msg_0 = f"{spec.topic}: index {spec.index} outside {spec.field}"
            raise ValueError(msg_0)
        value = value[spec.index]
    factor = {"identity": 1.0, "rad_to_deg": 180 / math.pi, "deg_to_rad": math.pi / 180}[spec.conversion]
    result = float(value) * factor * spec.scale + spec.offset
    if not math.isfinite(result) or abs(result) > np.finfo(np.float32).max:
        msg_0 = f"{spec.topic}: nonfinite/out-of-range value for {spec.name}"
        raise ValueError(msg_0)
    return result


def inspect_episode(path: Path, config: MappingConfig) -> EpisodePlan:
    """Validate every selected message before opening an output writer.

    Returns:
        The inclusive source interval, frame count, and camera shapes.

    Raises:
        ValueError: Required data is missing, inconsistent, or exceeds limits.
    """
    first: dict[str, int] = {}
    last: dict[str, int] = {}
    shapes: dict[str, tuple[int, ...]] = {}
    vector_shapes: dict[tuple[str, str], int] = {}
    for topic, timestamp, msg in _messages(path, config):
        first.setdefault(topic, timestamp)
        last[topic] = timestamp
        for spec in (*config.state, *config.action):
            if spec.topic == topic:
                _component(msg, spec)
                if spec.index is not None or spec.joint is not None:
                    key = (topic, spec.field)
                    size = len(getattr(msg, spec.field))
                    if vector_shapes.setdefault(key, size) != size:
                        msg_0 = f"{topic}: changing numeric dimensions"
                        raise ValueError(msg_0)
        for camera in config.cameras:
            if camera.topic == topic:
                shape = decode_image(msg, camera.message_type, config.limits).shape
                if shapes.setdefault(camera.name, shape) != shape:
                    msg_0 = f"{topic}: changing image dimensions"
                    raise ValueError(msg_0)
    if set(first) != set(config.topics()):
        msg_0 = f"{path.name}: empty required topics"
        raise ValueError(msg_0)
    start, end = max(first.values()), min(last.values())
    if start > end:
        msg_0 = f"{path.name}: required topics have no common interval"
        raise ValueError(msg_0)
    # Invert floor(n * N / fps) <= end-start exactly, including rounded endpoints.
    frames = ((end - start + 1) * config.fps - 1) // 1_000_000_000 + 1
    if frames > config.limits.episode_frames:
        msg_0 = f"{path.name}: episode_frames limit exceeded ({frames})"
        raise ValueError(msg_0)
    return EpisodePlan(path, start, end, frames, shapes)


def sample_episode(plan: EpisodePlan, config: MappingConfig) -> Iterator[dict[str, Any]]:
    """Select latest samples at each tick; never borrow a future sample.

    Yields:
        LeRobot frames with numeric arrays, RGB images, and task text.

    Raises:
        ValueError: A required topic is stale or missing at a frame timestamp.
    """
    topics = config.topics()
    latest: dict[str, tuple[int, Any]] = {}
    image_cache: dict[str, tuple[int, NDArray[np.uint8]]] = {}
    messages = _messages(plan.path, config)
    try:
        pending = next(messages, None)
        for tick in frame_times(plan.start_ns, plan.end_ns, config.fps):
            while pending is not None and pending[1] <= tick:
                topic, timestamp, msg = pending
                latest[topic] = (timestamp, msg)
                pending = next(messages, None)
            for topic, (_, max_age) in topics.items():
                if topic not in latest or tick - latest[topic][0] > max_age:
                    msg_0 = f"{plan.path.name}: missing/stale {topic} at {tick} ns"
                    raise ValueError(msg_0)
            frame: dict[str, Any] = {"task": config.task}
            for key, components in (("observation.state", config.state), ("action", config.action)):
                frame[key] = np.array(
                    [_component(latest[spec.topic][1], spec) for spec in components], dtype=np.float32
                )
            for camera in config.cameras:
                timestamp, msg = latest[camera.topic]
                if camera.name not in image_cache or image_cache[camera.name][0] != timestamp:
                    image_cache[camera.name] = (timestamp, decode_image(msg, camera.message_type, config.limits))
                frame[f"observation.images.{camera.name}"] = image_cache[camera.name][1]
            yield frame
    finally:
        messages.close()
