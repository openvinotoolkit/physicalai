# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Lossless offline backpressure for the pinned LeRobot 0.6.0 streaming writer."""

from __future__ import annotations

import queue
import time
from typing import TYPE_CHECKING

from lerobot.datasets.compute_stats import RunningQuantileStats, auto_downsample_height_width
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_utils import StreamingVideoEncoder

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    from lerobot.configs import DepthEncoderConfig, RGBEncoderConfig
    from numpy.typing import NDArray

_QUEUE_POLL_SECONDS = 0.05


class _OfflineVideoEncoder(StreamingVideoEncoder):
    """Wait on bounded queues instead of dropping live-recording frames.

    LeRobot 0.6.0 has no public lossless-queue option. Keep this version-specific
    adapter isolated, and test slow/dead encoders when updating that dependency.
    """

    def start_episode(
        self,
        video_keys: list[str],
        temp_dir: Path,
        depth_video_keys: list[str] | None = None,
    ) -> None:
        """Start bounded encoding and retain at most one frame per camera for stats."""
        super().start_episode(video_keys, temp_dir, depth_video_keys)
        self._frame_counts: dict[str, int] = {}
        self._first_frames: dict[str, NDArray[np.uint8]] = {}

    def _check_thread(self, video_key: str) -> None:
        if not self._episode_active or not self._threads[video_key].is_alive():
            msg = f"Offline video encoder is not running: {video_key}"
            raise RuntimeError(msg)

    def feed_frame(self, video_key: str, image: NDArray[np.uint8]) -> None:
        """Enqueue every frame, preserving bounded memory and interruptibility."""
        frame = image.copy()
        while True:
            self._check_thread(video_key)
            try:
                self._frame_queues[video_key].put(frame, timeout=_QUEUE_POLL_SECONDS)
            except queue.Full:
                continue
            else:
                count = self._frame_counts.get(video_key, 0) + 1
                self._frame_counts[video_key] = count
                if count == 1:
                    self._first_frames[video_key] = frame
                else:
                    self._first_frames.pop(video_key, None)
                return

    def finish_episode(self) -> dict[str, tuple[Path, dict | None]]:
        """Drain queues before the parent sends its blocking sentinel.

        Returns:
            Native video paths and channel statistics for every camera.

        Raises:
            RuntimeError: An encoder failed to return complete statistics.
        """
        # A dead consumer with a full queue must fail, not hang on the sentinel.
        # This caller is the only producer, so queues stay empty until super().
        for key, frames in self._frame_queues.items():
            while not frames.empty():
                self._check_thread(key)
                time.sleep(_QUEUE_POLL_SECONDS)
        results = super().finish_episode()
        for key, (path, stats) in results.items():
            if stats is None:
                # LeRobot omits camera stats for one-frame episodes, even though
                # each image contains enough pixel samples for channel statistics.
                if self._frame_counts[key] != 1:
                    msg = f"Offline video encoder did not finish: {key}"
                    raise RuntimeError(msg)
                image = auto_downsample_height_width(self._first_frames[key].transpose(2, 0, 1))
                tracker = RunningQuantileStats()
                tracker.update(image.transpose(1, 2, 0).reshape(-1, 3))
                results[key] = (path, tracker.get_statistics())
        self._first_frames.clear()
        return results


class OfflineLeRobotDataset(LeRobotDataset):
    """Native dataset writer with a lossless bounded streaming encoder."""

    @staticmethod
    def _build_streaming_encoder(
        fps: int,
        rgb_encoder: RGBEncoderConfig | None,
        depth_encoder: DepthEncoderConfig | None,
        encoder_queue_maxsize: int,
        encoder_threads: int | None,
    ) -> StreamingVideoEncoder:
        return _OfflineVideoEncoder(
            fps=fps,
            rgb_encoder=rgb_encoder,
            depth_encoder=depth_encoder,
            queue_maxsize=encoder_queue_maxsize,
            encoder_threads=encoder_threads,
        )
