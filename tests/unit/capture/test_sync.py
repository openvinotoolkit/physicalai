# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for multi-camera synchronisation."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from physicalai.capture.errors import CaptureTimeoutError
from physicalai.capture.frame import Frame
from physicalai.capture.multi import SyncedFrames, async_read_cameras, read_cameras
from tests.unit.capture.fake import FakeCamera


class SlowCamera(FakeCamera):
    """Camera that sleeps before returning a frame to test timeout handling."""

    def __init__(self, delay_s: float = 1.0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._delay_s = delay_s

    def read(self, timeout: float = 2.0) -> Frame:
        time.sleep(self._delay_s)
        return super().read(timeout)

    def read_latest(self) -> Frame:
        time.sleep(self._delay_s)
        return super().read_latest()

    async def async_read(self, timeout: float = 2.0) -> Frame:
        await asyncio.sleep(self._delay_s)
        return await super().async_read(timeout)


class TestReadCameras:
    """read_cameras() with multiple FakeCamera instances."""

    def test_returns_synced_frames(self) -> None:
        cameras = {
            "left": FakeCamera(device_name="left"),
            "right": FakeCamera(device_name="right"),
        }
        for cam in cameras.values():
            cam.connect()
        try:
            synced = read_cameras(cameras)
            assert isinstance(synced, SyncedFrames)
            assert "left" in synced.frames
            assert "right" in synced.frames
        finally:
            for cam in cameras.values():
                cam.disconnect()

    def test_skew_is_small(self) -> None:
        cameras = {
            "a": FakeCamera(device_name="a"),
            "b": FakeCamera(device_name="b"),
        }
        for cam in cameras.values():
            cam.connect()
        try:
            synced = read_cameras(cameras)
            # FakeCamera reads are near-instant so skew should be tiny
            assert synced.max_skew_ms < 100.0
        finally:
            for cam in cameras.values():
                cam.disconnect()

    def test_single_camera(self) -> None:
        cam = FakeCamera(device_name="solo")
        cam.connect()
        try:
            synced = read_cameras({"solo": cam})
            assert synced.max_skew_ms == 0.0
            assert synced.frames["solo"].sequence == 0
        finally:
            cam.disconnect()

    def test_empty_cameras_mapping(self) -> None:
        synced = read_cameras({})
        assert isinstance(synced, SyncedFrames)
        assert synced.frames == {}
        assert synced.max_skew_ms == 0.0

    def test_invalid_timeout_raises(self) -> None:
        cam = FakeCamera()
        cam.connect()
        try:
            with pytest.raises(ValueError, match="timeout must be positive"):
                read_cameras({"cam": cam}, timeout=0.0)
            with pytest.raises(ValueError, match="timeout must be positive"):
                read_cameras({"cam": cam}, timeout=-1.0)
        finally:
            cam.disconnect()

    def test_timeout_raises_capture_timeout_error(self) -> None:
        cameras = {
            "fast": FakeCamera(device_name="fast"),
            "slow": SlowCamera(delay_s=1.0, device_name="slow"),
        }
        for cam in cameras.values():
            cam.connect()
        try:
            with pytest.raises(CaptureTimeoutError, match=r"Cameras did not respond within 0\.1s") as exc_info:
                read_cameras(cameras, timeout=0.1, latest=False)
            assert "slow" in str(exc_info.value)
        finally:
            for cam in cameras.values():
                cam.disconnect()

    def test_timeout_does_not_block_beyond_timeout(self) -> None:
        slow_cam = SlowCamera(delay_s=2.0, device_name="slow")
        slow_cam.connect()
        t0 = time.perf_counter()
        try:
            with pytest.raises(CaptureTimeoutError):
                read_cameras({"slow": slow_cam}, timeout=0.1, latest=True)
            elapsed = time.perf_counter() - t0
            assert elapsed < 1.0
        finally:
            slow_cam.disconnect()

    def test_read_sequential_mode(self) -> None:
        cameras = {
            "c1": FakeCamera(device_name="c1"),
            "c2": FakeCamera(device_name="c2"),
        }
        for cam in cameras.values():
            cam.connect()
        try:
            synced = read_cameras(cameras, latest=False)
            assert len(synced.frames) == 2
            assert synced.frames["c1"].sequence == 0
            assert synced.frames["c2"].sequence == 0
        finally:
            for cam in cameras.values():
                cam.disconnect()


class TestAsyncReadCameras:
    """async_read_cameras() with FakeCamera instances."""

    def test_async_returns_synced_frames(self) -> None:
        async def _run() -> SyncedFrames:
            cameras = {
                "left": FakeCamera(device_name="left"),
                "right": FakeCamera(device_name="right"),
            }
            for cam in cameras.values():
                cam.connect()
            try:
                return await async_read_cameras(cameras)
            finally:
                for cam in cameras.values():
                    cam.disconnect()

        synced = asyncio.run(_run())
        assert "left" in synced.frames
        assert "right" in synced.frames

    def test_async_empty_cameras_mapping(self) -> None:
        synced = asyncio.run(async_read_cameras({}))
        assert isinstance(synced, SyncedFrames)
        assert synced.frames == {}
        assert synced.max_skew_ms == 0.0

    def test_async_invalid_timeout_raises(self) -> None:
        cam = FakeCamera()
        cam.connect()
        try:
            with pytest.raises(ValueError, match="timeout must be positive"):
                asyncio.run(async_read_cameras({"cam": cam}, timeout=0.0))
            with pytest.raises(ValueError, match="timeout must be positive"):
                asyncio.run(async_read_cameras({"cam": cam}, timeout=-0.5))
        finally:
            cam.disconnect()

    def test_async_timeout_raises_capture_timeout_error(self) -> None:
        async def _run() -> None:
            cameras = {
                "fast": FakeCamera(device_name="fast"),
                "slow": SlowCamera(delay_s=1.0, device_name="slow"),
            }
            for cam in cameras.values():
                cam.connect()
            try:
                with pytest.raises(CaptureTimeoutError, match=r"Cameras did not respond within 0\.1s"):
                    await async_read_cameras(cameras, timeout=0.1, latest=False)
            finally:
                for cam in cameras.values():
                    cam.disconnect()

        asyncio.run(_run())
