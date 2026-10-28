# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import errno
import os
import sys

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux V4L2 ABI")


def test_saved_video_state_uses_queries_only_until_explicit_restore(monkeypatch: pytest.MonkeyPatch) -> None:
    from physicalai.diagnostics.platforms import video

    calls = []
    closed = []
    opens = []
    def open_fd(path, flags):
        opens.append(flags)
        return 12
    def ioctl(fd, request, value):
        calls.append(request)
        if request == video.VIDIOC_G_FMT:
            value.fmt.pix.width = 640
            value.fmt.pix.height = 480
            value.fmt.pix.pixelformat = int.from_bytes(b'MJPG', 'little')
        elif request == video.VIDIOC_G_PARM:
            value.parm.capture.timeperframe.numerator = 1
            value.parm.capture.timeperframe.denominator = 30
    monkeypatch.setattr(video.os, "open", open_fd)
    monkeypatch.setattr(video.os, "close", closed.append)
    monkeypatch.setattr(video, "xioctl", ioctl)
    state = video.VideoState.read("fake")
    assert calls == [video.VIDIOC_G_FMT, video.VIDIOC_G_PARM]
    assert opens == [os.O_RDONLY | os.O_NONBLOCK]
    assert state.describe() == {"width": 640, "height": 480, "fps": 30, "pixel_format": "MJPG"}
    state.restore("fake")
    assert calls[-2:] == [video.VIDIOC_S_FMT, video.VIDIOC_S_PARM]
    assert opens[-1] == os.O_RDWR | os.O_NONBLOCK
    assert closed == [12, 12]


def test_failed_video_query_always_closes_descriptor(monkeypatch: pytest.MonkeyPatch) -> None:
    from physicalai.diagnostics.platforms import video
    from physicalai.diagnostics.models import Endpoint

    closed = []
    monkeypatch.setattr(video.os, "open", lambda *a: 12)
    monkeypatch.setattr(video.os, "close", closed.append)
    def denied(*args):
        raise OSError(errno.ENODEV, "device removed")
    monkeypatch.setattr(video, "xioctl", denied)
    with pytest.raises(OSError):
        video.probe_video(Endpoint(id="video", path="fake", kind="video"))
    assert closed == [12]


def test_per_node_capabilities_distinguish_metadata_from_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    from physicalai.diagnostics.platforms import video
    from physicalai.diagnostics.models import Endpoint
    monkeypatch.setattr(video.os, "open", lambda *a: 12)
    monkeypatch.setattr(video.os, "close", lambda _: None)
    def query(fd, request, cap):
        assert request == video.VIDIOC_QUERYCAP
        cap.capabilities = video.V4L2_CAP_DEVICE_CAPS | video.V4L2_CAP_VIDEO_CAPTURE | 0x00800000
        cap.device_caps = 0x00800000
    monkeypatch.setattr(video, "xioctl", query)
    endpoint = Endpoint(id="metadata", path="fake", kind="video")
    video.probe_video(endpoint)
    assert endpoint.role == "metadata" and not endpoint.modes
