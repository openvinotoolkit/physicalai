# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import socket

from physicalai.diagnostics.platforms.events import LinuxChanges


def test_native_wakeup_and_notification_loss_fallback() -> None:
    receiver, sender = socket.socketpair()
    try:
        source = LinuxChanges(receiver)
        sender.send(b"SUBSYSTEM=usb\x00ACTION=add\x00")
        source.wait(0.01)
        assert not source.failed
        receiver.close()
        source.wait(0.001)
        assert source.failed and "polling" in source.detail
    finally:
        receiver.close()
        sender.close()
