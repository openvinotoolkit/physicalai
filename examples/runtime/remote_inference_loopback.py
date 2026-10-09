#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Minimal loopback handshake example for the remote inference classes.

Start the server in one terminal, then run the client in another. The
built-in Zenoh configs use a single explicit loopback endpoint and disable
multicast and gossip.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from physicalai.inference import InferenceModel, RemoteInferenceModel
from physicalai.inference.remote import InferenceServer

_ENDPOINT = "tcp/127.0.0.1:7447"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("server", "client"))
    parser.add_argument("--name", required=True, help="Deployment slot name shared by this server and client")
    parser.add_argument("--export-dir", type=Path, help="Server-side exported model directory (server mode only)")
    parser.add_argument("--policy-name", help="Policy name inside the export (server mode only)")
    parser.add_argument("--device", default="auto", help="Inference device (server mode only)")
    args = parser.parse_args()

    if args.mode == "server":
        if args.export_dir is None:
            parser.error("server mode requires --export-dir")
        model = InferenceModel(
            export_dir=args.export_dir,
            policy_name=args.policy_name,
            device=args.device,
        )
        server = InferenceServer(model, name=args.name, listen=_ENDPOINT)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.stop()
        return

    with RemoteInferenceModel(name=args.name, endpoint=_ENDPOINT) as model:
        print(model.metadata)
        # Call model.predict_action_chunk(observation) with model-ready inputs here.


if __name__ == "__main__":
    main()
