# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Serve and inspect remote inference endpoints."""

from __future__ import annotations

import errno
import ipaddress
import logging
import signal
import socket
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from jsonargparse import ArgumentParser

from physicalai.cli._spec import SubcommandSpec  # noqa: PLC2701
from physicalai.transport._zenoh import derive_port, endpoint_for_key, model_key_prefix

if TYPE_CHECKING:
    from jsonargparse import Namespace

HELP = "Serve a policy over Zenoh or print its deterministic endpoint port."
_SERVE_HELP = "Load and serve an exported policy over Zenoh."
_PORT_HELP = "Print the deterministic Zenoh TCP port for a model name."
_HELP_TEMPLATE = """usage: {prog} {{serve,port}} ...

{description}

subcommands:
  serve   {serve_help}
  port    {port_help}

Run '{prog} serve --help' or '{prog} port --help' for subcommand options.
"""


def _build_serve_parser() -> ArgumentParser:
    parser = ArgumentParser(description=_SERVE_HELP)
    parser.add_argument("--name", required=True, help="Remote inference namespace name")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--export-dir", type=Path, help="Exported inference model directory")
    source.add_argument("--hub-id", help="Hugging Face repository or local model path")
    parser.add_argument(
        "--revision",
        default=None,
        help="Hub revision; use a reviewed commit SHA for reproducible model loading",
    )
    parser.add_argument("--policy-name", default=None, help="Policy name; auto-detected from the export if omitted")
    parser.add_argument("--backend", choices=("auto", "openvino", "onnx"), default="auto", help="Inference backend")
    parser.add_argument("--device", default="auto", help="Inference device")
    listen = parser.add_mutually_exclusive_group()
    listen.add_argument("--listen", help="Full TCP listen endpoint")
    listen.add_argument("--port", type=int, help="TCP port; otherwise derived from --name")
    parser.add_argument("--allow-all-interfaces", action="store_true", help="Allow non-loopback listen endpoints")
    parser.add_argument("--zenoh-config", type=Path, help="Replace the built-in Zenoh configuration")
    return parser


def _build_port_parser() -> ArgumentParser:
    parser = ArgumentParser(description=_PORT_HELP)
    parser.add_argument("name", help="Model namespace segment, e.g. pi05")
    return parser


def build_parser() -> ArgumentParser:
    """Build the nested ``physicalai inference`` parser."""
    parser = ArgumentParser(prog="physicalai inference", description=HELP)
    subcommands = parser.add_subcommands(required=True)
    subcommands.add_subcommand("serve", _build_serve_parser(), help=_SERVE_HELP)
    subcommands.add_subcommand("port", _build_port_parser(), help=_PORT_HELP)
    return parser


def print_help(prog: str) -> None:
    """Print lightweight group help."""
    print(
        _HELP_TEMPLATE.format(
            prog=prog,
            description=HELP,
            serve_help=_SERVE_HELP,
            port_help=_PORT_HELP,
        )
    )  # noqa: T201


def _serve(cfg: Namespace) -> int:
    from physicalai.inference import InferenceModel  # noqa: PLC0415
    from physicalai.inference.remote import InferenceServer  # noqa: PLC0415

    model_key_prefix(cfg.name)
    if cfg.export_dir is not None:
        model = InferenceModel(
            export_dir=cfg.export_dir,
            policy_name=cfg.policy_name,
            backend=cfg.backend,
            device=cfg.device,
        )
    else:
        model = InferenceModel.from_pretrained(
            cfg.hub_id,
            revision=cfg.revision,
            policy_name=cfg.policy_name,
            backend=cfg.backend,
            device=cfg.device,
        )
    listen = cfg.listen
    if listen is None:
        host = "0.0.0.0" if cfg.allow_all_interfaces else "127.0.0.1"
        listen = endpoint_for_key(model_key_prefix(cfg.name), host, cfg.port)
    server = InferenceServer(
        model,
        cfg.name,
        listen=listen,
        allow_all_interfaces=cfg.allow_all_interfaces,
        zenoh_config=cfg.zenoh_config,
    )
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    previous_handlers: dict[signal.Signals, object] = {}

    def stop_server(_signum: int, _frame: object) -> None:
        server.stop()

    try:
        server.start()
        listen_host = server.endpoint.removeprefix("tcp/").rsplit(":", maxsplit=1)[0].strip("[]")
        try:
            loopback = ipaddress.ip_address(listen_host).is_loopback
        except ValueError:
            loopback = listen_host.lower() == "localhost"
        if loopback:
            port = server.endpoint.rsplit(":", maxsplit=1)[-1]
            logger.info("SSH tunnel: ssh -N -L %s:127.0.0.1:%s %s", port, port, socket.gethostname())
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.signal(signum, stop_server)
        server.serve_forever()
    except Exception as error:
        message = str(error).lower()
        port = server.endpoint.rsplit(":", maxsplit=1)[-1]
        if (
            (isinstance(error, OSError) and error.errno == errno.EADDRINUSE)
            or "address already in use" in message
            or "eaddrinuse" in message
            or f"errno {errno.EADDRINUSE}" in message
        ):
            print(f"port {port} in use — choose another --name or --port", file=sys.stderr)  # noqa: T201
        else:
            logger.exception("Inference server failed")
        return 1
    finally:
        server.stop()
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
    return 0


def _port(cfg: Namespace) -> int:
    model_key_prefix(cfg.name)
    print(derive_port("inference", cfg.name))  # noqa: T201
    return 0


def _dispatch(_parser: ArgumentParser, cfg: Namespace) -> int:
    if cfg.subcommand == "serve":
        return _serve(cfg.serve)
    if cfg.subcommand == "port":
        return _port(cfg.port)
    msg = f"unknown inference subcommand: {cfg.subcommand!r}"
    raise AssertionError(msg)


def register() -> SubcommandSpec:
    """Register ``physicalai inference`` with the CLI host."""
    return SubcommandSpec(name="inference", parser=build_parser(), dispatch=_dispatch, help=HELP)
