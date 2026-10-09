# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the physicalai inference CLI subcommands."""

from __future__ import annotations

import errno

import pytest

from physicalai.cli import inference
from physicalai.transport._zenoh import derive_port


def test_port_prints_model_derived_port(capsys) -> None:  # noqa: ANN001
    parser = inference.build_parser()
    cfg = parser.parse_args(["port", "pi05"])

    assert inference._dispatch(parser, cfg) == 0  # noqa: SLF001
    assert capsys.readouterr().out.strip() == str(derive_port("inference", "pi05"))


@pytest.mark.parametrize(
    ("name", "expected_port"),
    [("so101-act", 31339), ("cell1-act", 20212), ("cell2-pi05", 37278)],
)
def test_port_matches_reference_values(name: str, expected_port: int, capsys) -> None:  # noqa: ANN001
    parser = inference.build_parser()
    cfg = parser.parse_args(["port", name])

    assert inference._dispatch(parser, cfg) == 0  # noqa: SLF001
    assert capsys.readouterr().out.strip() == str(expected_port)


def test_serve_parser_accepts_required_name_and_export_source() -> None:
    cfg = inference.build_parser().parse_args([
        "serve",
        "--name=pi05",
        "--export-dir=/models/pi05",
        "--policy-name=pi05",
        "--backend=openvino",
        "--device=GPU",
        "--listen=tcp/127.0.0.1:12345",
    ])

    assert cfg.subcommand == "serve"
    assert cfg.serve.name == "pi05"
    assert cfg.serve.export_dir.as_posix() == "/models/pi05"
    assert cfg.serve.policy_name == "pi05"
    assert cfg.serve.device == "GPU"
    assert cfg.serve.listen == "tcp/127.0.0.1:12345"


def test_serve_forwards_hub_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    import physicalai.inference as inference_package
    import physicalai.inference.remote as remote_package

    captured: dict[str, object] = {}

    class _ModelLoader:
        @classmethod
        def from_pretrained(cls, hub_id: str, **kwargs: object) -> object:
            captured["hub_id"] = hub_id
            captured.update(kwargs)
            return object()

    class _Server:
        endpoint = "tcp/127.0.0.1:12345"

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def start(self) -> None:
            pass

        def serve_forever(self) -> None:
            pass

        def stop(self) -> None:
            pass

    monkeypatch.setattr(inference_package, "InferenceModel", _ModelLoader)
    monkeypatch.setattr(remote_package, "InferenceServer", _Server)
    parser = inference.build_parser()
    cfg = parser.parse_args([
        "serve",
        "--name=pi05",
        "--hub-id=ORG/REPO",
        "--revision=0123456789abcdef0123456789abcdef01234567",
    ])

    assert inference._dispatch(parser, cfg) == 0  # noqa: SLF001
    assert captured["hub_id"] == "ORG/REPO"
    assert captured["revision"] == "0123456789abcdef0123456789abcdef01234567"


def test_serve_requires_name_and_exactly_one_model_source() -> None:
    parser = inference.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["serve", "--export-dir=/models/pi05"])
    with pytest.raises(SystemExit):
        parser.parse_args(["serve", "--name=pi05"])
    with pytest.raises(SystemExit):
        parser.parse_args(["serve", "--name=pi05", "--export-dir=/models/pi05", "--hub-id=repo/pi05"])


def test_serve_port_conflict_returns_nonzero_without_fallback(monkeypatch: pytest.MonkeyPatch, capsys) -> None:  # noqa: ANN001
    import physicalai.inference as inference_package
    import physicalai.inference.remote as remote_package

    class _Model:
        def __init__(self, **_kwargs: object) -> None:
            self.policy_name = "pi05"

    class _Server:
        def __init__(self, _model: object, _name: str, *, listen: str, **_kwargs: object) -> None:
            self.endpoint = listen
            self.start_count = 0

        def start(self) -> None:
            self.start_count += 1
            raise OSError(errno.EADDRINUSE, "Address already in use")

        def stop(self) -> None:
            pass

    monkeypatch.setattr(inference_package, "InferenceModel", _Model)
    monkeypatch.setattr(remote_package, "InferenceServer", _Server)
    parser = inference.build_parser()
    cfg = parser.parse_args(["serve", "--name", "pi05", "--export-dir", "/models/pi05", "--port", "23456"])

    assert inference._dispatch(parser, cfg) == 1  # noqa: SLF001
    assert capsys.readouterr().err.strip() == "port 23456 in use — choose another --name or --port"
