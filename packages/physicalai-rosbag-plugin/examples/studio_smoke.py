# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Exercise real Studio HTTP import, previews, directory import, and CPU ACT export."""

import argparse
from collections.abc import Callable
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
from uuid import uuid4
from typing import Any

import httpx


def wait_job(
    client: httpx.Client,
    job_id: str,
    predicate: Callable[[dict[str, Any]], bool],
    timeout: float = 180,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(f"/api/jobs/{job_id}")
        response.raise_for_status()
        job = response.json()
        if job["status"] in {"failed", "cancelled"}:
            raise RuntimeError(f"Import job failed: {job}")
        if predicate(job):
            return job
        time.sleep(0.5)
    raise TimeoutError(f"Import job timed out: {job_id}")


def train_export(dataset_path: Path, work: Path) -> dict[str, Any]:
    import numpy as np
    import torch
    from physicalai.data.lerobot import LeRobotDataModule, FormatConverter
    from physicalai.inference import InferenceModel
    from physicalai.policies import ACT
    from physicalai.train import Trainer

    torch.set_num_threads(2)
    torch.manual_seed(42)
    data = LeRobotDataModule(
        root=dataset_path, train_batch_size=2, num_workers=0, persistent_workers=False, pin_memory=False, val_split=0.0
    )
    policy = ACT(
        dataset_stats=data.train_dataset.stats,
        chunk_size=2,
        n_action_steps=2,
        pretrained_backbone_weights=None,
        image_size=(64, 64),
        dim_model=32,
        n_heads=4,
        dim_feedforward=64,
        n_encoder_layers=1,
        n_decoder_layers=1,
        use_vae=False,
        compile_model=False,
    )
    before = {name: value.detach().clone() for name, value in policy.named_parameters()}
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        max_steps=2,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        default_root_dir=work,
        num_sanity_val_steps=0,
        limit_val_batches=0,
    )
    trainer.fit(model=policy, datamodule=data)
    assert trainer.global_step == 2
    assert any(not torch.equal(before[name], value.detach()) for name, value in policy.named_parameters())
    losses = {key: float(value) for key, value in trainer.callback_metrics.items() if "loss" in key}
    assert losses and all(np.isfinite(value) for value in losses.values()), losses
    checkpoint = work / "trained.ckpt"
    trainer.save_checkpoint(checkpoint)
    trained = ACT.load_from_checkpoint(checkpoint, map_location="cpu").eval()
    observation = FormatConverter.to_observation(next(iter(data.train_dataloader())))[0:1].to("cpu")
    with torch.no_grad():
        expected = trained.predict_action_chunk(observation)
    if isinstance(expected, tuple):
        expected = expected[0]
    exported = work / "exported"
    trained.export(exported, backend="openvino")
    runtime = InferenceModel(export_dir=exported)
    result = runtime.select_action(observation.to_numpy().to_dict(flatten=False))
    actual = np.asarray(result).reshape(-1)
    expected = expected.detach().cpu().numpy()[0, 0].reshape(-1)
    assert actual.shape == expected.shape and np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, rtol=0.2, atol=0.2)
    return {"steps": trainer.global_step, "losses": losses, "export": str(exported), "action": actual.tolist()}


def run(dataset: Path, work: Path) -> None:
    work.mkdir(parents=True, exist_ok=False)
    storage = work / "storage"
    environment = {
        **os.environ,
        "STORAGE_DIR": str(storage),
        "SETTINGS_FILE": str(work / "settings.json"),
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
        "TOKENIZERS_PARALLELISM": "false",
    }
    executable = Path(sys.executable).parent / "physicalai-studio"
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    archive = Path(shutil.make_archive(str(work / "dataset"), "zip", dataset))
    report = {"studio_revision": "894ea9fb9c502cfa996b6cce4195a9993301f3c9", "dataset": str(dataset)}
    with (work / "backend.log").open("w") as log:
        backend = subprocess.Popen(
            [str(executable), "serve", "--host", "127.0.0.1", "--port", str(port)],
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=60) as client:
                deadline = time.monotonic() + 180
                while True:
                    if backend.poll() is not None:
                        raise RuntimeError(f"Backend exited; see {work / 'backend.log'}")
                    try:
                        if client.get("/api/health").status_code == 200:
                            break
                    except httpx.TransportError:
                        pass
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Studio backend did not start")
                    time.sleep(0.5)
                project_id, environment_id = str(uuid4()), str(uuid4())
                response = client.post("/api/projects", json={"id": project_id, "name": "Rosbag smoke"})
                response.raise_for_status()
                response = client.post(
                    f"/api/projects/{project_id}/environments",
                    json={
                        "id": environment_id,
                        "name": "Offline import",
                        "robots": [],
                        "cameras": [],
                    },
                )
                response.raise_for_status()
                prefix = f"/api/projects/{project_id}/imports/datasets"
                response = client.post(
                    f"{prefix}:prepare", data={"format_hint": "auto", "dataset_name": "ROS bag demo"}
                )
                response.raise_for_status()
                job_id = response.json()["id"]
                with archive.open("rb") as stream:
                    response = client.put(
                        f"{prefix}/{job_id}:upload", files={"archive": ("dataset.zip", stream, "application/zip")}
                    )
                response.raise_for_status()
                review = wait_job(client, job_id, lambda job: job["payload"]["step"] == "awaiting_user_review")
                payload = review["payload"]
                manifest = payload["dataset_manifest_draft"]
                info = json.loads((dataset / "meta/info.json").read_text())
                assert manifest["statistics"]["episode_count"] == info["total_episodes"]
                assert manifest["statistics"]["frame_count"] == info["total_frames"]
                schema = manifest["dataset_schema"]
                assert {camera["name"] for camera in schema["cameras"]} == {"front", "wrist"}
                assert set(schema["robots"][0]["joints"]) == {"shoulder_pan", "gripper"}
                assert "Move the synthetic arm" in schema["tasks"]
                validation = payload.get("validation_report") or {}
                assert not any(message["severity"] == "error" for message in validation.get("messages", []))
                response = client.post(
                    f"{prefix}/{job_id}:finalize",
                    json={"environment_id": environment_id, "default_task": "Move the synthetic arm"},
                )
                response.raise_for_status()
                completed = wait_job(client, job_id, lambda job: job["status"] == "completed")
                dataset_id = completed["payload"]["result_dataset_id"]
                for route in (
                    f"/api/dataset/{dataset_id}",
                    f"/api/dataset/{dataset_id}/episodes",
                    f"/api/dataset/{dataset_id}/episodes/0",
                ):
                    response = client.get(route)
                    response.raise_for_status()
                    assert response.content
                episode = response.json()
                assert episode["actions"] and episode["action_keys"]
                for video in episode["videos"].values():
                    response = client.get(f"/api/dataset/{dataset_id}/video/{video['path']}")
                    response.raise_for_status()
                    assert response.headers["content-type"].startswith("video/")
                    assert b"ftyp" in response.content[:32]
                for camera in ("front", "wrist"):
                    response = client.get(
                        f"/api/dataset/{dataset_id}/episodes/0/thumbnail",
                        params={"camera": camera, "width": 64, "height": 64},
                    )
                    response.raise_for_status()
                    assert response.content.startswith(b"\x89PNG")
                report["zip_import"] = {"job_id": job_id, "dataset_id": dataset_id, "manifest": manifest}
                copied = subprocess.run(
                    [
                        str(executable),
                        "datasets",
                        "import-dir",
                        "--source-dir",
                        str(dataset),
                        "--project-id",
                        project_id,
                        "--environment-id",
                        environment_id,
                        "--dataset-name",
                        "Directory copy",
                        "--copy",
                    ],
                    env=environment,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                (work / "directory-import.log").write_text(copied.stdout + copied.stderr)
                assert "Dataset ID:" in copied.stdout and dataset.is_dir()
                report["directory_import"] = "passed (copy)"
                report["training"] = train_export(storage / "datasets" / dataset_id, work)
        finally:
            backend.terminate()
            try:
                backend.wait(timeout=30)
            except subprocess.TimeoutExpired:
                backend.kill()
                backend.wait()
    (work / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    args = parser.parse_args()
    run(args.dataset.resolve(), args.work_dir.resolve())
