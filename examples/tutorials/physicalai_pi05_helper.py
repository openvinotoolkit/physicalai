# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from io import BytesIO
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download, snapshot_download
from IPython.display import Image as IPyImage
from PIL import Image, ImageDraw
from physicalai.inference import InferenceModel


STATE_KEY = "state"
TASK_KEY = "task"
DEFAULT_ACTION_NAMES = (
    "base/ee-x",
    "shoulder/ee-y",
    "elbow/ee-z",
    "wrist-rx",
    "wrist-ry",
    "wrist-rz",
    "gripper",
)

SO101_JOINT_ORDER = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


@dataclass(frozen=True)
class ReplayEpisode:
    dataset_root: Path
    episode_id: int
    episode_df: pd.DataFrame
    video_paths: dict[str, Path]
    video_start_frames: dict[str, int]
    image_keys: list[str]
    fps: float
    action_dim: int
    state_dim: int
    task: str | None


def dataset_image_key_to_model_key(dataset_key: str) -> str:
    return dataset_key.removeprefix("observation.")


def _format_lerobot_path(template: str, **values: int | str) -> str:
    return template.format(**values)


def _join_dataset_prefix(dataset_name: str, relative_path: str) -> str:
    return f"{dataset_name}/{relative_path}" if dataset_name else relative_path


def download_pi05_package(repo_id: str, assets_dir: Path, model_dir: Path | None = None) -> Path:
    if model_dir is None:
        model_dir = Path(
            snapshot_download(
                repo_id=repo_id,
                local_dir=assets_dir / "models" / repo_id.replace("/", "__"),
                allow_patterns=[
                    "manifest.json",
                    "pi05.xml",
                    "pi05.bin",
                    "tokenizer.xml",
                    "tokenizer.bin",
                    "metadata.yaml",
                    "README.md",
                ],
                local_dir_use_symlinks=False,
            )
        ).resolve()

    required = ["manifest.json", "pi05.xml", "pi05.bin", "tokenizer.xml", "tokenizer.bin"]
    missing = [name for name in required if not (model_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required PhysicalAI package files in {model_dir}: {missing}")
    return model_dir


def resolve_dataset_root(path: Path, dataset_name: str) -> Path:
    path = Path(path).resolve()
    nested = path / dataset_name
    return nested if nested.exists() else path


def prepare_replay_episode(
    *,
    repo_id: str,
    dataset_name: str,
    assets_dir: Path,
    episode_id: int,
    dataset_dir: Path | None = None,
) -> ReplayEpisode:
    if dataset_dir is None:
        dataset_cache = assets_dir / "datasets" / repo_id.replace("/", "__")
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=dataset_cache,
            allow_patterns=[_join_dataset_prefix(dataset_name, "meta/**")],
            local_dir_use_symlinks=False,
        )
        dataset_root = resolve_dataset_root(dataset_cache, dataset_name)
    else:
        dataset_root = resolve_dataset_root(dataset_dir, dataset_name)

    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing LeRobot dataset metadata: {info_path}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    fps = float(info.get("fps", 30))
    features = info.get("features", {})
    image_keys = [
        key
        for key, spec in features.items()
        if key.startswith("observation.images.") and spec.get("dtype") in {"image", "video"}
    ]
    if not image_keys:
        raise ValueError(f"No observation image/video keys found in {info_path}")
    action_dim = int(features.get("action", {}).get("shape", [7])[0])
    state_dim = int(features.get("observation.state", {}).get("shape", [action_dim])[0])
    data_path_template = info.get("data_path", "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet")
    video_path_template = info.get(
        "video_path",
        "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
    )

    def download_dataset_file(relative_path: str) -> Path:
        local_path = dataset_root / relative_path
        if local_path.exists():
            return local_path
        downloaded = Path(
            hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                filename=_join_dataset_prefix(dataset_name, relative_path),
                local_dir=dataset_root.parent if dataset_name else dataset_root,
                local_dir_use_symlinks=False,
            )
        )
        return downloaded.resolve()

    episode_files = sorted((dataset_root / "meta" / "episodes").glob("chunk-*/*.parquet"))
    if not episode_files:
        raise FileNotFoundError(f"No episode metadata files found under {dataset_root / 'meta' / 'episodes'}")

    episodes = pd.concat([pd.read_parquet(path) for path in episode_files], ignore_index=True)
    matches = episodes[episodes["episode_index"] == episode_id]
    if matches.empty:
        available = episodes["episode_index"].tolist()
        raise ValueError(f"Episode {episode_id} not found. Available examples: {available[:10]}")
    episode_meta = matches.iloc[0]

    data_chunk = int(episode_meta["data/chunk_index"])
    data_file = int(episode_meta["data/file_index"])
    parquet = download_dataset_file(
        _format_lerobot_path(data_path_template, chunk_index=data_chunk, file_index=data_file)
    )
    episode_df = pd.read_parquet(parquet)
    episode_df = episode_df[episode_df["episode_index"] == episode_id].reset_index(drop=True)
    task_text: str | None = None
    tasks_path = dataset_root / "meta" / "tasks.parquet"
    if tasks_path.exists() and "task_index" in episode_df.columns:
        tasks = pd.read_parquet(tasks_path)
        task_index = int(episode_df.iloc[0]["task_index"])
        task_matches = tasks[tasks["task_index"] == task_index]
        if not task_matches.empty:
            task_text = str(task_matches.index[0])

    def video_info(video_key: str) -> tuple[Path, int]:
        chunk_col = f"videos/{video_key}/chunk_index"
        file_col = f"videos/{video_key}/file_index"
        start_col = f"videos/{video_key}/from_timestamp"
        chunk_index = int(episode_meta[chunk_col]) if chunk_col in episode_meta.index else 0
        file_index = int(episode_meta[file_col]) if file_col in episode_meta.index else data_file
        start_seconds = float(episode_meta[start_col]) if start_col in episode_meta.index else 0.0
        video_path = download_dataset_file(
            _format_lerobot_path(
                video_path_template,
                video_key=video_key,
                chunk_index=chunk_index,
                file_index=file_index,
            )
        )
        return video_path, int(round(start_seconds * fps))

    video_paths: dict[str, Path] = {}
    video_start_frames: dict[str, int] = {}
    for image_key in image_keys:
        if image_key in episode_df.columns and features.get(image_key, {}).get("dtype") == "image":
            continue
        video_paths[image_key], video_start_frames[image_key] = video_info(image_key)

    return ReplayEpisode(
        dataset_root=dataset_root,
        episode_id=episode_id,
        episode_df=episode_df,
        video_paths=video_paths,
        video_start_frames=video_start_frames,
        image_keys=image_keys,
        fps=fps,
        action_dim=action_dim,
        state_dim=state_dim,
        task=task_text,
    )


def read_video_rgb(path: Path, max_frames: int | None = None, start_frame: int = 0) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if start_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if max_frames is not None and len(frames) >= max_frames:
            break
    cap.release()
    if frames:
        return frames

    try:
        import imageio

        reader = imageio.get_reader(str(path), "ffmpeg")
        try:
            for frame_index, frame in enumerate(reader):
                if frame_index < start_frame:
                    continue
                frames.append(np.asarray(frame[..., :3], dtype=np.uint8))
                if max_frames is not None and len(frames) >= max_frames:
                    break
        finally:
            reader.close()
    except Exception as exc:
        raise RuntimeError(
            "Could not decode replay video. This dataset may require AV1 decode support; "
            "install imageio-ffmpeg or re-encode the videos to H.264."
        ) from exc

    if not frames:
        raise RuntimeError(f"Decoded zero frames from {path}; check codec support and start_frame={start_frame}.")
    return frames


def decode_image_cell(value: Any) -> np.ndarray:
    if isinstance(value, dict) and value.get("bytes") is not None:
        return np.asarray(Image.open(BytesIO(value["bytes"])).convert("RGB"), dtype=np.uint8)
    if isinstance(value, (bytes, bytearray)):
        return np.asarray(Image.open(BytesIO(value)).convert("RGB"), dtype=np.uint8)
    arr = np.asarray(value)
    if arr.ndim == 3:
        return arr[..., :3].astype(np.uint8)
    raise ValueError(f"Unsupported image cell type: {type(value)!r}")


def read_replay_rgb_sequence(
    replay: ReplayEpisode,
    image_key: str,
    max_frames: int | None = None,
) -> list[np.ndarray]:
    if image_key in replay.video_paths:
        return read_video_rgb(
            replay.video_paths[image_key],
            max_frames=max_frames,
            start_frame=replay.video_start_frames[image_key],
        )
    rows = replay.episode_df[image_key]
    if max_frames is not None:
        rows = rows.iloc[:max_frames]
    return [decode_image_cell(value) for value in rows]


def as_vector(values: Any, name: str, expected_dim: int | None = None) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 0:
        raise ValueError(f"Expected {name} to be a vector, got scalar value {arr!r}")
    if arr.ndim > 1:
        arr = arr.reshape(-1, arr.shape[-1])[0]
    arr = arr.astype(np.float32)
    if expected_dim is not None:
        arr = arr[:expected_dim]
        if arr.shape[0] != expected_dim:
            raise ValueError(f"Expected {name} length {expected_dim}, got shape {arr.shape}")
    return arr


def as_action_vector(values: Any, name: str = "action", expected_dim: int | None = None) -> np.ndarray:
    return as_vector(values, name=name, expected_dim=expected_dim)


def make_policy_observation(frames: dict[str, np.ndarray], state: Any, task: str, state_dim: int) -> dict[str, Any]:
    obs: dict[str, Any] = {
        STATE_KEY: as_vector(state, name="observation.state", expected_dim=state_dim)[None, :],
        TASK_KEY: [task],
    }
    for dataset_key, frame in frames.items():
        obs[dataset_image_key_to_model_key(dataset_key)] = frame.astype(np.float32)[None, ...] / 255.0
    return obs


def openvino_config_for_device(cache_dir: Path) -> dict[str, str]:
    return {"CACHE_DIR": str(cache_dir)}


def benchmark_pi05(
    *,
    model_dir: Path,
    replay: ReplayEpisode,
    task: str,
    device: str,
    cache_dir: Path,
    runs: int = 5,
) -> tuple[InferenceModel, dict[str, Any]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    frames = {key: read_replay_rgb_sequence(replay, key, max_frames=1)[0] for key in replay.image_keys}
    obs = make_policy_observation(frames, replay.episode_df.iloc[0]["observation.state"], task, replay.state_dim)

    start = time.perf_counter()
    model = InferenceModel.load(model_dir, backend="openvino", device=device, **openvino_config_for_device(cache_dir))
    load_ms = (time.perf_counter() - start) * 1000

    model.reset()
    _ = model.predict_action_chunk(obs)
    first_action = model.select_action(obs)
    timings = []
    for _ in range(runs):
        model.reset()
        start = time.perf_counter()
        chunk = model.predict_action_chunk(obs)
        timings.append((time.perf_counter() - start) * 1000)

    avg_ms = float(np.mean(timings))
    return model, {
        "device": device,
        "load_ms": load_ms,
        "avg_ms": avg_ms,
        "p50_ms": float(np.percentile(timings, 50)),
        "p95_ms": float(np.percentile(timings, 95)),
        "fps": float(1000 / avg_ms),
        "chunk_shape": tuple(chunk.shape),
        "select_action_shape": tuple(np.asarray(first_action).shape),
    }


def benchmark_with_fallback(**kwargs: Any) -> tuple[InferenceModel, dict[str, Any]]:
    device = kwargs["device"]
    try:
        return benchmark_pi05(**kwargs)
    except RuntimeError as exc:
        print(f"[WARN] Pi0.5 OpenVINO failed on {device}: {type(exc).__name__}: {exc}")
        if device == "CPU":
            raise
        print("[INFO] Falling back to CPU so the notebook can continue.")
        kwargs["device"] = "CPU"
        return benchmark_pi05(**kwargs)


def _so101_points(values: np.ndarray, origin: tuple[int, int] = (590, 330), scale: float = 1.0) -> list[tuple[int, int]]:
    vals = np.asarray(values, dtype=np.float32)
    lengths = np.array([70, 58, 48, 34], dtype=np.float32) * scale
    angles = np.deg2rad(
        [
            -90 + vals[0] * 0.55,
            vals[1] * 0.45,
            vals[2] * 0.35,
            vals[3] * 0.25 + vals[4] * 0.08,
        ]
    )
    pts = [np.array(origin, dtype=np.float32)]
    heading = 0.0
    for length, angle in zip(lengths, angles):
        heading += angle
        pts.append(pts[-1] + np.array([np.cos(heading), np.sin(heading)]) * length)
    return [(int(x), int(y)) for x, y in pts]


def _draw_arm(draw: ImageDraw.ImageDraw, values: np.ndarray, color: tuple[int, int, int], width: int = 9) -> None:
    pts = _so101_points(values)
    for a, b in zip(pts[:-1], pts[1:]):
        draw.line([a, b], fill=color, width=width)
        draw.line([a, b], fill=(245, 248, 250), width=max(2, width // 3))
    for p in pts:
        draw.ellipse([p[0] - 7, p[1] - 7, p[0] + 7, p[1] + 7], fill=(20, 24, 30), outline=color, width=3)
    gripper = float(np.asarray(values)[5])
    ee = pts[-1]
    span = int(8 + np.clip(gripper, 0, 100) * 0.12)
    draw.line([(ee[0] - span, ee[1] - 9), (ee[0] + span, ee[1] + 9)], fill=color, width=4)


def _draw_bars(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    width: int,
    height: int,
    pred: np.ndarray,
    expert: np.ndarray,
) -> None:
    names = list(DEFAULT_ACTION_NAMES[: len(pred)])
    row_h = height // len(names)
    center = x + width // 2
    draw.line([(center, y), (center, y + height)], fill=(80, 90, 102), width=1)
    for i, name in enumerate(names):
        yy = y + i * row_h + 6
        draw.text((x, yy), name, fill=(220, 226, 234))
        for val, color, offset in [(expert[i], (95, 170, 255), 13), (pred[i], (255, 190, 85), 28)]:
            v = float(np.clip(val, -100, 100))
            bar = int((v / 100.0) * (width * 0.32))
            draw.rectangle([min(center, center + bar), yy + offset, max(center, center + bar), yy + offset + 8], fill=color)


def _make_overlay_frame(
    camera_frames: dict[str, np.ndarray],
    state: np.ndarray,
    pred: np.ndarray,
    expert: np.ndarray,
    frame_idx: int,
    latency_ms: float,
    device: str,
    task: str,
) -> Image.Image:
    canvas = Image.new("RGB", (1120, 720), (18, 22, 28))
    camera_items = list(camera_frames.items())[:2]
    if not camera_items:
        raise ValueError("At least one camera frame is required for replay visualization.")
    if len(camera_items) == 1:
        camera_items.append(camera_items[0])
    for (key, frame), pos in zip(camera_items, [(20, 72), (20, 374)]):
        image = Image.fromarray(frame).resize((512, 288))
        draw_image = ImageDraw.Draw(image)
        label = dataset_image_key_to_model_key(key)
        draw_image.rectangle([0, 0, 220, 26], fill=(0, 0, 0))
        draw_image.text((8, 6), label, fill=(255, 255, 255))
        canvas.paste(image, pos)

    draw = ImageDraw.Draw(canvas)
    draw.text((20, 24), "SO-101 Pi0.5 Replay: PhysicalAI + OpenVINO", fill=(242, 246, 250))
    draw.text((20, 48), f"task={task} | device={device} | frame={frame_idx:03d} | latency={latency_ms:.1f} ms", fill=(170, 184, 199))
    draw.rectangle([560, 72, 1098, 430], outline=(65, 76, 90), width=2)
    draw.text((580, 92), "Joint-space viewer", fill=(235, 241, 245))
    draw.text((580, 116), "blue: observed state   amber: Pi0.5 predicted target", fill=(170, 184, 199))
    _draw_arm(draw, state[:6], (95, 170, 255), width=11)
    _draw_arm(draw, pred, (255, 190, 85), width=7)
    draw.text((580, 398), f"mean |pred - expert| = {float(np.mean(np.abs(pred - expert))):.2f}", fill=(235, 241, 245))
    draw.rectangle([560, 454, 1098, 704], outline=(65, 76, 90), width=2)
    draw.text((580, 468), "Action comparison in SO-101 normalized joint space", fill=(235, 241, 245))
    _draw_bars(draw, 580, 500, 490, 186, pred, expert)
    return canvas


def describe_replay_mae(mean_mae: float) -> str:
    if mean_mae < 5.0:
        return "low offline MAE on this replay dataset; replay domain looks consistent"
    if mean_mae < 15.0:
        return "moderate offline MAE; inspect camera/domain match in the overlay"
    return "high offline MAE; often caused by dataset, camera, task, or action-distribution mismatch"


def run_replay_visualization(
    *,
    model: InferenceModel,
    model_dir: Path,
    replay: ReplayEpisode,
    task: str,
    device: str,
    cache_dir: Path,
    output_dir: Path,
    max_rendered_frames: int = 120,
    render_stride: int = 3,
) -> dict[str, Any]:
    max_replay_steps = max_rendered_frames * render_stride
    video_frames = {
        key: read_replay_rgb_sequence(replay, key, max_frames=max_replay_steps) for key in replay.image_keys
    }
    n = min(*(len(frames) for frames in video_frames.values()), len(replay.episode_df), max_replay_steps)
    if n == 0:
        raise RuntimeError("No replay frames were decoded.")

    if model is None:
        model = InferenceModel.load(model_dir, backend="openvino", device=device, **openvino_config_for_device(cache_dir))
    model.reset()

    vis_frames: list[Image.Image] = []
    latencies: list[float] = []
    errors: list[float] = []
    predictions: list[np.ndarray] = []
    expert_actions: list[np.ndarray] = []

    for frame_idx in range(n):
        state = as_vector(
            replay.episode_df.iloc[frame_idx]["observation.state"],
            name="observation.state",
            expected_dim=replay.state_dim,
        )
        expert = as_action_vector(
            replay.episode_df.iloc[frame_idx]["action"],
            name="expert action",
            expected_dim=replay.action_dim,
        )
        frame_batch = {key: frames[frame_idx] for key, frames in video_frames.items()}
        obs = make_policy_observation(frame_batch, state, task, replay.state_dim)

        start = time.perf_counter()
        pred = as_action_vector(model.select_action(obs), name="predicted action", expected_dim=replay.action_dim)
        latency_ms = (time.perf_counter() - start) * 1000
        latencies.append(latency_ms)
        errors.append(float(np.mean(np.abs(pred - expert))))
        predictions.append(pred)
        expert_actions.append(expert)
        if frame_idx % render_stride == 0 and len(vis_frames) < max_rendered_frames:
            vis_frames.append(
                _make_overlay_frame(
                    frame_batch,
                    state,
                    pred,
                    expert,
                    frame_idx,
                    latency_ms,
                    device,
                    task,
                )
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    gif_path = output_dir / "so101_pick_place_pi05_openvino.gif"
    vis_frames[0].save(gif_path, save_all=True, append_images=vis_frames[1:], duration=66, loop=0)
    mean_mae = float(np.mean(errors))
    per_joint_mae = np.mean(np.abs(np.stack(predictions) - np.stack(expert_actions)), axis=0)
    action_names = list(DEFAULT_ACTION_NAMES[: len(per_joint_mae)])
    return {
        "gif_path": gif_path,
        "gif": IPyImage(filename=str(gif_path)),
        "steps": len(predictions),
        "rendered_frames": len(vis_frames),
        "avg_select_action_ms": float(np.mean(latencies)),
        "avg_mae": mean_mae,
        "per_joint_mae": dict(zip(action_names, per_joint_mae.round(3).tolist())),
        "interpretation": describe_replay_mae(mean_mae),
        "predicted_actions": np.stack(predictions),
        "expert_actions": np.stack(expert_actions),
    }


def _mujoco_pick_place_xml() -> str:
    return """
<mujoco model="cartesian_pick_place_visualizer">
  <compiler angle="degree" autolimits="true"/>
  <option timestep="0.02" gravity="0 0 -9.81"/>
  <visual>
    <global offwidth="960" offheight="640"/>
  </visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.18 0.20 0.22" rgb2="0.25 0.27 0.30" width="256" height="256"/>
    <material name="floor" texture="grid" texrepeat="6 6" reflectance="0.15"/>
    <material name="arm_blue" rgba="0.20 0.58 0.95 1"/>
    <material name="arm_orange" rgba="1.00 0.62 0.18 1"/>
    <material name="joint_dark" rgba="0.08 0.10 0.13 1"/>
    <material name="target" rgba="0.25 0.85 0.50 1"/>
    <material name="goal" rgba="0.20 0.55 0.95 0.35"/>
  </asset>
  <worldbody>
    <light pos="0 -1.5 2.5" dir="0 0 -1" diffuse="0.9 0.9 0.9"/>
    <geom type="plane" size="1.4 1.4 0.02" material="floor"/>
    <geom name="place_goal" type="box" pos="-0.22 0.22 0.006" size="0.06 0.06 0.006" material="goal"/>
    <body name="target" pos="0.24 -0.18 0.035">
      <freejoint name="target_free"/>
      <geom type="box" size="0.035 0.035 0.035" material="target"/>
    </body>
    <body name="base" pos="-0.48 -0.38 0.02">
      <geom type="cylinder" size="0.065 0.04" material="joint_dark"/>
      <geom type="capsule" fromto="0 0 0.03 0 0 0.40" size="0.015" material="joint_dark"/>
    </body>
    <body name="ee" pos="0.24 -0.18 0.25">
      <freejoint name="ee_free"/>
      <geom type="capsule" fromto="0 0 0.22 0 0 0.02" size="0.018" material="arm_blue"/>
      <geom type="box" pos="0 0 0" size="0.055 0.032 0.018" material="arm_orange"/>
      <body name="left_finger" pos="0 0.018 -0.055">
        <joint name="left_finger_slide" type="slide" axis="0 1 0" range="0 0.035"/>
        <geom type="box" pos="0 0 -0.03" size="0.014 0.006 0.045" material="arm_orange"/>
      </body>
      <body name="right_finger" pos="0 -0.018 -0.055">
        <joint name="right_finger_slide" type="slide" axis="0 -1 0" range="0 0.035"/>
        <geom type="box" pos="0 0 -0.03" size="0.014 0.006 0.045" material="arm_orange"/>
      </body>
    </body>
    <camera name="overview" pos="0.72 -0.92 0.58" xyaxes="0.80 0.60 0.00 -0.32 0.43 0.85"/>
  </worldbody>
</mujoco>
"""


def _smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def _segment(progress: float, start: float, end: float) -> float:
    return _smoothstep((progress - start) / max(1e-6, end - start))


def _interp(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    return a * (1.0 - alpha) + b * alpha


def _pick_place_state(frame_idx: int, total_frames: int) -> tuple[np.ndarray, np.ndarray, float]:
    progress = frame_idx / max(1, total_frames - 1)
    cube_start = np.array([0.24, -0.18, 0.035], dtype=np.float64)
    cube_goal = np.array([-0.22, 0.22, 0.035], dtype=np.float64)
    above_start = cube_start + np.array([0.0, 0.0, 0.22])
    grasp = cube_start + np.array([0.0, 0.0, 0.105])
    lift = cube_start + np.array([0.0, 0.0, 0.30])
    above_goal = cube_goal + np.array([0.0, 0.0, 0.30])
    place = cube_goal + np.array([0.0, 0.0, 0.105])
    retreat = cube_goal + np.array([-0.08, 0.0, 0.30])

    if progress < 0.18:
        ee_pos = above_start
        cube_pos = cube_start
        grip_gap = 0.030
    elif progress < 0.34:
        ee_pos = _interp(above_start, grasp, _segment(progress, 0.18, 0.34))
        cube_pos = cube_start
        grip_gap = 0.030
    elif progress < 0.44:
        ee_pos = grasp
        cube_pos = cube_start
        grip_gap = 0.030 * (1.0 - _segment(progress, 0.34, 0.44)) + 0.004 * _segment(progress, 0.34, 0.44)
    elif progress < 0.58:
        alpha = _segment(progress, 0.44, 0.58)
        ee_pos = _interp(grasp, lift, alpha)
        cube_pos = ee_pos + np.array([0.0, 0.0, -0.07])
        grip_gap = 0.004
    elif progress < 0.76:
        alpha = _segment(progress, 0.58, 0.76)
        ee_pos = _interp(lift, above_goal, alpha)
        cube_pos = ee_pos + np.array([0.0, 0.0, -0.07])
        grip_gap = 0.004
    elif progress < 0.88:
        alpha = _segment(progress, 0.76, 0.88)
        ee_pos = _interp(above_goal, place, alpha)
        cube_pos = ee_pos + np.array([0.0, 0.0, -0.07])
        grip_gap = 0.004
    elif progress < 0.94:
        ee_pos = place
        cube_pos = cube_goal
        grip_gap = 0.004 * (1.0 - _segment(progress, 0.88, 0.94)) + 0.030 * _segment(progress, 0.88, 0.94)
    else:
        ee_pos = _interp(place, retreat, _segment(progress, 0.94, 1.0))
        cube_pos = cube_goal
        grip_gap = 0.030

    return ee_pos, cube_pos, float(grip_gap)


def _annotate_mujoco_frame(
    frame: np.ndarray,
    *,
    frame_idx: int,
    grip_gap: float,
    source: str,
) -> Image.Image:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, 960, 64], fill=(12, 16, 22))
    draw.text((18, 14), "MuJoCo pick-and-place rollout visualization", fill=(245, 248, 250))
    draw.text((18, 38), f"source={source} | frame={frame_idx:03d}", fill=(178, 190, 204))
    draw.text((680, 38), f"finger gap={grip_gap:.3f} m", fill=(178, 190, 204))
    return image


def run_mujoco_visualization(
    *,
    actions: np.ndarray | None,
    output_dir: Path,
    source: str,
    max_rendered_frames: int = 180,
) -> dict[str, Any]:
    """Render a scripted Cartesian gripper pick-and-place scene in MuJoCo."""
    try:
        import mujoco
    except ImportError as exc:
        raise RuntimeError("Install mujoco before running the MuJoCo visualization cell.") from exc

    frame_count = min(max_rendered_frames, len(actions)) if actions is not None and len(actions) else max_rendered_frames
    source = "scripted MuJoCo pick-and-place trajectory"

    model = mujoco.MjModel.from_xml_string(_mujoco_pick_place_xml())
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=640, width=960)
    ee_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ee_free")
    target_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "target_free")
    left_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_finger_slide")
    right_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_finger_slide")
    ee_qpos_addr = int(model.jnt_qposadr[ee_joint_id])
    target_qpos_addr = int(model.jnt_qposadr[target_joint_id])
    left_qpos_addr = int(model.jnt_qposadr[left_joint_id])
    right_qpos_addr = int(model.jnt_qposadr[right_joint_id])

    frames: list[Image.Image] = []
    try:
        for frame_idx in range(frame_count):
            ee_pos, cube_pos, grip_gap = _pick_place_state(frame_idx, frame_count)
            data.qpos[ee_qpos_addr : ee_qpos_addr + 7] = [ee_pos[0], ee_pos[1], ee_pos[2], 1.0, 0.0, 0.0, 0.0]
            data.qpos[target_qpos_addr : target_qpos_addr + 7] = [
                cube_pos[0],
                cube_pos[1],
                cube_pos[2],
                1.0,
                0.0,
                0.0,
                0.0,
            ]
            data.qpos[left_qpos_addr] = grip_gap
            data.qpos[right_qpos_addr] = grip_gap
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera="overview")
            frame = renderer.render()
            frames.append(_annotate_mujoco_frame(frame, frame_idx=frame_idx, grip_gap=grip_gap, source=source))
    finally:
        renderer.close()

    output_dir.mkdir(parents=True, exist_ok=True)
    gif_path = output_dir / "mujoco_pick_place_visualization.gif"
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=50, loop=0)
    return {
        "gif_path": gif_path,
        "gif": IPyImage(filename=str(gif_path)),
        "frames": len(frames),
        "source": source,
    }
