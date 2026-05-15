# pyrefly: ignore-errors
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Async threaded inference for SO-101 with lerp-blended action chunking.

Supports two backends:
- **PyTorch/CUDA**: Pi05 checkpoint (.ckpt file) — GPU inference via select_action.
- **OpenVINO**: Exported model directory — CPU/GPU inference via InferenceModel.

The backend is auto-detected from the checkpoint path:
- File ending in .ckpt → PyTorch Pi05
- Directory → OpenVINO InferenceModel

The main thread runs the robot control loop at a fixed FPS.
A background thread runs model inference and pushes action chunks
into a shared queue with lerp blending for smooth transitions.

Uses LeRobot SO101Follower (degrees) for robot control and
physicalai SharedCamera (iceoryx2) for camera input.

Usage (PyTorch):
    python inference_async_pi05.py \
        --model-path pi05/pi05_eugene.ckpt \
        --device cuda \
        --robot-port /dev/ttyACM1 \
        --task "pick a can and place it in a bowl"

Usage (OpenVINO):
    python inference_async_pi05.py \
        --model-path exports/pi05_ov \
        --device CPU \
        --robot-port /dev/ttyACM1 \
        --camera-map overhead=top arm=wrist \
        --task "pick a can and place it in a bowl"
"""

from __future__ import annotations

import argparse
import logging
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from physicalai.capture import SharedCamera
from physicalai.capture.discovery import DeviceInfo, discover_all

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(threadName)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
CAMERA_FPS = 30


# ---------------------------------------------------------------------------
# Queue Mixer — lerp-blends old and new action chunks
# ---------------------------------------------------------------------------

class QueueMixer:
    """Action queue with linear-interpolation blending between chunks.

    When a new chunk arrives while old actions remain, the overlapping
    region is lerp-blended so the robot doesn't jerk.  An *offset*
    parameter lets us skip the first N actions of the new chunk to
    compensate for inference latency.
    """

    def __init__(self, lerp_duration: int = 5) -> None:
        self.queue: np.ndarray | None = None
        self.lerp_duration = lerp_duration
        self.index = 0

    def add(self, chunk: np.ndarray, offset: int = 0) -> None:
        """Merge *chunk* (H, action_dim) into the queue."""
        if self.queue is None or self.index >= len(self.queue):
            self.queue = chunk[offset:]
            self.index = 0
            logger.info(
                "[QueueMixer] First/exhausted → replaced queue (len=%d, offset=%d)",
                len(self.queue), offset,
            )
            return

        remaining = self.queue[self.index:]
        incoming = chunk[offset:]
        n_remain = len(remaining)
        lerp_dur = min(n_remain, self.lerp_duration)

        weights = np.maximum(1.0 - np.arange(n_remain) / max(lerp_dur, 1), 0.0)
        weights = weights[:, np.newaxis]

        n_blend = min(n_remain, len(incoming))
        blended = weights[:n_blend] * remaining[:n_blend] + (1.0 - weights[:n_blend]) * incoming[:n_blend]

        self.queue = np.concatenate([blended, incoming[n_blend:]], axis=0).astype(np.float32)
        self.index = 0
        logger.info(
            "[QueueMixer] Blended chunk (blended=%d, appended=%d, total=%d, lerp=%d, offset=%d)",
            n_blend, max(len(incoming) - n_blend, 0), len(self.queue), lerp_dur, offset,
        )

    def pop(self) -> np.ndarray | None:
        """Pop the next action, or return None if empty."""
        if self.queue is None or self.index >= len(self.queue):
            return None
        action = self.queue[self.index]
        self.index += 1
        return action

    @property
    def remaining(self) -> int:
        if self.queue is None:
            return 0
        return max(len(self.queue) - self.index, 0)

    @property
    def empty(self) -> bool:
        return self.remaining == 0


# ---------------------------------------------------------------------------
# Inference Thread (PyTorch / CUDA)
# ---------------------------------------------------------------------------

class InferenceThread:
    """Background thread that runs model inference on demand.

    Supports two backends:
    - "torch": Pi05 GPU inference via select_action + action_queue draining.
    - "openvino": InferenceModel inference via model(dict)["action"].

    The control loop writes an observation dict into `obs_slot` and sets
    `obs_ready`.  The inference thread picks it up, runs inference, and
    pushes the result (chunk + timing) into `result_slot`.
    """

    def __init__(
        self,
        model: object,
        backend: str,
        device: str,
        task: str,
        flip_cameras: set[str],
        camera_name_map: dict[str, str] | None = None,
        blank_cameras: list[str] | None = None,
    ) -> None:
        self.model = model
        self.backend = backend
        self.device = device
        self.task = task
        self.flip_cameras = flip_cameras
        self.camera_name_map = camera_name_map or {}
        self.blank_cameras = blank_cameras or []

        self._lock = threading.Lock()
        self._obs_slot: dict | None = None
        self._obs_ready = threading.Event()
        self._running_inference = False
        self._request_time = 0.0

        self._result_lock = threading.Lock()
        self._result_slot: tuple[np.ndarray, float] | None = None

        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="InferenceThread", daemon=True)
        self.inference_count = 0

    def start(self) -> None:
        logger.info("[InferenceThread] Starting background thread")
        self._thread.start()

    def stop(self) -> None:
        logger.info("[InferenceThread] Requesting stop")
        self._stop.set()
        self._obs_ready.set()
        self._thread.join(timeout=10.0)
        logger.info("[InferenceThread] Stopped (ran %d inferences)", self.inference_count)

    def request(self, observation: dict) -> bool:
        """Submit an observation dict for inference. Returns False if busy."""
        with self._lock:
            if self._obs_slot is not None or self._running_inference:
                return False
            self._obs_slot = observation
            self._request_time = time.perf_counter()
        self._obs_ready.set()
        logger.debug("[InferenceThread] Observation submitted")
        return True

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._obs_slot is not None or self._running_inference

    @property
    def alive(self) -> bool:
        return self._thread.is_alive()

    @property
    def busy_duration(self) -> float:
        """Seconds since last request was submitted. 0 if not busy."""
        with self._lock:
            if not (self._obs_slot is not None or self._running_inference):
                return 0.0
            return time.perf_counter() - self._request_time

    def force_reset(self) -> None:
        """Clear stuck state so new requests can be submitted."""
        with self._lock:
            self._obs_slot = None
            self._running_inference = False
        logger.warning("[InferenceThread] Force reset — cleared stuck state")

    def get_result(self) -> tuple[np.ndarray, float] | None:
        """Non-blocking fetch of the latest result. Returns (chunk, latency_s) or None."""
        with self._result_lock:
            r = self._result_slot
            self._result_slot = None
        return r

    def _build_torch_observation(self, obs_dict: dict) -> object:
        """Convert raw observation dict to Pi05 Observation on device (torch)."""
        import torch
        from physicalai.data.observation import Observation

        state = torch.tensor(
            [[obs_dict[f"{jn}.pos"] for jn in JOINT_NAMES]],
            dtype=torch.float32,
            device=self.device,
        )

        images = {}
        for cam_key in [k for k in obs_dict if isinstance(obs_dict[k], np.ndarray) and obs_dict[k].ndim == 3]:
            img = np.ascontiguousarray(obs_dict[cam_key])
            if cam_key in self.flip_cameras:
                img = cv2.rotate(img, cv2.ROTATE_180)
            t = torch.from_numpy(img.copy()).float() / 255.0
            t = t.permute(2, 0, 1).unsqueeze(0)
            images[cam_key] = t.to(self.device)

        return Observation(state=state, images=images, task=self.task)

    def _build_ov_observation(self, obs_dict: dict) -> dict[str, np.ndarray]:
        """Convert raw observation dict to InferenceModel input (numpy)."""
        state = np.array([[obs_dict[f"{jn}.pos"] for jn in JOINT_NAMES]], dtype=np.float32)
        observation: dict[str, np.ndarray] = {"state": state, "task": [self.task]}

        for cam_key in [k for k in obs_dict if isinstance(obs_dict[k], np.ndarray) and obs_dict[k].ndim == 3]:
            img = np.ascontiguousarray(obs_dict[cam_key])
            if cam_key in self.flip_cameras:
                img = cv2.rotate(img, cv2.ROTATE_180)
            img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
            img = img.transpose(2, 0, 1).astype(np.float32)[np.newaxis] / 255.0
            model_name = self.camera_name_map.get(cam_key, cam_key)
            observation[f"images.{model_name}"] = img

        for blank_cam in self.blank_cameras:
            observation[f"images.{blank_cam}"] = np.zeros(
                (1, 3, IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.float32,
            )

        return observation

    def _infer_torch(self, obs_dict: dict) -> np.ndarray:
        """Run Pi05 inference and return the full action chunk as numpy."""
        import torch
        from collections import deque

        observation = self._build_torch_observation(obs_dict)
        self.model._action_queue = deque()

        with torch.no_grad():
            action = self.model.select_action(observation)
        if self.device == "cuda":
            torch.cuda.synchronize()

        actions = [action.detach().cpu().numpy()]
        while len(self.model._action_queue) > 0:
            a = self.model._action_queue.popleft()
            actions.append(a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else a)

        return np.array([np.squeeze(a) for a in actions], dtype=np.float32)

    def _infer_ov(self, obs_dict: dict) -> np.ndarray:
        """Run OpenVINO inference and return the full action chunk as numpy."""
        model_input = self._build_ov_observation(obs_dict)
        output = self.model(model_input)["action"]
        chunk = np.squeeze(output)
        if chunk.ndim == 1:
            chunk = chunk[np.newaxis, :]
        return chunk.astype(np.float32)

    def _run(self) -> None:
        logger.info("[InferenceThread] Thread started, waiting for observations")
        while not self._stop.is_set():
            self._obs_ready.wait()
            self._obs_ready.clear()

            if self._stop.is_set():
                break

            with self._lock:
                obs_dict = self._obs_slot
                self._obs_slot = None
                self._running_inference = True

            if obs_dict is None:
                with self._lock:
                    self._running_inference = False
                continue

            self.inference_count += 1
            logger.info("[InferenceThread] >>> Inference #%d START", self.inference_count)
            t0 = time.perf_counter()

            try:
                if self.backend == "torch":
                    chunk = self._infer_torch(obs_dict)
                else:
                    chunk = self._infer_ov(obs_dict)
            except Exception:
                logger.exception("[InferenceThread] Inference failed")
                with self._lock:
                    self._running_inference = False
                continue

            latency = time.perf_counter() - t0
            logger.info(
                "[InferenceThread] <<< Inference #%d DONE  latency=%.1fms  chunk_shape=%s",
                self.inference_count, latency * 1000, chunk.shape,
            )

            with self._result_lock:
                self._result_slot = (chunk, latency)
            with self._lock:
                self._running_inference = False


# ---------------------------------------------------------------------------
# Observation helpers
# ---------------------------------------------------------------------------

def get_full_observation(
    robot: SO101Follower,
    cameras: dict[str, SharedCamera],
) -> dict:
    """Combine robot state + SharedCamera images into one dict.

    LeRobot SO101Follower returns joint values in the correct
    model domain (body joints in degrees, gripper in [0, 100]).
    """
    obs = robot.get_observation()
    for cam_key, cam in cameras.items():
        frame = cam.read_latest()
        obs[cam_key] = frame.data
    return obs


def action_to_robot_dict(action: np.ndarray) -> dict[str, float]:
    """Convert action array to a LeRobot send_action dict."""
    return {f"{jn}.pos": float(action[i]) for i, jn in enumerate(JOINT_NAMES)}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Camera discovery
# ---------------------------------------------------------------------------

def select_cameras() -> dict[str, str]:
    """Discover cameras and interactively ask user to assign overhead/arm."""
    print("\n[cameras] Discovering cameras ...", flush=True)

    all_devices = discover_all()

    # Flatten into a single list (deduplicate across drivers by hardware_id)
    seen: set[str] = set()
    devices: list[DeviceInfo] = []
    for driver_devices in all_devices.values():
        for dev in driver_devices:
            key = dev.hardware_id or f"{dev.driver}:{dev.device_id}"
            if key not in seen:
                seen.add(key)
                devices.append(dev)

    if not devices:
        raise RuntimeError("No cameras found. Check connections and permissions.")

    print(f"\n  Found {len(devices)} camera(s):", flush=True)
    for i, dev in enumerate(devices):
        stable_id = dev.hardware_id or ""
        print(f"    [{i}] {dev.name or dev.model or 'Unknown'}  ({stable_id or dev.device_id})", flush=True)

    cameras: dict[str, str] = {}
    for role in ("overhead", "arm"):
        while True:
            choice = input(f"\n  Select {role} camera (0-{len(devices)-1}), or 's' to skip: ").strip()
            if choice.lower() == "s":
                print(f"    Skipping {role} camera", flush=True)
                break
            if choice.isdigit() and 0 <= int(choice) < len(devices):
                dev = devices[int(choice)]
                # Prefer stable by-id path, fall back to device_id
                path = dev.hardware_id if dev.hardware_id else dev.device_id
                cameras[role] = path
                print(f"    {role} → {path}", flush=True)
                break
            print(f"    Invalid choice. Enter 0-{len(devices)-1} or 's'.")

    if not cameras:
        raise RuntimeError("At least one camera must be selected.")

    return cameras


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Async threaded SO-101 inference (PyTorch or OpenVINO).")
    p.add_argument("--model-path", type=str, required=True,
                    help="Path to Pi05 checkpoint (.ckpt) or OpenVINO export directory")
    p.add_argument("--device", type=str, default="cuda",
                    help="'cuda' or 'cpu' for torch; 'CPU', 'GPU', 'NPU' for OpenVINO")
    p.add_argument("--robot-port", type=str, default="/dev/ttyACM1")
    p.add_argument("--robot-id", type=str, default="my_so101_follower")
    p.add_argument("--overhead-camera", type=str, default=None,
                    help="UVC device path for overhead camera (skip interactive selection)")
    p.add_argument("--arm-camera", type=str, default=None,
                    help="UVC device path for arm camera (skip interactive selection)")
    p.add_argument("--task", type=str, default="pick a can and place it in a bowl")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--queue-threshold", type=float, default=0.5,
                    help="Request new inference when queue drops below this fraction of chunk size.")
    p.add_argument("--lerp-duration", type=int, default=5,
                    help="Number of frames over which to blend old/new chunks.")
    p.add_argument("--flip-cameras", nargs="*", default=[],
                    help="Camera names to rotate 180 degrees.")
    p.add_argument("--infer-timeout", type=float, default=15.0,
                    help="Seconds before force-resetting a stuck inference thread.")
    p.add_argument("--max-speed", type=float, default=90.0,
                    help="Max joint speed in degrees/second. Clamps action deltas per step.")
    p.add_argument("--ramp-steps", type=int, default=30,
                    help="Number of initial steps to ramp up speed (starts at 1/4 max-speed).")
    p.add_argument("--no-compile", action="store_true", default=True,
                    help="Disable torch.compile (default: disabled, torch only)")
    # OpenVINO-specific options
    p.add_argument("--camera-map", nargs="*",
                    help="Map camera names to model input names, e.g. overhead=top arm=wrist (OV only)")
    p.add_argument("--blank-cameras", nargs="*",
                    help="Model camera inputs to fill with zeros, e.g. side (OV only)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Auto-detect backend
    if model_path.is_dir():
        backend = "openvino"
        # Default device for OpenVINO is CPU when user left the torch default
        if args.device in ("cuda", "cuda:0"):
            print(f"[init] Warning: OpenVINO backend does not support CUDA. Falling back to CPU.", flush=True)
            args.device = "CPU"
        else:
            args.device = args.device.upper()
    elif model_path.suffix == ".ckpt":
        backend = "torch"
    else:
        raise ValueError(
            f"Cannot determine backend from '{model_path}'. "
            "Use a .ckpt file for PyTorch or a directory for OpenVINO."
        )

    flip_cameras = set(args.flip_cameras or [])
    camera_name_map = dict(kv.split("=") for kv in (args.camera_map or []))
    blank_cameras = list(args.blank_cameras or [])

    print(f"[init] Model: {model_path}", flush=True)
    print(f"[init] Backend: {backend}", flush=True)
    print(f"[init] Device: {args.device}", flush=True)
    print(f"[init] Flip cameras: {flip_cameras}", flush=True)
    if backend == "openvino":
        print(f"[init] Camera map: {camera_name_map}", flush=True)
        print(f"[init] Blank cameras: {blank_cameras}", flush=True)

    # --- Set up cameras ---
    # Use CLI paths if provided, otherwise run interactive selection
    if args.overhead_camera or args.arm_camera:
        camera_paths: dict[str, str] = {}
        if args.overhead_camera:
            camera_paths["overhead"] = args.overhead_camera
        if args.arm_camera:
            camera_paths["arm"] = args.arm_camera
    else:
        camera_paths = select_cameras()

    cameras: dict[str, SharedCamera] = {}
    for cam_name, cam_device in camera_paths.items():
        cameras[cam_name] = SharedCamera(
            "uvc", device=cam_device,
            width=IMAGE_WIDTH, height=IMAGE_HEIGHT, fps=CAMERA_FPS,
        )

    for name, cam in cameras.items():
        cam.connect()
        print(f"[init] Camera '{name}' connected: {cam.actual_width}x{cam.actual_height} @ {cam.actual_fps}fps", flush=True)

    # --- Load model ---
    if backend == "torch":
        import torch
        from physicalai.policies.pi05 import Pi05

        print("[init] Loading Pi05 model (map_location=cpu) ...", flush=True)
        model = Pi05.load_from_checkpoint(
            str(model_path), map_location="cpu", compile_model=not args.no_compile,
        )
        print("[init] Checkpoint loaded to CPU, moving to device ...", flush=True)
        model = model.to(args.device)
        model.eval()
        print(f"[init] Model loaded on {args.device}", flush=True)
        if args.device == "cuda":
            print(f"[init] VRAM used: {torch.cuda.memory_allocated() / 1e9:.2f} GB", flush=True)
    else:
        import openvino_tokenizers  # noqa: F401
        from physicalai.inference import InferenceModel
        from physicalai.inference.runners import SinglePass

        print(f"[init] Loading OpenVINO model from {model_path} ...", flush=True)
        model = InferenceModel(
            export_dir=str(model_path),
            device=args.device,
            runner=SinglePass(),
        )
        print("[init] Model loaded", flush=True)


    # --- Connect robot (LeRobot SO101Follower, no cameras) ---
    print(f"[init] Connecting robot on {args.robot_port} ...", flush=True)
    robot_cfg = SO101FollowerConfig(
        port=args.robot_port,
        id=args.robot_id,
    )
    robot = SO101Follower(robot_cfg)
    robot.connect()
    print(f"[init] Robot connected (id={args.robot_id})", flush=True)

    # --- Warm-up inference to determine chunk size ---
    print("[init] Getting warm-up observation ...", flush=True)
    warmup_obs = get_full_observation(robot, cameras)
    print(f"[init] Got observation, keys: {list(warmup_obs.keys())}", flush=True)

    print("[init] Running warm-up inference ...", flush=True)
    t0 = time.perf_counter()

    if backend == "torch":
        import torch
        from collections import deque
        from physicalai.data.observation import Observation

        state = torch.tensor(
            [[warmup_obs[f"{jn}.pos"] for jn in JOINT_NAMES]],
            dtype=torch.float32, device=args.device,
        )
        warmup_images = {}
        for cam_key in cameras:
            img = np.ascontiguousarray(warmup_obs[cam_key])
            if cam_key in flip_cameras:
                img = cv2.rotate(img, cv2.ROTATE_180)
            t = torch.from_numpy(img.copy()).float() / 255.0
            t = t.permute(2, 0, 1).unsqueeze(0).to(args.device)
            warmup_images[cam_key] = t

        warmup_observation = Observation(state=state, images=warmup_images, task=args.task)
        model._action_queue = deque()
        with torch.no_grad():
            warmup_action = model.select_action(warmup_observation)
        if args.device == "cuda":
            torch.cuda.synchronize()

        warmup_actions = [warmup_action.detach().cpu().numpy()]
        while len(model._action_queue) > 0:
            a = model._action_queue.popleft()
            warmup_actions.append(a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else a)
        warmup_chunk = np.array([np.squeeze(a) for a in warmup_actions], dtype=np.float32)
    else:
        # OV warm-up
        warmup_state = np.array(
            [[warmup_obs[f"{jn}.pos"] for jn in JOINT_NAMES]], dtype=np.float32,
        )
        warmup_input: dict[str, np.ndarray] = {"state": warmup_state, "task": [args.task]}
        for cam_key in cameras:
            img = np.ascontiguousarray(warmup_obs[cam_key])
            if cam_key in flip_cameras:
                img = cv2.rotate(img, cv2.ROTATE_180)
            img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
            img = img.transpose(2, 0, 1).astype(np.float32)[np.newaxis] / 255.0
            model_name = camera_name_map.get(cam_key, cam_key)
            warmup_input[f"images.{model_name}"] = img
        for blank_cam in blank_cameras:
            warmup_input[f"images.{blank_cam}"] = np.zeros(
                (1, 3, IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.float32,
            )
        warmup_out = model(warmup_input)["action"]
        warmup_chunk = np.squeeze(warmup_out)
        if warmup_chunk.ndim == 1:
            warmup_chunk = warmup_chunk[np.newaxis, :]
        warmup_chunk = warmup_chunk.astype(np.float32)

    warmup_latency = time.perf_counter() - t0
    chunk_size = warmup_chunk.shape[0]
    action_dim = warmup_chunk.shape[1]
    print(f"[init] Warm-up done: chunk_size={chunk_size}, action_dim={action_dim}, "
          f"latency={warmup_latency*1000:.1f}ms", flush=True)

    # --- Init components ---
    queue_mixer = QueueMixer(lerp_duration=args.lerp_duration)
    queue_mixer.add(warmup_chunk, offset=0)

    inference_thread = InferenceThread(
        model, backend, args.device, args.task, flip_cameras,
        camera_name_map=camera_name_map, blank_cameras=blank_cameras,
    )
    inference_thread.start()

    goal_time = 1.0 / args.fps
    threshold = int(chunk_size * args.queue_threshold)
    max_delta_per_step = args.max_speed / args.fps  # degrees per step
    step = 0
    actions_from_queue = 0
    hold_count = 0
    last_action: np.ndarray = warmup_chunk[0]

    # Initialize commanded position from current robot state (degrees/[0,100])
    init_obs = robot.get_observation()
    commanded_pos = np.array(
        [init_obs[f"{jn}.pos"] for jn in JOINT_NAMES], dtype=np.float32,
    )
    print(f"[init] Current joint pos: {commanded_pos}", flush=True)
    print(f"[init] Max speed: {args.max_speed:.0f} deg/s → {max_delta_per_step:.1f} per step @ {args.fps}Hz", flush=True)
    print(f"[init] Ramp-up: {args.ramp_steps} steps", flush=True)
    print("[run] Starting — Ctrl+C to stop", flush=True)

    try:
        while True:
            loop_start = time.perf_counter()

            # 1. Check for inference results
            result = inference_thread.get_result()
            if result is not None:
                chunk, latency = result
                offset = int(latency * args.fps)
                logger.info(
                    "[Control] Got inference #%d: chunk=%s, latency=%.1fms, offset=%d",
                    inference_thread.inference_count, chunk.shape, latency * 1000, offset,
                )
                queue_mixer.add(chunk, offset=offset)
                queue_mixer.lerp_duration = max(offset, 1)

            # 2. Maybe request new inference (with stuck/dead detection)
            if queue_mixer.remaining <= threshold:
                if not inference_thread.alive:
                    logger.error("[Control] Inference thread DEAD — restarting")
                    inference_thread = InferenceThread(
                        model, backend, args.device, args.task, flip_cameras,
                        camera_name_map=camera_name_map, blank_cameras=blank_cameras,
                    )
                    inference_thread.start()
                elif inference_thread.busy_duration > args.infer_timeout:
                    logger.warning(
                        "[Control] Inference stuck for %.0fs — force resetting",
                        inference_thread.busy_duration,
                    )
                    inference_thread.force_reset()

                if not inference_thread.busy:
                    obs = get_full_observation(robot, cameras)
                    submitted = inference_thread.request(obs)
                    if submitted:
                        logger.info(
                            "[Control] Requested inference (queue_remaining=%d, threshold=%d)",
                            queue_mixer.remaining, threshold,
                        )

            # 3. Pop action from queue
            action = queue_mixer.pop()
            if action is not None:
                last_action = action
                actions_from_queue += 1
                hold_count = 0
            else:
                action = last_action
                hold_count += 1
                if hold_count == 1 or hold_count % 30 == 0:
                    logger.warning(
                        "[Control] Queue empty, holding position (hold_count=%d)", hold_count,
                    )

            # 4. Velocity clamp and send
            if step < args.ramp_steps:
                ramp_frac = 0.25 + 0.75 * (step / args.ramp_steps)
                effective_delta = max_delta_per_step * ramp_frac
            else:
                effective_delta = max_delta_per_step

            delta = action - commanded_pos
            clamped_delta = np.clip(delta, -effective_delta, effective_delta)
            commanded_pos = (commanded_pos + clamped_delta).astype(np.float32)
            robot.send_action(action_to_robot_dict(commanded_pos))

            # 5. Timing
            elapsed = time.perf_counter() - loop_start
            sleep_time = goal_time - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

            step += 1
            if step % 100 == 0:
                actual_hz = 1.0 / max(time.perf_counter() - loop_start, 1e-6)
                logger.info(
                    "[Control] Step %d | %.1f Hz | queue=%d | from_queue=%d | holds=%d | inferences=%d",
                    step, actual_hz, queue_mixer.remaining,
                    actions_from_queue, hold_count, inference_thread.inference_count,
                )

    except KeyboardInterrupt:
        logger.info("[Control] Interrupted by user")
    finally:
        inference_thread.stop()
        for name, cam in cameras.items():
            cam.disconnect()
            logger.info("Camera '%s' disconnected", name)
        robot.disconnect()
        logger.info(
            "[Control] Cleanup complete — %d steps, %d actions from queue, %d holds, %d inferences",
            step, actions_from_queue, hold_count, inference_thread.inference_count,
        )


if __name__ == "__main__":
    main(parse_args())
