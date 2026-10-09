#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Evaluate and benchmark the Zenoh Remote Inference API.

This script supports:
  1. Standalone loopback evaluation (runs both server and client benchmarks).
  2. Running as a dedicated Zenoh inference server.
  3. Running as a dedicated Zenoh inference client.

Usage examples:
  # 1. Quick local benchmark with synthetic observation & mock model:
  uv run python examples/runtime/zenoh_inference.py --mode loopback

  # 2. Run benchmark with custom chunk size and request count:
  uv run python examples/runtime/zenoh_inference.py --mode loopback --requests 50 --chunk-size 32

    # 3. Dedicated server mode (serves one model namespace):
  uv run python examples/runtime/zenoh_inference.py --mode server --endpoint tcp/127.0.0.1:7447

  # 4. Dedicated client mode:
  uv run python examples/runtime/zenoh_inference.py --mode client --endpoint tcp/127.0.0.1:7447
"""

from __future__ import annotations

import argparse
import threading
import time
from typing import Any

import numpy as np

from physicalai.inference.remote import InferenceServer, RemoteInferenceModel, RemoteInferenceUnavailableError
from physicalai.runtime import (
    AsyncExecution,
    ChunkedActionQueue,
    RTCActionQueue,
    RTCExecution,
    SyncExecution,
)
from physicalai.transport._zenoh import endpoint_for_key, model_key_prefix


class SyntheticInferenceModel:
    """Synthetic model for latency and throughput evaluation without heavy weights."""

    def __init__(self, action_dim: int = 6, chunk_size: int = 16, simulated_inference_s: float = 0.0) -> None:
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.simulated_inference_s = simulated_inference_s
        self.reset_count = 0
        self.predict_count = 0

    def predict_action_chunk(self, observation: dict[str, Any]) -> np.ndarray:
        if self.simulated_inference_s > 0:
            time.sleep(self.simulated_inference_s)
        self.predict_count += 1
        # Generate predictable synthetic trajectory
        t = np.linspace(0, 1, self.chunk_size, dtype=np.float32)[:, None]
        base = np.arange(self.action_dim, dtype=np.float32)[None, :]
        return np.sin(t + base)

    def __call__(self, inputs: dict[str, Any]) -> dict[str, np.ndarray]:
        return {"action": self.predict_action_chunk(inputs)[np.newaxis]}

    def reset(self) -> None:
        self.reset_count += 1


def make_sample_observation(image_h: int = 224, image_w: int = 224, joint_dim: int = 6) -> dict[str, Any]:
    """Generate sample observation containing RGB image and joint positions."""
    return {
        "state": np.random.randn(1, joint_dim).astype(np.float32),
        "images.overhead": np.random.randint(0, 256, (image_h, image_w, 3), dtype=np.uint8),
        "joint_positions": np.random.randn(joint_dim).astype(np.float32),
    }


def run_server(
    model_name: str,
    endpoint: str | None,
    listen_host: str,
    server_port: int | None,
    allow_all_interfaces: bool,
    simulated_latency_ms: float,
    chunk_size: int,
) -> None:
    print(f"[Server] Starting Zenoh Remote Inference Server for model '{model_name}'...")
    model = SyntheticInferenceModel(
        chunk_size=chunk_size,
        simulated_inference_s=simulated_latency_ms / 1000.0,
    )
    listen = endpoint or endpoint_for_key(model_key_prefix(model_name), listen_host, server_port)
    server = InferenceServer(
        model=model,  # type: ignore[arg-type]
        name=model_name,
        listen=listen,
        allow_all_interfaces=allow_all_interfaces,
    )
    try:
        server.start()
        print(f"[Server] Listening on endpoint: {server.endpoint}")
        print("[Server] Server is running. Press Ctrl+C to terminate.")
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[Server] Shutting down...")
    finally:
        server.stop()
        print("[Server] Stopped.")


def run_client(
    model_name: str,
    endpoint: str | None,
    server_host: str,
    server_port: int | None,
    execution_mode: str,
    num_requests: int,
    warmup_requests: int,
    startup_timeout_s: float = 0.0,
) -> None:
    remote_endpoint = endpoint or endpoint_for_key(model_key_prefix(model_name), server_host, server_port)
    print(f"[Client] Connecting to model '{model_name}' at {remote_endpoint}...")

    model = RemoteInferenceModel(
        name=model_name,
        endpoint=remote_endpoint,
        request_timeout_s=2.0,
    )
    deadline = time.monotonic() + startup_timeout_s
    while True:
        try:
            model.connect()
            break
        except RemoteInferenceUnavailableError:
            if time.monotonic() >= deadline:
                model.close()
                raise
            time.sleep(0.1)

    if execution_mode == "sync":
        execution = SyncExecution()
        queue = ChunkedActionQueue()
    elif execution_mode == "async":
        execution = AsyncExecution()
        queue = ChunkedActionQueue()
    else:
        execution = RTCExecution(
            chunk_size=model.chunk_size,
            execution_horizon=max(1, model.chunk_size // 2),
            fps=30.0,
        )
        queue = RTCActionQueue()

    execution.start(model, queue)

    sample_obs = make_sample_observation()

    try:
        print(f"[Client] Running warmup ({warmup_requests} request(s))...")
        execution.warmup(sample_obs)
        for _ in range(max(0, warmup_requests - 1)):
            model.predict_action_chunk(sample_obs)
        print(f"[Client] Warmup successful. Chunk size: {model.chunk_size}")

        execution.reset(reset_model=True)
        queue.reset()

        print(f"[Client] Benchmarking {num_requests} inference requests...")
        latencies_ms: list[float] = []

        for i in range(num_requests):
            if execution_mode == "rtc":
                while not queue.below_threshold(execution.queue_threshold):
                    queue.pop()
            else:
                while queue.remaining > 0:
                    queue.pop()

            previous_inference_count = execution.inference_count
            start_t = time.perf_counter()
            execution.maybe_request(sample_obs)

            # Wait for the selected execution strategy to complete its refill.
            deadline = time.perf_counter() + 5.0
            while execution.inference_count == previous_inference_count:
                if time.perf_counter() > deadline:
                    raise TimeoutError(f"Request {i + 1} timed out waiting for inference")
                time.sleep(0.0005)

            elapsed_ms = (time.perf_counter() - start_t) * 1000.0
            latencies_ms.append(elapsed_ms)

        # Print statistics
        latencies = np.array(latencies_ms)
        print("\n" + "=" * 50)
        print(f" Zenoh Remote Inference Benchmark ({execution_mode})")
        print("=" * 50)
        print(f"Total Requests  : {num_requests}")
        print(f"Total Time      : {latencies.sum() / 1000.0:.3f} s")
        print(f"Throughput      : {num_requests / (latencies.sum() / 1000.0):.2f} req/s")
        print(f"Latency Mean    : {latencies.mean():.2f} ms")
        print(f"Latency Median  : {np.median(latencies):.2f} ms")
        print(f"Latency Min     : {latencies.min():.2f} ms")
        print(f"Latency Max     : {latencies.max():.2f} ms")
        print(f"Latency P95     : {np.percentile(latencies, 95):.2f} ms")
        print(f"Latency P99     : {np.percentile(latencies, 99):.2f} ms")
        print("=" * 50 + "\n")
    finally:
        execution.stop()
        model.close()
        print("[Client] Disconnected.")


def run_loopback(
    model_name: str,
    execution_mode: str,
    requests: int,
    warmup: int,
    chunk_size: int,
    simulated_latency_ms: float,
) -> None:
    import socket

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    endpoint = f"tcp/127.0.0.1:{port}"
    print(f"[Loopback] Setting up model '{model_name}' on {endpoint}...")
    model = SyntheticInferenceModel(chunk_size=chunk_size, simulated_inference_s=simulated_latency_ms / 1000.0)
    server = InferenceServer(
        model=model,
        name=model_name,
        listen=endpoint,
    )  # type: ignore[arg-type]

    server.start()
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    try:
        run_client(
            model_name=model_name,
            endpoint=endpoint,
            server_host="127.0.0.1",
            server_port=port,
            execution_mode=execution_mode,
            num_requests=requests,
            warmup_requests=warmup,
            startup_timeout_s=10.0,
        )
    finally:
        print("[Loopback] Stopping server...")
        server.stop()
        server_thread.join(timeout=3.0)
        print("[Loopback] Completed.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the Zenoh Remote Inference API",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["loopback", "server", "client"],
        default="loopback",
        help="Evaluation mode: loopback (both in one process), server, or client",
    )
    parser.add_argument(
        "--model-name",
        default="eval",
        help="Model identity used to namespace inference keys and validate startup",
    )
    parser.add_argument(
        "--execution-mode",
        choices=("sync", "async", "rtc"),
        default="async",
        help="Scheduling strategy used with the same remote model",
    )
    parser.add_argument(
        "--endpoint",
        default=None,
        help="Explicit Zenoh endpoint override; otherwise the port is derived from --model-name",
    )
    parser.add_argument(
        "--server-host",
        default="127.0.0.1",
        help="Server host to connect to in client mode",
    )
    parser.add_argument(
        "--listen-host",
        default="127.0.0.1",
        help="Loopback host to bind in server mode (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--allow-all-interfaces",
        action="store_true",
        help="Permit a non-loopback listen host in server mode",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Optional fixed port; defaults to a deterministic port derived from --model-name",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=25,
        help="Number of inference requests to benchmark",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Number of warmup requests",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=16,
        help="Action chunk size returned by the model",
    )
    parser.add_argument(
        "--simulated-latency-ms",
        type=float,
        default=0.0,
        help="Simulated server-side model inference time in milliseconds",
    )

    args = parser.parse_args()

    if args.mode == "server":
        run_server(
            model_name=args.model_name,
            endpoint=args.endpoint,
            listen_host=args.listen_host,
            server_port=args.port,
            allow_all_interfaces=args.allow_all_interfaces,
            simulated_latency_ms=args.simulated_latency_ms,
            chunk_size=args.chunk_size,
        )
    elif args.mode == "client":
        run_client(
            model_name=args.model_name,
            endpoint=args.endpoint,
            server_host=args.server_host,
            server_port=args.port,
            execution_mode=args.execution_mode,
            num_requests=args.requests,
            warmup_requests=args.warmup,
        )
    else:
        run_loopback(
            model_name=args.model_name,
            execution_mode=args.execution_mode,
            requests=args.requests,
            warmup=args.warmup,
            chunk_size=args.chunk_size,
            simulated_latency_ms=args.simulated_latency_ms,
        )


if __name__ == "__main__":
    main()
