# Run Inference Remotely over Zenoh

The server disables Zenoh shared memory, including when a custom `zenoh_config`
is supplied. Same-host clients use TCP rather than shared-memory segments, so
repeated short-lived client processes do not depend on retained segment mappings.

Use a `RemoteInferenceModel` on the robot client and a normal exported `InferenceModel` on the inference server. Choose `SyncExecution`, `AsyncExecution`, or `RTCExecution` independently in the client runtime configuration.

Choose an explicit deployment-slot name, independently of the model policy and host, and use it on both client and server. For example, two servers running `pi05` can be named `cell1-act` and `cell2-act`. Zenoh keys are scoped below `physicalai/inference/{name}`, and the default TCP port is derived from that namespace. Both Python constructors accept `host` (default `127.0.0.1`) and optional `port`; Zenoh endpoint strings are internal. At startup, the client connects to that host and port with multicast and gossip disabled, then checks the protocol version, policy identity, and server identity in a handshake.

The exported model stays on the inference server. The client protocol accepts observations and returns model outputs; it does not accept model uploads, export paths, or class paths. Export the model on the server, then transfer the export separately:

```bash
rsync -a ./openvino-export/ user@inference-host:~/models/pi05/openvino/
```

Alternatively, let the server load a trusted Hub repository with `--hub-id`.

## Start the Server

Install/update the repository on the inference host and install the transport extra:

```bash
cd ~/local_dev/physicalai
uv sync --extra transport
```

Load and serve the exported policy:

```bash
uv run physicalai inference serve --name cell1-act \
  --export-dir ~/models/pi05/openvino \
  --policy-name pi05 \
  --backend openvino \
  --device GPU
```

To load directly from the Hugging Face Hub instead, replace `--export-dir ...` with `--hub-id ORG/REPO`:

```bash
uv run physicalai inference serve --name cell1-act \
  --hub-id ORG/REPO \
  --revision COMMIT_SHA \
  --backend openvino \
  --device GPU
```

Pin `--revision` to a reviewed commit SHA for reproducible Hub deployments. If omitted, the Hub's default revision is used.

The server defaults to host `127.0.0.1` and name-derived port `P`, and prints an SSH tunnel command. This is the recommended setup: SSH key authentication and encryption protect the connection, and the client leaves `host` and `port` unset so it connects through the local forwarded port. On the client, run the printed command, or:

```bash
ssh -N -L P:127.0.0.1:P user@inference-host
```

Keep the tunnel open. To find the deterministic port without starting the model server:

```bash
uv run physicalai inference port cell1-act
```

An explicit `--port PORT` changes the server's port; set the client's `port` to match. For a Tailscale deployment, bind the server's Tailscale address and use the same address on the client:

```bash
uv run physicalai inference serve --name cell1-act \
  --export-dir ~/models/pi05/openvino \
  --host 100.90.0.12 --port 45000
```

Set the client model's `host` to `100.90.0.12` and `port` to `45000`. Restrict access with tailnet ACLs. The same pattern applies to a WireGuard interface address, with peer access restricted by the WireGuard configuration.

For direct LAN access, bind the server's LAN address:

```bash
uv run physicalai inference serve --name cell1-act \
  --export-dir ~/models/pi05/openvino \
  --policy-name pi05 \
  --backend openvino \
  --device GPU \
  --host 192.168.1.20 --port 45000
```

Set the client model's `host` to `192.168.1.20` and `port` to `45000`. This transport has no Zenoh-level authentication or encryption: use it only on an isolated VLAN with firewall rules that limit access to trusted clients. The server logs an unauthenticated-exposure warning for every non-loopback listener. It never selects a fallback port when the requested port is occupied. Specific LAN, Tailscale, or WireGuard addresses do not require an additional opt-in. Wildcard binds (`--host 0.0.0.0` or `--host ::`) require `--allow-all-interfaces`. That flag alone does not change `--host`, which still defaults to `127.0.0.1`.

## Robots and GPUs

One server process and model instance serves one robot. Start separate server processes with distinct slot names for separate robots, even when they run the same policy; for example, use `cell1-act` for the first robot and `cell2-act` for the second.

For OpenVINO multi-GPU serving, run one process per GPU and use its device index:

```bash
uv run physicalai inference serve --name cell1-act --export-dir ~/models/pi05/openvino --backend openvino --device GPU.0
uv run physicalai inference serve --name cell2-act --export-dir ~/models/pi05/openvino --backend openvino --device GPU.1
```

For ONNX with CUDA, isolate each process with `CUDA_VISIBLE_DEVICES`:

```bash
CUDA_VISIBLE_DEVICES=0 uv run physicalai inference serve --name cell1-act --export-dir ~/models/pi05/onnx --backend onnx --device cuda
CUDA_VISIBLE_DEVICES=1 uv run physicalai inference serve --name cell2-act --export-dir ~/models/pi05/onnx --backend onnx --device cuda
```

## Configure the Robot Client

Install the transport, robot, camera, and optional Rerun dependencies on the robot host:

```bash
cd ~/local_dev/physicalai
uv sync --extra transport --extra so101 --extra capture --extra observer-rerun
```

Choose exactly one of these `runtime.action_source` YAML blocks. `SyncExecution` blocks the control loop; `AsyncExecution` runs inference in a worker and is the usual starting point for robot control; RTC requires an export with RTC metadata and a compatible RTC policy.

### Sync

```yaml
class_path: physicalai.runtime.PolicySource
init_args:
  model:
    class_path: physicalai.inference.RemoteInferenceModel
    init_args:
      name: cell1-act
      request_timeout_s: 2.0
  execution:
    class_path: physicalai.runtime.SyncExecution
    init_args: {}
  task: "pick up the red cube and place it in the blue bowl"
```

### Async

```yaml
class_path: physicalai.runtime.PolicySource
init_args:
  model:
    class_path: physicalai.inference.RemoteInferenceModel
    init_args:
      name: cell1-act
      request_timeout_s: 2.0
  execution:
    class_path: physicalai.runtime.AsyncExecution
    init_args:
      request_threshold: 0.5
  task: "pick up the red cube and place it in the blue bowl"
```

### RTC

```yaml
class_path: physicalai.runtime.PolicySource
init_args:
  model:
    class_path: physicalai.inference.RemoteInferenceModel
    init_args:
      name: cell1-act
      request_timeout_s: 2.0
  execution:
    class_path: physicalai.runtime.RTCExecution
    init_args:
      chunk_size: null
      execution_horizon: 15
      fps: 30.0
  action_queue:
    class_path: physicalai.runtime.RTCActionQueue
    init_args: {}
  task: "pick up the red cube and place it in the blue bowl"
```

With the default SSH tunnel, omit `host` and `port`; the client connects to loopback at the port derived from its name. For Tailscale, WireGuard, or direct LAN, set `host` and optional `port` to the values shown above. For example, add `host: 100.90.0.12` and `port: 45000` alongside `name` in the model's `init_args`. IPv6 hosts use address literals such as `::1`, without a `tcp/` prefix or port suffix.

Add the selected `action_source` block to your existing runtime YAML, keeping your robot and camera configuration, then run:

```bash
uv run physicalai run --config policy_runtime_remote_zenoh.yaml --run.duration_s=300
```

The client constructor performs no network I/O, so its configuration can be round-tripped through YAML. Protocol v1 uses MessagePack maps and shared tagged NumPy arrays; RGB images use JPEG by default and can use exact raw bytes with `image_codec: raw`.

During the handshake, the client verifies that its configured `image_codec` is
advertised by the server while tolerating additional codecs from newer servers.
It also enforces the server's advertised `max_request_bytes` before sending a
prediction. Returned `action` arrays must have a floating-point dtype and contain
only finite values. The client rejects chunks larger than an advertised size
greater than one; `1` is also the fallback when the export omits chunk-size
metadata, so it is not treated as a hard limit. Camera descriptors are not part
of the handshake metadata.

## Failure Behavior

If a prediction reply identifies a restarted or replaced server, that call raises
`RemoteInferenceModelMismatchError`. The client closes its stale connection and
clears cached actions, metadata, and timing. The next call re-handshakes and
rechecks `expected_policy` and `expected_manifest_sha256` as configuration
consistency guards before sending a new prediction. The manifest hash is
self-reported by the server and does not authenticate it or prove artifact
integrity. The detecting call is never automatically replayed; this permits
client reuse, not automatic recovery of a failed runtime execution worker.

The server keeps at most one pending prediction and one pending reset. A newer
request replaces an older pending request of the same kind, which receives a
`superseded` error rather than a success acknowledgement. Requests already
executing on the model worker are not cancelled. This bounds pending reset
queries even when a client sends resets without waiting for replies.

### Request Timing

After a successful prediction, `model.last_timing` reports durations in seconds:

- `server_decode_s`: server-side request parsing, validation, and array/image decoding, including JPEG decoding.
- `server_queue_s`: time from completed request decoding until the model worker starts processing the request.
- `server_compute_s`: time spent executing the model.
- `encode_s` and `decode_s`: client-side request encoding and reply decoding, respectively.
- `round_trip_s`: client-observed total duration, including encoding, transport, server work, and reply decoding.

Server debug logs and periodic summaries report decode, queue, and compute times
separately in milliseconds. Summaries are skipped when request, supersession, and
expiration counts have not changed since the previous summary interval.
The request budget still starts at receipt: decoding
does not extend the deadline. The current protocol requires `server_decode_s` in
every prediction reply; update the client and server together.

`request_timeout_s` defaults to 2 seconds and applies to the handshake and each request. Prediction budgets are capped at 60 seconds on both client and server; oversized wire budgets are rejected before entering the worker queue. The client does not retry failed model calls. Errors and timeouts propagate into the selected execution strategy: Sync raises in the control call, Async marks its worker failed and raises `WorkerDiedError` on a later tick, and RTC retries transient inference errors up to its consecutive-error limit.

## Troubleshooting

| Symptom                             | Check                                                                                                                                              |
| ----------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| No matching server                  | Confirm the server process is running with the same `--name`; run `physicalai inference port NAME` on both ends and compare the derived port.      |
| SSH tunnel down                     | Restart `ssh -N -L P:127.0.0.1:P user@inference-host` on the client and keep it open; test the local forwarded port with `nc -vz 127.0.0.1 P`.     |
| Firewall blocks a direct connection | Allow inbound TCP `P` only from trusted client addresses in the host and cloud firewalls; verify with `nc -vz SERVER_IP P`.                        |
| Port already in use                 | The server exits without selecting another port. Choose another `--name` or explicit `--port`, then update the client's `name` or `port` to match. |

For a model mismatch, compare the server's policy name and reported manifest
hash with the client's expected values. These are consistency checks, not server
authentication or artifact-integrity checks.

## Trust Boundary

Inference transport has no Zenoh-level authentication or encryption. It binds to loopback by default; SSH provides encrypted, key-authenticated access, while Tailscale requires tailnet ACLs. Direct LAN binding requires an isolated VLAN and restrictive firewall rules. Do not expose the listener to an untrusted network.
