# Add Runtime Callbacks

Use callbacks when you need product-specific behavior around the runtime loop.

The following example records observations and actions.

```python
class RecordingCallback:
    def on_action_ready(self, *, action, step):
        recorder.write_policy_action(step, action)
        return action

    def on_action_sent(self, *, action, step):
        recorder.write_sent_action(step, action)
```

Attach the callback when you construct the runtime.

```python
runtime = RobotRuntime(
    robot=robot,
    action_source=PolicySource(model=model, execution=execution),
    fps=30,
    callbacks=[RecordingCallback()],
)
```

The same callback instances receive every `run()` performed before the runtime is disconnected. Use the `start` and `shutdown` lifecycle events to separate per-run state. `RobotRuntime.disconnect()` or context exit closes callback resources exactly once and permanently disposes that runtime.

## Monitor the Runtime in Rerun

Install the optional visualization dependencies:

```bash
uv sync --extra observer-rerun
```

Add `RerunCallback(mode="spawn")` to your runtime's `callbacks` list. The **Monitoring** sidebar contains separate Inference Latency, System, and GPU sections, in that order. System and GPU collection are scheduled at most once per second; inference statistics update with each completed inference. CPU percentages use the total capacity of all host logical cores. The adjacent **CPU %** and **GPU %** tabs plot utilization history using the same samples: host/process CPU and one GPU series per vendor/device ID. Unsupported GPU counters do not produce numeric samples, so missing data is never logged as zero. The charts follow the recording's selected wall-clock or step timeline and do not issue additional GPU queries.

GPU monitoring uses read-only vendor commands found on `PATH`:

- **NVIDIA:** `nvidia-smi` (installed with the NVIDIA driver) reports utilization, used/total memory, and temperature.
- **Intel:** `xpu-smi` ([Intel XPU Manager](https://github.com/intel/xpumanager)) reports utilization, used memory, and temperature.

The sidebar includes all GPUs reported by each installed tool. Intel support depends on the GPU, driver, operating system, and tool version; not every integrated GPU exposes these counters. Intel memory can include shared system memory, so it is not labeled dedicated VRAM. Unsupported metrics show **N/A** rather than zero. Missing tools, drivers, or permissions produce an availability message and do not prevent runtime use. The callback never requests `sudo` or changes GPU settings.

GPU commands run on a single background worker, with a three-second timeout per command. A slow query cannot block the control thread, and new samples are not queued while a previous query is pending. Closing the runtime cancels queued queries without waiting for an in-flight command. Vendor tools still consume resources; measure monitoring overhead on the target hardware.

As a general rule, keep workflow-specific logic in callbacks unless the same behavior becomes a reusable runtime primitive.
