# OpenVINO Validation — Proposal

**Status:** Draft proposal · **Owner:** physicalai team · **Date:** 2026-06-29

Proposal for integrating physicalai tests into OpenVINO's automated CI/CD. It
defines the validation contract, who owns what, the test tiers, and the target
platforms, frequency, and reporting for the OV pipeline. Piotr Wolnowski owns
the pipeline implementation and contributes test cases via the physicalai repo.

## Coverage

Three areas, mapped to the tiers below:

| Area                            | Question                                        | Tier          |
| ------------------------------- | ----------------------------------------------- | ------------- |
| **Correctness + compatibility** | does physicalai load + run on this OV?          | smoke         |
| **Accuracy**                    | same numbers on target HW (per device/version)? | golden-action |
| **Functional**                  | does it still do the task end-to-end?           | gym E2E       |
| **Performance**                 | latency / throughput on target HW?              | benchmark     |

## Why

An OpenVINO upgrade broke physicalai inference — hence the current
`openvino==2026.1` pin in [`pyproject.toml`](../../pyproject.toml). We need an
automated signal that an OpenVINO change still works with physicalai _before_ we
adopt it.

## The Contract

"OpenVINO works with physicalai" means a real exported policy loads and runs
through the physicalai OpenVINO code paths. Three stages must pass:

| Stage         | Code path                 | What it proves                                                   |
| ------------- | ------------------------- | ---------------------------------------------------------------- |
| **load**      | `OpenVINOAdapter.load`    | `read_model` + `compile_model` succeed                           |
| **tokenizer** | `OVTokenizer`             | `openvino_tokenizers` extension loads + matches core OpenVINO    |
| **predict**   | `OpenVINOAdapter.predict` | inference runs, outputs have expected shape/dtype and are finite |

The tokenizer stage matters because `openvino` and `openvino_tokenizers` are
versioned **independently** and must stay a matched pair — a mismatch is what
broke us, and it only affects models that carry a tokenizer.

## Who Owns What

| Concern                                                       | Owner               |
| ------------------------------------------------------------- | ------------------- |
| Test **logic + code location** (contract, stages, assertions) | **physicalai** repo |
| Pass/fail criteria                                            | **physicalai**      |
| PR merge gate                                                 | **physicalai** CI   |
| Early-warning on OV bump PRs (Renovate-triggered)             | **physicalai** CI   |
| Running the same test in pre-release matrix                   | **OpenVINO**        |
| Weekly + per-RC runs on target hardware                       | **OpenVINO** CI     |
| Orchestration (workflow YAMLs, runners) + reporting           | **OpenVINO** repo   |
| Pre-release wheels + breakage notification                    | **OpenVINO**        |

**Decision:** split by layer.

- **Test logic lives in physicalai** — this is the only way to guarantee its
  quality and relevancy. Only physicalai knows its real OpenVINO usage, and the
  test sits next to the code it protects, so it changes in the same PR as that
  code. The OV team does not author test logic in-house.
- **Orchestration and reporting live in OpenVINO** — the workflow YAMLs, runners,
  matrix, and dashboards are OV-side tooling. OV consumes the physicalai test by
  pinned tag (`git clone physicalai@<tag>` → `pytest -m ov_smoke`).

Test-case contributions from the OV team land as PRs into the
physicalai repo and are reviewed by physicalai, so logic stays single-sourced
and quality-controlled while OV helps build it.

The test is self-contained (pytest + `ov_smoke` marker + model discovery via env
var), so OV can invoke it without pulling in physicalai internals.

Same test, multiple runners: physicalai's PR gate and Renovate early-warning
catch _our_ changes and released OV bumps; OpenVINO's pre-release matrix catches
_their_ changes before release.

**What OV runs:** smoke runs by pinned tag, so OV adds it to their pre-release
matrix. Golden-action runs on each device, comparing the OpenVINO output to a
stored PyTorch reference. Gym E2E uses the in-house LIBERO gyms on OV's GPU
runners. PAI maintains all tiers; OV provides the hardware.

## Smoke Test

A pytest integration test (marker `ov_smoke`) that runs the three contract
stages against discovered model exports and records the `openvino` /
`openvino_tokenizers` versions in each result.

Coverage needs **two models** to cover the contract:

| Model                                     | Covers                     |
| ----------------------------------------- | -------------------------- |
| **ACT** (no tokenizer)                    | load + predict             |
| **a tokenizer-bearing model** (e.g. pi05) | load + tokenizer + predict |

Verified 2026-06-26: under a core/tokenizer mismatch, ACT passes and the
tokenizer model fails at the tokenizer stage — so this pair detects the exact
break that caused the pin.

**Devices:** run the smoke across the device spectrum — CPU on hosted runners,
GPU on OV runners. CPU covers the version-pair contract; GPU catches
device-plugin regressions.

## Early Warning

physicalai CI runs `ov_smoke` when a Renovate PR bumps `openvino*`, against the
PR's resolved versions — event-driven, alert-only, fires exactly when a new
version appears. OV's weekly-vs-nightly and per-RC matrix (below) covers
pre-release wheels, so physicalai does not duplicate a weekly run.

## Golden-Action

A regression test that pins the OpenVINO output to a stored reference. It answers
a different question than smoke: smoke proves "it ran", golden proves it produced
the **right numbers**.

**Approach:**

1. **Export deterministically.** Build the test model with the PyTorch
   deterministic-denoising toggle on, so the policy uses fixed (non-random) noise.
   This removes the in-graph sampling that otherwise makes every inference
   different (see below).
2. **Record one reference.** Run the source **PyTorch** model (e.g. on CPU) on a
   fixed observation and store the action chunk as the ground-truth reference —
   recorded once, not per device.
3. **Check every run.** Run the OpenVINO export on the target device for the same
   observation and compare against the reference with an **L2 tolerance**.

Exact equality only holds for a device compared against itself; across devices,
kernel and precision differences make the numbers drift, so the L2 tolerance
absorbs that drift while still catching a real regression. One PyTorch reference
covers the whole device matrix.

**Why a deterministic export is needed (verified 2026-06-30).** Without it the
output is non-deterministic, and the cause is inside the exported graph — not in
physicalai's Python code, and not in any model input. The pi05 IR draws its
denoising noise from a single `RandomUniform` op exported with
`global_seed=0, op_seed=0`, which the OpenVINO spec defines as a non-deterministic
sequence; the stock model varies ~0.3 abs across identical calls. We confirmed the
noise can also be pinned at the IR level — seeding that op makes the OpenVINO
output reproducible bit-for-bit — which proves determinism is achievable, but
exporting deterministically from PyTorch is the cleaner source-level fix and also
makes the PyTorch reference itself stable.

## Performance

Latency / throughput on target hardware, built on the existing
`InferenceLatencyBenchmark`. Reports median/p99 per-iteration time and FPS per
model per device, tracked as a trend. Threshold-based and alert-only — hosted
variance makes absolute pass/fail unreliable, so flag large regressions, never
gate. Runs on the OV GPU runners across the platform matrix.

## End-to-End (gym)

Beyond smoke (loads + runs) and golden-action (same numbers), a gym rollout
checks the policy still **does the task** end-to-end: closed-loop
`reset → predict_action_chunk → step`, exercising the full pre/post pipeline and
the runtime action queue.

- **Gyms exist in-house:** LIBERO for SmolVLA and pi05 (a newer gym is in
  progress for later).
- **Reduced mode:** a small, fixed set of seeded episodes to bound runtime — a
  pass/fail signal, not a full acceptance benchmark.
- **Where:** OV GPU runners, using the in-house LIBERO gyms. Reduced/seeded,
  per-device reference, threshold-based, alert-only. Never on the PR gate — a
  distinct tier from smoke.

## Target Platforms

All targets are GPU — the pipeline runs on OV-provided Intel hardware:

| Platform | Type                        |
| -------- | --------------------------- |
| PTL iGPU | Panther Lake integrated GPU |
| B580     | Arc Battlemage discrete     |
| B60      | Arc Battlemage discrete     |
| B70      | Arc Battlemage discrete     |

CPU smoke stays on hosted runners as the portable baseline; GPU/accuracy/perf
tiers run across this matrix.

## Frequency

| Trigger                      | Scope                                | Gates?       |
| ---------------------------- | ------------------------------------ | ------------ |
| OV bump PR / PAI PR          | smoke (CPU)                          | yes          |
| **Weekly** vs OV **nightly** | smoke + golden + perf, all platforms | alert        |
| **Each RC** (pre-release)    | full suite, all platforms            | release gate |

## Reporting

Per-run record of OV / `openvino_tokenizers` versions, model, platform, tier,
and pass/fail + metrics. Failures notify the owner; perf/accuracy tracked as a
trend across versions. Dashboards and notification wiring are OV-side.

## Next

- Build the `ov_smoke` test (three contract stages, model discovery, version logging).
- Lock `openvino` + `openvino_tokenizers` as a matched pair in
  [`pyproject.toml`](../../pyproject.toml) — they are versioned independently
  and must be binary-compatible; constraining the tokenizer to the matching
  minor stops a resolver from installing the mismatch that broke us.
- Add a small tokenizer-bearing model the gate can run — the tokenizer break
  only shows on a model with an OV tokenizer; pi05 is 6.25 GB (too large for
  hosted CI) and ACT has none, so the gate needs a small tokenizer model.
- Wire `ov_smoke` into [`library.yml`](../../.github/workflows/library.yml) as a
  PR gate, plus the Renovate-triggered early-warning run.
- Add the golden-action tier — export the test model with deterministic denoising,
  store a PyTorch reference, and L2-compare the OpenVINO output per device.
- Provision OV GPU runners (PTL iGPU, B580, B60, B70) for GPU smoke, perf, and gym E2E.
- Add the performance tier across the platform matrix.
- Review with Piotr Wolnowski; agree pipeline ownership + RC release gate.
