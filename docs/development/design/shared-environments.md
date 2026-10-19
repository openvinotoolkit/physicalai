# Shared Isolation Infrastructure for Python Integrations

> **Status:** Discussion draft for cross-team review. This document defines requirements, a candidate architecture, and an evaluation plan. It does not establish a final API, package location, or choice of environment tool.
>
> **Working name:** `physicalai.isolation`. Here, isolation means separate Python dependencies and processes, not a security sandbox.

## Abstract

Applications that integrate independently developed machine-learning frameworks can encounter incompatible Python versions, dependencies, and native libraries. Separate environments and child processes address these conflicts, but each integration must otherwise implement environment preparation, source validation, process supervision, and communication independently.

This proposal examines a reusable isolation layer that training engines, inference adapters, and other packages can consume. It separates environment provisioning from framework behavior and from application scheduling. Existing tools should supply dependency resolution and as much environment/process management as their contracts permit; the shared layer should retain only the integration and verification responsibilities that remain necessary.

The intended outcome is a common infrastructure contract, not another training framework or package manager. The proposal also requires every plugin advertising inference to support deployment without `physicalai-train`. GR00T provides a concrete integration case, but the contract is independent of any model or plugin category.

## 1. Context and problem statement

Physical AI Studio provides training/export engines, while Physical AI Runtime loads exported artifacts and integrates them with application workflows. Framework and repository plugins allow upstream implementations to retain their native training and inference behavior.

The GR00T prototype in Studio [PR #23][gr00t-pr] illustrates the resulting infrastructure needs:

- Its native training environment uses the pinned Isaac-GR00T project and lock, Python 3.10, and CUDA dependencies.
- Its OpenVINO enablement applies the Robotics AI Suite patch series to the same upstream revision and uses a separate Python 3.12 environment.
- GR00T's dependency versions differ from the Runtime host's. Its adapter therefore invokes a persistent child process that owns model preprocessing, inference, and postprocessing.
- Environment preparation currently resides in `physicalai.engine`, distributed by `physicalai-train`. Deployment requires that distribution for automatic preparation/verification, or a manually provided interpreter without equivalent verification.
- Engine's finite-job runner and the GR00T adapter's persistent worker maintain related process-management code separately.

The prototype demonstrates selected training, export, inference, parity, and reset/replay paths. It does not establish a general shared API or complete failure, platform, offline, and cache-management guarantees.

Three concerns should be evaluated separately:

| Concern             | Question                                                                    |
| ------------------- | --------------------------------------------------------------------------- |
| Plugin contract     | Which package supplies the framework-specific behavior?                     |
| Isolation mechanism | Where do its dependencies run, and how is their process managed?            |
| Distribution        | Which independently installable package contains the common infrastructure? |

A plugin registry does not isolate conflicting dependencies. A separate distribution does not make two versions coexist in one interpreter. A process/environment tool does not define the model package or application API.

## 2. Objectives and scope

### 2.1 Requirements

| ID  | Requirement                                                                                                                                                   |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| R1  | Run selected framework code in a dependency environment distinct from the host, including a different supported Python version.                               |
| R2  | Reuse one environment description across training, export, and inference entry points where their requirements match.                                         |
| R3  | Reproduce reviewed source and dependencies from immutable inputs, including ordered external or locally maintained patches.                                   |
| R4  | Support finite jobs and persistent processes with bounded startup, request, cancellation, and shutdown behavior.                                              |
| R5  | Allow inference plugins to install, prepare/validate dependencies, load, predict, reset, and close without `physicalai-train` in the host or inference child. |
| R6  | Provide explicit online preparation and offline operation, without mutating exported model packages.                                                          |
| R7  | Permit an external tool or library to replace environment/process implementation without changing framework behavior.                                         |
| R8  | Expose no dependency on Engine, model classes, robot scheduling, Studio APIs, or training-specific events from the shared layer.                              |

### 2.2 Non-goals

The initial scope excludes a dependency solver, a universal framework config schema, arbitrary Python-object RPC, a distributed scheduler, automatic inference retries, and a security sandbox. Container/remote execution, Windows support, and additional accelerator paths require separate validation rather than being inferred from a tool's advertised capabilities.

The layer does not determine model quality, convert datasets, implement export optimizations, or remove a framework's upstream dependencies. It also does not make direct framework imports transparent across interpreters. Native-config serialization and compatibility remain plugin responsibilities.

## 3. System model and responsibility boundaries

The shared layer manages the conditions under which framework code runs. The consuming package owns the meaning of that code and its messages.

```text
Training Engine ────────────┐
Inference adapter ──────────┼──> Shared isolation layer
Other package integration ─┘      ├── sources and patch validation
                                  ├── provider: locked environment preparation
                                  ├── finite or persistent child process
                                  └── bounded communication and cleanup

Framework/model-specific code runs in the child environment.
The shared layer does not import its consumers.
```

### 3.1 Terminology

- **Host:** the application interpreter that loads the engine, adapter, or integration.
- **Child/worker:** a process running framework-specific code in a selected environment. This is not a DataLoader worker or Studio job-queue worker.
- **Environment:** the selected interpreter, dependency set, activation context, and source inputs required by that process.
- **Provider:** the integration with a tool that prepares or validates an environment and supplies launch information.
- **Session:** the lifetime of a managed child and its communication channels; it can serve one job or multiple requests.
- **Enablement:** a plugin-defined capability that can select a different source/dependency environment, such as patched OpenVINO export. It is not a shared-layer operation.

Runtime's `Execution` strategies schedule inference relative to the robot loop; Studio's `Environment` represents a robot/camera setup. The isolation layer addresses neither concept. The working name avoids those overloaded terms, but naming is not a prerequisite for evaluating the contract.

### 3.2 Division of responsibility

| Capability           | Shared layer                                                                       | Consumer/plugin                                                               |
| -------------------- | ---------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| Project inputs       | Materialize bundled or pinned Git source; verify patch layers                      | Select upstream revision, patches, licenses, and lock ownership               |
| Dependencies         | Delegate to the selected provider and authoritative lock                           | Select tested profiles and required framework versions                        |
| Launch               | Resolve executable/command, working directory, and activation variables            | Select entry point and native arguments                                       |
| Process lifecycle    | Startup, log draining, cancellation escalation, failure detection, close           | Decide job boundaries, reset semantics, and recovery policy                   |
| Communication        | Framing, deadlines, request identifiers, compatibility negotiation, generic errors | Define payload schemas and operations such as training progress or prediction |
| Model artifacts      | No model-specific interpretation                                                   | Export/load format, preprocessing, I/O mapping, weights, and asset revisions  |
| Application behavior | None                                                                               | Training results/hooks, robot timing, action queues, and safety policy        |

An environment does not belong to one script: training, export, and inference can reuse it when their dependencies match. Conversely, one plugin can declare multiple environments without claiming all operations work in every environment.

## 4. Capabilities of the proposed subpackage

### 4.1 Reproducible environment preparation

The layer accepts a bundled project/lock or an immutable Git revision. Optional patch layers can be published elsewhere or maintained inside the plugin. Existing source types in PR #23 are `BundledProject`, `GitProject`, `PublishedPatches`, and `BundledPatches`.

Published patches are verified by their recorded digests; bundled patches derive digests from their files. Ordered layers can additionally pin the resulting Git tree. This supports external ownership, a local mirror, locally authored changes, and gradual migration between teams.

Content identity and provenance are separate: moving identical patches into another repository need not change the prepared code, while provenance should still record their origin. Environment compatibility also depends on provider/profile, Python and platform requirements, and relevant project/lock inputs. Different providers or locks are not assumed equivalent merely because the patched source tree matches.

Preparation must be explicit or occur once at an approved loading boundary, never in an import, discovery call, or prediction request. It must not regenerate a lock. A missing or incompatible lock produces a diagnostic rather than a silent re-resolution or tool fallback.

### 4.2 Managed finite jobs and persistent sessions

A finite job runs a native training/export/tool command and returns generic completion information. Its consumer translates output/events into domain results such as Engine's `Run` and `TrainHooks`.

A persistent session starts once, loads a model once, and processes repeated requests. The adapter owns operations such as load, predict, and reset. Shutdown must close channels and reap owned processes within a bound, including when initialization or another cleanup action fails. Process-tree handling matters because native launchers can create further subprocesses.

Provisioning, startup, and per-request deadlines should be distinct. A failed or timed-out request must not be retried automatically: the child can be stateful, and the host may not know whether an action request completed.

### 4.3 Communication and observability

The initial transport can be local pipes with versioned framing. Logs remain separate from control messages. Metadata and numeric buffers have explicit size, dtype, shape, and message-type checks before allocation or decoding; object arrays and pickle are excluded from this protocol.

A small worker-side bridge must run on the child Python version without installing the host's Runtime or training wheel. For example, a Python 3.10 GR00T worker cannot depend on a Runtime distribution requiring Python 3.11. A separately packaged bridge is an option if compatible packaged script resources become insufficient.

The layer should expose preparation identity, provider/interpreter versions, startup/exit status, and phase-specific errors. Training metrics and inference timing remain consumer-defined events rather than mandatory framework-specific fields.

### 4.4 Cache and offline operation

Prepared environments reside outside source checkouts and model artifacts. Concurrent preparation requires a per-entry lock, staged source updates, and atomic readiness metadata. Warm reuse must validate the expected inputs without synchronizing dependencies on every prediction.

Cache cleanup must protect active sessions and retained deployment versions. It must not depend on Engine discovery or assume an environment is unused because training is absent. Inspection and dry-run cleanup should precede destructive operations.

Offline inference requires both a prepared dependency environment and locally available model/tokenizer/processor assets. Preparing a virtual environment alone does not make a model offline-capable. Missing assets must fail without unintended network access in offline mode.

## 5. Illustrative interface and usage

The following API fragments are **proposed interaction shapes, not implemented APIs**. Type names and signatures remain open. They use the working import path `physicalai.isolation` for readability and do not imply that the subpackage must ship in that distribution permanently.

A single lazy environment handle can keep preparation details internal. Consumers should not need to construct both a configuration object and a prepared-result object. Construction is side-effect-free; explicit preparation, a finite run, or session startup performs work.

### 5.1 A declaration shared by multiple consumers

```python
# In an integration package, not in an application-specific Engine implementation.
from physicalai.isolation import Environment, GitProject, Uv

NATIVE = Environment(
    name="gr00t-native",
    source=GitProject(
        repository="https://github.com/NVIDIA/Isaac-GR00T.git",
        revision="3df8b3825d67f755e69141446f4315f281b9b7e6",
    ),
    provider=Uv(python="3.10", extras=()),
)

NATIVE.prepare()  # Optional pre-provisioning; uses the project's existing lock.
```

The plugin can declare a second environment using a patched source for its OpenVINO enablement. Both Engine and adapter import the same declarations, without the adapter importing Engine. A user-managed interpreter is an alternative provider/attachment path with explicit validation and recorded assurance limits, not a fallback that silently skips compatibility checks.

### 5.2 A training or export package

```python
from integration_package.environments import NATIVE

result = NATIVE.run(
    entrypoint=train_worker_script,
    args=native_training_arguments,
    on_event=handle_training_event,
)
# The training integration translates completion/events into its native result API.
```

Only the training integration knows what constitutes a step, a checkpoint, or a training cancellation. An export integration or batch-processing tool can use the same finite-job interface without depending on Engine types.

### 5.3 An inference adapter

```python
from integration_package.environments import NATIVE

with NATIVE.start(
    entrypoint=inference_worker_script,
    startup_timeout=60,
    request_timeout=10,
) as session:
    session.request({"operation": "load", "model_path": model_path})
    for observation in observations:
        outputs = session.request({"operation": "predict", "inputs": observation})
    session.request({"operation": "reset"})
```

The session remains resident across requests. The inference adapter validates its own observation/output schemas and owns reset behavior. The generic layer does not attach robotics semantics to a message named `predict`. Another package can implement different operations over the same lifetime/transport facilities.

End users need not call this infrastructure directly. Applications continue to use their framework Engine and the inference API, for example `with InferenceModel(package) as model:`. The native configuration object remains owned by the framework/plugin; this proposal does not introduce another policy configuration schema.

### 5.4 Use by other packages

A dataset-conversion tool, evaluation package, or image-analysis application can use the same finite-job or session interface. It supplies an environment, a reviewed entry point, and its input/output contract; it does not implement or import `Engine` or `InferenceModel`. For example, a conversion package can run a converter with an older Python/native-library dependency set and exchange file paths, while an analysis adapter retains a loaded detector across numeric-array requests.

These consumers do not require the training/inference event vocabulary. Their integration depends only on the shared layer. Source-level independence does not imply a minimal installation, however: while the implementation ships inside `physicalai`, external consumers still install that parent distribution. Independent packaging or a lightweight InferenceKit base is needed if that dependency footprint is unacceptable.

## 6. Plugin packaging and deployment independence

Every plugin advertising inference—framework, policy, or repository—must install and perform its supported inference lifecycle without `physicalai-train` in either the host or inference child. Training-only integrations are outside this requirement. A training-checkpoint compatibility loader does not, by itself, establish a deployment path.

A self-contained plugin owns its adapters, optional Engine integration, entry points, source/lock declarations, patches or immutable references, tests, and compatibility documentation. It can depend on shared infrastructure and upstream libraries. Self-contained does not mean vendoring Python, frameworks, licensed weights, or device drivers into one wheel.

Training and inference entry-point registration must remain independent. Deployment imports and manifest components must not resolve to Studio-only modules. A single plugin distribution can expose deployment dependencies by default and training integration through an optional `train` extra.

The existing Studio convenience extra in PR #23 illustrates the installation contract:

```toml
[project.optional-dependencies]
gr00t = ["physicalai-gr00t-policy-plugin[train]"]
```

The intended published-package usage is:

```bash
# Training application plus selected integration.
uv pip install "physicalai-train[gr00t]"

# Deployment application: plugin plus declared Runtime/shared dependencies.
uv pip install physicalai-gr00t-policy-plugin
```

These examples describe the target package contract, not a claim that the prototype packages are published or that the current GR00T deployment path already meets it. Repository development uses workspace/local source mappings until release. Extras are curated installation shortcuts, not discovery: independently installed third-party plugins can register entry points without being listed by Studio.

The base shared-layer import should remain small and avoid mandatory model libraries. If shipped inside today's `physicalai` wheel, however, installing it still installs that wheel's base dependencies. A source-level boundary removes coupling to training; optional dependencies or an independent distribution are needed to reduce the installation footprint further.

## 7. Existing tools and alternative approaches

The maintenance question is whether an existing tool can provide the required guarantees with less owned code. These candidates address different layers; they are not interchangeable substitutes for the complete integration contract.

| Candidate                       | Relevant capability                                                                            | Integration/evaluation considerations                                                                                                                                                                                                                                     |
| ------------------------------- | ---------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [uv][uv-docs]                   | Python management, locked project synchronization, virtual environments                        | Current baseline and native lock format for GR00T. It does not define a persistent inference session or application protocol. `--frozen` skips lock freshness checks; project/lock agreement needs explicit validation.                                                   |
| [Pixi][pixi-docs]               | Multiple environments, Conda/PyPI dependencies, activation, tasks                              | Useful where native dependencies benefit from Conda. A Pixi project has its own lock; adopting it for GR00T requires an explicitly owned lock and compatibility test, not an assumption that it consumes `uv.lock`. Incompatible frameworks must not share a solve group. |
| [Isolate][isolate]              | Environment and cross-environment function execution                                           | Candidate for replacing a larger part of the implementation. Establish support for existing locks, persistent model ownership, Python versions, serialization, lifecycle, maintenance, and auditable source/license before adoption.                                      |
| [execnet][execnet]              | Persistent cross-interpreter gateways, channels, and builtin-type serialization without pickle | Candidate process/communication implementation. Environment installation remains separate; numeric-buffer transfer, bounded failure, and model lifecycle need assessment.                                                                                                 |
| [Ray runtime environments][ray] | Tasks/actors, runtime dependencies, distributed scheduling                                     | Relevant when a deployment already uses Ray or needs distributed placement. Evaluate operational cost before introducing it for a local child process.                                                                                                                    |
| OCI containers                  | Native/OS isolation and image-based distribution                                               | Appropriate where deployment already standardizes on images. Device access, image distribution, digest pinning, startup, and process lifecycle remain explicit concerns.                                                                                                  |

### 7.1 Implications of the Pixi demonstration

The [Pixi demonstration][pixi-demo] shows incompatible Python/PyTorch/NumPy environments selected by tasks in one project. The pipeline exchanges image/JSON files and starts a new detector process for a task. This is useful evidence for environment selection, but not evidence for low-latency persistent inference.

A comparable inference experiment should launch the same worker through Pixi once, retain its loaded model, and use the same request transport and observations as the uv baseline. File handoff remains suitable for finite jobs; repeated model loading and per-request disk serialization should not be confused with environment-provider overhead.

### 7.2 Assessment status and selection principle

The tool comparison is documentary, not a benchmark. The existing prototype exercised uv with selected GR00T/LeRobot paths. Pixi, Isolate, execnet, Ray, and containers have not been validated against this shared contract. The inspected Pixi demonstration had no declared license; code reuse requires clarification. Isolate's published package description was available, but its linked source repository was unavailable during the assessment; source and licensing review remains necessary.

An external library that satisfies preparation and lifecycle requirements should replace those implementation responsibilities. A thin provider wrapper is justified only for residual needs such as plugin-owned source references, provenance, compatibility policy, and integration with consumers. The design should not retain a custom process manager simply to preserve the prototype.

One authoritative project/lock and an explicit provider should be selected per environment. Automatic lock translation, implicit provider fallback, and a general provider marketplace are not initial requirements.

## 8. Evaluation methodology

Evaluation should separate the environment tool, process implementation, and application protocol. Compare alternatives using the same pinned source/model assets and worker entry point where feasible. Different locks require separately documented compatibility results.

| Dimension               | Method and acceptance evidence                                                                                                                                |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Dependency independence | Use a clean host and a child with incompatible supported dependencies; verify the host never imports the child framework.                                     |
| Deployment independence | Install the plugin wheel without training extras or a Studio checkout; verify `physicalai-train` is absent in host and inference child.                       |
| Reproducibility         | Check lock/source/profile identity, published/local patch equivalence, incompatible locks, changed trees, and recovery after interrupted preparation.         |
| Lifecycle               | Test startup failure, early exit, hung requests, cancellation, parent shutdown, repeated close, and cleanup failure; verify no owned processes/channels leak. |
| Persistent state        | Verify one model load for repeated predictions, correct reset behavior, and no implicit retries.                                                              |
| Performance             | Report cold provisioning, warm startup/model load, steady-state latency, transfer overhead, memory, and disk usage separately.                                |
| Offline operation       | Prepare dependencies and assets online; disable network and exercise the supported load/predict/reset/close flow.                                             |
| Artifact integrity      | Load from read-only packages and verify inference produces no changes inside them.                                                                            |
| Portability             | Run the actual host/child Python and OS/device matrix; do not extrapolate from provider documentation.                                                        |

Ordinary CI should use CPU-only fake workers, bounded failure tests, and packaged fixtures without external downloads. Gated tests should validate real framework environments and models on the claimed hardware/backend profiles.

For every inference plugin, the clean-install contract should cover discovery, environment preparation or attachment, load, preprocess, predict, reset, and close. Upstream dependencies used for both training and inference are permitted; Studio training must not be an undeclared deployment requirement.

GR00T remains a reference application: train a short run, export native and OpenVINO packages, compare identical observations/noise, and check reset/replay. Also test the released LIBERO checkpoint against recorded actions: agreement between backends alone cannot detect a shared camera/state/action mapping error. A short training run validates the workflow, not policy quality or closed-loop robot success.

## 9. Distribution and migration options

The working import path is `physicalai.isolation`, not `physicalai.utils.isolation`: the layer has explicit consumers, contracts, and tests rather than being a collection of unrelated helpers. Physical source placement can remain open while evaluating the interface.

| Placement                                      | Advantage                                                                  | Constraint                                                                                     |
| ---------------------------------------------- | -------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| Subpackage in the Runtime wheel                | Small initial migration; Studio already depends on Runtime                 | Retains the full Runtime installation footprint until optional-dependency or packaging changes |
| Independent library in the Runtime monorepo    | Independent dependencies and consumption by unrelated packages             | Requires a separate import root, compatibility policy, and release unit                        |
| Part of a future lightweight InferenceKit base | Reuses an inference-independent support layer across Studio and deployment | Acceptable only if training consumers do not acquire unnecessary inference dependencies        |

The [modular-packages design][packaging-design] supports incremental extraction without an immediate repository split. Do not distribute one public import path across independent wheels or maintain duplicate implementations. Use temporary compatibility aliases for source migration, and persist logical identities/data in artifacts rather than cache paths or serialized infrastructure classes.

A staged implementation can proceed as follows:

1. **Assess tools and freeze the minimum contract.** Compare the uv baseline with Pixi and one execution library. Decide what can be delegated before expanding owned infrastructure.
2. **Extract source/preparation services.** Move the reusable parts of Engine's `_sources.py`, `_environment.py`, and environment records behind the shared boundary. Remove Engine discovery and accelerator-selection policy from that boundary. Preserve cache/package compatibility or document a versioned migration.
3. **Migrate both consumers.** Release the shared functionality before requiring it from Studio/plugins. Keep Engine results/hooks in Studio. Remove GR00T's deployment requirement for training-based provisioning/verification and enforce the universal plugin contract.
4. **Consolidate lifecycle where demonstrated.** Reuse Engine runner and GR00T adapter supervision through the selected process implementation. Preserve finite and persistent modes, numeric transport, and the merged [Runtime close lifecycle][close-pr].
5. **Validate before expanding scope.** Run clean deployment tests, GR00T/Cosmos and LeRobot regressions, offline/failure tests, and provider comparisons. Add another provider only when it provides a measured capability or maintenance benefit.
6. **Separate distributions when justified.** Move the same implementation into InferenceKit or a smaller shared library when independent installation is required. A standalone InferenceKit deployment must then be tested with neither `physicalai` nor `physicalai-train` installed.

The current environment class names do not need to change during extraction. API naming, resolved-state representation, and dependency inversion can be reviewed independently rather than combined into a large behavior-changing rewrite.

## 10. Risks and unresolved design choices

- **Scope growth:** source materialization, provisioning, and RPC can become a general platform. Keep the local Python use cases and consumer contracts explicit.
- **Reproducibility limits:** a lock/tree identity does not alone pin system drivers, native ABI behavior, model assets, or every build-tool effect. Record and validate the relevant deployment assumptions.
- **Cache safety:** removing undeclared environments can break retained artifacts or running children. Cleanup needs ownership/reference policy, not just installed-plugin discovery.
- **Process recovery:** a restarted child loses model/episode state. The application must choose whether and how recovery is safe.
- **Native Python API expectations:** accepting a native config as data is different from importing an entire framework into the host. Transparent arbitrary-object execution is outside this proposal.
- **Security:** installing/building projects and launching their code grants operator-level privileges. A separate virtual environment does not sandbox files, credentials, devices, or network access. The existing [Runtime security model](../../getting-started/security.md) and [contributor requirements](../security.md) apply. Model metadata must not introduce peer-selected executable commands or provisioning code outside the reviewed plugin boundary.
- **External-tool adoption:** a provider is useful only if its lifecycle, lock behavior, licensing, and maintenance meet the requirements. Tool support claims are not integration evidence.

## 11. Questions for cross-team review

1. Which consumers need only environment preparation, and which also need a persistent process and common transport?
2. Can an existing library satisfy those requirements with less owned code, without replacing upstream locks or introducing unsafe object serialization?
3. Which Python/native dependency and platform profiles must be supported initially, including offline deployments?
4. What source/lock/model metadata must an integration publish to establish reproducibility and compatibility?
5. Is the proposed split between deployment dependencies and optional training integration sufficient for framework, policy, and repository plugins?
6. Which startup/request/cleanup guarantees and performance budgets should be release gates?
7. Should the shared layer initially ship inside Runtime, inside a lightweight InferenceKit base, or as an independent monorepo library?

## References

- [Runtime modular-packages proposal][packaging-design]
- [Native Framework Engines and Policy Plugins][engine-design]
- [Studio GR00T prototype and validation][gr00t-pr]
- [Runtime deterministic cleanup lifecycle][close-pr]
- [uv project lock and synchronization behavior][uv-docs]
- [Pixi multi-environment documentation][pixi-docs]
- [Pixi demonstration][pixi-demo]
- [Isolate package description][isolate]
- [execnet documentation][execnet]
- [Ray runtime environments][ray]

[packaging-design]: https://github.com/openvinotoolkit/physicalai/blob/docs/design/docs/design/packaging/modular-packages-in-one-repo.md
[engine-design]: https://github.com/open-edge-platform/physical-ai-studio/blob/feat/engine-based-plugin-approach/library/docs/explanation/policy/engine-and-frameworks.md
[gr00t-pr]: https://github.com/samet-akcay/physical-ai-studio/pull/23
[close-pr]: https://github.com/openvinotoolkit/physicalai/pull/372
[uv-docs]: https://docs.astral.sh/uv/concepts/projects/sync/
[pixi-docs]: https://pixi.sh/latest/tutorials/multi_environment/
[pixi-demo]: https://github.com/leoll2/pixi-demo
[isolate]: https://pypi.org/project/isolate/
[execnet]: https://execnet.readthedocs.io/
[ray]: https://docs.ray.io/en/latest/ray-core/handling-dependencies.html
