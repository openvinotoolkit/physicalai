# Security Model

The runtime treats several categories of input as trusted by design and does not sandbox them further. If you
load a config, manifest, or exported policy you did not author or fully review, you are running that content
with the same privileges as the `physicalai` process itself. The sections below describe each trust boundary.

## Configs and manifests can execute arbitrary code

YAML/JSON configs passed via `--config` flag and an exported policy's `manifest.json` both use `class_path` values to
dynamically import and construct Python objects - robots, cameras, preprocessors, postprocessors, action
sources, callbacks. Currently, nothing restricts which class a `class_path` may name.

Only load configs, exported policies, and manifests from sources you trust. See also the `class_path` note in
[Config Schema Reference](../reference/config-schema.md#security).

`--config <path>` is an explicit path selector: it may name an absolute path or a file outside the current
directory. This is intended for a trusted operator and does not confine the file beneath a project directory.
Automation must not populate it from untrusted input unless the automation first applies an allowlist or its
own base-directory containment policy.

## Use only trusted, reviewed policies

An exported policy package (`manifest.json` plus artifacts) runs with the same privileges as the
`physicalai` process.

**Loading via `InferenceModel.from_pretrained()`:** pin `revision` to the commit SHA of a version you have
reviewed and trust, rather than a mutable branch or tag, so the content you reviewed is exactly what gets
loaded on every run. This is currently an operator-controlled safeguard: policy loading accepts an omitted
revision, branch, tag, or commit and does not enforce commit-SHA pinning.

**Loading via `export_dir`:** `physicalai run` and direct `InferenceModel(export_dir=...)` construction both
load whatever package is already in that local directory. Only place a reviewed, trusted export there.
Treat populating that directory (downloading, copying, extracting) as the point where you decide to trust
its contents.

## Remote robot sharing has no built-in security controls

The `SharedRobot` network transport uses Zenoh `/action`, `/state`, and `/metadata` traffic to share one robot
connection across processes. It has no application-level authentication, access control, or encryption of
its own. Local-only mode is the default.

If you enable `allow_remote=True` (`--allow_remote` on `physicalai robot serve`/`discover`), use it only on
an isolated, firewalled robot-cell network (VLAN/firewall) or with Zenoh ACL/TLS configured yourself — the
same requirement documented in [CLI Reference](../reference/cli.md#physicalai-robot-serve). Without one of
those, anyone who can reach that network can observe robot state and, if nothing else restricts it, send
actions to the robot.

`SharedRobot` checks metadata protocol version, dimensions, and joint-name consistency before attaching.
These checks detect malformed or incompatible metadata; they do not authenticate the publisher or prove its
identity.

## Action validation currently varies by robot driver

Current drivers do not provide a uniform action-validation contract. Depending on the integration, a driver
may validate shape and clip targets locally, or delegate some checks to a composed or external driver. Do not
assume that transport decoding or an adapter's shape check validates numeric type, finite values, device
limits, timing, or all-or-nothing writes. Review and test each driver's guarantees before using it with
hardware.

## SharedCamera assumes trusted local processes

`SharedCamera` uses iceoryx2 shared memory to distribute camera frames between local processes. It is distinct
from the Zenoh `SharedRobot` transport and does not become remotely reachable through `allow_remote`.
Treat local processes that can discover or access its shared-memory resources as trusted. Its namespace
permissions, publisher identity, frame validation, and cleanup behavior have not received a dedicated
security assessment; this statement is an explicit coverage limit, not a finding that the transport is secure
or vulnerable.

## Runtime callbacks run with full trust

Callbacks registered with `RobotRuntime` (see
[Add Runtime Callbacks](../how-to/runtime/add-runtime-callbacks.md)) can inspect and modify the
action sent to the robot on every control tick. Currently, the runtime does not validate a callback's output before
sending it to hardware.

Only register callbacks you wrote or have reviewed, especially any callback that can transform the outgoing
action.

## Installed plugins load from the active Python environment

`physicalai <subcommand>` discovers third-party subcommands from the `physicalai.cli.subcommands` Python
entry-point group. Physical AI Studio separately discovers catalog plugins from
`physicalai.studio.catalog_plugins`. There is no allowlist of installed packages that may register either
entry point. The Studio catalog interface is a plugin-discovery surface; it does not itself imply dynamic
`class_path` loading.

Treat installing a package into either host environment as granting its registered entry-point code the
ability to run when that host loads it. Apply the same security review you would to any other dependency.
