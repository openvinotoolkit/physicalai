# Design: Robot Integration Packaging

**Status:** Proposed · **Decision target:** 2026-10-13 · **Author:** Samet Akcay

## Summary

Every robot family ships as its own plugin package, including SO-101 and WidowX AI, which are built into
`physicalai` today. `physicalai` keeps the runtime, the Robot protocol and the Plugin SDK (the contract plugins use
to add robots to Studio), and contains no robot drivers.

- **Users** still install robots through `physicalai`: `pip install "physicalai[b601-rs]"` installs the runtime,
  the reBot plugin and its vendor SDK. Existing class paths, configs and Studio projects keep working.
- **Robot fixes** do not need a new `physicalai`. Users get a fix by upgrading the plugin alone.
- **Studio** lists all first-party plugins as available but installs only the plugin a user chooses. Its Plugins
  page also offers LeRobot, MuJoCo and third-party plugins, and Studio no longer contains built-in robot code.
- **Maintainers** release everything from one release PR. Merging it checks, tags and publishes every changed
  package in dependency order, so nothing is released by hand or out of order.

The proposal depends on a release process that orders and checks releases automatically, described below, and on
keeping the API that plugins use backward compatible.

---

## Why Change

### The Boundary Is Accidental

SO-101 ([#3](https://github.com/openvinotoolkit/physicalai/pull/3), March) and WidowX AI
([#107](https://github.com/openvinotoolkit/physicalai/pull/107), May) were built into `physicalai` before the plugin
system existed. reBot, Star Arm and bimanual SO-101
([#263](https://github.com/openvinotoolkit/physicalai/pull/263)) and the LeRobot bridge
([#265](https://github.com/openvinotoolkit/physicalai/pull/265)) arrived later as packages. As a result:

- `BimanualWidowXAI` is part of `physicalai`, while `BimanualSO101` is a plugin that wraps the `SO101` driver
  inside `physicalai`.
- The SO-101 and WidowX Robot Types (the robots a Studio user can add to a project) live in Studio, while every
  other robot's Robot Types live in this repo ([#275](https://github.com/openvinotoolkit/physicalai/issues/275)).
- Nobody can say where the next robot belongs. The contributor skill still adds new robots inside
  `physicalai.robot`.

### Releases Need Manual Ordering

This repo releases seven packages, each from its own release PR. Between 2026-09-14 and 2026-10-06 it merged 11
release PRs, one of them for `physicalai`. Two went wrong:

- `physicalai-studio-plugin` 0.2.0 ([#268](https://github.com/openvinotoolkit/physicalai/pull/268)) failed to
  publish. The release workflow was fixed ([#271](https://github.com/openvinotoolkit/physicalai/pull/271)) and the
  release redone ([#272](https://github.com/openvinotoolkit/physicalai/pull/272)).
- `physicalai-stararm-plugin` 0.4.0 ([#326](https://github.com/openvinotoolkit/physicalai/pull/326)) was merged
  before `physicalai-studio-plugin` 0.3.0 existed. The smoke test failed with
  `No matching distribution found for physicalai-studio-plugin>=0.3.0`, and the release was reverted
  ([#339](https://github.com/openvinotoolkit/physicalai/pull/339)) and redone after the SDK release
  ([#333](https://github.com/openvinotoolkit/physicalai/pull/333),
  [#340](https://github.com/openvinotoolkit/physicalai/pull/340)).

Zero-pose calibration needed three releases in sequence, each waiting for the previous one on PyPI: the Plugin SDK
([#323](https://github.com/openvinotoolkit/physicalai/pull/323)), then reBot and Star Arm
([#325](https://github.com/openvinotoolkit/physicalai/pull/325)), then Studio's Curated Plugin List
([studio#1250](https://github.com/open-edge-platform/physical-ai-studio/pull/1250)).

### Version Requirements Are Never Tested

- The hardware plugins declare `physicalai>=0.1.1` but import `physicalai.config`, which 0.1.1 does not have.
- reBot and Star Arm import `physicalai.runtime._callback_bus._CallbackBus`, and the MuJoCo plugin imports
  `physicalai.robot.transport._lock`. Private modules can change without any version signal.
- Inside the uv workspace, packages resolve to each other's source and uv ignores the version requirements between
  them: a plugin requiring `physicalai>=5.0` locks without error against a 0.4 workspace. Wrong requirements only
  show up after publishing.

### Studio Environments Drift

- Studio locks `physicalai-studio-plugin` 0.2.0, but its Curated Plugin List (the reviewed plugins Studio can
  install) installs `physicalai-rebot-b601-plugin>=0.8.0`, which requires `physicalai-studio-plugin>=0.3.0`.
  Installing reBot from the Plugins page upgrades the SDK inside Studio's environment to a version Studio was never
  tested with.
- The Plugins page cannot update an installed plugin, and replacing a Docker container removes plugins installed
  from the UI.
- `physicalai` releases are rare (0.1.1 on 2026-06-02, 0.2.0 on 2026-09-21). Studio pins `physicalai` to git
  commits in between, and exempts each first-party package from its 7-day dependency cooldown.

### Robots Change on Their Own Schedule

- Since the first plugins moved into this repo on 2026-09-14, ten commits have changed them, not counting the moves
  and a release-workflow fix. Nine touched a single package, and none touched `src/`.
- reBot has had 11 releases on PyPI since 2026-06-22. `physicalai` has had one since 2026-06-02.

Robot code changes more often than the runtime, and almost always on its own.

---

## Decision

> `physicalai` contains no robot drivers. Every robot family, first-party or third-party, ships as its own plugin
> package. First-party plugins live in `packages/` and are released from the same release PR as `physicalai`.

### Packages

| Robot integration            | Today                                                                | Proposed                                                                                                                                            |
| ---------------------------- | -------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| SO-101                       | Driver in `physicalai`, Robot Types in Studio                        | `physicalai-so101-plugin`                                                                                                                           |
| Bimanual SO-101              | `physicalai-bimanual-so101-plugin`, wraps the driver in `physicalai` | Unchanged, wraps the driver in `physicalai-so101-plugin`                                                                                            |
| WidowX AI (single, bimanual) | Drivers in `physicalai`, Robot Types in Studio                       | `physicalai-trossen-plugin`                                                                                                                         |
| reBot B601 (DM, RS)          | `physicalai-rebot-b601-plugin`                                       | Unchanged                                                                                                                                           |
| Star Arm 102 (LD, HD, FL)    | `physicalai-stararm-plugin`                                          | Unchanged                                                                                                                                           |
| LeRobot bridge               | `physicalai-lerobot-plugin`                                          | Unchanged                                                                                                                                           |
| MuJoCo simulation            | `physicalai-mujoco-plugin`                                           | Unchanged                                                                                                                                           |
| Plugin SDK                   | `physicalai-studio-plugin`                                           | Moves into `physicalai` ([#269](https://github.com/openvinotoolkit/physicalai/issues/269)); the old package gets a final release that re-exports it |

Each plugin carries its drivers, its Robot Types and its URDF and mesh files. `physicalai` keeps the runtime, the
Robot protocol, the Plugin SDK and the camera backends.

### Names Do Not Change

- `physicalai.robot.SO101`, `physicalai.robot.WidowXAI` and `physicalai.robot.BimanualWidowXAI` keep working.
  `physicalai.robot` imports them lazily from the plugins, the way it lazily imports vendor SDKs today, and a
  missing plugin raises an error that names the extra to install. Configs exported after the move still use these
  class paths, so they also load on older runtimes.
- Imports from `physicalai.robot.so101` and `physicalai.robot.trossen` keep working through thin modules that
  forward to the plugins with a deprecation warning.
- Every other plugin keeps its current class paths, and Robot Type identifiers never change. Existing
  `runtime.yaml` files and Studio projects load unchanged.

---

## Installing

`physicalai` has one extra per robot family, plus aliases for robot models:

| Extra                              | Installs                           |
| ---------------------------------- | ---------------------------------- |
| `so101`                            | `physicalai-so101-plugin`          |
| `bimanual-so101`                   | `physicalai-bimanual-so101-plugin` |
| `trossen`, `widowx`                | `physicalai-trossen-plugin`        |
| `rebot-b601`, `b601-dm`, `b601-rs` | `physicalai-rebot-b601-plugin`     |
| `stararm`                          | `physicalai-stararm-plugin`        |
| `mujoco`                           | `physicalai-mujoco-plugin`         |
| `robots`                           | Every first-party hardware plugin  |

```bash
pip install "physicalai[b601-rs]"      # runtime, reBot plugin and motorbridge
pip install -U "physicalai[b601-rs]"   # upgrades the runtime and the plugin
```

- Each plugin depends on its vendor SDK, so one extra installs everything a robot needs. `pip install physicalai`
  alone installs no robot.
- The existing `plugin-*` extras stay as aliases. LeRobot has no extra because it pulls in torch; install
  `physicalai-lerobot-plugin` directly.
- Plugins support the same Python versions as `physicalai`, 3.11 and later. The extra for one robot fails with a
  clear error on a Python version that robot cannot run on; only the `robots` bundle skips such robots. Today the
  `plugin-*` extras install nothing on Python 3.11.
- Model aliases such as `b601-rs` are a promise to users. If a model later needs a different SDK, its alias can
  change without breaking anyone.

Studio's default environment installs no robot drivers. Its curated plugin manifest lists all first-party robot
types as available, including those whose plugins are not installed. The manifest contains enough metadata to show
a robot's name, type and role; CI checks that metadata against the installed plugin's catalog definitions. The
plugin provides the full configuration schema, builder, probes and assets after installation.

When a user chooses an uninstalled robot, Studio offers an explicit **Install and add** action. It installs only
the plugin package that provides that robot, restarts the backend (the catalog is loaded at startup), then resumes
adding the robot. For example, a user with a B601 installs the reBot plugin, not SO-101, Trossen or Star Arm. The
B601 plugin may provide both DM and RS types; they are one package-level install.

Studio installs the selected plugin distribution directly with versions constrained to Studio's tested
environment. It must not install `physicalai[b601-rs]` from the UI: resolving that extra could upgrade Studio's
`physicalai` version. If a plugin's minimum `physicalai` version exceeds Studio's, Studio reports that a Studio
update is required instead of changing its runtime dependencies. A container deployment must use a supported
persistent setup, such as including selected plugins in its image; installing into a replaceable container is
not durable.

The `physicalai[robots]` extra remains a convenience for users who deliberately want every first-party hardware
plugin in a standalone Runtime environment. Studio does not depend on it. Its Plugins page also offers LeRobot,
MuJoCo and reviewed third-party plugins, and Studio drops its SO-101 and WidowX catalog modules and
`sync_robot_assets.py`.

---

## Versions

- **Independent versions.** Each package has its own version. A reBot fix produces a new reBot version only, and
  users install it without upgrading `physicalai`.
- **Plugin requirements.** A plugin declares the oldest `physicalai` it works with and stays below 1.0, for example
  `physicalai>=0.2.0,<1`.
- **A stable plugin API.** `physicalai` keeps the API that plugins use backward compatible within 0.x: deprecate
  first, remove later. That API is the Robot protocol, the Plugin SDK, `export_config` and the helpers plugins need.
  Helpers that plugins copy or import privately today become public: reBot and Star Arm carry identical copies of
  `motion.py`, and `_CallbackBus` and `transport._lock` are private.
- **Minimum plugin versions in the extras.** Each `physicalai` release requires at least the plugin versions that
  were current when it was released, for example `physicalai-rebot-b601-plugin>=0.8.0`. Without these minimums,
  `pip install -U "physicalai[b601-rs]"` upgrades `physicalai` and leaves the plugin behind, because pip only
  upgrades a dependency when it has to. The release process keeps the minimums current.
- **Unreleased `physicalai` changes.** When a change makes a plugin depend on `physicalai` code that is not released
  yet, the plugin requires `physicalai>X`, where X is the latest release, which reads as "the next release". CI
  reports when this is needed, and the release checks refuse to publish the plugin unless the next `physicalai` is
  part of the same release.

---

## Releasing

Everything in this repo is released from one release PR.

1. **One release PR.** release-please keeps a single PR open that lists every package with releasable changes and
   its next version (`separate-pull-requests: false`). Packages without changes are not released.
2. **Checks before tagging.** CI on the release PR builds every package at its pending version, then:

   - installs each wheel, and `physicalai[robots]`, on Python 3.11 to 3.14 from the new wheels plus PyPI;
   - tests each plugin against the oldest `physicalai` it allows;
   - when `physicalai` is part of the release, tests the latest released version of every plugin against the new
     `physicalai`.

   A mismatch like [#326](https://github.com/openvinotoolkit/physicalai/pull/326) turns the release PR red instead
   of breaking a release.

3. **Release PR update.** After each release-please run, a step raises the minimum plugin versions in the
   `physicalai` extras and reruns `uv lock`.
4. **Publishing.** Merging the release PR tags every package in it. One publish job uploads `physicalai` first and
   the plugins after it, installs the result from PyPI in a clean environment, and notifies Studio to update its
   curated install manifest. Studio receives one update PR per release; the plugins do not enter its base
   dependency lock.

This requires:

- release-please running with a GitHub App token, because PRs opened with `GITHUB_TOKEN` do not trigger CI;
- `main` staying releasable, because the release PR contains every pending change;
- a PyPI trusted publisher for each new package before its first release, the only manual step.

A scheduled job can merge the release PR every two weeks when it is green. An urgent fix merges it immediately.

---

## Guardrails

- **Package consistency check.** A check in prek and CI fails when a package in `packages/` is missing from the
  release-please config, `tool.uv.sources`, the `physicalai` extras or the README robot table; when its supported
  Python versions differ from `physicalai`'s; or when it imports private `physicalai` modules.
- **Plugin scaffold.** A script creates a new plugin that passes the check, with entry points and tests.
- **Assets.** Each plugin keeps its URDF and mesh files inside its own Python package, and every mesh records its
  source and license. Today the reBot, Star Arm, bimanual SO-101 and LeRobot wheels install a shared top-level
  `urdf/` directory into `site-packages`.

---

## Migration

1. **Release process.** One release PR, checks before tagging, ordered publishing. Proven with one release of
   today's packages.
2. **Plugin platform.** The Plugin SDK moves into `physicalai`, the plugin API becomes public, every plugin supports
   Python 3.11, assets move inside each package, and every robot gets an extra.
3. **SO-101 and WidowX AI.** Drivers, Studio Robot Types and assets move into `physicalai-so101-plugin` and
   `physicalai-trossen-plugin`. `physicalai` keeps the old names working.
4. **Studio.** Studio lists first-party plugins as available, installs them only when selected, and removes its
   built-in robot catalog code. `physicalai[robots]` stays opt-in for Runtime users who want all hardware plugins.
5. **Documentation.** The README, user docs, examples and contributor skills describe plugins only.

---

## Considered Options

### First-Party Robots with Light SDKs Built Into `physicalai`

SO-101, WidowX AI, reBot and Star Arm inside `physicalai`, with plugins only for third-party integrations and heavy
ones such as LeRobot and MuJoCo. This was the previous draft of this proposal.

- **For:** one version for the runtime and its robots; four packages; no version requirements between first-party
  packages.
- **Against:**
  - Every robot fix needs a `physicalai` release, and users upgrade the whole runtime, including OpenVINO and
    Transformers, to get it.
  - Robot changes are local and frequent, so the runtime would have to release at the pace of the busiest robot.
  - The rule depends on who maintains a robot and how heavy its SDK is. Both can change (the reBot package already
    describes itself as a "Third-party Seeed reBot B601" plugin), and each change moves a robot again.
  - First-party robots stop using the path vendors use, and the examples that Studio's plugin guide points vendors
    to disappear from `packages/`.
  - It needs the same release process and Studio preinstallation to fix the problems above.

### Keep Today's Split and Fix the Release Process Only

- **For:** no code moves.
- **Against:** SO-101 and WidowX AI stay inside `physicalai` while every other robot is a plugin, so the boundary
  stays accidental.

### One Shared Version for Every Package

- **For:** a single version number and no version requirements to maintain.
- **Against:** every robot fix becomes a release of everything, which is the cost this design avoids.

---

## Trade-offs

- `main` must stay releasable.
- The plugin API needs deprecation discipline.
- Contributors occasionally update a plugin's minimum `physicalai` version.
- Eight packages, where the built-in option had four.
- Plugins with large meshes stay large: the reBot wheel is 35 MB, and runtime-only users download meshes they do
  not need.

---

## Out of Scope

- A common naming scheme for plugin drivers, such as `physicalai.robot.<Name>` for every plugin.
- Folding bimanual SO-101 into the SO-101 plugin, which would change its class path.
- Camera backends, which stay in `physicalai` with extras.
- The third-party plugin platform: rules for the Curated Plugin List, a plugin template repository and a contract
  test kit.
- A `physicalai[lerobot]` extra, and robot discovery in the CLI.

---

## References

- [#262](https://github.com/openvinotoolkit/physicalai/pull/262),
  [#263](https://github.com/openvinotoolkit/physicalai/pull/263),
  [#265](https://github.com/openvinotoolkit/physicalai/pull/265): the plugin system and the first-party plugins
- [#269](https://github.com/openvinotoolkit/physicalai/issues/269): merge the Plugin SDK into `physicalai`
- [#275](https://github.com/openvinotoolkit/physicalai/issues/275): extract SO-101 and WidowX AI as plugins
- [Robot Interface](../components/robot-interface.md): the Robot protocol
- [Modular Packages in One Repo](./modular-packages-in-one-repo.md): publishing several packages from one repo
- Studio: `application/docs/robot-plugins.md` and `application/docs/explanation/robot-plugin-architecture.md`
