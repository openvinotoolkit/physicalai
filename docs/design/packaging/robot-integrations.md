# Robot Integration Packaging

## Summary

This document proposes that every robot integration, including SO-101 and WidowX AI, is distributed as a separate
plugin package, and that `physicalai` contains no robot drivers. `physicalai` retains the runtime, the robot
interface and the Plugin SDK, which plugins use to describe their robots to Studio.

The proposal has four consequences:

- Users install a robot through an optional extra, for example `pip install "physicalai[b601-rs]"`. Existing
  configurations and Studio projects continue to load.
- A fix to a robot integration is released as a new version of that plugin, without a new `physicalai` release.
- Studio lists all supported robots but installs only the plugins a user selects.
- All packages in this repository are released from a single release pull request, which verifies them and
  publishes them in dependency order.

## Motivation

**The current boundary is historical.** SO-101 and WidowX AI were added to `physicalai` before the plugin system
existed; reBot, Star Arm, bimanual SO-101 and the LeRobot bridge were added later as plugins. Consequently,
bimanual WidowX AI is part of `physicalai` while bimanual SO-101 is a plugin, and the SO-101 and WidowX AI robot
definitions used by Studio are maintained in the Studio repository. There is no stated rule for where a new robot
belongs.

**Releases are coordinated by hand.** Each of the seven packages has its own release pull request. Between
2026-09-14 and 2026-10-06, two of eleven releases failed: one because of a workflow error
([#268](https://github.com/openvinotoolkit/physicalai/pull/268)), and one because a plugin was published before
the SDK version it required ([#326](https://github.com/openvinotoolkit/physicalai/pull/326), reverted in
[#339](https://github.com/openvinotoolkit/physicalai/pull/339)). A single feature, zero-pose calibration, required
three sequential releases across the SDK, the plugins and Studio.

**Version requirements between packages are not tested.** Within the development workspace, packages use each
other's source code regardless of their declared version requirements. As a result, the hardware plugins declare
`physicalai>=0.1.1` but depend on functionality introduced in 0.2.0, and some plugins import private
`physicalai` modules that may change without notice.

**Studio environments diverge from what was tested.** Installing a plugin from Studio can upgrade the SDK in
Studio's environment to an untested version. Installed plugins cannot be updated from the interface, and they are
lost when a Docker container is replaced.

**Robot integrations change independently of the runtime.** Since the plugins moved into this repository, every
change to them has been confined to plugin code; none modified `physicalai`. The reBot plugin has been
released eleven times since June, while `physicalai` has been released once.

## Proposal

`physicalai` contains no robot drivers. Each robot family is a separate plugin package, whether maintained by this
team or by a third party. First-party plugins are developed in `packages/` and released together with
`physicalai`.

| Robot integration                         | Current location                           | Proposed package                                       |
| ----------------------------------------- | ------------------------------------------ | ------------------------------------------------------ |
| SO-101                                    | `physicalai`; Studio definitions in Studio | `physicalai-so101-plugin`                              |
| Bimanual SO-101                           | `physicalai-bimanual-so101-plugin`         | Unchanged; depends on `physicalai-so101-plugin`        |
| WidowX AI (single, bimanual)              | `physicalai`; Studio definitions in Studio | `physicalai-trossen-plugin`                            |
| reBot B601, Star Arm 102, LeRobot, MuJoCo | Existing plugins                           | Unchanged                                              |
| Plugin SDK                                | `physicalai-studio-plugin`                 | Moved into `physicalai`; the old package re-exports it |

Each plugin contains its drivers, its Studio robot definitions and its 3D models. Camera backends remain in
`physicalai`.

### Compatibility

Existing names are preserved. `physicalai.robot.SO101`, `physicalai.robot.WidowXAI` and
`physicalai.robot.BimanualWidowXAI` remain valid and load the corresponding plugin; if the plugin is not installed,
the error names the extra to install. Other plugins keep their current class paths, and Studio robot identifiers do
not change. Existing configuration files and Studio projects therefore load without modification.

## Installation

`physicalai` provides one optional extra per robot family, with aliases for individual models:

| Extra                              | Installs                           |
| ---------------------------------- | ---------------------------------- |
| `so101`                            | `physicalai-so101-plugin`          |
| `bimanual-so101`                   | `physicalai-bimanual-so101-plugin` |
| `trossen`, `widowx`                | `physicalai-trossen-plugin`        |
| `rebot-b601`, `b601-dm`, `b601-rs` | `physicalai-rebot-b601-plugin`     |
| `stararm`                          | `physicalai-stararm-plugin`        |
| `mujoco`                           | `physicalai-mujoco-plugin`         |
| `robots`                           | All first-party hardware plugins   |

Each plugin depends on its vendor SDK, so a single extra installs everything a robot requires; `physicalai` alone
installs no robot. Plugins support the same Python versions as `physicalai`. The existing `plugin-*` extras remain
as aliases. LeRobot is installed directly, as it depends on PyTorch.

### Studio

Studio installs no robot plugins by default. It lists every supported first-party robot, and when a user selects
one that is not installed, Studio installs only the corresponding plugin and restarts. A user with a reBot B601,
for example, installs the reBot plugin and nothing else.

Studio installs plugins at reviewed versions and does not allow an installation to change its own `physicalai`
version or shared dependencies. If a plugin requires a newer `physicalai`, Studio reports that an update is needed.
Plugin selections must persist in Docker deployments.

## Versioning

Each package is versioned independently. A plugin declares the oldest `physicalai` version it supports and an
upper bound of 1.0, for example `physicalai>=0.2.0,<1`. In turn, `physicalai` keeps the interface used by plugins
backward compatible within 0.x: features are deprecated before they are removed. Helpers that plugins currently
duplicate or import from private modules become part of this public interface.

The extras in each `physicalai` release require at least the plugin versions available at that release, so that
upgrading through an extra also upgrades the plugin. When a plugin depends on unreleased `physicalai` changes, it declares a requirement on the
next `physicalai` release, and the release checks ensure both are published together.

## Release Process

All packages in this repository are released from one release pull request:

1. The pull request lists each package with releasable changes and its next version. Unchanged packages are not
   released.
2. Before anything is published, automated checks build each package at its new version, install it on every
   supported Python version, and test each plugin against the oldest and newest `physicalai` it supports. A
   dependency problem such as the one in [#326](https://github.com/openvinotoolkit/physicalai/pull/326) fails the
   pull request instead of the release.
3. Merging the pull request publishes `physicalai` first and the plugins after it, verifies the published packages,
   and notifies Studio.

This requires that the main branch remains releasable at all times, since the release pull request includes every
pending change. Each new package also requires a one-time publishing setup on PyPI.

Additional automated checks ensure that every package is registered consistently, supports the same Python versions
as `physicalai`, uses only its public interface, and ships its 3D models with a recorded source and license.

## Migration

1. Introduce the single release process and verify it with a release of the current packages.
2. Move the Plugin SDK into `physicalai`, make the plugin interface public, and align Python versions and extras
   across the existing plugins.
3. Move SO-101 and WidowX AI into plugins, preserving their existing names.
4. Update Studio to list robots and install them on demand, and remove its built-in robot definitions.
5. Update the documentation, examples and contributor guides.

## Alternatives Considered

**Include first-party robots in `physicalai`.** SO-101, WidowX AI, reBot and Star Arm would be part of
`physicalai`, and plugins would be reserved for third-party and heavy integrations such as LeRobot and MuJoCo. This
gives one version for the runtime and its robots and four packages in total. However, every robot fix would require
a `physicalai` release and a runtime upgrade; the rule depends on ownership and dependency size, both of which can
change; and first-party robots would no longer follow the path used by external developers.

**Keep the current split and improve only the release process.** This avoids moving code but leaves SO-101 and
WidowX AI as exceptions to the plugin model.

**Use one version for all packages.** This removes version requirements between packages, but every robot fix would
produce a release of every package.

## Trade-offs

- The main branch must remain releasable.
- The plugin interface requires a deprecation policy.
- The repository publishes eight packages instead of four under the first alternative.
- Plugins with large 3D models remain large for users who do not use Studio.

## Out of Scope

- A common naming scheme for all plugin drivers.
- Merging bimanual SO-101 into the SO-101 plugin.
- Distributing camera backends as plugins.
- Guidelines and tooling for third-party plugin authors.
- A `physicalai[lerobot]` extra.

## References

- [#262](https://github.com/openvinotoolkit/physicalai/pull/262),
  [#263](https://github.com/openvinotoolkit/physicalai/pull/263),
  [#265](https://github.com/openvinotoolkit/physicalai/pull/265): plugin system and first-party plugins
- [#269](https://github.com/openvinotoolkit/physicalai/issues/269),
  [#275](https://github.com/openvinotoolkit/physicalai/issues/275): original investigations
- [#352](https://github.com/openvinotoolkit/physicalai/issues/352): implementation epic
- [Robot Interface](../components/robot-interface.md)
- [Modular Packages in One Repo](./modular-packages-in-one-repo.md)
