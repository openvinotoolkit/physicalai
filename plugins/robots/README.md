# Robot Plugins

Robot plugins provide independently packaged robot integrations for
[Physical AI Runtime](../../README.md) and
[Physical AI Studio](https://github.com/open-edge-platform/physical-ai-studio).
They add hardware or simulated robots without adding robot-specific code to Studio.

Each package owns its robot drivers, vendor dependencies, Studio catalog definitions,
tests, and optional assets such as URDFs and meshes. Runtime supplies the common
robot interface, configuration system, and control loop. Studio discovers the
installed plugins and supplies project persistence, generated configuration forms,
teleoperation workflows, and hardware-process orchestration.

A driver can be used directly with Runtime without running Studio. Its Studio
integration lives separately in `studio_catalog.py`, using the
[`physicalai-studio-plugin` SDK](../physicalai-studio-plugin/README.md).

## Packages in This Directory

| Package                                                                | Integration                                          |
| ---------------------------------------------------------------------- | ---------------------------------------------------- |
| [physicalai-so101-plugin](physicalai-so101-plugin/README.md)           | SO-101 robot drivers and catalog entries             |
| [physicalai-trossen-plugin](physicalai-trossen-plugin/README.md)       | Trossen WidowX AI, including bimanual configurations |
| [physicalai-rebot-b601-plugin](physicalai-rebot-b601-plugin/README.md) | Seeed reBot B601 DM and RS followers                 |
| [physicalai-stararm-plugin](physicalai-stararm-plugin/README.md)       | Fashion Star Arm leaders and follower                |
| [physicalai-lerobot-plugin](physicalai-lerobot-plugin/README.md)       | Adapters for LeRobot robot integrations              |
| [physicalai-mujoco-plugin](physicalai-mujoco-plugin/README.md)         | MuJoCo simulation drivers, scenes, and visualization |

Each package has its own `pyproject.toml`, `src/`, and `tests/`. See its README for
supported configurations, installation requirements, and hardware-specific setup.

## Build a Robot Plugin

Start with these guides rather than copying Studio internals:

- [Runtime robot API](../../docs/reference/robot-api.md): the `Robot` and
  `RobotObservation` contracts, joint ordering, device ownership, and lifecycle safety.
- [Robot plugin development guide](../../skills/using/pai-create-robot-plugin/SKILL.md):
  the end-to-end workflow for implementing, packaging, testing, and integrating a driver.
- [Studio plugin SDK reference](../physicalai-studio-plugin/README.md): payloads,
  catalog registration, builders, discovery probes, assets, calibration, and simulation.
- [Studio robot plugin guide](https://github.com/open-edge-platform/physical-ai-studio/blob/main/application/docs/robot-plugins.md):
  installation, development, generated forms, curated plugins, and troubleshooting.
- [Studio plugin architecture](https://github.com/open-edge-platform/physical-ai-studio/blob/main/application/docs/explanation/robot-plugin-architecture.md):
  catalog discovery and the integration lifecycle.

The main implementation steps are:

1. Implement the Runtime `Robot` protocol in the plugin package. Keep driver code
   independent of Studio imports, preserve joint/action ordering, and implement safe
   connection and disconnection behavior.
2. Decorate driver classes returned to Studio with `physicalai.config.export_config`.
   Studio serializes the disconnected driver to construct it in its hardware-owner process.
3. Declare `physicalai`, `physicalai-studio-plugin`, and vendor dependencies in the
   plugin's `pyproject.toml`. Include optional robot assets in the built distributions.
4. Define Pydantic configuration payloads and register a `RobotCatalogDefinition`
   for each supported configuration. A **follower** executes actions; a **leader**
   supplies teleoperation input. Keep catalog `type` identifiers globally unique and
   stable because Studio persists them in projects.
5. Return a plain, unconnected driver from each async builder, not a `SharedRobot`.
   Studio owns the shared wrapper and connection lifetime. Add probes, assets,
   calibration, or simulation support only when the robot implements those capabilities.
6. Test drivers with mocked hardware and test catalog registration, payload validation,
   builders, and configuration round-trips before testing with Studio.

For an existing workspace package, run its tests and build its wheel from the Runtime
repository root. For example:

```bash
uv run --package physicalai-so101-plugin --extra tests pytest plugins/robots/physicalai-so101-plugin/tests
uv build --package physicalai-so101-plugin --wheel
```

## Integrate with Studio

### Register the Catalog Entry Point

Declare the entry point in the plugin's `pyproject.toml`:

```toml
[project.entry-points."physicalai.studio.catalog_plugins"]
my-robot = "physicalai_my_robot_plugin.studio_catalog:register_physicalai_studio_plugin"
```

The callable receives the catalog registry and calls
`registry.register_robot(definition)` for each robot configuration. Studio loads these
entry points at backend startup. It generates configuration forms from the payload
models; ordinary integrations do not need plugin-specific React components.

### Install into the Studio Backend Environment

For local development, run the following from your Studio checkout, replacing the
plugin path with your own package:

```bash
cd application/backend
uv add --editable /path/to/physicalai/plugins/robots/physicalai-so101-plugin
uv sync
```

Restart the backend after installation or changes to entry points, catalog definitions,
or payload schemas. Installing into another Python environment does not make a plugin
available to Studio. After restarting, open a project and select **Robots → Add robot**
to choose one of the plugin's types.

Check `/api/robots/catalog` and `/api/robots/catalog/{robot_type}/schema` on your Studio
instance if a type is missing or its generated form is incorrect. Backend startup logs
report entry-point import failures, duplicate types, and invalid catalog schemas.

### Offer Installation through Studio's Plugins Page

To make a plugin available on the curated **Plugins** page, submit a reviewed entry
to Studio's `application/backend/src/plugins/manifest.json`. The entry identifies the
Python distribution, its installation source, and the robot types it provides.

The manifest controls what the UI offers to install; the package entry point registers
the actual robot definitions. A privately installed plugin can appear in the robot
picker without being listed on the Plugins page. See the
[Studio robot plugin guide](https://github.com/open-edge-platform/physical-ai-studio/blob/main/application/docs/robot-plugins.md)
for the manifest format and installation workflow.
