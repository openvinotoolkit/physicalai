# Import ROS 2 recordings

The optional `physicalai-rosbag-plugin` converts offline ROS 2 demonstrations into
LeRobot v3 datasets accepted by Physical AI Studio. No ROS installation or physical
robot is required. Install the plugin in a separate Python 3.12 environment to keep
its LeRobot/Torch dependencies separate from a deployment-only Runtime installation.

```bash
physicalai rosbag convert --input ./bags --mapping ./mapping.yaml --output ./converted --repo-id local/demonstrations
```

Each recording becomes one episode. Explicit mappings select state, commands,
camera topics, units, and a fixed sampling rate. Missing or stale required data
fails conversion. Only uncompressed SQLite3/MCAP containers are supported; JPEG/PNG
camera messages are supported. Live ROS capture and raw-bag upload in Studio are
outside this plugin's scope.

Follow the [converter guide](https://github.com/openvinotoolkit/physicalai/tree/main/packages/physicalai-rosbag-plugin)
for source installation, the mapping schema, resource limits, and a worked synthetic
example. ZIP the completed output and use Studio's existing dataset import flow:
upload, review the detected cameras/joints, choose an environment, and finalize.

The guide also provides an opt-in verification script for the actual Studio backend,
dataset preview, CPU training, export, and Runtime inference. Synthetic examples
verify software interoperability, not real-world policy performance.
