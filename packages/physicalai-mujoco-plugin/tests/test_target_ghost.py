from __future__ import annotations

from unittest.mock import MagicMock, patch

import mujoco
import numpy as np
import pytest

from physicalai_mujoco_plugin.target_ghost import GHOST_OPACITY, TargetGhost, moved_bodies, servo_targets
from physicalai_mujoco_plugin.viewer import ViewerService

_XML = """<mujoco><worldbody>
  <body name="base"><geom size="0.05"/>
    <body name="upper"><joint name="shoulder" axis="0 1 0"/><geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02"/>
      <body name="lower" pos="0.3 0 0"><joint name="elbow" axis="0 1 0"/><geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02"/>
        <body name="hull"><geom size="0.01" group="3"/></body>
      </body>
    </body>
  </body>
  <body name="cube" pos="1 0 0"><freejoint/><geom type="box" size="0.02 0.02 0.02"/></body>
</worldbody>
<actuator>
  <position name="shoulder" joint="shoulder" kp="10" gear="2"/>
  <motor name="elbow" joint="elbow"/>
</actuator></mujoco>"""


@pytest.fixture
def model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(_XML)


def _ids(model: mujoco.MjModel, *names: str) -> list[int]:
    return [model.body(name).id for name in names]


def test_only_position_servos_on_a_joint_have_a_target(model: mujoco.MjModel) -> None:
    shoulder = model.joint("shoulder")
    assert servo_targets(model) == [(0, int(shoulder.qposadr[0]), 0.5, shoulder.id)]  # the motor has none


def test_the_ghost_covers_every_body_the_driven_joints_move(model: mujoco.MjModel) -> None:
    assert moved_bodies(model, {model.joint("elbow").id}) == _ids(model, "lower", "hull")
    assert moved_bodies(model, {model.joint("shoulder").id}) == _ids(model, "upper", "lower", "hull")


def test_the_ghost_stands_at_the_servo_and_driven_targets(model: mujoco.MjModel) -> None:
    server = MagicMock()
    data = mujoco.MjData(model)
    ghost = TargetGhost(server, model, driven_joints=[model.joint("elbow").id])
    assert ghost.bodies == _ids(model, "upper", "lower")  # not the collision hull, not the cube
    assert {call.kwargs["opacity"] for call in server.scene.add_mesh_simple.call_args_list} == {GHOST_OPACITY}

    data.ctrl[0] = 1.0  # 0.5 rad at the joint, through gear 2
    pose = ghost.pose(data, [(model.joint("elbow").id, 0.3)])

    assert pose.qpos[model.joint("shoulder").qposadr[0]] == pytest.approx(0.5)
    assert pose.qpos[model.joint("elbow").qposadr[0]] == pytest.approx(0.3)
    assert data.qpos[model.joint("shoulder").qposadr[0]] == 0.0  # the simulation itself is untouched
    np.testing.assert_allclose(pose.xpos[model.body("lower").id], [0.3 * np.cos(0.5), 0, -0.3 * np.sin(0.5)], atol=1e-9)


def test_the_ghost_hides_while_its_targets_mean_nothing(model: mujoco.MjModel) -> None:
    server = MagicMock()
    ghost = TargetGhost(server, model)
    data = mujoco.MjData(model)
    handles = [handle for _body, handle in ghost._handles]  # noqa: SLF001

    ghost.update(data, None)  # a replay
    assert all(handle.visible is False for handle in handles)
    ghost.update(data, [])
    assert all(handle.visible is True for handle in handles)
    ghost.set_visible(False)
    ghost.update(data, [])
    assert all(handle.visible is False for handle in handles)


def test_the_viewer_shows_the_ghost_with_a_toggle(model: mujoco.MjModel) -> None:
    server = MagicMock()
    viewer = ViewerService(host="127.0.0.1", port=9090, submit_command=MagicMock(), panel_state=MagicMock())
    with (
        patch("viser.ViserServer", return_value=server),
        patch("mjviser.ViserMujocoScene", return_value=MagicMock()),
        patch("physicalai_mujoco_plugin.viser_controls.SimControlPanel"),
    ):
        assert viewer.open(model, mujoco.MjData(model)) is True

    assert viewer.ghost is not None
    assert viewer.ghost.bodies == _ids(model, "upper", "lower")
    labels = [call.args[0] for call in server.gui.add_checkbox.call_args_list]
    assert "Target pose (ghost)" in labels
