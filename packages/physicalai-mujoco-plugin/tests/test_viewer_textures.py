"""Textured boxes keep their texture in the browser viewer, and only there."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

pytest.importorskip("mjviser")
import mjviser.conversions as conv  # noqa: E402
import trimesh  # noqa: E402

from physicalai_mujoco_plugin.scene_registry import get_scene  # noqa: E402
from physicalai_mujoco_plugin.viewer_textures import box_uvs, install_box_textures  # noqa: E402

MJCF = """
<mujoco>
  <asset>
    <texture name="wood" type="2d" builtin="checker" rgb1="0.6 0.4 0.3" rgb2="0.5 0.3 0.2" width="64" height="64"/>
    <material name="wood" texture="wood" texrepeat="3 3" texuniform="true"/>
    <material name="plain" rgba="0.2 0.3 0.4 1"/>
  </asset>
  <worldbody>
    <geom name="top" type="box" size="0.5 0.9 0.02" material="wood"/>
    <geom name="plain" type="box" size="0.1 0.1 0.1" pos="0 0 1" material="plain"/>
    <geom name="ball" type="sphere" size="0.1" pos="1 0 0" material="wood"/>
  </worldbody>
</mujoco>
"""


@pytest.fixture(scope="module")
def model() -> mujoco.MjModel:
    assert install_box_textures()
    return mujoco.MjModel.from_xml_string(MJCF)


def _geom(model: mujoco.MjModel, name: str) -> int:
    return int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name))


def test_a_textured_box_gets_its_texture_and_uvs(model: mujoco.MjModel) -> None:
    gid = _geom(model, "top")
    assert conv.get_geom_texture_id(model, gid) == int(model.mat_texid[model.geom_matid[gid]].max())
    mesh = conv.create_primitive_mesh(model, gid)
    assert isinstance(mesh.visual, trimesh.visual.TextureVisuals)
    assert mesh.visual.uv.shape == (len(mesh.vertices), 2)
    # texuniform with texrepeat 3: three repeats per metre across the 1.8 m long top.
    assert np.ptp(mesh.visual.uv, axis=0).max() == pytest.approx(1.8 * 3)


@pytest.mark.parametrize("name", ["plain", "ball"])
def test_untextured_boxes_and_other_primitives_stay_flat(model: mujoco.MjModel, name: str) -> None:
    gid = _geom(model, name)
    assert conv.get_geom_texture_id(model, gid) == -1
    assert not isinstance(conv.create_primitive_mesh(model, gid).visual, trimesh.visual.TextureVisuals)


def test_installing_twice_patches_once() -> None:
    patched = conv.create_primitive_mesh
    assert install_box_textures()
    assert conv.create_primitive_mesh is patched


def test_box_uvs_fit_one_face_without_texuniform() -> None:
    vertices = np.array([[0.5, -0.9, 0.02], [0.5, 0.9, 0.02], [0.5, 0.9, -0.02]])
    normals = np.tile([1.0, 0.0, 0.0], (3, 1))
    uv = box_uvs(vertices, normals, np.array([0.5, 0.9, 0.02]), np.array([1.0, 1.0]), uniform=False)
    np.testing.assert_allclose(uv, [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0]])


def test_the_bucket_sort_table_is_textured_in_the_viewer() -> None:
    install_box_textures()
    model = mujoco.MjModel.from_xml_path(str(get_scene("bucket_sort").scene_xml_path))
    gid = _geom(model, "table_top")
    assert isinstance(conv.create_primitive_mesh(model, gid).visual, trimesh.visual.TextureVisuals)
