# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Live camera edits: apply ``overview``/``wrist`` camera and camera-rig changes saved to the scene XML.

A development aid for placing cameras: edit the scene XML while the simulation runs, and the
camera moves within about a second, without recompiling the model.
"""

# MuJoCo model/data attributes are supplied by the C extension at runtime.
# pyrefly: ignore-errors [missing-attribute]

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from loguru import logger

POLL_INTERVAL_S = 1.0
"""Walking the include graph is far too expensive to do on every control cycle."""


class SceneXmlWatcher:
    """Poll a scene XML and everything it includes for edits to its cameras."""

    def __init__(self, xml_path: str | Path) -> None:
        """Watch *xml_path*, baselining the current file times."""
        self.reset(xml_path)

    def reset(self, xml_path: str | Path) -> None:
        """Watch *xml_path* from now on: re-walk its include graph and re-baseline the file times."""
        self._root = Path(xml_path)
        self._cached_paths: list[Path] | None = None
        self._mtimes = self._snapshot()
        self._next_check = time.monotonic() + POLL_INTERVAL_S

    def poll(self, model: object, data: object) -> None:
        """Apply camera edits if a watched file changed since the last poll (sim thread)."""
        now = time.monotonic()
        if now < self._next_check:
            return
        self._next_check = now + POLL_INTERVAL_S
        mtimes = self._snapshot()
        if mtimes == self._mtimes:
            return
        logger.info("Scene XML changed, updating camera")
        self._mtimes = mtimes
        self._apply_camera_edits(model, data)
        # The edit may have added or removed an <include>; re-walk next poll.
        self._cached_paths = None

    def _paths(self) -> list[Path]:
        """Return the scene XML and everything it includes, walking the XML only when stale."""
        if self._cached_paths is None:
            self._cached_paths = self._collect_paths()
        return self._cached_paths

    def _collect_paths(self) -> list[Path]:
        from defusedxml import ElementTree  # noqa: PLC0415

        visited: set[Path] = set()
        ordered: list[Path] = []

        def walk(path: Path) -> None:
            normalized = path.resolve()
            if normalized in visited:
                return
            visited.add(normalized)
            ordered.append(normalized)
            try:
                root = ElementTree.parse(normalized).getroot()
            except (ElementTree.ParseError, OSError):
                return
            for include in root.findall(".//include"):
                include_file = include.get("file")
                if include_file:
                    walk(normalized.parent / include_file)

        walk(self._root)
        return ordered

    def _snapshot(self) -> dict[str, float]:
        mtimes: dict[str, float] = {}
        for xml_path in self._paths():
            try:
                mtimes[str(xml_path)] = xml_path.stat().st_mtime
            except OSError:
                continue
        return mtimes

    def _apply_camera_edits(self, model: object, data: object) -> None:  # noqa: C901, PLR0912, PLR0915
        import mujoco  # noqa: PLC0415
        from defusedxml import ElementTree  # noqa: PLC0415

        roots = []
        for xml_path in self._paths():
            # _collect_scene_xml_paths keeps unreadable includes in the list, so
            # OSError is as survivable here as a parse error.
            try:
                roots.append(ElementTree.parse(xml_path).getroot())
            except (ElementTree.ParseError, OSError) as exc:
                logger.warning("Failed to parse XML {}: {}", xml_path, exc)

        if not roots:
            return

        def find_first(xpath: str) -> object | None:
            for root in roots:
                # pyrefly: ignore [missing-attribute]
                elem = root.find(xpath)
                if elem is not None:
                    return elem
            return None

        cam = find_first(".//camera[@name='overview']")
        if cam is None:
            return

        for body_name in (
            "overview_camera_rig",
            "overview_camera_tilt",
            "camera_mount",
            "camera_mount_wrist",
            "left_camera_mount",
            "right_camera_mount",
        ):
            body_elem = find_first(f".//body[@name='{body_name}']")
            # pyrefly: ignore [missing-attribute]
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_elem is None or body_id < 0:
                continue

            # pyrefly: ignore [missing-attribute]
            pos_str = body_elem.get("pos")
            if pos_str:
                # pyrefly: ignore [missing-attribute]
                model.body_pos[body_id] = [float(x) for x in pos_str.split()]

            # pyrefly: ignore [missing-attribute]
            euler_str = body_elem.get("euler")
            if euler_str:
                euler_vals = [float(x) for x in euler_str.split()]
                if len(euler_vals) == 3:  # noqa: PLR2004
                    quat = np.zeros(4, dtype=np.float64)
                    # pyrefly: ignore [missing-attribute]
                    mujoco.mju_euler2Quat(quat, euler_vals, "xyz")
                    # pyrefly: ignore [missing-attribute]
                    model.body_quat[body_id] = quat
                    logger.info("Updated {}: {}", body_name, euler_vals)
                else:
                    logger.warning("Invalid {} euler values: {}", body_name, euler_str)

            # pyrefly: ignore [missing-attribute]
            quat_str = body_elem.get("quat")
            if quat_str:
                quat_vals = [float(x) for x in quat_str.split()]
                if len(quat_vals) == 4:  # noqa: PLR2004
                    model.body_quat[body_id] = quat_vals  # pyrefly: ignore [missing-attribute]
                    logger.info("Updated {} quat: {}", body_name, quat_vals)
                else:
                    logger.warning("Invalid {} quat values: {}", body_name, quat_str)

        def update_camera_pose(camera_name: str, camera_elem: object) -> None:  # noqa: PLR0912
            camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)  # pyrefly: ignore [missing-attribute]
            if camera_id < 0:
                return

            # pyrefly: ignore [missing-attribute]
            pos_str = camera_elem.get("pos")
            if pos_str:
                pos = [float(x) for x in pos_str.split()]
                # pyrefly: ignore [missing-attribute]
                model.cam_pos[camera_id] = pos

            # pyrefly: ignore [missing-attribute]
            fovy_str = camera_elem.get("fovy")
            if fovy_str:
                # pyrefly: ignore [missing-attribute]
                model.cam_fovy[camera_id] = float(fovy_str)

            # pyrefly: ignore [missing-attribute]
            xyaxes_str = camera_elem.get("xyaxes")
            # pyrefly: ignore [missing-attribute]
            euler_str = camera_elem.get("euler")
            # pyrefly: ignore [missing-attribute]
            quat_str = camera_elem.get("quat")

            if quat_str:
                quat_vals = [float(x) for x in quat_str.split()]
                if len(quat_vals) == 4:  # noqa: PLR2004
                    model.cam_quat[camera_id] = quat_vals  # pyrefly: ignore [missing-attribute]
                    logger.info("Updated camera {} quat: {}", camera_name, quat_vals)
                else:
                    logger.warning("Invalid {} camera quat values: {}", camera_name, quat_str)
            elif xyaxes_str:
                vals = [float(x) for x in xyaxes_str.split()]
                if len(vals) == 6:  # noqa: PLR2004
                    # As the MJCF compiler does: normalize x, make y orthogonal
                    # to it, and use the axes as the rotation matrix's columns.
                    x_axis = np.asarray(vals[:3], dtype=np.float64)
                    x_axis /= np.linalg.norm(x_axis)
                    y_axis = np.asarray(vals[3:], dtype=np.float64)
                    y_axis -= (y_axis @ x_axis) * x_axis
                    y_axis /= np.linalg.norm(y_axis)
                    mat = np.column_stack((x_axis, y_axis, np.cross(x_axis, y_axis))).ravel()
                    quat = np.zeros(4, dtype=np.float64)
                    # pyrefly: ignore [missing-attribute]
                    mujoco.mju_mat2Quat(quat, mat)
                    # pyrefly: ignore [missing-attribute]
                    model.cam_quat[camera_id] = quat
                    logger.info("Updated camera {} xyaxes: {}", camera_name, vals)
                else:
                    logger.warning("Invalid {} camera xyaxes values: {}", camera_name, xyaxes_str)
            elif euler_str:
                euler_vals = [float(x) for x in euler_str.split()]
                if len(euler_vals) == 3:  # noqa: PLR2004
                    quat = np.zeros(4, dtype=np.float64)
                    # pyrefly: ignore [missing-attribute]
                    mujoco.mju_euler2Quat(quat, euler_vals, "xyz")
                    # pyrefly: ignore [missing-attribute]
                    model.cam_quat[camera_id] = quat
                    logger.info("Updated camera {} euler: {} -> quat={}", camera_name, euler_vals, quat.tolist())
                else:
                    logger.warning("Invalid {} camera euler values: {}", camera_name, euler_str)
            else:
                logger.info("No orientation attr (euler/xyaxes/quat) on {} camera", camera_name)

        update_camera_pose("overview", cam)
        for camera_name in ("wrist", "left_wrist", "right_wrist"):
            wrist_cam = find_first(f".//camera[@name='{camera_name}']")
            if wrist_cam is not None:
                update_camera_pose(camera_name, wrist_cam)

        # pyrefly: ignore [missing-attribute]
        mujoco.mj_forward(model, data)


__all__ = ["POLL_INTERVAL_S", "SceneXmlWatcher"]
