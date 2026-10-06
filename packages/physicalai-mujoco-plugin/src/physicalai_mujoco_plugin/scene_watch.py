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

_MIN_NORM = 1e-9
"""Shortest quaternion or camera axis an edit may give; MuJoCo cannot normalize a zero one."""

_ORIENTATION_SIZES = {"quat": 4, "xyaxes": 6, "euler": 3}


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
        if self.changed():
            self.apply(model, data)

    def changed(self) -> bool:
        """Return whether a watched file changed since the last `apply`; checks at most once a second.

        Returns:
            ``True`` at every check until `apply` runs; call `apply` then, with nothing else
            reading the model, or defer it to a later check.
        """
        now = time.monotonic()
        if now < self._next_check:
            return False
        self._next_check = now + POLL_INTERVAL_S
        return self._snapshot() != self._mtimes

    def apply(self, model: object, data: object) -> None:
        """Write the scene XML's camera poses into *model*.

        This mutates the model, so no other thread may be using it (MuJoCo shares a model between
        threads only while it is read-only): stop the camera thread first.
        """
        logger.info("Scene XML changed, updating camera")
        self._mtimes = self._snapshot()  # before reading: a save during the edit shows up next check
        try:
            self._apply_camera_edits(model, data)
        except Exception as exc:  # noqa: BLE001 - a half-typed edit must not stop the control loop
            logger.warning("Scene XML camera edit not applied: {}", exc)
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

    def _apply_camera_edits(self, model: object, data: object) -> None:  # noqa: C901, PLR0915
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

            for attr in ("euler", "quat"):  # as before: a quat wins over an euler
                # pyrefly: ignore [missing-attribute]
                text = body_elem.get(attr)
                if not text:
                    continue
                quat = _orientation_quat(attr, text)
                if quat is None:
                    logger.warning("Invalid {} {} values: {}", body_name, attr, text)
                    continue
                model.body_quat[body_id] = quat  # pyrefly: ignore [missing-attribute]
                logger.info("Updated {} {}: {}", body_name, attr, text)

        def update_camera_pose(camera_name: str, camera_elem: object) -> None:
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

            # The first orientation attribute present wins, as before.
            for attr in ("quat", "xyaxes", "euler"):
                # pyrefly: ignore [missing-attribute]
                text = camera_elem.get(attr)
                if not text:
                    continue
                quat = _orientation_quat(attr, text)
                if quat is None:
                    logger.warning("Invalid {} camera {} values: {}", camera_name, attr, text)
                else:
                    model.cam_quat[camera_id] = quat  # pyrefly: ignore [missing-attribute]
                    logger.info("Updated camera {} {}: {} -> quat={}", camera_name, attr, text, quat.tolist())
                return
            logger.info("No orientation attr (euler/xyaxes/quat) on {} camera", camera_name)

        update_camera_pose("overview", cam)
        for camera_name in ("wrist", "left_wrist", "right_wrist"):
            wrist_cam = find_first(f".//camera[@name='{camera_name}']")
            if wrist_cam is not None:
                update_camera_pose(camera_name, wrist_cam)

        # pyrefly: ignore [missing-attribute]
        mujoco.mj_forward(model, data)


def _orientation_quat(attr: str, text: str) -> np.ndarray | None:
    """Convert an MJCF ``quat``, ``xyaxes`` or ``euler`` value to a quaternion.

    Returns:
        The unit quaternion, or ``None`` if a half-typed edit left a value MuJoCo cannot use: the
        wrong count, a non-finite value, a zero or overflowing quaternion, or a zero or collinear
        ``xyaxes`` axis.
    """
    import mujoco  # noqa: PLC0415

    vals = np.asarray([float(x) for x in text.split()], dtype=np.float64)
    if vals.size != _ORIENTATION_SIZES[attr] or not np.isfinite(vals).all():
        return None
    quat = np.zeros(4, dtype=np.float64)
    if attr == "quat":
        quat = vals
    elif attr == "euler":
        # pyrefly: ignore [missing-attribute]
        mujoco.mju_euler2Quat(quat, vals, "xyz")
    else:
        # As the MJCF compiler does: normalize x, make y orthogonal to it, and use the axes as
        # the rotation matrix's columns.
        x_axis, y_axis = vals[:3], vals[3:]
        x_norm = np.linalg.norm(x_axis)
        if x_norm < _MIN_NORM:
            return None
        x_axis /= x_norm
        y_axis -= (y_axis @ x_axis) * x_axis
        y_norm = np.linalg.norm(y_axis)
        if y_norm < _MIN_NORM:
            return None
        y_axis /= y_norm
        mat = np.column_stack((x_axis, y_axis, np.cross(x_axis, y_axis))).ravel()
        # pyrefly: ignore [missing-attribute]
        mujoco.mju_mat2Quat(quat, mat)
    norm = float(np.linalg.norm(quat))
    if not np.isfinite(norm) or norm < _MIN_NORM:
        return None  # zero, or so large that the norm overflows
    return quat / norm


__all__ = ["POLL_INTERVAL_S", "SceneXmlWatcher"]
