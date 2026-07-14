# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest

from physicalai.robot.transport._codec import (
    ROBOT_TRANSPORT_PROTOCOL_VERSION,
    TransportObservation,
    decode_action,
    decode_metadata,
    decode_state,
    encode_action,
    encode_metadata,
    encode_state,
)


class TestStateRoundtrip:
    def test_float32_stays_float32(self) -> None:
        jp = np.arange(6, dtype=np.float32)
        state = np.arange(12, dtype=np.float32)
        blob = encode_state(joint_positions=jp, state=state, timestamp=1.5, sensor_data=None)

        obs = decode_state(blob)

        assert obs.joint_positions.dtype == np.float32
        assert obs.state.dtype == np.float32
        np.testing.assert_array_equal(obs.joint_positions, jp)
        np.testing.assert_array_equal(obs.state, state)
        assert obs.timestamp == 1.5

    def test_shipped_state_returned_not_joint_positions(self) -> None:
        jp = np.zeros(7, dtype=np.float32)
        state = np.ones(14, dtype=np.float32)
        obs = decode_state(encode_state(joint_positions=jp, state=state, timestamp=0.0, sensor_data=None))

        # The owner-computed vector must be shipped as-is (14 != 7).
        assert obs.state.shape == (14,)
        assert obs.joint_positions.shape == (7,)

    def test_sensor_data_roundtrip(self) -> None:
        jp = np.zeros(6, dtype=np.float32)
        sensor = {"velocities": np.arange(6, dtype=np.float32), "efforts": np.arange(6, dtype=np.float64)}
        obs = decode_state(encode_state(joint_positions=jp, state=jp, timestamp=0.0, sensor_data=sensor))

        assert obs.sensor_data is not None
        np.testing.assert_array_equal(obs.sensor_data["velocities"], sensor["velocities"])
        assert obs.sensor_data["efforts"].dtype == np.float64

    def test_sensor_data_none(self) -> None:
        jp = np.zeros(6, dtype=np.float32)
        obs = decode_state(encode_state(joint_positions=jp, state=jp, timestamp=0.0, sensor_data=None))
        assert obs.sensor_data is None

    def test_images_always_none(self) -> None:
        jp = np.zeros(6, dtype=np.float32)
        obs = decode_state(encode_state(joint_positions=jp, state=jp, timestamp=0.0, sensor_data=None))
        assert obs.images is None

    def test_non_contiguous_array(self) -> None:
        wide = np.arange(24, dtype=np.float32).reshape(4, 6)
        jp = wide[:, 0]  # non-contiguous view
        obs = decode_state(encode_state(joint_positions=jp, state=jp, timestamp=0.0, sensor_data=None))
        np.testing.assert_array_equal(obs.joint_positions, np.array([0, 6, 12, 18], dtype=np.float32))


class TestTransportObservation:
    def test_state_falls_back_to_joint_positions(self) -> None:
        jp = np.arange(6, dtype=np.float32)
        obs = TransportObservation(joint_positions=jp, timestamp=0.0)
        np.testing.assert_array_equal(obs.state, jp)

    def test_satisfies_robot_observation_protocol(self) -> None:
        from physicalai.robot.interface import RobotObservation

        obs = TransportObservation(joint_positions=np.zeros(6, dtype=np.float32), timestamp=0.0)
        assert isinstance(obs, RobotObservation)


class TestActionRoundtrip:
    def test_roundtrip(self) -> None:
        action = np.arange(6, dtype=np.float32)
        decoded, goal_time, ts = decode_action(encode_action(action, goal_time=0.25))

        np.testing.assert_array_equal(decoded, action)
        assert decoded.dtype == np.float32
        assert goal_time == 0.25
        assert ts > 0

    def test_float64_preserved(self) -> None:
        action = np.arange(6, dtype=np.float64)
        decoded, _, _ = decode_action(encode_action(action, goal_time=0.1))
        assert decoded.dtype == np.float64


class TestMetadataRoundtrip:
    def test_roundtrip(self) -> None:
        metadata = {
            "protocol_version": ROBOT_TRANSPORT_PROTOCOL_VERSION,
            "name": "left-arm",
            "robot_class": "physicalai.robot.so101.SO101",
            "device_ids": ["serial:ttyUSB0"],
            "host": "myhost",
            "joint_names": ["a", "b"],
            "num_joints": 2,
            "state_dim": 2,
        }
        assert decode_metadata(encode_metadata(metadata)) == metadata

    def test_bad_payload_raises(self) -> None:
        import msgpack

        with pytest.raises(TypeError, match="Expected a dict"):
            decode_metadata(msgpack.packb([1, 2, 3]))
