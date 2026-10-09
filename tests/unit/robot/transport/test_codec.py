# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import patch

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
from physicalai.transport._codec import decode_payload as decode_shared_payload
from physicalai.transport._codec import encode_numpy, pack_msgpack, unpack_msgpack


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

    def test_0d_scalar_joint_positions_shape_preserved(self) -> None:
        jp = np.array(1.5, dtype=np.float32)
        obs = decode_state(encode_state(joint_positions=jp, state=jp, timestamp=0.0, sensor_data=None))
        assert obs.joint_positions.shape == ()
        assert float(obs.joint_positions) == 1.5


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

    def test_0d_scalar_action_shape_preserved(self) -> None:
        action = np.array(True, dtype=np.bool_)
        decoded, _, _ = decode_action(encode_action(action, goal_time=0.1))
        assert decoded.shape == ()
        assert bool(decoded) == bool(action)


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

    def test_oversized_payload_rejected_before_unpacking(self) -> None:
        oversized = b"\x00" * (1024 * 1024 + 1)

        with (
            patch("msgpack.unpackb") as unpackb,
            pytest.raises(ValueError, match="exceeds the .* limit"),
        ):
            decode_metadata(oversized)

        unpackb.assert_not_called()

    def test_deeply_nested_payload_rejected(self) -> None:
        import msgpack

        payload: dict[str, object] = {"leaf": 1}
        for _ in range(100):
            payload = {"k": payload}

        with pytest.raises(ValueError, match="nesting exceeds"):
            decode_metadata(msgpack.packb(payload, use_bin_type=True))


class TestSharedNumpyCodec:
    def test_uses_existing_tagged_numpy_map(self) -> None:
        array = np.arange(6, dtype=np.float32).reshape(2, 3)

        assert encode_numpy(array) == {
            "__np__": True,
            "dtype": "float32",
            "shape": [2, 3],
            "data": array.tobytes(),
        }

        encoded = pack_msgpack({"array": array})
        decoded = unpack_msgpack(encoded)
        np.testing.assert_array_equal(decoded["array"], array)

    def test_scalar_array_shape_roundtrips_as_zero_dimensional(self) -> None:
        scalar = np.asarray(3.5, dtype=np.float32)

        decoded = unpack_msgpack(pack_msgpack({"scalar": scalar}))

        assert decoded["scalar"].shape == ()
        assert decoded["scalar"].dtype == np.float32
        assert decoded["scalar"].item() == pytest.approx(3.5)

    def test_rejects_non_numeric_dtype(self) -> None:
        tagged = {"__np__": True, "dtype": "|O", "shape": [1], "data": b"x"}

        with pytest.raises(ValueError, match="numeric or bool"):
            decode_shared_payload(tagged)

        with pytest.raises(ValueError, match="numeric or bool"):
            encode_numpy(np.asarray(["not numeric"]))

    def test_rejects_shape_dtype_data_length_mismatch(self) -> None:
        tagged = {"__np__": True, "dtype": "float32", "shape": [2], "data": b"1234"}

        with pytest.raises(ValueError, match="data length"):
            decode_shared_payload(tagged)

    def test_size_cap_is_checked_before_frombuffer(self) -> None:
        tagged = {"__np__": True, "dtype": "float32", "shape": [2], "data": b"12345678"}
        with (
            patch("physicalai.transport._codec.np.frombuffer") as frombuffer,
            pytest.raises(ValueError, match="limit"),
        ):
            decode_shared_payload(tagged, max_bytes=4)

        frombuffer.assert_not_called()

    def test_nesting_depth_is_bounded(self) -> None:
        nested: object = "value"
        for _ in range(4):
            nested = {"child": nested}

        with pytest.raises(ValueError, match="nesting exceeds the 2-level limit"):
            decode_shared_payload(nested, max_depth=2)
