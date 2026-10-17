# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Check the hardware smoke-test interaction without opening a serial port."""

import runpy
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call

import numpy as np
import pytest


@pytest.fixture
def smoke_test(monkeypatch, tmp_path):
    script = Path(__file__).resolve().parents[3] / "examples/runtime/shared_robot_torque_smoke.py"
    main = runpy.run_path(str(script))["main"]
    robot = MagicMock()
    robot.joint_names = ["shoulder_pan"]
    robot.get_observation.side_effect = [
        SimpleNamespace(joint_positions=np.array([0.0]), timestamp=1.0),
        SimpleNamespace(joint_positions=np.array([0.0]), timestamp=2.0),
        SimpleNamespace(joint_positions=np.array([0.0]), timestamp=3.0),
    ]
    monkeypatch.setitem(main.__globals__, "SO101", MagicMock())
    monkeypatch.setitem(main.__globals__, "SharedRobot", SimpleNamespace(from_config=lambda *args, **kwargs: robot))
    calibration = tmp_path / "calibration.json"
    calibration.write_text("{}")
    monkeypatch.setattr("sys.argv", [str(script), "--port", "unused", "--calibration", str(calibration), "--name", "test"])
    return main, robot


@pytest.mark.parametrize("move", [False, True])
def test_waits_for_operator_to_inspect_both_torque_states(smoke_test, monkeypatch, move):
    main, robot = smoke_test
    if move:
        monkeypatch.setattr("sys.argv", [*sys.argv, "--joint", "shoulder_pan", "--delta", "1"])
        robot.get_observation.side_effect = [
            SimpleNamespace(joint_positions=np.array([0.0]), timestamp=1.0),
            SimpleNamespace(joint_positions=np.array([0.0]), timestamp=2.0),
            SimpleNamespace(joint_positions=np.array([1.0]), timestamp=3.0),
        ]
    prompts = []

    def answer(prompt):
        prompts.append(prompt)
        if "CONNECT" in prompt:
            robot.connect.assert_not_called()
            return "CONNECT"
        if "ENABLE" in prompt:
            robot.set_torque.assert_not_called()
            return "ENABLE"
        if "MOVE" in prompt:
            robot.send_action.assert_not_called()
            return "MOVE"
        if "DISABLE" in prompt:
            assert robot.set_torque.call_args_list == [call(enabled=True)]
            robot.disconnect.assert_not_called()
            return "DISABLE"
        robot.set_torque.assert_has_calls([call(enabled=True), call(enabled=False)])
        robot.disconnect.assert_not_called()
        return ""

    monkeypatch.setattr("builtins.input", answer)
    main()

    assert len(prompts) == (5 if move else 4)
    np.testing.assert_array_equal(robot.send_action.call_args.args[0], [1.0 if move else 0.0])
    assert "Torque is OFF" in prompts[-1]
    assert robot.set_torque.call_args_list == [call(enabled=True), call(enabled=False)]
    robot.disconnect.assert_called_once()


@pytest.mark.parametrize("failure", [KeyboardInterrupt, EOFError])
def test_interruption_during_torque_on_inspection_still_disables(smoke_test, monkeypatch, failure):
    main, robot = smoke_test

    def answer(prompt):
        if "CONNECT" in prompt:
            return "CONNECT"
        if "ENABLE" in prompt:
            return "ENABLE"
        raise failure()

    monkeypatch.setattr("builtins.input", answer)
    with pytest.raises(failure):
        main()

    assert robot.set_torque.call_args_list == [call(enabled=True), call(enabled=False)]
    robot.disconnect.assert_called_once()
