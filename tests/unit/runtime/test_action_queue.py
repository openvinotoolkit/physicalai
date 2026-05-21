from __future__ import annotations

import threading

import numpy as np

from physicalai.runtime._action_queue import ActionQueue
from physicalai.runtime.smoothers import LerpSmoother, ReplaceSmoother


class TestActionQueue:
    def test_push_pop_roundtrip(self) -> None:
        queue = ActionQueue()
        chunk = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        queue.push_chunk(chunk)

        actions = [queue.pop() for _ in range(3)]
        assert all(a is not None for a in actions)
        for i, action in enumerate(actions):
            np.testing.assert_array_equal(action, chunk[i])

    def test_pop_empty_returns_none(self) -> None:
        queue = ActionQueue()
        assert queue.pop() is None

    def test_consecutive_holds_increment_and_reset(self) -> None:
        queue = ActionQueue()
        queue.pop()
        queue.pop()
        assert queue.consecutive_holds == 2

        queue.push_chunk(np.array([[1.0, 2.0]], dtype=np.float32))
        queue.pop()
        assert queue.consecutive_holds == 0

    def test_total_counters(self) -> None:
        queue = ActionQueue()
        queue.pop()
        queue.pop()
        queue.push_chunk(np.array([[1.0], [2.0], [3.0]], dtype=np.float32))
        queue.pop()
        queue.pop()
        queue.pop()
        queue.pop()

        assert queue.total_holds == 3
        assert queue.total_pops == 3

    def test_remaining_property(self) -> None:
        queue = ActionQueue()
        assert queue.remaining == 0

        queue.push_chunk(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
        assert queue.remaining == 2

        queue.pop()
        assert queue.remaining == 1

    def test_below_threshold(self) -> None:
        queue = ActionQueue()
        assert queue.below_threshold(1) is True

        queue.push_chunk(np.array([[1.0], [2.0], [3.0]], dtype=np.float32))
        assert queue.below_threshold(4) is True
        assert queue.below_threshold(3) is False
        assert queue.below_threshold(2) is False

    def test_clear(self) -> None:
        queue = ActionQueue()
        queue.push_chunk(np.array([[1.0], [2.0]], dtype=np.float32))
        queue.pop()
        queue.pop()
        queue.pop()
        assert queue.consecutive_holds == 1

        queue.push_chunk(np.array([[3.0], [4.0]], dtype=np.float32))
        queue.clear()

        assert queue.remaining == 0
        assert queue.consecutive_holds == 0
        assert queue.total_holds == 1
        assert queue.total_pops == 2

    def test_push_with_offset(self) -> None:
        queue = ActionQueue()
        chunk = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        queue.push_chunk(chunk, offset=2)

        assert queue.remaining == 1
        action = queue.pop()
        assert action is not None
        np.testing.assert_array_equal(action, [5.0, 6.0])

    def test_default_smoother_is_replace(self) -> None:
        queue = ActionQueue()
        assert isinstance(queue._smoother, ReplaceSmoother)

    def test_smoother_integration_lerp(self) -> None:
        queue = ActionQueue(smoother=LerpSmoother(duration_frames=5))
        first = np.array([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]], dtype=np.float32)
        queue.push_chunk(first)

        second = np.array(
            [[100.0, 100.0], [110.0, 110.0], [120.0, 120.0], [130.0, 130.0]],
            dtype=np.float32,
        )
        queue.push_chunk(second)

        first_action = queue.pop()
        assert first_action is not None
        assert not np.array_equal(first_action, second[0]), "LerpSmoother should blend, not replace"

    def test_smoother_integration_replace(self) -> None:
        queue = ActionQueue(smoother=ReplaceSmoother())
        first = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
        queue.push_chunk(first)

        second = np.array([[10.0], [20.0]], dtype=np.float32)
        queue.push_chunk(second)

        assert queue.remaining == 2
        action = queue.pop()
        assert action is not None
        np.testing.assert_array_equal(action, [10.0])


class TestActionQueueThreadSafety:
    def test_concurrent_push_pop(self) -> None:
        queue = ActionQueue()
        errors: list[Exception] = []
        action_dim = 4
        n_pushes = 100
        chunk_size = 10

        def pusher() -> None:
            try:
                for i in range(n_pushes):
                    chunk = np.full((chunk_size, action_dim), float(i), dtype=np.float32)
                    queue.push_chunk(chunk)
            except Exception as exc:
                errors.append(exc)

        def popper(stop_event: threading.Event) -> None:
            try:
                while not stop_event.is_set():
                    queue.pop()
            except Exception as exc:
                errors.append(exc)

        stop = threading.Event()
        push_thread = threading.Thread(target=pusher)
        pop_thread = threading.Thread(target=popper, args=(stop,))

        push_thread.start()
        pop_thread.start()

        push_thread.join(timeout=5.0)
        stop.set()
        pop_thread.join(timeout=5.0)

        assert not push_thread.is_alive(), "Push thread deadlocked"
        assert not pop_thread.is_alive(), "Pop thread deadlocked"
        assert not errors, f"Thread errors: {errors}"
