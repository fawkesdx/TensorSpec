"""Tests for multi-GPU ARPES runner hang/dead-worker recovery."""

import numpy as np
from queue import Queue

from tensorspec.core.arpes.one_step.grizzly_multigpu_collect import (
    apply_multigpu_result_message,
    collect_multigpu_block_results,
    missing_block_indices,
)


def test_missing_block_indices():
    assert missing_block_indices(5, {0, 2, 4}) == [1, 3]
    assert missing_block_indices(3, {0, 1, 2}) == []


def test_apply_ok_and_error_messages():
    cube = np.zeros((4, 2, 2), dtype=np.float32)
    completed = set()
    errors = []
    ok = ("ok", 1, 2, 4, np.ones((2, 2, 2), dtype=np.float32), 1.5)
    apply_multigpu_result_message(
        ok, cube, completed, errors, nphi=2, ne=2, is_oom=None
    )
    assert completed == {1}
    assert cube[2:4].sum() > 0
    assert errors == []

    err = ("error", 0, 0, "RuntimeError: boom")
    apply_multigpu_result_message(
        err, cube, completed, errors, nphi=2, ne=2, is_oom=None
    )
    assert completed == {0, 1}
    assert errors[0][0] == 0


def test_collect_marks_missing_when_workers_die():
    """Parent must not hang forever if a worker dies before sending a block."""

    class DeadProc:
        def is_alive(self):
            return False

        def terminate(self):
            pass

        def join(self, timeout=None):
            pass

        def kill(self):
            pass

    result_q = Queue()
    # Only block 0 reported; block 1 never arrives; workers already dead.
    result_q.put(("ok", 0, 0, 2, np.ones((2, 2, 2), dtype=np.float32), 0.1))

    cube = np.zeros((4, 2, 2), dtype=np.float32)
    blocks = [(0, 0, 2, [0.0, 1.0]), (1, 2, 4, [2.0, 3.0])]
    completed, errors = collect_multigpu_block_results(
        result_q,
        procs=[DeadProc(), DeadProc()],
        cube=cube,
        blocks=blocks,
        nphi=2,
        ne=2,
        poll_s=0.05,
    )
    assert completed == {0, 1}
    assert any(e[0] == 1 for e in errors)
    assert "without result" in errors[0][2].lower() or "died" in errors[0][2].lower()
