"""Collect multi-GPU θ-block results without hanging if a worker dies.

Used by ``chinook_remote_runner`` (FULL layout, ngpus>1). Kept Chinook-free
so unit tests can import it on the Mac GUI venv.
"""

from __future__ import annotations

from queue import Empty
from typing import Any, Callable, List, Optional, Sequence, Set, Tuple

import numpy as np


def missing_block_indices(n_blocks: int, completed: Set[int]) -> List[int]:
    return [i for i in range(n_blocks) if i not in completed]


def normalize_full_cube(Ig, ntheta_chunk: int, nphi: int, ne: int) -> np.ndarray:
    arr = np.asarray(Ig, dtype=np.float32)
    if arr.ndim == 3 and arr.shape == (ntheta_chunk, nphi, ne):
        return arr
    if arr.size == ntheta_chunk * nphi * ne:
        return arr.reshape(ntheta_chunk, nphi, ne)
    raise ValueError(
        f"Unexpected intensity shape {arr.shape}; "
        f"expected ({ntheta_chunk}, {nphi}, {ne})"
    )


def apply_multigpu_result_message(
    msg: tuple,
    cube: np.ndarray,
    completed: Set[int],
    errors: List[Tuple[int, int, str]],
    *,
    nphi: int,
    ne: int,
    is_oom: Optional[Callable[[BaseException], bool]] = None,
) -> None:
    """Mutate cube/completed/errors from one worker message. May raise on OOM."""
    if msg[0] == "error":
        _, bi, i0, err = msg
        completed.add(int(bi))
        errors.append((int(bi), int(i0), str(err)))
        print(f"  GPU block {bi} idx[{i0}:?] ERROR: {str(err)[:200]}", flush=True)
        if is_oom is not None and is_oom(RuntimeError(err)):
            raise RuntimeError(err)
        return

    _, bi, i0, i1, Ig, wall = msg
    bi, i0, i1 = int(bi), int(i0), int(i1)
    cube[i0:i1, :, :] = normalize_full_cube(Ig, i1 - i0, nphi, ne)
    completed.add(bi)
    print(f"  block {bi + 1} idx[{i0}:{i1}] wall={float(wall):.2f}s", flush=True)


def collect_multigpu_block_results(
    result_q,
    procs: Sequence[Any],
    cube: np.ndarray,
    blocks: Sequence[Tuple[int, int, int, Any]],
    *,
    nphi: int,
    ne: int,
    poll_s: float = 60.0,
    is_oom: Optional[Callable[[BaseException], bool]] = None,
    terminate_procs: bool = True,
) -> Tuple[Set[int], List[Tuple[int, int, str]]]:
    """Drain result queue until all θ-blocks accounted for or workers are dead.

    If every worker process is dead and the queue is empty while blocks remain,
    mark missing blocks as errors (zeros left in ``cube``) instead of hanging
    forever on ``Queue.get()``.
    """
    n_blocks = len(blocks)
    completed: Set[int] = set()
    errors: List[Tuple[int, int, str]] = []

    def _drain_nowait() -> int:
        n = 0
        while True:
            try:
                msg = result_q.get_nowait()
            except Empty:
                return n
            apply_multigpu_result_message(
                msg, cube, completed, errors, nphi=nphi, ne=ne, is_oom=is_oom
            )
            n += 1

    try:
        while len(completed) < n_blocks:
            try:
                msg = result_q.get(timeout=max(0.01, float(poll_s)))
            except Empty:
                alive = any(bool(getattr(p, "is_alive", lambda: False)()) for p in procs)
                _drain_nowait()
                if len(completed) >= n_blocks:
                    break
                if alive:
                    print(
                        f"  waiting for θ-blocks… {len(completed)}/{n_blocks} done "
                        f"(workers still alive)",
                        flush=True,
                    )
                    continue
                missing = missing_block_indices(n_blocks, completed)
                for bi in missing:
                    i0 = int(blocks[bi][1])
                    err = (
                        f"GPU worker died without result for θ-block {bi} "
                        f"idx[{i0}:?] (avoid infinite Queue.get hang)"
                    )
                    errors.append((bi, i0, err))
                    completed.add(bi)
                    print(f"  GPU block {bi} ERROR: {err}", flush=True)
                break

            apply_multigpu_result_message(
                msg, cube, completed, errors, nphi=nphi, ne=ne, is_oom=is_oom
            )
    finally:
        if terminate_procs:
            for p in procs:
                try:
                    if getattr(p, "is_alive", lambda: False)():
                        p.terminate()
                except Exception:
                    pass
            for p in procs:
                try:
                    p.join(timeout=5)
                except Exception:
                    pass
                try:
                    if getattr(p, "is_alive", lambda: False)():
                        p.kill()
                except Exception:
                    pass

    return completed, errors
