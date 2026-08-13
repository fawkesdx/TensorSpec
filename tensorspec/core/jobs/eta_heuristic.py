"""Order-of-magnitude wall-time guesses for DFT Queue and ARPES Simulate.

Mirrors ``tensorspec/web/static/js/job_timer.js``. Labels in the UI must stay
``heuristic`` / ``last run`` — these numbers are not countdowns.
"""

from __future__ import annotations


def estimate_dft_seconds(
    backend: str,
    nbnd: int,
    kx: int,
    ky: int,
    kz: int,
    soc: bool,
    ranks: int,
) -> int:
    base = 20 * 60 if backend == "einstein_ssh" else 60 * 60
    seconds = (
        base
        * (nbnd / 162)
        * ((kx * ky * kz) / 216)
        * (2 if soc else 1)
        * (8 / max(ranks, 1))
    )
    return int(max(120, min(12 * 3600, seconds)))


def estimate_arpes_seconds(n_energy: int, n_kx: int, n_ky: int) -> int:
    voxels = max(1, n_energy * n_kx * n_ky)
    seconds = 180 * (voxels / (48 * 64 * 64))
    return int(max(30, min(2 * 3600, seconds)))


def format_elapsed(seconds: float | int) -> str:
    total = max(0, int(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def format_estimate(seconds: float | int) -> str:
    total = max(0, int(seconds))
    if total >= 90:
        minutes = max(1, round(total / 60))
        return f"~{minutes} min"
    return f"~{total} s"
