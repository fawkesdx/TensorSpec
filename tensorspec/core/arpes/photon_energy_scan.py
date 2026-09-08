"""Photon-energy scan helpers for Chinook/Grizzly ARPES stacking."""
from __future__ import annotations

import numpy as np


def build_hv_list(start: float, finish: float, step: float) -> np.ndarray:
    if step <= 0:
        raise ValueError(f"step must be > 0, got {step}")
    if finish < start:
        raise ValueError(f"finish ({finish}) < start ({start})")
    hv = np.arange(float(start), float(finish) + float(step) / 2.0, float(step))
    if hv.size == 0:
        raise ValueError("hv list empty")
    return hv.astype(float)


def stack_hv_cubes(
    cubes: list[np.ndarray], hv_list: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if len(cubes) == 0:
        raise ValueError("no cubes to stack")
    if len(cubes) != len(hv_list):
        raise ValueError("cubes and hv_list length mismatch")
    stacked = np.stack([np.asarray(c) for c in cubes], axis=0)
    return stacked, np.asarray(hv_list, dtype=float)


def is_dispersion_axes(theta: np.ndarray, phi: np.ndarray, tol: float = 1e-9) -> bool:
    def _deg(a: np.ndarray) -> bool:
        a = np.asarray(a)
        return a.size <= 1 or float(np.ptp(a)) < tol

    return _deg(theta) or _deg(phi)
