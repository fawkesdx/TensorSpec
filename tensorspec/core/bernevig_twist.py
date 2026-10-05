"""Jiang et al., Science 393, eadu1550 (2026) homobilayer twist."""
from __future__ import annotations

import math

import numpy as np

_Q = {
    ("hexagonal", "Γ"): "triangular",
    ("hexagonal", "K"): "honeycomb",
    ("hexagonal", "M"): "kagome",
    ("hexagonal", "nHSP"): "nHSP",
    ("square", "Γ"): "square",
    ("square", "M"): "square",
    ("square", "X"): "nested square",
    ("square", "nHSP"): "nHSP",
    ("rectangular", "HSP"): "rectangular",
    ("rectangular", "nHSP"): "nHSP",
    ("oblique", "HSP"): "oblique",
    ("oblique", "nHSP"): "nHSP",
}


def classify_bravais(ab: np.ndarray) -> str:
    ab = np.asarray(ab, dtype=float)
    a1, a2 = ab[0], ab[1]
    la, lb = float(np.linalg.norm(a1)), float(np.linalg.norm(a2))
    mean = 0.5 * (la + lb)
    rel = abs(la - lb) / mean if mean > 0 else 1.0
    cosang = float(np.dot(a1, a2) / (la * lb))
    cosang = max(-1.0, min(1.0, cosang))
    ang = math.degrees(math.acos(cosang))
    if rel < 1e-3 and min(abs(ang - 60.0), abs(ang - 120.0)) < 1.0:
        return "hexagonal"
    if rel < 1e-3 and abs(ang - 90.0) < 1.0:
        return "square"
    if rel >= 1e-3 and abs(ang - 90.0) < 1.0:
        return "rectangular"
    return "oblique"


def q_lattice_label(bravais: str, valley: str) -> str:
    key = (bravais, valley)
    if key not in _Q:
        raise ValueError(f"No Q lattice for bravais={bravais!r} valley={valley!r}.")
    return _Q[key]


def hexagonal_pair(m: int, n: int) -> tuple[float, np.ndarray]:
    if m <= n or n < 0:
        raise ValueError("hexagonal pair needs m > n >= 0.")
    N = m * m + m * n + n * n
    cos_th = (m * m + n * n + 4 * m * n) / (2.0 * N)
    theta = math.degrees(math.acos(cos_th))
    S = np.array([[m, -n], [n, m + n]], dtype=int)
    return theta, S


def square_pair(m: int, n: int) -> tuple[float, np.ndarray]:
    if m <= n or n <= 0:
        raise ValueError("square pair needs m > n > 0.")
    theta = math.degrees(2.0 * math.atan2(n, m))
    S = np.array([[m, n], [-n, m]], dtype=int)
    return theta, S
