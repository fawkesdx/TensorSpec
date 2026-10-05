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


def _rot(theta_deg: float) -> np.ndarray:
    th = math.radians(theta_deg)
    c, s = math.cos(th), math.sin(th)
    return np.array([[c, -s], [s, c]], dtype=float)


def _residual(ab: np.ndarray, moire: np.ndarray, theta_deg: float) -> float:
    coeffs = (moire @ _rot(theta_deg)) @ np.linalg.inv(ab)
    return float(np.max(np.abs(coeffs - np.round(coeffs))))


def _search_integer_S(ab: np.ndarray, theta_deg: float, max_cells: int) -> dict:
    B = int(math.isqrt(max_cells)) + 1
    best = None
    for i1 in range(-B, B + 1):
        for j1 in range(-B, B + 1):
            for i2 in range(-B, B + 1):
                for j2 in range(-B, B + 1):
                    det = i1 * j2 - j1 * i2
                    if det <= 0 or det > max_cells:
                        continue
                    S = np.array([[i1, j1], [i2, j2]], dtype=int)
                    moire = S.astype(float) @ ab
                    res = _residual(ab, moire, theta_deg)
                    if best is None or res < best[0] - 1e-15 or (
                        abs(res - best[0]) <= 1e-15 and det < best[1]
                    ):
                        best = (res, det, S)
    if best is None or best[0] >= 1e-6:
        shown = best[0] if best else float("inf")
        raise ValueError(
            f"No unstrained commensurate cell for θ={theta_deg}° "
            f"within max_cells={max_cells} (best residual {shown:.3e})."
        )
    return {
        "bravais": classify_bravais(ab),
        "theta_used": float(theta_deg),
        "S": best[2],
        "n_cells": int(best[1]),
        "residual": float(best[0]),
    }


def _pair_snap_result(
    ab: np.ndarray, kind: str, theta_deg: float, th: float, S_pos: np.ndarray
) -> dict:
    n_cells = int(abs(round(np.linalg.det(S_pos))))
    if theta_deg >= 0:
        moire = S_pos.astype(float) @ ab
        return {
            "bravais": kind,
            "theta_used": float(th),
            "S": S_pos,
            "n_cells": n_cells,
            "residual": _residual(ab, moire, th),
        }
    moire = S_pos.astype(float) @ ab
    T = np.round((moire @ _rot(th)) @ np.linalg.inv(ab)).astype(int)
    theta_used = -th
    moire_t = T.astype(float) @ ab
    return {
        "bravais": kind,
        "theta_used": float(theta_used),
        "S": T,
        "n_cells": int(abs(round(np.linalg.det(T)))),
        "residual": _residual(ab, moire_t, theta_used),
    }


def snap_commensurate(ab: np.ndarray, theta_deg: float, max_cells: int = 217) -> dict:
    ab = np.asarray(ab, dtype=float)
    if max_cells < 1:
        raise ValueError("max_cells must be >= 1.")
    kind = classify_bravais(ab)
    if abs(theta_deg) < 1e-8:
        S = np.eye(2, dtype=int)
        return {
            "bravais": kind,
            "theta_used": 0.0,
            "S": S,
            "n_cells": 1,
            "residual": 0.0,
        }
    if kind == "hexagonal":
        best = None
        # m^2 < N <= max_cells => m upper bound
        m_max = int(math.sqrt(max_cells)) + 2
        for m in range(1, m_max + 1):
            for n in range(0, m):
                N = m * m + m * n + n * n
                if N > max_cells or n == 0:
                    continue
                th, S = hexagonal_pair(m, n)
                err = abs(th - abs(theta_deg))
                if best is None or err < best[0] - 1e-12 or (
                    abs(err - best[0]) <= 1e-12 and N < best[1]
                ):
                    best = (err, N, th, S)
        if best is None:
            raise ValueError(
                f"No unstrained commensurate cell for θ={theta_deg}° "
                f"within max_cells={max_cells}."
            )
        return _pair_snap_result(ab, kind, theta_deg, best[2], best[3])
    if kind == "square":
        best = None
        m_max = int(math.sqrt(max_cells)) + 2
        for m in range(2, m_max + 1):
            for n in range(1, m):
                N = m * m + n * n
                if N > max_cells:
                    continue
                th, S = square_pair(m, n)
                err = abs(th - abs(theta_deg))
                if best is None or err < best[0] - 1e-12 or (
                    abs(err - best[0]) <= 1e-12 and N < best[1]
                ):
                    best = (err, N, th, S)
        if best is None:
            raise ValueError(
                f"No unstrained commensurate cell for θ={theta_deg}° "
                f"within max_cells={max_cells}."
            )
        return _pair_snap_result(ab, kind, theta_deg, best[2], best[3])
    return _search_integer_S(ab, theta_deg, max_cells)
