"""Miller-plane helpers for the Crystal Suite cut-plane guide.

Mirrors the browser formulas in viewer_3d.js so orientation/depth stay
consistent with crystallography convention (n = h a* + k b* + l c*).
"""
from __future__ import annotations

import numpy as np


def miller_normal(cell_matrix: np.ndarray, hkl: tuple[int, int, int]) -> np.ndarray:
    h, k, l = (int(hkl[0]), int(hkl[1]), int(hkl[2]))
    if h == 0 and k == 0 and l == 0:
        raise ValueError("Miller index (0,0,0) is undefined")
    cell = np.asarray(cell_matrix, dtype=float).reshape(3, 3)
    a, b, c = cell[0], cell[1], cell[2]
    a_star = np.cross(b, c)
    b_star = np.cross(c, a)
    c_star = np.cross(a, b)
    n = h * a_star + k * b_star + l * c_star
    norm = np.linalg.norm(n)
    if norm < 1e-12:
        raise ValueError("Miller normal has zero length")
    return n / norm


def plane_offset(cell_matrix: np.ndarray, normal: np.ndarray, depth_frac: float) -> np.ndarray:
    """Offset from cell center along normal; depth_frac in [-1, 1].

    Matches the viewer frame where atoms are drawn relative to geometry.center.
    depth_frac=0 → plane through cell center; ±1 → cell AABB faces along n.
    """
    cell = np.asarray(cell_matrix, dtype=float).reshape(3, 3)
    n = np.asarray(normal, dtype=float)
    n = n / np.linalg.norm(n)
    a, b, c = cell[0], cell[1], cell[2]
    corners = [
        np.zeros(3),
        a, b, c,
        a + b, a + c, b + c,
        a + b + c,
    ]
    center = sum(corners) / 8.0
    projs = [float(np.dot(p - center, n)) for p in corners]
    half = 0.5 * (max(projs) - min(projs))
    frac = float(np.clip(depth_frac, -1.0, 1.0))
    return n * (frac * half)


def plane_size(cell_matrix: np.ndarray) -> float:
    cell = np.asarray(cell_matrix, dtype=float).reshape(3, 3)
    a, b, c = cell[0], cell[1], cell[2]
    face_diags = [
        np.linalg.norm(a + b),
        np.linalg.norm(a + c),
        np.linalg.norm(b + c),
    ]
    return 1.2 * max(face_diags)
