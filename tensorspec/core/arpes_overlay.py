"""Overlay DFT bands and simulated ARPES intensity onto experimental cuts."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from scipy.interpolate import RegularGridInterpolator


def project_k_component(k_vecs: np.ndarray, component: str | int = "kx") -> np.ndarray:
    """Reduce 3D k-path vectors to a 1D cut coordinate."""
    k = np.asarray(k_vecs, dtype=float)
    if k.ndim != 2 or k.shape[1] < 2:
        raise ValueError("k_vecs must have shape (nk, ≥2).")
    if component in (0, "0", "kx", "x"):
        return k[:, 0]
    if component in (1, "1", "ky", "y"):
        return k[:, 1]
    if component in (2, "2", "kz", "z") and k.shape[1] >= 3:
        return k[:, 2]
    if component in ("inplane", "abs", "kpara", "k∥"):
        return np.sqrt(k[:, 0] ** 2 + k[:, 1] ** 2)
    raise ValueError("component must be kx/ky/kz/inplane.")


def bands_to_polylines(
    k_vecs: np.ndarray,
    eigenvalues: np.ndarray,
    *,
    k_component: str | int = "kx",
    e_fermi: float = 0.0,
    k_offset: float = 0.0,
    band_indices: Optional[Sequence[int]] = None,
    e_min: Optional[float] = None,
    e_max: Optional[float] = None,
    k_min: Optional[float] = None,
    k_max: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Build ImageViewer polylines for each band on a (k, E) cut."""
    k1d = project_k_component(k_vecs, k_component) + float(k_offset)
    ev = np.asarray(eigenvalues, dtype=float)
    if ev.ndim != 2:
        raise ValueError("eigenvalues must be (nk, nbands).")
    if ev.shape[0] != k1d.shape[0]:
        raise ValueError("k_vecs and eigenvalues length mismatch.")
    nb = ev.shape[1]
    if band_indices is None:
        band_indices = list(range(nb))
    else:
        band_indices = [int(i) for i in band_indices if 0 <= int(i) < nb]

    polylines: List[Dict[str, Any]] = []
    for b in band_indices:
        pts = []
        for i in range(len(k1d)):
            kk = float(k1d[i])
            ee = float(ev[i, b] - e_fermi)
            if k_min is not None and kk < k_min:
                continue
            if k_max is not None and kk > k_max:
                continue
            if e_min is not None and ee < e_min:
                continue
            if e_max is not None and ee > e_max:
                continue
            pts.append({"x": kk, "y": ee})
        if len(pts) >= 2:
            polylines.append({"band": int(b), "points": pts})
    return polylines


def _axis_from_tensor(tensor, idx: int) -> np.ndarray:
    return np.asarray(tensor.axes[idx], dtype=float)


def resample_sim_plane(
    sim_tensor,
    *,
    cut_x: np.ndarray,
    cut_y: np.ndarray,
    sim_x_idx: int,
    sim_y_idx: int,
    sim_fixed: Optional[Dict[int, int]] = None,
) -> np.ndarray:
    """
    Resample a simulated spectroscopy tensor onto the experimental cut axes.

    Returns values with shape (len(cut_y), len(cut_x)) matching ImageViewer / extract_slice.
    """
    from tensorspec.core import tensor_ops as ops

    sim_fixed = {int(k): int(v) for k, v in (sim_fixed or {}).items()}
    # Fill missing fixed dims at mid-plane
    for i in range(sim_tensor.value.ndim):
        if i in (sim_x_idx, sim_y_idx) or i in sim_fixed:
            continue
        sim_fixed[i] = int(sim_tensor.value.shape[i] // 2)

    extracted = ops.extract_slice(sim_tensor, sim_x_idx, sim_y_idx, sim_fixed)
    sim_plane = np.asarray(extracted["values"], dtype=float)
    sx = np.asarray(extracted["x_axis"], dtype=float)
    sy = np.asarray(extracted["y_axis"], dtype=float)
    cut_x = np.asarray(cut_x, dtype=float)
    cut_y = np.asarray(cut_y, dtype=float)

    # RegularGridInterpolator wants ascending axes
    sx_order = np.argsort(sx)
    sy_order = np.argsort(sy)
    sx_s = sx[sx_order]
    sy_s = sy[sy_order]
    plane_s = sim_plane[np.ix_(sy_order, sx_order)]

    interp = RegularGridInterpolator(
        (sy_s, sx_s),
        plane_s,
        bounds_error=False,
        fill_value=np.nan,
    )
    yy, xx = np.meshgrid(cut_y, cut_x, indexing="ij")
    pts = np.column_stack([yy.ravel(), xx.ravel()])
    out = interp(pts).reshape(len(cut_y), len(cut_x))
    # Replace NaN (outside sim domain) with 0 so overlay stays quiet
    return np.nan_to_num(out, nan=0.0)


def infer_momentum_energy_axes(labels: Sequence[str]) -> Dict[str, Optional[int]]:
    """Best-effort axis roles for overlay registration."""
    energy = None
    momentum = None
    for i, label in enumerate(labels):
        low = (label or "").lower()
        if energy is None and ("energy" in low or low in ("e", "eb", "e_b", "ω", "omega")):
            energy = i
        if momentum is None and (
            low.startswith("k")
            or "angle" in low
            or "theta" in low
            or "phi" in low
            or "slit" in low
            or "momentum" in low
        ):
            momentum = i
    return {"energy_axis": energy, "momentum_axis": momentum}
