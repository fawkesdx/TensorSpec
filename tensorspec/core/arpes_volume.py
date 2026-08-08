"""BZ-shaped ARPES volume extraction for the 3D cutout viewer.

Builds a downsampled intensity cube plus a 2D Brillouin-zone polygon (rectangle
or hexagon) that the browser extrudes into a prism with optional indentations.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from tensorspec.core import arpes_process


def downsample_volume(
    volume: np.ndarray,
    axes: Sequence[np.ndarray],
    max_per_axis: int = 64,
) -> Tuple[np.ndarray, List[np.ndarray], Tuple[int, int, int]]:
    """Stride a 3D array so each axis is at most ``max_per_axis`` samples."""
    volume = np.asarray(volume, dtype=float)
    if volume.ndim != 3:
        raise ValueError("volume must be 3D.")
    steps = []
    out_axes = []
    slices = []
    for i in range(3):
        n = volume.shape[i]
        step = max(1, int(np.ceil(n / max_per_axis)))
        steps.append(step)
        slices.append(slice(None, None, step))
        out_axes.append(np.asarray(axes[i], dtype=float)[::step])
    return volume[tuple(slices)], out_axes, tuple(steps)


def extract_volume(
    tensor,
    *,
    x_idx: int,
    y_idx: int,
    z_idx: int,
    fixed: Optional[Dict[int, int]] = None,
    max_per_axis: int = 64,
) -> Dict[str, Any]:
    """
    Pull a 3D intensity block ``I[z, y, x]`` (display order matching 2D slices).

    ``x``/``y`` are in-plane (k or angle); ``z`` is typically energy or kz.
    """
    if tensor.ndim < 3:
        raise ValueError("Need at least 3D data for a volume view (e.g. kx × ky × E).")
    axes_used = {x_idx, y_idx, z_idx}
    if len(axes_used) != 3:
        raise ValueError("x_idx, y_idx, and z_idx must be distinct.")
    for idx in axes_used:
        if not 0 <= idx < tensor.ndim:
            raise ValueError(f"Axis {idx} out of range for {tensor.ndim}D tensor.")

    fixed = {int(k): int(v) for k, v in (fixed or {}).items()}
    selectors: List[Any] = []
    for i in range(tensor.ndim):
        if i in axes_used:
            selectors.append(slice(None))
        else:
            selectors.append(fixed.get(i, tensor.value.shape[i] // 2))

    block = np.asarray(tensor.value[tuple(selectors)], dtype=float)
    # After fancy indexing, free axes keep their relative order in the tensor.
    free = [i for i in range(tensor.ndim) if i in axes_used]
    # Map to (z, y, x)
    order = [free.index(z_idx), free.index(y_idx), free.index(x_idx)]
    volume = np.transpose(block, axes=order)
    ax = [
        np.asarray(tensor.axes[z_idx], dtype=float),
        np.asarray(tensor.axes[y_idx], dtype=float),
        np.asarray(tensor.axes[x_idx], dtype=float),
    ]
    volume, ax, steps = downsample_volume(volume, ax, max_per_axis=max_per_axis)
    finite = volume[np.isfinite(volume)]
    return {
        "values": volume,
        "z_axis": ax[0],
        "y_axis": ax[1],
        "x_axis": ax[2],
        "shape": [int(n) for n in volume.shape],
        "steps": [int(s) for s in steps],
        "vmin": float(finite.min()) if finite.size else 0.0,
        "vmax": float(finite.max()) if finite.size else 1.0,
        "x_label": tensor.labels[x_idx],
        "y_label": tensor.labels[y_idx],
        "z_label": tensor.labels[z_idx],
        "x_unit": tensor.units[x_idx],
        "y_unit": tensor.units[y_idx],
        "z_unit": tensor.units[z_idx],
        "x_idx": int(x_idx),
        "y_idx": int(y_idx),
        "z_idx": int(z_idx),
    }


def bounding_rectangle_polygon(x_axis: np.ndarray, y_axis: np.ndarray) -> Dict[str, Any]:
    """Closed rectangle from data extents (fallback when no crystal BZ)."""
    x0, x1 = float(np.min(x_axis)), float(np.max(x_axis))
    y0, y1 = float(np.min(y_axis)), float(np.max(y_axis))
    kx = [x0, x1, x1, x0, x0]
    ky = [y0, y0, y1, y1, y0]
    return {"kx": kx, "ky": ky, "n_vertices": 4, "shape": "rectangle", "source": "data_extent"}


def regular_polygon(
    n: int,
    *,
    radius: float,
    center: Tuple[float, float] = (0.0, 0.0),
    rotation: float = 0.0,
) -> Dict[str, Any]:
    """Closed regular n-gon in the kx–ky plane."""
    n = max(3, int(n))
    angles = rotation + np.linspace(0, 2 * np.pi, n, endpoint=False)
    cx, cy = center
    kx = list(cx + radius * np.cos(angles))
    ky = list(cy + radius * np.sin(angles))
    kx.append(kx[0])
    ky.append(ky[0])
    shape = "hexagon" if n == 6 else ("rectangle" if n == 4 else f"{n}-gon")
    return {"kx": kx, "ky": ky, "n_vertices": n, "shape": shape, "source": "regular"}


def infer_volume_axes(tensor) -> Dict[str, Optional[int]]:
    """Pick default (x,y,z) = (momentum/angles, energy) for a volume view."""
    roles = arpes_process.infer_axis_roles(tensor)
    energy = roles.get("energy_axis")
    angle = roles.get("angle_axis")
    beta = roles.get("beta_axis")
    photon = roles.get("photon_axis")

    x_idx = angle if angle is not None else 0
    if beta is not None and beta != x_idx:
        y_idx = beta
    elif photon is not None and photon != x_idx and photon != energy:
        y_idx = photon
    else:
        # Second non-energy axis
        y_idx = None
        for i in range(tensor.ndim):
            if i not in (x_idx, energy):
                y_idx = i
                break
        if y_idx is None:
            y_idx = 1 if x_idx != 1 else 0

    z_idx = energy if energy is not None else (
        next(i for i in range(tensor.ndim) if i not in (x_idx, y_idx))
    )
    return {"x_idx": x_idx, "y_idx": y_idx, "z_idx": z_idx, **roles}


def build_prism_spec(
    *,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    structure=None,
    h: int = 0,
    k: int = 0,
    l: int = 1,
    shape_mode: str = "auto",
) -> Dict[str, Any]:
    """
    2D BZ footprint for extrusion.

    ``shape_mode``: ``auto`` | ``rectangle`` | ``hexagon`` | ``crystal``.
    """
    mode = (shape_mode or "auto").lower().strip()
    x_axis = np.asarray(x_axis, dtype=float)
    y_axis = np.asarray(y_axis, dtype=float)
    rx = 0.5 * (float(np.max(x_axis)) - float(np.min(x_axis)))
    ry = 0.5 * (float(np.max(y_axis)) - float(np.min(y_axis)))
    radius = max(rx, ry, 1e-6)
    cx = 0.5 * (float(np.min(x_axis)) + float(np.max(x_axis)))
    cy = 0.5 * (float(np.min(y_axis)) + float(np.max(y_axis)))

    if mode == "rectangle":
        return bounding_rectangle_polygon(x_axis, y_axis)
    if mode == "hexagon":
        return regular_polygon(6, radius=radius, center=(cx, cy), rotation=np.pi / 6)

    if structure is not None and mode in ("auto", "crystal"):
        poly = arpes_process.surface_bz_polygon_2d(structure, h=h, k=k, l=l)
        if poly and poly.get("n_vertices", 0) >= 3:
            n = int(poly["n_vertices"])
            shape = "hexagon" if n >= 5 else "rectangle"
            return {
                "kx": poly["kx"],
                "ky": poly["ky"],
                "n_vertices": n,
                "shape": shape,
                "source": "crystal",
                "hkl": poly.get("hkl"),
            }

    return bounding_rectangle_polygon(x_axis, y_axis)
