from __future__ import annotations

import numpy as np


def roi_to_mask(ny: int, nx: int, roi: dict) -> np.ndarray:
    """
    Return bool mask (ny, nx).
    roi["kind"] in {"rect","ellipse","polygon"}.
    Raises ValueError on invalid/empty.
    """
    if ny < 1 or nx < 1:
        raise ValueError("image dimensions must be positive")

    kind = roi.get("kind")
    if kind == "rect":
        mask = _rect_mask(ny, nx, roi)
    elif kind == "ellipse":
        mask = _ellipse_mask(ny, nx, roi)
    elif kind == "polygon":
        mask = _polygon_mask(ny, nx, roi)
    else:
        raise ValueError(f"unsupported ROI kind: {kind!r}")

    if not mask.any():
        raise ValueError("empty ROI mask")
    return mask


def _rect_mask(ny: int, nx: int, roi: dict) -> np.ndarray:
    try:
        x0, y0, x1, y1 = roi["x0"], roi["y0"], roi["x1"], roi["y1"]
    except KeyError as exc:
        raise ValueError("rect ROI requires x0, y0, x1, y1") from exc

    x_lo, x_hi = (x0, x1) if x0 <= x1 else (x1, x0)
    y_lo, y_hi = (y0, y1) if y0 <= y1 else (y1, y0)

    x_lo = max(0, int(x_lo))
    x_hi = min(nx - 1, int(x_hi))
    y_lo = max(0, int(y_lo))
    y_hi = min(ny - 1, int(y_hi))

    mask = np.zeros((ny, nx), dtype=bool)
    if x_lo <= x_hi and y_lo <= y_hi:
        mask[y_lo : y_hi + 1, x_lo : x_hi + 1] = True
    return mask


def _ellipse_mask(ny: int, nx: int, roi: dict) -> np.ndarray:
    try:
        cx, cy, rx, ry = roi["cx"], roi["cy"], roi["rx"], roi["ry"]
    except KeyError as exc:
        raise ValueError("ellipse ROI requires cx, cy, rx, ry") from exc

    if rx <= 0 or ry <= 0:
        raise ValueError("ellipse radii must be positive")

    ys, xs = np.mgrid[0:ny, 0:nx]
    return (((xs - cx) / rx) ** 2 + ((ys - cy) / ry) ** 2) <= 1.0


def _polygon_mask(ny: int, nx: int, roi: dict) -> np.ndarray:
    points = roi.get("points")
    if points is None:
        raise ValueError("polygon ROI requires points")
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] < 3:
        raise ValueError("polygon requires at least 3 [x, y] points")

    xs = np.arange(nx, dtype=float)
    ys = np.arange(ny, dtype=float)
    grid_x, grid_y = np.meshgrid(xs, ys)
    flat_x = grid_x.ravel()
    flat_y = grid_y.ravel()

    inside = np.zeros(flat_x.shape[0], dtype=bool)
    n = pts.shape[0]
    j = n - 1
    for i in range(n):
        xi, yi = pts[i]
        xj, yj = pts[j]
        if yj == yi:
            j = i
            continue
        y_cross = (yi > flat_y) != (yj > flat_y)
        x_intersect = (xj - xi) * (flat_y - yi) / (yj - yi) + xi
        inside ^= y_cross & (flat_x < x_intersect)
        j = i

    return inside.reshape(ny, nx)
