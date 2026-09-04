from __future__ import annotations

import numpy as np

_VALID_KINDS = frozenset({"rect", "ellipse", "polygon"})


def roi_to_mask(ny: int, nx: int, roi: dict) -> np.ndarray:
    """
    Return bool mask (ny, nx).
    roi["kind"] in {"rect","ellipse","polygon"}.
    Raises ValueError on invalid/empty.
    """
    if ny < 1 or nx < 1:
        raise ValueError("image dimensions must be positive")

    if not isinstance(roi, dict):
        raise ValueError("roi must be a dict")

    kind = roi.get("kind")
    if not _kind_is_valid(kind):
        raise ValueError(f"unsupported ROI kind: {kind!r}")

    if kind == "rect":
        mask = _rect_mask(ny, nx, roi)
    elif kind == "ellipse":
        mask = _ellipse_mask(ny, nx, roi)
    else:
        mask = _polygon_mask(ny, nx, roi)

    if not mask.any():
        raise ValueError("empty ROI mask")
    return mask


def _kind_is_valid(kind: object) -> bool:
    try:
        return kind in _VALID_KINDS
    except TypeError:
        return False


def _coerce_float(value: object, field: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if not np.isfinite(out):
        raise ValueError(f"{field} must be numeric")
    return out


def _coerce_int(value: object, field: str) -> int:
    try:
        out = int(_coerce_float(value, field))
    except ValueError as exc:
        if f"{field} must be numeric" in str(exc):
            raise
        raise ValueError(f"{field} must be numeric") from exc
    return out


def _require_fields(roi: dict, fields: tuple[str, ...], label: str) -> None:
    missing = [name for name in fields if name not in roi]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"{label} ROI requires {joined}")


def _rect_mask(ny: int, nx: int, roi: dict) -> np.ndarray:
    _require_fields(roi, ("x0", "y0", "x1", "y1"), "rect")
    x0 = _coerce_int(roi["x0"], "x0")
    y0 = _coerce_int(roi["y0"], "y0")
    x1 = _coerce_int(roi["x1"], "x1")
    y1 = _coerce_int(roi["y1"], "y1")

    x_lo, x_hi = (x0, x1) if x0 <= x1 else (x1, x0)
    y_lo, y_hi = (y0, y1) if y0 <= y1 else (y1, y0)

    x_lo = max(0, x_lo)
    x_hi = min(nx - 1, x_hi)
    y_lo = max(0, y_lo)
    y_hi = min(ny - 1, y_hi)

    mask = np.zeros((ny, nx), dtype=bool)
    if x_lo <= x_hi and y_lo <= y_hi:
        mask[y_lo : y_hi + 1, x_lo : x_hi + 1] = True
    return mask


def _ellipse_mask(ny: int, nx: int, roi: dict) -> np.ndarray:
    _require_fields(roi, ("cx", "cy", "rx", "ry"), "ellipse")
    cx = _coerce_float(roi["cx"], "cx")
    cy = _coerce_float(roi["cy"], "cy")
    rx = _coerce_float(roi["rx"], "rx")
    ry = _coerce_float(roi["ry"], "ry")

    if rx <= 0 or ry <= 0:
        raise ValueError("ellipse radii must be positive")

    ys, xs = np.mgrid[0:ny, 0:nx]
    return (((xs - cx) / rx) ** 2 + ((ys - cy) / ry) ** 2) <= 1.0


def _polygon_mask(ny: int, nx: int, roi: dict) -> np.ndarray:
    if "points" not in roi:
        raise ValueError("polygon ROI requires points")

    points = roi["points"]
    if not isinstance(points, (list, tuple, np.ndarray)):
        raise ValueError("polygon points must be a sequence")

    try:
        pts = np.asarray(points, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("polygon points must be numeric") from exc

    if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] < 3:
        raise ValueError("polygon requires at least 3 [x, y] points")
    if not np.all(np.isfinite(pts)):
        raise ValueError("polygon points must be numeric")

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
