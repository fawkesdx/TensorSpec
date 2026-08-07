# File: tensorspec/core/tensor_ops.py
"""
Slicing and curve extraction for N-dimensional spectroscopic tensors.

These operations were previously written inline inside the Qt data viewer,
which meant no other front end could reuse them and the export path carried a
second, drifting copy. They live here so any caller -- a Qt widget, the web
service, or a batch script -- gets identical numbers.

Everything in this module is pure: arrays in, arrays out, no rendering, no
colormaps, no toolkit imports.

Conventions
-----------
A 2D slice is returned in row-major display order, meaning ``values[y, x]``:
rows follow the Y axis and columns follow the X axis. Integration widths are
half-widths in index units, so a width of 2 spans 5 samples centred on the
crosshair, matching the ``dx px`` spin boxes in the viewer.
"""
import numpy as np

# Profile aggregation modes.
#   SUM        -- integrate counts over the window (the physical default)
#   MEAN       -- average, for comparing windows of different widths
#   NORMALIZED -- integrate, then scale the peak to 1 for shape comparison
MODE_SUM = "sum"
MODE_MEAN = "mean"
MODE_NORMALIZED = "normalized"
PROFILE_MODES = (MODE_SUM, MODE_MEAN, MODE_NORMALIZED)


def describe_axes(tensor) -> list[dict]:
    """Summarises each dimension so a UI can build axis pickers and sliders."""
    described = []
    for index, axis in enumerate(tensor.axes):
        described.append({
            "index": index,
            "label": tensor.labels[index],
            "unit": tensor.units[index],
            "size": int(len(axis)),
            "min": float(np.min(axis)),
            "max": float(np.max(axis)),
        })
    return described


def coord_to_index(axis, value: float) -> int:
    """Nearest sample index to a physical coordinate."""
    return int(np.abs(np.asarray(axis) - value).argmin())


def axis_extent(axis) -> tuple[float, float]:
    """
    Outer edges of an axis, padded by half a sample step.

    Image renderers place pixel centres on the sample coordinates, so the drawn
    area extends half a step beyond the first and last samples.
    """
    axis = np.asarray(axis)
    if len(axis) > 1:
        step = (axis[-1] - axis[0]) / (len(axis) - 1)
    else:
        step = 0.1
    return float(axis[0] - step / 2), float(axis[-1] + step / 2)


def integration_bounds(center: int, half_width: int, size: int) -> tuple[int, int]:
    """
    Half-open index range covering ``center`` +/- ``half_width``, clipped to the
    array. A half_width of 0 yields a single sample.
    """
    return max(0, center - half_width), min(size, center + half_width + 1)


def extract_slice(tensor, x_idx: int, y_idx: int, fixed: dict) -> dict:
    """
    Cuts a 2D plane out of an N-dimensional tensor.

    `fixed` maps each remaining dimension index to the sample index held
    constant. Dimensions absent from `fixed` fall back to their midpoint.

    Returns the plane as ``values[y, x]`` together with the two coordinate
    axes and the drawing extent.
    """
    if x_idx == y_idx:
        raise ValueError("The X and Y axes must differ.")
    for index in (x_idx, y_idx):
        if not 0 <= index < tensor.ndim:
            raise ValueError(f"Axis {index} is outside this {tensor.ndim}D tensor.")

    selector = []
    for dim in range(tensor.ndim):
        if dim in (x_idx, y_idx):
            selector.append(slice(None))
        else:
            held = fixed.get(dim, len(tensor.axes[dim]) // 2)
            selector.append(int(np.clip(held, 0, len(tensor.axes[dim]) - 1)))

    plane = tensor.value[tuple(selector)]

    # Indexing preserves the tensor's own dimension order, so when X comes
    # first the plane arrives as [x, y] and has to be flipped to [y, x].
    if x_idx < y_idx:
        plane = plane.T

    x_axis = np.asarray(tensor.axes[x_idx])
    y_axis = np.asarray(tensor.axes[y_idx])
    x_lo, x_hi = axis_extent(x_axis)
    y_lo, y_hi = axis_extent(y_axis)

    return {
        "values": plane,
        "x_axis": x_axis,
        "y_axis": y_axis,
        "extent": (x_lo, x_hi, y_lo, y_hi),
        "vmin": float(np.nanmin(plane)),
        "vmax": float(np.nanmax(plane)),
    }


def _aggregate(chunk, axis, mode: str):
    if mode == MODE_MEAN:
        return np.mean(chunk, axis=axis)
    return np.sum(chunk, axis=axis)


def _normalize(values, mode: str):
    if mode != MODE_NORMALIZED:
        return values
    peak = np.nanmax(values)
    return values / peak if peak != 0 else values


def extract_profiles(plane, x_bounds: tuple, y_bounds: tuple, mode: str = MODE_SUM) -> dict:
    """
    Integrates a 2D plane down to its two 1D curves.

    In ARPES terms, with energy on Y the ``y`` curve is the EDC and the ``x``
    curve is the MDC; swapping the displayed axes swaps which is which, so the
    names stay geometric rather than physical.

    `x_bounds` is the column window the Y curve integrates over, and `y_bounds`
    the row window for the X curve.
    """
    if mode not in PROFILE_MODES:
        raise ValueError(f"Unknown profile mode '{mode}'. Expected one of {PROFILE_MODES}.")

    x1, x2 = x_bounds
    y1, y2 = y_bounds

    profile_x = _aggregate(plane[y1:y2, :], axis=0, mode=mode)
    profile_y = _aggregate(plane[:, x1:x2], axis=1, mode=mode)

    return {"x": _normalize(profile_x, mode), "y": _normalize(profile_y, mode)}


def extract_orthogonal_profile(tensor, ortho_idx: int, x_idx: int, y_idx: int,
                               x_bounds: tuple, y_bounds: tuple, fixed: dict,
                               mode: str = MODE_SUM) -> dict:
    """
    Integrates through the displayed plane to profile a third dimension.

    This is how a photon-energy or delay-stage dependence is read out: the
    crosshair window selects a patch of the visible image, and the intensity
    inside that patch is summed for every sample along `ortho_idx`.
    """
    if ortho_idx in (x_idx, y_idx):
        raise ValueError("The orthogonal axis must differ from the displayed axes.")
    if mode not in PROFILE_MODES:
        raise ValueError(f"Unknown profile mode '{mode}'. Expected one of {PROFILE_MODES}.")

    x1, x2 = x_bounds
    y1, y2 = y_bounds

    selector = []
    for dim in range(tensor.ndim):
        if dim == ortho_idx:
            selector.append(slice(None))
        elif dim == x_idx:
            selector.append(slice(x1, x2))
        elif dim == y_idx:
            selector.append(slice(y1, y2))
        else:
            held = int(fixed.get(dim, len(tensor.axes[dim]) // 2))
            selector.append(slice(held, held + 1))

    chunk = tensor.value[tuple(selector)]
    collapse = tuple(dim for dim in range(tensor.ndim) if dim != ortho_idx)
    values = _normalize(_aggregate(chunk, axis=collapse, mode=mode), mode)

    return {"axis": np.asarray(tensor.axes[ortho_idx]), "values": values}


def downsample_plane(plane, x_axis, y_axis, max_points: int = 900):
    """
    Decimates a plane so no dimension exceeds `max_points`.

    A browser cannot resolve more samples than it has pixels, and a detector
    map can be far larger than the panel showing it. Striding keeps the
    coordinate axes aligned with the values they label. Returns the plane
    unchanged when it already fits.
    """
    step_y = max(1, int(np.ceil(plane.shape[0] / max_points)))
    step_x = max(1, int(np.ceil(plane.shape[1] / max_points)))
    if step_y == 1 and step_x == 1:
        return plane, np.asarray(x_axis), np.asarray(y_axis), (1, 1)
    return (
        plane[::step_y, ::step_x],
        np.asarray(x_axis)[::step_x],
        np.asarray(y_axis)[::step_y],
        (step_y, step_x),
    )
