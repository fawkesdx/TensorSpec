"""Slit calibration and physical-grid resampling for SSL samples."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from tensorspec.core.ml.ssl.spec import ResampleSpec

DEG_PER_RAW_PX = {
    ("R4000", "Angular30"): 30.0 / 1260.0,
}


def slit_axis_degrees(
    n_pixels: int,
    scale_offset: float,
    scale_delta: float,
    deg_per_raw_px: float,
    centre_px: float | None = None,
) -> np.ndarray:
    """Convert a slit axis expressed in raw-pixel coordinates to degrees."""
    raw_px = scale_offset + scale_delta * np.arange(n_pixels, dtype=np.float64)
    if centre_px is None:
        centre_px = 0.5 * (raw_px[0] + raw_px[-1])
    return (raw_px - centre_px) * deg_per_raw_px


def _ascending_grid(
    sample: np.ndarray,
    axes: tuple[np.ndarray, ...],
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    values = np.asarray(sample)
    if values.ndim != len(axes):
        raise ValueError(
            f"sample has {values.ndim} dimensions; expected {len(axes)}"
        )

    ordered_axes: list[np.ndarray] = []
    for dimension, axis in enumerate(axes):
        coordinates = np.asarray(axis, dtype=np.float64)
        if coordinates.ndim != 1 or coordinates.size != values.shape[dimension]:
            raise ValueError(
                f"axis {dimension} shape {coordinates.shape} does not match "
                f"sample dimension {values.shape[dimension]}"
            )
        differences = np.diff(coordinates)
        if np.all(differences < 0.0):
            coordinates = coordinates[::-1]
            values = np.flip(values, axis=dimension)
        elif not np.all(differences > 0.0):
            raise ValueError(f"axis {dimension} must be strictly monotonic")
        ordered_axes.append(coordinates)

    return values, tuple(ordered_axes)


def _resample(
    sample: np.ndarray,
    axes: tuple[np.ndarray, ...],
    target_sizes: tuple[int, ...],
) -> np.ndarray:
    values, source_axes = _ascending_grid(sample, axes)
    target_axes = tuple(
        np.linspace(axis[0], axis[-1], size, dtype=np.float64)
        for axis, size in zip(source_axes, target_sizes)
    )
    target_grid = np.meshgrid(*target_axes, indexing="ij")
    target_points = np.stack(target_grid, axis=-1)
    interpolator = RegularGridInterpolator(
        source_axes,
        values,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )
    return np.asarray(interpolator(target_points), dtype=np.float32)


def resample_disp2d(
    sample: np.ndarray,
    energy_axis: np.ndarray,
    slit_axis_deg: np.ndarray,
    spec: ResampleSpec,
) -> np.ndarray:
    """Resample an ``(energy, slit)`` sample to a fixed physical grid."""
    return _resample(
        sample,
        (energy_axis, slit_axis_deg),
        (spec.energy_size, spec.slit_size),
    )


def resample_fermi3d(
    sample: np.ndarray,
    defl_axis: np.ndarray,
    energy_axis: np.ndarray,
    slit_axis_deg: np.ndarray,
    spec: ResampleSpec,
) -> np.ndarray:
    """Resample a ``(defl, energy, slit)`` sample to a fixed physical grid."""
    return _resample(
        sample,
        (defl_axis, energy_axis, slit_axis_deg),
        (spec.defl_size, spec.energy_size, spec.slit_size),
    )
