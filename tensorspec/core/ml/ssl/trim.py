"""Apply TrimSpec in physical units via axis coordinate arrays."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tensorspec.core.ml.ssl.resolve import role_for_label
from tensorspec.core.ml.ssl.spec import AxisRole, TrimSpec


@dataclass
class TrimResult:
    slices: dict[AxisRole, slice]
    warnings: list[str]
    out_shape: tuple[int, ...]


def _axis_extent(axis: np.ndarray) -> tuple[float, float]:
    return float(np.min(axis)), float(np.max(axis))


def _slice_length(size: int, s: slice) -> int:
    start, stop, step = s.indices(size)
    if step > 0:
        return max(0, (stop - start + step - 1) // step)
    return max(0, (start - stop - step - 1) // (-step))


def _slice_for_range(
    axis: np.ndarray,
    lo: float,
    hi: float,
    role: AxisRole,
    warnings: list[str],
) -> slice:
    axis_lo, axis_hi = _axis_extent(axis)
    clamped_lo, clamped_hi = lo, hi
    if lo < axis_lo:
        clamped_lo = axis_lo
        warnings.append(
            f"trim {role!r} lo {lo} clamped to axis minimum {axis_lo}"
        )
    if hi > axis_hi:
        clamped_hi = axis_hi
        warnings.append(
            f"trim {role!r} hi {hi} clamped to axis maximum {axis_hi}"
        )
    if clamped_lo > clamped_hi:
        warnings.append(
            f"trim {role!r} range ({lo}, {hi}) empty after clamp; keeping full axis"
        )
        return slice(None)

    mask = (axis >= clamped_lo) & (axis <= clamped_hi)
    if not np.any(mask):
        warnings.append(
            f"trim {role!r} range ({clamped_lo}, {clamped_hi}) matches no indices; keeping full axis"
        )
        return slice(None)

    indices = np.flatnonzero(mask)
    return slice(int(indices[0]), int(indices[-1]) + 1)


def apply_trim(
    labels: list[str],
    axes: list[np.ndarray],
    shape: tuple[int, ...],
    spec: TrimSpec,
) -> TrimResult:
    """Convert physical ranges to index slices; clamp with warnings."""
    if len(labels) != len(axes) or len(labels) != len(shape):
        msg = (
            f"labels ({len(labels)}), axes ({len(axes)}), "
            f"and shape ({len(shape)}) must have equal length"
        )
        raise ValueError(msg)

    warnings: list[str] = []
    role_slices: dict[AxisRole, slice] = {}

    for label, axis in zip(labels, axes):
        role = role_for_label(label)
        if role in role_slices:
            continue
        if role not in spec.ranges:
            role_slices[role] = slice(None)
            continue
        lo, hi = spec.ranges[role]
        role_slices[role] = _slice_for_range(axis, lo, hi, role, warnings)

    out_shape = tuple(
        _slice_length(shape[i], role_slices[role_for_label(labels[i])])
        for i in range(len(shape))
    )

    return TrimResult(slices=role_slices, warnings=warnings, out_shape=out_shape)


def suggest_default_trim(labels: list[str], axes: list[np.ndarray]) -> TrimSpec:
    """Outer 5% of slit axis; energy full; source_kind left as caller fill."""
    if len(labels) != len(axes):
        msg = f"labels length {len(labels)} != axes length {len(axes)}"
        raise ValueError(msg)

    ranges: dict[str, tuple[float, float]] = {}
    for label, axis in zip(labels, axes):
        role = role_for_label(label)
        axis_lo, axis_hi = _axis_extent(axis)
        if role == "energy":
            ranges["energy"] = (axis_lo, axis_hi)
        elif role == "slit":
            span = axis_hi - axis_lo
            margin = 0.05 * span
            ranges["slit"] = (axis_lo + margin, axis_hi - margin)

    return TrimSpec(ranges=ranges, source_kind="")
