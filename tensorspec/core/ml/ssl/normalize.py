"""Percentile clip, dead-pixel handling, and min-max normalization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import median_filter

from tensorspec.core.ml.ssl.spec import NormSpec


@dataclass
class FileNormStats:
    lo: float
    hi: float
    dead_pixel_mask: np.ndarray | None  # detector frame (E, A) or sample shape


def _subsample_frames(frames: np.ndarray, n_points: int) -> np.ndarray:
    n_frames = frames.shape[0]
    if n_frames <= n_points:
        return frames
    indices = np.linspace(0, n_frames - 1, n_points, dtype=int)
    return frames[indices]


def _detector_median(frames: np.ndarray) -> np.ndarray:
    return np.median(frames, axis=0)


def _estimate_dead_pixel_mask(
    pixel_medians: np.ndarray,
    dead_pixel_sigma: float,
) -> np.ndarray | None:
    if dead_pixel_sigma <= 0.0:
        return None

    local = median_filter(pixel_medians, size=3, mode="nearest")
    deviation = np.abs(pixel_medians - local)
    mad = float(np.median(deviation))
    if mad <= 0.0:
        mad = float(np.std(deviation))
    if mad <= 0.0:
        return np.zeros(pixel_medians.shape, dtype=bool)

    return deviation > dead_pixel_sigma * mad


def _broadcast_mask(mask: np.ndarray, sample_shape: tuple[int, ...]) -> np.ndarray:
    if mask.shape == sample_shape:
        return mask
    if len(sample_shape) > len(mask.shape) and sample_shape[-len(mask.shape) :] == mask.shape:
        broadcast = np.broadcast_to(mask, sample_shape)
        return np.asarray(broadcast)
    msg = f"dead_pixel_mask shape {mask.shape} incompatible with sample shape {sample_shape}"
    raise ValueError(msg)


def _replace_dead_pixels(sample: np.ndarray, mask: np.ndarray) -> np.ndarray:
    local = median_filter(sample, size=3, mode="nearest")
    out = sample.copy()
    out[mask] = local[mask]
    return out


def estimate_file_stats(
    frames: np.ndarray,
    spec: NormSpec,
) -> FileNormStats:
    """Estimate percentile clip bounds and dead-pixel mask from frame subsample."""
    if frames.ndim < 2:
        msg = f"frames must be at least 2D (N, ...); got shape {frames.shape}"
        raise ValueError(msg)

    subsample = _subsample_frames(frames, spec.subsample_points)
    values = subsample[np.isfinite(subsample)]
    if values.size == 0:
        return FileNormStats(lo=0.0, hi=1.0, dead_pixel_mask=None)

    lo, hi = np.percentile(values, spec.clip_percentiles)
    lo_f, hi_f = float(lo), float(hi)
    if hi_f <= lo_f:
        hi_f = lo_f + 1.0

    pixel_medians = _detector_median(subsample)
    dead_pixel_mask = _estimate_dead_pixel_mask(pixel_medians, spec.dead_pixel_sigma)

    return FileNormStats(lo=lo_f, hi=hi_f, dead_pixel_mask=dead_pixel_mask)


def normalize_sample(
    sample: np.ndarray,
    stats: FileNormStats,
    spec: NormSpec,
) -> np.ndarray:
    """Clip to [lo, hi], replace dead pixels, min-max to [0, 1]."""
    if sample.size == 0:
        return sample.astype(np.float64, copy=False)

    if np.all(sample == 0):
        return np.zeros_like(sample, dtype=np.float64)

    out = np.nan_to_num(sample.astype(np.float64, copy=True), nan=0.0, posinf=stats.hi, neginf=stats.lo)

    if stats.dead_pixel_mask is not None:
        mask = _broadcast_mask(stats.dead_pixel_mask, out.shape)
        out = _replace_dead_pixels(out, mask)

    out = np.clip(out, stats.lo, stats.hi)

    if np.all(out == 0):
        return out

    if spec.scope == "per_file":
        lo, hi = stats.lo, stats.hi
    else:
        lo, hi = float(np.min(out)), float(np.max(out))

    span = hi - lo
    if span <= 0.0:
        return np.zeros_like(out)

    return (out - lo) / span
