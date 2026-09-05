"""Tests for SSL percentile clip and min-max normalization."""

import numpy as np

from tensorspec.core.ml.ssl.normalize import (
    FileNormStats,
    estimate_file_stats,
    normalize_sample,
)
from tensorspec.core.ml.ssl.spec import NormSpec


def test_per_sample_minmax_in_unit_interval():
    rng = np.random.default_rng(0)
    frames = rng.uniform(10.0, 100.0, size=(8, 4, 4))
    spec = NormSpec(clip_percentiles=(0.0, 100.0), dead_pixel_sigma=0.0)

    stats = estimate_file_stats(frames, spec)
    sample = rng.uniform(20.0, 80.0, size=(4, 4))
    out = normalize_sample(sample, stats, spec)

    assert out.shape == sample.shape
    assert np.all(np.isfinite(out))
    assert out.min() >= 0.0
    assert out.max() <= 1.0
    assert out.min() == 0.0
    assert out.max() == 1.0


def test_per_file_scope_uses_file_anchors():
    spec = NormSpec(
        clip_percentiles=(0.0, 100.0),
        scope="per_file",
        dead_pixel_sigma=0.0,
    )
    frames = np.array([[[10.0, 50.0], [30.0, 90.0]]])
    stats = estimate_file_stats(frames, spec)

    sample = np.array([[20.0, 40.0], [60.0, 80.0]], dtype=np.float64)
    out = normalize_sample(sample, stats, spec)

    expected = (sample - stats.lo) / (stats.hi - stats.lo)
    np.testing.assert_allclose(out, expected)
    assert out.min() > 0.0
    assert out.max() < 1.0


def test_dead_pixels_replaced():
    spec = NormSpec(clip_percentiles=(0.0, 100.0), dead_pixel_sigma=3.0)
    base = np.linspace(1.0, 8.0, 8).reshape(2, 4)
    frames = np.stack([base, base * 1.1, base * 0.9], axis=0)
    frames[:, 0, 2] = 1.0e6  # persistently hot dead pixel

    stats = estimate_file_stats(frames, spec)
    assert stats.dead_pixel_mask is not None
    assert stats.dead_pixel_mask.shape == base.shape
    assert stats.dead_pixel_mask[0, 2]

    sample = base.copy()
    sample[0, 2] = 1.0e6
    out = normalize_sample(sample, stats, spec)

    assert np.isfinite(out[0, 2])
    assert out[0, 2] < 1.0
    assert out[0, 2] != sample[0, 2]


def test_drop_nonfinite_caller_side():
    spec = NormSpec(clip_percentiles=(0.0, 100.0), dead_pixel_sigma=0.0)
    frames = np.ones((4, 3, 3))
    stats = estimate_file_stats(frames, spec)

    all_zero = np.zeros((3, 3))
    out_zero = normalize_sample(all_zero, stats, spec)
    np.testing.assert_array_equal(out_zero, all_zero)

    noisy = np.array([[0.0, 1.0, 2.0], [np.nan, 4.0, 5.0], [6.0, 7.0, np.inf]])
    out_noisy = normalize_sample(noisy, stats, spec)
    assert np.all(np.isfinite(out_noisy))
