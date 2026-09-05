"""Tests for SSL physical-axis calibration and regular-grid resampling."""

import numpy as np

from tensorspec.core.ml.ssl.calibrate import (
    DEG_PER_RAW_PX,
    resample_disp2d,
    resample_fermi3d,
    slit_axis_degrees,
)
from tensorspec.core.ml.ssl.spec import ResampleSpec


def test_slit_axis_degrees_is_monotonic_and_centred():
    axis = slit_axis_degrees(
        n_pixels=5,
        scale_offset=10.0,
        scale_delta=2.0,
        deg_per_raw_px=DEG_PER_RAW_PX[("R4000", "Angular30")],
    )

    assert axis.shape == (5,)
    assert np.all(np.diff(axis) > 0.0)
    assert axis[2] == 0.0


def test_resample_disp2d_has_fixed_shape_and_is_idempotent():
    energy = np.linspace(-1.0, 1.0, 7)
    slit = np.linspace(-3.0, 3.0, 9)
    sample = energy[:, None] + 2.0 * slit[None, :]
    spec = ResampleSpec(energy_size=5, slit_size=6)

    first = resample_disp2d(sample, energy, slit, spec)
    target_energy = np.linspace(energy.min(), energy.max(), spec.energy_size)
    target_slit = np.linspace(slit.min(), slit.max(), spec.slit_size)
    second = resample_disp2d(first, target_energy, target_slit, spec)

    assert first.shape == (5, 6)
    assert first.dtype == np.float32
    np.testing.assert_allclose(second, first, rtol=1e-6, atol=1e-6)


def test_resample_fermi3d_has_fixed_shape_and_is_idempotent():
    defl = np.linspace(-2.0, 2.0, 5)
    energy = np.linspace(-1.0, 1.0, 7)
    slit = np.linspace(-3.0, 3.0, 9)
    sample = (
        defl[:, None, None]
        + energy[None, :, None]
        + 2.0 * slit[None, None, :]
    )
    spec = ResampleSpec(defl_size=4, energy_size=5, slit_size=6)

    first = resample_fermi3d(sample, defl, energy, slit, spec)
    target_defl = np.linspace(defl.min(), defl.max(), spec.defl_size)
    target_energy = np.linspace(energy.min(), energy.max(), spec.energy_size)
    target_slit = np.linspace(slit.min(), slit.max(), spec.slit_size)
    second = resample_fermi3d(
        first,
        target_defl,
        target_energy,
        target_slit,
        spec,
    )

    assert first.shape == (4, 5, 6)
    assert first.dtype == np.float32
    np.testing.assert_allclose(second, first, rtol=1e-6, atol=1e-6)
