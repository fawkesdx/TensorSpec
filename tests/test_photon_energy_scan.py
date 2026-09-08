import numpy as np
import pytest
from tensorspec.core.arpes.photon_energy_scan import (
    build_hv_list,
    stack_hv_cubes,
    is_dispersion_axes,
    resolve_photon_energies,
)


def test_build_hv_list_inclusive_finish():
    hv = build_hv_list(80.0, 90.0, 5.0)
    assert np.allclose(hv, [80.0, 85.0, 90.0])


def test_build_hv_list_rejects_bad_step():
    with pytest.raises(ValueError):
        build_hv_list(80.0, 90.0, 0.0)
    with pytest.raises(ValueError):
        build_hv_list(90.0, 80.0, 5.0)


def test_stack_dispersion_shapes_3d():
    # single-hv cube (ntheta, nphi=1, ne)
    cubes = [np.ones((10, 1, 5)) * i for i in range(3)]
    hv = np.array([80.0, 85.0, 90.0])
    stacked, hv_out = stack_hv_cubes(cubes, hv)
    assert stacked.shape == (3, 10, 1, 5)
    assert np.allclose(hv_out, hv)
    assert stacked[2, 0, 0, 0] == 2.0


def test_stack_fermi_shapes_4d_raw():
    cubes = [np.ones((8, 9, 4)) * i for i in range(2)]
    hv = np.array([84.0, 90.0])
    stacked, _ = stack_hv_cubes(cubes, hv)
    assert stacked.shape == (2, 8, 9, 4)


def test_is_dispersion_degenerate_phi():
    theta = np.linspace(-15, 15, 31)
    phi = np.array([0.0])
    assert is_dispersion_axes(theta, phi) is True
    phi2 = np.linspace(-10, 10, 21)
    assert is_dispersion_axes(theta, phi2) is False


def test_resolve_single():
    assert resolve_photon_energies(mode="single", single=90.0) == [90.0]


def test_resolve_range():
    out = resolve_photon_energies(
        mode="range", start=80.0, finish=90.0, step=5.0
    )
    assert out == [80.0, 85.0, 90.0]
