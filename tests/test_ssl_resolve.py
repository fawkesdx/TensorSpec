"""Tests for SSL axis role resolution and sample mode enumeration."""

from tensorspec.core.ml.ssl.resolve import enumerate_modes


def test_enumerate_5d():
    labels = ["Y", "X", "Slit Defl.", "Energy", "Angle"]
    shape = (81, 81, 17, 230, 316)
    modes = {m.name: m for m in enumerate_modes(labels, shape)}
    assert modes["fermi3d"].n_samples == 81 * 81
    assert modes["fermi3d"].sample_shape == (17, 230, 316)
    assert modes["disp2d"].n_samples == 81 * 81 * 17
    assert modes["disp2d"].sample_shape == (230, 316)


def test_enumerate_4d_xy():
    labels = ["Y", "X", "Energy", "Angle"]
    shape = (81, 81, 230, 316)
    modes = enumerate_modes(labels, shape)
    assert [m.name for m in modes] == ["disp2d"]
    assert modes[0].n_samples == 6561


def test_enumerate_defl_x():
    labels = ["X", "Slit Defl.", "Energy", "Angle"]
    shape = (161, 37, 472, 568)
    modes = {m.name: m for m in enumerate_modes(labels, shape)}
    assert modes["fermi3d"].n_samples == 161
    assert modes["disp2d"].n_samples == 161 * 37
