"""Tests for SSL physical-unit trim application."""

import numpy as np

from tensorspec.core.ml.ssl.spec import TrimSpec
from tensorspec.core.ml.ssl.trim import apply_trim, suggest_default_trim


def test_trim_physical_units_binning_invariant():
    """Same physical window, two binnings -> same physical span retained."""
    labels = ["Energy", "Angle"]
    shape = (100, 50)
    fine_energy = np.linspace(0.0, 10.0, 100)
    coarse_energy = np.linspace(0.0, 10.0, 50)
    slit = np.linspace(-15.0, 15.0, 50)
    spec = TrimSpec(ranges={"energy": (2.0, 8.0)}, source_kind="test")

    fine = apply_trim(labels, [fine_energy, slit], shape, spec)
    coarse = apply_trim(labels, [coarse_energy, slit], (50, 50), spec)

    fine_axis = fine_energy[fine.slices["energy"]]
    coarse_axis = coarse_energy[coarse.slices["energy"]]
    for axis in (fine_axis, coarse_axis):
        assert axis.min() >= 2.0
        assert axis.max() <= 8.0
    assert abs(fine_axis.min() - coarse_axis.min()) < 0.15
    assert abs(fine_axis.max() - coarse_axis.max()) < 0.15
    assert fine.warnings == []
    assert coarse.warnings == []


def test_trim_clamp_warns():
    labels = ["Energy", "Angle"]
    energy = np.linspace(0.0, 10.0, 11)
    slit = np.linspace(-10.0, 10.0, 21)
    spec = TrimSpec(ranges={"energy": (-5.0, 15.0)}, source_kind="test")

    result = apply_trim(labels, [energy, slit], (11, 21), spec)

    assert result.slices["energy"] == slice(0, 11)
    assert any("clamp" in w.lower() for w in result.warnings)
    assert result.out_shape == (11, 21)


def test_trim_absent_role_keeps_full_axis():
    labels = ["Energy", "Angle"]
    energy = np.linspace(0.0, 10.0, 11)
    slit = np.linspace(-10.0, 10.0, 21)
    spec = TrimSpec(ranges={"energy": (2.0, 8.0)}, source_kind="test")

    result = apply_trim(labels, [energy, slit], (11, 21), spec)

    assert result.slices["slit"] == slice(None)
    assert result.out_shape == (7, 21)


def test_trim_empty_match_warns_keeps_full():
    labels = ["Energy"]
    energy = np.linspace(0.0, 10.0, 11)
    spec = TrimSpec(ranges={"energy": (20.0, 30.0)}, source_kind="test")

    result = apply_trim(labels, [energy], (11,), spec)

    assert result.slices["energy"] == slice(None)
    assert any("empty" in w.lower() or "no index" in w.lower() for w in result.warnings)
    assert result.out_shape == (11,)


def test_suggest_default_trim():
    labels = ["Energy", "Angle"]
    energy = np.linspace(-2.0, 2.0, 41)
    slit = np.linspace(-20.0, 20.0, 401)

    spec = suggest_default_trim(labels, [energy, slit])

    assert spec.ranges["energy"] == (-2.0, 2.0)
    assert spec.ranges["slit"] == (-18.0, 18.0)
    assert spec.source_kind == ""
