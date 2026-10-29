"""core/arpes/peakfit + products on a synthetic two-Lorentzian cut (no Qt, no data files)."""
import json

import numpy as np
import pytest

from tensorspec.core.arpes import peakfit as pf
from tensorspec.core.arpes.products import (
    band_model_of,
    is_peakfit_dataset,
    line_components,
    line_table,
    nearest_line,
    peakfit_dataset,
    settings_of,
)
from tensorspec.core.data_models import TensorData
from tensorspec.core.workspace import WorkspaceManager

K = np.linspace(-0.3, 0.3, 121)
E = np.linspace(-0.2, 0.05, 26)
VF = 2.0            # eV A
W = 0.04            # Lorentzian FWHM in 1/A


def _cut(seed=0):
    """Two linear bands, k = -/+ (0.12 + E / v_F), that never overlap, on a sloping
    background."""
    rng = np.random.default_rng(seed)
    cut = np.zeros((K.size, E.size))
    for i, e in enumerate(E):
        kc = 0.12 + e / VF
        line = 300 * pf.peak_profile("lorentzian", K, -kc, W) \
            + 200 * pf.peak_profile("lorentzian", K, kc, W) + 20 + 10 * K
        cut[:, i] = rng.poisson(line)
    return cut


def _bands(shape="lorentzian"):
    left = pf.Band("L", [pf.Seed(-0.2, -0.02, 0.05, 300), pf.Seed(0.05, -0.145, 0.05, 300)], shape)
    right = pf.Band("R", [pf.Seed(-0.2, 0.02, 0.05, 200), pf.Seed(0.05, 0.145, 0.05, 200)],
                    shape, share_width="L")
    return pf.BandModel([left, right])


def test_peak_profile_conventions():
    x = np.linspace(-1, 1, 2001)
    for shape in pf.PEAK_SHAPES:
        y = pf.peak_profile(shape, x, 0.0, 0.2, 0.05 if shape == "voigt" else 0.0)
        assert y.max() == pytest.approx(1.0)
        half = x[y >= 0.5]
        if shape == "lorentzian":
            assert half.max() - half.min() == pytest.approx(0.2, abs=2e-3)
    with pytest.raises(ValueError):
        pf.peak_profile("sech", x, 0.0, 0.1)


def test_band_model_roundtrips_through_json():
    model = _bands("voigt")
    back = pf.BandModel.from_json(model.to_json())
    assert back.names == ["L", "R"]
    assert back.bands[1].share_width == "L" and back.bands[0].shape == "voigt"
    assert back.bands[0].seeds[0].fwhm == 0.05
    settings = pf.FitSettings(background="linear", resolution=0.01, fermi=True, temperature=20)
    assert pf.FitSettings.from_json(settings.to_json()) == settings
    with pytest.raises(ValueError):
        pf.BandModel([pf.Band("A"), pf.Band("A")])
    with pytest.raises(ValueError):
        pf.Band("A", shape="sech")


def test_fit_line_recovers_two_lorentzians():
    cut = _cut()
    i = 5
    fit = pf.fit_line(K, cut[:, i], _bands(), pf.FitSettings(), position=E[i])
    kc = 0.12 + E[i] / VF
    assert fit.names == ["L", "R"]
    assert fit.centres[0] == pytest.approx(-kc, abs=0.004)
    assert fit.centres[1] == pytest.approx(kc, abs=0.004)
    assert fit.widths[0] == pytest.approx(W, abs=0.01)
    assert fit.widths[1] == fit.widths[0]            # shared
    assert fit.heights[0] == pytest.approx(300, rel=0.1)
    assert np.all(np.isfinite(fit.centre_errors)) and fit.chi2 < 3
    assert fit.component(0).shape == K.shape


def test_fit_series_tracks_the_dispersion_and_builds_a_dataset():
    cut = _cut()
    model = _bands()
    settings = pf.FitSettings()
    series = pf.fit_series(cut, (K, E), model, settings, direction="mdc", bounds=(-0.15, 0.0),
                           combine=3)
    assert series.direction == "mdc" and series.names == ["L", "R"]
    assert series.positions.min() >= -0.15 and series.positions.max() <= 0.0
    expected = -(0.12 + series.positions / VF)
    assert np.nanmax(np.abs(series.centres[:, 0] - expected)) < 0.005
    k, e, err = series.band(1)
    assert k.size == e.size == err.size == series.positions.size

    ds = peakfit_dataset(series, model, settings, resolution_source="none")
    assert is_peakfit_dataset(ds)
    assert ds.attrs["kind"] == "mdc_peakfit" and ds.attrs["direction"] == "mdc"
    assert ds["data"].dims == ("Energy", "kx") and ds["centre"].dims == ("Energy", "band")
    assert ds.sizes["kx"] == K.size and ds.sizes["band"] == 2
    assert list(ds.coords["band"].values) == ["L", "R"]
    assert json.loads(ds.attrs["band_model"])["bands"][1]["share_width"] == "L"
    assert band_model_of(ds).names == ["L", "R"]
    assert settings_of(ds) == settings
    assert ds.attrs["resolution_source"] == "none"
    i = nearest_line(ds, -0.1)
    assert ds.coords["Energy"].values[i] == pytest.approx(-0.1, abs=0.01)
    comps = line_components(ds, i)
    assert [c[0] for c in comps] == ["L", "R"]
    # the sum of the components (minus the background counted twice) is the model
    bg = ds["background"].values[i]
    total = comps[0][2] + comps[1][2] - bg
    np.testing.assert_allclose(total, ds["model"].values[i], rtol=1e-6, atol=1e-6)
    rows = line_table(ds, i)
    assert len(rows) == 2 and rows[0][0] == "L" and rows[0][3] == pytest.approx(W, abs=0.01)

    # it goes through the workspace and comes back as a tensor the DataViewer can open
    ws = WorkspaceManager()
    ws.push_spectroscopy_data("cut", TensorData(cut, [K, E], ["kx", "Energy"], ["1/A", "eV"],
                                                "Experimental ARPES"))
    assert ws.write_analysis_data("cut", "mdc_peakfit", ds)
    assert ws.pull_analysis_data("cut", "mdc_peakfit").attrs["kind"] == "mdc_peakfit"
    tensor = ws.pull_tensor_data("cut", "analysis/mdc_peakfit")
    assert tensor.labels == ["Energy", "kx"] and tensor.value.shape == ds["data"].shape


def test_edc_series_with_fermi_cutoff():
    cut = _cut()
    # EDC seeds: position = k, centre = E of the left band, E = -v_F (k + 0.12)
    left = pf.Band("L", [pf.Seed(-0.11, -VF * (-0.11 + 0.12), 0.08, 300),
                         pf.Seed(-0.03, -VF * (-0.03 + 0.12), 0.08, 300)], "lorentzian")
    model = pf.BandModel([left])
    settings = pf.FitSettings(background="linear", resolution=0.01, fermi=True, ef=0.0,
                              temperature=30.0)
    series = pf.fit_series(cut, (K, E), model, settings, direction="edc", bounds=(-0.11, -0.03))
    assert series.direction == "edc"
    good = np.isfinite(series.centres[:, 0])
    assert good.sum() >= 5
    expected = -VF * (series.positions[good] + 0.12)
    assert np.nanmax(np.abs(series.centres[good, 0] - expected)) < 0.02
    ds = peakfit_dataset(series, model, settings)
    assert ds.attrs["kind"] == "edc_peakfit" and ds["data"].dims == ("kx", "Energy")
    comps = line_components(ds, 0)
    assert len(comps) == 1 and comps[0][2].shape == E.shape


def test_line_at_and_seeds_from_points():
    cut = _cut()
    pos, along, line = pf.line_at(cut, (K, E), -0.1, direction="mdc", combine=3)
    assert pos == pytest.approx(-0.1, abs=0.01) and along is K and line.shape == K.shape
    seeds = pf.seeds_from_points(cut, (K, E), [(-0.1, -0.07), (0.0, -0.12)], 0.04)
    assert len(seeds) == 2 and seeds[0].height > 0 and seeds[0].fwhm == 0.04


def test_core_peakfit_has_no_qt_or_plotting():
    import re
    from pathlib import Path

    pkg = Path(pf.__file__).parent
    pat = re.compile(r"^\s*(?:from|import)\s+(PySide6|PyQt5|PyQt6|matplotlib|tensorspec\.gui|tensorspec\.plotting)\b", re.M)
    for name in ("peakfit.py", "products.py"):
        assert pat.findall((pkg / name).read_text()) == []
