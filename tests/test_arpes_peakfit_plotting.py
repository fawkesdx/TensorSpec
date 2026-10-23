"""draw_cut_with_tracks / draw_line_fit on a synthetic fitted cut (Agg figure, no window)."""
import numpy as np
from matplotlib.figure import Figure

from tensorspec.core.arpes import peakfit as pf
from tensorspec.core.arpes.products import peakfit_dataset
from tensorspec.core.data_models import TensorData
from tensorspec.plotting.backends.matplotlib_engine import (
    band_colours,
    draw_cut_with_tracks,
    draw_line_fit,
    orient_cut,
)

K = np.linspace(-0.3, 0.3, 61)
E = np.linspace(-0.2, 0.05, 26)


def _fitted():
    cut = np.zeros((K.size, E.size))
    for i, e in enumerate(E):
        kc = 0.12 + e / 2.0
        cut[:, i] = 300 * pf.peak_profile("lorentzian", K, -kc, 0.04) \
            + 200 * pf.peak_profile("lorentzian", K, kc, 0.04) + 20
    model = pf.BandModel([
        pf.Band("L", [pf.Seed(-0.2, -0.02, 0.05, 300), pf.Seed(0.05, -0.145, 0.05, 300)], colour=2),
        pf.Band("R", [pf.Seed(-0.2, 0.02, 0.05, 200), pf.Seed(0.05, 0.145, 0.05, 200)],
                share_width="L")])
    settings = pf.FitSettings()
    series = pf.fit_series(cut, (K, E), model, settings, bounds=(-0.15, 0.0), refine=False)
    tensor = TensorData(cut, [K, E], ["kx", "Energy"], ["1/A", "eV"], "Experimental ARPES")
    return tensor, peakfit_dataset(series, model, settings)


def test_orient_cut_puts_energy_vertical():
    tensor, _ = _fitted()
    x, y, image, xl, yl, xu, yu = orient_cut(tensor)
    assert xl == "kx" and yl == "Energy" and image.shape == (E.size, K.size)
    flipped = TensorData(tensor.value.T, [E, K], ["Energy", "kx"], ["eV", "1/A"], "ARPES")
    x2, y2, image2, xl2, yl2, _, _ = orient_cut(flipped)
    assert xl2 == "kx" and yl2 == "Energy"
    np.testing.assert_array_equal(image2, image)


def test_draw_cut_and_line():
    tensor, ds = _fitted()
    assert band_colours(ds)[0] == "#2E8B57"          # Band.colour=2 picks the third palette entry
    fig = Figure()
    ax = fig.add_subplot(111)
    out = draw_cut_with_tracks(ax, tensor, ds, crosshair=-0.1, direction="mdc")
    assert out["crosshair"].get_ydata()[0] == -0.1
    assert ax.get_xlabel().startswith("kx") and ax.get_ylabel().startswith("Energy")
    assert len(ax.lines) >= 3                          # E=0 line, two tracks, crosshair
    out["crosshair"].set_ydata([-0.05, -0.05])
    fig.canvas.draw()

    ax2 = fig.add_subplot(122)
    rows = draw_line_fit(ax2, ds, 3, direction="mdc")
    assert [r[0] for r in rows] == ["L", "R"]
    assert "χ²" in ax2.get_title() and len(ax2.lines) >= 5   # 2 components, bg, model, data, 0-line
    rows_hidden = draw_line_fit(ax2, ds, 3, show_fit=False)
    assert rows_hidden == [] and "hidden" in ax2.get_title()
    rows_raw = draw_line_fit(ax2, None, None, raw_line=(K, tensor.value[:, 0], "kx (1/A)"))
    assert rows_raw == [] and "no fit" in ax2.get_title()
    # EDC orientation: a vertical crosshair
    out = draw_cut_with_tracks(ax, tensor, None, crosshair=0.05, direction="edc")
    assert out["crosshair"].get_xdata()[0] == 0.05
