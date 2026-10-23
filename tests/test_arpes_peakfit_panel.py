"""Headless checks of the peak-fit explorer tab: pickers, crosshair, table, dialog, worker and
its place in the ARPES suite."""
import numpy as np
import pytest

from tensorspec.core.arpes import peakfit as pf
from tensorspec.core.arpes.products import peakfit_dataset
from tensorspec.core.data_models import TensorData
from tensorspec.core.workspace import global_workspace

K = np.linspace(-0.3, 0.3, 61)
E = np.linspace(-0.2, 0.05, 26)


def _cut():
    cut = np.zeros((K.size, E.size))
    for i, e in enumerate(E):
        kc = 0.12 + e / 2.0
        cut[:, i] = 300 * pf.peak_profile("lorentzian", K, -kc, 0.04) \
            + 200 * pf.peak_profile("lorentzian", K, kc, 0.04) + 20
    return cut


def _model():
    return pf.BandModel([
        pf.Band("L", [pf.Seed(-0.2, -0.02, 0.05, 300), pf.Seed(0.05, -0.145, 0.05, 300)]),
        pf.Band("R", [pf.Seed(-0.2, 0.02, 0.05, 200), pf.Seed(0.05, 0.145, 0.05, 200)],
                share_width="L")])


@pytest.fixture
def make_panel(qapp):
    """Panels built here are flushed and deleted before the test ends, so no canvas redraw
    timer fires on an already-deleted widget during a later test's processEvents()."""
    made = []

    def build():
        from tensorspec.gui.components.arpes_tabs.peakfit_explorer_panel import PeakfitExplorerPanel
        panel = PeakfitExplorerPanel()
        made.append(panel)
        return panel

    yield build
    qapp.processEvents()
    for panel in made:
        panel.deleteLater()
    qapp.processEvents()


@pytest.fixture
def workspace_cut():
    name = "peakfit_test_cut"
    cut = _cut()
    global_workspace.push_spectroscopy_data(
        name, TensorData(cut, [K, E], ["kx", "Energy"], ["1/A", "eV"], "Experimental ARPES",
                         {"info": {"Temperature": {"Sample": 13.0}, "Beamline": {"Eresolution": 0.005}}}))
    settings = pf.FitSettings()
    series = pf.fit_series(cut, (K, E), _model(), settings, bounds=(-0.15, 0.0), refine=False)
    global_workspace.write_analysis_data(name, "mdc_peakfit", peakfit_dataset(series, _model(), settings))
    yield name, cut
    global_workspace.remove(name)


def test_panel_pickers_crosshair_and_table(qapp, workspace_cut, make_panel):
    from tensorspec.gui.components.arpes_tabs.peakfit_explorer_panel import NO_FIT

    name, cut = workspace_cut
    panel = make_panel()
    assert panel.select_dataset(name, "raw")
    assert panel.current_source() == (name, "raw")
    # the stored MDC fit is listed and picked up by default
    assert [panel.cmb_node.itemText(i) for i in range(panel.cmb_node.count())] == [NO_FIT, "mdc_peakfit"]
    assert panel.cmb_node.currentText() == "mdc_peakfit" and panel.dataset is not None
    assert panel.position is not None

    panel.set_position(-0.1)
    assert panel.position == pytest.approx(-0.1, abs=0.011)           # snapped to a fitted line
    assert panel.table.rowCount() == 2
    assert panel.table.item(0, 0).text() == "L"
    assert float(panel.table.item(0, 1).text()) == pytest.approx(-(0.12 + panel.position / 2.0), abs=0.005)
    assert "χ²" in panel.ax_right.get_title()
    assert panel.lbl_readout.text().startswith("Energy = ")

    before = panel.position
    panel.step_line(+1)
    assert panel.position > before
    panel.chk_show_fit.setChecked(False)
    assert panel.table.rowCount() == 0 and "hidden" in panel.ax_right.get_title()
    panel.chk_show_fit.setChecked(True)

    # outside the fitted window the raw MDC is shown
    panel.set_position(0.04)
    assert panel.table.rowCount() == 0 and "no fit" in panel.ax_right.get_title()

    # EDC direction: no stored EDC fit, vertical crosshair on k
    panel.set_direction("edc")
    assert panel.cmb_node.count() == 1 and panel.dataset is None
    panel.set_position(0.1)
    assert panel._artists["crosshair"].get_xdata()[0] == pytest.approx(0.1)
    assert panel.lbl_readout.text().startswith("kx = ")
    panel.set_direction("mdc")
    assert panel.cmb_node.currentText() == "mdc_peakfit"

    panel.cmb_node.setCurrentIndex(0)
    assert panel.dataset is None and panel.table.rowCount() == 0


def test_dialog_prefills_from_stored_fit_and_parses_seeds(qapp, workspace_cut):
    from tensorspec.gui.components.arpes_tabs.peakfit_explorer_panel import (
        BandModelDialog, parse_seed_text)

    assert parse_seed_text("0.0 -0.085; -0.025, -0.093 0.04\n") == [(0.0, -0.085), (-0.025, -0.093, 0.04)]
    with pytest.raises(ValueError):
        parse_seed_text("1 2 3 4")

    name, cut = workspace_cut
    ds = global_workspace.pull_analysis_data(name, "mdc_peakfit")
    from tensorspec.core.arpes.products import band_model_of, settings_of
    tensor = global_workspace.pull_tensor_data(name)
    dialog = BandModelDialog("mdc", (K, E, cut), tensor.metadata, band_model_of(ds), settings_of(ds),
                             "mdc_peakfit")
    assert dialog.table.rowCount() == 2
    model = dialog.band_model()
    assert model.names == ["L", "R"] and model.bands[1].share_width == "L"
    assert model.bands[0].seeds[0].fwhm == 0.05 and model.bands[0].seeds[0].height > 0
    assert dialog.settings().background == "linear"
    opts = dialog.options()
    assert opts["direction"] == "mdc" and opts["bounds"] == (-0.15, 0.0) and opts["node"] == "mdc_peakfit"
    dialog.spin_vf.setValue(2.0)
    dialog._resolution_from_metadata()
    assert dialog.spin_resolution.value() == pytest.approx(0.0025)
    assert dialog.resolution_source().startswith("info.Beamline.Eresolution / v_F=2")

    edc = BandModelDialog("edc", (K, E, cut), tensor.metadata)
    assert edc.chk_fermi.isChecked() and edc.spin_temperature.value() == 13.0
    assert edc.spin_resolution.value() == pytest.approx(0.005)
    assert edc.options()["node"] == "edc_peakfit"


def test_worker_and_store_series(qapp, workspace_cut, make_panel):
    from tensorspec.gui.components.arpes_tabs.peakfit_explorer_panel import FitWorker

    name, cut = workspace_cut
    panel = make_panel()
    assert panel.select_dataset(name, "raw")
    settings = pf.FitSettings(resolution=0.002)
    options = {"direction": "mdc", "bounds": (-0.1, -0.05), "step": 1, "combine": 3, "refine": False,
               "node": "mdc_outer", "resolution_source": "test"}
    worker = FitWorker(cut, (K, E), _model(), settings, options, ("kx", "Energy"), ("1/A", "eV"))
    got = {}
    worker.finished_series.connect(lambda s: got.setdefault("series", s))
    worker.failed.connect(lambda m: got.setdefault("failed", m))
    worker.run()                       # synchronously, same code path as start()
    assert "series" in got and "failed" not in got
    series = got["series"]
    assert series.positions.min() >= -0.1 and series.positions.max() <= -0.05

    target = panel.store_series(series, _model(), settings, options)
    assert target == "mdc_outer"
    ds = global_workspace.pull_analysis_data(name, "mdc_outer")
    assert ds.attrs["resolution_source"] == "test" and ds.attrs["source_node"] == "raw"
    assert ds.attrs["temperature"] == 13.0 and ds.attrs["combine"] == 3
    assert panel.cmb_node.currentText() == "mdc_outer" and panel.table.rowCount() == 2

    cancelled = FitWorker(cut, (K, E), _model(), settings, options, ("kx", "Energy"), ("1/A", "eV"))
    cancelled.cancel()
    msgs = []
    cancelled.failed.connect(msgs.append)
    cancelled.run()
    assert msgs == ["cancelled"]


def test_arpes_suite_has_the_tab_next_to_the_data_viewer(qapp):
    from tensorspec.gui.suites.arpes_suite import ARPESSuite

    suite = ARPESSuite()
    titles = [suite.tabs.tabText(i) for i in range(suite.tabs.count())]
    assert titles == ["Simulations (Chinook/SPRKKR)", "Data Viewer", "Peak-fit Explorer"]
    assert suite.tabs.widget(2) is suite.peakfit_panel
    qapp.processEvents()
    suite.close()                      # WA_DeleteOnClose: flush the deletion here, not later
    qapp.processEvents()
