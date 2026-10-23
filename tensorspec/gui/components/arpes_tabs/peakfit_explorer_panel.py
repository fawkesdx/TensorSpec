"""MDC / EDC peak-fit explorer tab of the ARPES suite.

Pick a 2-D cut from the workspace (``/raw`` or a ``/processed`` child) and a fit stored under
``/analysis``; drag a crosshair over the E-k image (arrow keys step one line) and see the MDC
or EDC at that position with the total model, the background, every band's component in its
colour and a table of centre ± err, FWHM and height with the reduced chi2 in the title.
"Fit…" opens a minimal band-model dialog and runs ``core.arpes.peakfit.fit_series`` in a
QThread; the result goes to ``/analysis/<node>`` through ``workspace.write_analysis_data``.

Numerics live in core/arpes/peakfit.py and products.py, drawing in
plotting/backends/matplotlib_engine.py (draw_cut_with_tracks, draw_line_fit).
"""
from __future__ import annotations

import re

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from tensorspec.core.arpes import peakfit as pf
from tensorspec.core.arpes.products import (
    band_model_of,
    is_peakfit_dataset,
    nearest_line,
    peakfit_dataset,
    peakfit_kind,
    settings_of,
)
from tensorspec.core.io.matlab_arpes import arpes_metadata_number
from tensorspec.core.workspace import global_workspace
from tensorspec.plotting.backends.matplotlib_engine import (
    band_colours,
    draw_cut_with_tracks,
    draw_line_fit,
    orient_cut,
)

NO_FIT = "(no fit)"
#: Default v_F used to turn the energy resolution into a momentum resolution for MDC fits.
DEFAULT_VF = 3.0


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------
class FitWorker(QThread):
    """Runs ``fit_series`` off the GUI thread; ``cancel()`` stops it between lines."""

    progress = Signal(int, int)
    finished_series = Signal(object)
    failed = Signal(str)

    def __init__(self, values, axes, model, settings, options: dict, labels, units, parent=None):
        super().__init__(parent)
        self._args = (np.asarray(values, float), axes, model, settings)
        self._options = dict(options)
        self._labels, self._units = tuple(labels), tuple(units)
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def _progress(self, done, total):
        self.progress.emit(int(done), int(total))
        return not self._cancelled

    def run(self):
        values, axes, model, settings = self._args
        o = self._options
        try:
            series = pf.fit_series(values, axes, model, settings, direction=o["direction"],
                                   step=o.get("step", 1), combine=o.get("combine", 1),
                                   bounds=o.get("bounds"), refine=o.get("refine", True),
                                   progress=self._progress, labels=self._labels,
                                   units=self._units)
        except Exception as exc:          # reported in the GUI, never raised in the thread
            self.failed.emit(str(exc))
            return
        if self._cancelled:
            self.failed.emit("cancelled")
            return
        self.finished_series.emit(series)


# ---------------------------------------------------------------------------
# Band-model dialog
# ---------------------------------------------------------------------------
def parse_seed_text(text: str) -> list:
    """``"0.0 -0.085; -0.025 -0.093 0.04"`` -> ``[(position, centre[, fwhm]), ...]``.
    Seeds are separated by ``;`` or newlines, numbers by spaces or commas."""
    out = []
    for chunk in re.split(r"[;\n]+", str(text)):
        numbers = [float(t) for t in re.split(r"[\s,]+", chunk.strip()) if t]
        if not numbers:
            continue
        if len(numbers) not in (2, 3):
            raise ValueError(f"a seed is 'position centre [fwhm]', not {chunk.strip()!r}")
        out.append(tuple(numbers))
    return out


def seed_text(band: pf.Band) -> str:
    return "; ".join(f"{s.position:g} {s.centre:g} {s.fwhm:g}" for s in band.sorted_seeds())


class BandModelDialog(QDialog):
    """Bands (name, shape, shared / fixed width, starting FWHM, seeds as text) and the fit
    settings. Prefilled from an existing fit so a stored model can be rerun on the next cut."""

    COLUMNS = ("Band", "Shape", "Share width of", "Fix width", "FWHM₀", "Seeds: position centre [fwhm]; …")

    def __init__(self, direction: str, cut, metadata: dict | None = None, model=None,
                 settings=None, node_name: str = "", parent=None):
        super().__init__(parent)
        self.direction = direction
        self._cut = cut                      # (x, y, values[x, y])
        x, y, _ = cut
        along, across = (x, y) if direction == "mdc" else (y, x)
        self.setWindowTitle(f"{direction.upper()} band model")
        self.resize(900, 520)
        metadata = metadata or {}
        eres = arpes_metadata_number(metadata, "info", "Beamline", "Eresolution", default=0.0) or 0.0
        temp = arpes_metadata_number(metadata, "info", "Temperature", "Sample", default=30.0) or 30.0

        root = QVBoxLayout(self)
        hint = ("Seeds: one or more 'position centre [fwhm]' triples per band, separated by ';'. "
                f"Position is on the {'energy' if direction == 'mdc' else 'momentum'} axis "
                f"(the line), centre along the {'momentum' if direction == 'mdc' else 'energy'} "
                "axis. Heights are read off the data; FWHM₀ is the starting width when a seed "
                "has none. A band exists only between its first and last seed.")
        lab = QLabel(hint)
        lab.setWordWrap(True)
        root.addWidget(lab)

        self.table = QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        self.table.horizontalHeader().setSectionResizeMode(len(self.COLUMNS) - 1, QHeaderView.Stretch)
        root.addWidget(self.table, 1)
        row = QHBoxLayout()
        btn_add = QPushButton("Add band")
        btn_add.clicked.connect(lambda: self._add_row())
        btn_del = QPushButton("Remove band")
        btn_del.clicked.connect(self._remove_row)
        row.addWidget(btn_add)
        row.addWidget(btn_del)
        row.addStretch(1)
        root.addLayout(row)

        box = QGroupBox("Fit settings")
        form = QFormLayout(box)
        self.cmb_background = QComboBox()
        self.cmb_background.addItems(list(pf.BACKGROUNDS))
        self.cmb_background.setCurrentText("linear")
        form.addRow("Background", self.cmb_background)
        self.spin_resolution = QDoubleSpinBox()
        self.spin_resolution.setDecimals(6)
        self.spin_resolution.setRange(0.0, 10.0)
        self.spin_resolution.setSingleStep(0.0005)
        self.spin_vf = QDoubleSpinBox()
        self.spin_vf.setDecimals(3)
        self.spin_vf.setRange(0.001, 100.0)
        self.spin_vf.setValue(DEFAULT_VF)
        res_row = QHBoxLayout()
        res_row.addWidget(self.spin_resolution)
        unit = "eV" if direction == "edc" else "1/A"
        res_row.addWidget(QLabel(unit))
        if direction == "mdc":
            res_row.addWidget(QLabel("  = ΔE / v_F, v_F (eV Å) ="))
            res_row.addWidget(self.spin_vf)
        self._eres = float(eres)
        btn_meta = QPushButton(f"From metadata (ΔE = {eres * 1e3:.2f} meV)")
        btn_meta.setEnabled(eres > 0)
        btn_meta.clicked.connect(self._resolution_from_metadata)
        res_row.addWidget(btn_meta)
        res_row.addStretch(1)
        form.addRow("Resolution FWHM", res_row)
        self.cmb_weighting = QComboBox()
        self.cmb_weighting.addItems(["poisson", "none"])
        form.addRow("Weighting", self.cmb_weighting)
        self.chk_fermi = QCheckBox("multiply the model by the Fermi-Dirac occupation (EDC)")
        self.chk_fermi.setEnabled(direction == "edc")
        self.chk_fermi.setChecked(direction == "edc")
        self.spin_ef = QDoubleSpinBox()
        self.spin_ef.setDecimals(4)
        self.spin_ef.setRange(-100, 100)
        self.spin_temperature = QDoubleSpinBox()
        self.spin_temperature.setDecimals(2)
        self.spin_temperature.setRange(0.01, 5000)
        self.spin_temperature.setValue(float(temp))
        fd_row = QHBoxLayout()
        fd_row.addWidget(self.chk_fermi)
        fd_row.addWidget(QLabel("E_F (eV)"))
        fd_row.addWidget(self.spin_ef)
        fd_row.addWidget(QLabel("T (K)"))
        fd_row.addWidget(self.spin_temperature)
        fd_row.addStretch(1)
        form.addRow("Fermi cut-off", fd_row)

        self.spin_lo = QDoubleSpinBox()
        self.spin_hi = QDoubleSpinBox()
        for spin in (self.spin_lo, self.spin_hi):
            spin.setDecimals(4)
            spin.setRange(float(np.nanmin(across)) - 1, float(np.nanmax(across)) + 1)
        if direction == "mdc":
            self.spin_lo.setValue(max(-0.15, float(np.nanmin(across))))
            self.spin_hi.setValue(min(0.0, float(np.nanmax(across))))
        else:
            self.spin_lo.setValue(float(np.nanmin(across)))
            self.spin_hi.setValue(float(np.nanmax(across)))
        win_row = QHBoxLayout()
        win_row.addWidget(self.spin_lo)
        win_row.addWidget(QLabel("to"))
        win_row.addWidget(self.spin_hi)
        self.spin_step = QSpinBox()
        self.spin_step.setRange(1, 1000)
        self.spin_combine = QSpinBox()
        self.spin_combine.setRange(1, 999)
        self.spin_combine.setSingleStep(2)
        self.spin_combine.setValue(5 if direction == "mdc" else 3)
        win_row.addWidget(QLabel("  every"))
        win_row.addWidget(self.spin_step)
        win_row.addWidget(QLabel("line(s), summing"))
        win_row.addWidget(self.spin_combine)
        win_row.addWidget(QLabel("neighbours"))
        win_row.addStretch(1)
        form.addRow("Lines to fit", win_row)
        self.chk_refine = QCheckBox("second pass seeded from the first (follows the fitted dispersion)")
        self.chk_refine.setChecked(True)
        form.addRow("Refine", self.chk_refine)
        self.spin_iterations = QSpinBox()
        self.spin_iterations.setRange(10, 100000)
        self.spin_iterations.setValue(800)
        form.addRow("Max iterations", self.spin_iterations)
        self.edit_node = QLineEdit(node_name or peakfit_kind(direction))
        form.addRow("Store as /analysis/", self.edit_node)
        root.addWidget(box)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Fit")
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        if model is not None:
            for band in model.bands:
                self._add_row(band)
        else:
            self._add_row()
        if settings is not None:
            self.set_settings(settings)
        elif eres > 0:
            self._resolution_from_metadata()

    # -- rows -------------------------------------------------------------------
    def _add_row(self, band: pf.Band | None = None):
        r = self.table.rowCount()
        self.table.insertRow(r)
        self.table.setItem(r, 0, QTableWidgetItem(band.name if band else f"band{r + 1}"))
        shape = QComboBox()
        shape.addItems(list(pf.PEAK_SHAPES))
        shape.setCurrentText(band.shape if band else "voigt")
        self.table.setCellWidget(r, 1, shape)
        self.table.setItem(r, 2, QTableWidgetItem(band.share_width if band else ""))
        fix = QTableWidgetItem()
        fix.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
        fix.setCheckState(Qt.Checked if band and band.fix_width else Qt.Unchecked)
        self.table.setItem(r, 3, fix)
        fwhm0 = band.sorted_seeds()[0].fwhm if band and band.seeds else (0.035 if self.direction == "mdc" else 0.03)
        self.table.setItem(r, 4, QTableWidgetItem(f"{fwhm0:g}"))
        self.table.setItem(r, 5, QTableWidgetItem(seed_text(band) if band else ""))

    def _remove_row(self):
        r = self.table.currentRow()
        if r >= 0:
            self.table.removeRow(r)

    def _resolution_from_metadata(self):
        res = self._eres if self.direction == "edc" else self._eres / max(self.spin_vf.value(), 1e-9)
        self.spin_resolution.setValue(res)

    def resolution_source(self) -> str:
        if self.direction == "edc":
            return "info.Beamline.Eresolution" if abs(self.spin_resolution.value() - self._eres) < 1e-12 else "entered by hand"
        from_meta = abs(self.spin_resolution.value() - self._eres / max(self.spin_vf.value(), 1e-9)) < 1e-12
        return (f"info.Beamline.Eresolution / v_F={self.spin_vf.value():g} eV A" if from_meta
                else "entered by hand")

    def set_settings(self, settings: pf.FitSettings):
        self.cmb_background.setCurrentText(settings.background)
        self.spin_resolution.setValue(float(settings.resolution))
        self.cmb_weighting.setCurrentText(settings.weighting)
        self.chk_fermi.setChecked(bool(settings.fermi) and self.direction == "edc")
        self.spin_ef.setValue(float(settings.ef))
        self.spin_temperature.setValue(float(settings.temperature))
        self.spin_iterations.setValue(int(settings.max_iterations))

    # -- results ----------------------------------------------------------------
    def _text(self, r, c) -> str:
        item = self.table.item(r, c)
        return item.text().strip() if item else ""

    def band_model(self) -> pf.BandModel:
        x, y, values = self._cut
        bands = []
        for r in range(self.table.rowCount()):
            name = self._text(r, 0) or f"band{r + 1}"
            fwhm0 = float(self._text(r, 4) or 0.03)
            parsed = parse_seed_text(self._text(r, 5))
            if not parsed:
                raise ValueError(f"band {name!r} has no seeds")
            points = [(p[0], p[1]) for p in parsed]
            seeds = pf.seeds_from_points(values, (x, y), points, fwhm0, direction=self.direction,
                                         combine=self.spin_combine.value(), window=None)
            for seed, p in zip(seeds, parsed):
                if len(p) == 3:
                    seed.fwhm = float(p[2])
            bands.append(pf.Band(name, seeds, self.table.cellWidget(r, 1).currentText(),
                                 self._text(r, 2), self.table.item(r, 3).checkState() == Qt.Checked))
        return pf.BandModel(bands)

    def settings(self) -> pf.FitSettings:
        return pf.FitSettings(self.cmb_background.currentText(), float(self.spin_resolution.value()),
                              self.cmb_weighting.currentText(),
                              bool(self.chk_fermi.isChecked()) and self.direction == "edc",
                              float(self.spin_ef.value()), float(self.spin_temperature.value()),
                              int(self.spin_iterations.value()))

    def options(self) -> dict:
        lo, hi = sorted((float(self.spin_lo.value()), float(self.spin_hi.value())))
        node = re.sub(r"[^0-9A-Za-z_\-]+", "_", self.edit_node.text().strip()) or peakfit_kind(self.direction)
        return {"direction": self.direction, "bounds": (lo, hi), "step": int(self.spin_step.value()),
                "combine": int(self.spin_combine.value()), "refine": bool(self.chk_refine.isChecked()),
                "node": node, "resolution_source": self.resolution_source()}

    def _accept(self):
        try:
            model = self.band_model()
            self.settings()
        except Exception as exc:
            QMessageBox.warning(self, "Band model", str(exc))
            return
        if not model.bands:
            QMessageBox.warning(self, "Band model", "Add at least one band with seeds.")
            return
        self.accept()


# ---------------------------------------------------------------------------
# The panel
# ---------------------------------------------------------------------------
class PeakfitExplorerPanel(QWidget):
    """Dataset + fit pickers, MDC/EDC switch, Show-fit toggle, the E-k image with a draggable
    crosshair (left) and the fitted line with its parameter table (right)."""

    position_changed = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._tensor = None
        self._cut = None                 # (x, y, values[x, y]) with x = momentum, y = energy
        self._labels = ("kx", "Energy")
        self._units = ("1/A", "eV")
        self._dataset = None
        self._position = None
        self._dragging = False
        self._artists = {}
        self._worker = None
        self._progress = None
        self._init_ui()
        self.refresh_datasets()

    # -- UI ---------------------------------------------------------------------
    def _init_ui(self):
        root = QVBoxLayout(self)
        bar = QHBoxLayout()
        bar.addWidget(QLabel("Dataset"))
        self.cmb_dataset = QComboBox()
        self.cmb_dataset.setMinimumWidth(220)
        self.cmb_dataset.currentIndexChanged.connect(self._on_dataset_changed)
        bar.addWidget(self.cmb_dataset)
        self.btn_refresh = QPushButton("🔄")
        self.btn_refresh.setToolTip("Rescan the workspace")
        self.btn_refresh.clicked.connect(self.refresh_datasets)
        bar.addWidget(self.btn_refresh)
        bar.addWidget(QLabel("Fit"))
        self.cmb_node = QComboBox()
        self.cmb_node.setMinimumWidth(160)
        self.cmb_node.currentIndexChanged.connect(self._on_node_changed)
        bar.addWidget(self.cmb_node)
        self.cmb_direction = QComboBox()
        self.cmb_direction.addItems(["MDC", "EDC"])
        self.cmb_direction.currentIndexChanged.connect(self._on_direction_changed)
        bar.addWidget(self.cmb_direction)
        self.chk_show_fit = QCheckBox("Show fit")
        self.chk_show_fit.setChecked(True)
        self.chk_show_fit.toggled.connect(lambda _: self._redraw())
        bar.addWidget(self.chk_show_fit)
        self.btn_fit = QPushButton("Fit…")
        self.btn_fit.clicked.connect(self.open_fit_dialog)
        bar.addWidget(self.btn_fit)
        self.lbl_readout = QLabel("—")
        self.lbl_readout.setStyleSheet("font-family: monospace; padding: 0 8px;")
        bar.addWidget(self.lbl_readout)
        bar.addStretch(1)
        self.lbl_hint = QLabel("Drag on the image, or use the arrow keys. Dots = fitted centres.")
        self.lbl_hint.setStyleSheet("color: gray;")
        bar.addWidget(self.lbl_hint)
        root.addLayout(bar)

        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.setChildrenCollapsible(False)
        root.addWidget(self.splitter, 1)

        self.fig_left = Figure(layout="constrained")
        self.canvas_left = FigureCanvasQTAgg(self.fig_left)
        self.canvas_left.setFocusPolicy(Qt.ClickFocus)
        self.ax_left = self.fig_left.add_subplot(111)
        self.splitter.addWidget(self.canvas_left)
        self.canvas_left.mpl_connect("button_press_event", self._on_press)
        self.canvas_left.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas_left.mpl_connect("button_release_event", self._on_release)
        self.canvas_left.mpl_connect("key_press_event", self._on_key)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        self.fig_right = Figure(layout="constrained")
        self.canvas_right = FigureCanvasQTAgg(self.fig_right)
        self.ax_right = self.fig_right.add_subplot(111)
        right_layout.addWidget(self.canvas_right, 3)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["band", "centre", "± err", "FWHM", "height"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setMaximumHeight(180)
        right_layout.addWidget(self.table, 1)
        self.splitter.addWidget(right)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 1)

    # -- state ------------------------------------------------------------------
    @property
    def direction(self) -> str:
        return "mdc" if self.cmb_direction.currentIndex() == 0 else "edc"

    @property
    def dataset(self):
        return self._dataset

    @property
    def position(self):
        return self._position

    def current_source(self):
        """``(workspace name, node)`` of the cut shown, or ``(None, None)``."""
        data = self.cmb_dataset.currentData()
        return tuple(data) if data else (None, None)

    def find_dataset(self, name: str, node: str = "raw") -> int:
        """Combo index of ``name : /node`` (-1 if absent); QComboBox.findData cannot compare
        Python tuples, so look by hand."""
        for i in range(self.cmb_dataset.count()):
            if tuple(self.cmb_dataset.itemData(i) or ()) == (name, node):
                return i
        return -1

    def select_dataset(self, name: str, node: str = "raw") -> bool:
        idx = self.find_dataset(name, node)
        if idx >= 0:
            self.cmb_dataset.setCurrentIndex(idx)
        return idx >= 0

    def refresh_datasets(self):
        """List every 2-D-capable node of the workspace's spectroscopy trees."""
        current = self.cmb_dataset.currentData()
        self.cmb_dataset.blockSignals(True)
        self.cmb_dataset.clear()
        for name, item in list(global_workspace._data.items()):
            if not isinstance(item, dict) or item.get("type") != "spectroscopy_tree":
                continue
            nodes = ["raw"]
            try:
                if "data" in item["tree"]["processed"].ds:
                    nodes.append("processed")
            except Exception:
                pass
            nodes += [f"processed/{child}" for child in global_workspace.list_processed_children(name)]
            for node in nodes:
                self.cmb_dataset.addItem(f"{name} : /{node}", (name, node))
        self.cmb_dataset.blockSignals(False)
        if current is not None:
            idx = self.find_dataset(*tuple(current))
            if idx >= 0:
                self.cmb_dataset.setCurrentIndex(idx)
        self._on_dataset_changed()

    def _analysis_nodes(self, name: str) -> list:
        """``[(node, Dataset)]`` under /analysis of ``name`` that are peak-fit products of the
        current direction."""
        item = global_workspace.get(name)
        if not item or item.get("type") != "spectroscopy_tree":
            return []
        out = []
        try:
            analysis = item["tree"]["analysis"]
        except Exception:
            return out
        for node, child in analysis.children.items():
            ds = child.to_dataset() if hasattr(child, "to_dataset") else child.ds
            if is_peakfit_dataset(ds) and str(ds.attrs.get("direction")) == self.direction:
                out.append((str(node), ds))
        return sorted(out, key=lambda t: t[0])

    def refresh_nodes(self, select: str | None = None):
        name, _ = self.current_source()
        current = select if select is not None else self.cmb_node.currentText()
        self.cmb_node.blockSignals(True)
        self.cmb_node.clear()
        self.cmb_node.addItem(NO_FIT, None)
        if name:
            for node, ds in self._analysis_nodes(name):
                self.cmb_node.addItem(node, node)
        idx = self.cmb_node.findText(current) if current and current != NO_FIT else -1
        if idx < 0:
            idx = 1 if self.cmb_node.count() > 1 else 0      # the first stored fit, if any
        self.cmb_node.setCurrentIndex(idx)
        self.cmb_node.blockSignals(False)
        self._on_node_changed()

    def _on_dataset_changed(self, *_):
        name, node = self.current_source()
        self._tensor = global_workspace.pull_tensor_data(name, node) if name else None
        self._cut = None
        if self._tensor is not None and np.ndim(self._tensor.value) == 2:
            x, y, image, xl, yl, xu, yu = orient_cut(self._tensor)
            self._cut = (x, y, np.ascontiguousarray(image.T))
            self._labels, self._units = (xl, yl), (xu, yu)
            self._position = None
        self.btn_fit.setEnabled(self._cut is not None)
        self.refresh_nodes(select="")

    def _on_direction_changed(self, *_):
        self._position = None
        self.refresh_nodes(select="")

    def _on_node_changed(self, *_):
        name, _ = self.current_source()
        node = self.cmb_node.currentData()
        self._dataset = global_workspace.pull_analysis_data(name, node) if (name and node) else None
        self._redraw()

    def set_direction(self, direction: str):
        self.cmb_direction.setCurrentIndex(0 if direction == "mdc" else 1)

    # -- crosshair --------------------------------------------------------------
    def _line_axis(self) -> np.ndarray:
        if self._dataset is not None:
            return np.asarray(self._dataset.coords[str(self._dataset.attrs["line_axis"])].values, float)
        x, y, _ = self._cut
        return y if self.direction == "mdc" else x

    def _line_spacing(self) -> float:
        axis = self._line_axis()
        return float(abs(axis[1] - axis[0])) if axis.size > 1 else 1.0

    def set_position(self, position: float, *, snap: bool = True):
        """Move the crosshair to ``position`` on the line axis (snapped to the nearest fitted
        line when a fit is shown) and refresh the line view."""
        if self._cut is None:
            return
        x, y, _ = self._cut
        axis = y if self.direction == "mdc" else x
        position = float(np.clip(position, np.nanmin(axis), np.nanmax(axis)))
        if snap and self._dataset is not None:
            line_axis = self._line_axis()
            i = int(np.argmin(np.abs(line_axis - position)))
            if abs(line_axis[i] - position) <= 1.5 * self._line_spacing():
                position = float(line_axis[i])
        self._position = position
        cross = self._artists.get("crosshair")
        if cross is not None:
            if self.direction == "mdc":
                cross.set_ydata([position, position])
            else:
                cross.set_xdata([position, position])
            self.canvas_left.draw_idle()
        self._draw_right()
        self.position_changed.emit(position)

    def step_line(self, n: int):
        """Arrow-key step: ``n`` lines along the line axis."""
        if self._position is None:
            return
        self.set_position(self._position + n * self._line_spacing())

    def _position_from_event(self, event):
        if event.inaxes is not self.ax_left or self._cut is None:
            return
        value = event.ydata if self.direction == "mdc" else event.xdata
        if value is not None:
            self.set_position(float(value))

    def _on_press(self, event):
        if event.button != 1:
            return
        self._dragging = True
        self.canvas_left.setFocus()
        self._position_from_event(event)

    def _on_motion(self, event):
        if self._dragging:
            self._position_from_event(event)

    def _on_release(self, _event):
        self._dragging = False

    def _on_key(self, event):
        forward = ("up", "right") if self.direction == "mdc" else ("right", "up")
        backward = ("down", "left") if self.direction == "mdc" else ("left", "down")
        if event.key in forward:
            self.step_line(+1)
        elif event.key in backward:
            self.step_line(-1)

    # -- drawing ----------------------------------------------------------------
    def _redraw(self):
        self.ax_left.clear()
        self._artists = {}
        if self._cut is None:
            self.ax_left.set_title("Pick a 2-D cut (k × E) from the workspace", fontsize=9)
            self.canvas_left.draw_idle()
            self.ax_right.clear()
            self.canvas_right.draw_idle()
            self.table.setRowCount(0)
            self.lbl_readout.setText("—")
            return
        axis = self._line_axis()
        if self._position is None:
            default = 0.0 if self.direction == "mdc" and axis.min() < 0 < axis.max() else axis[len(axis) // 2]
            self._position = float(axis[int(np.argmin(np.abs(axis - default)))])
        elif self._dataset is not None:
            # a newly chosen fit: bring the crosshair into its window
            self._position = float(np.clip(self._position, np.nanmin(axis), np.nanmax(axis)))
        self._artists = draw_cut_with_tracks(self.ax_left, self._tensor, self._dataset,
                                             self._position, direction=self.direction,
                                             show_fit=self.chk_show_fit.isChecked())
        name, node = self.current_source()
        self.ax_left.set_title(f"{name} : /{node}", fontsize=9)
        self.canvas_left.draw_idle()
        self.set_position(self._position)

    def _draw_right(self):
        x, y, values = self._cut
        rows = []
        index = None
        if self._dataset is not None:
            index = nearest_line(self._dataset, self._position)
            line_axis = self._line_axis()
            if abs(line_axis[index] - self._position) > 1.5 * self._line_spacing():
                index = None
        if index is not None:
            rows = draw_line_fit(self.ax_right, self._dataset, index, direction=self.direction,
                                 show_fit=self.chk_show_fit.isChecked())
            colours = band_colours(self._dataset)
        else:
            actual, along, line = pf.line_at(values, (x, y), self._position, direction=self.direction)
            label = self._labels[0] if self.direction == "mdc" else self._labels[1]
            unit = self._units[0] if self.direction == "mdc" else self._units[1]
            kind = "MDC" if self.direction == "mdc" else "EDC"
            line_label = self._labels[1] if self.direction == "mdc" else self._labels[0]
            draw_line_fit(self.ax_right, None, None, direction=self.direction,
                          raw_line=(along, line, f"{label} ({unit})" if unit else label),
                          title=f"{kind} at {line_label} = {actual:+.4f}   (no fit here)")
            colours = []
        self.canvas_right.draw_idle()
        self._fill_table(rows, colours)
        line_label = self._labels[1] if self.direction == "mdc" else self._labels[0]
        line_unit = self._units[1] if self.direction == "mdc" else self._units[0]
        self.lbl_readout.setText(f"{line_label} = {self._position:+.4f} {line_unit}")

    def _fill_table(self, rows, colours):
        self.table.setRowCount(len(rows))
        band_index = {}
        if self._dataset is not None:
            band_index = {str(n): j for j, n in enumerate(self._dataset.coords["band"].values)}
        for r, (name, centre, err, fwhm, height) in enumerate(rows):
            item = QTableWidgetItem(str(name))
            j = band_index.get(str(name), r)
            if colours:
                item.setBackground(QColor(colours[j % len(colours)]))
                item.setForeground(QColor("white"))
            self.table.setItem(r, 0, item)
            for c, value in enumerate((centre, err, fwhm, height), start=1):
                cell = QTableWidgetItem(f"{value:.4f}" if abs(value) < 100 else f"{value:.4g}")
                cell.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.table.setItem(r, c, cell)

    # -- fitting ----------------------------------------------------------------
    def open_fit_dialog(self):
        if self._cut is None:
            QMessageBox.information(self, "Fit", "Pick a 2-D cut first.")
            return
        model = settings = None
        node = self.cmb_node.currentData() or ""
        if self._dataset is not None:
            try:
                model, settings = band_model_of(self._dataset), settings_of(self._dataset)
            except Exception:
                model = settings = None
        dialog = BandModelDialog(self.direction, self._cut, self._tensor.metadata, model, settings,
                                 node, parent=self)
        if dialog.exec() != QDialog.Accepted:
            return
        try:
            self.start_fit(dialog.band_model(), dialog.settings(), dialog.options())
        except Exception as exc:
            QMessageBox.warning(self, "Fit", str(exc))

    def start_fit(self, model: pf.BandModel, settings: pf.FitSettings, options: dict):
        """Run ``fit_series`` in a worker thread; the result lands in /analysis/<options['node']>."""
        if self._worker is not None and self._worker.isRunning():
            QMessageBox.information(self, "Fit", "A fit is already running.")
            return
        x, y, values = self._cut
        self._fit_options = dict(options)
        self._fit_model, self._fit_settings = model, settings
        self._worker = FitWorker(values, (x, y), model, settings, options, self._labels, self._units,
                                 parent=self)
        self._progress = QProgressDialog("Fitting lines…", "Cancel", 0, 100, self)
        self._progress.setWindowModality(Qt.WindowModal)
        self._progress.setMinimumDuration(300)
        self._progress.canceled.connect(self._worker.cancel)
        self._worker.progress.connect(self._on_fit_progress)
        self._worker.finished_series.connect(self._on_fit_done)
        self._worker.failed.connect(self._on_fit_failed)
        self.btn_fit.setEnabled(False)
        self._worker.start()

    def _on_fit_progress(self, done: int, total: int):
        if self._progress is not None:
            self._progress.setMaximum(max(total, 1))
            self._progress.setValue(min(done, total))

    def _finish_progress(self):
        if self._progress is not None:
            self._progress.reset()
            self._progress = None
        self.btn_fit.setEnabled(self._cut is not None)

    def _on_fit_failed(self, message: str):
        self._finish_progress()
        if message != "cancelled":
            QMessageBox.critical(self, "Fit failed", message)

    def _on_fit_done(self, series):
        self._finish_progress()
        self.store_series(series, self._fit_model, self._fit_settings, self._fit_options)

    def store_series(self, series, model, settings, options: dict) -> str | None:
        """Build the product Dataset and write it under /analysis of the current tree."""
        name, node = self.current_source()
        if not name:
            return None
        lo, hi = options.get("bounds", (None, None))
        extra = {"source_dataset": name, "source_node": node,
                 "fit_window": [lo, hi], "step": options.get("step", 1),
                 "combine": options.get("combine", 1), "refine": options.get("refine", True)}
        temperature = arpes_metadata_number(self._tensor.metadata, "info", "Temperature", "Sample")
        if temperature is not None:
            extra["temperature"] = temperature
        ds = peakfit_dataset(series, model, settings,
                             resolution_source=options.get("resolution_source", ""), extra_attrs=extra)
        target = options.get("node") or peakfit_kind(series.direction)
        if not global_workspace.write_analysis_data(name, target, ds):
            QMessageBox.critical(self, "Fit", f"Could not write /analysis/{target} of {name}.")
            return None
        n_fitted = int(ds.attrs.get("n_fitted", 0))
        chi2 = float(np.nanmedian(series.chi2)) if np.isfinite(series.chi2).any() else float("nan")
        self.lbl_hint.setText(f"Stored /analysis/{target}: {n_fitted} lines, median χ² {chi2:.3g}")
        self.refresh_nodes(select=target)
        return target
