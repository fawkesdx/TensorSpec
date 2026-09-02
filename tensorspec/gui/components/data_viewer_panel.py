import weakref

# Global registry to track all active Data Viewer windows for crosshair syncing
GLOBAL_SYNC_REGISTRY = weakref.WeakSet()
import numpy as np
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, 
                               QComboBox, QPushButton, QCheckBox, QFrame, QMenu, QSpinBox, 
                               QFileDialog, QMessageBox, QDoubleSpinBox, QGridLayout, QMainWindow, QSplitter)
from PySide6.QtCore import Qt
from PySide6.QtGui import QCursor
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.patches as patches

from tensorspec.core.data_models import TensorData


def is_ml_label_layer(layer_name: str) -> bool:
    return (
        layer_name.startswith("domains_")
        or layer_name.startswith("Labels_")
        or layer_name == "Supervised Probabilities"
    )


def ml_label_spatial_map(layer_data: np.ndarray, value_shape: tuple) -> np.ndarray | None:
    """Reshape flat ML domain vectors (or prob cubes) to (nY, nX) for spatial maps."""
    if len(value_shape) != 4:
        return None
    nY, nX = value_shape[2], value_shape[3]
    arr = np.asarray(layer_data)
    if arr.ndim == 1 and arr.size == nY * nX:
        return arr.reshape(nY, nX)
    if arr.ndim == 3 and arr.shape[:2] == (nY, nX):
        return np.argmax(arr, axis=2)
    return None

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=4, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi, layout='constrained')
        super().__init__(fig)
        from PySide6.QtWidgets import QSizePolicy
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.updateGeometry()

class SliceWidget(QFrame):
    """A modular 2D viewer that synchronizes with the Global N-Dimensional State."""
    def __init__(self, parent_panel, x_idx: int, y_idx: int, grid_row: int = 0, grid_col: int = 0):
        super().__init__(parent_panel)
        self.parent_panel = parent_panel
        self.tensor_data = parent_panel.tensor_data
        
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        self.setLineWidth(2)
        from PySide6.QtWidgets import QSizePolicy
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(200, 150)
        
        self.x_idx = x_idx
        self.y_idx = y_idx
        
        # Track position in the puzzle grid
        self.grid_row = grid_row
        self.grid_col = grid_col
        
        self.slider_widgets = {}
        self.spin_widgets = {}
        
        self._init_ui()
        self._populate_dropdowns()

        # Register for cross-window syncing
        GLOBAL_SYNC_REGISTRY.add(self)
        self._is_syncing = False # Prevents infinite feedback loops
        
    def _init_ui(self):
        layout = QVBoxLayout(self)
        
        # --- Top Controls ---
        ctrl_layout_v = QVBoxLayout()
        
        # ROW 1 (Always Visible)
        row1 = QHBoxLayout()
        self.combo_x = QComboBox()
        self.combo_y = QComboBox()
        self.combo_layer = QComboBox()
        self.combo_x.currentIndexChanged.connect(self._on_dropdown_changed)
        self.combo_y.currentIndexChanged.connect(self._on_dropdown_changed)
        self.combo_layer.currentIndexChanged.connect(self.redraw)
        
        row1.addWidget(QLabel("X:"))
        row1.addWidget(self.combo_x)
        row1.addWidget(QLabel("Y:"))
        row1.addWidget(self.combo_y)
        row1.addWidget(QLabel("Layer:"))
        row1.addWidget(self.combo_layer)
        
        self.combo_aspect = QComboBox()
        self.combo_aspect.addItems(["Stretch (Auto)", "Square (Equal)"])
        self.combo_aspect.currentIndexChanged.connect(self.update_aspect)
        row1.addWidget(self.combo_aspect)

        self.chk_profiles = QCheckBox("XY Profiles")
        self.chk_profiles.stateChanged.connect(self.toggle_profiles)
        row1.addWidget(self.chk_profiles)
        
        self.combo_profile_mode = QComboBox()
        self.combo_profile_mode.addItems(["Raw (Sum)", "Mean", "Normalized to Max"])
        self.combo_profile_mode.currentIndexChanged.connect(self.redraw)
        row1.addWidget(self.combo_profile_mode)
        
        # NEW: Zen Mode Toggle Button
        self.btn_toggle = QPushButton("👁️ Toggle UI")
        self.btn_toggle.setStyleSheet("font-weight: bold; background-color: #5bc0de; color: black;")
        self.btn_toggle.clicked.connect(self.toggle_ui_controls)
        row1.addWidget(self.btn_toggle)
        
        btn_close = QPushButton("✕")
        btn_close.setFixedWidth(25)
        btn_close.setStyleSheet("background-color: #d9534f; color: white;")
        btn_close.clicked.connect(self.close_widget)
        row1.addWidget(btn_close)
        
        # ROW 2 (Wrapped in a collapsible widget)
        self.row2_widget = QWidget()
        row2 = QHBoxLayout(self.row2_widget)
        row2.setContentsMargins(0, 0, 0, 0)
        
        row2.addWidget(QLabel("ΔX:"))
        self.spin_dx = QSpinBox(); self.spin_dx.setRange(0, 100); self.spin_dx.setSuffix(" px")
        self.slider_dx = QSlider(Qt.Horizontal); self.slider_dx.setRange(0, 100); self.slider_dx.setFixedWidth(50)
        self.spin_dx.valueChanged.connect(self.slider_dx.setValue)
        self.slider_dx.valueChanged.connect(self.spin_dx.setValue)
        self.spin_dx.valueChanged.connect(self.redraw)
        row2.addWidget(self.spin_dx); row2.addWidget(self.slider_dx)
        
        row2.addWidget(QLabel("ΔY:"))
        self.spin_dy = QSpinBox(); self.spin_dy.setRange(0, 100); self.spin_dy.setSuffix(" px")
        self.slider_dy = QSlider(Qt.Horizontal); self.slider_dy.setRange(0, 100); self.slider_dy.setFixedWidth(50)
        self.spin_dy.valueChanged.connect(self.slider_dy.setValue)
        self.slider_dy.valueChanged.connect(self.spin_dy.setValue)
        self.spin_dy.valueChanged.connect(self.redraw)
        row2.addWidget(self.spin_dy); row2.addWidget(self.slider_dy)
        
        row2.addWidget(QLabel("  |  "))
        self.chk_ortho = QCheckBox("Extract Orthogonal:")
        self.chk_ortho.stateChanged.connect(self.toggle_profiles)
        self.combo_ortho = QComboBox()
        self.combo_ortho.currentIndexChanged.connect(self.redraw)
        row2.addWidget(self.chk_ortho)
        row2.addWidget(self.combo_ortho)
        
        ctrl_layout_v.addLayout(row1)
        ctrl_layout_v.addWidget(self.row2_widget)
        layout.addLayout(ctrl_layout_v)
        
        # --- Canvas Setup (2x3 Grid) ---
        self.canvas = MplCanvas(self)
        self.gs = self.canvas.figure.add_gridspec(2, 3, width_ratios=[4, 1, 2], height_ratios=[4, 1])
        
        self.ax_main = self.canvas.figure.add_subplot(self.gs[0, 0])
        self.ax_prof_y = self.canvas.figure.add_subplot(self.gs[0, 1], sharey=self.ax_main)
        self.ax_prof_x = self.canvas.figure.add_subplot(self.gs[1, 0], sharex=self.ax_main)
        self.ax_prof_ortho = self.canvas.figure.add_subplot(self.gs[:, 2]) 
        
        self.im_main = self.ax_main.imshow(np.zeros((10, 10)), origin='lower', aspect='auto', cmap='magma')
        self.line_x, = self.ax_prof_x.plot([], [], 'b-')
        self.line_y, = self.ax_prof_y.plot([], [], 'r-')
        self.line_ortho, = self.ax_prof_ortho.plot([], [], 'g-', linewidth=1.5)
        
        self.vline = self.ax_main.axvline(0, color='cyan', ls='--', alpha=0.7)
        self.hline = self.ax_main.axhline(0, color='cyan', ls='--', alpha=0.7)
        self.rect_window = patches.Rectangle((0,0), 0, 0, linewidth=1.5, edgecolor='cyan', facecolor='cyan', alpha=0.2, linestyle=':')
        self.ax_main.add_patch(self.rect_window)
        
        self.ax_prof_y.set_visible(False)
        self.ax_prof_x.set_visible(False)
        self.ax_prof_ortho.set_visible(False)
        
        self.canvas.mpl_connect('button_press_event', self._on_click)
        self.canvas.mpl_connect('motion_notify_event', self._on_drag) # NEW: Listen for mouse movement
        layout.addWidget(self.canvas)
        layout.setStretchFactor(self.canvas, 1) # Force canvas to expand vertically!
        
        # --- Sliders Wrapper (Collapsible) ---
        self.sliders_widget = QWidget()
        self.sliders_layout = QVBoxLayout(self.sliders_widget)
        self.sliders_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.sliders_widget)
        
        # Automatically collapse GridSpec unused space on boot
        self.toggle_profiles()

        # --- NEW: Sync Crosshairs Checkbox ---
        self.chk_sync = QCheckBox("🔗 Sync")
        self.chk_sync.setToolTip("Lock crosshairs across all open windows with matching axes.")
        row1.addWidget(self.chk_sync)

    def _populate_dropdowns(self):
        self.combo_x.blockSignals(True)
        self.combo_y.blockSignals(True)
        self.combo_layer.blockSignals(True)
        
        self.combo_layer.clear()
        self.combo_layer.addItem("Intensity")
        layers = self.tensor_data.metadata.get('layers', {}) if hasattr(self.tensor_data, 'metadata') and self.tensor_data.metadata else {}
        for layer_name in layers:
            self.combo_layer.addItem(layer_name)

        for idx, (label, unit) in enumerate(zip(self.tensor_data.labels, self.tensor_data.units)):
            display = f"{label} ({unit})" if unit else label
            self.combo_x.addItem(display, userData=idx)
            self.combo_y.addItem(display, userData=idx)
            
        self.combo_x.setCurrentIndex(self.x_idx)
        self.combo_y.setCurrentIndex(self.y_idx)
        self.combo_x.blockSignals(False)
        self.combo_y.blockSignals(False)
        self.combo_layer.blockSignals(False)
        self._update_ortho_combo()
        self._rebuild_sliders()

    def _update_ortho_combo(self):
        self.combo_ortho.blockSignals(True)
        self.combo_ortho.clear()
        for i in range(self.tensor_data.ndim):
            if i not in (self.x_idx, self.y_idx):
                label = self.tensor_data.labels[i]
                unit = self.tensor_data.units[i]
                display = f"{label} ({unit})" if unit else label
                self.combo_ortho.addItem(display, userData=i)
                
        has_rem_dims = self.combo_ortho.count() > 0
        self.chk_ortho.setEnabled(has_rem_dims)
        self.combo_ortho.setEnabled(has_rem_dims)
        if not has_rem_dims: self.chk_ortho.setChecked(False)
        self.combo_ortho.blockSignals(False)

    def _rebuild_sliders(self):
        while self.sliders_layout.count():
            item = self.sliders_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()
            elif item.layout():
                while item.layout().count():
                    sub = item.layout().takeAt(0)
                    if sub.widget(): sub.widget().deleteLater()
                    
        self.slider_widgets.clear()
        self.spin_widgets.clear()
        
        for i in range(self.tensor_data.ndim):
            if i in (self.x_idx, self.y_idx): continue
            
            row = QHBoxLayout()
            lbl = QLabel(f"{self.tensor_data.labels[i]}:")
            lbl.setMinimumWidth(80)
            ax_arr = self.tensor_data.axes[i]
            current_idx = self.parent_panel.global_coords[i]
            
            spin = QDoubleSpinBox()
            spin.setDecimals(3)
            spin.setRange(min(ax_arr), max(ax_arr))
            spin.setValue(ax_arr[current_idx])
            spin.setSuffix(f" {self.tensor_data.units[i]}")
            spin.setKeyboardTracking(False)
            
            slider = QSlider(Qt.Horizontal)
            slider.setRange(0, len(ax_arr)-1)
            slider.setValue(current_idx)
            
            slider.valueChanged.connect(
                lambda val, dim=i: self.parent_panel.update_global_coord(
                    dim, val, cross_sync=self.chk_sync.isChecked()
                )
            )
            spin.valueChanged.connect(
                lambda val, dim=i, arr=ax_arr: self.parent_panel.update_global_coord(
                    dim, int((np.abs(arr - val)).argmin()), cross_sync=self.chk_sync.isChecked()
                )
            )
            
            row.addWidget(lbl); row.addWidget(spin); row.addWidget(slider)
            self.sliders_layout.addLayout(row)
            self.slider_widgets[i] = slider
            self.spin_widgets[i] = spin

    def _on_dropdown_changed(self):
        self.x_idx = self.combo_x.currentData()
        self.y_idx = self.combo_y.currentData()
        self._update_ortho_combo()
        self._rebuild_sliders()
        self.parent_panel.broadcast_redraw()

    def toggle_ui_controls(self):
        """Collapses or expands the integration controls and dimension sliders."""
        is_visible = self.row2_widget.isVisible()
        self.row2_widget.setVisible(not is_visible)
        self.sliders_widget.setVisible(not is_visible)

    def toggle_profiles(self):
        """Toggles profile visibility and reclaims space for the main canvas."""
        vis_xy = self.chk_profiles.isChecked()
        vis_ortho = self.chk_ortho.isChecked()
        
        self.ax_prof_x.set_visible(vis_xy)
        self.ax_prof_y.set_visible(vis_xy)
        self.ax_prof_ortho.set_visible(vis_ortho)
        
        # Rebuild GridSpec dynamically so constrained_layout works perfectly
        if vis_xy and vis_ortho:
            gs = self.canvas.figure.add_gridspec(2, 3, width_ratios=[4, 1, 2], height_ratios=[4, 1])
            self.ax_main.set_subplotspec(gs[0, 0])
            self.ax_prof_y.set_subplotspec(gs[0, 1])
            self.ax_prof_x.set_subplotspec(gs[1, 0])
            self.ax_prof_ortho.set_subplotspec(gs[:, 2])
        elif vis_xy:
            gs = self.canvas.figure.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[4, 1])
            self.ax_main.set_subplotspec(gs[0, 0])
            self.ax_prof_y.set_subplotspec(gs[0, 1])
            self.ax_prof_x.set_subplotspec(gs[1, 0])
        elif vis_ortho:
            gs = self.canvas.figure.add_gridspec(1, 2, width_ratios=[4, 2])
            self.ax_main.set_subplotspec(gs[0, 0])
            self.ax_prof_ortho.set_subplotspec(gs[0, 1])
        else:
            gs = self.canvas.figure.add_gridspec(1, 1)
            self.ax_main.set_subplotspec(gs[0, 0])
            
        self.gs = gs
        self.canvas.draw_idle()
        self.redraw()

    def sync_sliders_to_global(self):
        for i, slider in self.slider_widgets.items():
            slider.blockSignals(True)
            slider.setValue(self.parent_panel.global_coords[i])
            slider.blockSignals(False)
            
        for i, spin in self.spin_widgets.items():
            spin.blockSignals(True)
            val_idx = self.parent_panel.global_coords[i]
            spin.setValue(self.tensor_data.axes[i][val_idx])
            spin.blockSignals(False)

    def update_aspect(self):
        mode = "equal" if "Square" in self.combo_aspect.currentText() else "auto"
        self.ax_main.set_aspect(mode)
        self.canvas.draw_idle()

    def redraw(self):
        # --- NEW: Prevent 1D shape crash ---
        if self.x_idx == self.y_idx:
            return
        
        self.sync_sliders_to_global()
        slices = []
        for i in range(self.tensor_data.ndim):
            if i in (self.x_idx, self.y_idx):
                slices.append(slice(None))
            else:
                slices.append(self.parent_panel.global_coords[i])
                
        layer_text = self.combo_layer.currentText()
        layers = self.tensor_data.metadata.get('layers', {}) if self.tensor_data.metadata else {}
        if layer_text != "Intensity" and layer_text in layers:
            layer_data = layers[layer_text]
            spatial = ml_label_spatial_map(np.asarray(layer_data), self.tensor_data.value.shape)
            if spatial is not None and {self.x_idx, self.y_idx} <= {2, 3}:
                sliced = spatial
            else:
                data_source = np.asarray(layer_data)
                if data_source.ndim == self.tensor_data.ndim:
                    sliced = data_source[tuple(slices)]
                elif spatial is not None:
                    sliced = spatial
                else:
                    sliced = self.tensor_data.value[tuple(slices)]
        else:
            sliced = self.tensor_data.value[tuple(slices)]
        if self.x_idx < self.y_idx: sliced = sliced.T
        
        x_arr, y_arr = self.tensor_data.axes[self.x_idx], self.tensor_data.axes[self.y_idx]
        dx_step = (x_arr[-1] - x_arr[0]) / max(1, len(x_arr) - 1) if len(x_arr) > 1 else 0.1
        dy_step = (y_arr[-1] - y_arr[0]) / max(1, len(y_arr) - 1) if len(y_arr) > 1 else 0.1
        extent = (x_arr[0] - dx_step/2, x_arr[-1] + dx_step/2, y_arr[0] - dy_step/2, y_arr[-1] + dy_step/2)
        
        self.im_main.set_data(sliced)
        self.im_main.set_extent(extent)
        if is_ml_label_layer(layer_text):
            self.im_main.set_clim(-0.5, 19.5)
            try:
                import matplotlib as mpl
                self.im_main.set_cmap(mpl.colormaps["tab20"].with_extremes(under="#333333"))
            except Exception:
                self.im_main.set_cmap("tab20")
        else:
            vmin, vmax = np.nanmin(sliced), np.nanmax(sliced)
            self.im_main.set_clim(vmin, vmax)
            self.im_main.set_cmap('magma')
        
        self.ax_main.set_xlabel(self.combo_x.currentText())
        self.ax_main.set_ylabel(self.combo_y.currentText())
        
        x_cross = self.parent_panel.global_coords[self.x_idx]
        y_cross = self.parent_panel.global_coords[self.y_idx]
        dx_px = self.spin_dx.value()
        dy_px = self.spin_dy.value()
        
        x1, x2 = max(0, x_cross - dx_px), min(sliced.shape[1], x_cross + dx_px + 1)
        y1, y2 = max(0, y_cross - dy_px), min(sliced.shape[0], y_cross + dy_px + 1)
        
        self.vline.set_xdata([x_arr[x_cross]])
        self.hline.set_ydata([y_arr[y_cross]])
        
        rect_x = x_arr[x1] - dx_step/2
        rect_y = y_arr[y1] - dy_step/2
        rect_w = (x2 - x1) * dx_step
        rect_h = (y2 - y1) * dy_step
        self.rect_window.set_bounds(rect_x, rect_y, rect_w, rect_h)
        
        calc_mode = self.combo_profile_mode.currentText()

        if self.chk_profiles.isChecked():
            if calc_mode == "Mean":
                prof_x = np.mean(sliced[y1:y2, :], axis=0)
                prof_y = np.mean(sliced[:, x1:x2], axis=1)
            else:
                prof_x = np.sum(sliced[y1:y2, :], axis=0)
                prof_y = np.sum(sliced[:, x1:x2], axis=1)
                
            if calc_mode == "Normalized to Max":
                mx, my = np.nanmax(prof_x), np.nanmax(prof_y)
                if mx != 0: prof_x = prof_x / mx
                if my != 0: prof_y = prof_y / my
            
            self.line_x.set_data(x_arr, prof_x)
            self.line_y.set_data(prof_y, y_arr)
            self.ax_prof_x.relim()
            self.ax_prof_x.autoscale(enable=True, axis='y')
            self.ax_prof_y.relim()
            self.ax_prof_y.autoscale(enable=True, axis='x')
            
        if self.chk_ortho.isChecked() and self.combo_ortho.count() > 0:
            ortho_idx = self.combo_ortho.currentData()
            slices_ortho = []
            for i in range(self.tensor_data.ndim):
                if i == ortho_idx:
                    slices_ortho.append(slice(None))
                elif i == self.x_idx:
                    slices_ortho.append(slice(x1, x2))
                elif i == self.y_idx:
                    slices_ortho.append(slice(y1, y2))
                else:
                    val = self.parent_panel.global_coords[i]
                    slices_ortho.append(slice(val, val+1)) 
                    
            ortho_chunk = self.tensor_data.value[tuple(slices_ortho)]
            eval_axes = tuple(i for i in range(self.tensor_data.ndim) if i != ortho_idx)
            
            if calc_mode == "Mean":
                prof_ortho = np.mean(ortho_chunk, axis=eval_axes)
            else:
                prof_ortho = np.sum(ortho_chunk, axis=eval_axes)
                
            if calc_mode == "Normalized to Max":
                mo = np.nanmax(prof_ortho)
                if mo != 0: prof_ortho = prof_ortho / mo
            
            ortho_arr = self.tensor_data.axes[ortho_idx]
            self.line_ortho.set_data(ortho_arr, prof_ortho)
            self.ax_prof_ortho.set_xlabel(self.combo_ortho.currentText())
            self.ax_prof_ortho.set_ylabel(f"Intensity ({calc_mode})")
            self.ax_prof_ortho.relim(); self.ax_prof_ortho.autoscale_view()
        
        self.ax_main.set_xlim(extent[0], extent[1])
        self.ax_main.set_ylim(extent[2], extent[3])
        self.canvas.draw_idle()

    

    def _on_click(self, event):
        if event.inaxes != self.ax_main: return
        if event.xdata is None or event.ydata is None: return
        
        # --- NEW: Double-click to instantly spawn a new panel to the right ---
        if event.dblclick:
            dashboard = self.parentWidget()
            while dashboard and not hasattr(dashboard, 'spawn_view'):
                dashboard = dashboard.parentWidget()
            if dashboard:
                dashboard.spawn_view(self.x_idx, self.y_idx, self, "Right")
            return
            
        # 1. Map the raw mouse coordinates to the nearest valid index in the data arrays
        x_arr = self.tensor_data.axes[self.x_idx]
        y_arr = self.tensor_data.axes[self.y_idx]
        
        new_x_idx = int((np.abs(x_arr - event.xdata)).argmin())
        new_y_idx = int((np.abs(y_arr - event.ydata)).argmin())
        
        # 2. Update the global state manager (cross-dataset sync only if 🔗 Sync is on)
        want_sync = bool(getattr(self, 'chk_sync', None) and self.chk_sync.isChecked())
        self.parent_panel.update_global_coord(
            self.x_idx, new_x_idx, broadcast=False, cross_sync=want_sync
        )
        self.parent_panel.update_global_coord(
            self.y_idx, new_y_idx, broadcast=True, cross_sync=want_sync
        )

        # 3. Broadcast physical crosshair coords to other Sync-enabled windows
        if want_sync and not self._is_syncing:
            x_label = self.combo_x.currentText()
            y_label = self.combo_y.currentText()
            
            for widget in list(GLOBAL_SYNC_REGISTRY):
                try:
                    if widget is not self and hasattr(widget, 'chk_sync') and widget.chk_sync.isChecked():
                        widget.receive_sync(x_label, event.xdata, y_label, event.ydata)
                except RuntimeError:
                    # Clean up deleted C++ objects silently
                    GLOBAL_SYNC_REGISTRY.discard(widget)

    def receive_sync(self, source_x_label, source_x_val, source_y_label, source_y_val):
        """Receives crosshair coordinates from another window and updates if physical axes match."""
        if self._is_syncing: return
        self._is_syncing = True
        
        needs_redraw = False
        my_x_label = self.combo_x.currentText()
        my_y_label = self.combo_y.currentText()
        
        # Safely fetch current crosshair positions
        try:
            new_x = self.vline.get_xdata()[0]
            new_y = self.hline.get_ydata()[0]
        except (IndexError, TypeError):
            new_x, new_y = 0, 0
        
        # Check if the incoming X or Y matches our local X axis
        if my_x_label == source_x_label:
            new_x = source_x_val
            needs_redraw = True
        elif my_x_label == source_y_label:
            new_x = source_y_val
            needs_redraw = True
            
        # Check if the incoming X or Y matches our local Y axis
        if my_y_label == source_y_label:
            new_y = source_y_val
            needs_redraw = True
        elif my_y_label == source_x_label:
            new_y = source_x_val
            needs_redraw = True
            
        if needs_redraw:
            self.vline.set_xdata([new_x, new_x])
            self.hline.set_ydata([new_y, new_y])
            self.redraw()
            
        self._is_syncing = False
    
    def _on_drag(self, event):
        """Allows the crosshair to update continuously while clicking and dragging."""
        # event.button == 1 ensures this only triggers if the left mouse button is held down
        if event.inaxes == self.ax_main and event.button == 1:
            self._on_click(event)
    
    def export_profiles(self):
        # (Export logic remains perfectly intact)
        if not self.chk_profiles.isChecked():
            QMessageBox.warning(self, "Export Error", "Please enable '1D Profiles' (checkbox at top) before exporting.")
            return

        path, _ = QFileDialog.getSaveFileName(self, "Export 1D Profiles", "spectra_profiles.csv", "CSV Files (*.csv);;Text Files (*.txt)")
        if not path: return

        try:
            slices = []
            for i in range(self.tensor_data.ndim):
                if i in (self.x_idx, self.y_idx): slices.append(slice(None))
                else: slices.append(self.parent_panel.global_coords[i])
                    
            sliced = self.tensor_data.value[tuple(slices)]
            if self.x_idx < self.y_idx: sliced = sliced.T
            
            x_arr, y_arr = self.tensor_data.axes[self.x_idx], self.tensor_data.axes[self.y_idx]
            x_cross = self.parent_panel.global_coords[self.x_idx]
            y_cross = self.parent_panel.global_coords[self.y_idx]
            
            x1, x2 = max(0, x_cross - self.spin_dx.value()), min(sliced.shape[1], x_cross + self.spin_dx.value() + 1)
            y1, y2 = max(0, y_cross - self.spin_dy.value()), min(sliced.shape[0], y_cross + self.spin_dy.value() + 1)
            
            calc_mode = self.combo_profile_mode.currentText()
            if calc_mode == "Mean":
                prof_x = np.mean(sliced[y1:y2, :], axis=0)
                prof_y = np.mean(sliced[:, x1:x2], axis=1)
            else:
                prof_x = np.sum(sliced[y1:y2, :], axis=0)
                prof_y = np.sum(sliced[:, x1:x2], axis=1)
                
            if calc_mode == "Normalized to Max":
                mx, my = np.nanmax(prof_x), np.nanmax(prof_y)
                if mx != 0: prof_x = prof_x / mx
                if my != 0: prof_y = prof_y / my

            max_len = max(len(x_arr), len(y_arr))
            out_x_ax = np.pad(x_arr, (0, max_len - len(x_arr)), constant_values=np.nan)
            out_x_pr = np.pad(prof_x, (0, max_len - len(prof_x)), constant_values=np.nan)
            out_y_ax = np.pad(y_arr, (0, max_len - len(y_arr)), constant_values=np.nan)
            out_y_pr = np.pad(prof_y, (0, max_len - len(prof_y)), constant_values=np.nan)

            x_lbl = f"Horizontal_{self.tensor_data.labels[self.x_idx]}"
            y_lbl = f"Vertical_{self.tensor_data.labels[self.y_idx]}"
            header = f"{x_lbl}_{self.tensor_data.units[self.x_idx]},{x_lbl}_Intensity,{y_lbl}_{self.tensor_data.units[self.y_idx]},{y_lbl}_Intensity"

            data_matrix = np.column_stack((out_x_ax, out_x_pr, out_y_ax, out_y_pr))
            np.savetxt(path, data_matrix, delimiter=',', header=header, comments='', fmt='%g')
            QMessageBox.information(self, "Export Successful", f"Profiles safely exported to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export data:\n{str(e)}")

    def export_vector_figure(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export Vector Figure", "arpes_figure.pdf", "PDF Files (*.pdf);;SVG Files (*.svg)")
        if not path: return
        try:
            import matplotlib as mpl
            mpl.rcParams['pdf.fonttype'] = 42
            mpl.rcParams['ps.fonttype'] = 42
            mpl.rcParams['svg.fonttype'] = 'none'
            self.canvas.figure.savefig(path, transparent=True, bbox_inches='tight', dpi=300)
            QMessageBox.information(self, "Export Successful", f"Vector figure safely exported to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export figure:\n{str(e)}")

    def contextMenuEvent(self, event):
        """Builds a dynamic right-click menu to handle snapping, de-snapping, and reattaching."""
        menu = QMenu(self)
        
        dashboard = self.parentWidget()
        while dashboard and not hasattr(dashboard, 'get_neighbor'):
            dashboard = dashboard.parentWidget()
            
        if not dashboard:
            return

        is_floating = hasattr(self, 'is_detached') and self.is_detached
        
        if is_floating:
            action_reattach = menu.addAction("🔗 Reattach to Main Grid")
            action_reattach.triggered.connect(lambda: dashboard.reattach_view(self))
        else:
            # NEW: Add options to spawn new connected panels
            spawn_menu = menu.addMenu("➕ Snap New Panel...")
            spawn_menu.addAction("Top").triggered.connect(lambda: dashboard.spawn_view(self.x_idx, self.y_idx, self, "Top"))
            spawn_menu.addAction("Bottom").triggered.connect(lambda: dashboard.spawn_view(self.x_idx, self.y_idx, self, "Bottom"))
            spawn_menu.addAction("Left").triggered.connect(lambda: dashboard.spawn_view(self.x_idx, self.y_idx, self, "Left"))
            spawn_menu.addAction("Right").triggered.connect(lambda: dashboard.spawn_view(self.x_idx, self.y_idx, self, "Right"))
            
            menu.addSeparator()

            # Existing directional de-snap options
            pos = dashboard.get_widget_position(self)
            if pos:
                row, col = pos[0], pos[1]
                neighbors = {
                    "Up": dashboard.get_neighbor(row, col, "up"),
                    "Bottom": dashboard.get_neighbor(row, col, "down"),
                    "Left": dashboard.get_neighbor(row, col, "left"),
                    "Right": dashboard.get_neighbor(row, col, "right")
                }
                
                has_neighbors = False
                for direction, neighbor in neighbors.items():
                    if neighbor:
                        action = menu.addAction(f"✂️ De-snap {direction} Connection")
                        action.triggered.connect(lambda checked, n=neighbor: dashboard.detach_view(n))
                        has_neighbors = True
                        
                if not has_neighbors:
                    action_none = menu.addAction("No connected neighbors to de-snap.")
                    action_none.setEnabled(False)

        menu.exec(event.globalPos())

    def close_widget(self):
        self.parent_panel.remove_view(self)


class DataViewerPanel(QWidget):
    """The unified Dashboard that synchronizes crosshairs across all spawned views."""
    
    # --- NEW: Global registry for Cross-Dataset Sync ---
    _active_instances = weakref.WeakSet()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.tensor_data = None
        self.views = []
        self.detached_windows = [] 
        
        # Central State Manager
        self.global_coords = {}
        
        # Register this panel to the global sync network
        DataViewerPanel._active_instances.add(self)
        
        self._init_ui()

    def _init_ui(self):
        self.main_layout = QVBoxLayout(self)
        
        top_bar = QHBoxLayout()
        top_bar.addWidget(QLabel("<b>Dynamic Cross-Correlated Dashboard</b>"))
        top_bar.addStretch()
        self.main_layout.addLayout(top_bar)
        
        # Root splitter for vertical rows
        self.v_splitter = QSplitter(Qt.Vertical)
        self.main_layout.addWidget(self.v_splitter)
        self.main_layout.setStretch(1, 1)
        
    def load_data(self, tensor_data: TensorData):
        self.tensor_data = tensor_data
        self.global_coords = {i: len(ax)//2 for i, ax in enumerate(self.tensor_data.axes)}
        
        self._clear_tree()
        self.views.clear()
        
        default_x = 1 if self.tensor_data.ndim > 1 else 0
        default_y = 0
        self.spawn_view(default_x, default_y)

    def _clear_tree(self):
        """Safely detach all SliceWidgets and destroy intermediate splitters/spacers."""
        for w in self.views:
            w.setParent(None)
        while self.v_splitter.count():
            child = self.v_splitter.widget(0)
            child.setParent(None)
            if isinstance(child, QSplitter):
                child.deleteLater()

    def _rebuild_grid(self):
        """Rebuilds the grid geometry using synchronized QSplitters and spacers."""
        self._clear_tree()
        
        if not self.views:
            return
            
        max_row = max(w.grid_row for w in self.views)
        max_col = max(w.grid_col for w in self.views)
        
        for r in range(max_row + 1):
            h_splitter = QSplitter(Qt.Horizontal)
            
            # Check if this row actually has any widgets
            row_has_widgets = any(w.grid_row == r for w in self.views)
            if not row_has_widgets:
                continue
                
            for c in range(max_col + 1):
                w = next((v for v in self.views if v.grid_row == r and v.grid_col == c), None)
                if w:
                    h_splitter.addWidget(w)
                else:
                    # Blank space to maintain grid alignment
                    spacer = QWidget()
                    spacer.setStyleSheet("background-color: transparent;")
                    h_splitter.addWidget(spacer)
            
            # Distribute width evenly
            h_splitter.setSizes([1000] * (max_col + 1))
            self.v_splitter.addWidget(h_splitter)
            
        # Distribute height evenly
        self.v_splitter.setSizes([1000] * self.v_splitter.count())

    def spawn_view(self, x_idx: int, y_idx: int, ref_widget: SliceWidget = None, direction: str = None):
        """Creates a new SliceWidget and places it in the grid relative to ref_widget."""
        row, col = 0, 0
        
        if ref_widget and direction:
            row, col = ref_widget.grid_row, ref_widget.grid_col
            if direction == "Top": row -= 1
            elif direction == "Bottom": row += 1
            elif direction == "Left": col -= 1
            elif direction == "Right": col += 1
            
            # Prevent negative indexing by shifting the entire puzzle board
            if row < 0:
                for w in self.views: w.grid_row += 1
                row = 0
            if col < 0:
                for w in self.views: w.grid_col += 1
                col = 0

        new_widget = SliceWidget(self, x_idx, y_idx, row, col)
        self.views.append(new_widget)
        self._rebuild_grid()
        new_widget.redraw()

    def remove_view(self, widget: SliceWidget):
        if len(self.views) <= 1: return 
        if widget in self.views:
            self.views.remove(widget)
        self._rebuild_grid()
        widget.deleteLater()

    def detach_view(self, widget):
        """Rips a panel out of the grid and puts it in an independent floating window."""
        if len(self.views) <= 1:
            QMessageBox.warning(self, "Detach Error", "Cannot detach the last remaining panel.")
            return 
            
        if widget in self.views:
            self.views.remove(widget)
        
        widget.setParent(None)
        self._rebuild_grid()
        
        floating_window = QWidget()
        floating_window.setWindowTitle("Detached TensorSpec Panel")
        floating_window.resize(600, 500)
        floating_window.setWindowFlags(Qt.Window)
        layout = QVBoxLayout(floating_window)
        layout.addWidget(widget)
        widget.is_detached = True
        widget.floating_container = floating_window
        floating_window.show()
        self.detached_windows.append(floating_window)

    def reattach_view(self, widget):
        """Pulls a floating panel back into the grid."""
        if hasattr(widget, 'floating_container'):
            floating_window = widget.floating_container
            floating_window.layout().removeWidget(widget)
            widget.setParent(None)
            floating_window.close()
            if floating_window in self.detached_windows:
                self.detached_windows.remove(floating_window)
            widget.is_detached = False
            
            max_row = max((w.grid_row for w in self.views), default=-1)
            widget.grid_row = max_row + 1
            widget.grid_col = 0
            
            if widget not in self.views:
                self.views.append(widget)
            self._rebuild_grid()

    def _sync_enabled(self) -> bool:
        """True if any SliceWidget owned by this panel has 🔗 Sync checked."""
        candidates = list(self.views)
        for win in self.detached_windows:
            try:
                candidates.extend(win.findChildren(SliceWidget))
            except RuntimeError:
                continue
        for view in candidates:
            chk = getattr(view, 'chk_sync', None)
            if chk is not None and chk.isChecked():
                return True
        return False

    def update_global_coord(self, dim_idx: int, val_idx: int, broadcast=True, cross_sync=False):
        """Update this panel's cursor index.

        cross_sync defaults False: cross-dataset linking must be opted in via
        🔗 Sync on the source (and peer) viewers — never the silent default.
        """
        self.global_coords[dim_idx] = val_idx
        
        # Caller already opted in via cross_sync=True (from chk_sync on click/slider).
        # Still require peer Sync on so one toggled window cannot drag an untoggled peer.
        if cross_sync and self.tensor_data is not None:
            phys_val = self.tensor_data.axes[dim_idx][val_idx]
            target_label = self.tensor_data.labels[dim_idx]
            target_unit = self.tensor_data.units[dim_idx]
            
            for peer in list(DataViewerPanel._active_instances):
                try:
                    if peer is self or peer.tensor_data is None:
                        continue
                    if not peer._sync_enabled():
                        continue
                    for p_dim_idx, (p_label, p_unit) in enumerate(zip(peer.tensor_data.labels, peer.tensor_data.units)):
                        if p_label == target_label and p_unit == target_unit:
                            peer_axis = peer.tensor_data.axes[p_dim_idx]
                            nearest_idx = int(np.argmin(np.abs(peer_axis - phys_val)))
                            if peer.global_coords.get(p_dim_idx) != nearest_idx:
                                peer.update_global_coord(p_dim_idx, nearest_idx, broadcast=True, cross_sync=False)
                            break 
                except RuntimeError:
                    DataViewerPanel._active_instances.discard(peer)
        
        if broadcast:
            self.broadcast_redraw()

    def get_widget_position(self, widget):
        if widget not in self.views:
            return None
        return (widget.grid_row, widget.grid_col, 1, 1)

    def get_neighbor(self, row, col, direction):
        if direction == "up": r, c = row - 1, col
        elif direction == "down": r, c = row + 1, col
        elif direction == "left": r, c = row, col - 1
        elif direction == "right": r, c = row, col + 1
        else: return None
        
        for w in self.views:
            if w.grid_row == r and w.grid_col == c:
                return w
        return None

    def get_current_coords(self):
        return (self.global_coords.get(1, 0), self.global_coords.get(0, 0))

    def get_slider_values(self):
        return (0, 0, 0, 0)

    def get_dispersion_contrast(self):
        return 100

    def sync_ml_layers(self, source_dict: dict):
        """Merge ML domain / label arrays into viewer metadata and refresh layer combos."""
        if self.tensor_data is None:
            return
        meta = dict(self.tensor_data.metadata or {})
        layers = dict(meta.get("layers") or {})
        for key, val in source_dict.items():
            if key.startswith(("Labels_", "domains_", "probs_")) or key == "Supervised Probabilities":
                layers[key] = val
        meta["layers"] = layers
        self.tensor_data.metadata = meta
        for view in self.views:
            current = view.combo_layer.currentText()
            view.combo_layer.blockSignals(True)
            view.combo_layer.clear()
            view.combo_layer.addItem("Intensity")
            for layer_name in layers:
                view.combo_layer.addItem(layer_name)
            restore = view.combo_layer.findText(current)
            view.combo_layer.setCurrentIndex(restore if restore >= 0 else 0)
            view.combo_layer.blockSignals(False)

    def focus_spatial_layer(self, layer_name: str, y_idx: int = 2, x_idx: int = 3):
        """Show an ML label layer on the Y vs X spatial map."""
        view = next((v for v in self.views if v.x_idx == x_idx and v.y_idx == y_idx), None)
        if view is None:
            self.spawn_view(x_idx, y_idx)
            view = next(v for v in self.views if v.x_idx == x_idx and v.y_idx == y_idx)
        if view.combo_layer.findText(layer_name) < 0:
            view.combo_layer.addItem(layer_name)
        view.combo_layer.setCurrentText(layer_name)
        view.redraw()

    def add_overlay_mode(self, mode_name):
        """Legacy hook — prefer sync_ml_layers() so combo entries have backing arrays."""
        for view in self.views:
            if hasattr(view, "combo_layer") and view.combo_layer.findText(mode_name) < 0:
                view.combo_layer.addItem(mode_name)

    def broadcast_redraw(self):
        for view in self.views:
            view.redraw()