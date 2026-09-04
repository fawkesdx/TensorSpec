import numpy as np
import matplotlib.patches as patches
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QSlider, QComboBox, QCheckBox, QMenu, QFileDialog, QMessageBox, QSizePolicy)
from PySide6.QtCore import Qt
from PySide6.QtGui import QCursor
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=10, height=6, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(fig)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.updateGeometry()

class Maestro4DViewer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_data = None
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0) # Remove padding so it fits flush inside the main app

        # --- TOGGLES ---
        viewer_toggles_layout = QHBoxLayout()
        viewer_toggles_layout.addWidget(QLabel("<b>3. Interactive 4D Viewer</b>"))
        self.chk_spat_prof = QCheckBox("Show X/Y Profiles")
        self.chk_disp_prof = QCheckBox("Show EDC/MDC Profiles")
        self.chk_spat_prof.stateChanged.connect(self.toggle_profiles)
        self.chk_disp_prof.stateChanged.connect(self.toggle_profiles)
        viewer_toggles_layout.addSpacing(20)
        viewer_toggles_layout.addWidget(self.chk_spat_prof)
        viewer_toggles_layout.addWidget(self.chk_disp_prof)
        viewer_toggles_layout.addStretch()
        main_layout.addLayout(viewer_toggles_layout)

        # --- VIEW MODE DROPDOWNS ---
        view_mode_layout = QVBoxLayout()
        spat_ctrl = QHBoxLayout()
        spat_ctrl.addWidget(QLabel("Spatial Map (Left):"))
        self.combo_spatial_mode = QComboBox()
        self.combo_spatial_mode.addItem("Intensity")
        self.combo_spatial_mode.currentTextChanged.connect(self.update_viewer)
        spat_ctrl.addWidget(self.combo_spatial_mode)
        
        spat_ctrl.addWidget(QLabel("Contrast %:"))
        self.sl_c_spat = QSlider(Qt.Orientation.Horizontal)
        self.sl_c_spat.setRange(1, 100); self.sl_c_spat.setValue(100)
        self.sl_c_spat.valueChanged.connect(self.update_viewer)
        spat_ctrl.addWidget(self.sl_c_spat)
        
        spat_ctrl.addWidget(QLabel("Cmap:"))
        self.combo_cmap_spat = QComboBox()
        self.combo_cmap_spat.addItems(['viridis', 'magma', 'inferno', 'plasma', 'cividis', 'gray', 'hsv', 'twilight_shifted', 'PiYG'])
        self.combo_cmap_spat.currentTextChanged.connect(self.update_viewer)
        spat_ctrl.addWidget(self.combo_cmap_spat)
        
        disp_ctrl = QHBoxLayout()
        disp_ctrl.addWidget(QLabel("Dispersion Map (Right) - Contrast %:"))
        self.sl_c_disp = QSlider(Qt.Orientation.Horizontal)
        self.sl_c_disp.setRange(1, 100); self.sl_c_disp.setValue(100)
        self.sl_c_disp.valueChanged.connect(self.update_viewer)
        disp_ctrl.addWidget(self.sl_c_disp)
        
        disp_ctrl.addWidget(QLabel("Cmap:"))
        self.combo_cmap_disp = QComboBox()
        self.combo_cmap_disp.addItems(['magma', 'viridis', 'inferno', 'plasma', 'cividis', 'gray', 'hsv', 'twilight_shifted', 'PiYG'])
        self.combo_cmap_disp.currentTextChanged.connect(self.update_viewer)
        disp_ctrl.addWidget(self.combo_cmap_disp)
        
        view_mode_layout.addLayout(spat_ctrl)
        view_mode_layout.addLayout(disp_ctrl)
        main_layout.addLayout(view_mode_layout)

        # --- CANVAS SETUP ---
        self.canvas = MplCanvas(self, width=10, height=6)
        
        # THE FIX: Hardcode the margins and spacing so the boxes never shift
        gs = self.canvas.figure.add_gridspec(
            2, 4, 
            width_ratios=[4, 1, 4, 1], 
            height_ratios=[4, 1],
            left=0.08, right=0.95, top=0.90, bottom=0.12, 
            wspace=0.35, hspace=0.35
        )
        
        self.ax_spat = self.canvas.figure.add_subplot(gs[0, 0])
        self.ax_y_prof = self.canvas.figure.add_subplot(gs[0, 1], sharey=self.ax_spat)
        self.ax_x_prof = self.canvas.figure.add_subplot(gs[1, 0], sharex=self.ax_spat)
        
        self.ax_disp = self.canvas.figure.add_subplot(gs[0, 2])
        self.ax_edc = self.canvas.figure.add_subplot(gs[0, 3], sharey=self.ax_disp)
        self.ax_mdc = self.canvas.figure.add_subplot(gs[1, 2], sharex=self.ax_disp)
        
        self.im_spat = self.ax_spat.imshow(np.zeros((10, 10)), origin='lower', aspect='auto', cmap='viridis')
        self.im_disp = self.ax_disp.imshow(np.zeros((10, 10)), origin='lower', aspect='auto', cmap='magma')

        self.line_x_prof, = self.ax_x_prof.plot([], [], 'b-')
        self.line_y_prof, = self.ax_y_prof.plot([], [], 'b-')
        self.line_edc, = self.ax_edc.plot([], [], 'r-')
        self.line_mdc, = self.ax_mdc.plot([], [], 'r-')

        self.ax_y_prof.tick_params(labelleft=False)
        self.ax_x_prof.tick_params(labelbottom=True)
        self.ax_spat.tick_params(labelbottom=True)
        self.ax_edc.tick_params(labelleft=False)
        self.ax_mdc.tick_params(labelbottom=True)
        self.ax_disp.tick_params(labelbottom=True)
        
        self.ax_spat.set_ylabel("Y Position")
        self.ax_x_prof.set_xlabel("X Position"); self.ax_x_prof.set_ylabel("Int.")
        self.ax_y_prof.set_xlabel("Int.")
        
        self.ax_disp.set_ylabel("Energy (eV)")
        self.ax_mdc.set_xlabel("Angle (Degrees)"); self.ax_mdc.set_ylabel("Int.")
        self.ax_edc.set_xlabel("Int.")
        
        self.ax_y_prof.set_visible(False)
        self.ax_x_prof.set_visible(False)
        self.ax_edc.set_visible(False)
        self.ax_mdc.set_visible(False)
        
        # Crosshairs
        self.vline_spat = self.ax_spat.axvline(0, color='red', linestyle='--', alpha=0.7)
        self.hline_spat = self.ax_spat.axhline(0, color='red', linestyle='--', alpha=0.7)
        self.rect_spat = patches.Rectangle((0,0), 1, 1, linewidth=2, edgecolor='red', facecolor='none')
        self.ax_spat.add_patch(self.rect_spat)
        
        self.vline_disp = self.ax_disp.axvline(0, color='cyan', linestyle='--', alpha=0.7)
        self.hline_disp = self.ax_disp.axhline(0, color='cyan', linestyle='--', alpha=0.7)
        self.rect_disp = patches.Rectangle((0,0), 1, 1, linewidth=2, edgecolor='cyan', facecolor='none')
        self.ax_disp.add_patch(self.rect_disp)
        
        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.canvas.mpl_connect('motion_notify_event', self.on_click)
        main_layout.addWidget(self.canvas)
        
        # --- SLIDERS ---
        sliders_layout = QHBoxLayout()
        self.sl_x = QSlider(Qt.Orientation.Horizontal); self.sl_x.valueChanged.connect(self.update_viewer)
        self.sl_dx = QSlider(Qt.Orientation.Horizontal); self.sl_dx.valueChanged.connect(self.update_viewer)
        self.sl_y = QSlider(Qt.Orientation.Horizontal); self.sl_y.valueChanged.connect(self.update_viewer)
        self.sl_dy = QSlider(Qt.Orientation.Horizontal); self.sl_dy.valueChanged.connect(self.update_viewer)
        
        self.sl_e = QSlider(Qt.Orientation.Horizontal); self.sl_e.valueChanged.connect(self.update_viewer)
        self.sl_de = QSlider(Qt.Orientation.Horizontal); self.sl_de.valueChanged.connect(self.update_viewer)
        self.sl_a = QSlider(Qt.Orientation.Horizontal); self.sl_a.valueChanged.connect(self.update_viewer)
        self.sl_da = QSlider(Qt.Orientation.Horizontal); self.sl_da.valueChanged.connect(self.update_viewer)
        
        s1 = QVBoxLayout()
        s1.addWidget(QLabel("X Center")); s1.addWidget(self.sl_x)
        s1.addWidget(QLabel("X Width ±")); s1.addWidget(self.sl_dx)
        s1.addWidget(QLabel("Y Center")); s1.addWidget(self.sl_y)
        s1.addWidget(QLabel("Y Width ±")); s1.addWidget(self.sl_dy)
        
        s2 = QVBoxLayout()
        s2.addWidget(QLabel("Energy Center")); s2.addWidget(self.sl_e)
        s2.addWidget(QLabel("Energy Width ±")); s2.addWidget(self.sl_de)
        s2.addWidget(QLabel("Angle Center")); s2.addWidget(self.sl_a)
        s2.addWidget(QLabel("Angle Width ±")); s2.addWidget(self.sl_da)
        
        sliders_layout.addLayout(s1)
        sliders_layout.addLayout(s2)
        main_layout.addLayout(sliders_layout)

    # --- API METHODS (The "Bridge" for the Main App) ---
    def set_data(self, data):
        self.current_data = data
        val = data['value']
        dim_E, dim_A, nY, nX = val.shape
        
        sliders = [self.sl_x, self.sl_dx, self.sl_y, self.sl_dy, self.sl_e, self.sl_de, self.sl_a, self.sl_da]
        for sl in sliders: sl.blockSignals(True)
        
        self.sl_x.setMaximum(nX-1); self.sl_x.setValue(int(nX/2))
        self.sl_dx.setMaximum(int(nX/2)); self.sl_dx.setValue(0)
        self.sl_y.setMaximum(nY-1); self.sl_y.setValue(int(nY/2))
        self.sl_dy.setMaximum(int(nY/2)); self.sl_dy.setValue(0)
        self.sl_e.setMaximum(dim_E-1); self.sl_e.setValue(int(dim_E/2))
        self.sl_de.setMaximum(int(dim_E/2)); self.sl_de.setValue(0)
        self.sl_a.setMaximum(dim_A-1); self.sl_a.setValue(int(dim_A/2))
        self.sl_da.setMaximum(int(dim_A/2)); self.sl_da.setValue(0)
        
        for sl in sliders: sl.blockSignals(False)
        
        self.combo_spatial_mode.blockSignals(True)
        self.combo_spatial_mode.clear()
        self.combo_spatial_mode.addItem("Intensity")
        for k in data.keys():
            if k.startswith("domains_") or k == "Supervised Probabilities":
                self.combo_spatial_mode.addItem(k)
        self.combo_spatial_mode.blockSignals(False)
        
        X_arr, Y_arr, E_arr, A_arr = data['x'], data['y'], data['E'], data['angle']
        px_x = (X_arr[-1] - X_arr[0]) / max(1, nX-1) if (X_arr[-1] - X_arr[0]) / max(1, nX-1) != 0 else 0.1
        px_y = (Y_arr[-1] - Y_arr[0]) / max(1, nY-1) if (Y_arr[-1] - Y_arr[0]) / max(1, nY-1) != 0 else 0.1
        px_A = (A_arr[-1] - A_arr[0]) / max(1, dim_A-1) if (A_arr[-1] - A_arr[0]) / max(1, dim_A-1) != 0 else 0.1
        px_E = (E_arr[-1] - E_arr[0]) / max(1, dim_E-1) if (E_arr[-1] - E_arr[0]) / max(1, dim_E-1) != 0 else 0.1
        
        self.im_spat.set_extent((X_arr[0]-px_x/2, X_arr[-1]+px_x/2, Y_arr[0]-px_y/2, Y_arr[-1]+px_y/2))
        self.im_disp.set_extent((A_arr[0]-px_A/2, A_arr[-1]+px_A/2, E_arr[0]-px_E/2, E_arr[-1]+px_E/2))
        
        self.ax_spat.set_xlim(X_arr[0]-px_x/2, X_arr[-1]+px_x/2); self.ax_spat.set_ylim(Y_arr[0]-px_y/2, Y_arr[-1]+px_y/2)
        self.ax_disp.set_xlim(A_arr[0]-px_A/2, A_arr[-1]+px_A/2); self.ax_disp.set_ylim(E_arr[0]-px_E/2, E_arr[-1]+px_E/2)
        
        self.update_viewer()

    def add_overlay_mode(self, mode_name):
        if self.combo_spatial_mode.findText(mode_name) == -1:
            self.combo_spatial_mode.addItem(mode_name)

    def get_current_coords(self):
        """Used by the Supervised Tab to grab training points."""
        return self.sl_x.value(), self.sl_y.value()

    def get_slider_values(self):
        """Used by the Clustering Tab to compute dynamic EDC/MDC."""
        return self.sl_e.value(), self.sl_de.value(), self.sl_a.value(), self.sl_da.value()

    def get_dispersion_contrast(self):
        """Used by the Dendrogram to match contrast."""
        return self.sl_c_disp.value() / 100.0

    def get_current_spatial_mode(self):
        """Used by UMAP to know what domain colors to plot."""
        return self.combo_spatial_mode.currentText()

    # --- INTERNAL LOGIC ---
    def toggle_profiles(self):
        spat_on = self.chk_spat_prof.isChecked()
        disp_on = self.chk_disp_prof.isChecked()
        self.ax_y_prof.set_visible(spat_on); self.ax_x_prof.set_visible(spat_on)
        self.ax_spat.tick_params(labelbottom=not spat_on)
        self.ax_edc.set_visible(disp_on); self.ax_mdc.set_visible(disp_on)
        self.ax_disp.tick_params(labelbottom=not disp_on)
        self.update_viewer()

    def update_viewer(self):
        if not self.current_data: return
        data = self.current_data
        val = data['value']
        dim_E, dim_A, nY, nX = val.shape
        X_arr, Y_arr, E_arr, A_arr = data['x'], data['y'], data['E'], data['angle']
        
        x_c, dx, y_c, dy = self.sl_x.value(), self.sl_dx.value(), self.sl_y.value(), self.sl_dy.value()
        e_c, de, a_c, da = self.sl_e.value(), self.sl_de.value(), self.sl_a.value(), self.sl_da.value()
        
        x1, x2 = max(0, x_c - dx), min(nX, x_c + dx + 1)
        y1, y2 = max(0, y_c - dy), min(nY, y_c + dy + 1)
        e1, e2 = max(0, e_c - de), min(dim_E, e_c + de + 1)
        a1, a2 = max(0, a_c - da), min(dim_A, a_c + da + 1)
        
        mode = self.combo_spatial_mode.currentText()
        c_spat = self.sl_c_spat.value() / 100.0
        c_disp = self.sl_c_disp.value() / 100.0
        cmap_spat = self.combo_cmap_spat.currentText()
        cmap_disp = self.combo_cmap_disp.currentText()
        
        if mode == "Intensity" or mode not in data:
            spat = np.sum(val[e1:e2, a1:a2, :, :], axis=(0,1))
            self.im_spat.set_data(spat)
            s_min, s_max = spat.min(), spat.max()
            self.im_spat.set_clim(s_min, s_min + (s_max - s_min) * c_spat + 1e-8)
            self.im_spat.set_cmap(cmap_spat)
            self.ax_spat.set_title(f"Spatial Map\nE:[{E_arr[e1]:.2f}:{E_arr[e2-1]:.2f}] eV | Ang:[{A_arr[a1]:.1f}:{A_arr[a2-1]:.1f}]°")
        
        elif mode == "Supervised Probabilities":
            prob_map = data[mode] 
            nY, nX, nC = prob_map.shape
            import matplotlib.cm as cm
            base_colors = cm.get_cmap('tab10').colors 
            color_names = ["Blue", "Orange", "Green", "Red", "Purple", "Brown", "Pink", "Gray", "Olive", "Cyan"]
            
            rgb_map = np.zeros((nY, nX, 3))
            for c in range(nC):
                color = np.array(base_colors[c % 10])
                rgb_map += prob_map[:, :, c:c+1] * color.reshape(1, 1, 3)
            rgb_map = np.clip(rgb_map, 0, 1)
            self.im_spat.set_data(rgb_map)
            pixel_probs = prob_map[y_c, x_c, :] 
            legend_str = " | ".join([f"L{c+1} ({color_names[c%10]})" for c in range(nC)])
            prob_str = "  ".join([f"L{c+1}: {pixel_probs[c]*100:.1f}%" for c in range(nC)])
            self.ax_spat.set_title(f"Supervised Labels: {legend_str}\nCrosshair Confidence: {prob_str}", fontsize=10)
            
        elif mode.startswith("domains_Align"):
            shift_1d = data[mode]
            shift_2d = shift_1d.reshape((nY, nX))
            self.im_spat.set_data(shift_2d)
            if 'Defl' in mode or 'Slit' in mode:
                max_abs = np.max(np.abs(shift_2d)); max_abs = 1.0 if max_abs == 0 else max_abs
                self.im_spat.set_clim(-max_abs, max_abs * c_spat) 
                title_name = "Surface Normal Tilt (Pixels)"
            elif 'Azimuth' in mode:
                self.im_spat.set_clim(0, 360 * c_spat) 
                title_name = "Azimuthal Rotation (\u03D5)\n(Measured in Degrees)"
            else:
                d_min, d_max = shift_2d.min(), shift_2d.max()
                self.im_spat.set_clim(d_min, d_min + (d_max - d_min) * c_spat + 1e-8)
                title_name = "Match Quality Score"
            self.im_spat.set_cmap(cmap_spat) 
            self.ax_spat.set_title(title_name)
            
        else:
            import matplotlib.cm as cm
            labels_1d = data[mode]
            labels_2d = labels_1d.reshape((nY, nX))
            self.im_spat.set_data(labels_2d)
            self.im_spat.set_clim(-0.5, 19.5) 
            try:
                import matplotlib as mpl
                cmap = mpl.colormaps['tab20'].with_extremes(under='#333333')
            except:
                cmap = cm.get_cmap('tab20').copy(); cmap.set_under('#333333') 
            self.im_spat.set_cmap(cmap)
            clean_name = mode.replace('domains_', '')
            self.ax_spat.set_title(f"Clustering Domains: {clean_name}\n(Intensity Integration Disabled)")
            
        disp = np.sum(val[:, :, y1:y2, x1:x2], axis=(2,3)) 
        self.im_disp.set_data(disp)
        self.im_disp.set_cmap(cmap_disp) 
        d_min, d_max = disp.min(), disp.max()
        self.im_disp.set_clim(d_min, d_min + (d_max - d_min) * c_disp + 1e-8)
        
        px_x = (X_arr[-1] - X_arr[0]) / max(1, nX-1) if (X_arr[-1] - X_arr[0]) / max(1, nX-1) != 0 else 0.1
        px_y = (Y_arr[-1] - Y_arr[0]) / max(1, nY-1) if (Y_arr[-1] - Y_arr[0]) / max(1, nY-1) != 0 else 0.1
        px_A = (A_arr[-1] - A_arr[0]) / max(1, dim_A-1) if (A_arr[-1] - A_arr[0]) / max(1, dim_A-1) != 0 else 0.1
        px_E = (E_arr[-1] - E_arr[0]) / max(1, dim_E-1) if (E_arr[-1] - E_arr[0]) / max(1, dim_E-1) != 0 else 0.1
        
        self.vline_spat.set_xdata([X_arr[x_c]]); self.hline_spat.set_ydata([Y_arr[y_c]])
        self.rect_spat.set_bounds(X_arr[x1]-px_x/2, Y_arr[y1]-px_y/2, (x2-x1)*px_x, (y2-y1)*px_y)
        
        self.vline_disp.set_xdata([A_arr[a_c]]); self.hline_disp.set_ydata([E_arr[e_c]])
        self.rect_disp.set_bounds(A_arr[a1]-px_A/2, E_arr[e1]-px_E/2, (a2-a1)*px_A, (e2-e1)*px_E)
        
        self.ax_disp.set_title(f"ARPES Dispersion\nX:[{X_arr[x1]:.3f}:{X_arr[x2-1]:.3f}] | Y:[{Y_arr[y1]:.3f}:{Y_arr[y2-1]:.3f}]")
        
        if self.chk_spat_prof.isChecked():
            raw_spat = np.sum(val[e1:e2, a1:a2, :, :], axis=(0,1)) 
            x_prof = np.sum(raw_spat[y1:y2, :], axis=0)
            y_prof = np.sum(raw_spat[:, x1:x2], axis=1)
            self.line_x_prof.set_data(X_arr, x_prof); self.ax_x_prof.relim(); self.ax_x_prof.autoscale_view()
            self.line_y_prof.set_data(y_prof, Y_arr); self.ax_y_prof.relim(); self.ax_y_prof.autoscale_view()

        if self.chk_disp_prof.isChecked():
            mdc = np.sum(disp[e1:e2, :], axis=0)
            edc = np.sum(disp[:, a1:a2], axis=1)
            self.line_mdc.set_data(A_arr, mdc); self.ax_mdc.relim(); self.ax_mdc.autoscale_view()
            self.line_edc.set_data(edc, E_arr); self.ax_edc.relim(); self.ax_edc.autoscale_view()

        self.canvas.draw_idle()

    def on_click(self, event):
        if not self.current_data: return
        data = self.current_data
        
        if event.button == 1: 
            if event.inaxes == self.ax_spat:
                x_idx = np.argmin(np.abs(data['x'] - event.xdata))
                y_idx = np.argmin(np.abs(data['y'] - event.ydata))
                self.sl_x.setValue(int(x_idx)); self.sl_y.setValue(int(y_idx))
            elif event.inaxes == self.ax_disp:
                a_idx = np.argmin(np.abs(data['angle'] - event.xdata))
                e_idx = np.argmin(np.abs(data['E'] - event.ydata))
                self.sl_a.setValue(int(a_idx)); self.sl_e.setValue(int(e_idx))
                
        elif event.button == 3: 
            menu = QMenu(self)
            if event.inaxes == self.ax_spat:
                act_spat = menu.addAction("Save Spatial Map (No Crosshairs)")
                action = menu.exec(QCursor.pos())
                if action == act_spat: self.export_plot("spatial_no_cross")
            elif event.inaxes == self.ax_disp:
                act_full_cross = menu.addAction("Save Full Viewer (WITH Crosshairs)")
                act_full_no_cross = menu.addAction("Save Full Viewer (NO Crosshairs)")
                action = menu.exec(QCursor.pos())
                if action == act_full_cross: self.export_plot("full_with_cross")
                elif action == act_full_no_cross: self.export_plot("full_no_cross")

    def set_crosshairs_visible(self, ax, visible):
        if ax == self.ax_spat:
            self.vline_spat.set_visible(visible); self.hline_spat.set_visible(visible); self.rect_spat.set_visible(visible)
        elif ax == self.ax_disp:
            self.vline_disp.set_visible(visible); self.hline_disp.set_visible(visible); self.rect_disp.set_visible(visible)

    def export_plot(self, plot_type):
        path, _ = QFileDialog.getSaveFileName(self, "Export Plot", "MAESTRO_Plot.eps", "EPS Vector (*.eps);;PDF Files (*.pdf);;TIFF Images (*.tiff *.tif);;PNG Images (*.png)")
        if not path: return
        try:
            if plot_type == "spatial_no_cross":
                self.set_crosshairs_visible(self.ax_spat, False); self.ax_disp.set_visible(False)
                mode = self.combo_spatial_mode.currentText()
                legend = None
                if mode != "Intensity":
                    import matplotlib.patches as mpatches
                    legend_patches = []
                    if mode == "Supervised Probabilities":
                        color_names = ["Blue", "Orange", "Green", "Red", "Purple", "Brown", "Pink", "Gray", "Olive", "Cyan"]
                        import matplotlib.cm as cm
                        base_colors = cm.get_cmap('tab10').colors
                        nC = self.current_data[mode].shape[2]
                        for c in range(nC):
                            legend_patches.append(mpatches.Patch(color=base_colors[c%10], label=f"Label {c+1} ({color_names[c%10]})"))
                    else:
                        labels = self.current_data[mode]
                        for lbl in np.unique(labels):
                            if lbl == -1: continue 
                            color = self.im_spat.cmap(self.im_spat.norm(lbl))
                            legend_patches.append(mpatches.Patch(color=color, label=f"Cluster {lbl}"))
                    if legend_patches: legend = self.ax_spat.legend(handles=legend_patches, loc='center left', bbox_to_anchor=(1.05, 0.5))
                self.canvas.figure.savefig(path, bbox_inches='tight', dpi=300)
                if legend: legend.remove()
                self.ax_disp.set_visible(True); self.set_crosshairs_visible(self.ax_spat, True)

            elif plot_type == "full_with_cross":
                self.canvas.figure.savefig(path, bbox_inches='tight', dpi=300)
            elif plot_type == "full_no_cross":
                self.set_crosshairs_visible(self.ax_spat, False); self.set_crosshairs_visible(self.ax_disp, False)
                self.canvas.figure.savefig(path, bbox_inches='tight', dpi=300)
                self.set_crosshairs_visible(self.ax_spat, True); self.set_crosshairs_visible(self.ax_disp, True)

        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to save plot:\n{str(e)}")
            self.ax_disp.set_visible(True); self.set_crosshairs_visible(self.ax_spat, True); self.set_crosshairs_visible(self.ax_disp, True)
        self.canvas.draw_idle()