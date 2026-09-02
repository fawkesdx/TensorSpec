import numpy as np
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, 
                             QLabel, QSlider, QComboBox)
from PySide6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.patches as patches # <-- NEW: Imported patches for the rectangles!

class FermiViewerWindow(QDialog):
    def __init__(self, data, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"3D Fermi Map Inspector: {data.get('kind', 'Unknown')}")
        self.resize(1100, 850)
        
        self.data = data
        raw_E = data['E'].copy()
        if np.mean(raw_E) > 0:
            raw_E = -raw_E
            
        sort_idx = np.argsort(raw_E)
        self.E_arr = raw_E[sort_idx]
        self.val = data['value'].squeeze(axis=2)[sort_idx, :, :]
        
        self.A_arr = data['angle']
        self.D_arr = data['x']     
            
        self.dim_E, self.dim_A, self.dim_D = self.val.shape
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        self.canvas = FigureCanvas(Figure(figsize=(10, 8), layout='tight'))
        layout.addWidget(self.canvas)
        fig = self.canvas.figure
        
        gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1], height_ratios=[1.5, 1])
        self.ax_iso = fig.add_subplot(gs[0, 0])
        self.ax_disp_defl = fig.add_subplot(gs[0, 1]) 
        self.ax_disp_slit = fig.add_subplot(gs[1, 0]) 
        
        self.im_iso = self.ax_iso.imshow(np.zeros((10,10)), origin='lower', aspect='auto', cmap='viridis')
        self.im_disp_slit = self.ax_disp_slit.imshow(np.zeros((10,10)), origin='lower', aspect='auto', cmap='magma')
        self.im_disp_defl = self.ax_disp_defl.imshow(np.zeros((10,10)), origin='lower', aspect='auto', cmap='magma')
        
        # Crosshair lines
        self.iso_v = self.ax_iso.axvline(0, color='red', ls='--')
        self.iso_h = self.ax_iso.axhline(0, color='cyan', ls='--')
        self.ds_v = self.ax_disp_slit.axvline(0, color='red', ls='--')
        self.ds_h = self.ax_disp_slit.axhline(0, color='yellow', ls='--')
        self.dd_v = self.ax_disp_defl.axvline(0, color='cyan', ls='--')
        self.dd_h = self.ax_disp_defl.axhline(0, color='yellow', ls='--')
        
        # --- NEW: Width Rectangles ---
        self.rect_iso = patches.Rectangle((0,0), 1, 1, linewidth=1.5, edgecolor='white', facecolor='none', alpha=0.8, linestyle=':')
        self.ax_iso.add_patch(self.rect_iso)
        
        self.rect_ds = patches.Rectangle((0,0), 1, 1, linewidth=1.5, edgecolor='white', facecolor='none', alpha=0.8, linestyle=':')
        self.ax_disp_slit.add_patch(self.rect_ds)
        
        self.rect_dd = patches.Rectangle((0,0), 1, 1, linewidth=1.5, edgecolor='white', facecolor='none', alpha=0.8, linestyle=':')
        self.ax_disp_defl.add_patch(self.rect_dd)
        # -----------------------------
        
        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.canvas.mpl_connect('motion_notify_event', self.on_click)
        
        sliders_layout = QHBoxLayout()
        self.sl_e = QSlider(Qt.Orientation.Horizontal); self.sl_e.setMaximum(self.dim_E-1); self.sl_e.setValue(self.dim_E//2)
        self.sl_de = QSlider(Qt.Orientation.Horizontal); self.sl_de.setMaximum(self.dim_E//2); self.sl_de.setValue(0)
        self.sl_s = QSlider(Qt.Orientation.Horizontal); self.sl_s.setMaximum(self.dim_A-1); self.sl_s.setValue(self.dim_A//2)
        self.sl_ds = QSlider(Qt.Orientation.Horizontal); self.sl_ds.setMaximum(self.dim_A//2); self.sl_ds.setValue(0)
        self.sl_d = QSlider(Qt.Orientation.Horizontal); self.sl_d.setMaximum(self.dim_D-1); self.sl_d.setValue(self.dim_D//2)
        self.sl_dd = QSlider(Qt.Orientation.Horizontal); self.sl_dd.setMaximum(self.dim_D//2); self.sl_dd.setValue(0)
        
        for sl in [self.sl_e, self.sl_de, self.sl_s, self.sl_ds, self.sl_d, self.sl_dd]: 
            sl.valueChanged.connect(self.update_plots)
            
        s1 = QVBoxLayout(); s1.addWidget(QLabel("Energy Center")); s1.addWidget(self.sl_e); s1.addWidget(QLabel("Energy Width ±")); s1.addWidget(self.sl_de)
        s2 = QVBoxLayout(); s2.addWidget(QLabel("Slit Angle Center")); s2.addWidget(self.sl_s); s2.addWidget(QLabel("Slit Width ±")); s2.addWidget(self.sl_ds)
        s3 = QVBoxLayout(); s3.addWidget(QLabel("Deflection Center")); s3.addWidget(self.sl_d); s3.addWidget(QLabel("Deflection Width ±")); s3.addWidget(self.sl_dd)
        sliders_layout.addLayout(s1); sliders_layout.addLayout(s2); sliders_layout.addLayout(s3)
        layout.addLayout(sliders_layout)
        
        ctrl_layout = QVBoxLayout()
        iso_ctrl = QHBoxLayout()
        iso_ctrl.addWidget(QLabel("Isoenergy Map - Contrast %:"))
        self.sl_c_iso = QSlider(Qt.Orientation.Horizontal)
        self.sl_c_iso.setRange(1, 500); self.sl_c_iso.setValue(100)
        self.sl_c_iso.valueChanged.connect(self.update_plots)
        iso_ctrl.addWidget(self.sl_c_iso)
        iso_ctrl.addWidget(QLabel("Cmap:"))
        self.combo_cmap_iso = QComboBox()
        self.combo_cmap_iso.addItems(['viridis', 'magma', 'inferno', 'plasma', 'cividis', 'gray', 'hsv', 'twilight_shifted', 'PiYG'])
        self.combo_cmap_iso.currentTextChanged.connect(self.update_plots)
        iso_ctrl.addWidget(self.combo_cmap_iso)
        
        disp_ctrl = QHBoxLayout()
        disp_ctrl.addWidget(QLabel("Dispersions - Contrast %:"))
        self.sl_c_disp = QSlider(Qt.Orientation.Horizontal)
        self.sl_c_disp.setRange(1, 300); self.sl_c_disp.setValue(100)
        self.sl_c_disp.valueChanged.connect(self.update_plots)
        disp_ctrl.addWidget(self.sl_c_disp)
        disp_ctrl.addWidget(QLabel("Cmap:"))
        self.combo_cmap_disp = QComboBox()
        self.combo_cmap_disp.addItems(['magma', 'viridis', 'inferno', 'plasma', 'cividis', 'gray', 'hsv', 'twilight_shifted', 'PiYG'])
        self.combo_cmap_disp.currentTextChanged.connect(self.update_plots)
        disp_ctrl.addWidget(self.combo_cmap_disp)
        
        ctrl_layout.addLayout(iso_ctrl); ctrl_layout.addLayout(disp_ctrl)
        layout.addLayout(ctrl_layout)
        
        self.update_plots()
        
    def update_plots(self):
        e_c, de = self.sl_e.value(), self.sl_de.value()
        s_c, ds = self.sl_s.value(), self.sl_ds.value()
        d_c, dd = self.sl_d.value(), self.sl_dd.value()
        
        e1, e2 = max(0, e_c-de), min(self.dim_E, e_c+de+1)
        s1, s2 = max(0, s_c-ds), min(self.dim_A, s_c+ds+1)
        d1, d2 = max(0, d_c-dd), min(self.dim_D, d_c+dd+1)
        
        c_iso = self.sl_c_iso.value() / 100.0
        c_disp = self.sl_c_disp.value() / 100.0
        
        iso = np.sum(self.val[e1:e2, :, :], axis=0).T
        self.im_iso.set_data(iso)
        self.im_iso.set_cmap(self.combo_cmap_iso.currentText())
        i_min, i_max = iso.min(), iso.max()
        self.im_iso.set_clim(i_min, i_min + (i_max-i_min)*c_iso + 1e-8)
        self.im_iso.set_extent((self.A_arr[0], self.A_arr[-1], self.D_arr[0], self.D_arr[-1]))
        
        disp_s = np.sum(self.val[:, :, d1:d2], axis=2)
        self.im_disp_slit.set_data(disp_s)
        self.im_disp_slit.set_cmap(self.combo_cmap_disp.currentText())
        ds_min, ds_max = disp_s.min(), disp_s.max()
        self.im_disp_slit.set_clim(ds_min, ds_min + (ds_max-ds_min)*c_disp + 1e-8)
        
        disp_d = np.sum(self.val[:, s1:s2, :], axis=1)
        self.im_disp_defl.set_data(disp_d)
        self.im_disp_defl.set_cmap(self.combo_cmap_disp.currentText())
        dd_min, dd_max = disp_d.min(), disp_d.max()
        self.im_disp_defl.set_clim(dd_min, dd_min + (dd_max-dd_min)*c_disp + 1e-8)
        
        e_start, e_end = self.E_arr[0], self.E_arr[-1]
        self.im_disp_slit.set_extent((self.A_arr[0], self.A_arr[-1], e_start, e_end))
        self.im_disp_defl.set_extent((self.D_arr[0], self.D_arr[-1], e_start, e_end))
        
        # Update lines
        self.iso_v.set_xdata([self.A_arr[s_c]]); self.iso_h.set_ydata([self.D_arr[d_c]])
        self.ds_v.set_xdata([self.A_arr[s_c]]); self.ds_h.set_ydata([self.E_arr[e_c]])
        self.dd_v.set_xdata([self.D_arr[d_c]]); self.dd_h.set_ydata([self.E_arr[e_c]])
        
        # --- NEW: Update Rectangles ---
        px_A = (self.A_arr[-1] - self.A_arr[0]) / max(1, self.dim_A-1) if self.dim_A > 1 else 0.1
        px_D = (self.D_arr[-1] - self.D_arr[0]) / max(1, self.dim_D-1) if self.dim_D > 1 else 0.1
        px_E = (self.E_arr[-1] - self.E_arr[0]) / max(1, self.dim_E-1) if self.dim_E > 1 else 0.1
        
        # Iso Map (Slit vs Defl)
        self.rect_iso.set_bounds(self.A_arr[s1]-px_A/2, self.D_arr[d1]-px_D/2, (s2-s1)*px_A, (d2-d1)*px_D)
        # Slit Map (Slit vs Energy)
        self.rect_ds.set_bounds(self.A_arr[s1]-px_A/2, self.E_arr[e1]-px_E/2, (s2-s1)*px_A, (e2-e1)*px_E)
        # Defl Map (Defl vs Energy)
        self.rect_dd.set_bounds(self.D_arr[d1]-px_D/2, self.E_arr[e1]-px_E/2, (d2-d1)*px_D, (e2-e1)*px_E)
        # ------------------------------
        
        self.ax_iso.set_title(f"Isoenergy Map\nE: [{self.E_arr[e1]:.2f} : {self.E_arr[e2-1]:.2f}] eV")
        self.ax_disp_slit.set_title(f"Slit Dispersion\nDefl: [{self.D_arr[d1]:.2f} : {self.D_arr[d2-1]:.2f}]°")
        self.ax_disp_defl.set_title(f"Deflection Dispersion\nSlit: [{self.A_arr[s1]:.2f} : {self.A_arr[s2-1]:.2f}]°")
        self.canvas.draw_idle()
        
    def on_click(self, event):
        if event.button != 1 or not event.inaxes: return
        
        if event.inaxes == self.ax_iso:
            self.sl_s.setValue(int(np.argmin(np.abs(self.A_arr - event.xdata))))
            self.sl_d.setValue(int(np.argmin(np.abs(self.D_arr - event.ydata))))
        elif event.inaxes == self.ax_disp_slit:
            self.sl_s.setValue(int(np.argmin(np.abs(self.A_arr - event.xdata))))
            self.sl_e.setValue(int(np.argmin(np.abs(self.E_arr - event.ydata))))
        elif event.inaxes == self.ax_disp_defl:
            self.sl_d.setValue(int(np.argmin(np.abs(self.D_arr - event.xdata))))
            self.sl_e.setValue(int(np.argmin(np.abs(self.E_arr - event.ydata))))