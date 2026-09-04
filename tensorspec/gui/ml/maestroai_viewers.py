import numpy as np
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, 
                             QLabel, QSlider, QComboBox, QSizePolicy)
from PySide6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.colors as mcolors
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.ndimage import map_coordinates


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100, is_3d=False, orientation='horizontal'):
        fig = Figure(figsize=(width, height), dpi=dpi, layout='tight')
        
        if not is_3d:
            if orientation == 'vertical':
                self.axes = fig.subplots(2, 1)
            elif orientation == 'vertical_3':
                self.axes = fig.subplots(3, 1)
            elif orientation == 'horizontal_3':
                self.axes = fig.subplots(1, 3)
            else:
                self.axes = fig.subplots(1, 2)
        else:
            self.axes = fig.subplots(1, 1)
            
        super().__init__(fig)
        # Expanding (not Ignored) so the canvas claims leftover space instead of
        # being squeezed to nothing by the control widgets stacked above it.
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(240, 220)
        self.updateGeometry()


class DendrogramDialog(QDialog):
    def __init__(self, embeds, val_array, k, E_arr, A_arr, contrast_scale, gamma_scale, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Hierarchical Family Tree & Representative Spectra")
        self.resize(1100, 700)
        layout = QVBoxLayout(self)

        self.canvas = FigureCanvas(Figure(figsize=(12, 8), dpi=100))
        layout.addWidget(self.canvas)
        fig = self.canvas.figure

        gs = fig.add_gridspec(2, k, height_ratios=[1.5, 1])
        ax_tree = fig.add_subplot(gs[0, :])

        Z = linkage(embeds, method='ward')
        dendro = dendrogram(Z, truncate_mode='lastp', p=k, ax=ax_tree, show_leaf_counts=True)
        ax_tree.set_title(f"Hierarchical Dendrogram (Top {k} Branches)")
        ax_tree.set_ylabel("Mathematical Distance (Merging Threshold)")
        ax_tree.set_xlabel("Number of Pixels in Branch")

        labels = fcluster(Z, k, criterion='maxclust')
        dim_E, dim_A, nY, nX = val_array.shape
        flat_val = val_array.transpose(2, 3, 0, 1).reshape(nY * nX, dim_E, dim_A)

        px_A = (A_arr[-1] - A_arr[0]) / max(1, dim_A-1)
        px_E = (E_arr[-1] - E_arr[0]) / max(1, dim_E-1)
        extent = (A_arr[0]-px_A/2, A_arr[-1]+px_A/2, E_arr[0]-px_E/2, E_arr[-1]+px_E/2)

        for i in range(k):
            cluster_id = i + 1
            mask = (labels == cluster_id)
            
            ax_img = fig.add_subplot(gs[1, i])
            
            if np.any(mask):
                avg_dispersion = flat_val[mask].mean(axis=0)
                d_min, d_max = avg_dispersion.min(), avg_dispersion.max()
                d_max_adj = d_min + (d_max - d_min) * contrast_scale + 1e-8
                norm = mcolors.PowerNorm(gamma=gamma_scale, vmin=d_min, vmax=d_max_adj)
                
                ax_img.imshow(avg_dispersion, aspect='auto', cmap='magma', origin='lower', extent=extent, norm=norm)
                ax_img.set_title(f"Cluster {cluster_id}\n({np.sum(mask)} pixels)", fontsize=10)
            else:
                ax_img.text(0.5, 0.5, "Empty", ha='center', va='center')
                
            ax_img.set_xticks([]); ax_img.set_yticks([])

        fig.tight_layout()


class AzimuthTemplateViewer(QDialog):
    def __init__(self, ref_data, gamma_s_deg, gamma_d_deg, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Interactive Coupled Azimuth & Tilt Visualizer")
        self.resize(1000, 550)
        
        layout = QVBoxLayout(self)
        self.canvas = MplCanvas(self, width=10, height=4, is_3d=False)
        self.canvas.figure.clf() 
        self.ax_iso = self.canvas.figure.add_subplot(121)
        self.ax_cut = self.canvas.figure.add_subplot(122)
        layout.addWidget(self.canvas)
        
        top_ctrl = QHBoxLayout()
        ang_layout = QHBoxLayout()
        self.sl_angle = QSlider(Qt.Orientation.Horizontal)
        self.sl_angle.setRange(0, 359); self.sl_angle.setValue(0)
        self.lbl_angle = QLabel("Rotation: 0°")
        self.sl_angle.valueChanged.connect(self.on_slide)
        ang_layout.addWidget(self.lbl_angle); ang_layout.addWidget(self.sl_angle)
        
        tilt_layout = QHBoxLayout()
        self.sl_tilt = QSlider(Qt.Orientation.Horizontal)
        self.sl_tilt.setRange(-50, 50); self.sl_tilt.setValue(0)
        self.lbl_tilt = QLabel("Deflection Tilt Shift: 0 px")
        self.sl_tilt.valueChanged.connect(self.on_slide)
        tilt_layout.addWidget(self.lbl_tilt); tilt_layout.addWidget(self.sl_tilt)
        
        top_ctrl.addLayout(ang_layout); top_ctrl.addSpacing(20); top_ctrl.addLayout(tilt_layout)
        layout.addLayout(top_ctrl)
        
        split_ctrl = QHBoxLayout()
        iso_ctrl = QHBoxLayout()
        iso_ctrl.addWidget(QLabel("Isoenergy Map Contrast %:"))
        self.sl_c_iso = QSlider(Qt.Orientation.Horizontal)
        self.sl_c_iso.setRange(1, 1000); self.sl_c_iso.setValue(100)
        self.sl_c_iso.valueChanged.connect(self.on_slide)
        iso_ctrl.addWidget(self.sl_c_iso)
        iso_ctrl.addWidget(QLabel("Cmap:"))
        self.combo_cmap_iso = QComboBox(); self.combo_cmap_iso.addItems(['viridis', 'magma', 'inferno', 'plasma', 'cividis', 'gray', 'hsv', 'twilight_shifted', 'PiYG'])
        self.combo_cmap_iso.currentTextChanged.connect(self.on_slide)
        iso_ctrl.addWidget(self.combo_cmap_iso)
        
        cut_ctrl = QHBoxLayout()
        cut_ctrl.addWidget(QLabel("Extracted Cut Contrast %:"))
        self.sl_c_cut = QSlider(Qt.Orientation.Horizontal)
        self.sl_c_cut.setRange(1, 1000); self.sl_c_cut.setValue(100)
        self.sl_c_cut.valueChanged.connect(self.on_slide)
        cut_ctrl.addWidget(self.sl_c_cut)
        cut_ctrl.addWidget(QLabel("Cmap:"))
        self.combo_cmap_cut = QComboBox(); self.combo_cmap_cut.addItems(['magma', 'viridis', 'inferno', 'plasma', 'cividis', 'gray', 'hsv', 'twilight_shifted', 'PiYG'])
        self.combo_cmap_cut.currentTextChanged.connect(self.on_slide)
        cut_ctrl.addWidget(self.combo_cmap_cut)
        
        split_ctrl.addLayout(iso_ctrl); split_ctrl.addSpacing(20); split_ctrl.addLayout(cut_ctrl)
        layout.addLayout(split_ctrl)
        
        try:
            self.prep_data(ref_data, gamma_s_deg, gamma_d_deg)
            self.on_slide()
        except Exception as e:
            self.ax_iso.set_title(f"Error: {e}")
            self.canvas.draw_idle()
        
    def prep_data(self, ref_data, gamma_s_deg, gamma_d_deg):
        self.A_arr = ref_data['angle']
        self.D_arr = ref_data['x']
        
        self.A_scale = abs(self.A_arr[-1] - self.A_arr[0]) / max(1, len(self.A_arr)-1) or 1.0
        self.D_scale = abs(self.D_arr[-1] - self.D_arr[0]) / max(1, len(self.D_arr)-1) or 1.0
        self.aspect_ratio = self.A_scale / self.D_scale
        
        self.gamma_s_px = int(np.argmin(np.abs(self.A_arr - gamma_s_deg)))
        self.gamma_d_px = int(np.argmin(np.abs(self.D_arr - gamma_d_deg)))
        
        E_ref = ref_data['E'].copy()
        if np.mean(E_ref) > 0: E_ref = -E_ref
            
        sort_idx = np.argsort(E_ref)
        self.E_arr = E_ref[sort_idx]
        self.val = ref_data['value'].squeeze(axis=2)[sort_idx, :, :]
        self.val = np.flip(self.val, axis=0)
        
        self.dim_E, self.dim_A, self.dim_D = self.val.shape
        self.E_extent = [self.E_arr[0], self.E_arr[-1]]
        
        self.iso_map = np.sum(self.val, axis=0).T
        self.iso_extent = (self.A_arr[0], self.A_arr[-1], self.D_arr[0], self.D_arr[-1])
        self.a_max = (self.dim_A / 2) * self.A_scale
        
        self.im_iso = self.ax_iso.imshow(self.iso_map, origin='lower', aspect='auto', cmap='viridis', extent=self.iso_extent)
        self.ax_iso.set_title("3D Fermi Volume\n(Isoenergy Projection)")
        self.ax_iso.set_xlabel("Slit Angle (°)"); self.ax_iso.set_ylabel("Deflection Angle (°)")
        
        g_s_phys, g_d_phys = self.A_arr[self.gamma_s_px], self.D_arr[self.gamma_d_px]
        self.ax_iso.plot(g_s_phys, g_d_phys, 'ro', markersize=8)
        self.line_iso, = self.ax_iso.plot([], [], 'r--', linewidth=2)
        
        self.im_cut = self.ax_cut.imshow(np.zeros((self.dim_E, self.dim_A)), aspect='auto', origin='lower', cmap='magma', extent=[-self.a_max, self.a_max, self.E_extent[0], self.E_extent[-1]])
        self.ax_cut.set_title("Extracted Reference Cut")
        self.ax_cut.set_xlabel("Angle from Γ (°)"); self.ax_cut.set_ylabel("Energy (eV)")
        
        self.canvas.figure.tight_layout()
        
    def on_slide(self):
        if not hasattr(self, 'val'): return 
        
        angle = self.sl_angle.value()
        dd = self.sl_tilt.value()
        
        self.lbl_angle.setText(f"Rotation: {angle}°")
        self.lbl_tilt.setText(f"Deflection Tilt Shift: {dd} px")
        
        c_iso = self.sl_c_iso.value() / 100.0
        c_cut = self.sl_c_cut.value() / 100.0
        
        self.im_iso.set_cmap(self.combo_cmap_iso.currentText())
        self.im_cut.set_cmap(self.combo_cmap_cut.currentText())
        
        i_min, i_max = self.iso_map.min(), self.iso_map.max()
        self.im_iso.set_clim(i_min, i_min + (i_max-i_min)*c_iso + 1e-8)
        
        theta = np.deg2rad(angle)
        ds = np.arange(self.dim_A) - (self.dim_A // 2)
        E_coords = np.arange(self.dim_E)
        
        slit_coords = self.gamma_s_px + ds * np.cos(theta) + dd * np.sin(theta) / self.aspect_ratio
        defl_coords = self.gamma_d_px - ds * self.aspect_ratio * np.sin(theta) + dd * np.cos(theta)
        
        EE, SS = np.meshgrid(E_coords, slit_coords, indexing='ij')
        _, DD = np.meshgrid(E_coords, defl_coords, indexing='ij')
        coords = np.stack([EE, SS, DD])
        
        template = map_coordinates(self.val, coords, order=1, mode='constant', cval=0.0)
        
        g_s_phys, g_d_phys = self.A_arr[self.gamma_s_px], self.D_arr[self.gamma_d_px]
        
        x_shift = dd * np.sin(theta) * self.D_scale
        y_shift = dd * np.cos(theta) * self.D_scale
        
        x0 = g_s_phys - self.a_max * np.cos(theta) + x_shift
        x1 = g_s_phys + self.a_max * np.cos(theta) + x_shift
        y0 = g_d_phys + self.a_max * np.sin(theta) + y_shift
        y1 = g_d_phys - self.a_max * np.sin(theta) + y_shift
        
        self.line_iso.set_data([x0, x1], [y0, y1])
        
        self.im_cut.set_data(template)
        vmin_t, vmax_t = np.nanpercentile(template, 2), np.nanpercentile(template, 98)
        if vmax_t == vmin_t: vmax_t = vmin_t + 1
        if not np.isnan(vmin_t) and not np.isnan(vmax_t):
            self.im_cut.set_clim(vmin_t, vmin_t + (vmax_t - vmin_t) * c_cut)
            
        self.ax_cut.set_title(f"Extracted Cut (Rot: {angle}°, Shift: {dd}px)")
        self.canvas.draw_idle()