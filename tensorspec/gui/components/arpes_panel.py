import numpy as np
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                               QComboBox, QPushButton, QDoubleSpinBox, 
                               QFormLayout, QGroupBox, QMessageBox, QSlider, 
                               QSpinBox, QScrollArea, QApplication,QInputDialog)
from PySide6.QtCore import Qt, QThread, Signal

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from tensorspec.core.arpes_engine import ARPESEngineRouter
from tensorspec.core.workspace import global_workspace

class ARPESRunnerThread(QThread):
    """Runs the heavy 2D matrix element loop in the background to prevent UI freezing."""
    finished_signal = Signal(bool, object, str)

    def __init__(self, engine_router, model_choice, crystal_data, experiment_kwargs):
        super().__init__()
        self.engine_router = engine_router
        self.model_choice = model_choice
        self.crystal_data = crystal_data
        self.experiment_kwargs = experiment_kwargs

    def run(self):
        try:
            results = self.engine_router.run_simulation(
                model_choice=self.model_choice,
                crystal_data=self.crystal_data,
                experiment_kwargs=self.experiment_kwargs
            )
            self.finished_signal.emit(True, results, "Success")
        except Exception as e:
            self.finished_signal.emit(False, None, str(e))


class ARPESPanel(QWidget):
    """
    Standalone modular panel for ARPES Matrix Element Simulations.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.engine_router = ARPESEngineRouter()
        self._setup_ui()

    def _setup_ui(self):
        main_sim_layout = QHBoxLayout(self)
        
        # --- LEFT PANEL: CONTROLS ---
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)

        # Workspace & SOC Settings
        ws_group = QGroupBox("0. Crystal Structure & Bands (Workspace)")
        ws_layout = QHBoxLayout(ws_group)
        self.ws_combo = QComboBox()
        self.btn_ws_refresh = QPushButton("🔄 Refresh")
        self.btn_ws_refresh.clicked.connect(self.refresh_workspace)
        ws_layout.addWidget(self.ws_combo)
        ws_layout.addWidget(self.btn_ws_refresh)
        control_layout.addWidget(ws_group)

        # 1. Engine Selector
        engine_group = QGroupBox("1. Simulation Engine Selector")
        engine_layout = QFormLayout(engine_group)
        self.engine_dropdown = QComboBox()
        self.engine_dropdown.addItem("Option B1: One-Step Model (Chinook TB)", "B1")
        self.engine_dropdown.addItem("Option B2: Bare Spectral Function (ME Off)", "B2")
        self.engine_dropdown.addItem("Option B3: Full Multiple Scattering (SPR-KKR)", "B3")
        self.engine_dropdown.addItem("Option A: Phenomenological Three-Step Model", "A")
        engine_layout.addRow(QLabel("Physics Model:"), self.engine_dropdown)
        control_layout.addWidget(engine_group)
        
        # 2. Final State & Thermodynamics
        param_group = QGroupBox("2. Final State & Thermodynamics")
        param_layout = QFormLayout(param_group)
        
        self.photon_energy_spin = QDoubleSpinBox(); self.photon_energy_spin.setRange(5.0, 2000.0); self.photon_energy_spin.setValue(90.0); self.photon_energy_spin.setSuffix(" eV")
        self.work_function_spin = QDoubleSpinBox(); self.work_function_spin.setRange(0.0, 10.0); self.work_function_spin.setValue(4.5); self.work_function_spin.setSuffix(" eV")
        self.inner_potential_spin = QDoubleSpinBox(); self.inner_potential_spin.setRange(0.0, 30.0); self.inner_potential_spin.setValue(15.0); self.inner_potential_spin.setSuffix(" eV")
        self.temperature_spin = QDoubleSpinBox(); self.temperature_spin.setRange(0.1, 1000.0); self.temperature_spin.setValue(10.0); self.temperature_spin.setSuffix(" K")
        
        param_layout.addRow("Photon (hv):", self.photon_energy_spin)
        param_layout.addRow("Work Func (Φ):", self.work_function_spin)
        param_layout.addRow("Inner Pot (V0):", self.inner_potential_spin)
        param_layout.addRow("Temperature:", self.temperature_spin)
        control_layout.addWidget(param_group)

        # 3. Beam & Manipulator Geometry
        beam_group = QGroupBox("3. Beam & Manipulator Geometry")
        beam_layout = QFormLayout(beam_group)
        
        self.manip_theta_spin = QDoubleSpinBox(); self.manip_theta_spin.setRange(-180.0, 180.0); self.manip_theta_spin.setSuffix(" °")
        self.manip_azi_spin = QDoubleSpinBox(); self.manip_azi_spin.setRange(-180.0, 180.0); self.manip_azi_spin.setSuffix(" °")
        self.manip_tilt_spin = QDoubleSpinBox(); self.manip_tilt_spin.setRange(-90.0, 90.0); self.manip_tilt_spin.setSuffix(" °")
        self.incidence_angle_spin = QDoubleSpinBox(); self.incidence_angle_spin.setRange(0.0, 90.0); self.incidence_angle_spin.setValue(55.0); self.incidence_angle_spin.setSuffix(" °")
        
        self.polarization_combo = QComboBox()
        self.polarization_combo.addItems([
            "Linear Horizontal (p-pol)", 
            "Linear Vertical (s-pol)", 
            "Linear Arbitrary", 
            "Circular Right (CR)", 
            "Circular Left (CL)"
        ])
        
        self.lin_pol_angle_spin = QDoubleSpinBox()
        self.lin_pol_angle_spin.setRange(0.0, 360.0)
        self.lin_pol_angle_spin.setValue(45.0)
        self.lin_pol_angle_spin.setSuffix(" °")
        self.lin_pol_angle_spin.setVisible(False)
        
        self.polarization_combo.currentTextChanged.connect(
            lambda text: self.lin_pol_angle_spin.setVisible("Arbitrary" in text)
        )
        
        self.matrix_element_combo = QComboBox()
        self.matrix_element_combo.addItems(["Full Matrix Elements", "Polarization Dipole Only", "Bare Spectral Function (ME Off)"])
        
        beam_layout.addRow("Manipulator Θ (Lab Z):", self.manip_theta_spin)
        beam_layout.addRow("Manipulator Azimuth:", self.manip_azi_spin)
        beam_layout.addRow("Manipulator Tilt:", self.manip_tilt_spin)
        beam_layout.addRow("Beam Incidence (Lab):", self.incidence_angle_spin)
        beam_layout.addRow("Polarization:", self.polarization_combo)
        beam_layout.addRow("Pol. Angle (Arbitrary):", self.lin_pol_angle_spin)
        beam_layout.addRow("Intensity Mode:", self.matrix_element_combo)
        control_layout.addWidget(beam_group)
        
        # 4. Analyzer Domain & Resolution
        domain_group = QGroupBox("4. Analyzer Domain & Resolution")
        domain_layout = QVBoxLayout(domain_group)
        
        analyzer_layout = QFormLayout()
        self.slit_size_spin = QDoubleSpinBox(); self.slit_size_spin.setRange(0.1, 5.0); self.slit_size_spin.setValue(0.5); self.slit_size_spin.setSingleStep(0.1); self.slit_size_spin.setSuffix(" mm")
        self.slit_angle_spin = QDoubleSpinBox(); self.slit_angle_spin.setRange(-180.0, 180.0); self.slit_angle_spin.setValue(0.0); self.slit_angle_spin.setSuffix(" °")
        self.deflector_angle_spin = QDoubleSpinBox(); self.deflector_angle_spin.setRange(-15.0, 15.0); self.deflector_angle_spin.setValue(0.0); self.deflector_angle_spin.setSuffix(" °")
        analyzer_layout.addRow("Analyzer Slit Size:", self.slit_size_spin)
        analyzer_layout.addRow("Analyzer Slit Angle:", self.slit_angle_spin)
        analyzer_layout.addRow("Deflector Angle:", self.deflector_angle_spin)
        domain_layout.addLayout(analyzer_layout)
        
        def create_range_row(label_text, min_val, max_val, pts_val):
            row = QHBoxLayout()
            row.addWidget(QLabel(label_text))
            vmin = QDoubleSpinBox(); vmin.setRange(-10.0, 10.0); vmin.setValue(min_val); vmin.setSingleStep(0.1)
            vmax = QDoubleSpinBox(); vmax.setRange(-10.0, 10.0); vmax.setValue(max_val); vmax.setSingleStep(0.1)
            vpts = QSpinBox(); vpts.setRange(10, 1000); vpts.setValue(40); vpts.setSingleStep(10)
            row.addWidget(QLabel("Min:")); row.addWidget(vmin)
            row.addWidget(QLabel("Max:")); row.addWidget(vmax)
            row.addWidget(QLabel("Pts:")); row.addWidget(vpts)
            return row, vmin, vmax, vpts

        # Relabeled strictly to Slit and Deflector axes
        row_kx, self.spin_kx_min, self.spin_kx_max, self.spin_kx_steps = create_range_row("kx (Slit):", -2.0, 2.0, 100)
        row_ky, self.spin_ky_min, self.spin_ky_max, self.spin_ky_steps = create_range_row("ky (Deflect):", -2.0, 2.0, 100)
        row_e, self.spin_e_min, self.spin_e_max, self.spin_e_steps = create_range_row("E:", -2.0, 0.5, 100)

        domain_layout.addLayout(row_kx)
        domain_layout.addLayout(row_ky)
        domain_layout.addLayout(row_e)
        
        res_layout = QFormLayout()
        self.ui_se_spinbox = QDoubleSpinBox(); self.ui_se_spinbox.setRange(0.001, 1.0); self.ui_se_spinbox.setValue(0.01); self.ui_se_spinbox.setSingleStep(0.01); self.ui_se_spinbox.setDecimals(3); self.ui_se_spinbox.setSuffix(" eV")
        self.ui_res_e_spinbox = QDoubleSpinBox(); self.ui_res_e_spinbox.setRange(0.001, 1.0); self.ui_res_e_spinbox.setValue(0.02); self.ui_res_e_spinbox.setSingleStep(0.01); self.ui_res_e_spinbox.setDecimals(3); self.ui_res_e_spinbox.setSuffix(" eV")
        self.ui_res_k_spinbox = QDoubleSpinBox(); self.ui_res_k_spinbox.setRange(0.001, 1.0); self.ui_res_k_spinbox.setValue(0.02); self.ui_res_k_spinbox.setSingleStep(0.01); self.ui_res_k_spinbox.setDecimals(3); self.ui_res_k_spinbox.setSuffix(" 1/A")
        
        res_layout.addRow("Peak Linewidth (SE):", self.ui_se_spinbox)
        res_layout.addRow("Energy Resolution (dE):", self.ui_res_e_spinbox)
        res_layout.addRow("Momentum Resolution (dk):", self.ui_res_k_spinbox)
        domain_layout.addLayout(res_layout)
        
        control_layout.addWidget(domain_group)
        
        # Run Button
        self.run_sim_btn = QPushButton("🚀 Run ARPES Simulation")
        self.run_sim_btn.setStyleSheet("font-weight: bold; padding: 10px; background-color: #2b5c8f; color: white;")
        self.run_sim_btn.clicked.connect(self.trigger_simulation)
        control_layout.addWidget(self.run_sim_btn)
        
        control_layout.addStretch() 
        
        scroll_area = QScrollArea()
        scroll_area.setWidget(control_panel)
        scroll_area.setWidgetResizable(True)
        scroll_area.setMinimumWidth(400) 
        main_sim_layout.addWidget(scroll_area, stretch=0)

        # --- MIDDLE PANEL: SCHEMATIC VIEWER ---
        schematic_panel = QWidget()
        schematic_layout = QVBoxLayout(schematic_panel)
        self.schematic_figure = Figure(figsize=(4, 5))
        self.schematic_canvas = FigureCanvas(self.schematic_figure)
        
        self.ax_schematic = self.schematic_figure.add_subplot(111, projection='3d')
        schematic_layout.addWidget(self.schematic_canvas)
        main_sim_layout.addWidget(schematic_panel, stretch=1)

        self.incidence_angle_spin.valueChanged.connect(self.update_schematic)
        self.polarization_combo.currentTextChanged.connect(self.update_schematic)
        self.deflector_angle_spin.valueChanged.connect(self.update_schematic)
        self.slit_size_spin.valueChanged.connect(self.update_schematic)
        self.manip_theta_spin.valueChanged.connect(self.update_schematic)
        self.manip_azi_spin.valueChanged.connect(self.update_schematic)
        self.manip_tilt_spin.valueChanged.connect(self.update_schematic)
        self.lin_pol_angle_spin.valueChanged.connect(self.update_schematic)
        self.slit_angle_spin.valueChanged.connect(self.update_schematic)

        # --- RIGHT PANEL: INTERACTIVE PLOTTER ---
        plot_panel = QWidget()
        plot_layout = QVBoxLayout(plot_panel)
        self.figure = Figure(figsize=(5, 5))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("Simulated ARPES Intensity")
        plot_layout.addWidget(self.canvas, stretch=1)
        
        slider_layout = QHBoxLayout()
        self.energy_label = QLabel("Binding Energy: 0.00 eV")
        self.energy_label.setFixedWidth(150)
        self.energy_slider = QSlider(Qt.Horizontal)
        self.energy_slider.setEnabled(False)
        self.energy_slider.valueChanged.connect(self.update_plot_slice)
        slider_layout.addWidget(self.energy_label)
        slider_layout.addWidget(self.energy_slider)
        plot_layout.addLayout(slider_layout)
        
        contrast_layout = QHBoxLayout()
        self.vmin_spin = QDoubleSpinBox(); self.vmin_spin.setRange(0.0, 1.0); self.vmin_spin.setValue(0.0); self.vmin_spin.setSingleStep(0.05)
        self.vmax_spin = QDoubleSpinBox(); self.vmax_spin.setRange(0.0, 1.0); self.vmax_spin.setValue(1.0); self.vmax_spin.setSingleStep(0.05)
        self.gamma_spin = QDoubleSpinBox(); self.gamma_spin.setRange(0.1, 5.0); self.gamma_spin.setValue(1.0); self.gamma_spin.setSingleStep(0.1)
        
        contrast_layout.addWidget(QLabel("Min:"))
        contrast_layout.addWidget(self.vmin_spin)
        contrast_layout.addWidget(QLabel("Max:"))
        contrast_layout.addWidget(self.vmax_spin)
        contrast_layout.addWidget(QLabel("γ (Gamma):"))
        contrast_layout.addWidget(self.gamma_spin)
        plot_layout.addLayout(contrast_layout)
        
        self.vmin_spin.valueChanged.connect(lambda: self.update_plot_slice(self.energy_slider.value()))
        self.vmax_spin.valueChanged.connect(lambda: self.update_plot_slice(self.energy_slider.value()))
        self.gamma_spin.valueChanged.connect(lambda: self.update_plot_slice(self.energy_slider.value()))
        
        main_sim_layout.addWidget(plot_panel, stretch=1)
        self.update_schematic()

        # Run Button
        self.run_sim_btn = QPushButton("🚀 Run ARPES Simulation")
        self.run_sim_btn.setStyleSheet("font-weight: bold; padding: 10px; background-color: #2b5c8f; color: white;")
        self.run_sim_btn.clicked.connect(self.trigger_simulation)
        control_layout.addWidget(self.run_sim_btn)
        
        # --- NEW: Save Button ---
        self.save_data_btn = QPushButton("💾 Save Simulated Data")
        self.save_data_btn.setStyleSheet("font-weight: bold; padding: 10px; background-color: #2e7d32; color: white;")
        self.save_data_btn.setEnabled(False) # Disabled until a simulation finishes
        self.save_data_btn.clicked.connect(self.save_current_simulation)
        control_layout.addWidget(self.save_data_btn)

    def update_schematic(self, *args):
        self.draw_hemisphere_schematic(
            self.manip_theta_spin.value(),
            self.manip_azi_spin.value(),
            self.manip_tilt_spin.value(),
            self.incidence_angle_spin.value(),
            self.polarization_combo.currentText(),
            self.deflector_angle_spin.value(),
            self.slit_size_spin.value(),
            self.slit_angle_spin.value()
        )

    def draw_hemisphere_schematic(self, theta, azimuth, tilt, incidence_angle, pol_mode, deflector_angle, slit_width, slit_angle):
        self.ax_schematic.clear()
        t_rad, a_rad, tilt_rad = np.radians(theta), np.radians(azimuth), np.radians(tilt)
        R_z = np.array([[np.cos(t_rad), -np.sin(t_rad), 0], [np.sin(t_rad), np.cos(t_rad), 0], [0, 0, 1]])
        R_y = np.array([[np.cos(a_rad), 0, np.sin(a_rad)], [0, 1, 0], [-np.sin(a_rad), 0, np.cos(a_rad)]])
        R_x = np.array([[1, 0, 0], [0, np.cos(tilt_rad), -np.sin(tilt_rad)], [0, np.sin(tilt_rad), np.cos(tilt_rad)]])
        R_total = R_z @ R_y @ R_x
        R_inv = np.linalg.inv(R_total)

        inc_rad = np.radians(incidence_angle)
        lin_ang = np.radians(self.lin_pol_angle_spin.value())
        is_circular = "Circular" in pol_mode
        
        if "Horizontal" in pol_mode: eps_lab = np.array([np.cos(inc_rad), -np.sin(inc_rad), 0.0])
        elif "Vertical" in pol_mode: eps_lab = np.array([0.0, 0.0, 1.0])
        elif "Arbitrary" in pol_mode:
            eps_lab = np.cos(lin_ang)*np.array([np.cos(inc_rad), -np.sin(inc_rad), 0.0]) + np.sin(lin_ang)*np.array([0.0, 0.0, 1.0])
        else:
            sign = 1 if "Right" in pol_mode else -1
            eps_lab = (np.array([np.cos(inc_rad), -np.sin(inc_rad), 0.0]) + sign*1j*np.array([0.0, 0.0, 1.0])) / np.sqrt(2)
        
        eps_sample = R_inv @ eps_lab

        phi = np.linspace(0, 2*np.pi, 40)
        theta_emi = np.linspace(0, np.pi/2, 20)
        PHI, THETA_EMI = np.meshgrid(phi, theta_emi)
        
        X_hemi_loc = 1.2 * np.sin(THETA_EMI) * np.cos(PHI)
        Z_hemi_loc = 1.2 * np.sin(THETA_EMI) * np.sin(PHI)
        Y_hemi_loc = 1.2 * np.cos(THETA_EMI)
        
        dot_product = eps_sample[0]*(X_hemi_loc/1.2) + eps_sample[1]*(Y_hemi_loc/1.2) + eps_sample[2]*(Z_hemi_loc/1.2)
        ME_heatmap = np.abs(dot_product)**2
        
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
        norm = mcolors.Normalize(vmin=ME_heatmap.min(), vmax=ME_heatmap.max())
        colors = cm.afmhot(norm(ME_heatmap))
        
        shape = X_hemi_loc.shape
        coords = np.vstack([X_hemi_loc.flatten(), Y_hemi_loc.flatten(), Z_hemi_loc.flatten()])
        coords_rot = R_total @ coords
        X_hemi_lab, Y_hemi_lab, Z_hemi_lab = coords_rot[0].reshape(shape), coords_rot[1].reshape(shape), coords_rot[2].reshape(shape)
        
        self.ax_schematic.plot_surface(X_hemi_lab, Y_hemi_lab, Z_hemi_lab, facecolors=colors, alpha=0.7, shade=False)

        s_range = np.linspace(-1, 1, 2)
        X_s, Z_s = np.meshgrid(s_range, s_range)
        Y_s = np.zeros_like(X_s)
        s_coords = R_total @ np.vstack([X_s.flatten(), Y_s.flatten(), Z_s.flatten()])
        self.ax_schematic.plot_surface(s_coords[0].reshape(2,2), s_coords[1].reshape(2,2), s_coords[2].reshape(2,2), color='darkgray', alpha=0.9, edgecolor='k')

        axes_lab = R_total @ np.array([[1.5, 0, 0], [0, 1.5, 0], [0, 0, 1.5]]).T
        self.ax_schematic.quiver(0,0,0, axes_lab[0,0], axes_lab[1,0], axes_lab[2,0], color='red', linewidth=2, label='Sample X')
        self.ax_schematic.quiver(0,0,0, axes_lab[0,1], axes_lab[1,1], axes_lab[2,1], color='lime', linewidth=2, label='Sample Y (Normal)')
        self.ax_schematic.quiver(0,0,0, axes_lab[0,2], axes_lab[1,2], axes_lab[2,2], color='blue', linewidth=2, label='Sample Z')

        beam_dir = np.array([-np.sin(inc_rad), -np.cos(inc_rad), 0.0]) 
        start_pt = -4 * beam_dir
        t_beam = np.linspace(0, 4, 150)
        
        if is_circular:
            freq = 6 * np.pi
            radius = 0.3
            sign = 1 if "Right" in pol_mode else -1
            perp1 = np.array([0, 0, 1])
            perp2 = np.cross(beam_dir, perp1)
            wave_pts = start_pt[:, None] + beam_dir[:, None] * t_beam + radius * (np.cos(freq * t_beam) * perp1[:, None] + sign * np.sin(freq * t_beam) * perp2[:, None])
            self.ax_schematic.plot(wave_pts[0], wave_pts[1], wave_pts[2], color='magenta', linewidth=2, label='Circular Light')
        else:
            freq = 5 * np.pi
            amplitude = 0.3
            osc_dir = np.real(eps_lab)
            wave_pts = start_pt[:, None] + beam_dir[:, None] * t_beam + osc_dir[:, None] * amplitude * np.sin(freq * t_beam)
            self.ax_schematic.plot(wave_pts[0], wave_pts[1], wave_pts[2], color='gold', linewidth=2, label='Linear Light')
            self.ax_schematic.quiver(start_pt[0], start_pt[1], start_pt[2], np.real(eps_lab)[0], np.real(eps_lab)[1], np.real(eps_lab)[2], length=1.5, color='magenta', linewidth=2, label='E-Field')
        
        defl_rad, slit_rot = np.radians(deflector_angle), np.radians(slit_angle)
        
        # --- FIXED DETECTOR PLANE ---
        # The detector is physically bolted to the Lab Frame at Y = 1.25
        y_det = 1.25
        plane_size = 0.5
        px, pz = np.meshgrid(np.linspace(-plane_size, plane_size, 5), np.linspace(-plane_size, plane_size, 5))
        py = np.full_like(px, y_det)
        
        self.ax_schematic.plot_surface(px, py, pz, color='gray', alpha=0.3, edgecolor='blue', label='Detector Plane')

        # --- DRAW ANALYZER SLIT (Fixed Cyan Line) ---
        slit_len = 0.5
        sx = slit_len * np.cos(slit_rot)
        sz = slit_len * np.sin(slit_rot)
        self.ax_schematic.plot([-sx, sx], [y_det, y_det], [-sz, sz], color='cyan', linewidth=4, zorder=10, label='Analyzer Slit')

        # --- DRAW DEFLECTOR CUT (Animated Yellow Line) ---
        # The deflector shifts the measurement orthogonally to the slit
        shift_dist = y_det * np.tan(defl_rad)
        dx_shift = shift_dist * (-np.sin(slit_rot))
        dz_shift = shift_dist * (np.cos(slit_rot))
        
        self.ax_schematic.plot([-sx + dx_shift, sx + dx_shift], 
                               [y_det, y_det], 
                               [-sz + dz_shift, sz + dz_shift], 
                               color='yellow', linestyle='--', linewidth=3, zorder=11, label='Deflector Cut')
                               
        # Draw a small arrow showing the deflection direction
        if abs(deflector_angle) > 0.1:
            self.ax_schematic.quiver(0, y_det, 0, dx_shift, 0, dz_shift, color='yellow', linewidth=2, arrow_length_ratio=0.3)

        self.ax_schematic.plot([0, 0], [0, 0], [-3, 3], color='black', linestyle='-.', linewidth=1, label='Lab Z (Theta)')

        self.ax_schematic.set_xlim([-3, 3]); self.ax_schematic.set_ylim([-3, 3]); self.ax_schematic.set_zlim([-3, 3])
        self.ax_schematic.set_xlabel('Lab X (Side)'); self.ax_schematic.set_ylabel('Lab Y (Forward)'); self.ax_schematic.set_zlabel('Lab Z (Vertical)')
        self.ax_schematic.set_title("Experimental Geometry Schematic", pad=0)
        self.ax_schematic.legend(loc='center left', fontsize=7, bbox_to_anchor=(1.05, 0.5))
        self.ax_schematic.xaxis.pane.fill = False; self.ax_schematic.yaxis.pane.fill = False; self.ax_schematic.zaxis.pane.fill = False
        
        self.schematic_figure.subplots_adjust(left=0.0, right=0.8, top=1.0, bottom=0.0) 
        self.schematic_canvas.draw()

    def trigger_simulation(self):
        target_crystal = self.ws_combo.currentText()
        model_choice = self.engine_dropdown.currentData()
        me_mode = self.matrix_element_combo.currentText()

        print("\n" + "="*50)
        print("🚀 INITIATING ARPES MATRIX ELEMENT SIMULATION")
        print("="*50)
        print(f"Target Structure : {target_crystal}")
        print(f"Selected Physics : {model_choice} ({me_mode})")
        print(f"Polarization     : {self.polarization_combo.currentText()}")
        print(f"Incidence Angle  : {self.incidence_angle_spin.value()}\u00B0")
        print("="*50 + "\n")

        e_min, e_max, e_steps = self.spin_e_min.value(), self.spin_e_max.value(), self.spin_e_steps.value()
        kx_min, kx_max, kx_steps = self.spin_kx_min.value(), self.spin_kx_max.value(), self.spin_kx_steps.value()
        ky_min, ky_max, ky_steps = self.spin_ky_min.value(), self.spin_ky_max.value(), self.spin_ky_steps.value()

        band_data = global_workspace.pull_band_structure(target_crystal)

        if not band_data:
            print(">> WARNING: No pre-calculated band structure found.")
            return
            
        print(">> SUCCESS: Band structure loaded. Routing to Matrix Element Engine...")
        
        experiment_kwargs = {
            'photon_energy': self.photon_energy_spin.value(),
            'work_function': self.work_function_spin.value(),
            'inner_potential': self.inner_potential_spin.value(), 
            'temperature': self.temperature_spin.value(),         
            'incidence_angle': self.incidence_angle_spin.value(),
            'polarization': self.polarization_combo.currentText(),
            'lin_pol_angle': self.lin_pol_angle_spin.value(),
            'matrix_element_mode': me_mode,
            'manip_theta': self.manip_theta_spin.value(),
            'manip_azimuth': self.manip_azi_spin.value(),
            'manip_tilt': self.manip_tilt_spin.value(),
            'k_bounds': {'X': [kx_min, kx_max, kx_steps], 'Y': [ky_min, ky_max, ky_steps], 'E': [e_min, e_max, e_steps]},
            'se_width': self.ui_se_spinbox.value(),
            'res_E': self.ui_res_e_spinbox.value(),
            'res_k': self.ui_res_k_spinbox.value(),
            'slit_angle': self.slit_angle_spin.value()
        }
        
        # --- THREADED LAUNCH ---
        self.ax.clear()
        self.ax.set_title("Calculating 2D Matrix Elements... (Running in Background)")
        self.canvas.draw()
        
        self.run_sim_btn.setEnabled(False)
        self.run_sim_btn.setText("⏳ Calculating... Please Wait")
        self.run_sim_btn.setStyleSheet("font-weight: bold; padding: 10px; background-color: #555555; color: white;")

        self.arpes_thread = ARPESRunnerThread(self.engine_router, model_choice, band_data, experiment_kwargs)
        self.arpes_thread.finished_signal.connect(self.on_simulation_finished)
        self.arpes_thread.start()

    def on_simulation_finished(self, success, results, message):
        self.run_sim_btn.setEnabled(True)
        self.run_sim_btn.setText("🚀 Run ARPES Simulation")
        self.run_sim_btn.setStyleSheet("font-weight: bold; padding: 10px; background-color: #2b5c8f; color: white;")

        if success:
            self.save_data_btn.setEnabled(True)  # <--- NEW: Enable the save button
            
            e_min, e_max, e_steps = self.spin_e_min.value(), self.spin_e_max.value(), self.spin_e_steps.value()
            kx_min, kx_max, kx_steps = self.spin_kx_min.value(), self.spin_kx_max.value(), self.spin_kx_steps.value()
            ky_min, ky_max, ky_steps = self.spin_ky_min.value(), self.spin_ky_max.value(), self.spin_ky_steps.value()

            self.sim_intensity = results['intensity_broadened']
            
            # --- PATCH: DYNAMIC DIMENSION DETECTION ---
            # Read the actual dimensions returned by the backend, ignoring UI spinboxes if they were bypassed
            nx, ny, ne = self.sim_intensity.shape
            
            self.sim_E_axis = np.linspace(e_min, e_max, ne)
            self.sim_kx = np.linspace(kx_min, kx_max, nx)
            self.sim_ky = np.linspace(ky_min, ky_max, ny)
            
            if ny == 1 or nx == 1:
                # It is a 2D Band Dispersion Map (Energy vs Momentum)
                self.energy_slider.setEnabled(False)
                self.update_plot_slice(index=None)
            else:
                # It is a 3D Cube (Constant Energy Contours)
                self.energy_slider.setRange(0, ne - 1)
                fermi_idx = np.abs(self.sim_E_axis).argmin()
                self.energy_slider.setEnabled(True)
                self.energy_slider.setValue(fermi_idx)
                
                self.update_plot_slice(fermi_idx)
        else:
            QMessageBox.critical(self, "Simulation Error", f"An error occurred in the physics router:\n{message}")
            self.ax.set_title("Simulation Failed")
            self.canvas.draw()

    def update_plot_slice(self, index=None):
        if not hasattr(self, 'sim_intensity'):
            return
            
        nx, ny, ne = self.sim_intensity.shape
        self.ax.clear()

        # --- ROUTE 1: PLOT BAND DISPERSION (E vs k) ---
        if ny == 1 or nx == 1:
            self.energy_label.setText(f"Band Dispersion Map")
            
            if ny == 1:
                slice_2d = self.sim_intensity[:, 0, :].T  
                x_axis = self.sim_kx
                x_label = r"$k_x$ ($\mathrm{\AA}^{-1}$)"
            else:
                slice_2d = self.sim_intensity[0, :, :].T  
                x_axis = self.sim_ky
                x_label = r"$k_y$ ($\mathrm{\AA}^{-1}$)"
                
            slice_max = np.max(slice_2d) if np.max(slice_2d) > 0 else 1.0
            norm_slice = slice_2d / slice_max
            gamma_corrected = np.power(norm_slice, self.gamma_spin.value())
            
            self.ax.pcolormesh(x_axis, self.sim_E_axis, gamma_corrected, shading='auto', cmap='afmhot', 
                               vmin=self.vmin_spin.value(), vmax=self.vmax_spin.value())
            
            self.ax.set_aspect('auto')  # Free the aspect ratio for bands
            self.ax.set_title("Simulated ARPES Band Dispersion")
            self.ax.set_xlabel(x_label)
            self.ax.set_ylabel("Binding Energy (eV)")

        # --- ROUTE 2: PLOT CONSTANT ENERGY CONTOUR (kx vs ky) ---
        else:
            if index is None: 
                index = self.energy_slider.value()
                
            E_val = self.sim_E_axis[index]
            self.energy_label.setText(f"Binding Energy: {E_val:.3f} eV")
            
            slice_2d = self.sim_intensity[:, :, index].T 
            
            slice_max = np.max(slice_2d) if np.max(slice_2d) > 0 else 1.0
            norm_slice = slice_2d / slice_max
            gamma_corrected = np.power(norm_slice, self.gamma_spin.value())
            
            self.ax.pcolormesh(self.sim_kx, self.sim_ky, gamma_corrected, shading='auto', cmap='afmhot', 
                               vmin=self.vmin_spin.value(), vmax=self.vmax_spin.value())
            
            self.ax.set_aspect('equal') # Lock aspect ratio for Fermi surfaces
            self.ax.set_title(f"Constant Energy Contour: {E_val:.2f} eV")
            self.ax.set_xlabel(r"$k_x$ ($\mathrm{\AA}^{-1}$)")
            self.ax.set_ylabel(r"$k_y$ ($\mathrm{\AA}^{-1}$)")
            
        self.figure.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)
        self.canvas.draw()
    

    def save_current_simulation(self):
        if not hasattr(self, 'sim_intensity'):
            return
            
        # Prompt the user for a dataset name
        name, ok = QInputDialog.getText(self, "Save Simulation", "Enter dataset name (e.g., WTe2_75eV_CR):")
        
        if ok and name:
            # Gather the metadata from the UI
            metadata = {
                'crystal': self.ws_combo.currentText(),
                'photon_energy': self.photon_energy_spin.value(),
                'work_function': self.work_function_spin.value(),
                'temperature': self.temperature_spin.value(),
                'polarization': self.polarization_combo.currentText(),
                'manip_theta': self.manip_theta_spin.value(),
                'manip_azi': self.manip_azi_spin.value(),
                'manip_tilt': self.manip_tilt_spin.value(),
                'slit_angle': self.slit_angle_spin.value()
            }
            
            # Save via workspace
            try:
                global_workspace.save_simulated_arpes(
                    name, 
                    self.sim_intensity, 
                    self.sim_kx, 
                    self.sim_ky, 
                    self.sim_E_axis, 
                    metadata
                )
                QMessageBox.information(self, "Success", f"Data successfully saved as '{name}'")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save data:\n{e}")
    
    def refresh_workspace(self):
        self.ws_combo.clear()
        bands = global_workspace.list_band_structures()
        if bands:
            self.ws_combo.addItems(bands)
        else:
            self.ws_combo.addItem("No band structures available")