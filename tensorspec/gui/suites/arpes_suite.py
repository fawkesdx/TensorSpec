import sys
import numpy as np
import matplotlib.patches as patches
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, 
                               QLabel, QComboBox, QPushButton, QDoubleSpinBox, 
                               QFormLayout, QGroupBox, QMessageBox, QSlider, 
                               QSpinBox, QScrollArea, QApplication)
from PySide6.QtCore import Qt

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from tensorspec.core.arpes_engine import ARPESEngineRouter
from tensorspec.core.workspace import global_workspace

class ARPESSuite(QWidget):
    """
    Main container for all ARPES-related tools. 
    Implements a tabbed architecture to separate simulation from real data analysis.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.setAttribute(Qt.WA_DeleteOnClose)
        self._is_closing = False
        
        # Initialize the backend router 
        self.engine_router = ARPESEngineRouter()
        
        self.layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        
        # Tab 1: Hierarchical Simulation Engine
        self.simulation_tab = QWidget()
        self.init_simulation_tab()
        self.tabs.addTab(self.simulation_tab, "Matrix Element Simulator")
        
        # Tab 2: Experimental Data Viewer
        self.data_tab = QWidget()
        data_layout = QVBoxLayout(self.data_tab)
        data_layout.addWidget(QLabel("Real ARPES Data Viewer & Crosshairs will go here."))
        self.tabs.addTab(self.data_tab, "Data Viewer")
        
        self.layout.addWidget(self.tabs)
        
    def init_simulation_tab(self):
        """Builds the UI for the Simulation Engine Selection and experimental parameters."""
        main_sim_layout = QHBoxLayout(self.simulation_tab)
        
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
        self.engine_dropdown.addItem("Option B1: One-Step Model (Chinook)", "B1")
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
        
        # --- UPGRADED POLARIZATION ---
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
        self.lin_pol_angle_spin.setVisible(False) # Hidden by default
        
        self.polarization_combo.currentTextChanged.connect(
            lambda text: self.lin_pol_angle_spin.setVisible("Arbitrary" in text)
        )
        
        self.matrix_element_combo = QComboBox()
        self.matrix_element_combo.addItems(["Full Matrix Elements", "Polarization Dipole Only", "Bare Spectral Function (ME Off)"])
        
        self.slit_size_spin = QDoubleSpinBox(); self.slit_size_spin.setRange(0.1, 5.0); self.slit_size_spin.setValue(0.5); self.slit_size_spin.setSingleStep(0.1); self.slit_size_spin.setSuffix(" mm")
        self.slit_angle_spin = QDoubleSpinBox(); self.slit_angle_spin.setRange(-180.0, 180.0); self.slit_angle_spin.setValue(0.0); self.slit_angle_spin.setSuffix(" °")
        self.deflector_angle_spin = QDoubleSpinBox(); self.deflector_angle_spin.setRange(-15.0, 15.0); self.deflector_angle_spin.setValue(0.0); self.deflector_angle_spin.setSuffix(" °")

        beam_layout.addRow("Manipulator Θ (Lab Z):", self.manip_theta_spin)
        beam_layout.addRow("Manipulator Azimuth:", self.manip_azi_spin)
        beam_layout.addRow("Manipulator Tilt:", self.manip_tilt_spin)
        beam_layout.addRow("Beam Incidence (Lab):", self.incidence_angle_spin)
        beam_layout.addRow("Polarization:", self.polarization_combo)
        beam_layout.addRow("Pol. Angle (Arbitrary):", self.lin_pol_angle_spin)
        beam_layout.addRow("Intensity Mode:", self.matrix_element_combo)
        beam_layout.addRow("Analyzer Slit Size:", self.slit_size_spin)
        beam_layout.addRow("Analyzer Slit Angle:", self.slit_angle_spin)
        beam_layout.addRow("Deflector Angle:", self.deflector_angle_spin)
        control_layout.addWidget(beam_group)
        
        # 4. Domain & Resolution
        domain_group = QGroupBox("4. Domain & Resolution")
        domain_layout = QVBoxLayout(domain_group)
        
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

        row_kx, self.spin_kx_min, self.spin_kx_max, self.spin_kx_steps = create_range_row("kx:", -2.0, 2.0, 100)
        row_ky, self.spin_ky_min, self.spin_ky_max, self.spin_ky_steps = create_range_row("ky:", -2.0, 2.0, 100)
        row_e, self.spin_e_min, self.spin_e_max, self.spin_e_steps = create_range_row("E:", -2.0, 0.5, 100)

        domain_layout.addLayout(row_kx)
        domain_layout.addLayout(row_ky)
        domain_layout.addLayout(row_e)
        
        # --- NEW: Resolution & Broadening Controls ---
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
        
        # Create strictly ONE Scroll Area for the left panel
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
        
        # Make the schematic a 3D projection
        self.ax_schematic = self.schematic_figure.add_subplot(111, projection='3d')
        
        schematic_layout.addWidget(self.schematic_canvas)
        main_sim_layout.addWidget(schematic_panel, stretch=1)

        # Connect UI changes to update the schematic LIVE
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
        
        # --- NEW: Contrast & Gamma Controls ---
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
        
        # Connect contrast updates (re-draws current slice dynamically)
        self.vmin_spin.valueChanged.connect(lambda: self.update_plot_slice(self.energy_slider.value()))
        self.vmax_spin.valueChanged.connect(lambda: self.update_plot_slice(self.energy_slider.value()))
        self.gamma_spin.valueChanged.connect(lambda: self.update_plot_slice(self.energy_slider.value()))
        
        main_sim_layout.addWidget(plot_panel, stretch=1)
        self.update_schematic()

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
        
        # 1. Rotation Matrices
        t_rad, a_rad, tilt_rad = np.radians(theta), np.radians(azimuth), np.radians(tilt)
        R_z = np.array([[np.cos(t_rad), -np.sin(t_rad), 0], [np.sin(t_rad), np.cos(t_rad), 0], [0, 0, 1]])
        R_y = np.array([[np.cos(a_rad), 0, np.sin(a_rad)], [0, 1, 0], [-np.sin(a_rad), 0, np.cos(a_rad)]])
        R_x = np.array([[1, 0, 0], [0, np.cos(tilt_rad), -np.sin(tilt_rad)], [0, np.sin(tilt_rad), np.cos(tilt_rad)]])
        R_total = R_z @ R_y @ R_x
        R_inv = np.linalg.inv(R_total)

        # 2. Polarization Vector Setup (Lab Frame)
        inc_rad = np.radians(incidence_angle)
        lin_ang = np.radians(self.lin_pol_angle_spin.value())
        is_circular = "Circular" in pol_mode
        
        if "Horizontal" in pol_mode: eps_lab = np.array([np.cos(inc_rad), -np.sin(inc_rad), 0.0])
        elif "Vertical" in pol_mode: eps_lab = np.array([0.0, 0.0, 1.0])
        elif "Arbitrary" in pol_mode:
            eps_lab = np.cos(lin_ang)*np.array([np.cos(inc_rad), -np.sin(inc_rad), 0.0]) + np.sin(lin_ang)*np.array([0.0, 0.0, 1.0])
        else: # Circular Right / Left
            sign = 1 if "Right" in pol_mode else -1
            eps_lab = (np.array([np.cos(inc_rad), -np.sin(inc_rad), 0.0]) + sign*1j*np.array([0.0, 0.0, 1.0])) / np.sqrt(2)
        
        # Transform light to sample frame to calculate matrix elements
        eps_sample = R_inv @ eps_lab

        # 3. Draw Colored Emission Hemisphere (Sideways, facing Local +Y)
        # We generate a hemisphere in the local sample frame (Normal = Y)
        phi = np.linspace(0, 2*np.pi, 40)
        theta_emi = np.linspace(0, np.pi/2, 20)
        PHI, THETA_EMI = np.meshgrid(phi, theta_emi)
        
        # Coordinates in Sample Frame (radius = 1.2)
        X_hemi_loc = 1.2 * np.sin(THETA_EMI) * np.cos(PHI)
        Z_hemi_loc = 1.2 * np.sin(THETA_EMI) * np.sin(PHI)
        Y_hemi_loc = 1.2 * np.cos(THETA_EMI) # Emitting along normal
        
        # Calculate Matrix Element Heatmap (Dipole Approximation | A \cdot k |^2)
        # Using the unit vectors of the hemisphere as k_f
        dot_product = eps_sample[0]*(X_hemi_loc/1.2) + eps_sample[1]*(Y_hemi_loc/1.2) + eps_sample[2]*(Z_hemi_loc/1.2)
        ME_heatmap = np.abs(dot_product)**2
        
        # Normalize heatmap for coloring
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
        norm = mcolors.Normalize(vmin=ME_heatmap.min(), vmax=ME_heatmap.max())
        colors = cm.afmhot(norm(ME_heatmap))
        
        # Rotate Hemisphere to Lab Frame for drawing
        shape = X_hemi_loc.shape
        coords = np.vstack([X_hemi_loc.flatten(), Y_hemi_loc.flatten(), Z_hemi_loc.flatten()])
        coords_rot = R_total @ coords
        X_hemi_lab, Y_hemi_lab, Z_hemi_lab = coords_rot[0].reshape(shape), coords_rot[1].reshape(shape), coords_rot[2].reshape(shape)
        
        self.ax_schematic.plot_surface(X_hemi_lab, Y_hemi_lab, Z_hemi_lab, facecolors=colors, alpha=0.7, shade=False)

        # 4. Draw the Sample Square Plate & Axes
        s_range = np.linspace(-1, 1, 2)
        X_s, Z_s = np.meshgrid(s_range, s_range)
        Y_s = np.zeros_like(X_s)
        s_coords = R_total @ np.vstack([X_s.flatten(), Y_s.flatten(), Z_s.flatten()])
        self.ax_schematic.plot_surface(s_coords[0].reshape(2,2), s_coords[1].reshape(2,2), s_coords[2].reshape(2,2), color='darkgray', alpha=0.9, edgecolor='k')

        axes_lab = R_total @ np.array([[1.5, 0, 0], [0, 1.5, 0], [0, 0, 1.5]]).T
        self.ax_schematic.quiver(0,0,0, axes_lab[0,0], axes_lab[1,0], axes_lab[2,0], color='red', linewidth=2, label='Sample X')
        self.ax_schematic.quiver(0,0,0, axes_lab[0,1], axes_lab[1,1], axes_lab[2,1], color='lime', linewidth=2, label='Sample Y (Normal)')
        self.ax_schematic.quiver(0,0,0, axes_lab[0,2], axes_lab[1,2], axes_lab[2,2], color='blue', linewidth=2, label='Sample Z')

        # 5. Draw Photon Beam (Wavy or Spiral)
        beam_dir = np.array([-np.sin(inc_rad), -np.cos(inc_rad), 0.0]) 
        start_pt = -4 * beam_dir
        t_beam = np.linspace(0, 4, 150)
        
        if is_circular:
            # Draw a 3D Helix/Corkscrew for Circular Polarization
            freq = 6 * np.pi
            radius = 0.3
            sign = 1 if "Right" in pol_mode else -1
            perp1 = np.array([0, 0, 1])
            perp2 = np.cross(beam_dir, perp1)
            wave_pts = start_pt[:, None] + beam_dir[:, None] * t_beam + radius * (np.cos(freq * t_beam) * perp1[:, None] + sign * np.sin(freq * t_beam) * perp2[:, None])
            self.ax_schematic.plot(wave_pts[0], wave_pts[1], wave_pts[2], color='magenta', linewidth=2, label='Circular Light')
        else:
            # Draw a planar sine wave for Linear Polarization
            freq = 5 * np.pi
            amplitude = 0.3
            # Extract real part of eps_lab to get the physical oscillation plane
            osc_dir = np.real(eps_lab)
            wave_pts = start_pt[:, None] + beam_dir[:, None] * t_beam + osc_dir[:, None] * amplitude * np.sin(freq * t_beam)
            self.ax_schematic.plot(wave_pts[0], wave_pts[1], wave_pts[2], color='gold', linewidth=2, label='Linear Light')
            self.ax_schematic.quiver(start_pt[0], start_pt[1], start_pt[2], np.real(eps_lab)[0], np.real(eps_lab)[1], np.real(eps_lab)[2], length=1.5, color='magenta', linewidth=2, label='E-Field')
        
        # 6. Draw Analyzer Acceptance Window (Slit + Deflector)
        win_w = np.radians(15.0)  # Standard acceptance FOV 
        slit_h = np.radians(slit_width / 10.0) 
        defl_rad, slit_rot = np.radians(deflector_angle), np.radians(slit_angle)
        
        w_theta = np.linspace(-slit_h/2, slit_h/2, 5) + defl_rad
        w_phi = np.linspace(-win_w/2, win_w/2, 20)
        W_THETA, W_PHI = np.meshgrid(w_theta, w_phi)
        
        # Apply slit rotation
        w_theta_rot = W_THETA * np.cos(slit_rot) - W_PHI * np.sin(slit_rot)
        w_phi_rot = W_THETA * np.sin(slit_rot) + W_PHI * np.cos(slit_rot)
        
        # Project onto the top of the lab-frame hemisphere
        X_win = 1.25 * np.sin(np.pi/2 - w_theta_rot) * np.cos(np.pi/2 + w_phi_rot)
        Z_win = 1.25 * np.sin(np.pi/2 - w_theta_rot) * np.sin(np.pi/2 + w_phi_rot)
        Y_win = 1.25 * np.cos(np.pi/2 - w_theta_rot)
        win_coords = R_total @ np.vstack([X_win.flatten(), Y_win.flatten(), Z_win.flatten()])
        
        self.ax_schematic.plot_surface(win_coords[0].reshape(W_THETA.shape), 
                                       win_coords[1].reshape(W_THETA.shape), 
                                       win_coords[2].reshape(W_THETA.shape), 
                                       color='cyan', alpha=0.8, edgecolor='blue', label='Analyzer Slit')

        # Draw Lab Z Pole
        self.ax_schematic.plot([0, 0], [0, 0], [-3, 3], color='black', linestyle='-.', linewidth=1, label='Lab Z (Theta)')

        # Formatting
        self.ax_schematic.set_xlim([-3, 3]); self.ax_schematic.set_ylim([-3, 3]); self.ax_schematic.set_zlim([-3, 3])
        self.ax_schematic.set_xlabel('Lab X (Side)'); self.ax_schematic.set_ylabel('Lab Y (Forward)'); self.ax_schematic.set_zlabel('Lab Z (Vertical)')
        self.ax_schematic.set_title("Experimental Geometry Schematic", pad=0)
        self.ax_schematic.legend(loc='center left', fontsize=7, bbox_to_anchor=(1.05, 0.5))
        self.ax_schematic.xaxis.pane.fill = False; self.ax_schematic.yaxis.pane.fill = False; self.ax_schematic.zaxis.pane.fill = False
        
        # Explicitly lock margins to bypass Apple Silicon auto-layout segfault
        self.schematic_figure.subplots_adjust(left=0.0, right=0.8, top=1.0, bottom=0.0) 
        self.schematic_canvas.draw()

    def trigger_simulation(self):
        """Validates workspace data, outputs ME info to terminal, and triggers computation."""
        target_crystal = self.ws_combo.currentText()
        model_choice = self.engine_dropdown.currentData()
        me_mode = self.matrix_element_combo.currentText()
        deflector_angle = self.deflector_angle_spin.value()
        slit_size = self.slit_size_spin.value()

        # --- TERMINAL INFORMATIVE OUTPUT ---
        print("\n" + "="*50)
        print("🚀 INITIATING ARPES MATRIX ELEMENT SIMULATION")
        print("="*50)
        print(f"Target Structure : {target_crystal}")
        print(f"Selected Physics : {model_choice} ({me_mode})")
        print(f"Polarization     : {self.polarization_combo.currentText()}")
        print(f"Incidence Angle  : {self.incidence_angle_spin.value()}\u00B0")
        
        if me_mode == "Full Matrix Elements":
            print(">> Calculating full dipole transition matrix: M_fi = <\u03C8_f | A\u00B7p | \u03C8_i>")
            print(">> Note: This includes spatial phase factors and orbital character projections.")
        elif me_mode == "Polarization Dipole Only":
            print(">> Calculating simplified geometry-only transitions.")
            print(">> Ignoring deep orbital overlaps; applying A\u00B7r geometric constraints only.")
        else:
            print(">> Matrix Elements DISABLED. Calculating bare spectral function A(k, \u03C9).")
            print(">> Intensity directly reflects the imaginary part of the Green's function.")
        print("="*50 + "\n")

        # --- PREPARE EXPERIMENT PARAMETERS FIRST (Fixes UnboundLocalError) ---
        e_min, e_max, e_steps = self.spin_e_min.value(), self.spin_e_max.value(), self.spin_e_steps.value()
        kx_min, kx_max, kx_steps = self.spin_kx_min.value(), self.spin_kx_max.value(), self.spin_kx_steps.value()
        ky_min, ky_max, ky_steps = self.spin_ky_min.value(), self.spin_ky_max.value(), self.spin_ky_steps.value()

        # --- CHECK WORKSPACE FOR PRE-CALCULATED BANDS ---
        band_data = global_workspace.pull_band_structure(target_crystal)

        if not band_data:
            print(">> WARNING: No pre-calculated band structure found. Plotting pure geometry matrix element.")
            self.ax.clear()
            
            # Generate a 2D grid for purely geometric matrix elements
            kx = np.linspace(kx_min, kx_max, kx_steps)
            ky = np.linspace(ky_min, ky_max, ky_steps)
            KX, KY = np.meshgrid(kx, ky)
            
            # Re-calculate local polarization for the 2D plane projection
            inc_rad = np.radians(self.incidence_angle_spin.value())
            pol_mode = self.polarization_combo.currentText()
            lin_ang = np.radians(self.lin_pol_angle_spin.value())
            
            if "Horizontal" in pol_mode: eps_lab = np.array([np.cos(inc_rad), -np.sin(inc_rad), 0.0])
            elif "Vertical" in pol_mode: eps_lab = np.array([0.0, 0.0, 1.0])
            elif "Arbitrary" in pol_mode: eps_lab = np.cos(lin_ang)*np.array([np.cos(inc_rad), -np.sin(inc_rad), 0.0]) + np.sin(lin_ang)*np.array([0.0, 0.0, 1.0])
            else: 
                sign = 1 if "Right" in pol_mode else -1
                eps_lab = (np.array([np.cos(inc_rad), -np.sin(inc_rad), 0.0]) + sign*1j*np.array([0.0, 0.0, 1.0])) / np.sqrt(2)
            
            t_rad, a_rad, tilt_rad = np.radians(self.manip_theta_spin.value()), np.radians(self.manip_azi_spin.value()), np.radians(self.manip_tilt_spin.value())
            R_z = np.array([[np.cos(t_rad), -np.sin(t_rad), 0], [np.sin(t_rad), np.cos(t_rad), 0], [0, 0, 1]])
            R_y = np.array([[np.cos(a_rad), 0, np.sin(a_rad)], [0, 1, 0], [-np.sin(a_rad), 0, np.cos(a_rad)]])
            R_x = np.array([[1, 0, 0], [0, np.cos(tilt_rad), -np.sin(tilt_rad)], [0, np.sin(tilt_rad), np.cos(tilt_rad)]])
            R_total = R_z @ R_y @ R_x
            R_inv = np.linalg.inv(R_total)
            eps_sample = R_inv @ eps_lab
            
            # Approx KZ from inner potential (assuming E ~ Fermi Level = 0 for pure geometry)
            V0 = self.inner_potential_spin.value()
            hv = self.photon_energy_spin.value()
            W = self.work_function_spin.value()
            E_kin = hv - W 
            k_norm2 = 0.262 * (E_kin + V0)
            KZ = np.sqrt(np.maximum(k_norm2 - KX**2 - KY**2, 0.0))
            
            # Normalize k-vector array
            k_mag = np.sqrt(KX**2 + KY**2 + KZ**2)
            k_mag[k_mag == 0] = 1e-10
            kx_norm, ky_norm, kz_norm = KX/k_mag, KY/k_mag, KZ/k_mag
            
            # Dipole interaction |A.k|^2
            me = np.abs(eps_sample[0]*kx_norm + eps_sample[1]*ky_norm + eps_sample[2]*kz_norm)**2
            
            self.ax.pcolormesh(kx, ky, me, shading='auto', cmap='afmhot')
            self.ax.set_aspect('equal')
            self.ax.set_title("Pure Geometric Matrix Element |A·k|²")
            self.ax.set_xlabel(r"$k_x$ ($\mathrm{\AA}^{-1}$)")
            self.ax.set_ylabel(r"$k_y$ ($\mathrm{\AA}^{-1}$)")
            self.figure.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)
            self.canvas.draw()
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
        
        try:
            # Force UI update to show calculation state
            self.ax.clear()
            self.ax.set_title("Calculating Matrix Elements... Please wait.")
            self.canvas.draw()
            QApplication.processEvents() 
            
            # Route to the Core Physics Engine 
            # We now pass the loaded band_data dict instead of invoking a tight-binding class locally
            results = self.engine_router.run_simulation(
                model_choice=model_choice, 
                crystal_data=band_data, 
                experiment_kwargs=experiment_kwargs
            )
            
            # Cache the 3D data block and axes for the slider
            self.sim_intensity = results['intensity_broadened']
            self.sim_E_axis = np.linspace(e_min, e_max, e_steps)
            self.sim_kx = np.linspace(kx_min, kx_max, kx_steps)
            self.sim_ky = np.linspace(ky_min, ky_max, ky_steps)
            
            # Configure the slider to match the number of energy steps
            self.energy_slider.setRange(0, e_steps - 1)
            
            # Default to the Fermi level (or closest to 0.0 eV)
            fermi_idx = np.abs(self.sim_E_axis).argmin()
            self.energy_slider.setEnabled(True)
            self.energy_slider.setValue(fermi_idx)
            
            # Force a plot update and draw the schematic since a successful run happened
            self.update_plot_slice(fermi_idx)
            
        except Exception as e:
            QMessageBox.critical(self, "Simulation Error", f"An error occurred in the physics router:\n{str(e)}")
            self.ax.set_title("Simulation Failed")
            self.canvas.draw()

    def update_plot_slice(self, index):
        """Slices the 3D intensity block at the chosen energy index and redraws the heatmap."""
        if not hasattr(self, 'sim_intensity'):
            return
        
        E_val = self.sim_E_axis[index]
        self.energy_label.setText(f"Binding Energy: {E_val:.3f} eV")
        
        # Extract the 2D slice and transpose for pcolormesh (Y, X format)
        slice_2d = self.sim_intensity[:, :, index].T 
        
        # --- NEW: Apply Contrast and Gamma ---
        slice_max = np.max(slice_2d) if np.max(slice_2d) > 0 else 1.0
        norm_slice = slice_2d / slice_max
        gamma_corrected = np.power(norm_slice, self.gamma_spin.value())
        
        self.ax.clear()
        self.ax.pcolormesh(self.sim_kx, self.sim_ky, gamma_corrected, shading='auto', cmap='afmhot', 
                           vmin=self.vmin_spin.value(), vmax=self.vmax_spin.value())
        self.ax.set_aspect('equal')
        self.ax.set_title(f"Constant Energy Contour: {E_val:.2f} eV")
        self.ax.set_xlabel(r"$k_x$ ($\mathrm{\AA}^{-1}$)")
        self.ax.set_ylabel(r"$k_y$ ($\mathrm{\AA}^{-1}$)")
        
        # Explicitly lock margins to bypass Apple Silicon auto-layout segfault
        self.figure.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)
        self.canvas.draw()
    
    def refresh_workspace(self):
        """Pulls the list of pre-calculated band structures from the central workspace."""
        self.ws_combo.clear()
        
        # We only want to list band structures ready for ARPES simulation
        bands = global_workspace.list_band_structures()
        
        if bands:
            self.ws_combo.addItems(bands)
        else:
            self.ws_combo.addItem("No band structures available")

    def closeEvent(self, event):
        self._is_closing = True
        event.accept()

# Standalone runner for independent testing
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ARPESSuite()
    window.resize(1000, 700)
    window.show()
    sys.exit(app.exec())