import numpy as np
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                               QComboBox, QPushButton, QDoubleSpinBox, 
                               QFormLayout, QGroupBox, QMessageBox, QSlider, 
                               QSpinBox, QScrollArea, QApplication, QInputDialog, QSplitter,
                               QCheckBox, QTextEdit)
from PySide6.QtCore import Qt, QThread, Signal

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from tensorspec.core.arpes_engine import ARPESEngineRouter
from tensorspec.core.workspace import global_workspace
from tensorspec.core.data_models import TensorData

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
        # Use QSplitter instead of QHBoxLayout for resizable panels
        main_sim_layout = QVBoxLayout(self)
        self.main_splitter = QSplitter(Qt.Horizontal)
        main_sim_layout.addWidget(self.main_splitter)
        
        # --- LEFT PANEL: CONTROLS ---
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)

        # 1. Engine Selector
        engine_group = QGroupBox("1. Simulation Engine Selector")
        engine_layout = QFormLayout(engine_group)
        self.engine_dropdown = QComboBox()
        self.engine_dropdown.addItem("Option B1: One-Step Model (Chinook TB)", "B1")
        self.engine_dropdown.addItem("Option B2: Bare Spectral Function (ME Off)", "B2")
        self.engine_dropdown.addItem("Option B3: Full Multiple Scattering (SPR-KKR)", "B3")
        self.engine_dropdown.addItem("Option A: Phenomenological Three-Step Model", "A")
        
        self.target_dropdown = QComboBox()
        self.target_dropdown.addItem("💻 Local Machine (Fast TB / Preview)", "local")
        self.populate_clusters()
        
        engine_layout.addRow(QLabel("Physics Model:"), self.engine_dropdown)
        engine_layout.addRow(QLabel("Compute Target:"), self.target_dropdown)
        control_layout.addWidget(engine_group)

        # Remote Chinook/Grizzly debug toggles (shown for B1/B2/A + Remote)
        self.remote_tb_opts = QGroupBox("1b. Remote TB backend (Chinook ↔ Grizzly debug)")
        remote_tb_form = QFormLayout(self.remote_tb_opts)
        self.combo_remote_me = QComboBox()
        self.combo_remote_me.addItem("Chinook (CPU reference)", "chinook")
        self.combo_remote_me.addItem("GrizzlyME", "grizzly")
        self.combo_remote_me.addItem("Auto (Grizzly if installed)", "auto")
        self.combo_remote_me.setCurrentIndex(0)  # default chinook while debugging
        self.combo_remote_device = QComboBox()
        self.combo_remote_device.addItem("CUDA", "cuda")
        self.combo_remote_device.addItem("CPU", "cpu")
        self.combo_remote_layout = QComboBox()
        self.combo_remote_layout.addItem("θ-slices (safe; large TB)", "slices")
        self.combo_remote_layout.addItem("Full cube (1 GPU job; may OOM)", "full")
        self.combo_remote_layout.addItem("Auto (full if CUDA Grizzly)", "auto")
        self.combo_remote_layout.setCurrentIndex(0)
        self.remote_tb_cores_spin = QSpinBox()
        self.remote_tb_cores_spin.setRange(1, 128)
        self.remote_tb_cores_spin.setValue(40)
        self.remote_tb_cores_spin.setPrefix("Cores: ")
        remote_tb_form.addRow("ME engine:", self.combo_remote_me)
        remote_tb_form.addRow("Grizzly device:", self.combo_remote_device)
        remote_tb_form.addRow("Layout:", self.combo_remote_layout)
        remote_tb_form.addRow("CPU workers:", self.remote_tb_cores_spin)
        self.combo_remote_me.setToolTip(
            "Chinook = reference CPU path. GrizzlyME = GPU/CPU accelerated ME. "
            "Toggle to A/B debug intensity and speed."
        )
        self.combo_remote_layout.setToolTip(
            "Full cube feeds one GPU job (fast when it fits VRAM). "
            "Large Wannier TB often OOMs on full — use θ-slices then."
        )
        control_layout.addWidget(self.remote_tb_opts)
        self.remote_tb_opts.hide()

        # Workspace & SOC Settings
        self.ws_group = QGroupBox("0. Crystal Structure & Bands (Workspace)")
        ws_layout = QVBoxLayout(self.ws_group)
        ws_row = QHBoxLayout()
        self.ws_combo = QComboBox()
        self.btn_ws_refresh = QPushButton("🔄 Refresh")
        self.btn_ws_refresh.clicked.connect(self.refresh_workspace)
        ws_row.addWidget(self.ws_combo)
        ws_row.addWidget(self.btn_ws_refresh)
        ws_layout.addLayout(ws_row)
        self.lbl_band_energy_meta = QLabel("onsite / EF: (select a pushed band structure)")
        self.lbl_band_energy_meta.setWordWrap(True)
        self.lbl_band_energy_meta.setStyleSheet("color: #9ad; font-size: 11px;")
        ws_layout.addWidget(self.lbl_band_energy_meta)
        self.ws_combo.currentIndexChanged.connect(self._update_band_energy_meta_label)
        control_layout.addWidget(self.ws_group)
        
        # --- Remote Vaults for SPRKKR ---
        self.vault_group = QGroupBox("0. SPRKKR Remote Vault")
        vault_layout = QHBoxLayout(self.vault_group)
        self.vault_combo = QComboBox()
        self.btn_vault_refresh = QPushButton("🔄 Refresh")
        self.btn_vault_refresh.clicked.connect(self.refresh_workspace)
        self.btn_vault_delete = QPushButton("🗑️ Delete")
        self.btn_vault_delete.clicked.connect(self.delete_vault)
        
        self.remote_cores_spin = QSpinBox()
        self.remote_cores_spin.setRange(1, 128)
        self.remote_cores_spin.setValue(40)
        self.remote_cores_spin.setPrefix("Cores: ")
        
        vault_layout.addWidget(self.vault_combo)
        vault_layout.addWidget(self.remote_cores_spin)
        vault_layout.addWidget(self.btn_vault_refresh)
        vault_layout.addWidget(self.btn_vault_delete)
        control_layout.addWidget(self.vault_group)
        self.vault_group.hide()
        
        self.engine_dropdown.currentIndexChanged.connect(self.on_engine_changed)
        self.target_dropdown.currentIndexChanged.connect(self.on_target_changed)
        
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

        # --- ADD THIS INSIDE _setup_ui() (e.g., right before beam_layout.addRow("Manipulator Θ...")) ---
        hkl_layout = QHBoxLayout()
        self.spin_h = QSpinBox(); self.spin_h.setRange(-10, 10); self.spin_h.setValue(0)
        self.spin_k = QSpinBox(); self.spin_k.setRange(-10, 10); self.spin_k.setValue(0)
        self.spin_l = QSpinBox(); self.spin_l.setRange(-10, 10); self.spin_l.setValue(1)
        
        hkl_layout.addWidget(self.spin_h)
        hkl_layout.addWidget(self.spin_k)
        hkl_layout.addWidget(self.spin_l)
        
        beam_layout.addRow("Cleavage Plane [h k l]:", hkl_layout)
        # ---------------------------------------------------------------------------------------------
        beam_layout.addRow("Beam Incidence (Lab):", self.incidence_angle_spin)
        beam_layout.addRow("Polarization:", self.polarization_combo)
        beam_layout.addRow("Pol. Angle (Arbitrary):", self.lin_pol_angle_spin)
        beam_layout.addRow("Intensity Mode:", self.matrix_element_combo)
        control_layout.addWidget(beam_group)

        # --- SARPES CONFIGURATION ---
        self.group_sarpes = QGroupBox("Spin-Resolved ARPES (SARPES)")
        sarpes_layout = QFormLayout()
        
        self.chk_enable_sarpes = QCheckBox("Enable SARPES Filter")
        self.chk_enable_sarpes.setChecked(False)
        
        self.combo_spin_axis = QComboBox()
        self.combo_spin_axis.addItems(["Sz (Z-Axis)", "Sx (X-Axis)", "Sy (Y-Axis)"])
        
        self.combo_spin_comp = QComboBox()
        self.combo_spin_comp.addItems(["Spin Up (+)", "Spin Down (-)"])
        
        sarpes_layout.addRow(self.chk_enable_sarpes)
        sarpes_layout.addRow("Projection Axis:", self.combo_spin_axis)
        sarpes_layout.addRow("Spin Filter:", self.combo_spin_comp)
        self.group_sarpes.setLayout(sarpes_layout)
        
        control_layout.addWidget(self.group_sarpes)


        
        
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
            vmin = QDoubleSpinBox(); vmin.setRange(-90.0, 90.0); vmin.setValue(min_val); vmin.setSingleStep(0.1)
            vmax = QDoubleSpinBox(); vmax.setRange(-90.0, 90.0); vmax.setValue(max_val); vmax.setSingleStep(0.1)
            vpts = QSpinBox(); vpts.setRange(10, 1000); vpts.setValue(40); vpts.setSingleStep(10)
            row.addWidget(QLabel("Min:")); row.addWidget(vmin)
            row.addWidget(QLabel("Max:")); row.addWidget(vmax)
            row.addWidget(QLabel("Pts:")); row.addWidget(vpts)
            return row, vmin, vmax, vpts

        # Relabeled strictly to Angular axes (Theta and Phi)
        row_kx, self.spin_kx_min, self.spin_kx_max, self.spin_kx_steps = create_range_row("Θ (Slit) [°]:", -15.0, 15.0, 100)
        row_ky, self.spin_ky_min, self.spin_ky_max, self.spin_ky_steps = create_range_row("Φ (Deflect) [°]:", -15.0, 15.0, 100)
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
        
        control_layout.addStretch()
        
        scroll_area = QScrollArea()
        scroll_area.setWidget(control_panel)
        scroll_area.setWidgetResizable(True)
        scroll_area.setMinimumWidth(350) 
        self.main_splitter.addWidget(scroll_area)

        # --- MIDDLE PANEL: SCHEMATIC VIEWER ---
        schematic_panel = QWidget()
        schematic_layout = QVBoxLayout(schematic_panel)
        self.schematic_figure = Figure(figsize=(4, 5))
        self.schematic_canvas = FigureCanvas(self.schematic_figure)
        
        self.ax_schematic = self.schematic_figure.add_subplot(111, projection='3d')
        schematic_layout.addWidget(self.schematic_canvas)
        self.main_splitter.addWidget(schematic_panel)

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
        self.energy_label = QLabel("Binding Energy (eV):")
        
        # New QDoubleSpinBox for keyboard input
        self.energy_spinbox = QDoubleSpinBox()
        self.energy_spinbox.setDecimals(3)
        self.energy_spinbox.setSingleStep(0.01)
        self.energy_spinbox.setEnabled(False)
        self.energy_spinbox.setKeyboardTracking(False) # Only update when Enter is pressed or focus is lost
        
        self.energy_slider = QSlider(Qt.Horizontal)
        self.energy_slider.setEnabled(False)
        
        # Connect both widgets to sync and update the plot
        self.energy_slider.valueChanged.connect(self._sync_slider_to_spinbox)
        self.energy_spinbox.valueChanged.connect(self._sync_spinbox_to_slider)
        
        slider_layout.addWidget(self.energy_label)
        slider_layout.addWidget(self.energy_spinbox)
        slider_layout.addWidget(self.energy_slider)
        plot_layout.addLayout(slider_layout)
        
        contrast_layout = QHBoxLayout()
        self.vmin_spin = QDoubleSpinBox(); self.vmin_spin.setRange(0.0, 1.0); self.vmin_spin.setValue(0.0); self.vmin_spin.setSingleStep(0.05)
        self.vmax_spin = QDoubleSpinBox(); self.vmax_spin.setRange(0.0, 1.0); self.vmax_spin.setValue(1.0); self.vmax_spin.setSingleStep(0.05)
        self.gamma_spin = QDoubleSpinBox(); self.gamma_spin.setRange(0.1, 5.0); self.gamma_spin.setValue(1.0); self.gamma_spin.setSingleStep(0.1)
        
        # --- NEW: BZ OVERLAY TOGGLE ---
        self.chk_overlay_bz = QCheckBox("Overlay Surface BZ")
        self.chk_overlay_bz.setStyleSheet("font-weight: bold; color: #0F6A8B;")
        self.chk_overlay_bz.stateChanged.connect(lambda: self.update_plot_slice())
        # ------------------------------

        contrast_layout.addWidget(QLabel("Min:"))
        contrast_layout.addWidget(self.vmin_spin)
        contrast_layout.addWidget(QLabel("Max:"))
        contrast_layout.addWidget(self.vmax_spin)
        contrast_layout.addWidget(QLabel("γ (Gamma):"))
        contrast_layout.addWidget(self.gamma_spin)
        contrast_layout.addWidget(self.chk_overlay_bz) # <-- Added here
        plot_layout.addLayout(contrast_layout)

        
        
        self.vmin_spin.valueChanged.connect(lambda: self.update_plot_slice(self.energy_slider.value()))
        self.vmax_spin.valueChanged.connect(lambda: self.update_plot_slice(self.energy_slider.value()))
        self.gamma_spin.valueChanged.connect(lambda: self.update_plot_slice(self.energy_slider.value()))
        
        # --- Live Logs (Hidden by default, shown for SPRKKR) ---
        self.live_log_widget = QWidget()
        live_log_layout = QVBoxLayout(self.live_log_widget)
        live_log_layout.setContentsMargins(0, 5, 0, 0)
        
        self.btn_start_live = QPushButton("▶️ Start Live Monitor")
        self.btn_start_live.setStyleSheet("background-color: #d35400; color: white; font-weight: bold; padding: 4px;")
        self.btn_start_live.clicked.connect(self.toggle_embedded_monitor)
        
        self.btn_fetch_results = QPushButton("📥 Fetch Remote Results")
        self.btn_fetch_results.setStyleSheet("background-color: #27ae60; color: white; font-weight: bold; padding: 4px;")
        self.btn_fetch_results.clicked.connect(self.fetch_remote_results)
        self.btn_fetch_results.setEnabled(False)
        self.btn_fetch_results.setToolTip(
            "Pull finished cube from the selected remote cluster. Enabled whenever "
            "Compute Target is Remote (Chinook → chinook_gui_run; SPR-KKR → vault / sprkkr_gui_run)."
        )
        
        self.txt_live_logs = QTextEdit()
        self.txt_live_logs.setReadOnly(True)
        self.txt_live_logs.setFixedHeight(120)
        self.txt_live_logs.setStyleSheet("background-color: #0c0c0c; color: #00ff00; font-family: monospace; font-size: 10px;")
        
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.btn_start_live)
        btn_layout.addWidget(self.btn_fetch_results)
        
        live_log_layout.addLayout(btn_layout)
        live_log_layout.addWidget(self.txt_live_logs)
        plot_layout.addWidget(self.live_log_widget)
        self.live_log_widget.hide()
        
        self.live_monitor = None

        # After remote widgets exist: sync Fetch/Monitor vs Physics×Target.
        self._sync_remote_ui()
        
        self.main_splitter.addWidget(plot_panel)
        
        # Set initial relative sizes for the splitter panels (e.g., 30% left, 35% middle, 35% right)
        self.main_splitter.setSizes([350, 400, 400])
        
        self.update_schematic()

        # Run Button
        self.run_sim_btn = QPushButton("🚀 Run ARPES Simulation")
        self.run_sim_btn.setStyleSheet("font-weight: bold; padding: 10px; background-color: #2b5c8f; color: white;")
        self.run_sim_btn.clicked.connect(self.trigger_simulation)
        control_layout.addWidget(self.run_sim_btn)
        
        # --- NEW: Save & Push Buttons ---
        self.btn_push_workspace = QPushButton("📊 Push to Workspace")
        self.btn_push_workspace.setStyleSheet("font-weight: bold; padding: 10px; background-color: #E67E22; color: white;")
        self.btn_push_workspace.setEnabled(False)
        self.btn_push_workspace.clicked.connect(self.push_arpes_to_workspace)
        
        self.btn_save_disk = QPushButton("💾 Save to Disk (.npz)")
        self.btn_save_disk.setStyleSheet("font-weight: bold; padding: 10px; background-color: #D35400; color: white;")
        self.btn_save_disk.setEnabled(False)
        self.btn_save_disk.clicked.connect(self.save_arpes_to_disk)
        
        control_layout.addWidget(self.btn_push_workspace)
        control_layout.addWidget(self.btn_save_disk)

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
        model_choice = self.engine_dropdown.currentData()
        
        if model_choice == "B3":
            # --- SPRKKR Remote Execution ---
            import os
            vault_name = self.vault_combo.currentText()
            if not vault_name or vault_name == "No SPRKKR Vaults Found":
                QMessageBox.warning(self, "Error", "No valid SPRKKR Vault selected! Run an SCF job first.")
                return
                
            try:
                import json, paramiko
                cluster = self.get_selected_cluster()
                if not cluster:
                    QMessageBox.critical(self, "Error", "No remote cluster found in configuration!")
                    return
                
                if vault_name == "Temporary Scratch Run (sprkkr_gui_run)":
                    remote_dir = f"/mnt/data/{cluster['user']}/tensorspec_heavy/sprkkr_gui_run"
                    cluster_name = cluster.get('name', cluster['host'])
                else:
                    vault = global_workspace.get(vault_name)
                    remote_dir = vault.get('remote_path') if vault else f"/mnt/data/{cluster['user']}/tensorspec_heavy/sprkkr_gui_run"
                    cluster_name = vault.get('cluster_name', vault.get('cluster', cluster.get('name', cluster['host'])))

                ssh = self._ssh_connect(cluster)
                
                task_str = "ARPES"
                polar = self.polarization_combo.currentText()
                
                from tensorspec.core.dft.sprkkr_generator import SPRKKRInputGenerator
                gen = SPRKKRInputGenerator(None)
                
                out_dir = "scratch/sprkkr_gui_run"
                gen.write_arpes_input(
                    out_dir,
                    task=task_str,
                    ne=self.spin_e_steps.value(),
                    e_min=self.spin_e_min.value(),
                    e_max=self.spin_e_max.value(),
                    ephot=self.photon_energy_spin.value(),
                    temp=self.temperature_spin.value(),
                    workf=self.work_function_spin.value(),
                    polar=polar,
                    hkl=(self.spin_h.value(), self.spin_k.value(), self.spin_l.value())
                )
                
                sftp = ssh.open_sftp()
                sftp.put(f"{out_dir}/sys.inp", f"{remote_dir}/sys.inp")
                
                # Upload the parallel mapping python script
                local_runner = os.path.join(os.path.dirname(__file__), "..", "..", "core", "dft", "arpes_map_runner_template.py")
                sftp.put(local_runner, f"{remote_dir}/arpes_map_runner.py")
                sftp.close()
                
                kx_min, kx_max, kx_steps = self.spin_kx_min.value(), self.spin_kx_max.value(), self.spin_kx_steps.value()
                ky_min, ky_max, ky_steps = self.spin_ky_min.value(), self.spin_ky_max.value(), self.spin_ky_steps.value()
                hv = self.photon_energy_spin.value()
                workf = self.work_function_spin.value()
                cores = self.remote_cores_spin.value()
                
                bin_path = f"/mnt/data/{cluster['user']}/tensorspec_heavy/SPRKKR/bin/kkrspec9.7"
                run_args = f"--theta_min {kx_min} --theta_max {kx_max} --ntheta {kx_steps} --phi_min {ky_min} --phi_max {ky_max} --nphi {ky_steps} --bin {bin_path} --cores {cores}"

                # SARPES Args
                if self.chk_enable_sarpes.isChecked():
                    axis_map = {"Sz (Z-Axis)": "Z", "Sx (X-Axis)": "X", "Sy (Y-Axis)": "Y"}
                    comp_map = {"Spin Up (+)": 1, "Spin Down (-)": -1}
                    run_args += f" --sarpes True --spin_axis {axis_map[self.combo_spin_axis.currentText()]} --spin_comp {comp_map[self.combo_spin_comp.currentText()]}"

                
                is_slurm = cluster.get('mode', '').upper() == 'SLURM'
                if is_slurm:
                    sbatch_content = f"""#!/bin/bash
#SBATCH --job-name=sprkkr_arpes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cores}
#SBATCH --output=sys.out.full
#SBATCH --error=sys.out.full

cd {remote_dir}
export TMPDIR=/mnt/data/{cluster['user']}/tmp
mkdir -p $TMPDIR
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PATH="/home/{cluster['user']}/miniconda3/envs/qe/bin:$PATH"
export LD_LIBRARY_PATH="/home/{cluster['user']}/miniconda3/envs/qe/lib:$LD_LIBRARY_PATH"

/home/{cluster['user']}/TensorSpec/TensorSpec_env/bin/python -u arpes_map_runner.py {run_args}
"""
                    sftp = ssh.open_sftp()
                    with sftp.file(f"{remote_dir}/job.sbatch", "w") as f:
                        f.write(sbatch_content)
                    sftp.close()
                    
                    cmd = f"cd {remote_dir} && sbatch job.sbatch"
                else:
                    cmd = f"bash -c 'cd {remote_dir} && export TMPDIR=/mnt/data/{cluster['user']}/tmp && mkdir -p $TMPDIR && export OMP_NUM_THREADS=1 && export MKL_NUM_THREADS=1 && export OPENBLAS_NUM_THREADS=1 && export PATH=\"/home/{cluster['user']}/miniconda3/envs/qe/bin:$PATH\" && export LD_LIBRARY_PATH=\"/home/{cluster['user']}/miniconda3/envs/qe/lib:$LD_LIBRARY_PATH\" && nohup /home/{cluster['user']}/TensorSpec/TensorSpec_env/bin/python -u arpes_map_runner.py {run_args} > sys.out.full 2>&1 &'"
                
                ssh.exec_command(cmd)
                ssh.close()
                self.btn_fetch_results.setEnabled(True)
                msg_type = "via SLURM Queue" if is_slurm else "via Background Daemon"
                QMessageBox.information(self, "Success", f"SPRKKR ARPES submitted to {cluster.get('name', cluster['host'])} ({msg_type})!\n\nLogs: {remote_dir}/sys.out.full\n\nCheck 'Calculation Live Logs' tab to watch the calculation!")
                
                # Auto-start the live monitor so the user sees the output immediately
                self.toggle_embedded_monitor(force_start=True)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to run SPRKKR ARPES:\n{str(e)}")
            return
            
        # --- CHINOOK EXECUTION (LOCAL OR REMOTE CLUSTER) ---
        target_crystal = self.ws_combo.currentText()
        me_mode = self.matrix_element_combo.currentText()

        print("\n" + "="*50)
        print("🚀 INITIATING ARPES MATRIX ELEMENT SIMULATION")
        print("="*50)
        print(f"Target Structure : {target_crystal}")
        print(f"Selected Physics : {model_choice} ({me_mode})")
        print(f"Compute Target   : {self.target_dropdown.currentText()}")
        print(f"Polarization     : {self.polarization_combo.currentText()}")
        print(f"Incidence Angle  : {self.incidence_angle_spin.value()}\u00B0")
        print("="*50 + "\n")

        e_min, e_max, e_steps = self.spin_e_min.value(), self.spin_e_max.value(), self.spin_e_steps.value()
        kx_min, kx_max, kx_steps = self.spin_kx_min.value(), self.spin_kx_max.value(), self.spin_kx_steps.value()
        ky_min, ky_max, ky_steps = self.spin_ky_min.value(), self.spin_ky_max.value(), self.spin_ky_steps.value()
        # Same as local chinook_wrapper: min==max → 1 sample (avoid linspace(0,0,40) waste).
        if kx_min == kx_max:
            kx_steps = 1
        if ky_min == ky_max:
            ky_steps = 1
        if e_min == e_max:
            e_steps = 1

        band_data = global_workspace.pull_band_structure(target_crystal)

        if not band_data:
            print(">> WARNING: No pre-calculated band structure found.")
            return
        
        print("\n[LOG 2 - ARPES PANEL]")
        print(f"Dictionary keys pulled: {list(band_data.keys())}")
        print(">> SUCCESS: Band structure loaded. Routing to Matrix Element Engine...")
        
        if self.target_dropdown.currentData() != "local":
            # --- REMOTE CHINOOK DISPATCH ---
            try:
                import os, json, paramiko
                cluster = self.get_selected_cluster()
                if not cluster:
                    QMessageBox.critical(self, "Error", "No remote cluster found in configuration!")
                    return
                
                remote_dir = f"/mnt/data/{cluster['user']}/tensorspec_heavy/chinook_gui_run"
                ssh = self._ssh_connect(cluster)
                sftp = ssh.open_sftp()
                
                try:
                    sftp.mkdir(f"/mnt/data/{cluster['user']}/tensorspec_heavy")
                except:
                    pass
                try:
                    sftp.mkdir(remote_dir)
                except:
                    pass
                    
                # Upload runner + shared kmesh module
                local_template = "tensorspec/core/arpes/one_step/chinook_remote_runner_template.py"
                local_kmesh = "tensorspec/core/arpes/one_step/chinook_arpes_kmesh.py"
                local_schedule = "tensorspec/core/arpes/one_step/grizzly_cuda_schedule.py"
                sftp.put(local_template, f"{remote_dir}/chinook_remote_runner.py")
                sftp.put(local_kmesh, f"{remote_dir}/chinook_arpes_kmesh.py")
                sftp.put(local_schedule, f"{remote_dir}/grizzly_cuda_schedule.py")
                
                # Save and upload TB data
                os.makedirs("scratch/chinook_gui_run", exist_ok=True)
                tb_path_local = "scratch/chinook_gui_run/tb_data.npz"
                physics_path_local = "scratch/chinook_gui_run/arpes_physics.json"
                
                # Compress massive H_dict into binary arrays.
                # R vectors are Cartesian floats (Wannier/SK) — must stay float64.
                # (int32 truncation zeroed remote ARPES intensity at finite angle.)
                h_list = band_data.get('H_dict', {}).get('list', [])
                if len(h_list) > 0:
                    indices = np.array(
                        [[h[0], h[1], h[2], h[3], h[4]] for h in h_list],
                        dtype=np.float64,
                    )
                    values = np.array([h[5] for h in h_list], dtype=np.complex128)
                else:
                    indices = np.empty((0, 5), dtype=np.float64)
                    values = np.empty(0, dtype=np.complex128)
                

                # Extract basic basis info
                basis = band_data.get('basis', [])
                if isinstance(basis, dict) and 'bulk' in basis:
                    basis = basis['bulk']
                elif isinstance(basis, dict) and 'orbitals' in basis:
                    basis = basis['orbitals']
                    
                basis_list = []

                if isinstance(basis, (list, tuple, np.ndarray)):
                    for b in basis:
                        if hasattr(b, 'pos'):
                            basis_list.append({
                                'pos': b.pos, 
                                'label': getattr(b, 'label', '10'), 
                                'spin': getattr(b, 'spin', 1.0),
                                'Z': getattr(b, 'Z', 1)
                            })
                
                b_matrix = band_data.get("recip_matrix")
                if b_matrix is None and band_data.get("structure") is not None:
                    b_matrix = band_data["structure"].lattice.reciprocal_lattice.matrix
                elif b_matrix is None:
                    a_list = band_data.get("H_dict", {}).get("a")
                    if a_list is not None:
                        A = np.array(a_list, dtype=float)
                        b_matrix = 2 * np.pi * np.linalg.inv(A).T
                if b_matrix is None:
                    b_matrix = 2 * np.pi * np.eye(3)

                np.savez_compressed(
                    tb_path_local,
                    indices=indices,
                    values=values,
                    basis_list=basis_list,
                    a_mat=band_data.get('H_dict', {}).get('a', []),
                    b_matrix=np.asarray(b_matrix, dtype=float),
                    e_fermi=float(
                        band_data.get(
                            'arpes_e_fermi_shift',
                            band_data.get('fermi_energy', 0.0),
                        )
                        or 0.0
                    ),
                    onsite_e=float(band_data.get('onsite_e', 0.0) or 0.0),
                    fermi_energy_qe=float(band_data.get('fermi_energy', 0.0) or 0.0),
                    h_includes_onsite=bool(band_data.get('h_includes_onsite', False)),
                )
                print(
                    f"[ARPES remote] TB upload: hops={len(indices)} "
                    f"indices.dtype={indices.dtype} "
                    f"onsite_e={band_data.get('onsite_e', 'MISSING')} "
                    f"QE_fermi={band_data.get('fermi_energy', 'MISSING')} "
                    f"arpes_e_shift={band_data.get('arpes_e_fermi_shift', band_data.get('fermi_energy', 0.0))} "
                    f"R_sample={indices[0, 2:5] if len(indices) else 'n/a'}",
                    flush=True,
                )

                from tensorspec.core.arpes.one_step.chinook_arpes_kmesh import (
                    physics_from_experiment_kwargs,
                )

                remote_physics = physics_from_experiment_kwargs(
                    {
                        "photon_energy": self.photon_energy_spin.value(),
                        "work_function": self.work_function_spin.value(),
                        "inner_potential": self.inner_potential_spin.value(),
                        "temperature": self.temperature_spin.value(),
                        "incidence_angle": self.incidence_angle_spin.value(),
                        "polarization": self.polarization_combo.currentText(),
                        "lin_pol_angle": self.lin_pol_angle_spin.value(),
                        "matrix_element_mode": me_mode,
                        "manip_theta": self.manip_theta_spin.value(),
                        "manip_azimuth": self.manip_azi_spin.value(),
                        "manip_tilt": self.manip_tilt_spin.value(),
                        "hkl": (
                            self.spin_h.value(),
                            self.spin_k.value(),
                            self.spin_l.value(),
                        ),
                        "slit_angle": self.slit_angle_spin.value(),
                        "se_width": self.ui_se_spinbox.value(),
                        "res_E": self.ui_res_e_spinbox.value(),
                        "res_k": self.ui_res_k_spinbox.value(),
                    }
                )
                with open(physics_path_local, "w") as pf:
                    json.dump(
                        {**remote_physics, "hkl": list(remote_physics["hkl"])},
                        pf,
                        indent=2,
                    )
                    
                sftp.put(tb_path_local, f"{remote_dir}/tb_data.npz")
                sftp.put(physics_path_local, f"{remote_dir}/arpes_physics.json")
                sftp.close()
                
                cores = self.remote_tb_cores_spin.value()
                hv = self.photon_energy_spin.value()
                workf = self.work_function_spin.value()
                v0 = self.inner_potential_spin.value()
                temp = self.temperature_spin.value()
                polar = "P" if "Linear" in self.polarization_combo.currentText() else "S"
                e_fermi = float(
                    band_data.get(
                        'arpes_e_fermi_shift',
                        band_data.get('fermi_energy', 0.0),
                    )
                    or 0.0
                )
                me_engine = self.combo_remote_me.currentData()
                me_device = self.combo_remote_device.currentData()
                me_layout = self.combo_remote_layout.currentData()

                run_args = (
                    f"--tb_file tb_data.npz --theta_min {kx_min} --theta_max {kx_max} --ntheta {kx_steps} "
                    f"--phi_min {ky_min} --phi_max {ky_max} --nphi {ky_steps} "
                    f"--e_min {e_min} --e_max {e_max} --ne {e_steps} "
                    f"--hv {hv} --workf {workf} --v0 {v0} --temp {temp} --polar {polar} "
                    f"--cores {cores} --engine {me_engine} --device {me_device} "
                    f"--layout {me_layout} --e_fermi {e_fermi} --theta_chunk 0 --ngpus 0"
                )

                # SARPES Args (forces chinook fallback inside runner; GrizzlyME v0.1 is spinless)
                if self.chk_enable_sarpes.isChecked():
                    axis_map = {"Sz (Z-Axis)": "Z", "Sx (X-Axis)": "X", "Sy (Y-Axis)": "Y"}
                    comp_map = {"Spin Up (+)": 1, "Spin Down (-)": -1}
                    run_args += f" --sarpes True --spin_axis {axis_map[self.combo_spin_axis.currentText()]} --spin_comp {comp_map[self.combo_spin_comp.currentText()]}"

                
                is_slurm = cluster.get('mode', '').upper() == 'SLURM'
                if is_slurm:
                    sbatch_content = f"""#!/bin/bash
#SBATCH --job-name=chinook_arpes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cores}
#SBATCH --output=sys.out.full
#SBATCH --error=sys.out.full

cd {remote_dir}
export TMPDIR=/mnt/data/{cluster['user']}/tmp
mkdir -p $TMPDIR
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

/home/{cluster['user']}/TensorSpec/TensorSpec_env/bin/python -u chinook_remote_runner.py {run_args}
"""
                    sftp = ssh.open_sftp()
                    with sftp.file(f"{remote_dir}/job.sbatch", "w") as f:
                        f.write(sbatch_content)
                    sftp.close()
                    cmd = f"cd {remote_dir} && sbatch job.sbatch"
                else:
                    cmd = f"bash -c 'cd {remote_dir} && export TMPDIR=/mnt/data/{cluster['user']}/tmp && mkdir -p $TMPDIR && export OMP_NUM_THREADS=1 && export MKL_NUM_THREADS=1 && export OPENBLAS_NUM_THREADS=1 && nohup /home/{cluster['user']}/TensorSpec/TensorSpec_env/bin/python -u chinook_remote_runner.py {run_args} > sys.out.full 2>&1 &'"
                
                ssh.exec_command(cmd)
                ssh.close()
                
                self.btn_fetch_results.setEnabled(True)
                cluster_label = cluster.get('name', cluster['host'])
                msg_type = "via SLURM Queue" if is_slurm else "via Background Daemon"
                QMessageBox.information(self, "Success", f"CHINOOK ARPES map dispatched to {cluster_label} ({msg_type}) with {cores} cores!\n\nLogs: {remote_dir}/sys.out.full\n\nCheck 'Calculation Live Logs' tab to watch progress!")
                self.toggle_embedded_monitor(force_start=True)
                return
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Error",
                    f"Failed to dispatch CHINOOK to {cluster_label}:\n{str(e)}",
                )
                return
        
        # --- LOCAL EXECUTION ---
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
            'hkl': (self.spin_h.value(), self.spin_k.value(), self.spin_l.value()),

            # Send standard relative bounds (e.g., -2.0 to 0.5) to the engine
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
            self.btn_push_workspace.setEnabled(True)
            self.btn_save_disk.setEnabled(True)
            
            e_min, e_max, e_steps = self.spin_e_min.value(), self.spin_e_max.value(), self.spin_e_steps.value()
            kx_min, kx_max, kx_steps = self.spin_kx_min.value(), self.spin_kx_max.value(), self.spin_kx_steps.value()
            ky_min, ky_max, ky_steps = self.spin_ky_min.value(), self.spin_ky_max.value(), self.spin_ky_steps.value()
            if kx_min == kx_max:
                kx_steps = 1
            if ky_min == ky_max:
                ky_steps = 1
            if e_min == e_max:
                e_steps = 1

            self.sim_intensity = results['intensity_broadened']
            
            # Prefer axes from remote npz when present (truth for remote fetches).
            nx, ny, ne = self.sim_intensity.shape
            if results.get('energy') is not None:
                self.sim_E_axis = np.asarray(results['energy'], dtype=float)
            else:
                self.sim_E_axis = np.linspace(e_min, e_max, ne)
            if results.get('theta') is not None:
                self.sim_kx = np.asarray(results['theta'], dtype=float)
            else:
                self.sim_kx = np.linspace(kx_min, kx_max, nx)
            if results.get('phi') is not None:
                self.sim_ky = np.asarray(results['phi'], dtype=float)
            else:
                self.sim_ky = np.linspace(ky_min, ky_max, ny)

            # Remote/UI grids are emission angles (deg), not Å⁻¹.
            self.sim_axes_are_angles = True
            
            if (
                ny == 1
                or nx == 1
                or self._axis_degenerate(self.sim_ky)
                or self._axis_degenerate(self.sim_kx)
            ):
                # 2D Band Dispersion Map (Energy vs angle)
                self.energy_slider.setEnabled(False)
                self.energy_spinbox.setEnabled(False)
                self.update_plot_slice(index=None)
            else:
                # 3D Cube (Constant Energy Contours)
                self.energy_slider.setRange(0, ne - 1)
                
                # Setup Spinbox Range
                self.energy_spinbox.setRange(float(self.sim_E_axis[0]), float(self.sim_E_axis[-1]))
                
                fermi_idx = int(np.abs(self.sim_E_axis).argmin())
                self.energy_slider.setEnabled(True)
                self.energy_spinbox.setEnabled(True)
                
                # Block signals to prevent double-firing during initialization
                self.energy_slider.blockSignals(True)
                self.energy_spinbox.blockSignals(True)
                self.energy_slider.setValue(fermi_idx)
                self.energy_spinbox.setValue(self.sim_E_axis[fermi_idx])
                self.energy_slider.blockSignals(False)
                self.energy_spinbox.blockSignals(False)
                
                self.update_plot_slice(fermi_idx)
        else:
            QMessageBox.critical(self, "Simulation Error", f"An error occurred in the physics router:\n{message}")
            self.ax.set_title("Simulation Failed")
            self.canvas.draw()

    def _sync_slider_to_spinbox(self, index):
        """Updates the spinbox text when the slider moves."""
        if not hasattr(self, 'sim_E_axis'): return
        self.energy_spinbox.blockSignals(True)
        self.energy_spinbox.setValue(self.sim_E_axis[index])
        self.energy_spinbox.blockSignals(False)
        self.update_plot_slice(index)

    def _sync_spinbox_to_slider(self, val):
        """Finds the nearest array index when the user types a value in the spinbox."""
        if not hasattr(self, 'sim_E_axis'): return
        nearest_idx = (np.abs(self.sim_E_axis - val)).argmin()
        self.energy_slider.blockSignals(True)
        self.energy_slider.setValue(nearest_idx)
        self.energy_slider.blockSignals(False)
        self.update_plot_slice(nearest_idx)

    def update_plot_slice(self, index=None):
        if not hasattr(self, 'sim_intensity'):
            return
            
        nx, ny, ne = self.sim_intensity.shape
        self.ax.clear()

        use_angles = getattr(self, 'sim_axes_are_angles', True)
        x_lab = r"$\Theta$ (slit) [deg]" if use_angles else r"$k_x$ ($\mathrm{\AA}^{-1}$)"
        y_lab = r"$\Phi$ (deflect) [deg]" if use_angles else r"$k_y$ ($\mathrm{\AA}^{-1}$)"

        # Degenerate φ=0…0 (or θ) → band map, not a squeezed contour.
        deg_y = ny == 1 or self._axis_degenerate(self.sim_ky)
        deg_x = nx == 1 or self._axis_degenerate(self.sim_kx)

        # --- ROUTE 1: PLOT BAND DISPERSION (E vs angle) ---
        if deg_y or deg_x:
            self.energy_label.setText("Band Dispersion Map")
            
            if deg_y:
                slice_2d = self.sim_intensity[:, 0, :].T
                x_axis = self.sim_kx
                x_label = x_lab
            else:
                slice_2d = self.sim_intensity[0, :, :].T
                x_axis = self.sim_ky
                x_label = y_lab
                
            slice_max = np.max(slice_2d) if np.max(slice_2d) > 0 else 1.0
            norm_slice = slice_2d / slice_max
            gamma_corrected = np.power(norm_slice, self.gamma_spin.value())
            
            self.ax.pcolormesh(x_axis, self.sim_E_axis, gamma_corrected, shading='auto', cmap='afmhot', 
                               vmin=self.vmin_spin.value(), vmax=self.vmax_spin.value())
            
            self.ax.set_aspect('auto')
            self.ax.set_title("Simulated ARPES Band Dispersion")
            self.ax.set_xlabel(x_label)
            self.ax.set_ylabel(r"$E - E_F$ [eV]")

        # --- ROUTE 2: PLOT CONSTANT ENERGY CONTOUR ---
        else:
            if index is None: 
                index = self.energy_slider.value()
                
            E_val = self.sim_E_axis[index]
            
            slice_2d = self.sim_intensity[:, :, index].T
            
            slice_max = np.max(slice_2d) if np.max(slice_2d) > 0 else 1.0
            norm_slice = slice_2d / slice_max
            gamma_corrected = np.power(norm_slice, self.gamma_spin.value())
            
            self.ax.pcolormesh(self.sim_kx, self.sim_ky, gamma_corrected, shading='auto', cmap='afmhot', 
                               vmin=self.vmin_spin.value(), vmax=self.vmax_spin.value())

            span_x = float(np.ptp(self.sim_kx))
            span_y = float(np.ptp(self.sim_ky))
            if span_x > 1e-9 and span_y > 1e-9 and 0.05 < (span_y / span_x) < 20:
                self.ax.set_aspect('equal')
            else:
                self.ax.set_aspect('auto')
            self.ax.set_title(f"Constant Energy Contour: {E_val:.2f} eV")
            self.ax.set_xlabel(x_lab)
            self.ax.set_ylabel(y_lab)
            
            if getattr(self, 'chk_overlay_bz', None) and self.chk_overlay_bz.isChecked():
                self._draw_bz_overlay()
            
        self.figure.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)
        self.canvas.draw()
        
    def populate_clusters(self):
        import json, os
        config_file = os.path.expanduser('~/.tensorspec_clusters.json')
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    clusters = json.load(f)
                for c in clusters:
                    self.target_dropdown.addItem(f"🚀 Remote: {c.get('name', c.get('host'))}", c)
            except Exception as e:
                print(f"Error reading cluster config: {e}")

    def get_selected_cluster(self):
        selected_data = self.target_dropdown.currentData()
        if isinstance(selected_data, dict):
            return selected_data
            
        import json, os
        config_file = os.path.expanduser('~/.tensorspec_clusters.json')
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    clusters = json.load(f)
                if clusters:
                    return clusters[0]
            except:
                pass
        return None

    def _ssh_connect(self, cluster):
        """Paramiko connect with longer banner/kex timeouts (slow remote hosts)."""
        import paramiko
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        pwd = cluster.get('password', '') or None
        # banner_timeout / auth_timeout matter more than connect timeout for kex stalls
        ssh.connect(
            cluster['host'],
            port=cluster.get('port', 22),
            username=cluster['user'],
            password=pwd,
            timeout=30,
            banner_timeout=90,
            auth_timeout=60,
            allow_agent=True,
            look_for_keys=True,
        )
        return ssh

    @staticmethod
    def _axis_degenerate(axis) -> bool:
        axis = np.asarray(axis, dtype=float)
        return axis.size <= 1 or float(np.ptp(axis)) < 1e-9

    def toggle_embedded_monitor(self, force_start=False):
        if self.live_monitor and self.live_monitor.isRunning():
            if force_start:
                return
            # Stop it
            self.live_monitor.stop()
            self.live_monitor.wait()
            self.btn_start_live.setText("▶️ Start Live Monitor")
            self.btn_start_live.setStyleSheet("background-color: #d35400; color: white; font-weight: bold; padding: 4px;")
            return
            
        active_cluster = self.get_selected_cluster()
        if not active_cluster:
            QMessageBox.warning(self, "No Cluster", "No remote cluster configured in Cluster Connections!")
            return
        
        from tensorspec.gui.components.compute_panel import LiveMonitorThread
        self.live_monitor = LiveMonitorThread(active_cluster)
        self.live_monitor.data_ready.connect(self.update_embedded_logs)
        self.live_monitor.start()
        self.btn_start_live.setText("⏹️ Stop Live Monitor")
        self.btn_start_live.setStyleSheet("background-color: #c0392b; color: white; font-weight: bold; padding: 4px;")

    def update_embedded_logs(self, data):
        content = "=== REMOTE LIVE LOGS ===\n\n"
        content += data.get('text_info', '') + "\n"
        content += "----------------------------------------\n"
        content += data.get('full_log_tail', '')
        
        self.txt_live_logs.setPlainText(content)
        sb = self.txt_live_logs.verticalScrollBar()
        sb.setValue(sb.maximum())

    def get_simulation_metadata(self):
        """Helper to grab all current UI parameters for saving."""
        return {
            'crystal': self.ws_combo.currentText(),
            'engine': self.engine_dropdown.currentText(),
            'photon_energy': self.photon_energy_spin.value(),
            'work_function': self.work_function_spin.value(),
            'inner_potential': self.inner_potential_spin.value(),
            'temperature': self.temperature_spin.value(),
            'polarization': self.polarization_combo.currentText(),
            'hkl': [self.spin_h.value(), self.spin_k.value(), self.spin_l.value()]
        }



    def fetch_remote_results(self):
        """Fetch remote ARPES cube (Chinook or SPR-KKR)."""
        return self.fetch_sprkkr_results()

    def fetch_sprkkr_results(self):
        vault_name = self.vault_combo.currentText()
        model_choice = self.engine_dropdown.currentData()
        
        import json, os, paramiko
        config_file = os.path.expanduser('~/.tensorspec_clusters.json')
        with open(config_file, 'r') as f:
            clusters = json.load(f)
        cluster = self.get_selected_cluster()
        if not cluster: return
        
        # Chinook / bare / three-step remote → chinook_gui_run.
        # Only SPR-KKR (B3) uses SPRKKR vault paths.
        is_chinook_remote = (model_choice != "B3" and self.target_dropdown.currentData() != "local")
        
        if is_chinook_remote:
            remote_dir = f"/mnt/data/{cluster['user']}/tensorspec_heavy/chinook_gui_run"
            target_cube_name = "chinook_arpes_cube.npz"
        elif vault_name == "Temporary Scratch Run (sprkkr_gui_run)":
            remote_dir = f"/mnt/data/{cluster['user']}/tensorspec_heavy/sprkkr_gui_run"
            target_cube_name = "arpes_cube.npz"
        else:
            vault = global_workspace.get(vault_name)
            remote_dir = vault.get('remote_path') if vault else f"/mnt/data/{cluster['user']}/tensorspec_heavy/sprkkr_gui_run"
            target_cube_name = "arpes_cube.npz"

        try:
            ssh = self._ssh_connect(cluster)
            
            sftp = ssh.open_sftp()
            os.makedirs("scratch", exist_ok=True)
            local_path = f"scratch/{target_cube_name}"
            sftp.get(f"{remote_dir}/{target_cube_name}", local_path)
            sftp.close()
            ssh.close()
            
            # Load the file
            data = np.load(local_path, allow_pickle=True)
            if 'cube' in data:
                intensity_3d = data['cube']
                results = {
                    'intensity_broadened': intensity_3d,
                    'energy': data['energy'] if 'energy' in data else None,
                    'theta': data['theta'] if 'theta' in data else None,
                    'phi': data['phi'] if 'phi' in data else None,
                }
            else:
                intensity_3d = data['intensity']
                results = {
                    'intensity_broadened': intensity_3d,
                    'energy': data['e_axis'] if 'e_axis' in data else None,
                    'theta': data['kx'] if 'kx' in data else None,
                    'phi': data['ky'] if 'ky' in data else None,
                }
            
            self.on_simulation_finished(True, results, "Fetched successfully")
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to fetch results:\n{str(e)}\n\n"
                "If this is a Paramiko key-exchange timeout, retry Fetch "
                "(slow network / VPN). Connect timeout was increased to 90s banner.",
            )

    def push_arpes_to_workspace(self):
        if not hasattr(self, 'sim_intensity'): return
        
        name, ok = QInputDialog.getText(self, "Push to Workspace", "Enter dataset name (e.g., WTe2_75eV_CR):")
        if ok and name:
            try:
                # Transpose dimensions for the viewer (kx, ky, E) -> (E, kx, ky)
                tensor_value = np.transpose(self.sim_intensity, (2, 0, 1))
                
                sim_tensor = TensorData(
                    value=tensor_value,
                    axes=[self.sim_E_axis, self.sim_kx, self.sim_ky],
                    labels=["Energy", "Θ (Slit)", "Φ (Deflect)"],
                    units=["eV", "deg", "deg"],
                    data_type="Simulated ARPES Matrix Elements",
                    metadata=self.get_simulation_metadata()
                )
                
                global_workspace.push_spectroscopy_data(name, sim_tensor)
                QMessageBox.information(self, "Success", f"Data pushed to Workspace as '{name}'.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to push data:\n{e}")

    def save_arpes_to_disk(self):
        if not hasattr(self, 'sim_intensity'): return
        
        name, ok = QInputDialog.getText(self, "Save to Disk", "Enter file name (e.g., WTe2_75eV_CR):")
        if ok and name:
            try:
                global_workspace.save_simulated_arpes(
                    name, 
                    self.sim_intensity, 
                    self.sim_kx, 
                    self.sim_ky, 
                    self.sim_E_axis, 
                    self.get_simulation_metadata()
                )
                QMessageBox.information(self, "Success", f"Data safely saved to disk as '{name}.npz'.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save data:\n{e}")
    

    def _is_remote_target(self) -> bool:
        return self.target_dropdown.currentData() != "local"

    def _sync_remote_ui(self):
        """Show/enable live-monitor + fetch from Physics Model × Compute Target.

        Rules:
        - Local + (B1/B2/A): hide remote panel; fetch off.
        - Remote + (B1/B2/A): show panel; fetch on → chinook_gui_run cube.
        - B3 (SPR-KKR): always show panel + vault; fetch on (vault / sprkkr path).
        Fetch must NOT require a fresh submit this session (finished jobs on remote cluster).
        """
        model = self.engine_dropdown.currentData()
        remote = self._is_remote_target()

        if model == "B3":
            self.ws_group.hide()
            self.vault_group.show()
            self.remote_tb_opts.hide()
            self.matrix_element_combo.hide()
            self.live_log_widget.show()
            self.btn_fetch_results.setEnabled(True)
            self.btn_start_live.setEnabled(True)
        else:
            self.ws_group.show()
            self.vault_group.hide()
            self.matrix_element_combo.show()
            if remote:
                self.remote_tb_opts.show()
                self.live_log_widget.show()
                self.btn_fetch_results.setEnabled(True)
                self.btn_start_live.setEnabled(True)
            else:
                self.remote_tb_opts.hide()
                self.live_log_widget.hide()
                self.btn_fetch_results.setEnabled(False)
                self.btn_start_live.setEnabled(False)

    def on_target_changed(self):
        self._sync_remote_ui()

    def on_engine_changed(self):
        self._sync_remote_ui()


    def _update_band_energy_meta_label(self):
        name = self.ws_combo.currentText()
        if not name or name == "No band structures found":
            self.lbl_band_energy_meta.setText("onsite / EF: (select a pushed band structure)")
            return
        band = global_workspace.pull_band_structure(name)
        if not band:
            self.lbl_band_energy_meta.setText("onsite / EF: (not found in workspace)")
            return
        onsite = band.get('onsite_e', 'MISSING')
        ef = band.get('fermi_energy', band.get('e_fermi', 'MISSING'))
        shift = band.get('arpes_e_fermi_shift', ef)
        baked = band.get('h_includes_onsite', False)
        self.lbl_band_energy_meta.setText(
            f"From DFT push: onsite_e={onsite} eV | QE Fermi={ef} eV | "
            f"ARPES E-shift={shift} eV | onsite baked in H={baked}"
        )

    def refresh_workspace(self):
        bands = global_workspace.list_band_structures()
        self.ws_combo.blockSignals(True)
        self.ws_combo.clear()
        if not bands:
            self.ws_combo.addItem("No band structures found")
        else:
            self.ws_combo.addItems(bands)
        self.ws_combo.blockSignals(False)
        self._update_band_energy_meta_label()
            
        vaults = global_workspace.list_remote_runs(engine="SPRKKR")
        self.vault_combo.clear()
        
        # Always add the Scratch Run as the default!
        self.vault_combo.addItem("Temporary Scratch Run (sprkkr_gui_run)")
        
        if vaults:
            self.vault_combo.addItems(vaults)
            
    def delete_vault(self):
        vault_name = self.vault_combo.currentText()
        if not vault_name or "Temporary Scratch Run" in vault_name or vault_name == "No SPRKKR Vaults Found":
            QMessageBox.warning(self, "Invalid", "You cannot delete the temporary scratch run or an empty vault.")
            return
            
        reply = QMessageBox.question(
            self,
            'Delete Vault',
            f"Are you sure you want to delete '{vault_name}' from the Workspace?\n\n"
            "(This will also SSH into the remote cluster and delete the remote folder!)",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        
        if reply == QMessageBox.Yes:
            vault = global_workspace.get(vault_name)
            remote_dir = vault.get('remote_path')
            
            # SSH and delete
            try:
                import json, paramiko, os
                cluster = self.get_selected_cluster()
                if cluster:
                    ssh = paramiko.SSHClient()
                    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                    pwd = cluster.get('password', '') or None
                    ssh.connect(cluster['host'], port=cluster.get('port', 22), username=cluster['user'], password=pwd, timeout=10)
                    
                    cmd = f"rm -rf {remote_dir}"
                    ssh.exec_command(cmd)
                    ssh.close()
            except Exception as e:
                QMessageBox.warning(self, "SSH Warning", f"Removed from workspace, but failed to delete remote folder on cluster: {e}")
                
            global_workspace.remove(vault_name)
            self.refresh_workspace()
            QMessageBox.information(self, "Deleted", f"Vault '{vault_name}' has been deleted!")
    
    def _draw_bz_overlay(self):
        """Fetches the 3D Brillouin Zone, projects it to the hkl surface, and rotates it onto the (kx, ky) plot axes."""
        target_crystal = self.ws_combo.currentText()
        band_data = global_workspace.pull_band_structure(target_crystal)
        
        # Ensure the DFT suite provided the structure and reciprocal matrix
        if not band_data or 'structure' not in band_data or 'recip_matrix' not in band_data:
            return

        structure = band_data['structure']
        recip_matrix = band_data['recip_matrix']
        
        from tensorspec.core.crystallography import CrystalEngine
        import numpy as np
        
        # 1. Calculate the 3D Bulk Brillouin Zone vertices
        bz_data = CrystalEngine.calculate_brillouin_zone(structure)
        if not bz_data: return
        
        # 2. Recreate the exact hkl -> Sample Frame rotation matrix used by Chinook
        hkl = (self.spin_h.value(), self.spin_k.value(), self.spin_l.value())
        Z_surf, Y_surf = CrystalEngine.get_hkl_surface_frame(hkl, recip_matrix, azimuthal_ref=None)
        X_surf = np.cross(Y_surf, Z_surf)
        
        # R_bulk_to_hkl projects 3D Bulk vectors into the (kx, ky, kz) sample frame
        R_hkl_to_bulk = np.column_stack((X_surf, Y_surf, Z_surf))
        R_bulk_to_hkl = np.linalg.inv(R_hkl_to_bulk)
        
        # 3. Calculate the 2D projected boundary silhouette
        scaled_points = np.array(bz_data["points"])
        surf_data = CrystalEngine.calculate_surface_projection(scaled_points, structure, hkl[0], hkl[1], hkl[2])
        
        if surf_data:
            # The silhouette vertices are returned in the 3D Bulk Cartesian frame
            proj_bounds_bulk = np.array(surf_data["projected_bounds"])
            
            # Rotate the vertices into the ARPES (kx, ky) Frame
            bounds_sample = proj_bounds_bulk @ R_bulk_to_hkl.T
            
            kx_poly = bounds_sample[:, 0]
            ky_poly = bounds_sample[:, 1]
            
            # Close the polygon loop so the drawn line connects the last vertex back to the first
            kx_poly = np.append(kx_poly, kx_poly[0])
            ky_poly = np.append(ky_poly, ky_poly[0])
            
            # Overlay on the plot
            self.ax.plot(kx_poly, ky_poly, color='cyan', linestyle='-', linewidth=1.5, label="Projected Surface BZ")
            self.ax.legend(loc='upper right', fontsize=8, facecolor='black', edgecolor='white', labelcolor='white')