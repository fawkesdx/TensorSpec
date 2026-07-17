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
        self.photon_energy_spin = QDoubleSpinBox(); self.photon_energy_spin.setRange(5.0, 2000.0); self.photon_energy_spin.setValue(21.2); self.photon_energy_spin.setSuffix(" eV")
        self.work_function_spin = QDoubleSpinBox(); self.work_function_spin.setRange(0.0, 10.0); self.work_function_spin.setValue(4.5); self.work_function_spin.setSuffix(" eV")
        
        param_layout.addRow("Photon (hv):", self.photon_energy_spin)
        param_layout.addRow("Work Func (\u03A6):", self.work_function_spin)
        control_layout.addWidget(param_group)

        # 3. Beam & Analyzer Geometry
        beam_group = QGroupBox("3. Beam & Analyzer Geometry")
        beam_layout = QFormLayout(beam_group)
        self.incidence_angle_spin = QDoubleSpinBox(); self.incidence_angle_spin.setRange(0.0, 90.0); self.incidence_angle_spin.setValue(55.0); self.incidence_angle_spin.setSuffix(" \u00B0")
        
        self.polarization_combo = QComboBox()
        self.polarization_combo.addItems(["Linear Horizontal (p-pol)", "Linear Vertical (s-pol)", "Circular Right (CR)", "Circular Left (CL)"])

        self.matrix_element_combo = QComboBox()
        self.matrix_element_combo.addItems(["Full Matrix Elements", "Polarization Dipole Only", "Bare Spectral Function (ME Off)"])
        
        # New Analyzer controls
        self.slit_size_spin = QDoubleSpinBox(); self.slit_size_spin.setRange(0.1, 5.0); self.slit_size_spin.setValue(0.5); self.slit_size_spin.setSingleStep(0.1); self.slit_size_spin.setSuffix(" mm")
        self.deflector_angle_spin = QDoubleSpinBox(); self.deflector_angle_spin.setRange(-15.0, 15.0); self.deflector_angle_spin.setValue(0.0); self.deflector_angle_spin.setSuffix(" \u00B0")

        beam_layout.addRow("Incidence Angle:", self.incidence_angle_spin)
        beam_layout.addRow("Polarization:", self.polarization_combo)
        beam_layout.addRow("Intensity Mode:", self.matrix_element_combo)
        beam_layout.addRow("Analyzer Slit Size:", self.slit_size_spin)
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
            vpts = QSpinBox(); vpts.setRange(10, 500); vpts.setValue(40); vpts.setSingleStep(10)
            row.addWidget(QLabel("Min:")); row.addWidget(vmin)
            row.addWidget(QLabel("Max:")); row.addWidget(vmax)
            row.addWidget(QLabel("Pts:")); row.addWidget(vpts)
            return row, vmin, vmax, vpts

        row_kx, self.spin_kx_min, self.spin_kx_max, self.spin_kx_steps = create_range_row("kx:", -1.0, 1.0, 40)
        row_ky, self.spin_ky_min, self.spin_ky_max, self.spin_ky_steps = create_range_row("ky:", -1.0, 1.0, 40)
        row_e, self.spin_e_min, self.spin_e_max, self.spin_e_steps = create_range_row("E:", -2.0, 0.5, 40)

        domain_layout.addLayout(row_kx)
        domain_layout.addLayout(row_ky)
        domain_layout.addLayout(row_e)
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
        scroll_area.setFixedWidth(450)
        main_sim_layout.addWidget(scroll_area)

        # --- MIDDLE PANEL: SCHEMATIC VIEWER ---
        schematic_panel = QWidget()
        schematic_layout = QVBoxLayout(schematic_panel)
        self.schematic_figure = Figure(figsize=(4, 5))
        self.schematic_canvas = FigureCanvas(self.schematic_figure)
        self.ax_schematic = self.schematic_figure.add_subplot(111)
        self.ax_schematic.set_title("Experimental Geometry Schematic")
        schematic_layout.addWidget(self.schematic_canvas)
        main_sim_layout.addWidget(schematic_panel, stretch=1)

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
        
        main_sim_layout.addWidget(plot_panel, stretch=1)

    def draw_hemisphere_schematic(self, deflector_angle, slit_width):
        """Draws the emission hemisphere heatmap and highlights the measurement slit."""
        self.ax_schematic.clear()
        
        # Draw base hemisphere as a polar-like projection (theta, phi)
        theta = np.linspace(-90, 90, 100)
        phi = np.linspace(-90, 90, 100)
        T, P = np.meshgrid(theta, phi)
        
        # Mock heatmap data mimicking standard photoemission intensity
        intensity = np.cos(np.radians(T)) * np.cos(np.radians(P))
        
        self.ax_schematic.pcolormesh(T, P, intensity, cmap='Blues', shading='auto', alpha=0.5)
        
        # Add the deflector measurement rectangle
        # Slit acts along the phi axis (vertical), deflector shifts along theta (horizontal)
        rect_height = 180  # Full angular acceptance vertically
        rect_width = slit_width * 2  # Scaled for visual representation
        
        rect = patches.Rectangle(
            (deflector_angle - rect_width/2, -90), 
            rect_width, 
            rect_height, 
            linewidth=2, 
            edgecolor='red', 
            facecolor='none',
            linestyle='--'
        )
        self.ax_schematic.add_patch(rect)
        
        self.ax_schematic.set_xlim(-90, 90)
        self.ax_schematic.set_ylim(-90, 90)
        self.ax_schematic.set_title(f"Hemisphere (Deflector: {deflector_angle}\u00B0, Slit: {slit_width}mm)")
        self.ax_schematic.set_xlabel(r"Deflector Angle $\theta_{x}$ (\u00B0)")
        self.ax_schematic.set_ylabel(r"Analyzer Angle $\theta_{y}$ (\u00B0)")
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

        # --- CHECK WORKSPACE FOR PRE-CALCULATED BANDS ---
        # Pull the dictionary containing eigenvalues, eigenvectors, and k_vecs
        band_data = global_workspace.pull_band_structure(target_crystal)

        if not band_data:
            print(">> WARNING: No pre-calculated band structure found in Workspace.")
            print(">> Falling back to Analyzer Geometry Schematic generation.")
            self.draw_hemisphere_schematic(deflector_angle, slit_size)
            self.ax.clear()
            self.ax.set_title("Awaiting Band Data from DFT Suite")
            self.canvas.draw()
            return
            
        print(">> SUCCESS: Band structure loaded. Routing to Matrix Element Engine...")
        
        # --- PREPARE EXPERIMENT PARAMETERS ---
        e_min, e_max, e_steps = self.spin_e_min.value(), self.spin_e_max.value(), self.spin_e_steps.value()
        kx_min, kx_max, kx_steps = self.spin_kx_min.value(), self.spin_kx_max.value(), self.spin_kx_steps.value()
        ky_min, ky_max, ky_steps = self.spin_ky_min.value(), self.spin_ky_max.value(), self.spin_ky_steps.value()

        experiment_kwargs = {
            'photon_energy': self.photon_energy_spin.value(),
            'work_function': self.work_function_spin.value(),
            'incidence_angle': self.incidence_angle_spin.value(),
            'polarization': self.polarization_combo.currentText(),
            'matrix_element_mode': me_mode,
            'k_bounds': {'X': [kx_min, kx_max, kx_steps], 'Y': [ky_min, ky_max, ky_steps], 'E': [e_min, e_max, e_steps]}
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
            self.draw_hemisphere_schematic(deflector_angle, slit_size)
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
        
        self.ax.clear()
        self.ax.pcolormesh(self.sim_kx, self.sim_ky, slice_2d, shading='auto', cmap='afmhot')
        self.ax.set_aspect('equal')
        self.ax.set_title(f"Constant Energy Contour: {E_val:.2f} eV")
        self.ax.set_xlabel(r"$k_x$ ($\mathrm{\AA}^{-1}$)")
        self.ax.set_ylabel(r"$k_y$ ($\mathrm{\AA}^{-1}$)")
        
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