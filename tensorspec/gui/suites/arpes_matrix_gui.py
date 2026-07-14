import sys
import numpy as np
import matplotlib.patches as patches
from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                               QFormLayout, QSlider, QLabel, QComboBox, QGroupBox, 
                               QMainWindow, QDoubleSpinBox, QSpinBox)
from PySide6.QtCore import Qt

from tensorspec.core.arpes_engine import ARPESGeometryEngine
from tensorspec.plotting.viewers.matrix_setup_viewer import MatrixSetupViewer

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PySide6.QtWidgets import QPushButton, QMessageBox, QScrollArea
from tensorspec.core.workspace import global_workspace

import platform
is_legacy_mac = (platform.system() == "Darwin" and platform.machine() == "x86_64")

class MatrixElementGUI(QWidget):
    """
    Standalone modular GUI for the ARPES Matrix Element Simulator.
    Can be embedded into the main ARPES Suite later.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ARPES Matrix Element Simulator")
        self.resize(1000, 700)
        
        # Initialize Core Engine
        self.engine = ARPESGeometryEngine()
        
        self._setup_ui()
        self._connect_signals()

        # Trigger an initial render so the window isn't blank on startup
        self.update_simulation()

    def _setup_ui(self):
        main_layout = QHBoxLayout(self)
        
        # Left Panel: Controls
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        control_panel.setFixedWidth(420)
        
        # Sample Controls
        sample_group = QGroupBox("Sample & Manipulator")
        sample_form = QFormLayout(sample_group)
        # Sample Manipulator Sliders
        row_theta, self.theta_slider = self._make_slider_with_input(-180, 180, 0)
        row_beta, self.beta_slider = self._make_slider_with_input(-180, 180, 0)
        row_azimuth, self.azimuth_slider = self._make_slider_with_input(-180, 180, 0)
        # You will also need to update the addRow lines to pass the layout (row_theta) instead of the slider:
        sample_form.addRow("Theta (Z-axis):", row_theta)
        sample_form.addRow("Beta (Tilt):", row_beta)
        sample_form.addRow("Azimuth:", row_azimuth)

        self.orbital_combo = QComboBox()
        self.orbital_combo.addItems([
            "s", "px", "py", "pz", 
            "dxy", "dxz", "dyz", "dx2-y2", "dz2",
            "sp2 (x-directed)", "sp2 (y-directed)", "sp3"
        ])
        sample_form.addRow("Initial Orbital:", self.orbital_combo)
        
        # Beam & Slit Controls
        beam_group = QGroupBox("Beam & Slit")
        beam_form = QFormLayout(beam_group)
        # Beam Slit Sliders
        row_beam, self.beam_angle_slider = self._make_slider_with_input(0, 90, 45)
        row_slit, self.slit_angle_slider = self._make_slider_with_input(-90, 90, 0)
        
        beam_form.addRow("Beam Angle:", row_beam)
        beam_form.addRow("Slit Rotation:", row_slit)

        self.pol_combo = QComboBox()
        self.pol_combo.addItems(["LH", "LV", "CP", "CM"])
        self.mode_combo = QComboBox()
        self.mode_combo.addItems([
            "Geometric Dipole", 
            "Orbital Form Factor", 
            "Total (Geom x Orbital)",
            "Full Crystal (Geom x Orb x Lattice)"
        ])
        beam_form.addRow("Simulation Mode:", self.mode_combo)
        beam_form.addRow("Polarization:", self.pol_combo)
        
        
        # Workspace Integration Panel
        workspace_group = QGroupBox("Crystal Structure (Workspace)")
        ws_layout = QHBoxLayout(workspace_group)
        self.ws_combo = QComboBox()
        self.ws_combo.addItem("No structures available")
        
        self.btn_ws_refresh = QPushButton("🔄 Refresh")
        self.btn_ws_load = QPushButton("📥 Load Basis")
        
        ws_layout.addWidget(self.ws_combo)
        ws_layout.addWidget(self.btn_ws_refresh)
        ws_layout.addWidget(self.btn_ws_load)
        
        control_layout.addWidget(workspace_group)
        control_layout.addWidget(sample_group)
        control_layout.addWidget(beam_group)
        # ARPES Simulation Settings
        arpes_group = QGroupBox("E vs k ARPES Simulation")
        arpes_form = QFormLayout(arpes_group)
        
        self.plot_mode_combo = QComboBox()
        self.plot_mode_combo.addItems([
            "1. Matrix Element Heatmap (Angle vs Angle)", 
            "2. Raw Bands (E vs k, No ME)", 
            "3. Full ARPES (E vs k, with ME)",
            "4. Raw Constant Energy (kx vs ky, No ME)", 
            "5. Full Fermi Surface (kx vs ky, with ME)"
        ])
        
        # Band Structure Workspace loader
        ws_bands_layout = QHBoxLayout()
        self.ws_bands_combo = QComboBox()
        self.btn_ws_bands_refresh = QPushButton("🔄")
        self.btn_ws_bands_load = QPushButton("📥 Load Bands")
        ws_bands_layout.addWidget(self.ws_bands_combo)
        ws_bands_layout.addWidget(self.btn_ws_bands_refresh)
        ws_bands_layout.addWidget(self.btn_ws_bands_load)

        # --- NEW: Energy Slider for Slicing ---
        self.spin_energy = QDoubleSpinBox()
        self.spin_energy.setRange(-10.0, 10.0)
        self.spin_energy.setValue(0.0)  # Default to Fermi Level
        self.spin_energy.setSingleStep(0.1)
        
        # Physics Adjustments
        self.spin_temp = QDoubleSpinBox(); self.spin_temp.setRange(1.0, 1000.0); self.spin_temp.setValue(10.0)
        self.spin_broadening = QDoubleSpinBox(); self.spin_broadening.setRange(0.01, 1.0); self.spin_broadening.setValue(0.1); self.spin_broadening.setSingleStep(0.05)
        self.spin_noise = QDoubleSpinBox(); self.spin_noise.setRange(0.0, 1.0); self.spin_noise.setValue(0.05); self.spin_noise.setSingleStep(0.05)
        
        arpes_form.addRow("Plot Output:", self.plot_mode_combo)
        arpes_form.addRow("Band Data:", ws_bands_layout)
        arpes_form.addRow("Binding E (eV):", self.spin_energy)
        arpes_form.addRow("Temp (K):", self.spin_temp)
        arpes_form.addRow("Broadening Σ'' (eV):", self.spin_broadening)
        arpes_form.addRow("Noise Level:", self.spin_noise)
        
        control_layout.addWidget(arpes_group)
        control_layout.addStretch()
        
        # Center/Right Layout: Split between 3D view and 2D simulated map
        visualization_layout = QHBoxLayout()

        
        if is_legacy_mac:
            # Bypass PyVista to prevent complete GUI freeze on older Macs
            print("🔌 ARPES Suite: PyVista 3D Geometry Viewer disabled for Legacy Mac.")
            self.viewer = None
            placeholder = QLabel("<b>3D Geometry Viewer Disabled</b><br>PyVista is unsupported on this hardware architecture.<br>The 2D Matrix Element & ARPES physics simulations are fully functional.")
            placeholder.setAlignment(Qt.AlignCenter)
            placeholder.setStyleSheet("background-color: #2b2b2b; color: #888888; border: 1px dashed #555555;")
            visualization_layout.addWidget(placeholder, stretch=1)
        else:
            self.viewer = MatrixSetupViewer()
            visualization_layout.addWidget(self.viewer, stretch=1)
        
        # 2D Heatmap Canvas setup
        self.map_figure = Figure(figsize=(5, 5))
        self.map_canvas = FigureCanvas(self.map_figure)
        self.ax = self.map_figure.add_subplot(111)
        
        
        visualization_layout.addWidget(self.map_canvas, stretch=1)
        
        # --- NEW: Wrap the entire left panel in a Scroll Area ---
        scroll_area = QScrollArea()
        scroll_area.setWidget(control_panel)
        scroll_area.setWidgetResizable(True)
        scroll_area.setFixedWidth(440) # Slightly wider than the panel to fit the scrollbar
        scroll_area.setMinimumWidth(440)
        
        # Add the Scroll Area instead of the raw control panel
        main_layout.addWidget(scroll_area)
        main_layout.addLayout(visualization_layout, stretch=1)

    def _make_slider_with_input(self, min_val, max_val, default_val):
        """Creates a slider and a synced spinbox for manual number entry."""
        layout = QHBoxLayout()
        slider = QSlider(Qt.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setValue(default_val)
        
        spinbox = QSpinBox()
        spinbox.setRange(min_val, max_val)
        spinbox.setValue(default_val)
        spinbox.setFixedWidth(60) # Keep it compact
        
        # Sync them together
        slider.valueChanged.connect(spinbox.setValue)
        spinbox.valueChanged.connect(slider.setValue)
        
        layout.addWidget(slider)
        layout.addWidget(spinbox)
        return layout, slider

    def _create_slider(self, min_val, max_val, default_val):
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(min_val)
        slider.setMaximum(max_val)
        slider.setValue(default_val)
        return slider

    def _connect_signals(self):
        # Connect UI changes to update logic
        self.theta_slider.valueChanged.connect(self.update_simulation)
        self.beta_slider.valueChanged.connect(self.update_simulation)
        self.azimuth_slider.valueChanged.connect(self.update_simulation)
        self.beam_angle_slider.valueChanged.connect(self.update_simulation)
        self.slit_angle_slider.valueChanged.connect(self.update_simulation)
        self.pol_combo.currentTextChanged.connect(self.update_simulation)
        self.mode_combo.currentTextChanged.connect(self.update_simulation)
        self.orbital_combo.currentTextChanged.connect(self.update_simulation)
        # Workspace signals
        self.btn_ws_refresh.clicked.connect(self.refresh_workspace_list)
        self.btn_ws_load.clicked.connect(self.load_workspace_structure)
        # ARPES settings signals
        self.plot_mode_combo.currentTextChanged.connect(self.update_simulation)
        self.spin_temp.valueChanged.connect(self.update_simulation)
        self.spin_broadening.valueChanged.connect(self.update_simulation)
        self.spin_noise.valueChanged.connect(self.update_simulation)
        
        self.btn_ws_bands_refresh.clicked.connect(self.refresh_workspace_list)
        self.btn_ws_bands_load.clicked.connect(self.load_workspace_bands)

        self.spin_energy.valueChanged.connect(self.update_simulation)

    def refresh_workspace_list(self):
        """Pulls structures and band data from the central memory."""
        # Refresh Crystals
        self.ws_combo.clear()
        structures = global_workspace.list_crystal_structures()
        self.ws_combo.addItems(structures if structures else ["No structures available"])
        
        # Refresh Bands
        self.ws_bands_combo.clear()
        bands = global_workspace.list_band_structures()
        self.ws_bands_combo.addItems(bands if bands else ["No bands available"])

    def load_workspace_bands(self):
        target = self.ws_bands_combo.currentText()
        if not target or target == "No bands available":
            QMessageBox.warning(self, "Warning", "No valid band structure selected.")
            return
            
        if self.engine.load_bands_from_workspace(target):
            QMessageBox.information(self, "Success", f"Loaded Band Structure '{target}'!\nSwitch Plot Output to modes 2, 3, 4, or 5 to view it.")
            self.update_simulation()
        else:
            QMessageBox.critical(self, "Error", "Failed to load band structure from workspace.")

    def load_workspace_structure(self):
        """Loads the selected structure basis from the workspace into the math engine."""
        target = self.ws_combo.currentText()
        if target == "No structures available" or not target:
            QMessageBox.warning(self, "Warning", "No valid structure selected.")
            return
        
        success = self.engine.load_basis_from_workspace(target)
        if success:
            QMessageBox.information(self, "Success", f"Loaded real crystal basis '{target}'!\nStructure Factor calculations will now use these atomic coordinates.")
            self.update_simulation()
        else:
            QMessageBox.critical(self, "Error", "Failed to load structure from workspace.")

    def update_simulation(self):
        """Pushes UI values to the core engine, updates 3D viewer, and recalculates output."""
        self.engine.theta = self.theta_slider.value()
        self.engine.beta = self.beta_slider.value()
        self.engine.azimuth = self.azimuth_slider.value()
        self.engine.beam_angle = self.beam_angle_slider.value()
        self.engine.slit_angle = self.slit_angle_slider.value()
        self.engine.polarization = self.pol_combo.currentText()
        self.engine.initial_orbital = self.orbital_combo.currentText()
        
        if self.viewer is not None:
            self.viewer.update_geometry(self.engine)
        self.ax.clear()
        
        plot_mode = self.plot_mode_combo.currentIndex()
        
        # MODE 0: Hemisphere Matrix Element Heatmap
        if plot_mode == 0:
            txt = self.mode_combo.currentText()
            if txt == "Geometric Dipole":
                sim_mode = "geometric"
            elif txt == "Orbital Form Factor":
                sim_mode = "structure_factor"
            elif txt == "Total (Geom x Orbital)":
                sim_mode = "total"
            else:
                sim_mode = "full_crystal"
            
            # 1. Ask engine for the full hemisphere projection (Deflection vs Slit)
            defl_vals, slit_vals, intensity = self.engine.simulate_matrix_element_map(
                mode=sim_mode, defl_bounds=(-90, 90), slit_bounds=(-90, 90)
            )
            
            vmax = np.max(intensity) if np.max(intensity) > 1e-6 else 1e-6
            self.ax.pcolormesh(slit_vals, defl_vals, intensity, shading='auto', cmap='magma', vmin=0, vmax=vmax)
            
            # 2. Draw the ARPES Measurement Window Patch
            slit_width = 30 
            defl_height = 30
            
            # The analyzer acceptance is permanently centered on the detector
            rect = patches.Rectangle(
                (-slit_width/2, -defl_height/2),
                slit_width, defl_height,
                linewidth=2, edgecolor='cyan', facecolor='none', linestyle='--'
            )
            self.ax.add_patch(rect)
            
            # 3. Formatting
            self.ax.set_xlim(-90, 90)
            self.ax.set_ylim(-90, 90)
            self.ax.set_title(f"Hemisphere Matrix Element ({self.mode_combo.currentText()})")
            self.ax.set_xlabel("Analyzer Slit Angle (deg)")
            self.ax.set_ylabel("Deflection Angle (deg)")
            
        # MODE 1 & 2: 2D E vs k ARPES Spectrum Simulation
        elif plot_mode == 1 or plot_mode == 2:
            physics_mode = 'bands_only' if plot_mode == 1 else 'full'
            res = self.engine.simulate_arpes_cut(
                mode=physics_mode,
                temp=self.spin_temp.value(),
                broadening=self.spin_broadening.value(),
                noise=self.spin_noise.value()
            )
            
            if res is None:
                self.ax.set_title("⚠️ Please load Band Data from Workspace first.")
            else:
                k_dist, E_axis, intensity, node_idx, labels = res
                
                # Use a standard ARPES colormap (e.g., 'afmhot' or 'magma')
                self.ax.pcolormesh(k_dist, E_axis, intensity, shading='auto', cmap='afmhot')
                self.ax.set_title("Full ARPES Simulation" if plot_mode == 2 else "Raw Bands (No ME)")
                self.ax.set_xlabel("Momentum path (Å⁻¹)")
                self.ax.set_ylabel("Binding Energy (eV)")
                
                # Draw high symmetry lines
                for i in node_idx:
                    self.ax.axvline(k_dist[i], color='white', linewidth=0.5, linestyle=':')
                self.ax.set_xticks([k_dist[i] for i in node_idx])
                self.ax.set_xticklabels(labels)
        
        # MODE 3 & 4: 2D Constant Energy Map (kx vs ky)
        elif plot_mode == 3 or plot_mode == 4:
            physics_mode = 'bands_only' if plot_mode == 3 else 'full'
            res = self.engine.simulate_isoenergy_contour(
                target_energy=self.spin_energy.value(),
                mode=physics_mode,
                temp=self.spin_temp.value(),
                broadening=self.spin_broadening.value(),
                noise=self.spin_noise.value()
            )
            
            if res is None:
                self.ax.set_title("⚠️ Please load 2D Band Data from Workspace first.")
            else:
                kx, ky, intensity = res
                self.ax.pcolormesh(kx, ky, intensity, shading='auto', cmap='afmhot')
                self.ax.set_aspect('equal')
                self.ax.set_title(f"Constant Energy Contour (E = {self.spin_energy.value():.2f} eV)")
                self.ax.set_xlabel(r"$k_x$ ($\mathrm{\AA}^{-1}$)")
                self.ax.set_ylabel(r"$k_y$ ($\mathrm{\AA}^{-1}$)")
    
        self.map_figure.tight_layout()
        self.map_canvas.draw()



# Standalone runner
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MatrixElementGUI()
    window.show()
    sys.exit(app.exec())